"""Deferred tools: the model starts with a search tool and finds the rest.

Each request starts with the six prompt-named tools (always loaded, since the
system prompt names them) plus search_tools. When the model searches, BM25 over
tool names and descriptions returns the top five, and the model gets a second
turn with those tools added. Both models use this same harness.
"""
import json
import math
import re
from collections import Counter

from .catalog import PROMPT_TOOLS
from .client import complete
from .prompt import build_messages, select_tools

TOP_K = 5
SEARCH_TOOL = {"type": "function", "function": {
    "name": "search_tools",
    "description": "Search the phone's tool catalog. Returns the names and descriptions of the most "
                   "relevant tools. Use it to find a tool that is not already listed.",
    "parameters": {"type": "object", "properties": {"query": {
        "type": "string", "description": "What the user wants to do, in a few words"}},
        "required": ["query"]}}}
DEFERRED_NOTE = ("\n\nMost of the phone's tools are not listed above. To use one, first call "
                 "search_tools to find it.")


def tokens(text):
    """Lowercase words, with a plural 's' dropped so "alarm" matches "alarms"."""
    words = re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words]


class BM25:
    def __init__(self, tools, k1=1.5, b=0.75):
        self.names = [t["function"]["name"] for t in tools]
        self.descs = {t["function"]["name"]: t["function"].get("description", "") for t in tools}
        self.docs = [Counter(tokens(n) + tokens(self.descs[n])) for n in self.names]
        self.avg = sum(sum(d.values()) for d in self.docs) / len(self.docs)
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def search(self, query, k=TOP_K):
        q = tokens(query)
        scores = []
        for name, d in zip(self.names, self.docs):
            length = sum(d.values())
            s = sum(self.idf.get(w, 0) * d[w] * (self.k1 + 1) /
                    (d[w] + self.k1 * (1 - self.b + self.b * length / self.avg)) for w in q if w in d)
            scores.append((s, name))
        return [name for s, name in sorted(scores, key=lambda x: -x[0])[:k] if s > 0]


def assistant_turn(calls):
    """Rebuild the model's tool calls as a chat message, with our own ids."""
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"call_{i}", "type": "function",
         "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {})}}
        for i, c in enumerate(calls)]}


def run_deferred(row, tools, index, sampling, url):
    """Two-step request. Returns the final result plus search details and summed timings."""
    messages = build_messages(row)
    messages[0]["content"] += DEFERRED_NOTE
    loaded = select_tools(tools, PROMPT_TOOLS) + [SEARCH_TOOL]
    first = complete(messages, loaded, sampling=sampling, url=url)
    searches = [c for c in first["calls"] if c["name"] == "search_tools"]
    record = {"step1_calls": first["calls"], "queries": [], "found": [], "steps": 1}
    if not searches or first["format_error"]:
        final = first
    else:
        found = []
        for c in searches:
            query = str((c.get("arguments") or {}).get("query", ""))
            hits = index.search(query)
            record["queries"].append(query)
            found += [h for h in hits if h not in found]
        record["found"] = found
        results = [{"role": "tool", "tool_call_id": f"call_{i}",
                    "content": json.dumps([{"name": n, "description": index.descs[n]} for n in index.search(
                        str((c.get("arguments") or {}).get("query", "")))])}
                   for i, c in enumerate(first["calls"]) if c["name"] == "search_tools"]
        messages = messages + [assistant_turn(first["calls"])] + results
        second_tools = loaded + select_tools(tools, [n for n in found if n not in PROMPT_TOOLS])
        final = complete(messages, second_tools, sampling=sampling, url=url)
        record["steps"] = 2
        for key in ("prompt_tokens", "completion_tokens", "prompt_ms", "decode_ms", "wall_ms"):
            final[key] = (final.get(key) or 0) + (first.get(key) or 0)
    gold = {c["name"] for c in json.loads(row["calls"])}
    record["search_recall"] = (len(gold & set(record["found"]) | (gold & set(PROMPT_TOOLS))) / len(gold)) if gold else None
    return {**final, **record}
