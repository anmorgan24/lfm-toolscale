"""Deferred tools: the model starts with a search tool and finds the rest.

Each request starts with the six prompt-named tools (always loaded, since the
system prompt names them) plus search_tools, and the system prompt's no-tool rule
tells the model to search first. Search is BM25 over each tool's name,
description, and parameters, returning the top five.

LFM3 uses Liquid's own format from its chat template: every tool is passed on
every request, with the non-loaded ones marked defer_loading (left out of the
system prompt), and search results carry full definitions inside
<additional_tools>, as the template renders them. MiniCPM5 has no such mechanism,
so it gets the closest equivalent: search results carry the definitions as JSON,
and the found tools join its tool list for the next step.
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
    "description": "Search the phone's tool catalog. Returns the most relevant tools with their "
                   "definitions. Use it to find a tool that is not already listed.",
    "parameters": {"type": "object", "properties": {"query": {
        "type": "string", "description": "What the user wants to do, in a few words"}},
        "required": ["query"]}}}


def tokens(text):
    """Lowercase words, with a plural 's' dropped so "alarm" matches "alarms"."""
    words = re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words]


def tool_text(tool):
    fn = tool["function"]
    params = fn.get("parameters", {}).get("properties", {})
    parts = [fn["name"], fn.get("description", "")]
    parts += [f"{k} {v.get('description', '')}" for k, v in params.items() if isinstance(v, dict)]
    return " ".join(parts)


class BM25:
    def __init__(self, tools, k1=1.5, b=0.75):
        self.names = [t["function"]["name"] for t in tools]
        self.descs = {t["function"]["name"]: t["function"].get("description", "") for t in tools}
        self.docs = [Counter(tokens(tool_text(t))) for t in tools]
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


def function_json(fn):
    """One tool definition as LFM3's template renders it in <additional_tools>."""
    return json.dumps({k: v for k, v in fn.items() if k != "defer_loading"}, ensure_ascii=False)


def deferred_tools(tools):
    """All tools, with everything outside the loaded set marked defer_loading (LFM3 format)."""
    out = []
    for t in tools:
        fn = dict(t["function"])
        if fn["name"] not in PROMPT_TOOLS:
            fn["defer_loading"] = True
        out.append({"type": "function", "function": fn})
    return out + [SEARCH_TOOL]


def search_result(names, by_name, native):
    fns = [by_name[n]["function"] for n in names]
    if native:
        return "<additional_tools>\n" + "\n".join(function_json(fn) for fn in fns) + "\n</additional_tools>"
    return json.dumps(fns, ensure_ascii=False)


def run_deferred(row, tools, index, sampling, url, native, forced=False):
    """Up to two steps. Returns the final result plus search details and summed timings.

    forced: the harness searches on the model's behalf, with the user's request as the
    query, and presents the results as though the model had searched. Only the
    model's decision to search is removed; formats and tools are unchanged.
    """
    by_name = {t["function"]["name"]: t for t in tools}
    messages = build_messages(row, deferred=True)
    loaded = select_tools(tools, PROMPT_TOOLS) + [SEARCH_TOOL]
    step1_tools = deferred_tools(tools) if native else loaded
    if forced:
        first = {"calls": [{"name": "search_tools", "arguments": {"query": row["request"]}}],
                 "format_error": False}
    else:
        first = complete(messages, step1_tools, sampling=sampling, url=url)
    searches = [(i, c) for i, c in enumerate(first["calls"]) if c["name"] == "search_tools"]
    record = {"step1_calls": first["calls"], "queries": [], "found": [], "steps": 1,
              "native": native, "forced": forced}
    if not searches or first["format_error"]:
        final = first
    else:
        found, results = [], []
        for i, c in searches:
            query = str((c.get("arguments") or {}).get("query", ""))
            hits = index.search(query)
            record["queries"].append(query)
            found += [h for h in hits if h not in found]
            results.append({"role": "tool", "tool_call_id": f"call_{i}",
                            "content": search_result(hits, by_name, native)})
        record["found"] = found
        messages = messages + [assistant_turn(first["calls"])] + results
        step2_tools = step1_tools if native else loaded + select_tools(
            tools, [n for n in found if n not in PROMPT_TOOLS])
        final = complete(messages, step2_tools, sampling=sampling, url=url)
        record["steps"] = 2
        for key in ("prompt_tokens", "completion_tokens", "prompt_ms", "decode_ms", "wall_ms"):
            final[key] = (final.get(key) or 0) + (first.get(key) or 0)
    gold = {c["name"] for c in json.loads(row["calls"])}
    reachable = set(record["found"]) | set(PROMPT_TOOLS)
    record["search_recall"] = len(gold & reachable) / len(gold) if gold else None
    return {**final, **record}
