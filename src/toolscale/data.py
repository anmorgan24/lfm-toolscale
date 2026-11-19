"""Load Liquid's phone-assistant-tools dataset (Apache 2.0) and its tool catalog."""
import json
import re
from collections import Counter
from difflib import SequenceMatcher

from datasets import load_dataset
from huggingface_hub import hf_hub_download

DATASET_ID = "LiquidAI/phone-assistant-tools"
NO_CALL_CATEGORIES = {"no_tool", "clarify"}


def load_catalog():
    """Return (tools, scoring): the 108 tool schemas and per-argument comparison kinds."""
    tools = json.load(open(hf_hub_download(DATASET_ID, "tools.json", repo_type="dataset")))
    scoring = json.load(open(hf_hub_download(DATASET_ID, "scoring.json", repo_type="dataset")))
    return tools, scoring


def load_requests():
    return load_dataset(DATASET_ID)


def tool_name(tool):
    return tool["function"]["name"]


def gold_calls(example):
    calls = example["calls"]
    return json.loads(calls) if isinstance(calls, str) else calls


def gold_names(example):
    return {call["name"] for call in gold_calls(example)}


def name_tokens(name):
    return set(re.split(r"[_\W]+", name.lower())) - {""}


def lookalike_pairs(tools, threshold=0.5):
    """Heuristic lookalikes: overlapping name tokens or similar descriptions.

    Used only to guide core selection; check the pairs by hand.
    """
    pairs = []
    for i, a in enumerate(tools):
        for b in tools[i + 1:]:
            ta, tb = name_tokens(tool_name(a)), name_tokens(tool_name(b))
            name_sim = len(ta & tb) / len(ta | tb)
            desc_sim = SequenceMatcher(
                None, a["function"].get("description", ""), b["function"].get("description", "")
            ).ratio()
            score = max(name_sim, desc_sim)
            if score >= threshold:
                pairs.append((tool_name(a), tool_name(b), round(score, 2)))
    return sorted(pairs, key=lambda p: -p[2])


def eligible(examples, core):
    """Requests answerable within the core: gold tools all in core, or no call expected."""
    core = set(core)
    return [e for e in examples if e["category"] in NO_CALL_CATEGORIES or gold_names(e) <= core]


def greedy_core(examples, tools, size=10, lookalikes=None):
    """Grow a core that maximizes eligible tool-call requests.

    Ties go to tools that have a lookalike outside the core, so that growing the
    catalog adds lookalikes for the core's requests.
    """
    has_lookalike = Counter()
    for a, b, _ in lookalikes or []:
        has_lookalike[a] += 1
        has_lookalike[b] += 1
    calling = [e for e in examples if e["category"] not in NO_CALL_CATEGORIES]
    core = []
    candidates = [tool_name(t) for t in tools]
    while len(core) < size:
        def gain(name):
            covered = sum(1 for e in calling if gold_names(e) <= set(core + [name]))
            return (covered, has_lookalike[name])
        best = max((c for c in candidates if c not in core), key=gain)
        core.append(best)
    return core
