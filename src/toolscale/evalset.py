"""Build the evaluation requests for the scaling experiment."""
import random

from .data import gold_names

CALL_CATEGORIES = {"single", "multi", "lookalike", "partial"}


def build_eval_set(examples, core_tools, prompt_tools, max_calls=300,
                   no_tool_share=0.15, language="en", seed=0):
    """English requests answerable within the core, plus a capped share of no-tool requests.

    Clarify requests are left out: the right answer depends on whether the tool
    the user wanted is in the catalog, and the data doesn't say which tool that is.
    Lookalike requests for prompt tools are kept but flagged, since their
    lookalikes are in every catalog.
    """
    core = set(core_tools)
    pool = [e for e in examples if e["language"] == language]
    calls = [e for e in pool if e["category"] in CALL_CATEGORIES and gold_names(e) <= core]
    no_tool = [e for e in pool if e["category"] == "no_tool"]
    rng = random.Random(seed)
    rng.shuffle(calls)
    calls = calls[:max_calls]
    n_no_tool = round(len(calls) * no_tool_share / (1 - no_tool_share))
    rng.shuffle(no_tool)
    chosen = calls + no_tool[:n_no_tool]
    rows = []
    for e in chosen:
        names = gold_names(e)
        rows.append({
            "id": e["id"],
            "category": e["category"],
            "request": e["request"],
            "context": e["context"],
            "calls": e["calls"],
            "lookalike_scaling": e["category"] == "lookalike" and not (names & set(prompt_tools)),
        })
    return sorted(rows, key=lambda r: r["id"])

