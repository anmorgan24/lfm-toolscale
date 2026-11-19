"""Inspect phone-assistant-tools to choose the core tools for the experiment.

Run: python scripts/inspect_dataset.py
Paste the output back into the conversation.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toolscale.data import (  # noqa: E402
    NO_CALL_CATEGORIES, eligible, gold_names, greedy_core, load_catalog,
    load_requests, lookalike_pairs, tool_name,
)

tools, scoring = load_catalog()
dataset = load_requests()
print(f"Tools: {len(tools)}")
print("Splits:", {name: len(split) for name, split in dataset.items()})

for name, split in dataset.items():
    print(f"\n[{name}] categories:", dict(Counter(split["category"])))
    print(f"[{name}] languages:", dict(Counter(split["language"]).most_common(5)), "...")

usage = Counter(n for split in ("train", "test") for e in dataset[split] for n in gold_names(e))
print("\nMost used tools (train + test):", usage.most_common(25))
print("Never used in train + test:", [tool_name(t) for t in tools if tool_name(t) not in usage])

pairs = lookalike_pairs(tools)
print(f"\nLookalike candidates ({len(pairs)}), top 30:")
for a, b, s in pairs[:30]:
    print(f"  {s:.2f}  {a}  ~  {b}")

english = [e for split in ("train", "test") for e in dataset[split] if e["language"] == "en"]
pool_all = [e for split in ("train", "test") for e in dataset[split]]
for label, pool in (("English only", english), ("all languages", pool_all)):
    for size in (10, 15):
        core = greedy_core(pool, tools, size=size, lookalikes=pairs)
        ok = eligible(pool, core)
        calls = [e for e in ok if e["category"] not in NO_CALL_CATEGORIES]
        print(f"\nCore of {size} ({label}): {core}")
        print(f"  eligible requests: {len(ok)} ({len(calls)} with calls)")
        print("  by category:", dict(Counter(e["category"] for e in ok)))

print("\nExample request and context:")
e = dataset["test"][0]
print({k: e[k] for k in e if k != "calls"}, "\ncalls:", e["calls"])
