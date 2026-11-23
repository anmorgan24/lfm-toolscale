"""Build the nested catalogs and the evaluation set, and save them to data/.

Run: python scripts/build_experiment.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toolscale.catalog import (  # noqa: E402
    PROMPT_TOOLS, ROOT, SIZES, all_catalogs, core, load_groups, nested_catalogs, validate_groups,
)
from toolscale.data import load_catalog, load_requests  # noqa: E402
from toolscale.evalset import build_eval_set  # noqa: E402

tools, _ = load_catalog()
groups = load_groups()
groups_by_tool = validate_groups(groups, tools)
core_tools = core(groups_by_tool)
catalogs = nested_catalogs(tools, core_tools)
catalog_file = all_catalogs(tools, core_tools)

ds = load_requests()
examples = [e for split in ("train", "test") for e in ds[split]]
rows = build_eval_set(examples, core_tools, PROMPT_TOOLS)

print(f"Core ({len(core_tools)}): {core_tools}")
print(f"Catalog sizes: {[len(c) for c in catalogs.values()]}")
print(f"Ungrouped tools: {sorted({t['function']['name'] for t in tools} - set(groups_by_tool))}")
print(f"\nEvaluation requests: {len(rows)}")
print("By category:", dict(Counter(r["category"] for r in rows)))
print("Gold tools:", dict(Counter(c["name"] for r in rows for c in json.loads(r["calls"]))))
scaling = [r for r in rows if r["lookalike_scaling"]]
print(f"\nLookalike requests used for scaling: {len(scaling)}")

# Trap-removed catalogs: the 108-tool catalog minus one trap, for the paired
# comparison on lookalike requests. Needs data/lookalike_traps.json
# (scripts/propose_traps.py), so it is skipped on the first run.
traps_path = ROOT / "data" / "lookalike_traps.json"
trap_removed = {}
if traps_path.exists():
    labeled = json.loads(traps_path.read_text())
    # A null trap means no wrong tool is tempting; those requests stay in the
    # scaling runs but are left out of the trap-removed comparison.
    lookalike_ids = {r["id"] for r in scaling}
    unlabeled = sorted(lookalike_ids - set(labeled))
    if unlabeled:
        print(f"\n{len(unlabeled)} lookalike requests have no label yet; run scripts/propose_traps.py")
    labeled = {rid: e for rid, e in labeled.items() if rid in lookalike_ids}
    traps = {rid: e for rid, e in labeled.items() if e["trap"]}
    print(f"\nTrap labels: {len(traps)} requests with a trap, {len(labeled) - len(traps)} marked as no clear trap")
    full = catalogs[max(catalogs)]
    for rid, entry in traps.items():
        trap = entry["trap"]
        if trap not in full:
            raise ValueError(f"{rid}: trap {trap} is not in the catalog")
        if trap in core_tools:
            raise ValueError(f"{rid}: trap {trap} is in the core, so it is never absent")
        trap_removed.setdefault(trap, [t for t in full if t != trap])
    by_trap = Counter(e["trap"] for e in traps.values())
    print("\nLabeled traps (requests, first catalog size containing the trap):")
    for trap, n in by_trap.most_common():
        first = min(size for size, c in catalogs.items() if trap in c)
        print(f"  {trap}: {n} requests, joins at {first}")
else:
    print("\nNo trap labels yet: run scripts/propose_traps.py, then rerun this script.")

out = ROOT / "data"
if trap_removed:
    (out / "trap_removed_catalogs.json").write_text(json.dumps(trap_removed, indent=1))
    print(f"Wrote {len(trap_removed)} trap-removed catalogs to {out / 'trap_removed_catalogs.json'}")
(out / "catalogs.json").write_text(json.dumps(catalog_file, indent=1))
print(f"Catalogs written: {list(catalog_file)}")
with open(out / "eval_set.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\nWrote {out / 'catalogs.json'} and {out / 'eval_set.jsonl'}")
