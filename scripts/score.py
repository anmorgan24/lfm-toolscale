"""Label saved outputs, summarize them with cluster bootstrap intervals, and log to Opik.

  python scripts/score.py --machine gpu            # summary only
  python scripts/score.py --machine gpu --opik     # also log each file as an Opik experiment
  python scripts/score.py --agreement laptop gpu   # compare outputs on shared requests
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from toolscale.data import load_catalog  # noqa: E402
from toolscale.scoring import Scorer  # noqa: E402

LABELS = ["correct", "format_error", "no_call", "unneeded_call", "invented_function",
          "substituted_call", "spurious_call", "dropped_call", "extra_call", "argument_error"]

tools, kinds = load_catalog()
scorer = Scorer(tools, kinds)
rows = {r["id"]: r for r in map(json.loads, open(ROOT / "data" / "eval_set.jsonl"))}
catalogs = json.loads((ROOT / "data" / "catalogs.json").read_text())
traps_path = ROOT / "data" / "lookalike_traps.json"
traps = json.loads(traps_path.read_text()) if traps_path.exists() else {}
removed_path = ROOT / "data" / "trap_removed_catalogs.json"
removed = json.loads(removed_path.read_text()) if removed_path.exists() else {}


def catalog_for(rec):
    if rec["cell"] == "trap_removed":
        return removed[traps[rec["id"]]["trap"]]
    return catalogs[rec["cell"]]


def label(rec):
    return scorer.failure_label(json.loads(rows[rec["id"]]["calls"]), rec["calls"],
                                catalog_for(rec), rec["format_error"])


def cluster(rid):
    """Requests for the same gold tool are close paraphrases, so they resample together."""
    names = sorted({c["name"] for c in json.loads(rows[rid]["calls"])})
    return "+".join(names) if names else f"no_tool:{rid}"


def cluster_ci(ids, correct, n_boot=2000, seed=0):
    groups = defaultdict(list)
    for rid in ids:
        groups[cluster(rid)].append(correct[rid])
    keys = list(groups)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        picked = [v for k in rng.choices(keys, k=len(keys)) for v in groups[k]]
        stats.append(sum(picked) / len(picked))
    stats.sort()
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot)]


def load(machine):
    files = sorted((ROOT / "results" / machine).glob("*/*.jsonl"))
    return {f: [json.loads(line) for line in open(f)] for f in files}


def summarize(machine):
    out_rows = []
    for f, recs in load(machine).items():
        labels = {r["id"]: label(r) for r in recs}
        correct = {rid: lab == "correct" for rid, lab in labels.items()}
        ids = list(labels)
        lo, hi = cluster_ci(ids, correct)
        counts = Counter(labels.values())
        look = [rid for rid in ids if rows[rid]["lookalike_scaling"]]
        row = {"file": str(f.relative_to(ROOT)), "n": len(ids),
               "accuracy": round(sum(correct.values()) / len(ids), 3),
               "ci_low": round(lo, 3), "ci_high": round(hi, 3),
               "lookalike_accuracy": round(sum(correct[r] for r in look) / len(look), 3) if look else "",
               "mean_prompt_tokens": round(sum(r["prompt_tokens"] or 0 for r in recs) / len(recs)),
               "mean_prompt_s": round(sum(r["prompt_ms"] or 0 for r in recs) / len(recs) / 1000, 2),
               "mean_decode_tok_s": round(sum(r["decode_tok_per_s"] or 0 for r in recs) / len(recs), 1),
               **{lab: counts.get(lab, 0) for lab in LABELS}}
        out_rows.append(row)
        print(f"{row['file']}: n={row['n']} accuracy {row['accuracy']:.1%} "
              f"[{lo:.1%}, {hi:.1%}] | {dict(counts)}")
    if out_rows:
        path = ROOT / "results" / machine / "summary.csv"
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nWrote {path}")


def agreement(a, b):
    """Share of shared requests where two machines produced the same calls."""
    def index(machine):
        return {(r["model"], r["cell"], r["sampling"], r["id"]): r
                for recs in load(machine).values() for r in recs}
    ia, ib = index(a), index(b)
    shared = sorted(set(ia) & set(ib))
    if not shared:
        print("No shared requests yet.")
        return
    same_calls = sum(ia[k]["calls"] == ib[k]["calls"] for k in shared)
    same_label = sum(label(ia[k]) == label(ib[k]) for k in shared)
    print(f"{len(shared)} shared requests: identical calls {same_calls / len(shared):.1%}, "
          f"same label {same_label / len(shared):.1%}")


def log_to_opik(machine):
    import opik
    from opik.evaluation import evaluate
    from opik.evaluation.metrics import base_metric, score_result

    class LabelIs(base_metric.BaseMetric):
        def __init__(self, target):
            super().__init__(name=target)
            self.target = target

        def score(self, label, **ignored):
            return score_result.ScoreResult(name=self.name, value=float(label == self.target))

    class PromptSeconds(base_metric.BaseMetric):
        def score(self, prompt_s, **ignored):
            return score_result.ScoreResult(name=self.name, value=prompt_s)

    client = opik.Opik()
    dataset = client.get_or_create_dataset("toolscale-eval-set")
    dataset.insert([{"request_id": r["id"], "request": r["request"], "category": r["category"],
                     "gold_calls": r["calls"]} for r in rows.values()])
    items = dataset.get_items()
    if items and "id" not in items[0]:
        sys.exit("Opik dataset items have no 'id' field in this SDK version; update the mapping in log_to_opik.")
    item_ids = {item["request_id"]: item["id"] for item in items}

    for f, recs in load(machine).items():
        by_id = {r["id"]: r for r in recs}

        def task(item):
            rec = by_id[item["request_id"]]
            return {"output": json.dumps(rec["calls"], ensure_ascii=False) or rec["content"],
                    "label": label(rec), "prompt_s": (rec["prompt_ms"] or 0) / 1000}

        first = recs[0]
        evaluate(dataset=dataset, task=task,
                 scoring_metrics=[LabelIs(lab) for lab in LABELS] + [PromptSeconds(name="prompt_seconds")],
                 experiment_name=f"{first['model']}-{f.stem}-{machine}",
                 experiment_config={k: first[k] for k in ("model", "cell", "sampling", "machine", "server")},
                 dataset_item_ids=[item_ids[rid] for rid in by_id], verbose=0)
        print(f"Logged {f.relative_to(ROOT)} to Opik")


parser = argparse.ArgumentParser()
parser.add_argument("--machine")
parser.add_argument("--opik", action="store_true")
parser.add_argument("--agreement", nargs=2, metavar=("A", "B"))
args = parser.parse_args()
if args.agreement:
    agreement(*args.agreement)
if args.machine:
    summarize(args.machine)
    if args.opik:
        log_to_opik(args.machine)
