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
from toolscale.catalog import load_groups, parse_cell  # noqa: E402
from toolscale.data import load_catalog  # noqa: E402
from toolscale.scoring import Scorer, is_lookup  # noqa: E402

import re as _re
CLAIM = _re.compile(r"\b(I've|I have|I just)\s+(added|sent|set|created|deleted|removed|started|booked|scheduled|"
                    r"turned|saved|marked|played|called|logged|texted|paid|transferred|completed)\b", _re.I)

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
group_of = {t: g for g, members in load_groups().items() for t in members}


def catalog_for(rec):
    cell = rec["cell"]
    if cell == "trap_removed":
        return removed[traps[rec["id"]]["trap"]]
    if cell.startswith("deferred"):
        return catalogs["108"]
    if cell.startswith("followup_"):
        return catalogs[cell.split("_", 1)[1]]
    return catalogs[cell]


def label(rec):
    return scorer.failure_label(json.loads(rows[rec["id"]]["calls"]), rec["calls"],
                                catalog_for(rec), rec["format_error"])


def substitution_kind(rec):
    """For a substituted call: did the model look something up instead of acting,
    pick the labeled trap, another tool from a gold tool's lookalike group, or an
    unrelated tool? Checked in that order."""
    gold = {c["name"] for c in json.loads(rows[rec["id"]]["calls"])}
    wrong = [c.get("name") for c in rec["calls"] if c.get("name") not in gold]
    if wrong and all(is_lookup(n) for n in wrong) and not all(is_lookup(n) for n in gold):
        return "sub_lookup"
    trap = (traps.get(rec["id"]) or {}).get("trap")
    if trap and trap in wrong:
        return "sub_trap"
    gold_groups = {group_of.get(n) for n in gold} - {None}
    if any(group_of.get(n) in gold_groups for n in wrong):
        return "sub_group"
    return "sub_unrelated"


def mentions_tool(rec):
    """A no-call answer that names a tool it could have called: it described the call instead of making it."""
    text = rec.get("content") or ""
    offered = rec.get("found") or catalog_for(rec)
    return any(name in text for name in offered)


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
        subs = Counter(substitution_kind(r) for r in recs if labels[r["id"]] == "substituted_call")
        look = [rid for rid in ids if rows[rid]["lookalike_scaling"]]
        size, order = parse_cell(recs[0]["cell"])
        row = {"file": str(f.relative_to(ROOT)), "model": recs[0]["model"], "size": size,
               "order": "" if order is None else order, "n": len(ids),
               "accuracy": round(sum(correct.values()) / len(ids), 3),
               "ci_low": round(lo, 3), "ci_high": round(hi, 3),
               "lookalike_accuracy": round(sum(correct[r] for r in look) / len(look), 3) if look else "",
               "mean_prompt_tokens": round(sum(r["prompt_tokens"] or 0 for r in recs) / len(recs)),
               "mean_prompt_s": round(sum(r["prompt_ms"] or 0 for r in recs) / len(recs) / 1000, 2),
               "mean_decode_tok_s": round(sum(r["decode_tok_per_s"] or 0 for r in recs) / len(recs), 1),
               **{lab: counts.get(lab, 0) for lab in LABELS},
               **{k: subs.get(k, 0) for k in ("sub_lookup", "sub_trap", "sub_group", "sub_unrelated")},
               "narrated_call": sum(1 for r in recs if labels[r["id"]] == "no_call" and mentions_tool(r)),
               "claimed_action": sum(1 for r in recs if labels[r["id"]] == "no_call" and CLAIM.search(r.get("content") or "")),
               "searched": sum(1 for r in recs if r.get("steps") == 2) if recs[0]["cell"].startswith("deferred") else "",
               "accuracy_when_found": (lambda f: round(sum(correct[r["id"]] for r in f) / len(f), 3) if f else "")(
                   [r for r in recs if r.get("steps") == 2 and r.get("search_recall") == 1])
               if recs[0]["cell"].startswith("deferred") else "",
               "search_recall": round(sum(r["search_recall"] for r in recs if r.get("search_recall") is not None)
                                      / max(1, sum(r.get("search_recall") is not None for r in recs)), 3)
               if recs[0]["cell"].startswith("deferred") else ""}
        out_rows.append(row)
        extra = (f" | searched {row['searched']}, search recall {row['search_recall']:.0%}, "
                 f"accuracy when the right tool was found {row['accuracy_when_found']:.1%}"
                 if row["searched"] != "" and row["accuracy_when_found"] != "" else "")
        print(f"{row['file']}: n={row['n']} accuracy {row['accuracy']:.1%} "
              f"[{lo:.1%}, {hi:.1%}] | {dict(counts)}" + (f" | substitutions: {dict(subs)}" if subs else "") + extra)
    by_size = defaultdict(list)
    for row in out_rows:
        cell = Path(row["file"]).stem.split("_")[0]
        if row["n"] > 100 and (cell.isdigit() or "s" in cell and cell.split("s")[0].isdigit()):
            by_size[(row["model"], row["size"])].append(row["accuracy"])
    if by_size:
        print("\nAccuracy by catalog size, averaged across arrival orders:")
        for (model, size), accs in sorted(by_size.items()):
            print(f"  {model:<9} {size:>3} tools: {sum(accs) / len(accs):.1%}  (orders: {', '.join(f'{a:.1%}' for a in accs)})")
    if out_rows:
        path = ROOT / "results" / machine / "summary.csv"
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nWrote {path}")


def substitution_pairs(machine):
    """Most common (gold -> chosen) substitutions in each file."""
    for f, recs in load(machine).items():
        pairs = Counter()
        for r in recs:
            if label(r) != "substituted_call":
                continue
            gold = {c["name"] for c in json.loads(rows[r["id"]]["calls"])}
            for c in r["calls"]:
                if c.get("name") not in gold:
                    pairs[(", ".join(sorted(gold)), c.get("name"))] += 1
        if pairs:
            print(f"{f.relative_to(ROOT)}: " + "; ".join(f"{a} -> {b} x{n}" for (a, b), n in pairs.most_common(5)))


def trap_comparison(machine):
    """The same lookalike requests at 108 tools, with and without each request's trap."""
    for model_dir in sorted((ROOT / "results" / machine).glob("*/")):
        full_path, tr_path = model_dir / "108_all_greedy.jsonl", model_dir / "trap_removed_all_greedy.jsonl"
        if not (full_path.exists() and tr_path.exists()):
            continue
        full = {r["id"]: r for r in map(json.loads, open(full_path))}
        tr = [json.loads(line) for line in open(tr_path)]
        present = sum(label(full[r["id"]]) == "correct" for r in tr)
        absent = sum(label(r) == "correct" for r in tr)
        print(f"{model_dir.name}: {len(tr)} lookalike requests at 108 tools, correct with trap {present}, without {absent}")


def correctness(machine, model, cell):
    path = ROOT / "results" / machine / model / f"{cell}_all_greedy.jsonl"
    if not path.exists():
        return None
    return {r["id"]: label(r) == "correct" for r in map(json.loads, open(path))}


def paired_ci(ids, diff, n_boot=2000, seed=0):
    """Cluster bootstrap interval for a mean per-request difference, resampling gold tools."""
    groups = defaultdict(list)
    for rid in ids:
        groups[cluster(rid)].append(diff[rid])
    keys, rng, stats = list(groups), random.Random(seed), []
    for _ in range(n_boot):
        picked = [v for k in rng.choices(keys, k=len(keys)) for v in groups[k]]
        stats.append(sum(picked) / len(picked))
    stats.sort()
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot)]


def compare(machine, a, b):
    """Same requests, two conditions, per model: B minus A, with a paired cluster interval."""
    for model in ("lfm3", "minicpm5"):
        ca, cb = correctness(machine, model, a), correctness(machine, model, b)
        if not ca or not cb:
            continue
        ids = sorted(set(ca) & set(cb))
        diff = {i: int(cb[i]) - int(ca[i]) for i in ids}
        lo, hi = paired_ci(ids, diff)
        only_a = sum(ca[i] and not cb[i] for i in ids)
        only_b = sum(cb[i] and not ca[i] for i in ids)
        print(f"{model:<9} {b} minus {a}: {sum(diff.values()) / len(ids):+.1%} [{lo:+.1%}, {hi:+.1%}] "
              f"on {len(ids)} requests (right only in {a}: {only_a}, only in {b}: {only_b})")


def compare_drop(machine, a, b):
    """Does one model lose more than the other from condition A to B? Paired on the same requests."""
    drops = {}
    for model in ("lfm3", "minicpm5"):
        ca, cb = correctness(machine, model, a), correctness(machine, model, b)
        if ca and cb:
            drops[model] = {i: int(cb[i]) - int(ca[i]) for i in set(ca) & set(cb)}
    if len(drops) < 2:
        print("Need both models for", a, "and", b)
        return
    ids = sorted(set(drops["lfm3"]) & set(drops["minicpm5"]))
    diff = {i: drops["lfm3"][i] - drops["minicpm5"][i] for i in ids}
    lo, hi = paired_ci(ids, diff)
    for model in drops:
        print(f"{model:<9} {a} to {b}: {sum(drops[model][i] for i in ids) / len(ids):+.1%}")
    print(f"LFM3's change minus MiniCPM5's: {sum(diff.values()) / len(ids):+.1%} [{lo:+.1%}, {hi:+.1%}]")


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


def describe(model, cell):
    """A readable condition name for Opik."""
    size, order = parse_cell(cell)
    if cell == "trap_removed":
        return "in context, 107 tools, trap removed"
    if cell == "deferred":
        return "deferred tools, model decides to search"
    if cell == "deferred_forced":
        return "harness search, " + ("Liquid format" if model == "lfm3" else "tool-list format")
    if cell == "deferred_forced_plain":
        return "harness search, tool-list format"
    if cell.startswith("followup_"):
        return f"follow-up turn after a lookup, {size} tools"
    return f"in context, {size} tools" + (f" (order {order})" if size in (36, 72) else "")


def log_to_opik(machine, only=None, project="lfm3-toolscale"):
    import opik
    from opik.exceptions import ExperimentNotFound
    from opik.evaluation import evaluate
    from opik.evaluation.metrics import base_metric, score_result

    class FieldIs(base_metric.BaseMetric):
        """1 when an output field equals a value: one metric per failure label or substitution kind."""
        def __init__(self, field, target):
            super().__init__(name=target)
            self.field, self.target = field, target

        def score(self, **kwargs):
            return score_result.ScoreResult(name=self.name, value=float(kwargs.get(self.field) == self.target))

    class Number(base_metric.BaseMetric):
        def __init__(self, field):
            super().__init__(name=field)
            self.field = field

        def score(self, **kwargs):
            return score_result.ScoreResult(name=self.name, value=float(kwargs.get(self.field) or 0))

    metrics = ([FieldIs("label", lab) for lab in LABELS]
               + [FieldIs("substitution", k) for k in ("sub_lookup", "sub_trap", "sub_group", "sub_unrelated")]
               + [Number("prompt_seconds"), Number("prompt_tokens"),
                  Number("decode_tokens"), Number("decode_seconds"), Number("decode_tok_s")])

    client = opik.Opik()
    dataset = client.get_or_create_dataset(
        "toolscale-eval-set", description="274 English requests from LiquidAI/phone-assistant-tools (Apache 2.0)")
    dataset.insert([{"request_id": r["id"], "request": r["request"], "category": r["category"],
                     "expected_calls": r["calls"]} for r in rows.values()])
    items = dataset.get_items()
    if items and "id" not in items[0]:
        sys.exit("Opik dataset items have no 'id' field in this SDK version; update the mapping in log_to_opik.")
    item_ids = {item["request_id"]: item["id"] for item in items}

    for f, recs in load(machine).items():
        if only and only not in str(f):
            continue
        first = recs[0]
        sample = "latency" if "_latency_" in f.stem else "all"
        condition = describe(first["model"], first["cell"])
        name = f"{first['model']} | {condition} | {sample} | {machine}"
        try:
            exists = bool(client.get_experiments_by_name(name))
        except ExperimentNotFound:
            exists = False
        if exists:
            print(f"Already logged, skipping: {name}")
            continue
        by_id = {r["id"]: r for r in recs}

        def task(item):
            rec = by_id[item["request_id"]]
            lab = label(rec)
            return {"calls": json.dumps(rec["calls"], ensure_ascii=False), "text": rec.get("content") or "",
                    "label": lab, "substitution": substitution_kind(rec) if lab == "substituted_call" else "",
                    "found_by_search": ", ".join(rec.get("found") or []),
                    "prompt_seconds": (rec.get("prompt_ms") or 0) / 1000,
                    "prompt_tokens": rec.get("prompt_tokens") or 0,
                    "decode_tokens": rec.get("completion_tokens") or 0,
                    "decode_seconds": (rec.get("decode_ms") or 0) / 1000,
                    "decode_tok_s": rec.get("decode_tok_per_s") or 0}

        size, order = parse_cell(first["cell"])
        evaluate(dataset=dataset, task=task, scoring_metrics=metrics, experiment_name=name,
                 project_name=project,
                 experiment_config={"model": first["model"], "condition": condition, "cell": first["cell"],
                                    "catalog_size": size, "arrival_order": order, "sample": sample,
                                    "machine": machine, "decoding": first["sampling"], "server": first["server"]},
                 experiment_tags=[first["model"], machine, f"{size} tools", condition.split(",")[0]],
                 dataset_item_ids=[item_ids[rid] for rid in by_id], verbose=0)
        print(f"Logged: {name}")


parser = argparse.ArgumentParser()
parser.add_argument("--machine")
parser.add_argument("--opik", action="store_true")
parser.add_argument("--only", help="with --opik, log only result files whose path contains this text")
parser.add_argument("--project", default="lfm3-toolscale", help="with --opik, the Opik project for traces")
parser.add_argument("--agreement", nargs=2, metavar=("A", "B"))
parser.add_argument("--pairs", action="store_true", help="list the most common substitutions per file")
parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="paired comparison of two cells, per model")
parser.add_argument("--compare-drop", nargs=2, metavar=("A", "B"), help="does one model change more from A to B?")
args = parser.parse_args()
if args.agreement:
    agreement(*args.agreement)
if args.compare:
    compare(args.machine or "gpu", *args.compare)
if args.compare_drop:
    compare_drop(args.machine or "gpu", *args.compare_drop)
if args.machine and not (args.compare or args.compare_drop):
    summarize(args.machine)
    if args.pairs:
        print()
        substitution_pairs(args.machine)
        trap_comparison(args.machine)
    if args.opik:
        log_to_opik(args.machine, args.only, args.project)
