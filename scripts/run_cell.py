"""Run one experiment cell against a running llama-server and save raw outputs.

Outputs go to results/<machine>/<model>/<cell>_<sample>_<sampling>.jsonl, one line
per request. Runs resume: requests already in the file are skipped. Scoring is a
separate step (scripts/score.py), so changing labels never requires a rerun.

Examples:
  python scripts/run_cell.py --model lfm3 --cell 18 --sample pilot --machine gpu
  python scripts/run_cell.py --model lfm3 --cell trap_removed --sample all --machine gpu
  python scripts/run_cell.py --model minicpm5 --cell 108 --sample latency --machine laptop
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from toolscale.client import GREEDY, RECOMMENDED, complete, server_info  # noqa: E402
from toolscale.data import load_catalog  # noqa: E402
from toolscale.evalset import sample_ids  # noqa: E402
from toolscale.deferred import BM25, run_deferred  # noqa: E402
from toolscale.prompt import build_messages, select_tools  # noqa: E402

MODEL_FILES = {"lfm3": "LFM3-1.7B", "minicpm5": "MiniCPM5-2B"}

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODEL_FILES), required=True)
parser.add_argument("--cell", required=True, help="18, 36, 72, 108, an extra order like 36s1 or 72s2, trap_removed, deferred, deferred_forced, or deferred_forced_plain")
parser.add_argument("--sample", choices=["all", "pilot", "latency"], default="all")
parser.add_argument("--sampling", choices=["greedy", "recommended"], default="greedy")
parser.add_argument("--machine", required=True, help="where this runs, e.g. laptop or gpu")
parser.add_argument("--url", default="http://localhost:8080")
args = parser.parse_args()

# Guard against running the wrong model: the server must be serving the named one.
info = server_info(args.url)
if MODEL_FILES[args.model] not in (info["model_path"] or ""):
    sys.exit(f"Server is running {info['model_path']}, not {args.model}. Restart it with the right model.")

tools, _ = load_catalog()
rows = {r["id"]: r for r in map(json.loads, open(ROOT / "data" / "eval_set.jsonl"))}
catalogs = json.loads((ROOT / "data" / "catalogs.json").read_text())

if args.cell == "trap_removed":
    traps = json.loads((ROOT / "data" / "lookalike_traps.json").read_text())
    removed = json.loads((ROOT / "data" / "trap_removed_catalogs.json").read_text())
    jobs = {rid: removed[e["trap"]] for rid, e in traps.items() if e["trap"] and rid in rows}
    if args.sample != "all":
        keep = set(sample_ids(rows.values(), args.sample))
        jobs = {rid: c for rid, c in jobs.items() if rid in keep}
elif args.cell in ("deferred", "deferred_forced", "deferred_forced_plain"):
    jobs = {rid: catalogs["108"] for rid in sample_ids(rows.values(), args.sample)}
else:
    if args.cell not in catalogs:
        sys.exit(f"Unknown cell {args.cell}; choose from {list(catalogs)} or trap_removed")
    catalog = catalogs[args.cell]
    jobs = {rid: catalog for rid in sample_ids(rows.values(), args.sample)}

sampling = GREEDY if args.sampling == "greedy" else RECOMMENDED[args.model]
out_path = ROOT / "results" / args.machine / args.model / f"{args.cell}_{args.sample}_{args.sampling}.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)
done = {json.loads(line)["id"] for line in open(out_path)} if out_path.exists() else set()
todo = [rid for rid in jobs if rid not in done]
print(f"{args.model} | cell {args.cell} | {args.sample} | {args.sampling} | {len(done)} done, {len(todo)} to go")
print(f"Server: {info}")

index = BM25(tools)
start = time.time()
with open(out_path, "a") as f:
    for i, rid in enumerate(todo, 1):
        catalog = jobs[rid]
        if args.cell.startswith("deferred"):
            # LFM3 gets Liquid's deferred format, except in the _plain cell, which gives it
            # MiniCPM5's format (found tools added to the tool list) to separate format from model.
            out = run_deferred(rows[rid], tools, index, sampling, args.url,
                               native=args.model == "lfm3" and not args.cell.endswith("_plain"),
                               forced=args.cell.startswith("deferred_forced"))
        else:
            out = complete(build_messages(rows[rid]), select_tools(tools, catalog), sampling=sampling, url=args.url)
        record = {"id": rid, "model": args.model, "cell": args.cell, "catalog_size": len(catalog),
                  "sampling": args.sampling, "machine": args.machine, "server": info, **out}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        elapsed = time.time() - start
        print(f"  {i}/{len(todo)}  {rid}  {out['prompt_tokens']} tok  {out['wall_ms'] / 1000:.1f}s"
              f"  (about {elapsed / i * (len(todo) - i) / 60:.0f} min left)")
print(f"Saved to {out_path}")
