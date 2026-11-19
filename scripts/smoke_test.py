"""Run a few evaluation requests against the running llama-server.

Start the server first (see README), then:
  python scripts/smoke_test.py --size 18 --n 10
  python scripts/smoke_test.py --size 108 --n 10 --sampling recommended --model lfm3
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from toolscale.client import GREEDY, RECOMMENDED, complete, server_info  # noqa: E402
from toolscale.data import load_catalog  # noqa: E402
from toolscale.prompt import build_messages, select_tools  # noqa: E402
from toolscale.scoring import Scorer  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=18)
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--sampling", choices=["greedy", "recommended"], default="greedy")
parser.add_argument("--model", choices=list(RECOMMENDED), help="needed for --sampling recommended")
args = parser.parse_args()

tools, kinds = load_catalog()
scorer = Scorer(tools, kinds)
catalog = json.loads((ROOT / "data" / "catalogs.json").read_text())[str(args.size)]
tool_schemas = select_tools(tools, catalog)
rows = [json.loads(line) for line in open(ROOT / "data" / "eval_set.jsonl")][: args.n]
sampling = GREEDY if args.sampling == "greedy" else RECOMMENDED[args.model]

print("Server:", server_info())
print(f"Catalog: {args.size} tools | sampling: {sampling}\n")
labels = Counter()
for r in rows:
    out = complete(build_messages(r), tool_schemas, sampling=sampling)
    gold = json.loads(r["calls"])
    label = scorer.failure_label(gold, out["calls"], catalog, out["format_error"])
    labels[label] += 1
    shown = json.dumps(out["calls"], ensure_ascii=False) if out["calls"] else f"(text) {out['content'][:80]!r}"
    print(f"{label:<18} {r['category']:<10} prompt {out['prompt_tokens']:>5} tok, "
          f"cached {out['cached_tokens']}, {out['prompt_ms']:.0f} ms | {r['request'][:60]!r}")
    print(f"{'':<18} -> {shown[:150]}")
print("\n", dict(labels))
