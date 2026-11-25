"""Follow-up turn: when the model looked something up instead of acting, does it act next?

Takes the requests in a finished cell where the model's substitution was a lookup
(a read-only tool such as list_alarms or search_contacts in place of an action),
returns a minimal result echoing the request's details, and gives the model one
more turn with the same tools. Outputs go to results/<machine>/<model>/followup_<cell>_greedy.jsonl.

  python scripts/run_followup.py --model lfm3 --cell 108 --machine gpu --url http://localhost:8091
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from toolscale.client import GREEDY, complete, server_info  # noqa: E402
from toolscale.data import load_catalog  # noqa: E402
from toolscale.deferred import assistant_turn  # noqa: E402
from toolscale.prompt import build_messages, select_tools  # noqa: E402
from toolscale.scoring import Scorer, is_lookup  # noqa: E402

MODEL_FILES = {"lfm3": "LFM3-1.7B", "minicpm5": "MiniCPM5-2B"}
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODEL_FILES), required=True)
parser.add_argument("--cell", default="108")
parser.add_argument("--machine", required=True)
parser.add_argument("--url", default="http://localhost:8080")
args = parser.parse_args()

info = server_info(args.url)
if MODEL_FILES[args.model] not in (info["model_path"] or ""):
    sys.exit(f"Server is running {info['model_path']}, not {args.model}.")

tools, kinds = load_catalog()
scorer = Scorer(tools, kinds)
rows = {r["id"]: r for r in map(json.loads, open(ROOT / "data" / "eval_set.jsonl"))}
catalog = json.loads((ROOT / "data" / "catalogs.json").read_text())[args.cell]
schemas = select_tools(tools, catalog)
source = ROOT / "results" / args.machine / args.model / f"{args.cell}_all_greedy.jsonl"
out_path = source.with_name(f"followup_{args.cell}_greedy.jsonl")
done = {json.loads(line)["id"] for line in open(out_path)} if out_path.exists() else set()


def mock_result(gold):
    """A minimal 'found it' result: the text and time details from the request's gold call."""
    details = {k: v for c in gold for k, v in (c.get("arguments") or {}).items()
               if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
    return json.dumps({"status": "ok", "results": [details]}, ensure_ascii=False)


todo = []
for rec in map(json.loads, open(source)):
    gold = json.loads(rows[rec["id"]]["calls"])
    label = scorer.failure_label(gold, rec["calls"], catalog, rec["format_error"])
    gold_names = {c["name"] for c in gold}
    wrong = [c["name"] for c in rec["calls"] if c["name"] not in gold_names]
    if label == "substituted_call" and wrong and all(is_lookup(n) for n in wrong) and rec["id"] not in done:
        todo.append((rec, gold))
print(f"{args.model} | follow-up on cell {args.cell} | {len(done)} done, {len(todo)} to go")

with open(out_path, "a") as f:
    for i, (rec, gold) in enumerate(todo, 1):
        messages = build_messages(rows[rec["id"]]) + [assistant_turn(rec["calls"])] + [
            {"role": "tool", "tool_call_id": f"call_{j}", "content": mock_result(gold)}
            for j in range(len(rec["calls"]))]
        out = complete(messages, schemas, sampling=GREEDY, url=args.url)
        f.write(json.dumps({"id": rec["id"], "model": args.model, "cell": f"followup_{args.cell}",
                            "source_cell": args.cell, "first_calls": rec["calls"], "sampling": "greedy",
                            "machine": args.machine, "server": info, **out}, ensure_ascii=False) + "\n")
        f.flush()
        print(f"  {i}/{len(todo)} {rec['id']} -> {[c['name'] for c in out['calls']]}")
print(f"Saved to {out_path}")
