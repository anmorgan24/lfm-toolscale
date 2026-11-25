"""Check that llama.cpp renders LFM3's deferred tools the way the chat template intends.

Start an LFM3 server first, then:
  python scripts/check_deferred_template.py --url http://localhost:8080

Uses /apply-template, which returns the formatted prompt without running the model.
"""
import argparse
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from toolscale.data import load_catalog  # noqa: E402
from toolscale.deferred import BM25, assistant_turn, deferred_tools, search_result  # noqa: E402
from toolscale.prompt import build_messages  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8080")
args = parser.parse_args()

tools, _ = load_catalog()
by_name = {t["function"]["name"]: t for t in tools}
row = json.loads(open(ROOT / "data" / "eval_set.jsonl").readline())
hits = BM25(tools).search("delete alarm")
messages = build_messages(row, deferred=True) + [
    assistant_turn([{"name": "search_tools", "arguments": {"query": "delete alarm"}}]),
    {"role": "tool", "tool_call_id": "call_0", "content": search_result(hits, by_name, native=True)}]

resp = requests.post(f"{args.url}/apply-template", json={
    "messages": messages, "tools": deferred_tools(tools),
    "chat_template_kwargs": {"enable_thinking": False}}, timeout=60)
resp.raise_for_status()
prompt = resp.json()["prompt"]
system = prompt.split("<|im_end|>")[0]

print("Loaded tools in the system prompt (expect all True):")
for name in ("translate_text", "calculate", "search_tools"):
    print(f"  {name}: {name in system}")
print("Deferred tools kept out of the system prompt (expect all True):")
for name in ("delete_alarm", "send_money", "list_alarms"):
    print(f"  {name}: {name not in system}")
print("defer_loading never shown to the model (expect True):", "defer_loading" not in prompt)
print("Search result wrapped as a tool response (expect True):",
      "<tool_response>\n<additional_tools>" in prompt)
start = prompt.find("<tool_response>")
print("\nRendered search result (first 600 characters):\n" + prompt[start:start + 600])
