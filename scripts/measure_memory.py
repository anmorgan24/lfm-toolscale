"""Measure peak memory of llama-server for each model and catalog size.

For each model and size, starts a server with a context window just large enough
for that catalog's prompt, records resident memory (RSS) after loading, sends one
request while sampling RSS every 50 ms, and records the peak. This follows the
memory axis in Liquid's LFM2 technical report (peak RSS at batch size 1).

Run on the laptop with nothing else running (about 15 minutes):
  python scripts/measure_memory.py
"""
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from toolscale.client import GREEDY, complete  # noqa: E402
from toolscale.data import load_catalog  # noqa: E402
from toolscale.evalset import sample_ids  # noqa: E402
from toolscale.prompt import build_messages, select_tools  # noqa: E402

LLAMA = os.environ.get("LLAMA", str(Path.home() / "Comet/llama.cpp/build/bin/llama-server"))
MODELS_DIR = Path(os.environ.get("MODELS", str(Path.home() / "Comet/models")))
THREADS = os.environ.get("THREADS", "4")
MACHINE = os.environ.get("MACHINE", "laptop")
FILES = {"lfm3": "LFM3-1.7B-Q4_K_M.gguf", "minicpm5": "MiniCPM5-2B-Q4_K_M.gguf"}
SIZES = ["18", "36", "72", "108"]
PORT = 8090


def prompt_tokens(model, size):
    """Mean prompt length from the latency runs, to size the context window."""
    path = ROOT / "results" / MACHINE / model / f"{size}_latency_greedy.jsonl"
    recs = [json.loads(line) for line in open(path)]
    return round(sum(r["prompt_tokens"] for r in recs) / len(recs))


def start_server(model, ctx, log_path):
    cmd = [LLAMA, "-m", str(MODELS_DIR / FILES[model]), "--port", str(PORT), "-c", str(ctx),
           "--parallel", "1", "-t", THREADS]
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            if requests.get(f"http://localhost:{PORT}/health", timeout=1).ok:
                return proc
        except requests.RequestException:
            pass
        if proc.poll() is not None:
            break
        time.sleep(1)
    proc.kill()
    sys.exit(f"Server for {model} did not start; see {log_path}")


def rss_mb(proc):
    return psutil.Process(proc.pid).memory_info().rss / 2**20


def main():
    tools, _ = load_catalog()
    catalogs = json.loads((ROOT / "data" / "catalogs.json").read_text())
    rows = {r["id"]: r for r in map(json.loads, open(ROOT / "data" / "eval_set.jsonl"))}
    request = rows[sample_ids(rows.values(), "latency")[0]]
    out_dir = ROOT / "results" / MACHINE
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    results = []
    for model in FILES:
        for size in SIZES:
            tokens = prompt_tokens(model, size)
            ctx = math.ceil((tokens + 1024) / 1024) * 1024  # prompt plus room for the reply
            log_path = out_dir / "logs" / f"memory_{model}_{size}.log"
            proc = start_server(model, ctx, log_path)
            time.sleep(2)
            after_load = rss_mb(proc)
            peak = [after_load]
            done = threading.Event()

            def sample():
                while not done.is_set():
                    peak[0] = max(peak[0], rss_mb(proc))
                    time.sleep(0.05)

            sampler = threading.Thread(target=sample)
            sampler.start()
            out = complete(build_messages(request), select_tools(tools, catalogs[size]),
                           sampling=GREEDY, url=f"http://localhost:{PORT}")
            done.set()
            sampler.join()
            proc.terminate()
            proc.wait()
            kv_lines = [line.strip() for line in open(log_path) if "KV" in line and "MiB" in line]
            row = {"model": model, "size": int(size), "ctx": ctx, "prompt_tokens": out["prompt_tokens"],
                   "rss_after_load_mb": round(after_load), "peak_rss_mb": round(peak[0]),
                   "kv_log": " | ".join(kv_lines)[:300]}
            results.append(row)
            print(f"{model:<9} {size:>3} tools  ctx {ctx:>6}  load {row['rss_after_load_mb']:>5} MB"
                  f"  peak {row['peak_rss_mb']:>5} MB")
    path = out_dir / "memory.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
