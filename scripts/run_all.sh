#!/usr/bin/env bash
# Run a set of cells for both models, starting and stopping llama-server for each.
#
# Laptop, overnight latency samples:
#   caffeinate -i ./scripts/run_all.sh latency
# GPU, pilot accuracy:
#   MACHINE=gpu EXTRA="-ngl 99" CELLS="18 108" ./scripts/run_all.sh pilot
# GPU, deferred tools and the follow-up turn on the 108-tool cell:
#   MACHINE=gpu EXTRA="-ngl 99" PORT=8091 CELLS="deferred" FOLLOWUP=108 ./scripts/run_all.sh all
# GPU, full accuracy (three arrival orders for 36 and 72):
#   MACHINE=gpu EXTRA="-ngl 99" CELLS="18 36 72 108 36s1 72s1 36s2 72s2 trap_removed" ./scripts/run_all.sh all
set -euo pipefail

SAMPLE=${1:-latency}
MACHINE=${MACHINE:-laptop}
CELLS=${CELLS:-"18 36 72 108"}
LLAMA=${LLAMA:-$HOME/Comet/llama.cpp/build/bin/llama-server}
MODELS=${MODELS:-$HOME/Comet/models}
THREADS=${THREADS:-4}
EXTRA=${EXTRA:-}        # "-ngl 99" on a GPU, to put every layer on it
PORT=${PORT:-8080}
FOLLOWUP=${FOLLOWUP:-}  # a finished cell, e.g. 108, to run the follow-up turn on

mkdir -p results/logs
for model in lfm3 minicpm5; do
  case $model in
    lfm3) file=LFM3-1.7B-Q4_K_M.gguf ;;
    minicpm5) file=MiniCPM5-2B-Q4_K_M.gguf ;;
  esac
  echo "== Starting $model"
  # 32k context: room for the longest prompts (about 9,600 tokens for MiniCPM5 at 108 tools).
  "$LLAMA" -m "$MODELS/$file" --port "$PORT" -c 32768 --parallel 1 -t "$THREADS" $EXTRA \
    > "results/logs/server_${MACHINE}_${model}.log" 2>&1 &
  pid=$!
  trap 'kill $pid 2>/dev/null' EXIT
  # Wait up to 2 minutes for the server; stop with its log if it dies or never starts.
  for i in $(seq 1 60); do
    curl -sf "http://localhost:$PORT/health" > /dev/null && break
    if ! kill -0 $pid 2>/dev/null || [ "$i" = 60 ]; then
      echo "Server for $model did not start. Last log lines:"
      tail -20 "results/logs/server_${MACHINE}_${model}.log"
      exit 1
    fi
    sleep 2
  done
  for cell in $CELLS; do
    python scripts/run_cell.py --model "$model" --cell "$cell" --sample "$SAMPLE" \
      --machine "$MACHINE" --url "http://localhost:$PORT"
  done
  if [ -n "$FOLLOWUP" ]; then
    python scripts/run_followup.py --model "$model" --cell "$FOLLOWUP" \
      --machine "$MACHINE" --url "http://localhost:$PORT"
  fi
  kill $pid; wait $pid 2>/dev/null || true
done
echo "== Done. Score with: python scripts/score.py --machine $MACHINE"
