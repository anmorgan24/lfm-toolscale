# Tool catalogs on small on-device models

How should a small on-device model be given a large tool catalog, and does its
architecture change the answer? This repo compares all tools in context against
deferred tools with search, on LFM3-1.7B (hybrid) and MiniCPM5-2B (transformer),
as the catalog grows from 10 to 108 tools. Evaluation runs locally with
llama.cpp and is tracked in Opik.

Status: experiment design. `scripts/build_experiment.py` builds the nested
catalogs (18, 36, 72, 108 tools) and the evaluation set from the lookalike
groups in `data/lookalike_groups.json`.

## Data

[`LiquidAI/phone-assistant-tools`](https://huggingface.co/datasets/LiquidAI/phone-assistant-tools)
(Apache 2.0): 108 tool schemas, requests with verified gold calls, and
per-argument scoring rules.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/inspect_dataset.py
python scripts/build_experiment.py
```

## Inference setup

Models run locally with [llama.cpp](https://github.com/ggml-org/llama.cpp), on CPU only.

- llama.cpp build 12555, commit `c406f4367`
- Built with `cmake -B build -DGGML_METAL=OFF -DGGML_OPENMP=OFF`
  - Metal off: all runs use the CPU, so latency measurements are comparable.
  - OpenMP off: llama.cpp uses its own thread pool instead. (On the test
    machine, OpenMP's headers and library were mismatched.)
- Hardware: Intel MacBook Pro (x86_64), CPU only
- LFM3-1.7B requires llama.cpp build 11960 or newer.

To build the same version:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout c406f4367
cmake -B build -DGGML_METAL=OFF -DGGML_OPENMP=OFF
cmake --build build --config Release -j
```
