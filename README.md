# Tool catalogs on small on-device models

How should a small on-device model be given a large tool catalog, and does its
architecture change the answer? This repo compares all tools in context against
deferred tools with search, on LFM3-1.7B (hybrid) and MiniCPM5-2B (transformer),
as the catalog grows from 10 to 108 tools. Evaluation runs locally with
llama.cpp and is tracked in Opik.

Status: dataset inspection. Experiment code follows once the core tools are chosen.

## Data

[`LiquidAI/phone-assistant-tools`](https://huggingface.co/datasets/LiquidAI/phone-assistant-tools)
(Apache 2.0): 108 tool schemas, requests with verified gold calls, and
per-argument scoring rules.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/inspect_dataset.py
```
