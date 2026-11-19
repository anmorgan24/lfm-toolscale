"""Send one request to a running llama-server and return the parsed result."""
import json
import time

import requests

GREEDY = {"temperature": 0.0, "top_k": 1}
RECOMMENDED = {
    "lfm3": {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.1},
    "minicpm5": {"temperature": 1.0, "top_p": 0.95},
}
# Text that means the model tried to call a tool but llama.cpp didn't parse it.
CALL_MARKERS = ("<tool_call", "<|tool_call", "<function", '"arguments"')


def server_info(url="http://localhost:8080"):
    props = requests.get(f"{url}/props", timeout=10).json()
    return {"model_path": props.get("model_path"), "build": props.get("build_info")}


def complete(messages, tools, sampling=GREEDY, seed=0, url="http://localhost:8080",
             cache_prompt=False, max_tokens=512, timeout=600):
    body = {
        "messages": messages,
        "tools": tools,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": max_tokens,
        "seed": seed,
        "cache_prompt": cache_prompt,
        **sampling,
    }
    start = time.perf_counter()
    resp = requests.post(f"{url}/v1/chat/completions", json=body, timeout=timeout)
    wall_ms = (time.perf_counter() - start) * 1000
    resp.raise_for_status()
    data = resp.json()
    message = data["choices"][0]["message"]
    content = message.get("content") or ""
    calls, format_error = [], False
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args, format_error = None, True
        calls.append({"name": fn.get("name"), "arguments": args})
    if not calls and any(m in content for m in CALL_MARKERS):
        format_error = True
    timings = data.get("timings", {})
    return {
        "calls": calls,
        "content": content,
        "format_error": format_error,
        "finish_reason": data["choices"][0].get("finish_reason"),
        "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": data.get("usage", {}).get("completion_tokens"),
        "cached_tokens": timings.get("cache_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "decode_ms": timings.get("predicted_ms"),
        "decode_tok_per_s": timings.get("predicted_per_second"),
        "wall_ms": wall_ms,
    }
