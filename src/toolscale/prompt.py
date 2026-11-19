"""Build chat messages exactly as Liquid's phone-assistant-tools notebook does.

Source: Liquid4All/cookbook, finetuning/notebooks/lfm3_grpo_with_trl.ipynb.
The system prompt is copied unchanged and used for both models.
"""
from datetime import datetime

SYSTEM_PROMPT = """You are a helpful assistant running on the user's phone. You can call the phone's tools.
Current date and time: {weekday}, {date} {time} ({timezone}). User locale: {locale}.

Rules:
- Call every tool needed to complete the request, with arguments taken from the request and the context above. Fill optional arguments only when the user gives them; never invent values.
- Resolve relative dates and times ("tomorrow", "on Friday", "in 20 minutes") against the current date and time. Times are local, in the formats the tool descriptions give.
- If the request needs a tool but a required detail is missing, ask a short clarifying question instead of calling a tool.
- Answer small talk and general-knowledge questions directly, without tools. Use translate_text, calculate, convert_units and convert_currency for translations, arithmetic and conversions, and search_web or get_news_headlines for current or local information.
- If no tool can do what is asked, say so briefly instead of calling a different tool. If only part of the request is possible, do that part and say what you could not do.
- Keep free-text arguments (messages, notes, search queries) in the user's language."""


def build_messages(example):
    now = datetime.fromisoformat(example["context"]["datetime"])
    system = SYSTEM_PROMPT.format(
        weekday=now.strftime("%A"), date=now.strftime("%Y-%m-%d"),
        time=now.strftime("%H:%M"), **example["context"],
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": example["request"]}]


def select_tools(tools, names):
    """Tool schemas for one catalog, in the given order."""
    by_name = {t["function"]["name"]: t for t in tools}
    return [by_name[n] for n in names]
