"""Lookalike groups, the core tools, and nested catalogs of growing size."""
import json
import random
from pathlib import Path

from .data import tool_name

ROOT = Path(__file__).resolve().parents[2]

# Tools named in Liquid's system prompt. They are in every catalog, so the prompt
# never refers to a tool that is missing.
PROMPT_TOOLS = ["translate_text", "calculate", "convert_units",
                "convert_currency", "search_web", "get_news_headlines"]

# One tool per lookalike group, chosen for the most English lookalike requests.
# Their lookalikes stay outside the core, so growing the catalog adds traps.
CORE_TOOLS = ["send_money", "create_contact", "delete_alarm", "play_audiobook",
              "start_navigation", "take_photo", "search_notes", "log_workout",
              "set_volume", "complete_todo", "make_video_call", "send_sms"]

SIZES = [18, 36, 72, 108]
# Extra arrival orders for the middle sizes. 18 is always the core and 108 is
# always every tool, so only 36 and 72 depend on the order.
EXTRA_ORDERS = [1, 2]
MIDDLE_SIZES = [36, 72]


def load_groups(path=ROOT / "data" / "lookalike_groups.json"):
    groups = json.loads(Path(path).read_text())
    groups.pop("_note", None)
    return groups


def validate_groups(groups, tools):
    names = {tool_name(t) for t in tools}
    seen = {}
    for group, members in groups.items():
        for m in members:
            if m not in names:
                raise ValueError(f"{m} in group {group} is not in the catalog")
            if m in seen:
                raise ValueError(f"{m} is in both {seen[m]} and {group}")
            seen[m] = group
    return seen  # tool -> group


def lookalikes_of(name, groups_by_tool, groups):
    group = groups_by_tool.get(name)
    return set(groups[group]) - {name} if group else set()


def core(groups_by_tool):
    """Prompt tools plus core tools; checks the core holds one tool per group."""
    tools = PROMPT_TOOLS + CORE_TOOLS
    used = {}
    for t in CORE_TOOLS:
        g = groups_by_tool.get(t)
        if g in used:
            raise ValueError(f"{t} and {used[g]} are lookalikes; keep one in the core")
        used[g] = t
    return tools


def nested_catalogs(tools, core_tools, sizes=SIZES, seed=0):
    """Return {size: [tool names]} with each catalog containing the previous one.

    Non-core tools join in one fixed random order, so lookalikes of core tools
    enter gradually as the catalog grows. Within a prompt, tools keep their order
    in tools.json, as in Liquid's notebook, so the 108-tool catalog matches
    Liquid's own evaluation setup.
    """
    names = [tool_name(t) for t in tools]
    rest = [n for n in names if n not in core_tools]
    random.Random(seed).shuffle(rest)
    catalogs = {}
    for size in sizes:
        members = set(core_tools) | set(rest[: size - len(core_tools)])
        catalogs[size] = [n for n in names if n in members]
    return catalogs


def all_catalogs(tools, core_tools):
    """Order 0 under plain size keys ("36"), extra orders as "36s1", "72s2"."""
    catalogs = {str(k): v for k, v in nested_catalogs(tools, core_tools, seed=0).items()}
    for seed in EXTRA_ORDERS:
        for size, members in nested_catalogs(tools, core_tools, seed=seed).items():
            if size in MIDDLE_SIZES:
                catalogs[f"{size}s{seed}"] = members
    return catalogs


def parse_cell(cell):
    """'36s1' -> (36, 1); '72' -> (72, 0); 'trap_removed' -> (107, None)."""
    if cell == "trap_removed":
        return 107, None
    if cell.startswith("deferred"):
        return 108, None
    if cell.startswith("followup_"):
        return int(cell.split("_")[1]), None
    size, _, order = cell.partition("s")
    return int(size), int(order or 0)
