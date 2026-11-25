"""Score predicted tool calls against gold calls.

calls_match and its helpers reproduce the comparison in Liquid's notebook
(Liquid4All/cookbook, finetuning/notebooks/lfm3_grpo_with_trl.ipynb), with the
tool schemas and argument kinds passed in instead of read from globals.
failure_label adds this project's failure categories on top.
"""
import ast
import itertools
import math
import operator
import re
import unicodedata
from collections import Counter
from datetime import date, datetime, time
from difflib import SequenceMatcher


LOOKUP_PREFIXES = ("list_", "search_", "get_", "read_")


def is_lookup(name):
    """Read-only tools: they list, search, get, or read, and change nothing."""
    return bool(name) and name.startswith(LOOKUP_PREFIXES)


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = "".join(" " if unicodedata.category(c)[0] in "PSZ" else c for c in unicodedata.normalize("NFC", s))
    return " ".join(s.split())


def contained(a, b):
    short, long_ = sorted((a, b), key=len)
    return bool(short) and short in long_ and len(short) / len(long_) >= 0.5


def name_equal(a, b):
    a, b = norm(a), norm(b)
    return a == b or contained(a, b)


def text_similar(a, b):
    a, b = norm(a), norm(b)
    return a == b or contained(a, b) or (bool(a) and bool(b) and SequenceMatcher(None, a, b).ratio() >= 0.6)


def parse_when(value, kind):
    value = str(value).strip()
    patterns = {"datetime": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?", "date": r"\d{4}-\d{2}-\d{2}",
                "time": r"\d{1,2}:\d{2}(:\d{2})?"}
    if not re.fullmatch(patterns[kind], value):
        return None
    cls = {"datetime": datetime, "date": date, "time": time}[kind]
    return cls.fromisoformat(value.zfill(5) if kind == "time" and len(value) == 4 else value)


OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
             ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculate(expression):
    """Safely evaluate an arithmetic expression, or return None."""
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            return OPERATORS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp):
            return OPERATORS[type(node.op)](walk(node.operand))
        raise ValueError
    try:
        expr = str(expression).replace("^", "**").replace("×", "*").replace("÷", "/")
        return float(walk(ast.parse(expr, mode="eval").body))
    except Exception:
        return None


def same_list(gold, pred, equal):
    return len(gold) == len(pred) and any(
        all(equal(g, p) for g, p in zip(gold, perm)) for perm in itertools.permutations(pred))


def same_value(kind, gold, pred):
    try:
        if kind == "bool":
            return isinstance(pred, bool) and gold == pred
        if kind in ("int", "number"):
            return isinstance(pred, (int, float)) and not isinstance(pred, bool) and abs(float(gold) - float(pred)) < 1e-6
        if kind == "enum":
            return isinstance(pred, str) and gold == pred.strip().lower()
        if kind == "code":
            return isinstance(pred, str) and re.sub(r"[\s\-().]", "", str(gold).casefold()) == re.sub(r"[\s\-().]", "", pred.casefold())
        if kind in ("datetime", "date", "time"):
            return parse_when(pred, kind) is not None and parse_when(gold, kind) == parse_when(pred, kind)
        if kind == "expr":
            g, p = calculate(gold), calculate(pred)
            return g is not None and p is not None and math.isclose(g, p, rel_tol=1e-9, abs_tol=1e-9)
        if kind == "name":
            return isinstance(pred, str) and name_equal(gold, pred)
        if kind == "text":
            return isinstance(pred, str) and text_similar(gold, pred)
        if kind == "list_enum":
            return isinstance(pred, list) and sorted(gold) == sorted(str(x).lower() for x in pred)
        if kind == "list_name":
            return isinstance(pred, list) and same_list(gold, [str(x) for x in pred], name_equal)
        if kind == "list_text":
            return isinstance(pred, list) and same_list(gold, [str(x) for x in pred], text_similar)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False
    return False


class Scorer:
    def __init__(self, tools, kinds):
        self.schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
        self.kinds = kinds

    def canonical(self, name, arguments):
        """Fill schema defaults, drop empty optional values, and treat end_date == date as a single day."""
        schema = self.schemas[name]
        required = set(schema.get("required", []))
        out = {k: v for k, v in (arguments or {}).items() if k in required or v not in (None, "", [])}
        for k, prop in schema["properties"].items():
            if k not in out and "default" in prop:
                out[k] = prop["default"]
        if "end_date" in out and out.get("end_date") == out.get("date"):
            del out["end_date"]
        return out

    def same_call(self, gold, pred):
        if gold["name"] != pred.get("name"):
            return False
        g = self.canonical(gold["name"], gold.get("arguments"))
        p = self.canonical(gold["name"], pred.get("arguments"))
        kinds = self.kinds[gold["name"]]
        return set(g) == set(p) and all(k in kinds and same_value(kinds[k], g[k], p[k]) for k in g)

    def calls_match(self, gold, pred):
        """True when the prediction makes exactly the gold calls, in any order."""
        if len(gold) != len(pred) or any(call.get("name") not in self.schemas for call in pred):
            return False
        return any(all(self.same_call(g, p) for g, p in zip(gold, perm)) for perm in itertools.permutations(pred))

    def failure_label(self, gold, pred, catalog, format_error=False):
        """One label per output, checked in order. See the experiment plan's metrics table.

        catalog: the tool names offered in this request's prompt.
        """
        if format_error:
            return "format_error"
        if self.calls_match(gold, pred):
            return "correct"
        if gold and not pred:
            return "no_call"
        if not gold and pred:
            return "unneeded_call"
        pred_names = Counter(c.get("name") for c in pred)
        gold_names = Counter(c["name"] for c in gold)
        if any(n not in set(catalog) for n in pred_names):
            return "invented_function"
        extra = [n for n in pred_names if n not in gold_names]
        if extra:
            # Spurious: every gold tool was called, plus unneeded extra tools.
            # Substituted: a wrong tool took the place of a gold one (the lookalike failure).
            return "spurious_call" if all(pred_names[n] >= gold_names[n] for n in gold_names) else "substituted_call"
        if pred_names == gold_names:
            return "argument_error"
        if all(pred_names[n] <= gold_names[n] for n in pred_names):
            return "dropped_call"
        return "extra_call"
