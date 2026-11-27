"""Score predicted tool calls against gold calls.

An independent implementation of the comparison rules for LiquidAI/phone-assistant-tools
(Apache 2.0): each tool argument has a kind in the dataset's scoring.json, and each kind
has a matching rule. The rules follow those used in Liquid's GRPO cookbook notebook, so
results are comparable with its evaluation; the code is written separately.
failure_label adds this project's failure categories on top.
"""
import ast
import math
import operator
import re
import unicodedata
from collections import Counter
from datetime import date, datetime, time
from difflib import SequenceMatcher

LOOKUP_PREFIXES = ("list_", "search_", "get_", "read_")

NAME_OVERLAP = 0.5     # a shorter name counts if it is contained and covers at least half
TEXT_SIMILARITY = 0.6  # free text counts if difflib's ratio reaches this


def is_lookup(name):
    """Read-only tools: they list, search, get, or read, and change nothing."""
    return bool(name) and name.startswith(LOOKUP_PREFIXES)


# ---------- normalization ----------

def _fold(value):
    """Case-fold, strip accents, and turn punctuation, symbols and spacing into single spaces."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    decomposed = unicodedata.normalize("NFD", text)
    no_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    recomposed = unicodedata.normalize("NFC", no_marks)
    spaced = "".join(" " if unicodedata.category(ch)[0] in ("P", "S", "Z") else ch for ch in recomposed)
    return " ".join(spaced.split())


def _covers(a, b):
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return bool(short) and short in long_ and len(short) >= NAME_OVERLAP * len(long_)


def _same_name(gold, pred):
    g, p = _fold(gold), _fold(pred)
    return g == p or _covers(g, p)


def _similar_text(gold, pred):
    g, p = _fold(gold), _fold(pred)
    if g == p or _covers(g, p):
        return True
    return bool(g) and bool(p) and SequenceMatcher(None, g, p).ratio() >= TEXT_SIMILARITY


# ---------- dates, times, arithmetic ----------

_WHEN_FORMS = {
    "datetime": (re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?"), datetime),
    "date": (re.compile(r"\d{4}-\d{2}-\d{2}"), date),
    "time": (re.compile(r"\d{1,2}:\d{2}(?::\d{2})?"), time),
}


def _when(value, kind):
    text = str(value).strip()
    pattern, cls = _WHEN_FORMS[kind]
    if not pattern.fullmatch(text):
        return None
    if kind == "time" and len(text) == 4:
        text = "0" + text
    return cls.fromisoformat(text)


_ARITH = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
_SIGN = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def evaluate_expression(expression):
    """Evaluate plain arithmetic safely; anything else returns None."""
    text = str(expression).replace("^", "**").replace("×", "*").replace("÷", "/")
    try:
        tree = ast.parse(text, mode="eval").body
    except (SyntaxError, ValueError):
        return None

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ARITH:
            return _ARITH[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SIGN:
            return _SIGN[type(node.op)](ev(node.operand))
        raise ValueError("not arithmetic")

    try:
        return float(ev(tree))
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None


# ---------- matching ----------

def _pairs_up(gold, pred, match):
    """True when every gold item can be paired with a distinct predicted item that matches it."""
    if len(gold) != len(pred):
        return False

    def assign(i, used):
        if i == len(gold):
            return True
        return any(j not in used and match(gold[i], pred[j]) and assign(i + 1, used | {j})
                   for j in range(len(pred)))

    return assign(0, frozenset())


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _strip_code(x):
    return re.sub(r"[\s\-().]", "", str(x).casefold())


def _match_list(item_match):
    return lambda g, p: isinstance(p, list) and _pairs_up(list(g), [str(x) for x in p], item_match)


MATCHERS = {
    "bool": lambda g, p: isinstance(p, bool) and g == p,
    "int": lambda g, p: _is_number(p) and abs(float(g) - float(p)) < 1e-6,
    "number": lambda g, p: _is_number(p) and abs(float(g) - float(p)) < 1e-6,
    "enum": lambda g, p: isinstance(p, str) and p.strip().lower() == g,
    "code": lambda g, p: isinstance(p, str) and _strip_code(g) == _strip_code(p),
    "datetime": lambda g, p: _when(p, "datetime") is not None and _when(g, "datetime") == _when(p, "datetime"),
    "date": lambda g, p: _when(p, "date") is not None and _when(g, "date") == _when(p, "date"),
    "time": lambda g, p: _when(p, "time") is not None and _when(g, "time") == _when(p, "time"),
    "expr": lambda g, p: (lambda a, b: a is not None and b is not None
                          and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9))(evaluate_expression(g),
                                                                             evaluate_expression(p)),
    "name": lambda g, p: isinstance(p, str) and _same_name(g, p),
    "text": lambda g, p: isinstance(p, str) and _similar_text(g, p),
    "list_enum": lambda g, p: isinstance(p, list) and sorted(g) == sorted(str(x).lower() for x in p),
    "list_name": _match_list(_same_name),
    "list_text": _match_list(_similar_text),
}


def values_match(kind, gold, pred):
    rule = MATCHERS.get(kind)
    if rule is None:
        return False
    try:
        return bool(rule(gold, pred))
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False


class Scorer:
    def __init__(self, tools, kinds):
        self.schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
        self.kinds = kinds

    def _normalized_args(self, name, arguments):
        """Keep required arguments and non-empty optional ones, fill schema defaults,
        and drop an end_date equal to date (a single day either way)."""
        schema = self.schemas[name]
        required = set(schema.get("required", []))
        args = {k: v for k, v in (arguments or {}).items()
                if k in required or v not in (None, "", [])}
        for key, prop in schema["properties"].items():
            if key not in args and "default" in prop:
                args[key] = prop["default"]
        if "end_date" in args and args["end_date"] == args.get("date"):
            args.pop("end_date")
        return args

    def same_call(self, gold, pred):
        name = gold["name"]
        if pred.get("name") != name:
            return False
        g = self._normalized_args(name, gold.get("arguments"))
        p = self._normalized_args(name, pred.get("arguments"))
        kinds = self.kinds[name]
        return g.keys() == p.keys() and all(k in kinds and values_match(kinds[k], g[k], p[k]) for k in g)

    def calls_match(self, gold, pred):
        """True when the prediction makes exactly the gold calls, in any order."""
        if any(call.get("name") not in self.schemas for call in pred):
            return False
        return _pairs_up(gold, pred, self.same_call)

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
