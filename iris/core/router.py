"""
iris/core/router.py
-------------------
Deterministic Request Classifier and Tool Handlers for IRIS Cloud Architecture.

Responsibilities:
- Classifies user requests using lightweight, deterministic Python logic (no LLM call).
- Evaluates exact calculations using a safe AST-based arithmetic evaluator.
- Answers simple date/time queries using system utilities.
- Retrieves conversation context from local Markdown session archives.
- Identifies complex reasoning tasks for Gemini Flash routing.
- Identifies everyday conversation for OpenRouter free-model routing.
- Detects user confirmation when degrading from unavailable models.
"""

from __future__ import annotations

import ast
import logging
import math
import operator
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from iris.services.archive import ConversationArchive, parse_session_markdown

logger = logging.getLogger(__name__)


class RouteType(str, Enum):
    CALCULATION = "calculation"
    DATETIME = "datetime"
    MEMORY = "memory"
    COMPLEX_REASONING = "complex_reasoning"
    CONVERSATION = "conversation"


@dataclass(frozen=True)
class RoutingDecision:
    route: RouteType
    reason: str
    direct_response: str | None = None
    target_provider: str = ""
    target_model: str = ""


# ── 1. Deterministic Safe Calculator ──────────────────────────────────────────

_ALLOWED_OPERATORS: dict[type[ast.AST], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_FUNCTIONS: dict[str, Any] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "pow": math.pow,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}

# Common algebraic expansions handled deterministically
_ALGEBRAIC_IDENTITIES: dict[str, str] = {
    r"\(a\s*\+\s*b\)\s*\^?\s*2": "(a + b)² = a² + 2ab + b²",
    r"\(a\s*-\s*b\)\s*\^?\s*2": "(a - b)² = a² - 2ab + b²",
    r"a\s*\^?\s*2\s*-\s*b\s*\^?\s*2": "a² - b² = (a - b)(a + b)",
    r"\(a\s*\+\s*b\)\s*\^?\s*3": "(a + b)³ = a³ + 3a²b + 3ab² + b³",
    r"\(a\s*-\s*b\)\s*\^?\s*3": "(a - b)³ = a³ - 3a²b + 3ab² - b³",
}


def _safe_eval_node(node: ast.AST, depth: int = 0) -> float | int:
    """Recursively evaluate an AST node restricting execution to safe math."""
    if depth > 25:
        raise ValueError("Expression exceeds maximum safe nesting depth")

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Non-numeric literal in expression")

    if isinstance(node, ast.UnaryOp):
        op_func = _ALLOWED_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval_node(node.operand, depth=depth + 1))

    if isinstance(node, ast.BinOp):
        op_func = _ALLOWED_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = _safe_eval_node(node.left, depth=depth + 1)
        right = _safe_eval_node(node.right, depth=depth + 1)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ZeroDivisionError("Division by zero")
        if isinstance(node.op, ast.Pow):
            if abs(left) > 1e6 or abs(right) > 1000:
                raise OverflowError("Exponentiation exceeds safe calculation bounds")
        return op_func(left, right)

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id.lower()
            func = _ALLOWED_FUNCTIONS.get(func_name)
            if callable(func):
                args = [_safe_eval_node(arg, depth=depth + 1) for arg in node.args]
                return func(*args)
        raise ValueError("Unauthorized or unknown function call")

    if isinstance(node, ast.Name):
        var_name = node.id.lower()
        if var_name in _ALLOWED_FUNCTIONS and not callable(_ALLOWED_FUNCTIONS[var_name]):
            return _ALLOWED_FUNCTIONS[var_name]
        raise ValueError(f"Undefined variable in expression: {node.id}")

    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def try_evaluate_calculation(query: str) -> str | None:
    """
    Attempt to evaluate an exact calculation or algebraic identity.
    Returns a clean, conversational string answer or None if not an exact calculation.
    """
    cleaned = query.strip()

    # 1. Check for standard algebraic expansions
    for pattern, expansion in _ALGEBRAIC_IDENTITIES.items():
        if re.search(pattern, cleaned, re.IGNORECASE):
            return f"The standard expansion is {expansion}."

    # 2. Percentage questions: e.g. "what is 15% of 200" or "15 percent of 200"
    pct_match = re.search(
        r"(?:what\s+is\s+|calculate\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d+(?:\.\d+)?)",
        cleaned,
        re.IGNORECASE,
    )
    if pct_match:
        pct = float(pct_match.group(1))
        base = float(pct_match.group(2))
        res = (pct / 100.0) * base
        res_str = f"{int(res)}" if res.is_integer() else f"{res:.4g}"
        return f"{pct:g}% of {base:g} is {res_str}."

    # 3. Square root: e.g. "square root of 144" or "sqrt of 25"
    sqrt_match = re.search(
        r"(?:what\s+is\s+(?:the\s+)?)?square\s+root\s+of\s+(\d+(?:\.\d+)?)",
        cleaned,
        re.IGNORECASE,
    )
    if sqrt_match:
        val = float(sqrt_match.group(1))
        res = math.sqrt(val)
        res_str = f"{int(res)}" if res.is_integer() else f"{res:.4g}"
        return f"The square root of {val:g} is {res_str}."

    # 4. Natural language arithmetic: "what is 25 times 4", "15 plus 27", "144 divided by 12"
    nl_arith = cleaned
    nl_arith = re.sub(r"^(?:what\s+is|what\'s|calculate|compute|solve)\s+", "", nl_arith, flags=re.IGNORECASE)
    nl_arith = nl_arith.rstrip("?. ")

    # Map words to symbols
    replacements = [
        (r"\btimes\b", "*"),
        (r"\bmultiplied\s+by\b", "*"),
        (r"\bdivided\s+by\b", "/"),
        (r"\bover\b", "/"),
        (r"\bplus\b", "+"),
        (r"\bminus\b", "-"),
        (r"\bto\s+the\s+power\s+of\b", "**"),
        (r"\^", "**"),
    ]
    for old, new in replacements:
        nl_arith = re.sub(old, new, nl_arith, flags=re.IGNORECASE)

    # Check if the candidate string contains only valid arithmetic characters/functions
    has_digit = any(ch.isdigit() for ch in nl_arith)
    has_op = any(op in nl_arith for op in ["+", "-", "*", "/", "%", "**"])
    allowed_expr_pattern = re.compile(r"^(?:[\d\s\+\-\*\/\%\.\(\)\,]|\b(?:sqrt|abs|round|pow|pi|e)\b)+$")

    if has_digit and has_op and allowed_expr_pattern.match(nl_arith):
        try:
            parsed = ast.parse(nl_arith, mode="eval")
            result = _safe_eval_node(parsed.body)
            if isinstance(result, float) and result.is_integer():
                res_display = str(int(result))
            elif isinstance(result, float):
                res_display = f"{result:.6g}"
            else:
                res_display = str(result)
            return f"The answer is {res_display}."
        except ZeroDivisionError:
            return "Division by zero is undefined."
        except OverflowError:
            return "The calculation exceeds allowable numerical bounds."
        except Exception:
            pass

    return None


# ── 2. Deterministic Date / Time Handler ──────────────────────────────────────

_TIME_PATTERNS = [
    r"\bwhat\s+time\s+is\s+it\b",
    r"\bwhat\s+is\s+the\s+time\b",
    r"\bwhat\'s\s+the\s+time\b",
    r"\bcurrent\s+time\b",
    r"\btell\s+me\s+the\s+time\b",
    r"\btime\s+right\s+now\b",
]

_DATE_PATTERNS = [
    r"\bwhat\s+is\s+today\'?s\s+date\b",
    r"\bwhat\'s\s+today\'?s\s+date\b",
    r"\bwhat\s+is\s+the\s+date\s+today\b",
    r"\bwhat\s+is\s+the\s+date\b",
    r"\bwhat\s+date\s+is\s+it\b",
    r"\btoday\'?s\s+date\b",
]

_DAY_PATTERNS = [
    r"\bwhat\s+day\s+of\s+the\s+week\s+is\s+it\b",
    r"\bwhat\s+day\s+is\s+it\s+today\b",
    r"\bwhat\s+day\s+is\s+it\b",
    r"\bwhat\s+day\s+is\s+today\b",
]

_YEAR_MONTH_PATTERNS = [
    (r"\bwhat\s+year\s+is\s+(?:it|this)\b", "year"),
    (r"\bwhat\s+month\s+is\s+(?:it|this)\b", "month"),
]


def try_evaluate_datetime(query: str, now_dt: datetime | None = None) -> str | None:
    """
    Evaluate simple date/time questions using system utilities.
    Returns a natural English answer or None if not a current date/time query.
    Rejects questions with location qualifiers, past/future tense, or event targets.
    """
    now = now_dt or datetime.now()
    cleaned = query.strip().lower()

    # Disqualify queries targeting external locations (e.g. "what time is it in Tokyo")
    if re.search(r"\b(?:in|at|for)\s+[a-z]+", cleaned) and not re.search(r"\b(?:right\s+now|here)\b", cleaned):
        return None

    # Disqualify queries asking about historical/future events (e.g. "what is the date of Easter", "what year was it created")
    if re.search(r"\b(?:was|will\s+be|were|did|born|released|happened|occurred)\b", cleaned):
        return None
    if re.search(r"\b(?:date|time|year|month)\s+of\b", cleaned):
        return None

    for pattern in _TIME_PATTERNS:
        if re.search(pattern, cleaned):
            # Format time cleanly e.g. "3:45 PM"
            time_str = now.strftime("%I:%M %p").lstrip("0")
            return f"It is currently {time_str}."

    for pattern in _DATE_PATTERNS:
        if re.search(pattern, cleaned):
            # Format date cleanly e.g. "Thursday, 5 November 2026"
            day_num = now.strftime("%d").lstrip("0")
            month_name = now.strftime("%B")
            weekday = now.strftime("%A")
            year = now.strftime("%Y")
            return f"Today is {weekday}, {day_num} {month_name} {year}."

    for pattern in _DAY_PATTERNS:
        if re.search(pattern, cleaned):
            weekday = now.strftime("%A")
            return f"Today is {weekday}."

    for pattern, kind in _YEAR_MONTH_PATTERNS:
        if re.search(pattern, cleaned):
            if kind == "year":
                return f"The year is {now.strftime('%Y')}."
            elif kind == "month":
                return f"The current month is {now.strftime('%B')}."

    return None


# ── 3. Deterministic Memory / Archive Retrieval ───────────────────────────────

_MEMORY_PATTERNS = [
    r"\bwhat\s+did\s+we\s+talk\s+about\b",
    r"\bwhat\s+did\s+we\s+discuss\b",
    r"\bwhat\s+was\s+our\s+last\s+(?:conversation|session|topic)\b",
    r"\bsummarize\s+(?:our\s+)?(?:last|previous)\s+session\b",
    r"\brecall\s+(?:our\s+)?previous\s+conversation\b",
    r"\bwhat\s+was\s+the\s+last\s+thing\s+i\s+(?:asked|said)\b",
    r"\bwhat\s+did\s+i\s+(?:just\s+)?say\s+before\b",
    r"\bsearch\s+(?:my\s+)?past\s+sessions\b",
    r"\bcheck\s+(?:my\s+)?previous\s+session\b",
]


def try_evaluate_memory(
    query: str,
    archive_dir: Path | None = None,
    persona_name: str = "IRIS",
) -> str | None:
    """
    Retrieve past conversational context from local Markdown session files.
    """
    cleaned = query.strip()
    is_memory_query = any(re.search(pat, cleaned, re.IGNORECASE) for pat in _MEMORY_PATTERNS)
    if not is_memory_query:
        return None

    if archive_dir is None or not archive_dir.exists():
        return "I do not have access to any recorded local conversation sessions yet."

    archive = ConversationArchive(archive_dir=archive_dir, persona_name=persona_name)
    session_files = archive.get_session_files()
    if not session_files:
        return "I don't have any prior conversation sessions recorded in my local archive."

    # Look at the most recent completed session file
    latest_file = session_files[-1]
    try:
        content = latest_file.read_text(encoding="utf-8")
        turns = parse_session_markdown(content, persona_name=persona_name)
    except Exception as exc:
        logger.warning("Could not read archive file %s: %s", latest_file, exc)
        return "I encountered an error reading the local conversation archive."

    if not turns:
        return "Our previous session was recorded, but contains no completed conversational turns."

    # Extract the last few turns
    user_turns = [t.content for t in turns if t.role == "user"]
    assistant_turns = [t.content for t in turns if t.role == "assistant"]

    cleaned_lower = cleaned.lower()
    if "last thing i asked" in cleaned_lower or "what did i say" in cleaned_lower:
        if user_turns:
            return f'In your last recorded turn, you asked: "{user_turns[-1]}".'
        return "I couldn't find a recorded user turn in that session."

    # General recap of last session
    sample_queries = user_turns[-3:] if len(user_turns) >= 3 else user_turns
    queries_summary = "; ".join(f'"{q}"' for q in sample_queries)
    return (
        f"In our last recorded session ({latest_file.name}), we covered {len(turns)} turns. "
        f"Key topics included: {queries_summary}."
    )


# ── 4. Complex Reasoning vs. Everyday Conversation Classifier ─────────────────

_COMPLEX_KEYWORDS = [
    # Explicit analysis & deduction
    r"\banalyze\b",
    r"\banalysis\b",
    r"\bcompare\s+and\s+contrast\b",
    r"\bdifference\s+between\b",
    r"\bdifferences\s+between\b",
    r"\bpros\s+and\s+cons\b",
    r"\badvantages\s+and\s+disadvantages\b",
    r"\btrade-?offs?\b",
    r"\bdeep\s+dive\b",
    r"\bstep-?by-?step\b",
    r"\bexplain\s+(?:why|how|in\s+detail)\b",
    r"\bphilosophical\b",
    r"\bphilosophy\b",
    r"\bimplications\s+of\b",
    r"\bconsequences\s+of\b",
    r"\bmathematical\s+proof\b",
    r"\bprove\s+that\b",
    r"\bderive\b",
    r"\bdeduce\b",
    # Technical architecture & code implementation
    r"\barchitect(?:ure)?\b",
    r"\bsystem\s+design\b",
    r"\balgorithm\b",
    r"\bimplement\s+(?:an?\s+)?(?:algorithm|class|pattern)\b",
    r"\bdebug\s+(?:this|the)\b",
    r"\bwrite\s+(?:a\s+)?(?:python|code|script|function|program)\b",
    r"\bcomplex\s+(?:reasoning|problem|logic)\b",
]

_CONFIRMATION_PATTERNS = [
    r"^(?:yes|please|sure|go\s+ahead|okay|ok|proceed|yes\s*,?\s*please)\.?$",
]


def is_user_confirmation(text: str) -> bool:
    """Check if the user input is a brief affirmative response."""
    cleaned = text.strip().lower()
    return any(re.match(pat, cleaned) for pat in _CONFIRMATION_PATTERNS)


def classify_request(
    query: str,
    archive_dir: Path | None = None,
    persona_name: str = "IRIS",
) -> RoutingDecision:
    """
    Deterministic routing decision engine:
    1. Exact calculations -> Deterministic Python functions
    2. Simple date/time operations -> Python/system utilities
    3. File and conversation-memory retrieval -> Local archive retrieval
    4. Complex reasoning & difficult questions -> Gemini Flash (Gemini Developer API free tier)
    5. Everyday conversation -> OpenRouter free models
    """
    cleaned = query.strip()

    # 1. Exact Calculations
    calc_result = try_evaluate_calculation(cleaned)
    if calc_result is not None:
        return RoutingDecision(
            route=RouteType.CALCULATION,
            reason="Deterministic arithmetic / algebra calculation",
            direct_response=calc_result,
            target_provider="deterministic",
            target_model="python_math",
        )

    # 2. Simple Date/Time Operations
    datetime_result = try_evaluate_datetime(cleaned)
    if datetime_result is not None:
        return RoutingDecision(
            route=RouteType.DATETIME,
            reason="Deterministic system date/time lookup",
            direct_response=datetime_result,
            target_provider="deterministic",
            target_model="python_datetime",
        )

    # 3. File & Conversation-Memory Retrieval
    memory_result = try_evaluate_memory(cleaned, archive_dir=archive_dir, persona_name=persona_name)
    if memory_result is not None:
        return RoutingDecision(
            route=RouteType.MEMORY,
            reason="Local conversation session archive retrieval",
            direct_response=memory_result,
            target_provider="local_archive",
            target_model="session_markdown",
        )

    # 4. Complex Reasoning and Difficult Questions
    is_complex = False
    match_reason = ""

    # Check for analytical / reasoning keywords
    for pat in _COMPLEX_KEYWORDS:
        m = re.search(pat, cleaned, re.IGNORECASE)
        if m:
            is_complex = True
            match_reason = f"Keyword match: '{m.group(0)}'"
            break

    # Substantive multi-question or deep analytical query heuristics
    if not is_complex:
        words = cleaned.split()
        num_questions = cleaned.count("?")
        has_analytical_markers = any(
            w.lower() in {"why", "how", "impact", "underlying", "principle", "theorem"}
            for w in words
        )
        if num_questions >= 2 and len(words) >= 15:
            is_complex = True
            match_reason = "Multi-part inquiry with substantive questioning"
        elif len(words) >= 28 and has_analytical_markers:
            is_complex = True
            match_reason = "Extended analytical query length with conceptual reasoning markers"

    if is_complex:
        return RoutingDecision(
            route=RouteType.COMPLEX_REASONING,
            reason=f"Complex reasoning ({match_reason})",
            target_provider="google",
            target_model="gemini-3.8-flash",
        )

    # 5. Everyday Conversation
    return RoutingDecision(
        route=RouteType.CONVERSATION,
        reason="Everyday conversational dialogue",
        target_provider="openrouter",
        target_model="openrouter/free",
    )
