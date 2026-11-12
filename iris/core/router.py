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
    WEATHER = "weather"
    LOCATION = "location"
    KNOWLEDGE_SAVE = "knowledge_save"
    KNOWLEDGE_SEARCH = "knowledge_search"
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

_TIMEZONE_PATTERNS = [
    r"\bwhat\s+timezone\s+(?:is\s+this|am\s+i\s+in|are\s+we\s+in)\b",
    r"\bwhat\s+is\s+my\s+timezone\b",
    r"\bwhat\s+is\s+the\s+timezone\b",
    r"\bwhat(?:\'s|\s+is)\s+our\s+timezone\b",
]


def try_evaluate_datetime(
    query: str,
    now_dt: datetime | None = None,
    context: Any | None = None,
) -> str | None:
    """
    Evaluate simple date/time questions using system utilities or environment context.
    Returns a natural English answer or None if not a current date/time query.
    Rejects questions with location qualifiers, past/future tense, or event targets.
    """
    cleaned = query.strip().lower()

    # Timezone query check
    for pattern in _TIMEZONE_PATTERNS:
        if re.search(pattern, cleaned):
            if context is not None and getattr(context, "timezone", None) is not None:
                tz = context.timezone
                offset = tz.utc_offset_hours
                offset_str = f"UTC{'+' if offset >= 0 else ''}{int(offset) if offset.is_integer() else offset}"
                return f"You are currently in the {tz.iana_id} timezone ({tz.timezone_name}, {offset_str})."
            return None

    now = now_dt
    if now is None and context is not None and getattr(context, "timezone", None) is not None:
        try:
            import zoneinfo
            tz_obj = zoneinfo.ZoneInfo(context.timezone.iana_id)
            now = datetime.now(tz_obj)
        except Exception:
            now = datetime.now()
    elif now is None:
        now = datetime.now()

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


# ── 3. Deterministic Weather Handler ──────────────────────────────────────────

_WEATHER_PATTERNS = [
    r"\bwhat(?:\'s|\s+is)\s+the\s+weather\b",
    r"\bhow(?:\'s|\s+is)\s+the\s+weather\b",
    r"\bweather\s+(?:today|outside|like|forecast|right\s+now)\b",
    r"\bwhat\s+is\s+the\s+temperature\b",
    r"\bwhat(?:\'s|\s+is)\s+it\s+like\s+outside\b",
    r"\bcurrent\s+weather\b",
    r"\btell\s+me\s+the\s+weather\b",
    r"\bwill\s+it\s+rain\b",
    r"\bis\s+it\s+raining\b",
    r"\bchance\s+of\s+rain\b",
    r"\bforecast\s+for\s+(?:today|tomorrow)\b",
    r"\btomorrow(?:\'s)?\s+weather\b",
]

_TEMPERATURE_PATTERNS = [
    r"\bwhat(?:\'s|\s+is)\s+the\s+temperature\b",
    r"\bhow\s+(?:cold|hot|warm)\s+is\s+it\b",
    r"\btemperature\s+outside\b",
    r"\bcurrent\s+temperature\b",
]

_RAIN_PATTERNS = [
    r"\bwill\s+it\s+rain\b",
    r"\bis\s+it\s+raining\b",
    r"\bchance\s+of\s+rain\b",
    r"\bis\s+there\s+any\s+rain\b",
]

_FORECAST_TOMORROW_PATTERNS = [
    r"\btomorrow(?:\'s)?\s+weather\b",
    r"\bweather\s+tomorrow\b",
    r"\bforecast\s+for\s+tomorrow\b",
]


def try_evaluate_weather(query: str, context: Any | None = None) -> str | None:
    """
    Evaluate weather questions using local Context Engine snapshot.
    Returns concise natural English answer or None if not an applicable local query.
    """
    if context is None or getattr(context, "weather", None) is None:
        return None

    cleaned = query.strip().lower()
    is_weather = any(re.search(pat, cleaned) for pat in _WEATHER_PATTERNS)
    if not is_weather:
        return None

    loc_name = context.location.safe_display_name().lower()

    # Disqualify queries asking about an external city not matching detected location
    target_match = re.search(r"\b(?:in|at|for)\s+([a-z\s]+)", cleaned)
    if target_match:
        target_place = target_match.group(1).strip()
        if target_place not in loc_name and not any(k in target_place for k in ["here", "today", "tomorrow", "this", "outside"]):
            return None

    weather = context.weather
    if not weather or not weather.current:
        return None

    display_loc = context.location.safe_display_name()

    # 1. Temperature query
    if any(re.search(pat, cleaned) for pat in _TEMPERATURE_PATTERNS):
        temp = int(round(weather.current.temperature_c))
        return f"It is currently {temp} degrees Celsius in {display_loc}."

    # 2. Rain query
    if any(re.search(pat, cleaned) for pat in _RAIN_PATTERNS):
        is_raining = "rain" in weather.current.condition.lower() or "drizzle" in weather.current.condition.lower()
        rain_phrase = "It is currently raining" if is_raining else "It is not currently raining"
        prob = weather.forecast[0].precipitation_probability_pct if weather.forecast else None
        prob_str = f" with a {prob}% chance of rain today" if prob is not None else ""
        return f"{rain_phrase} in {display_loc}{prob_str}."

    # 3. Tomorrow's forecast
    if any(re.search(pat, cleaned) for pat in _FORECAST_TOMORROW_PATTERNS) and len(weather.forecast) > 1:
        fc = weather.forecast[1]
        t_max = int(round(fc.temp_max_c))
        t_min = int(round(fc.temp_min_c))
        prob_str = f" with a {fc.precipitation_probability_pct}% chance of rain" if fc.precipitation_probability_pct is not None else ""
        return (
            f"Tomorrow in {display_loc} will be {fc.condition.lower()}, "
            f"with a high of {t_max} degrees and a low of {t_min} degrees{prob_str}."
        )

    # 4. General weather
    return context.weather_summary_for_voice()


# ── 4. Deterministic Location Handler ─────────────────────────────────────────

_LOCATION_PATTERNS = [
    r"\bwhere\s+am\s+i\b",
    r"\bwhat\s+is\s+my\s+(?:current\s+)?location\b",
    r"\bwhat\s+location\s+are\s+we\s+in\b",
    r"\bwhat\s+city\s+am\s+i\s+in\b",
    r"\bcurrent\s+location\b",
]


def try_evaluate_location(query: str, context: Any | None = None) -> str | None:
    """
    Evaluate location queries safely without exposing coordinates.
    """
    cleaned = query.strip().lower()
    if not any(re.search(pat, cleaned) for pat in _LOCATION_PATTERNS):
        return None

    if context is not None and getattr(context, "location", None) is not None:
        return f"Your current detected location is {context.location.safe_display_name()}."
    return "Your current location has not been determined."


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


# ── 3b. Deterministic Knowledge Vault Handlers (Explicit Intent Only) ──────────

_WAKE_PREFIX_RE = re.compile(r"^(?:hey\s+)?iris[,:\s]+", re.IGNORECASE)

_KNOWLEDGE_SAVE_PATTERNS = [
    re.compile(
        r"^(?:please\s+)?(?:add|save|put|store)\s+(?:this|that|it)"
        r"(?:\s+information)?\s+(?:to|in|into|as)\s+(?:my\s+)?"
        r"(?:notes|knowledge\s+vault|vault|a\s+note|inbox)\b(.*)$",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"^(?:please\s+)?(?:create|make|write|save|add)\s+(?:a\s+)?(?:new\s+)?note\b(.*)$",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"^(?:please\s+)?remember\s+this\s+information\b(.*)$",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"^(?:please\s+)?save\s+this\s+(?:as\s+a\s+note|to\s+(?:my\s+)?notes|"
        r"to\s+(?:my\s+)?(?:knowledge\s+)?vault|to\s+(?:my\s+)?inbox|information)\b(.*)$",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"^(?:please\s+)?capture\s+(?:this|to\s+(?:my\s+)?inbox|a\s+note)\b(.*)$",
        re.IGNORECASE | re.DOTALL,
    ),
]

_KNOWLEDGE_SEARCH_PATTERNS = [
    re.compile(
        r"^(?:please\s+)?(?:search|find|check|look\s+up|list|show)\s+(?:in\s+)?(?:my\s+)?"
        r"(?:knowledge\s+)?vault(?:\s+(?:for|about|on)\s+(.+))?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?(?:search|find|list|show)\s+(?:my\s+)?notes"
        r"(?:\s+(?:for|about|on|in|tagged)\s+(.+))?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?look\s+up\s+(.+?)\s+in\s+(?:my\s+)?(?:notes|knowledge\s+vault|vault)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:what|which)\s+notes\s+do\s+i\s+have(?:\s+(?:for|about|on|in)\s+(.+))?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?(?:check|list|show)\s+(?:my\s+)?(?:knowledge\s+)?inbox\b(.*)$",
        re.IGNORECASE,
    ),
]


def _derive_note_title(content: str, explicit_title: str = "") -> str:
    """Derive a concise title directly from explicit user/content text without inventing details."""
    if explicit_title.strip():
        return explicit_title.strip()[:80]
    first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
    clean = re.sub(r"^#+\s*", "", first_line).strip()
    if len(clean) <= 60:
        return clean or "Captured Note"
    truncated = clean[:57].rsplit(" ", 1)[0].strip()
    return f"{truncated or clean[:57]}..."


def _extract_save_payload(
    cleaned_query: str,
    remainder: str,
    prior_context: str | None = None,
) -> tuple[str, str, bool]:
    """
    Extract (title, body, target_inbox) from an explicit save request.

    Returns ("", "", target_inbox) if the user provided no inline content and
    there is no prior context to capture.
    """
    target_inbox = bool(re.search(r"\b(?:to|in|into)\s+(?:my\s+)?inbox\b|^capture\b", cleaned_query, re.IGNORECASE))
    title = ""

    rest = remainder.strip()
    # Optional explicit title marker: titled "..." / called "..."
    title_match = re.match(
        r'^(?:titled|called|named)\s+["\']([^"\']+)["\'](?:\s*[:\-–—]?\s*(.*))?$',
        rest,
        re.IGNORECASE | re.DOTALL,
    )
    if title_match:
        title = title_match.group(1).strip()
        rest = (title_match.group(2) or "").strip()

    # Strip leading connectors like ':', '-', 'saying', 'about', 'that'
    rest = re.sub(r"^(?:[:\-–—]+|(?:saying|about|that)\b\s*:?)\s*", "", rest, flags=re.IGNORECASE).strip()

    body = rest or (prior_context.strip() if prior_context else "")
    if not body:
        return "", "", target_inbox

    return _derive_note_title(body, explicit_title=title), body, target_inbox


def try_evaluate_knowledge_save(
    query: str,
    knowledge_service: Any | None = None,
    prior_context: str | None = None,
) -> str | None:
    """
    Persist a permanent note ONLY when the user expresses explicit save intent.

    Returns None when the input is ordinary dialogue or a question without save intent.
    """
    cleaned = _WAKE_PREFIX_RE.sub("", query.strip()).strip()
    if not cleaned:
        return None

    matched_remainder: str | None = None
    for pat in _KNOWLEDGE_SAVE_PATTERNS:
        m = pat.match(cleaned)
        if m:
            matched_remainder = m.group(1) or ""
            break

    if matched_remainder is None:
        return None

    if knowledge_service is None:
        return (
            "I recognized your request to save a note, but the Knowledge Vault "
            "service is not configured in this session."
        )

    title, body, target_inbox = _extract_save_payload(cleaned, matched_remainder, prior_context)
    if not body:
        return "What exact information would you like me to save to your Knowledge Vault?"

    if target_inbox:
        note = knowledge_service.capture_to_inbox(content=body, title=title)
        return (
            f"Captured to Knowledge Vault Inbox as \"{note.metadata.title}\" "
            f"({note.relative_path})."
        )

    note = knowledge_service.create_note(
        title=title,
        content=body,
        folder="Notes",
        note_type="note",
    )
    return (
        f"Saved permanent note \"{note.metadata.title}\" to your Knowledge Vault "
        f"at {note.relative_path} (separate from conversation archives)."
    )


def try_evaluate_knowledge_search(
    query: str,
    knowledge_service: Any | None = None,
) -> str | None:
    """
    Search permanent Markdown notes in the Knowledge Vault via SQLite FTS5.

    Explicitly separates permanent notes from conversation archives in responses.
    """
    cleaned = _WAKE_PREFIX_RE.sub("", query.strip()).strip()
    if not cleaned:
        return None

    matched = False
    search_clause = ""
    is_inbox_query = bool(
        re.match(r"^(?:please\s+)?(?:check|list|show)\s+(?:my\s+)?(?:knowledge\s+)?inbox\b", cleaned, re.IGNORECASE)
    )
    for pat in _KNOWLEDGE_SEARCH_PATTERNS:
        m = pat.match(cleaned)
        if m:
            matched = True
            search_clause = (m.group(1) or "").strip() if m.lastindex else ""
            break

    if not matched:
        return None

    if knowledge_service is None:
        return "The Knowledge Vault service is not configured in this session."

    folder_filter: str | None = "Inbox" if is_inbox_query else None
    tags_filter: list[str] = []

    if search_clause:
        folder_m = re.search(
            r"\bin\s+(Inbox|Notes|Projects|Research|Archive)\b",
            search_clause,
            re.IGNORECASE,
        )
        if folder_m:
            folder_filter = folder_m.group(1)
            search_clause = (search_clause[: folder_m.start()] + search_clause[folder_m.end() :]).strip()

        for tag_m in re.finditer(r"(?:#|tagged\s+)([a-zA-Z0-9_-]+)", search_clause, re.IGNORECASE):
            tags_filter.append(tag_m.group(1).lower())
        search_clause = re.sub(r"(?:#|tagged\s+)[a-zA-Z0-9_-]+", "", search_clause, flags=re.IGNORECASE).strip()

    results = knowledge_service.search(
        query=search_clause,
        tags=tags_filter or None,
        folder=folder_filter,
        limit=5,
    )
    if not results:
        scope_desc = f" in {folder_filter}" if folder_filter else ""
        return (
            f"I searched your permanent Knowledge Vault{scope_desc} "
            "(separate from session conversation archives) and found no matching notes."
        )

    lines = [
        f"Found {len(results)} permanent note{'s' if len(results) != 1 else ''} in your Knowledge Vault:"
    ]
    for item in results:
        snippet = f" — {item.excerpt}" if item.excerpt else ""
        lines.append(f"- [{item.folder}] {item.title} ({item.relative_path}){snippet}")
    return "\n".join(lines)


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
    context_snapshot: Any | None = None,
    knowledge_service: Any | None = None,
    prior_context: str | None = None,
) -> RoutingDecision:
    """
    Deterministic routing decision engine:
    1. Exact calculations -> Deterministic Python functions
    2. Simple date/time operations -> Python/system utilities / Context Engine
    3. Live weather queries -> Context Engine (Open-Meteo)
    4. Location queries -> Context Engine (privacy-safe device location)
    5. Explicit Knowledge Vault saves -> Permanent Markdown Vault
    6. Knowledge Vault search -> Disposable SQLite FTS5 index
    7. File and conversation-memory retrieval -> Local archive retrieval
    8. Complex reasoning & difficult questions -> Gemini Flash (Gemini Developer API free tier)
    9. Everyday conversation -> OpenRouter free models
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
    datetime_result = try_evaluate_datetime(cleaned, context=context_snapshot)
    if datetime_result is not None:
        return RoutingDecision(
            route=RouteType.DATETIME,
            reason="Deterministic system date/time lookup",
            direct_response=datetime_result,
            target_provider="deterministic",
            target_model="python_datetime",
        )

    # 3. Live Weather Operations
    weather_result = try_evaluate_weather(cleaned, context=context_snapshot)
    if weather_result is not None:
        return RoutingDecision(
            route=RouteType.WEATHER,
            reason="Deterministic Context Engine live weather lookup",
            direct_response=weather_result,
            target_provider="context_engine",
            target_model="open_meteo",
        )

    # 4. Device Location Inquiries
    location_result = try_evaluate_location(cleaned, context=context_snapshot)
    if location_result is not None:
        return RoutingDecision(
            route=RouteType.LOCATION,
            reason="Deterministic Context Engine location lookup",
            direct_response=location_result,
            target_provider="context_engine",
            target_model="device_location",
        )

    # 5. Explicit Knowledge Vault Save (requires explicit save intent)
    knowledge_save_result = try_evaluate_knowledge_save(
        cleaned,
        knowledge_service=knowledge_service,
        prior_context=prior_context,
    )
    if knowledge_save_result is not None:
        return RoutingDecision(
            route=RouteType.KNOWLEDGE_SAVE,
            reason="Explicit user request to save permanent knowledge note",
            direct_response=knowledge_save_result,
            target_provider="knowledge_vault",
            target_model="markdown_vault",
        )

    # 6. Knowledge Vault Search (separate from conversation archives)
    knowledge_search_result = try_evaluate_knowledge_search(
        cleaned,
        knowledge_service=knowledge_service,
    )
    if knowledge_search_result is not None:
        return RoutingDecision(
            route=RouteType.KNOWLEDGE_SEARCH,
            reason="Knowledge Vault full-text FTS5 retrieval",
            direct_response=knowledge_search_result,
            target_provider="knowledge_vault",
            target_model="sqlite_fts5",
        )

    # 7. File & Conversation-Memory Retrieval
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
