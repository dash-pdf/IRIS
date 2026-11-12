"""
iris/core/security.py
---------------------
Shared secret-redaction and error-sanitisation utilities for IRIS.

All exceptions, HTTP errors, and diagnostic strings written to Markdown archives
or application logs should pass through `sanitize_error_message` / `sanitize_sensitive_text`
so API keys, OAuth tokens, bearer headers, private keys, and sensitive query parameters
are never persisted in plaintext.
"""

from __future__ import annotations

import re

# Known token/key value prefixes (OpenRouter, OpenAI, Anthropic, Google API/OAuth, GitHub, JWT)
_SECRET_PREFIX_PATTERN = re.compile(
    r"\b(?:"
    r"sk-(?:or-v1-|ant-|proj-)?[A-Za-z0-9_\-]{12,}|"
    r"AIza[0-9A-Za-z_\-]{20,}|"
    r"ya29\.[0-9A-Za-z_\-\.]{12,}|"
    r"gh[pousr]_[A-Za-z0-9_]{16,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|"
    r"xi-[A-Za-z0-9_\-]{16,}|"
    r"eyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"
    r")"
)

# PEM private key blocks (e.g. Apple WeatherKit .p8 / RSA / EC keys)
_PEM_BLOCK_PATTERN = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    flags=re.DOTALL,
)

# Authorization / custom API-key headers (Bearer <token>, Basic <token>, xi-api-key: <token>, etc.)
_AUTH_HEADER_PATTERN = re.compile(
    r"\b(Authorization\s*:\s*(?:Bearer|Basic|Token)\s+|Bearer\s+|Basic\s+|xi-api-key\s*[:=]\s*|x-api-key\s*[:=]\s*)"
    r"([A-Za-z0-9_\-\.\+/=]{6,})",
    flags=re.IGNORECASE,
)

# Query strings attached to URLs (?Sec-MS-GEC=..., ?key=..., ?code=..., etc.)
_URL_QUERY_PATTERN = re.compile(r"\?[^ \t\n\r\"'>]+")

# Generic key=value / token: value assignments (single-line value, skipping already redacted tags)
_KV_SECRET_PATTERN = re.compile(
    r"\b(token|access_token|refresh_token|oauth_token|auth|api_key|apikey|key|"
    r"secret|client_secret|password|passwd|credential|Sec-MS-GEC|signature|sig)"
    r"([ \t]*[=:][ \t]*['\"]?)(?!\[REDACTED)([^'\"\s,&;}\]]+)(['\"]?)",
    flags=re.IGNORECASE,
)


def sanitize_sensitive_text(text: str) -> str:
    """Redact credential-like strings, tokens, PEM keys, and URL query parameters."""
    if not text:
        return ""

    cleaned = _PEM_BLOCK_PATTERN.sub("[REDACTED PRIVATE KEY]", text)
    cleaned = _URL_QUERY_PATTERN.sub("?[REDACTED]", cleaned)
    cleaned = _AUTH_HEADER_PATTERN.sub(r"\1[REDACTED]", cleaned)
    cleaned = _KV_SECRET_PATTERN.sub(r"\1\2[REDACTED]\4", cleaned)
    cleaned = _SECRET_PREFIX_PATTERN.sub("[REDACTED]", cleaned)
    return cleaned


def sanitize_error_message(error: str | BaseException) -> str:
    """
    Format and sanitise an exception or error string for logs and archives.

    Preserves the exception class name and non-sensitive diagnostic context while
    redacting any embedded credentials or URL parameters.
    """
    if isinstance(error, BaseException):
        exc_type = type(error).__name__
        raw_detail = sanitize_sensitive_text(str(error).strip())
        if not raw_detail:
            return exc_type
        if raw_detail.startswith(f"{exc_type}:"):
            return raw_detail
        return f"{exc_type}: {raw_detail}"

    return sanitize_sensitive_text(str(error))
