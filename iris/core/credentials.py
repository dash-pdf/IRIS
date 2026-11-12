"""
iris/core/credentials.py
------------------------
Secure OS credential-store abstraction for IRIS.

Uses `keyring` (macOS Keychain / Windows Credential Locker) when available and falls
back gracefully to environment variables (`.env` during local development) without
ever logging, archiving, or silently deleting/migrating existing secrets.

Manual Credential Migration Procedure (when ready to move off `.env`):
1. Store each key cleanly in OS Keychain using `CredentialStore`:
       python -c "from iris.core.credentials import CredentialStore; CredentialStore().set_secret('openrouter_api_key', input('Key: '))"
2. Verify resolution via `CredentialStore().get_source('openrouter_api_key') == 'keyring'`.
3. Remove the migrated secret line from your local `.env` once verified.
Never commit `.env`, pass raw keys on the command line (avoid shell history), or
write secrets into Markdown archives, SQLite databases, or logs.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any, Literal

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_NAME = "IRIS"

# Canonical credential key names managed in OS secure storage
CREDENTIAL_KEYS: dict[str, str] = {
    "elevenlabs_api_key": "ELEVENLABS_API_KEY",
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "google_api_key": "GOOGLE_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "weather_api_key": "IRIS_WEATHER_API_KEY",
    "google_calendar_oauth_token": "IRIS_GOOGLE_CALENDAR_TOKEN",
    "spotify_oauth_token": "IRIS_SPOTIFY_TOKEN",
}

CredentialSource = Literal["keyring", "env", "missing"]


def _is_placeholder(value: str) -> bool:
    clean = value.strip()
    return not clean or clean.startswith("your_") or clean.endswith("_here")


class CredentialStore:
    """
    Read/write credential wrapper backed by OS secure storage (`keyring`) with
    read-only environment variable fallback.
    """

    def __init__(
        self,
        service_name: str = DEFAULT_SERVICE_NAME,
        keyring_backend: Any | None = None,
        allow_env_fallback: bool = True,
        env_mapping: Mapping[str, str] | None = None,
    ) -> None:
        self._service_name = service_name
        self._allow_env_fallback = allow_env_fallback
        self._env = env_mapping if env_mapping is not None else os.environ
        self._keyring = keyring_backend if keyring_backend is not None else self._load_keyring()

    @staticmethod
    def _load_keyring() -> Any | None:
        try:
            import keyring
            from keyring.errors import NoKeyringError  # noqa: F401

            backend = keyring.get_keyring()
            backend_name = type(backend).__name__.lower()
            backend_module = type(backend).__module__.lower()
            if "fail" in backend_module or "null" in backend_module or backend_name == "nullkeyring":
                logger.warning("No usable OS keyring backend detected (%s).", type(backend).__name__)
                return None
            return keyring
        except Exception as exc:
            logger.debug("Keyring backend unavailable (%s); continuing with env fallback.", type(exc).__name__)
            return None

    @property
    def service_name(self) -> str:
        return self._service_name

    @property
    def is_secure_backend_available(self) -> bool:
        return self._keyring is not None

    def get_secret(self, key_name: str, env_var: str | None = None) -> str | None:
        """
        Retrieve a secret from OS secure storage, falling back to `.env` / environment
        variables if enabled. Returns None when missing or placeholder.
        """
        if self._keyring is not None:
            try:
                value = self._keyring.get_password(self._service_name, key_name)
                if value and not _is_placeholder(value):
                    return value
            except Exception as exc:
                logger.warning(
                    "Unable to read credential '%s' from secure storage (%s); checking fallback.",
                    key_name,
                    type(exc).__name__,
                )

        if self._allow_env_fallback:
            target_env = env_var or CREDENTIAL_KEYS.get(key_name, "")
            if target_env:
                raw_env = self._env.get(target_env, "")
                if raw_env and not _is_placeholder(raw_env):
                    return raw_env

        return None

    def set_secret(self, key_name: str, value: str) -> bool:
        """
        Store a secret in OS secure storage (`keyring`).
        Returns False cleanly if no secure backend is available or storage fails.
        """
        clean_value = value.strip()
        if not clean_value:
            raise ValueError(f"Credential value for '{key_name}' must not be empty.")

        if self._keyring is None:
            logger.warning("Cannot store credential '%s': OS secure storage backend unavailable.", key_name)
            return False

        try:
            self._keyring.set_password(self._service_name, key_name, clean_value)
            return True
        except Exception as exc:
            logger.error("Failed storing credential '%s' in secure storage (%s).", key_name, type(exc).__name__)
            return False

    def delete_secret(self, key_name: str) -> bool:
        """
        Remove a secret from OS secure storage (`keyring`).
        Returns False if backend is unavailable or key did not exist.
        """
        if self._keyring is None:
            return False

        try:
            self._keyring.delete_password(self._service_name, key_name)
            return True
        except Exception as exc:
            # Handle PasswordDeleteError cleanly when entry does not exist
            if "Delete" in type(exc).__name__ or "not found" in str(exc).lower():
                return False
            logger.warning("Failed deleting credential '%s' from secure storage (%s).", key_name, type(exc).__name__)
            return False

    def get_source(self, key_name: str, env_var: str | None = None) -> CredentialSource:
        """Identify where a credential is currently resolved from without returning its value."""
        if self._keyring is not None:
            try:
                value = self._keyring.get_password(self._service_name, key_name)
                if value and not _is_placeholder(value):
                    return "keyring"
            except Exception:
                pass

        if self._allow_env_fallback:
            target_env = env_var or CREDENTIAL_KEYS.get(key_name, "")
            if target_env:
                raw_env = self._env.get(target_env, "")
                if raw_env and not _is_placeholder(raw_env):
                    return "env"

        return "missing"
