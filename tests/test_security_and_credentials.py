"""
tests/test_security_and_credentials.py
--------------------------------------
Unit and integration tests for IRIS Phase 0:
- Error and archive redaction (`iris/core/security.py`, `archive.py`, `assistant.py`)
- OS secure credential store abstraction (`iris/core/credentials.py`, `config.py`)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from iris.adapters.model.mock_adapter import MockModelAdapter
from iris.adapters.stt.mock_stt import MockSTT
from iris.adapters.tts.mock_tts import MockTTS
from iris.audio.capture import AudioCapture
from iris.audio.player import AudioPlayer
from iris.audio.vad import VoiceActivityDetector
from iris.core.assistant import IRISAssistant
from iris.core.config import IRISConfig, load_config
from iris.core.credentials import CredentialStore
from iris.core.security import sanitize_error_message, sanitize_sensitive_text
from iris.core.state import AudioState
from iris.services.archive import ConversationArchive


class InMemoryKeyringBackend:
    """Deterministic in-memory mock compatible with the `keyring` API."""

    def __init__(self) -> None:
        self.secrets: dict[tuple[str, str], str] = {}

    def get_password(self, service_name: str, username: str) -> str | None:
        return self.secrets.get((service_name, username))

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self.secrets[(service_name, username)] = password

    def delete_password(self, service_name: str, username: str) -> None:
        if (service_name, username) not in self.secrets:
            raise RuntimeError("PasswordDeleteError: entry not found")
        del self.secrets[(service_name, username)]


class BrokenKeyringBackend:
    """Simulates a locked or unavailable OS keyring backend."""

    def get_password(self, service_name: str, username: str) -> str | None:
        raise RuntimeError("Keychain locked")

    def set_password(self, service_name: str, username: str, password: str) -> None:
        raise RuntimeError("Keychain write denied")

    def delete_password(self, service_name: str, username: str) -> None:
        raise RuntimeError("Keychain unavailable")


class TestErrorSanitisation:
    def test_redacts_api_keys_bearer_tokens_and_query_params(self) -> None:
        raw = (
            "HTTP 401 POST https://generativelanguage.googleapis.com/v1beta/models/flash:generate"
            "?key=AIzaSyDummyGoogleKey1234567890&token=secret99 "
            "Authorization: Bearer ya29.a0AfH6SM_DummyOAuthToken "
            "openrouter=sk-or-v1-1234567890abcdef1234567890 "
            "refresh_token=1//0gDummyRefresh"
        )
        sanitised = sanitize_sensitive_text(raw)
        assert "AIzaSyDummyGoogleKey1234567890" not in sanitised
        assert "secret99" not in sanitised
        assert "ya29.a0AfH6SM_DummyOAuthToken" not in sanitised
        assert "sk-or-v1-1234567890abcdef1234567890" not in sanitised
        assert "1//0gDummyRefresh" not in sanitised
        assert "HTTP 401 POST https://generativelanguage.googleapis.com/v1beta/models/flash:generate?[REDACTED]" in sanitised

    def test_redacts_pem_private_key_blocks(self) -> None:
        raw = (
            "Auth error loading key:\n"
            "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQ\n"
            "-----END PRIVATE KEY-----\nfailed cleanly"
        )
        sanitised = sanitize_sensitive_text(raw)
        assert "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQ" not in sanitised
        assert "[REDACTED PRIVATE KEY]" in sanitised
        assert "failed cleanly" in sanitised

    def test_sanitize_error_message_preserves_exception_type_and_context(self) -> None:
        exc = ConnectionError(
            "Timed out calling https://api.example.com/v1/chat?api_key=sk-proj-abc1234567890xyz after 30s"
        )
        msg = sanitize_error_message(exc)
        assert msg.startswith("ConnectionError: ")
        assert "sk-proj-abc1234567890xyz" not in msg
        assert "after 30s" in msg

    def test_archive_redacts_secrets_in_error_entries(self, tmp_path: Path) -> None:
        archive = ConversationArchive(archive_dir=tmp_path)
        archive.open_session()
        archive.write_error("Failed with Authorization: Bearer secret_token_xyz123 and ?token=abc")
        archive.write_turn("error", "API failure sk-or-v1-999888777666555444")
        archive.close_session()

        content = archive.session_file.read_text(encoding="utf-8")
        assert "secret_token_xyz123" not in content
        assert "?[REDACTED]" in content
        assert "999888777666555444" not in content
        assert "### [" in content
        assert "] Error" in content

    def test_assistant_sanitises_exception_before_archiving(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        cfg = IRISConfig(
            archive_dir=tmp_path / "conversations",
            context_cache_dir=tmp_path / "context",
            enable_location_tracking=False,
            cooldown_ms=10,
        )
        failing_model = MagicMock()
        failing_model.generate_response.side_effect = RuntimeError(
            "Upstream failure at https://api.openrouter.ai/v1/chat?api_key=sk-or-v1-secretkey99999999 "
            "Authorization: Bearer bearer_secret_12345"
        )

        assistant = IRISAssistant(
            config=cfg,
            model_adapter=failing_model,
            tts_adapter=MockTTS(),
            stt_adapter=MockSTT(default_response="Test question"),
        )
        assistant.capture = MagicMock(spec=AudioCapture)
        assistant.player = MagicMock(spec=AudioPlayer)
        assistant.vad = MagicMock(spec=VoiceActivityDetector)
        assistant.archive.open_session()

        turn_id = assistant.state_machine.new_turn()
        assistant.state_machine.transition_to(AudioState.THINKING)

        with caplog.at_level(logging.ERROR):
            assistant._process_utterance(
                utterance_bytes=b"\x01\x00" * 1600,
                turn_id=turn_id,
                speech_end_time=time.perf_counter(),
            )

        assert assistant.state_machine.state == AudioState.IDLE
        assert assistant.capture.unmute.called

        archive_content = assistant.archive.session_file.read_text(encoding="utf-8")
        assert "sk-or-v1-secretkey99999999" not in archive_content
        assert "bearer_secret_12345" not in archive_content
        assert "RuntimeError:" in archive_content
        assert "sk-or-v1-secretkey99999999" not in caplog.text
        assert "bearer_secret_12345" not in caplog.text


class TestCredentialStore:
    def test_store_retrieve_and_delete_via_mock_backend(self) -> None:
        backend = InMemoryKeyringBackend()
        store = CredentialStore(service_name="IRIS-TEST", keyring_backend=backend, env_mapping={})

        assert store.is_secure_backend_available is True
        assert store.get_secret("openrouter_api_key") is None
        assert store.get_source("openrouter_api_key") == "missing"

        assert store.set_secret("openrouter_api_key", "test-secret-value") is True
        assert store.get_secret("openrouter_api_key") == "test-secret-value"
        assert store.get_source("openrouter_api_key") == "keyring"

        assert store.delete_secret("openrouter_api_key") is True
        assert store.get_secret("openrouter_api_key") is None
        assert store.delete_secret("openrouter_api_key") is False

    def test_env_fallback_and_keyring_precedence(self) -> None:
        backend = InMemoryKeyringBackend()
        env = {
            "OPENROUTER_API_KEY": "env-fallback-key",
            "GOOGLE_API_KEY": "your_google_api_key_here",  # placeholder should be ignored
        }
        store = CredentialStore(keyring_backend=backend, env_mapping=env)

        assert store.get_secret("openrouter_api_key") == "env-fallback-key"
        assert store.get_source("openrouter_api_key") == "env"
        assert store.get_secret("google_api_key") is None
        assert store.get_source("google_api_key") == "missing"

        # Once stored in keyring, keyring takes precedence over env fallback
        store.set_secret("openrouter_api_key", "keychain-primary-key")
        assert store.get_secret("openrouter_api_key") == "keychain-primary-key"
        assert store.get_source("openrouter_api_key") == "keyring"
        # Existing env dict is never silently deleted or mutated
        assert env["OPENROUTER_API_KEY"] == "env-fallback-key"

    def test_broken_or_missing_backend_falls_back_gracefully(self) -> None:
        store = CredentialStore(
            keyring_backend=BrokenKeyringBackend(),
            env_mapping={"OPENROUTER_API_KEY": "dev-env-key"},
        )
        assert store.get_secret("openrouter_api_key") == "dev-env-key"
        assert store.set_secret("openrouter_api_key", "new-value") is False
        assert store.delete_secret("openrouter_api_key") is False

        no_backend_store = CredentialStore(
            keyring_backend=None,
            allow_env_fallback=False,
            env_mapping={"OPENROUTER_API_KEY": "ignored"},
        )
        # Force _keyring None to simulate missing/fail OS backend
        no_backend_store._keyring = None
        assert no_backend_store.is_secure_backend_available is False
        assert no_backend_store.get_secret("openrouter_api_key") is None
        assert no_backend_store.set_secret("openrouter_api_key", "abc") is False

    def test_empty_secret_rejected(self) -> None:
        store = CredentialStore(keyring_backend=InMemoryKeyringBackend(), env_mapping={})
        with pytest.raises(ValueError, match="must not be empty"):
            store.set_secret("google_api_key", "   ")

    def test_load_config_reads_from_credential_store_with_env_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = InMemoryKeyringBackend()
        backend.set_password("IRIS", "openrouter_api_key", "kr-openrouter-val")
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        monkeypatch.setenv("GOOGLE_API_KEY", "env-google-val")
        monkeypatch.setenv("ELEVENLABS_API_KEY", "")
        monkeypatch.setenv("ELEVENLABS_AGENT_ID", "")

        store = CredentialStore(keyring_backend=backend, allow_env_fallback=False)
        cfg = load_config(require_voice=False, credential_store=store)
        assert cfg.openrouter_api_key == "kr-openrouter-val"
        assert cfg.google_api_key == "env-google-val"

    def test_load_keyring_accepts_macos_and_rejects_fail_backends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        import keyring.backends.fail
        import keyring.backends.null

        if sys.platform == "darwin":
            store = CredentialStore(env_mapping={})
            assert store.is_secure_backend_available is True

        for bad_backend in (keyring.backends.fail.Keyring(), keyring.backends.null.Keyring()):
            monkeypatch.setattr("keyring.get_keyring", lambda b=bad_backend: b)
            assert CredentialStore._load_keyring() is None
