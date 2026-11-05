"""
iris/services/context/context_manager.py
-----------------------------------------
Unified Environment Context Manager for IRIS.

Coordinates location detection, timezone resolution, and live weather updates.
Maintains a background refresh worker that keeps environmental context fresh
without blocking the half-duplex voice loop.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from iris.services.context.location import LocationService
from iris.services.context.models import (
    IRISContextSnapshot,
    LocationInfo,
    TimezoneInfo,
    WeatherInfo,
)
from iris.services.context.timezone import TimezoneService
from iris.services.context.weather import WeatherService

if TYPE_CHECKING:
    from iris.core.config import IRISConfig

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Central manager for environmental awareness in IRIS.
    Thread-safe, non-blocking, and resilient to network/offline conditions.
    """

    def __init__(
        self,
        config: IRISConfig | None = None,
        location_service: LocationService | None = None,
        timezone_service: TimezoneService | None = None,
        weather_service: WeatherService | None = None,
        refresh_interval_s: int = 1800,
    ) -> None:
        self.config = config
        cache_dir = getattr(config, "context_cache_dir", Path("memory/context")) if config else Path("memory/context")
        self.cache_dir = Path(cache_dir)
        self.refresh_interval_s = getattr(config, "context_refresh_interval_s", refresh_interval_s) if config else refresh_interval_s

        # Initialize sub-services
        if location_service is not None:
            self.location_service = location_service
        else:
            self.location_service = LocationService(
                cache_dir=self.cache_dir,
                enable_tracking=getattr(config, "enable_location_tracking", True) if config else True,
                fallback_name=getattr(config, "fallback_location_name", "") if config else "",
                fallback_latitude=getattr(config, "fallback_latitude", None) if config else None,
                fallback_longitude=getattr(config, "fallback_longitude", None) if config else None,
            )

        self.timezone_service = timezone_service or TimezoneService()

        if weather_service is not None:
            self.weather_service = weather_service
        else:
            self.weather_service = WeatherService(
                cache_dir=self.cache_dir,
                provider=getattr(config, "weather_provider", "open-meteo") if config else "open-meteo",
                api_key=getattr(config, "weather_api_key", "") if config else "",
                cache_ttl_s=getattr(config, "weather_cache_ttl_s", 1800) if config else 1800,
            )

        # Thread-safe snapshot state
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._is_running = False

        # Initial baseline snapshot
        initial_loc = LocationInfo(name="Local Device", source="init", is_fallback=True)
        initial_tz = self.timezone_service.get_timezone_info(location=initial_loc)
        self._snapshot = IRISContextSnapshot(
            location=initial_loc,
            timezone=initial_tz,
            weather=None,
            is_online=True,
            last_refresh_at=datetime.now().isoformat(),
        )

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self, initial_refresh: bool = True) -> None:
        """Start the background context refresh thread."""
        with self._lock:
            if self._is_running:
                return
            self._is_running = True
            self._stop_event.clear()

        if initial_refresh:
            try:
                self.refresh(force=False)
            except Exception as exc:
                logger.warning("Initial context refresh warning: %s", exc)

        self._worker_thread = threading.Thread(
            target=self._background_loop,
            name="iris-context-engine",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info(
            "Context Engine started (refresh interval: %ds, cache dir: %s)",
            self.refresh_interval_s,
            self.cache_dir,
        )

    def stop(self) -> None:
        """Stop background worker thread cleanly."""
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False
            self._stop_event.set()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

        logger.info("Context Engine stopped")

    def refresh(self, force: bool = False) -> IRISContextSnapshot:
        """
        Synchronously refresh location, timezone, and weather.
        Guarantees thread-safe atomic snapshot update.
        """
        errors: list[str] = []
        is_online = True

        # 1. Location
        try:
            loc = self.location_service.get_location(force_refresh=force)
        except Exception as exc:
            errors.append(f"Location error: {exc}")
            loc = LocationInfo(name="Local Device", source="error_fallback", is_fallback=True)

        # 2. Weather
        weather: WeatherInfo | None = None
        try:
            weather = self.weather_service.get_weather(loc, force_refresh=force)
            if weather and weather.is_stale:
                is_online = False
        except Exception as exc:
            errors.append(f"Weather error: {exc}")
            is_online = False

        # 3. Timezone
        try:
            weather_tz = weather.timezone if (weather and weather.timezone) else None
            tz = self.timezone_service.get_timezone_info(iana_override=weather_tz, location=loc)
        except Exception as exc:
            errors.append(f"Timezone error: {exc}")
            tz = TimezoneInfo(iana_id="UTC", timezone_name="UTC")

        snapshot = IRISContextSnapshot(
            location=loc,
            timezone=tz,
            weather=weather,
            is_online=is_online,
            last_refresh_at=datetime.now().isoformat(),
            errors=errors,
        )

        with self._lock:
            self._snapshot = snapshot

        return snapshot

    def get_context(self) -> IRISContextSnapshot:
        """Return the current immutable context snapshot."""
        with self._lock:
            return self._snapshot

    def get_prompt_context(self) -> str:
        """Return privacy-safe formatted string block for LLM system prompt injection."""
        snapshot = self.get_context()
        # Always recompute fresh date/time string so prompt has exact minute precision
        tz_live = self.timezone_service.get_timezone_info(
            iana_override=snapshot.timezone.iana_id,
            location=snapshot.location,
        )
        live_snapshot = IRISContextSnapshot(
            location=snapshot.location,
            timezone=tz_live,
            weather=snapshot.weather,
            is_online=snapshot.is_online,
            last_refresh_at=snapshot.last_refresh_at,
            errors=snapshot.errors,
        )
        return live_snapshot.to_prompt_context()

    def get_weather_summary(self) -> str:
        """Spoken summary of current weather conditions."""
        return self.get_context().weather_summary_for_voice()

    def get_datetime_summary(self) -> str:
        """Spoken summary of current date/time."""
        return self.get_context().datetime_summary_for_voice()

    def get_location_name(self) -> str:
        """Current detected location name."""
        return self.get_context().location.safe_display_name()

    def _background_loop(self) -> None:
        """Periodic background refresh worker."""
        while not self._stop_event.is_set():
            # Wait for refresh interval or stop signal
            if self._stop_event.wait(timeout=float(self.refresh_interval_s)):
                break
            try:
                self.refresh(force=False)
            except Exception as exc:
                logger.warning("Context Engine background refresh error: %s", exc)
