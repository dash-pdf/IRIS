"""
iris/services/context/weather.py
---------------------------------
Weather and forecast service for IRIS.

Uses Open-Meteo's free, keyless REST API by default with robust
disk and memory caching, rate-limit awareness, staleness tracking,
and offline graceful degradation.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from iris.services.context.models import (
    LocationInfo,
    WeatherCurrent,
    WeatherDailyForecast,
    WeatherInfo,
    wmo_code_to_condition,
)

logger = logging.getLogger(__name__)

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherService:
    """
    Fetches and caches live weather conditions and multi-day forecasts.
    Guarantees offline resilience through persistent cache fallback.
    """

    def __init__(
        self,
        cache_dir: Path,
        provider: str = "open-meteo",
        api_key: str = "",
        cache_ttl_s: int = 1800,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "weather_cache.json"
        self.provider = provider.lower()
        self.api_key = api_key
        self.cache_ttl_s = cache_ttl_s

        self._cached_weather: WeatherInfo | None = None
        self._cached_timestamp: float = 0.0

        # Load existing disk cache if available
        self._load_cache()

    def get_weather(
        self,
        location: LocationInfo,
        force_refresh: bool = False,
    ) -> WeatherInfo | None:
        """
        Retrieve weather for the given location.

        Returns WeatherInfo with `is_stale=False` on live or fresh cached fetch,
        or `is_stale=True` when offline / using expired cache.
        Returns None only if coordinates are unavailable and no cache exists.
        """
        now = time.time()

        # Check in-memory fresh cache
        if (
            not force_refresh
            and self._cached_weather is not None
            and (now - self._cached_timestamp) < self.cache_ttl_s
        ):
            return self._cached_weather

        # If location lacks coordinates, rely on disk/memory cache
        if location.latitude is None or location.longitude is None:
            logger.debug("Location '%s' has no coordinates; serving cached weather if any", location.name)
            if self._cached_weather:
                self._cached_weather.is_stale = True
                return self._cached_weather
            return None

        # Attempt live API fetch
        try:
            live_weather = self._fetch_open_meteo(
                lat=location.latitude,
                lon=location.longitude,
                location_name=location.safe_display_name(),
            )
            if live_weather:
                self._cached_weather = live_weather
                self._cached_timestamp = now
                self._save_cache(live_weather, timestamp=now)
                return live_weather
        except Exception as exc:
            logger.warning(
                "Live weather fetch failed for '%s' (%s). Falling back to cached observations.",
                location.name,
                exc,
            )

        # Offline / Failure fallback: Serve existing cache marked stale
        if self._cached_weather is not None:
            self._cached_weather.is_stale = True
            return self._cached_weather

        return None

    def _fetch_open_meteo(
        self,
        lat: float,
        lon: float,
        location_name: str,
    ) -> WeatherInfo | None:
        """Call Open-Meteo REST endpoint and parse the response."""
        params = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 3,
        }

        with httpx.Client(timeout=8.0) as client:
            resp = client.get(OPEN_METEO_FORECAST_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        return self._parse_open_meteo_response(data, location_name)

    def _parse_open_meteo_response(
        self,
        data: dict[str, Any],
        location_name: str,
    ) -> WeatherInfo:
        """Transform raw Open-Meteo payload into structured WeatherInfo."""
        curr_data = data.get("current", {})
        temp_c = float(curr_data.get("temperature_2m", 0.0))
        wmo_code = int(curr_data.get("weather_code", 0))
        condition = wmo_code_to_condition(wmo_code)
        humidity = curr_data.get("relative_humidity_2m")
        wind_speed = curr_data.get("wind_speed_10m")
        apparent_temp = curr_data.get("apparent_temperature")
        is_day = bool(curr_data.get("is_day", 1))

        current = WeatherCurrent(
            temperature_c=temp_c,
            condition=condition,
            condition_code=wmo_code,
            humidity_pct=int(humidity) if humidity is not None else None,
            wind_speed_kmh=float(wind_speed) if wind_speed is not None else None,
            apparent_temperature_c=float(apparent_temp) if apparent_temp is not None else None,
            is_day=is_day,
        )

        daily_data = data.get("daily", {})
        dates = daily_data.get("time", [])
        wmo_codes = daily_data.get("weather_code", [])
        t_max = daily_data.get("temperature_2m_max", [])
        t_min = daily_data.get("temperature_2m_min", [])
        precip_sums = daily_data.get("precipitation_sum", [])
        precip_probs = daily_data.get("precipitation_probability_max", [])

        forecast: list[WeatherDailyForecast] = []
        for i in range(len(dates)):
            d_str = str(dates[i])
            try:
                dt = datetime.strptime(d_str, "%Y-%m-%d")
                weekday = dt.strftime("%A")
            except Exception:
                weekday = "Upcoming"

            code_i = int(wmo_codes[i]) if i < len(wmo_codes) else 0
            max_c = float(t_max[i]) if i < len(t_max) else temp_c
            min_c = float(t_min[i]) if i < len(t_min) else temp_c
            p_prob = int(precip_probs[i]) if i < len(precip_probs) and precip_probs[i] is not None else None
            p_sum = float(precip_sums[i]) if i < len(precip_sums) and precip_sums[i] is not None else None

            forecast.append(
                WeatherDailyForecast(
                    date=d_str,
                    weekday=weekday,
                    temp_max_c=max_c,
                    temp_min_c=min_c,
                    condition=wmo_code_to_condition(code_i),
                    precipitation_probability_pct=p_prob,
                    precipitation_sum_mm=p_sum,
                )
            )

        # Build natural spoken summary
        summary = f"{int(round(temp_c))}°C, {condition} in {location_name}."
        if forecast:
            today_fc = forecast[0]
            summary += (
                f" Today's forecast: High {int(round(today_fc.temp_max_c))}°C, "
                f"low {int(round(today_fc.temp_min_c))}°C"
            )
            if today_fc.precipitation_probability_pct is not None:
                summary += f" with {today_fc.precipitation_probability_pct}% chance of rain"
            summary += "."

        return WeatherInfo(
            current=current,
            forecast=forecast,
            provider="open-meteo",
            fetched_at=datetime.now().isoformat(),
            is_stale=False,
            summary=summary,
            timezone=str(data.get("timezone", "")),
        )

    def _load_cache(self) -> None:
        """Load weather cache from disk if available."""
        if not self.cache_file.exists():
            return
        try:
            content = self.cache_file.read_text(encoding="utf-8")
            data = json.loads(content)
            self._cached_timestamp = float(data.get("timestamp", 0.0))
            weather_data = data.get("weather")
            if weather_data:
                self._cached_weather = WeatherInfo.from_dict(weather_data)
                # If cache is older than TTL, mark stale
                if (time.time() - self._cached_timestamp) >= self.cache_ttl_s:
                    self._cached_weather.is_stale = True
        except Exception as exc:
            logger.warning("Failed reading weather cache from %s: %s", self.cache_file, exc)

    def _save_cache(self, weather: WeatherInfo, timestamp: float) -> None:
        """Persist weather cache to disk."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": timestamp,
                "weather": weather.to_dict(),
            }
            self.cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed writing weather cache to %s: %s", self.cache_file, exc)
