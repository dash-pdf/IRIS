"""
iris/services/context/weather.py
---------------------------------
Modular weather and forecast service for IRIS.

Architecture:
- Pluggable provider interface (BaseWeatherProvider)
- Apple WeatherKit Provider (Apple-first native REST API using ES256 JWT auth)
- Open-Meteo Provider (zero-key, free-tier default fallback)
- Provider-agnostic caching, TTL staleness detection, and offline resilience
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from iris.services.context.models import (
    LocationInfo,
    WeatherCurrent,
    WeatherDailyForecast,
    WeatherInfo,
    wmo_code_to_condition,
)

if TYPE_CHECKING:
    from iris.core.config import IRISConfig

logger = logging.getLogger(__name__)

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
APPLE_WEATHERKIT_BASE_URL = "https://weatherkit.apple.com/v1/weather"

# Apple WeatherKit condition code mapping
APPLE_WEATHERKIT_CONDITION_MAP: dict[str, str] = {
    "Clear": "Clear sky",
    "Cloudy": "Cloudy",
    "Dust": "Dusty",
    "Drizzle": "Light drizzle",
    "EscapeHatch": "Clear",
    "Foggy": "Foggy",
    "FreezingDrizzle": "Freezing drizzle",
    "FreezingRain": "Freezing rain",
    "Frigid": "Frigid",
    "Hail": "Hail",
    "Haze": "Hazy",
    "HeavyRain": "Heavy rain",
    "HeavySnow": "Heavy snow",
    "Hot": "Hot",
    "Hurricane": "Hurricane",
    "IsolatedThunderstorms": "Isolated thunderstorms",
    "MostlyClear": "Mostly clear",
    "MostlyCloudy": "Mostly cloudy",
    "PartlyCloudy": "Partly cloudy",
    "Rain": "Rain",
    "ScatteredThunderstorms": "Scattered thunderstorms",
    "Sleet": "Sleet",
    "Smoky": "Smoky",
    "Snow": "Snow",
    "StrongStorms": "Strong storms",
    "SunFlurries": "Sun flurries",
    "SunShowers": "Sun showers",
    "Thunderstorms": "Thunderstorms",
    "TropicalStorm": "Tropical storm",
    "Windy": "Windy",
    "Breezy": "Breezy",
    "Blizzard": "Blizzard",
    "BlowingSnow": "Blowing snow",
    "Flurries": "Flurries",
    "WintryMix": "Wintry mix",
}


def _urlsafe_b64(data: bytes) -> str:
    """Base64url encode bytes stripping trailing equals signs."""
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


# ── 1. Abstract Weather Provider Interface ────────────────────────────────────

class BaseWeatherProvider(ABC):
    """Abstract interface for weather data providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique provider identifier."""

    @abstractmethod
    def fetch_weather(
        self,
        lat: float,
        lon: float,
        location_name: str,
    ) -> WeatherInfo | None:
        """Fetch live current weather and forecast for given coordinates."""


# ── 2. Open-Meteo Provider (Default Free Tier) ────────────────────────────────

class OpenMeteoProvider(BaseWeatherProvider):
    """
    Open-Meteo keyless weather provider.
    Free for non-commercial open-source usage with zero API key requirement.
    """

    @property
    def provider_name(self) -> str:
        return "open-meteo"

    def fetch_weather(
        self,
        lat: float,
        lon: float,
        location_name: str,
    ) -> WeatherInfo | None:
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

        return self._parse_response(data, location_name)

    def _parse_response(self, data: dict[str, Any], location_name: str) -> WeatherInfo:
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


# ── 3. Apple WeatherKit Provider (Apple-First) ────────────────────────────────

class AppleWeatherKitProvider(BaseWeatherProvider):
    """
    Apple WeatherKit REST API provider.

    Requires:
    - Apple Developer Program membership ($99/yr)
    - Team ID, Key ID, Service ID, and ES256 Private Key (.p8 file)
    """

    def __init__(
        self,
        team_id: str,
        key_id: str,
        service_id: str,
        private_key: str,
    ) -> None:
        self.team_id = team_id.strip()
        self.key_id = key_id.strip()
        self.service_id = service_id.strip()
        self.private_key_raw = private_key.strip()

        if not (self.team_id and self.key_id and self.service_id and self.private_key_raw):
            raise ValueError(
                "Apple WeatherKit requires team_id, key_id, service_id, and private_key."
            )

        self._cached_jwt: str = ""
        self._jwt_expires_at: float = 0.0

    @property
    def provider_name(self) -> str:
        return "apple"

    def _get_auth_token(self) -> str:
        """Generate or reuse a signed ES256 JSON Web Token."""
        now = time.time()
        # Reuse existing token if valid for at least another 5 minutes
        if self._cached_jwt and now < (self._jwt_expires_at - 300):
            return self._cached_jwt

        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

        # Read key from file or raw string
        if os.path.exists(self.private_key_raw):
            key_bytes = Path(self.private_key_raw).read_bytes()
        else:
            key_bytes = self.private_key_raw.encode("utf-8")

        private_key = serialization.load_pem_private_key(key_bytes, password=None)
        if not isinstance(private_key, ec.EllipticCurvePrivateKey):
            raise ValueError("Apple WeatherKit private key must be an Elliptic Curve private key (P-256).")

        issued_at = int(now)
        expires_at = issued_at + 3600  # 1 hour expiration

        header = {
            "alg": "ES256",
            "kid": self.key_id,
            "id": f"{self.team_id}.{self.service_id}",
        }
        payload = {
            "iss": self.team_id,
            "iat": issued_at,
            "exp": expires_at,
            "sub": self.service_id,
        }

        header_b64 = _urlsafe_b64(json.dumps(header).encode("utf-8"))
        payload_b64 = _urlsafe_b64(json.dumps(payload).encode("utf-8"))
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")

        der_sig = private_key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der_sig)
        raw_sig = r.to_bytes(32, byteorder="big") + s.to_bytes(32, byteorder="big")
        sig_b64 = _urlsafe_b64(raw_sig)

        token = f"{header_b64}.{payload_b64}.{sig_b64}"
        self._cached_jwt = token
        self._jwt_expires_at = expires_at
        return token

    def fetch_weather(
        self,
        lat: float,
        lon: float,
        location_name: str,
    ) -> WeatherInfo | None:
        token = self._get_auth_token()
        url = f"{APPLE_WEATHERKIT_BASE_URL}/en/{lat:.4f}/{lon:.4f}"
        headers = {"Authorization": f"Bearer {token}"}
        params = {
            "dataSets": "currentWeather,forecastDaily",
            "timezone": "auto",
        }

        with httpx.Client(timeout=8.0) as client:
            resp = client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        return self._parse_response(data, location_name)

    def _parse_response(self, data: dict[str, Any], location_name: str) -> WeatherInfo:
        curr_data = data.get("currentWeather", {})
        temp_c = float(curr_data.get("temperature", 0.0))
        cond_code = curr_data.get("conditionCode", "Clear")
        condition = APPLE_WEATHERKIT_CONDITION_MAP.get(cond_code, cond_code)
        humidity_frac = curr_data.get("humidity")
        humidity = int(round(humidity_frac * 100)) if humidity_frac is not None else None
        wind_speed = curr_data.get("windSpeed")
        apparent_temp = curr_data.get("temperatureApparent")
        daylight = bool(curr_data.get("daylight", True))

        current = WeatherCurrent(
            temperature_c=temp_c,
            condition=condition,
            condition_code=0,
            humidity_pct=humidity,
            wind_speed_kmh=float(wind_speed) if wind_speed is not None else None,
            apparent_temperature_c=float(apparent_temp) if apparent_temp is not None else None,
            is_day=daylight,
        )

        daily_data = data.get("forecastDaily", {})
        days = daily_data.get("days", [])

        forecast: list[WeatherDailyForecast] = []
        for day in days[:3]:
            start_str = day.get("forecastStart", "")
            d_str = start_str[:10] if len(start_str) >= 10 else ""
            try:
                dt = datetime.strptime(d_str, "%Y-%m-%d")
                weekday = dt.strftime("%A")
            except Exception:
                weekday = "Upcoming"

            day_cond_code = day.get("conditionCode", "Clear")
            day_cond = APPLE_WEATHERKIT_CONDITION_MAP.get(day_cond_code, day_cond_code)
            max_c = float(day.get("temperatureMax", temp_c))
            min_c = float(day.get("temperatureMin", temp_c))
            p_chance = day.get("precipitationChance")
            p_prob = int(round(p_chance * 100)) if p_chance is not None else None
            p_amount = day.get("precipitationAmount")

            forecast.append(
                WeatherDailyForecast(
                    date=d_str,
                    weekday=weekday,
                    temp_max_c=max_c,
                    temp_min_c=min_c,
                    condition=day_cond,
                    precipitation_probability_pct=p_prob,
                    precipitation_sum_mm=float(p_amount) if p_amount is not None else None,
                )
            )

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
            provider="apple",
            fetched_at=datetime.now().isoformat(),
            is_stale=False,
            summary=summary,
            timezone=str(data.get("timezone", "")),
        )


# ── 4. Weather Provider Factory ───────────────────────────────────────────────

def create_weather_provider(
    config: IRISConfig | None = None,
    provider_name: str | None = None,
) -> BaseWeatherProvider:
    """
    Factory creating the configured weather provider.

    Hierarchy:
    1. If provider is 'apple' or 'weatherkit':
       - If full Apple credentials are present -> AppleWeatherKitProvider
       - If Apple credentials missing -> logs clear notice and falls back to OpenMeteoProvider
    2. Otherwise -> OpenMeteoProvider (default free tier)
    """
    req_provider = provider_name or (getattr(config, "weather_provider", "open-meteo") if config else "open-meteo")
    req_provider = req_provider.lower().strip()

    if req_provider in {"apple", "weatherkit"}:
        team_id = getattr(config, "apple_weather_team_id", "") if config else ""
        key_id = getattr(config, "apple_weather_key_id", "") if config else ""
        service_id = getattr(config, "apple_weather_service_id", "") if config else ""
        private_key = getattr(config, "apple_weather_private_key", "") if config else ""

        if team_id and key_id and service_id and private_key:
            try:
                logger.info("Initializing Apple WeatherKit provider (Team: %s, Key: %s)", team_id, key_id)
                return AppleWeatherKitProvider(
                    team_id=team_id,
                    key_id=key_id,
                    service_id=service_id,
                    private_key=private_key,
                )
            except Exception as exc:
                logger.warning("Failed initializing Apple WeatherKit (%s); falling back to Open-Meteo", exc)
        else:
            logger.info(
                "Apple WeatherKit selected, but Apple Developer credentials (APPLE_WEATHER_TEAM_ID, etc.) "
                "are incomplete. Gracefully falling back to Open-Meteo free tier."
            )

    return OpenMeteoProvider()


# ── 5. Main Weather Service ───────────────────────────────────────────────────

class WeatherService:
    """
    Coordinates weather retrieval across pluggable providers.
    Manages disk caching, TTL expiration, staleness tracking, and offline resilience.
    """

    def __init__(
        self,
        cache_dir: Path,
        provider: BaseWeatherProvider | str | None = None,
        config: IRISConfig | None = None,
        cache_ttl_s: int = 1800,
        api_key: str = "",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "weather_cache.json"
        self.cache_ttl_s = getattr(config, "weather_cache_ttl_s", cache_ttl_s) if config else cache_ttl_s

        if isinstance(provider, BaseWeatherProvider):
            self.provider = provider
        elif isinstance(provider, str):
            self.provider = create_weather_provider(config=config, provider_name=provider)
        else:
            self.provider = create_weather_provider(config=config)

        self._cached_weather: WeatherInfo | None = None
        self._cached_timestamp: float = 0.0

        # Load existing disk cache if available
        self._load_cache()

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    def get_weather(
        self,
        location: LocationInfo,
        force_refresh: bool = False,
    ) -> WeatherInfo | None:
        """
        Retrieve weather for the given location using the active provider.

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

        # Attempt live API fetch via active provider
        try:
            live_weather = self.provider.fetch_weather(
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
                "Live weather fetch via [%s] failed for '%s' (%s). Falling back to cached observations.",
                self.provider.provider_name,
                location.name,
                exc,
            )

        # Offline / Failure fallback: Serve existing cache marked stale
        if self._cached_weather is not None:
            self._cached_weather.is_stale = True
            return self._cached_weather

        return None

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
