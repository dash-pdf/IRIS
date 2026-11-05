"""
iris/services/context/models.py
--------------------------------
Structured data models for the IRIS Context Engine.

Represents environment awareness:
- Physical device location (privacy-safe, coordinates never leaked to logs/archives)
- Local timezone, daylight saving time (DST), and current date/time
- Live weather conditions and multi-day forecast
- Snapshot aggregation with data freshness and offline resilience
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Standard WMO Weather Interpretation Codes (WW)
WMO_CODE_MAP: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def wmo_code_to_condition(code: int) -> str:
    """Map a WMO weather code to a human-readable condition string."""
    return WMO_CODE_MAP.get(code, "Clear")


@dataclass
class LocationInfo:
    """
    Physical device location metadata.

    Privacy Guarantee:
    Coordinates (latitude/longitude) are maintained strictly in memory and local
    cache for weather fetching. They are NEVER printed in __repr__, NEVER included in
    conversation archives, and NEVER leaked into logs or LLM context prompts.
    """

    name: str = "Local Device"
    latitude: float | None = None
    longitude: float | None = None
    accuracy_m: float | None = None
    source: str = "unknown"  # "macos_corelocation" | "manual_fallback" | "cached" | "default_fallback" | "unknown"
    is_fallback: bool = False

    def __repr__(self) -> str:
        # Intentionally suppress latitude/longitude from string representations
        return (
            f"LocationInfo(name={self.name!r}, source={self.source!r}, "
            f"is_fallback={self.is_fallback})"
        )

    def safe_display_name(self) -> str:
        """Return a human-friendly location string without exposing coordinates."""
        return self.name or "Local Device"

    def to_dict(self, include_coordinates: bool = True) -> dict[str, Any]:
        """Serialize location info. Coordinates can be excluded for privacy."""
        data: dict[str, Any] = {
            "name": self.name,
            "source": self.source,
            "is_fallback": self.is_fallback,
        }
        if include_coordinates:
            data["latitude"] = self.latitude
            data["longitude"] = self.longitude
            data["accuracy_m"] = self.accuracy_m
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LocationInfo:
        return cls(
            name=data.get("name", "Local Device"),
            latitude=data.get("latitude"),
            longitude=data.get("longitude"),
            accuracy_m=data.get("accuracy_m"),
            source=data.get("source", "cached"),
            is_fallback=data.get("is_fallback", False),
        )


@dataclass
class TimezoneInfo:
    """Local timezone details, current date/time, and DST status."""

    iana_id: str = "UTC"
    utc_offset_hours: float = 0.0
    is_dst: bool = False
    timezone_name: str = "UTC"
    current_time_str: str = ""
    current_date_str: str = ""
    day_of_week: str = ""
    iso_timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "iana_id": self.iana_id,
            "utc_offset_hours": self.utc_offset_hours,
            "is_dst": self.is_dst,
            "timezone_name": self.timezone_name,
            "current_time_str": self.current_time_str,
            "current_date_str": self.current_date_str,
            "day_of_week": self.day_of_week,
            "iso_timestamp": self.iso_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimezoneInfo:
        return cls(
            iana_id=data.get("iana_id", "UTC"),
            utc_offset_hours=float(data.get("utc_offset_hours", 0.0)),
            is_dst=bool(data.get("is_dst", False)),
            timezone_name=data.get("timezone_name", "UTC"),
            current_time_str=data.get("current_time_str", ""),
            current_date_str=data.get("current_date_str", ""),
            day_of_week=data.get("day_of_week", ""),
            iso_timestamp=data.get("iso_timestamp", ""),
        )


@dataclass
class WeatherCurrent:
    """Current weather observations."""

    temperature_c: float
    condition: str
    condition_code: int = 0
    humidity_pct: int | None = None
    wind_speed_kmh: float | None = None
    apparent_temperature_c: float | None = None
    is_day: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature_c": self.temperature_c,
            "condition": self.condition,
            "condition_code": self.condition_code,
            "humidity_pct": self.humidity_pct,
            "wind_speed_kmh": self.wind_speed_kmh,
            "apparent_temperature_c": self.apparent_temperature_c,
            "is_day": self.is_day,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WeatherCurrent:
        return cls(
            temperature_c=float(data.get("temperature_c", 0.0)),
            condition=data.get("condition", "Clear"),
            condition_code=int(data.get("condition_code", 0)),
            humidity_pct=data.get("humidity_pct"),
            wind_speed_kmh=data.get("wind_speed_kmh"),
            apparent_temperature_c=data.get("apparent_temperature_c"),
            is_day=bool(data.get("is_day", True)),
        )


@dataclass
class WeatherDailyForecast:
    """Daily forecast data for a single upcoming day."""

    date: str  # YYYY-MM-DD
    weekday: str  # e.g. "Friday"
    temp_max_c: float
    temp_min_c: float
    condition: str
    precipitation_probability_pct: int | None = None
    precipitation_sum_mm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "weekday": self.weekday,
            "temp_max_c": self.temp_max_c,
            "temp_min_c": self.temp_min_c,
            "condition": self.condition,
            "precipitation_probability_pct": self.precipitation_probability_pct,
            "precipitation_sum_mm": self.precipitation_sum_mm,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WeatherDailyForecast:
        return cls(
            date=data.get("date", ""),
            weekday=data.get("weekday", ""),
            temp_max_c=float(data.get("temp_max_c", 0.0)),
            temp_min_c=float(data.get("temp_min_c", 0.0)),
            condition=data.get("condition", "Clear"),
            precipitation_probability_pct=data.get("precipitation_probability_pct"),
            precipitation_sum_mm=data.get("precipitation_sum_mm"),
        )


@dataclass
class WeatherInfo:
    """Aggregated weather package with current conditions, forecast, and staleness."""

    current: WeatherCurrent | None = None
    forecast: list[WeatherDailyForecast] = field(default_factory=list)
    provider: str = "open-meteo"
    fetched_at: str = ""
    is_stale: bool = False
    summary: str = ""
    timezone: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": self.current.to_dict() if self.current else None,
            "forecast": [f.to_dict() for f in self.forecast],
            "provider": self.provider,
            "fetched_at": self.fetched_at,
            "is_stale": self.is_stale,
            "summary": self.summary,
            "timezone": self.timezone,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WeatherInfo:
        current_data = data.get("current")
        current = WeatherCurrent.from_dict(current_data) if current_data else None
        forecast_list = [
            WeatherDailyForecast.from_dict(f)
            for f in data.get("forecast", [])
            if isinstance(f, dict)
        ]
        return cls(
            current=current,
            forecast=forecast_list,
            provider=data.get("provider", "open-meteo"),
            fetched_at=data.get("fetched_at", ""),
            is_stale=bool(data.get("is_stale", False)),
            summary=data.get("summary", ""),
            timezone=data.get("timezone", ""),
        )


@dataclass
class IRISContextSnapshot:
    """
    Unified immutable environment snapshot for IRIS components.

    Thread-safe copy provided to conversational router, prompt composer,
    and voice synthesizers.
    """

    location: LocationInfo
    timezone: TimezoneInfo
    weather: WeatherInfo | None = None
    is_online: bool = True
    last_refresh_at: str = ""
    errors: list[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """
        Generate a concise, privacy-safe context block for system prompt injection.
        Crucially excludes raw latitude and longitude coordinates.
        """
        lines = [
            "Current Context:",
            f"- Date & Time: {self.timezone.current_date_str}, {self.timezone.current_time_str} "
            f"({self.timezone.timezone_name}, {self.timezone.iana_id})",
            f"- Location: {self.location.safe_display_name()}",
        ]
        if self.weather and self.weather.summary:
            stale_note = " [Cached/Offline]" if self.weather.is_stale else ""
            lines.append(f"- Weather: {self.weather.summary}{stale_note}")
        return "\n".join(lines)

    def weather_summary_for_voice(self) -> str:
        """Natural spoken English summary of current weather conditions."""
        if not self.weather or not self.weather.current:
            return f"Weather information is currently unavailable for {self.location.safe_display_name()}."

        loc = self.location.safe_display_name()
        curr = self.weather.current
        temp = int(round(curr.temperature_c))
        cond = curr.condition.lower()

        if self.weather.forecast:
            today_fc = self.weather.forecast[0]
            t_max = int(round(today_fc.temp_max_c))
            t_min = int(round(today_fc.temp_min_c))
            msg = f"In {loc}, it is currently {temp} degrees and {cond}. Today's forecast calls for a high of {t_max} degrees and a low of {t_min} degrees."
        else:
            msg = f"In {loc}, it is currently {temp} degrees and {cond}."

        if self.weather.is_stale:
            msg += " (Note: This is based on cached observations)."
        return msg

    def datetime_summary_for_voice(self) -> str:
        """Natural spoken English summary of local date and time."""
        return (
            f"It is currently {self.timezone.current_time_str} on "
            f"{self.timezone.current_date_str} in the {self.timezone.timezone_name} timezone."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location.to_dict(include_coordinates=False),
            "timezone": self.timezone.to_dict(),
            "weather": self.weather.to_dict() if self.weather else None,
            "is_online": self.is_online,
            "last_refresh_at": self.last_refresh_at,
            "errors": list(self.errors),
        }
