"""
iris/services/context/__init__.py
----------------------------------
IRIS Context Engine package.

Exposes:
- ContextManager
- IRISContextSnapshot
- LocationInfo, LocationService
- TimezoneInfo, TimezoneService
- WeatherInfo, WeatherCurrent, WeatherDailyForecast, WeatherService
"""

from __future__ import annotations

from iris.services.context.context_manager import ContextManager
from iris.services.context.location import LocationService
from iris.services.context.models import (
    IRISContextSnapshot,
    LocationInfo,
    TimezoneInfo,
    WeatherCurrent,
    WeatherDailyForecast,
    WeatherInfo,
    WMO_CODE_MAP,
    wmo_code_to_condition,
)
from iris.services.context.timezone import TimezoneService
from iris.services.context.weather import (
    APPLE_WEATHERKIT_CONDITION_MAP,
    AppleWeatherKitProvider,
    BaseWeatherProvider,
    OpenMeteoProvider,
    WeatherService,
    create_weather_provider,
)

__all__ = [
    "ContextManager",
    "IRISContextSnapshot",
    "LocationInfo",
    "LocationService",
    "TimezoneInfo",
    "TimezoneService",
    "WeatherCurrent",
    "WeatherDailyForecast",
    "WeatherInfo",
    "WeatherService",
    "BaseWeatherProvider",
    "OpenMeteoProvider",
    "AppleWeatherKitProvider",
    "create_weather_provider",
    "APPLE_WEATHERKIT_CONDITION_MAP",
    "WMO_CODE_MAP",
    "wmo_code_to_condition",
]

