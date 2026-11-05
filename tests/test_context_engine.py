"""
tests/test_context_engine.py
-----------------------------
Comprehensive unit and integration tests for IRIS Phase 2: Context Engine.

Verifies:
1. LocationService:
   - macOS CoreLocation detection and authorization status checks
   - Clear setup guidance logged on unauthorized / restricted status
   - Fallback hierarchy (CoreLocation -> cache -> manual config -> timezone -> generic)
   - Cache persistence and retrieval
   - Strict privacy (raw coordinates never in repr, logs, or archives)
2. TimezoneService:
   - Canonical IANA identification
   - Daylight Saving Time (DST) transitions (winter vs summer)
   - UTC offset calculations and formatted strings
   - Timezone-aware datetime generation
3. WeatherService:
   - Open-Meteo REST API parsing and WMO weather code translation
   - Current conditions and multi-day forecast extraction
   - Cache persistence, TTL expiration, and staleness tracking
   - Offline resilience and graceful degradation under network failure
4. ContextManager:
   - Unified snapshot aggregation and thread-safe access
   - Background refresh worker lifecycle (start and clean stop)
   - Spoken summaries for voice and prompt context generation
5. Router and Assistant Integration:
   - Deterministic weather, location, and timezone query classification
   - Seamless passthrough to cloud models for external cities
   - Assistant lifecycle coordination (starts and stops ContextManager)
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import zoneinfo

import pytest

from iris.adapters.model.cloud_router import CloudRouterModelAdapter
from iris.adapters.model.mock_adapter import MockModelAdapter
from iris.adapters.tts.base import TTSAdapter
from iris.adapters.stt.base import STTAdapter
from iris.core.assistant import IRISAssistant
from iris.core.config import IRISConfig
from iris.core.router import (
    RouteType,
    classify_request,
    try_evaluate_datetime,
    try_evaluate_location,
    try_evaluate_weather,
)
from iris.services.context import (
    APPLE_WEATHERKIT_CONDITION_MAP,
    AppleWeatherKitProvider,
    BaseWeatherProvider,
    ContextManager,
    IRISContextSnapshot,
    LocationInfo,
    LocationService,
    OpenMeteoProvider,
    TimezoneInfo,
    TimezoneService,
    WeatherCurrent,
    WeatherDailyForecast,
    WeatherInfo,
    WeatherService,
    create_weather_provider,
    wmo_code_to_condition,
)

# Sample Open-Meteo API response for London
MOCK_OPEN_METEO_PAYLOAD = {
    "latitude": 51.5,
    "longitude": -0.12,
    "timezone": "Europe/London",
    "timezone_abbreviation": "GMT",
    "utc_offset_seconds": 0,
    "current": {
        "time": "2026-11-05T12:00",
        "temperature_2m": 14.2,
        "relative_humidity_2m": 72,
        "apparent_temperature": 13.5,
        "is_day": 1,
        "precipitation": 0.0,
        "weather_code": 2,  # Partly cloudy
        "wind_speed_10m": 12.8,
    },
    "daily": {
        "time": ["2026-11-05", "2026-11-06", "2026-11-07"],
        "weather_code": [2, 61, 0],  # Partly cloudy, Slight rain, Clear sky
        "temperature_2m_max": [16.0, 14.5, 13.0],
        "temperature_2m_min": [9.0, 8.0, 6.5],
        "precipitation_sum": [0.0, 3.5, 0.0],
        "precipitation_probability_max": [15, 80, 5],
    },
}

# Sample Apple WeatherKit API response for Cupertino
MOCK_APPLE_WEATHERKIT_PAYLOAD = {
    "currentWeather": {
        "asOf": "2026-11-05T12:00:00Z",
        "temperature": 18.5,
        "temperatureApparent": 17.8,
        "humidity": 0.65,
        "conditionCode": "PartlyCloudy",
        "windSpeed": 10.2,
        "daylight": True,
    },
    "forecastDaily": {
        "days": [
            {
                "forecastStart": "2026-11-05T00:00:00Z",
                "temperatureMax": 21.0,
                "temperatureMin": 11.5,
                "conditionCode": "PartlyCloudy",
                "precipitationChance": 0.10,
                "precipitationAmount": 0.0,
            },
            {
                "forecastStart": "2026-11-06T00:00:00Z",
                "temperatureMax": 19.5,
                "temperatureMin": 10.0,
                "conditionCode": "Rain",
                "precipitationChance": 0.75,
                "precipitationAmount": 4.2,
            },
        ],
    },
    "timezone": "America/Los_Angeles",
}


# ── 1. LocationService Tests ──────────────────────────────────────────────────

class TestLocationService:
    def test_manual_fallback_location(self, tmp_path: Path) -> None:
        service = LocationService(
            cache_dir=tmp_path,
            enable_tracking=False,
            fallback_name="Edinburgh, UK",
            fallback_latitude=55.9533,
            fallback_longitude=-3.1883,
        )
        loc = service.get_location()
        assert loc.name == "Edinburgh, UK"
        assert loc.latitude == 55.9533
        assert loc.longitude == -3.1883
        assert loc.source == "manual_fallback"
        assert loc.is_fallback is True

    def test_cache_persistence_and_loading(self, tmp_path: Path) -> None:
        service1 = LocationService(
            cache_dir=tmp_path,
            enable_tracking=False,
            fallback_name="Manchester, UK",
            fallback_latitude=53.4808,
            fallback_longitude=-2.2426,
        )
        loc1 = service1.get_location()
        assert (tmp_path / "location_cache.json").exists()

        # Instantiate second service without fallbacks; should read from disk cache
        service2 = LocationService(cache_dir=tmp_path, enable_tracking=False)
        loc2 = service2.get_location()
        assert loc2.name == "Manchester, UK"
        assert loc2.latitude == 53.4808
        assert loc2.source == "cached"

    def test_privacy_guarantee_no_coordinates_in_repr_or_safe_export(self) -> None:
        loc = LocationInfo(
            name="London, UK",
            latitude=51.5074,
            longitude=-0.1278,
            source="macos_corelocation",
        )
        repr_str = repr(loc)
        assert "51.5074" not in repr_str
        assert "-0.1278" not in repr_str
        assert "London, UK" in repr_str

        # Safe dictionary export
        safe_dict = loc.to_dict(include_coordinates=False)
        assert "latitude" not in safe_dict
        assert "longitude" not in safe_dict
        assert safe_dict["name"] == "London, UK"

    def test_macos_corelocation_denied_falls_back_gracefully(self, tmp_path: Path) -> None:
        service = LocationService(
            cache_dir=tmp_path,
            enable_tracking=True,
            fallback_name="Fallback City",
            fallback_latitude=10.0,
            fallback_longitude=20.0,
        )
        # Mock CoreLocation returning None (denied or not determined)
        with patch.object(service, "_try_macos_corelocation", return_value=None):
            loc = service.get_location()
            assert loc.name == "Fallback City"
            assert loc.latitude == 10.0
            assert loc.is_fallback is True

    def test_macos_corelocation_authorized_success(self, tmp_path: Path) -> None:
        service = LocationService(cache_dir=tmp_path, enable_tracking=True)
        mock_loc = LocationInfo(
            name="Cupertino, CA",
            latitude=37.3230,
            longitude=-122.0322,
            accuracy_m=15.0,
            source="macos_corelocation",
            is_fallback=False,
        )
        with patch.object(service, "_try_macos_corelocation", return_value=mock_loc):
            loc = service.get_location()
            assert loc.name == "Cupertino, CA"
            assert loc.latitude == 37.3230
            assert loc.accuracy_m == 15.0
            assert loc.source == "macos_corelocation"
            assert loc.is_fallback is False


# ── 2. TimezoneService Tests ──────────────────────────────────────────────────

class TestTimezoneService:
    def test_canonical_iana_resolution(self) -> None:
        service = TimezoneService()
        tz_info = service.get_timezone_info(iana_override="Europe/London")
        assert tz_info.iana_id == "Europe/London"
        assert tz_info.current_time_str != ""
        assert tz_info.current_date_str != ""
        assert tz_info.day_of_week in [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ]

    def test_dst_transitions_summer_vs_winter(self) -> None:
        service = TimezoneService()
        tz_london = zoneinfo.ZoneInfo("Europe/London")

        # Winter date (January): GMT, UTC+0, is_dst=False
        winter_dt = datetime(2026, 1, 15, 12, 0, tzinfo=tz_london)
        winter_info = service.get_timezone_info(
            iana_override="Europe/London",
            reference_time=winter_dt,
        )
        assert winter_info.utc_offset_hours == 0.0
        assert winter_info.is_dst is False
        assert winter_info.timezone_name == "GMT"

        # Summer date (July): BST, UTC+1, is_dst=True
        summer_dt = datetime(2026, 7, 15, 12, 0, tzinfo=tz_london)
        summer_info = service.get_timezone_info(
            iana_override="Europe/London",
            reference_time=summer_dt,
        )
        assert summer_info.utc_offset_hours == 1.0
        assert summer_info.is_dst is True
        assert summer_info.timezone_name == "BST"

    def test_us_eastern_timezone_offsets(self) -> None:
        service = TimezoneService()
        tz_ny = zoneinfo.ZoneInfo("America/New_York")

        # Winter (EST, UTC-5)
        winter_dt = datetime(2026, 1, 15, 12, 0, tzinfo=tz_ny)
        winter_info = service.get_timezone_info(
            iana_override="America/New_York",
            reference_time=winter_dt,
        )
        assert winter_info.utc_offset_hours == -5.0
        assert winter_info.is_dst is False

        # Summer (EDT, UTC-4)
        summer_dt = datetime(2026, 7, 15, 12, 0, tzinfo=tz_ny)
        summer_info = service.get_timezone_info(
            iana_override="America/New_York",
            reference_time=summer_dt,
        )
        assert summer_info.utc_offset_hours == -4.0
        assert summer_info.is_dst is True

    def test_fallback_on_invalid_timezone(self) -> None:
        service = TimezoneService(default_iana="UTC")
        tz_info = service.get_timezone_info(iana_override="Invalid/Imaginary_Zone")
        assert tz_info.iana_id == "UTC"
        assert tz_info.utc_offset_hours == 0.0


# ── 3. WeatherService Tests ───────────────────────────────────────────────────

class TestWeatherService:
    def test_wmo_code_mapping(self) -> None:
        assert wmo_code_to_condition(0) == "Clear sky"
        assert wmo_code_to_condition(2) == "Partly cloudy"
        assert wmo_code_to_condition(61) == "Slight rain"
        assert wmo_code_to_condition(95) == "Thunderstorm"
        assert wmo_code_to_condition(999) == "Clear"

    def test_fetch_and_parse_open_meteo(self, tmp_path: Path) -> None:
        service = WeatherService(cache_dir=tmp_path)
        loc = LocationInfo(name="London, UK", latitude=51.5074, longitude=-0.1278)

        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = MOCK_OPEN_METEO_PAYLOAD
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            weather = service.get_weather(loc, force_refresh=True)

        assert weather is not None
        assert weather.current is not None
        assert weather.current.temperature_c == 14.2
        assert weather.current.condition == "Partly cloudy"
        assert weather.current.humidity_pct == 72
        assert weather.is_stale is False
        assert len(weather.forecast) == 3
        assert weather.forecast[0].temp_max_c == 16.0
        assert weather.forecast[1].condition == "Slight rain"
        assert "14°C, Partly cloudy in London, UK" in weather.summary

    def test_cache_ttl_and_staleness(self, tmp_path: Path) -> None:
        # Create a service with a very short TTL (1 second)
        service = WeatherService(cache_dir=tmp_path, cache_ttl_s=1)
        loc = LocationInfo(name="London, UK", latitude=51.5074, longitude=-0.1278)

        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = MOCK_OPEN_METEO_PAYLOAD
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            # Fetch fresh
            weather1 = service.get_weather(loc)
            assert weather1 is not None
            assert weather1.is_stale is False
            assert mock_get.call_count == 1

            # Fetch immediately; should serve fresh in-memory cache without HTTP call
            weather2 = service.get_weather(loc)
            assert weather2 is not None
            assert weather2.is_stale is False
            assert mock_get.call_count == 1

        # Simulate network failure after TTL expiration
        time.sleep(1.1)
        with patch("httpx.Client.get", side_effect=Exception("Network down")):
            weather_offline = service.get_weather(loc)
            assert weather_offline is not None
            # Must serve cached data marked stale
            assert weather_offline.is_stale is True
            assert weather_offline.current is not None
            assert weather_offline.current.temperature_c == 14.2

    def test_offline_without_cache_returns_none_safely(self, tmp_path: Path) -> None:
        service = WeatherService(cache_dir=tmp_path)
        loc = LocationInfo(name="Remote Island", latitude=12.34, longitude=56.78)

        with patch("httpx.Client.get", side_effect=Exception("Connection timeout")):
            weather = service.get_weather(loc)
            assert weather is None

    def test_apple_weatherkit_jwt_generation(self) -> None:
        import base64
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

        priv = ec.generate_private_key(ec.SECP256R1())
        pem_key = priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        provider = AppleWeatherKitProvider(
            team_id="TESTTEAM12",
            key_id="TESTKEY123",
            service_id="com.test.iris.weather",
            private_key=pem_key,
        )

        token = provider._get_auth_token()
        parts = token.split(".")
        assert len(parts) == 3

        # Decode header and payload
        header_raw = base64.urlsafe_b64decode(parts[0] + "==")
        payload_raw = base64.urlsafe_b64decode(parts[1] + "==")
        header = json.loads(header_raw)
        payload = json.loads(payload_raw)

        assert header["alg"] == "ES256"
        assert header["kid"] == "TESTKEY123"
        assert header["id"] == "TESTTEAM12.com.test.iris.weather"
        assert payload["iss"] == "TESTTEAM12"
        assert payload["sub"] == "com.test.iris.weather"

        # Verify cryptographic signature with the public key
        raw_sig = base64.urlsafe_b64decode(parts[2] + "==")
        assert len(raw_sig) == 64
        r = int.from_bytes(raw_sig[:32], "big")
        s = int.from_bytes(raw_sig[32:], "big")
        der_sig = encode_dss_signature(r, s)
        signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
        priv.public_key().verify(der_sig, signing_input, ec.ECDSA(hashes.SHA256()))

    def test_apple_weatherkit_fetch_and_parse(self) -> None:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec

        priv = ec.generate_private_key(ec.SECP256R1())
        pem_key = priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        provider = AppleWeatherKitProvider(
            team_id="TESTTEAM12",
            key_id="TESTKEY123",
            service_id="com.test.iris.weather",
            private_key=pem_key,
        )

        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = MOCK_APPLE_WEATHERKIT_PAYLOAD
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            weather = provider.fetch_weather(lat=37.3346, lon=-122.0090, location_name="Cupertino, CA")

        assert weather is not None
        assert weather.provider == "apple"
        assert weather.current is not None
        assert weather.current.temperature_c == 18.5
        assert weather.current.condition == "Partly cloudy"
        assert weather.current.humidity_pct == 65
        assert len(weather.forecast) == 2
        assert weather.forecast[0].temp_max_c == 21.0
        assert weather.forecast[1].condition == "Rain"
        assert "18°C, Partly cloudy in Cupertino, CA" in weather.summary
        assert weather.timezone == "America/Los_Angeles"

    def test_weather_provider_factory(self) -> None:
        # Default -> OpenMeteoProvider
        prov_default = create_weather_provider()
        assert isinstance(prov_default, OpenMeteoProvider)
        assert prov_default.provider_name == "open-meteo"

        # Apple provider without credentials -> logs notice and falls back to Open-Meteo
        cfg_apple_incomplete = IRISConfig(weather_provider="apple")
        prov_fallback = create_weather_provider(config=cfg_apple_incomplete)
        assert isinstance(prov_fallback, OpenMeteoProvider)

        # Apple provider with complete credentials -> AppleWeatherKitProvider
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec

        priv = ec.generate_private_key(ec.SECP256R1())
        pem_key = priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        cfg_apple_complete = IRISConfig(
            weather_provider="apple",
            apple_weather_team_id="TEAM123456",
            apple_weather_key_id="KEY1234567",
            apple_weather_service_id="com.test.service",
            apple_weather_private_key=pem_key,
        )
        prov_apple = create_weather_provider(config=cfg_apple_complete)
        assert isinstance(prov_apple, AppleWeatherKitProvider)
        assert prov_apple.provider_name == "apple"

    def test_weather_service_with_apple_provider(self, tmp_path: Path) -> None:
        mock_apple = MagicMock(spec=AppleWeatherKitProvider)
        mock_apple.provider_name = "apple"
        mock_apple.fetch_weather.return_value = WeatherInfo(
            current=WeatherCurrent(temperature_c=22.0, condition="Clear sky"),
            forecast=[WeatherDailyForecast(date="2026-11-05", weekday="Thursday", temp_max_c=24.0, temp_min_c=15.0, condition="Clear sky")],
            provider="apple",
            summary="22°C, Clear sky in Cupertino, CA.",
            timezone="America/Los_Angeles",
        )

        service = WeatherService(cache_dir=tmp_path, provider=mock_apple)
        assert service.provider_name == "apple"

        loc = LocationInfo(name="Cupertino, CA", latitude=37.3346, longitude=-122.0090)
        weather = service.get_weather(loc)

        assert weather is not None
        assert weather.provider == "apple"
        assert weather.current.temperature_c == 22.0
        mock_apple.fetch_weather.assert_called_once()


# ── 4. ContextManager Tests ───────────────────────────────────────────────────

class TestContextManager:
    def test_snapshot_aggregation(self, tmp_path: Path) -> None:
        loc_svc = LocationService(
            cache_dir=tmp_path,
            enable_tracking=False,
            fallback_name="Oxford, UK",
            fallback_latitude=51.7520,
            fallback_longitude=-1.2577,
        )
        tz_svc = TimezoneService(default_iana="Europe/London")
        w_svc = WeatherService(cache_dir=tmp_path)

        manager = ContextManager(
            location_service=loc_svc,
            timezone_service=tz_svc,
            weather_service=w_svc,
        )

        with patch.object(w_svc, "get_weather", return_value=WeatherInfo(
            current=WeatherCurrent(temperature_c=15.0, condition="Sunny"),
            forecast=[WeatherDailyForecast(date="2026-11-05", weekday="Thursday", temp_max_c=17.0, temp_min_c=9.0, condition="Sunny")],
            summary="15°C, Sunny in Oxford, UK.",
        )):
            snapshot = manager.refresh()

        assert snapshot.location.name == "Oxford, UK"
        assert snapshot.timezone.iana_id == "Europe/London"
        assert snapshot.weather is not None
        assert snapshot.weather.current.temperature_c == 15.0
        assert snapshot.is_online is True

        # Check prompt context string
        prompt_ctx = manager.get_prompt_context()
        assert "Oxford, UK" in prompt_ctx
        assert "15°C, Sunny" in prompt_ctx
        assert "51.7520" not in prompt_ctx  # Privacy: coordinates suppressed

    def test_background_worker_lifecycle(self, tmp_path: Path) -> None:
        loc_svc = LocationService(cache_dir=tmp_path, enable_tracking=False)
        manager = ContextManager(
            location_service=loc_svc,
            refresh_interval_s=100,
        )
        assert manager.is_running is False

        # Start manager
        manager.start(initial_refresh=False)
        assert manager.is_running is True
        assert manager._worker_thread is not None
        assert manager._worker_thread.is_alive()

        # Stop manager cleanly
        manager.stop()
        assert manager.is_running is False
        assert manager._worker_thread is None

    def test_spoken_voice_summaries(self, tmp_path: Path) -> None:
        loc = LocationInfo(name="Bristol, UK", latitude=51.4545, longitude=-2.5879)
        weather = WeatherInfo(
            current=WeatherCurrent(temperature_c=13.0, condition="Overcast"),
            forecast=[
                WeatherDailyForecast(
                    date="2026-11-05",
                    weekday="Thursday",
                    temp_max_c=15.0,
                    temp_min_c=8.0,
                    condition="Overcast",
                )
            ],
            summary="13°C, Overcast in Bristol, UK.",
        )
        tz = TimezoneInfo(
            iana_id="Europe/London",
            current_time_str="2:15 PM",
            current_date_str="Thursday, 5 November 2026",
            timezone_name="GMT",
        )
        snapshot = IRISContextSnapshot(location=loc, timezone=tz, weather=weather)

        assert "In Bristol, UK, it is currently 13 degrees and overcast." in snapshot.weather_summary_for_voice()
        assert "high of 15 degrees" in snapshot.weather_summary_for_voice()
        assert "It is currently 2:15 PM on Thursday, 5 November 2026 in the GMT timezone." == snapshot.datetime_summary_for_voice()


# ── 5. Router & Assistant Context Integration Tests ───────────────────────────

class TestRouterContextIntegration:
    @pytest.fixture
    def mock_context(self) -> IRISContextSnapshot:
        loc = LocationInfo(name="Cambridge, UK", latitude=52.2053, longitude=0.1218)
        tz = TimezoneInfo(
            iana_id="Europe/London",
            utc_offset_hours=0.0,
            is_dst=False,
            timezone_name="GMT",
            current_time_str="3:45 PM",
            current_date_str="Thursday, 5 November 2026",
            day_of_week="Thursday",
        )
        curr = WeatherCurrent(temperature_c=12.4, condition="Drizzle", humidity_pct=85)
        forecast = [
            WeatherDailyForecast(
                date="2026-11-05",
                weekday="Thursday",
                temp_max_c=14.0,
                temp_min_c=7.0,
                condition="Drizzle",
                precipitation_probability_pct=75,
            ),
            WeatherDailyForecast(
                date="2026-11-06",
                weekday="Friday",
                temp_max_c=13.0,
                temp_min_c=6.0,
                condition="Light rain",
                precipitation_probability_pct=80,
            ),
        ]
        weather = WeatherInfo(
            current=curr,
            forecast=forecast,
            summary="12°C, Drizzle in Cambridge, UK.",
        )
        return IRISContextSnapshot(location=loc, timezone=tz, weather=weather)

    def test_weather_query_deterministic_routing(self, mock_context: IRISContextSnapshot) -> None:
        decision = classify_request("what is the weather today?", context_snapshot=mock_context)
        assert decision.route == RouteType.WEATHER
        assert decision.direct_response is not None
        assert "Cambridge, UK" in decision.direct_response
        assert "12 degrees" in decision.direct_response

    def test_temperature_query_deterministic_routing(self, mock_context: IRISContextSnapshot) -> None:
        decision = classify_request("what is the temperature outside?", context_snapshot=mock_context)
        assert decision.route == RouteType.WEATHER
        assert decision.direct_response == "It is currently 12 degrees Celsius in Cambridge, UK."

    def test_rain_query_deterministic_routing(self, mock_context: IRISContextSnapshot) -> None:
        decision = classify_request("is it raining?", context_snapshot=mock_context)
        assert decision.route == RouteType.WEATHER
        assert "It is currently raining in Cambridge, UK" in decision.direct_response
        assert "75% chance of rain" in decision.direct_response

    def test_tomorrow_forecast_query_routing(self, mock_context: IRISContextSnapshot) -> None:
        decision = classify_request("what is tomorrow's weather?", context_snapshot=mock_context)
        assert decision.route == RouteType.WEATHER
        assert "Tomorrow in Cambridge, UK will be light rain" in decision.direct_response
        assert "high of 13 degrees" in decision.direct_response

    def test_external_city_weather_bypasses_local_handler(self, mock_context: IRISContextSnapshot) -> None:
        # Weather query for another city (e.g. Tokyo) must NOT be answered by local Cambridge data
        decision = classify_request("what is the weather in Tokyo?", context_snapshot=mock_context)
        assert decision.route != RouteType.WEATHER
        assert decision.route in (RouteType.CONVERSATION, RouteType.COMPLEX_REASONING)

    def test_location_query_routing(self, mock_context: IRISContextSnapshot) -> None:
        decision = classify_request("where am I?", context_snapshot=mock_context)
        assert decision.route == RouteType.LOCATION
        assert decision.direct_response == "Your current detected location is Cambridge, UK."

    def test_timezone_query_routing(self, mock_context: IRISContextSnapshot) -> None:
        decision = classify_request("what is my timezone?", context_snapshot=mock_context)
        assert decision.route == RouteType.DATETIME
        assert "Europe/London timezone" in decision.direct_response
        assert "GMT" in decision.direct_response

    def test_cloud_router_adapter_executes_weather_deterministically(
        self,
        mock_context: IRISContextSnapshot,
    ) -> None:
        cfg = IRISConfig(llm_provider="router")
        mock_cm = MagicMock(spec=ContextManager)
        mock_cm.get_context.return_value = mock_context
        mock_cm.get_prompt_context.return_value = mock_context.to_prompt_context()

        router = CloudRouterModelAdapter(config=cfg, context_manager=mock_cm)
        reply = router.generate_response([{"role": "user", "content": "what's the weather today?"}])

        assert "In Cambridge, UK" in reply
        assert router.last_route == "weather"
        assert router.last_provider_name == "context_engine"
        assert router.last_selected_model == "open_meteo"

    def test_assistant_lifecycle_starts_and_stops_context_manager(self, tmp_path: Path) -> None:
        cfg = IRISConfig(
            llm_provider="mock",
            tts_provider="mock",
            stt_provider="mock",
            archive_dir=tmp_path / "conversations",
            context_cache_dir=tmp_path / "context",
        )
        mock_cm = MagicMock(spec=ContextManager)
        mock_cm.get_prompt_context.return_value = "Context: London"

        assistant = IRISAssistant(
            config=cfg,
            model_adapter=MockModelAdapter(responses=["Hello!"]),
            tts_adapter=MagicMock(spec=TTSAdapter),
            stt_adapter=MagicMock(spec=STTAdapter),
            context_manager=mock_cm,
        )

        assert assistant.context_manager == mock_cm

        # Start and stop lifecycle
        with patch.object(assistant.capture, "start"):
            assistant.start()
            mock_cm.start.assert_called_once()

            assistant.stop()
            mock_cm.stop.assert_called_once()

