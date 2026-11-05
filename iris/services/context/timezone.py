"""
iris/services/context/timezone.py
----------------------------------
Timezone and date/time service for IRIS.

Accurately tracks:
- Canonical IANA timezone identifiers (e.g., 'Europe/London', 'America/New_York')
- UTC offsets and Daylight Saving Time (DST) transitions via standard library zoneinfo
- Formatted local time, date, day of week, and ISO timestamps
"""

from __future__ import annotations

import logging
import os
import sys
import zoneinfo
from datetime import datetime
from typing import Any

from iris.services.context.models import LocationInfo, TimezoneInfo

logger = logging.getLogger(__name__)

# Common geographical location-to-timezone mappings for location fallback
_LOCATION_TZ_HINTS: dict[str, str] = {
    "london": "Europe/London",
    "oxford": "Europe/London",
    "cambridge": "Europe/London",
    "uk": "Europe/London",
    "united kingdom": "Europe/London",
    "edinburgh": "Europe/London",
    "manchester": "Europe/London",
    "birmingham": "Europe/London",
    "bristol": "Europe/London",
    "berlin": "Europe/Berlin",
    "paris": "Europe/Paris",
    "madrid": "Europe/Madrid",
    "rome": "Europe/Rome",
    "new york": "America/New_York",
    "boston": "America/New_York",
    "san francisco": "America/Los_Angeles",
    "cupertino": "America/Los_Angeles",
    "los angeles": "America/Los_Angeles",
    "seattle": "America/Los_Angeles",
    "chicago": "America/Chicago",
    "denver": "America/Denver",
    "toronto": "America/Toronto",
    "tokyo": "Asia/Tokyo",
    "kolkata": "Asia/Kolkata",
    "mumbai": "Asia/Kolkata",
    "delhi": "Asia/Kolkata",
    "singapore": "Asia/Singapore",
    "sydney": "Australia/Sydney",
}


class TimezoneService:
    """
    Computes local timezone properties, daylight saving time (DST) status,
    and formatted date/time strings using Python standard library zoneinfo.
    """

    def __init__(self, default_iana: str = "UTC") -> None:
        self.default_iana = default_iana

    def get_timezone_info(
        self,
        iana_override: str | None = None,
        location: LocationInfo | None = None,
        reference_time: datetime | None = None,
    ) -> TimezoneInfo:
        """
        Build and return a TimezoneInfo object.

        Args:
            iana_override: Specific IANA string (e.g. returned by weather API or config).
            location: Current LocationInfo if available.
            reference_time: Optional datetime to evaluate against (useful for tests/DST checks).
        """
        iana_id = self._resolve_iana_id(iana_override, location=location)

        try:
            tz = zoneinfo.ZoneInfo(iana_id)
        except Exception as exc:
            logger.warning("Invalid timezone '%s' (%s); falling back to UTC", iana_id, exc)
            tz = zoneinfo.ZoneInfo("UTC")
            iana_id = "UTC"

        now = reference_time or datetime.now(tz)
        if now.tzinfo is None:
            now = now.replace(tzinfo=tz)
        else:
            now = now.astimezone(tz)

        offset = now.utcoffset()
        offset_hours = (offset.total_seconds() / 3600.0) if offset else 0.0

        dst_delta = now.dst()
        is_dst = bool(dst_delta and dst_delta.total_seconds() != 0)

        tz_name = now.tzname() or iana_id

        # Formatted natural English representations
        time_str = now.strftime("%I:%M %p").lstrip("0")
        day_num = now.strftime("%d").lstrip("0")
        month_name = now.strftime("%B")
        weekday = now.strftime("%A")
        year = now.strftime("%Y")
        date_str = f"{weekday}, {day_num} {month_name} {year}"

        return TimezoneInfo(
            iana_id=iana_id,
            utc_offset_hours=offset_hours,
            is_dst=is_dst,
            timezone_name=tz_name,
            current_time_str=time_str,
            current_date_str=date_str,
            day_of_week=weekday,
            iso_timestamp=now.isoformat(),
        )

    def get_datetime_now(self, iana_id: str | None = None) -> datetime:
        """Return the current timezone-aware datetime object."""
        target_iana = self._resolve_iana_id(iana_id)
        try:
            tz = zoneinfo.ZoneInfo(target_iana)
        except Exception:
            tz = zoneinfo.ZoneInfo("UTC")
        return datetime.now(tz)

    def _resolve_iana_id(
        self,
        override: str | None = None,
        location: LocationInfo | None = None,
    ) -> str:
        """Resolve canonical IANA timezone identifier from override, location hints, OS, or UTC."""
        if override and override.strip():
            cand = override.strip()
            try:
                zoneinfo.ZoneInfo(cand)
                return cand
            except Exception:
                logger.debug("Provided IANA override '%s' is not valid in zoneinfo", cand)
                return self.default_iana or "UTC"

        # Check location-derived geographical timezone hints
        if location and location.name:
            loc_lower = location.name.lower()
            for key, tz_hint in _LOCATION_TZ_HINTS.items():
                if key in loc_lower:
                    return tz_hint

        # macOS / Linux /etc/localtime symlink inspection
        if sys.platform in ("darwin", "linux"):
            try:
                target = os.readlink("/etc/localtime")
                if "zoneinfo/" in target:
                    cand = target.split("zoneinfo/")[-1]
                    try:
                        zoneinfo.ZoneInfo(cand)
                        return cand
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("Failed reading /etc/localtime symlink: %s", exc)

        # Local datetime.astimezone() inspection
        try:
            local_tz = datetime.now().astimezone().tzinfo
            if local_tz is not None and hasattr(local_tz, "key") and local_tz.key:
                return str(local_tz.key)
        except Exception as exc:
            logger.debug("Failed deriving tzinfo from astimezone: %s", exc)

        return self.default_iana or "UTC"
