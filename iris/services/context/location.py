"""
iris/services/context/location.py
----------------------------------
Physical device location service for IRIS.

Prioritizes native macOS CoreLocation via lightweight ctypes Objective-C bindings,
with robust fallbacks to persistent disk cache, manual user configuration,
and timezone-based heuristics.

Privacy Guarantee:
Exact latitude and longitude coordinates are NEVER logged. Only city/region names
and detection sources are recorded in application logs.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from iris.services.context.models import LocationInfo

logger = logging.getLogger(__name__)

# macOS CoreLocation authorization statuses (CLAuthorizationStatus)
CL_AUTH_NOT_DETERMINED = 0
CL_AUTH_RESTRICTED = 1
CL_AUTH_DENIED = 2
CL_AUTH_AUTHORIZED_ALWAYS = 3
CL_AUTH_AUTHORIZED_WHEN_IN_USE = 4

# Heuristic coordinates for common primary timezones if offline without GPS or config
_TIMEZONE_CITY_DEFAULTS: dict[str, tuple[str, float, float]] = {
    "Europe/London": ("London, UK", 51.5074, -0.1278),
    "Europe/Berlin": ("Berlin, Germany", 52.5200, 13.4050),
    "Europe/Paris": ("Paris, France", 48.8566, 2.3522),
    "America/New_York": ("New York, NY", 40.7128, -74.0060),
    "America/Chicago": ("Chicago, IL", 41.8781, -87.6298),
    "America/Denver": ("Denver, CO", 39.7392, -104.9903),
    "America/Los_Angeles": ("Los Angeles, CA", 34.0522, -118.2437),
    "America/Toronto": ("Toronto, Canada", 43.6532, -79.3832),
    "Asia/Tokyo": ("Tokyo, Japan", 35.6762, 139.6503),
    "Asia/Kolkata": ("Kolkata, India", 22.5726, 88.3639),
    "Australia/Sydney": ("Sydney, Australia", -33.8688, 151.2093),
}


class CLLocationCoordinate2D(ctypes.Structure):
    _fields_ = [
        ("latitude", ctypes.c_double),
        ("longitude", ctypes.c_double),
    ]


class LocationService:
    """
    Detects physical device location via macOS CoreLocation, with
    graceful fallback to local cache, manual configuration, and timezone defaults.
    """

    def __init__(
        self,
        cache_dir: Path,
        enable_tracking: bool = True,
        fallback_name: str = "",
        fallback_latitude: float | None = None,
        fallback_longitude: float | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "location_cache.json"
        self.enable_tracking = enable_tracking
        self.fallback_name = fallback_name.strip()
        self.fallback_latitude = fallback_latitude
        self.fallback_longitude = fallback_longitude
        self._cached_location: LocationInfo | None = None

        # Pre-load cache if available
        self._cached_location = self._load_cache()

    def get_location(self, force_refresh: bool = False) -> LocationInfo:
        """
        Determine device location using the fallback hierarchy:
        1. macOS CoreLocation (if tracking enabled on macOS)
        2. Local disk cache (location_cache.json)
        3. Manual configuration fallback (IRISConfig)
        4. Timezone-based heuristic fallback
        5. Generic 'Local Device' fallback
        """
        if self._cached_location and not force_refresh and self._cached_location.latitude is not None:
            return self._cached_location

        if self.enable_tracking and sys.platform == "darwin":
            detected = self._try_macos_corelocation()
            if detected is not None:
                self._save_cache(detected)
                self._cached_location = detected
                return detected

        # Fallback 1: Local persistent cache from prior runs
        cached = self._load_cache()
        if cached is not None and cached.latitude is not None:
            logger.info("Using cached location: '%s' (source: %s)", cached.name, cached.source)
            self._cached_location = cached
            return cached

        # Fallback 2: Manual configuration
        if self.fallback_name or (self.fallback_latitude is not None and self.fallback_longitude is not None):
            logger.info("Using configured manual fallback location: '%s'", self.fallback_name or "Custom Coordinates")
            manual = LocationInfo(
                name=self.fallback_name or "Manual Location",
                latitude=self.fallback_latitude,
                longitude=self.fallback_longitude,
                source="manual_fallback",
                is_fallback=True,
            )
            self._save_cache(manual)
            self._cached_location = manual
            return manual

        # Fallback 3: Timezone-based heuristic
        tz_fallback = self._heuristic_from_timezone()
        if tz_fallback is not None:
            logger.info("Using timezone-derived location: '%s'", tz_fallback.name)
            self._save_cache(tz_fallback)
            self._cached_location = tz_fallback
            return tz_fallback

        # Fallback 4: Generic unknown device
        logger.info("No location available; defaulting to generic Local Device")
        generic = LocationInfo(
            name="Local Device",
            latitude=None,
            longitude=None,
            source="unknown",
            is_fallback=True,
        )
        self._cached_location = generic
        return generic

    def _try_macos_corelocation(self) -> LocationInfo | None:
        """
        Attempt to query location via macOS CoreLocation using ctypes.
        Returns LocationInfo on success, or None on failure or missing permissions.
        """
        try:
            objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
            cl_lib = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreLocation.framework/CoreLocation")
        except Exception as exc:
            logger.debug("CoreLocation ctypes load skipped: %s", exc)
            return None

        try:
            objc.objc_getClass.restype = ctypes.c_void_p
            objc.objc_getClass.argtypes = [ctypes.c_char_p]
            objc.sel_registerName.restype = ctypes.c_void_p
            objc.sel_registerName.argtypes = [ctypes.c_char_p]

            cls_mgr = objc.objc_getClass(b"CLLocationManager")
            if not cls_mgr:
                return None

            sel_alloc = objc.sel_registerName(b"alloc")
            sel_init = objc.sel_registerName(b"init")
            sel_services_enabled = objc.sel_registerName(b"locationServicesEnabled")
            sel_auth_status = objc.sel_registerName(b"authorizationStatus")
            sel_location = objc.sel_registerName(b"location")
            sel_start_updates = objc.sel_registerName(b"startUpdatingLocation")
            sel_stop_updates = objc.sel_registerName(b"stopUpdatingLocation")

            # Check if global location services are enabled
            objc.objc_msgSend.restype = ctypes.c_bool
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            services_enabled = objc.objc_msgSend(cls_mgr, sel_services_enabled)
            if not services_enabled:
                logger.info(
                    "macOS Location Services are disabled globally. "
                    "Enable via System Settings -> Privacy & Security -> Location Services."
                )
                return None

            # Allocate and initialize CLLocationManager
            objc.objc_msgSend.restype = ctypes.c_void_p
            mgr = objc.objc_msgSend(cls_mgr, sel_alloc)
            mgr = objc.objc_msgSend(mgr, sel_init)
            if not mgr:
                return None

            # Check authorization status
            objc.objc_msgSend.restype = ctypes.c_int
            status = objc.objc_msgSend(mgr, sel_auth_status)

            if status == CL_AUTH_NOT_DETERMINED:
                logger.info(
                    "macOS Location Services authorization status is NotDetermined (0). "
                    "To enable automatic location detection on macOS, ensure Location Services "
                    "is allowed under System Settings -> Privacy & Security -> Location Services "
                    "for your Terminal / Python process."
                )
                return None
            elif status in (CL_AUTH_RESTRICTED, CL_AUTH_DENIED):
                logger.info(
                    "macOS Location Services authorization is restricted or denied (status=%d). "
                    "Using configured fallback or cached location.",
                    status,
                )
                return None
            elif status not in (CL_AUTH_AUTHORIZED_ALWAYS, CL_AUTH_AUTHORIZED_WHEN_IN_USE):
                logger.debug("Unknown macOS CoreLocation auth status: %d", status)
                return None

            # Authorized! Request location update
            objc.objc_msgSend.restype = ctypes.c_void_p
            objc.objc_msgSend(mgr, sel_start_updates)

            # Check location up to a brief timeout
            loc_ptr = None
            for _ in range(5):
                loc_ptr = objc.objc_msgSend(mgr, sel_location)
                if loc_ptr:
                    break
                time.sleep(0.1)

            objc.objc_msgSend(mgr, sel_stop_updates)

            if not loc_ptr:
                logger.debug("CoreLocation returned null location pointer")
                return None

            # Extract coordinates
            # coordinate method returns CLLocationCoordinate2D struct
            # On arm64 and x86_64, msgSend_stret or direct msgSend is architecture-dependent.
            # Using ctypes prototype for objc_msgSend with struct return:
            msg_send_coord = ctypes.cast(
                objc.objc_msgSend,
                ctypes.CFUNCTYPE(CLLocationCoordinate2D, ctypes.c_void_p, ctypes.c_void_p),
            )
            sel_coordinate = objc.sel_registerName(b"coordinate")
            coord = msg_send_coord(loc_ptr, sel_coordinate)

            lat = float(coord.latitude)
            lon = float(coord.longitude)

            # Query horizontal accuracy
            objc.objc_msgSend.restype = ctypes.c_double
            sel_horiz_acc = objc.sel_registerName(b"horizontalAccuracy")
            accuracy = float(objc.objc_msgSend(loc_ptr, sel_horiz_acc))

            # Validate coordinate bounds
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0) or (lat == 0.0 and lon == 0.0):
                return None

            city_name = self.fallback_name or self._guess_city_name(lat, lon) or "Local Device"
            logger.info("macOS CoreLocation acquired successfully for '%s' (accuracy: %.1fm)", city_name, accuracy)

            return LocationInfo(
                name=city_name,
                latitude=lat,
                longitude=lon,
                accuracy_m=accuracy if accuracy >= 0 else None,
                source="macos_corelocation",
                is_fallback=False,
            )
        except Exception as exc:
            logger.debug("Error querying macOS CoreLocation: %s", exc)
            return None

    def _guess_city_name(self, lat: float, lon: float) -> str:
        """Lightweight offline matching to major city centers."""
        best_name = ""
        best_dist = float("inf")
        for city, (_, clat, clon) in _TIMEZONE_CITY_DEFAULTS.items():
            dist = (lat - clat) ** 2 + (lon - clon) ** 2
            if dist < best_dist:
                best_dist = dist
                best_name = city

        # If within ~1 degree (~110km) of a known city center
        if best_dist < 1.0:
            return best_name
        return "Local Device"

    def _heuristic_from_timezone(self) -> LocationInfo | None:
        """Derive approximate city coordinates from the system timezone link."""
        try:
            if sys.platform in ("darwin", "linux"):
                target = os.readlink("/etc/localtime")
                if "zoneinfo/" in target:
                    tz_key = target.split("zoneinfo/")[-1]
                    if tz_key in _TIMEZONE_CITY_DEFAULTS:
                        name, lat, lon = _TIMEZONE_CITY_DEFAULTS[tz_key]
                        return LocationInfo(
                            name=name,
                            latitude=lat,
                            longitude=lon,
                            source="default_fallback",
                            is_fallback=True,
                        )
        except Exception as exc:
            logger.debug("Timezone heuristic lookup skipped: %s", exc)
        return None

    def _load_cache(self) -> LocationInfo | None:
        """Read cached location from disk."""
        if not self.cache_file.exists():
            return None
        try:
            content = self.cache_file.read_text(encoding="utf-8")
            data = json.loads(content)
            loc = LocationInfo.from_dict(data)
            loc.source = "cached"
            return loc
        except Exception as exc:
            logger.warning("Failed to parse location cache from %s: %s", self.cache_file, exc)
            return None

    def _save_cache(self, location: LocationInfo) -> None:
        """Persist location to disk cache."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            payload = location.to_dict(include_coordinates=True)
            self.cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.debug("Saved location cache for '%s'", location.name)
        except Exception as exc:
            logger.warning("Failed to write location cache to %s: %s", self.cache_file, exc)
