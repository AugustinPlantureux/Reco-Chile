"""Coordinates, distance, and address geocoding.

Used by the recommendation engine to score proximity between a candidate
program and either the family's home address or the centroid of their current
wish list.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st

from sae_app.constants import (
    COMMUNE_COORDINATES_PATH,
    GEOCODING_TIMEOUT_SECONDS,
    GEOCODING_USER_AGENT,
    NOMINATIM_MIN_INTERVAL_SECONDS,
    PROGRAM_LATITUDE,
    PROGRAM_LONGITUDE,
    REGION,
    REGION_CENTROIDS,
)
from sae_app.data_loading import first_existing_column, read_csv_path
from sae_app.i18n import t
from sae_app.program_options import ProgramRecord
from sae_app.text_utils import normalize_geo_key, parse_coordinate

_nominatim_lock = threading.Lock()
_nominatim_last_call_at = 0.0


def _throttle_nominatim() -> None:
    """Block until at least NOMINATIM_MIN_INTERVAL_SECONDS have passed since the
    last outbound call, enforced across all sessions in this process.
    """
    global _nominatim_last_call_at
    with _nominatim_lock:
        wait = _nominatim_last_call_at + NOMINATIM_MIN_INTERVAL_SECONDS - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _nominatim_last_call_at = time.monotonic()


@st.cache_data(show_spinner=False)
def load_commune_coordinates() -> pd.DataFrame:
    """Load optional commune coordinates for continuous proximity scoring.

    Expected columns in data/commune_coordinates.csv:
    - commune
    - region or Region
    - latitude/lat/latitud
    - longitude/lon/lng/longitud

    If the file is absent, the recommendation engine falls back to approximate
    regional centroids. This keeps the app runnable without adding a new data
    dependency, while making it easy to improve proximity later.
    """
    if not COMMUNE_COORDINATES_PATH.exists():
        return pd.DataFrame(columns=["commune_key", "region_key", PROGRAM_LATITUDE, PROGRAM_LONGITUDE])

    df = read_csv_path(COMMUNE_COORDINATES_PATH, sep="auto")
    commune_col = first_existing_column(df, ["commune", "comuna", "school_commune"])
    region_col = first_existing_column(df, [REGION, "region", "región"])
    lat_col = first_existing_column(df, ["latitude", "lat", "latitud"])
    lon_col = first_existing_column(df, ["longitude", "lon", "lng", "longitud"])

    if not commune_col or not lat_col or not lon_col:
        return pd.DataFrame(columns=["commune_key", "region_key", PROGRAM_LATITUDE, PROGRAM_LONGITUDE])

    out = pd.DataFrame({
        "commune_key": df[commune_col].map(normalize_geo_key),
        "region_key": df[region_col].map(normalize_geo_key) if region_col else [""] * len(df),
        PROGRAM_LATITUDE: df[lat_col].map(parse_coordinate),
        PROGRAM_LONGITUDE: df[lon_col].map(parse_coordinate),
    })
    out = out.dropna(subset=[PROGRAM_LATITUDE, PROGRAM_LONGITUDE])
    out = out.drop_duplicates(["commune_key", "region_key"])
    return out


def valid_lat_lon(lat, lon) -> bool:
    try:
        lat = float(lat)
        lon = float(lon)
    except Exception:
        return False
    return np.isfinite(lat) and np.isfinite(lon) and -56 <= lat <= -17 and -76 <= lon <= -66


@st.cache_data(show_spinner=False)
def commune_coordinate_lookup() -> dict[tuple[str, str], tuple[float, float]]:
    coords = load_commune_coordinates()
    if coords.empty:
        return {}
    return {
        (str(row["commune_key"]), str(row["region_key"])): (float(row[PROGRAM_LATITUDE]), float(row[PROGRAM_LONGITUDE]))
        for _, row in coords.iterrows()
    }


def program_coordinates(program: ProgramRecord) -> tuple[float, float, str]:
    """Return best available coordinates for one program.

    Priority:
    1. program/school coordinates if present in the program metadata file;
    2. optional commune-level coordinates from data/commune_coordinates.csv;
    3. approximate region centroid.
    """
    if valid_lat_lon(program.program_latitude, program.program_longitude):
        return float(program.program_latitude), float(program.program_longitude), "program coordinate"

    commune = normalize_geo_key(program.school_commune)
    region = normalize_geo_key(program.region)
    lookup = commune_coordinate_lookup()
    for key in [(commune, region), (commune, "")]:
        if key in lookup:
            lat, lon = lookup[key]
            return lat, lon, "commune coordinate"

    if program.region in REGION_CENTROIDS:
        lat, lon = REGION_CENTROIDS[program.region]
        return lat, lon, "region approximation"

    return np.nan, np.nan, "No reliable coordinate"


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in kilometers."""
    if not (valid_lat_lon(lat1, lon1) and valid_lat_lon(lat2, lon2)):
        return np.nan
    r = 6371.0
    phi1, phi2 = np.radians([float(lat1), float(lat2)])
    dphi = np.radians(float(lat2) - float(lat1))
    dlambda = np.radians(float(lon2) - float(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def proximity_from_distance(distance_km: float, distance_scale_km: float = 50.0) -> float:
    """Continuous 0-1 proximity score; scale is the approximate half-score distance."""
    if pd.isna(distance_km):
        return 0.0
    distance_scale_km = max(float(distance_scale_km), 1.0)
    return float(1.0 / (1.0 + max(float(distance_km), 0.0) / distance_scale_km))


def normalize_address_for_geocoding(address: str) -> str:
    """Return a cleaned address query, restricted to Chile when no country is provided."""
    text = " ".join(str(address or "").strip().split())
    if not text:
        return ""
    if "chile" not in normalize_geo_key(text):
        text = f"{text}, Chile"
    return text


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def geocode_chilean_address(address: str) -> dict:
    """Geocode a family-entered address using OpenStreetMap/Nominatim.

    The app does not store the address permanently; Streamlit only keeps the
    returned coordinates in session/cache to avoid repeated calls on reruns.
    """
    original_address = " ".join(str(address or "").strip().split())
    query = normalize_address_for_geocoding(original_address)
    if not query:
        return {"ok": False, "address": original_address, "error": t("No address entered.")}

    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
        "countrycodes": "cl",
    })
    url = f"https://nominatim.openstreetmap.org/search?{params}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": GEOCODING_USER_AGENT,
            "Accept": "application/json",
        },
    )

    _throttle_nominatim()

    try:
        with urllib.request.urlopen(request, timeout=GEOCODING_TIMEOUT_SECONDS) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                return {
                    "ok": False,
                    "address": original_address,
                    "error": t("Geocoding service returned status {status}.", status=status),
                }
            payload = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        return {
            "ok": False,
            "address": original_address,
            "error": t("Could not reach the geocoding service: {error}", error=str(exc)),
        }
    except Exception as exc:
        return {
            "ok": False,
            "address": original_address,
            "error": t("Could not reach the geocoding service: {error}", error=str(exc)),
        }

    try:
        results = json.loads(payload)
    except Exception as exc:
        return {
            "ok": False,
            "address": original_address,
            "error": t("Could not read the geocoding response: {error}", error=str(exc)),
        }

    if not results:
        return {"ok": False, "address": original_address, "error": t("No result found for this address in Chile.")}

    best = results[0]
    lat = parse_coordinate(best.get("lat"))
    lon = parse_coordinate(best.get("lon"))
    if not valid_lat_lon(lat, lon):
        return {
            "ok": False,
            "address": original_address,
            "error": t("The geocoded result is outside Chile or has invalid coordinates."),
        }

    return {
        "ok": True,
        "address": original_address,
        "query": query,
        "lat": float(lat),
        "lon": float(lon),
        "display_name": str(best.get("display_name", query)),
    }
