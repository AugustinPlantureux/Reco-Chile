"""Coordinates, distance, and address geocoding.

Used by the recommendation engine to score proximity between a candidate
program and either the family's home address or the centroid of their current
wish list.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st

from sae_app.constants import (
    CHILE_COORDINATE_ZONES,
    COMMUNE_COORDINATES_PATH,
    GEOCODING_TIMEOUT_SECONDS,
    GEOCODING_USER_AGENT,
    NOMINATIM_MIN_INTERVAL_SECONDS,
    PROGRAM_LATITUDE,
    PROGRAM_LONGITUDE,
    REGION,
    REGION_CENTROIDS,
)
from sae_app.data_loading import first_existing_column, read_csv
from sae_app.program_options import ProgramRecord
from sae_app.text_utils import normalize_geo_key, parse_coordinate


@st.cache_data(show_spinner=False)
def _load_commune_coordinates(file_bytes: bytes) -> pd.DataFrame:
    """Parse optional commune coordinates using one coherent coordinate pair."""
    if not file_bytes.strip():
        return pd.DataFrame(
            columns=["commune_key", "region_key", PROGRAM_LATITUDE, PROGRAM_LONGITUDE]
        )
    df = read_csv(file_bytes, sep="auto")
    commune_col = first_existing_column(df, ["commune", "comuna", "school_commune"])
    region_col = first_existing_column(df, [REGION, "region", "región"])

    coordinate_pairs = [
        ("latitude", "longitude"),
        ("lat", "lon"),
        ("lat", "lng"),
        ("latitud", "longitud"),
    ]
    lat_col, lon_col = next(
        ((lat, lon) for lat, lon in coordinate_pairs if lat in df.columns and lon in df.columns),
        (None, None),
    )

    if not commune_col or not lat_col or not lon_col:
        return pd.DataFrame(
            columns=["commune_key", "region_key", PROGRAM_LATITUDE, PROGRAM_LONGITUDE]
        )

    out = pd.DataFrame({
        "commune_key": df[commune_col].map(normalize_geo_key),
        "region_key": df[region_col].map(normalize_geo_key) if region_col else [""] * len(df),
        PROGRAM_LATITUDE: df[lat_col].map(parse_coordinate),
        PROGRAM_LONGITUDE: df[lon_col].map(parse_coordinate),
    })
    valid_mask = [valid_lat_lon(lat, lon) for lat, lon in zip(
        out[PROGRAM_LATITUDE], out[PROGRAM_LONGITUDE]
    )]
    out = out.loc[valid_mask].copy()
    out = out.drop_duplicates(["commune_key", "region_key"])
    return out


# The former mainland-only check rejected Rapa Nui and Juan Fernández.
# The shared zones in constants.py keep ingestion and distance validation aligned.
def _inside_coordinate_zone(
    lat: float,
    lon: float,
    zone: tuple[str, float, float, float, float],
) -> bool:
    """Return whether a coordinate lies inside one configured Chile zone."""
    _, min_lat, max_lat, min_lon, max_lon = zone
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def valid_lat_lon(lat, lon) -> bool:
    """Return True for finite coordinates in mainland or insular Chile."""
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return False

    if not (np.isfinite(lat) and np.isfinite(lon)):
        return False

    return any(_inside_coordinate_zone(lat, lon, zone) for zone in CHILE_COORDINATE_ZONES)


def commune_coordinate_lookup() -> dict[tuple[str, str], tuple[float, float]]:
    """Return the commune lookup, caching by the coordinate file's contents."""
    if not COMMUNE_COORDINATES_PATH.exists():
        return {}
    file_bytes = COMMUNE_COORDINATES_PATH.read_bytes()
    if not file_bytes.strip():
        return {}
    return _commune_coordinate_lookup(file_bytes)


@st.cache_data(show_spinner=False)
def _commune_coordinate_lookup(
    file_bytes: bytes,
) -> dict[tuple[str, str], tuple[float, float]]:
    coords = _load_commune_coordinates(file_bytes)
    if coords.empty:
        return {}
    return {
        (str(row["commune_key"]), str(row["region_key"])): (
            float(row[PROGRAM_LATITUDE]),
            float(row[PROGRAM_LONGITUDE]),
        )
        for _, row in coords.iterrows()
    }


STRONG_PROGRAM_GEO_MATCH_LEVELS = frozenset({
    "exact_rbd_program_code",
    "exact_rbd",
    "exact",
})
MAX_MATCHED_COORDINATE_DISCREPANCY_KM = 5.0
MAX_RBD_SCHOOL_COORDINATE_SPREAD_KM = 10.0
RELIABLE_HOME_GEOCODING_PRECISIONS = frozenset({"address", "street"})


def program_coordinate_source_is_reliable(
    source: str,
    program: ProgramRecord | None = None,
) -> bool:
    """Return whether a coordinate is precise enough for hard exclusion.

    Generic, commune and regional coordinates remain useful for continuous
    proximity scoring, but never remove a program from the family-facing list.
    """
    source = str(source or "").strip()
    if program is None:
        return False

    if source == "school coordinate":
        spread = program.rbd_coordinate_spread_km
        return (
            np.isfinite(spread)
            and float(spread) <= MAX_RBD_SCHOOL_COORDINATE_SPREAD_KM
        )

    if source == "matched program coordinate":
        discrepancy = program.coordinate_discrepancy_km
        match_level = str(program.program_geo_match_level or "").strip().lower()
        spread = program.rbd_coordinate_spread_km
        spread_is_acceptable = (
            not np.isfinite(spread)
            or float(spread) <= MAX_RBD_SCHOOL_COORDINATE_SPREAD_KM
        )
        return (
            np.isfinite(discrepancy)
            and float(discrepancy) <= MAX_MATCHED_COORDINATE_DISCREPANCY_KM
            and match_level in STRONG_PROGRAM_GEO_MATCH_LEVELS
            and spread_is_acceptable
        )

    return False


def program_coordinate_reference_priority(source: str) -> int:
    """Rank coordinate sources for building the wish-list geographic reference."""
    return {
        "school coordinate": 4,
        "matched program coordinate": 3,
        "commune coordinate": 2,
        "generic coordinate": 1,
        "region approximation": 0,
    }.get(str(source or "").strip(), 0)


def home_geocoding_supports_hard_filter(home_geo_reference: dict | None) -> bool:
    precision = str((home_geo_reference or {}).get("precision", "")).strip().lower()
    return precision in RELIABLE_HOME_GEOCODING_PRECISIONS


def home_distance_filter_is_reliable(
    home_geo_reference: dict | None,
    program_coordinate_source: str,
    program: ProgramRecord | None = None,
) -> bool:
    return (
        home_geocoding_supports_hard_filter(home_geo_reference)
        and program_coordinate_source_is_reliable(
            program_coordinate_source,
            program,
        )
    )


def program_coordinates(
    program: ProgramRecord,
    commune_lookup: dict[tuple[str, str], tuple[float, float]] | None = None,
) -> tuple[float, float, str]:
    """Return best available coordinates for one program.

    Priority:
    1. program/school coordinates if present in the program metadata file;
    2. optional commune-level coordinates from data/commune_coordinates.csv;
    3. approximate region centroid.
    """
    if valid_lat_lon(program.program_latitude, program.program_longitude):
        source = str(program.program_coordinate_source or "generic coordinate").strip()
        return float(program.program_latitude), float(program.program_longitude), source

    commune = normalize_geo_key(program.school_commune)
    region = normalize_geo_key(program.region)
    lookup = commune_lookup if commune_lookup is not None else commune_coordinate_lookup()
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


GEOCODING_RESULT_LIMIT = 10

ADDRESS_LEVEL_ADDRESSTYPES = {
    "house",
    "building",
    "amenity",
    "shop",
    "office",
    "tourism",
    "leisure",
}

STREET_LEVEL_ADDRESSTYPES = {
    "road",
    "residential",
    "pedestrian",
    "footway",
    "cycleway",
    "path",
}

ROAD_ADDRESS_KEYS = (
    "road",
    "pedestrian",
    "footway",
    "cycleway",
    "path",
    "residential",
)

CITY_ADDRESS_KEYS = (
    "city",
    "town",
    "village",
    "municipality",
    "county",
    "state_district",
)

# Nominatim candidate-scoring weights. These stay named and local to geo.py
# because they only tune the internal choice of the best geocoding candidate.
HOUSE_NUMBER_MATCH_BONUS = 80.0
HOUSE_NUMBER_MISMATCH_PENALTY = -60.0
HOUSE_NUMBER_MISSING_PENALTY = -20.0
ROAD_MATCH_WEIGHT = 35.0
CITY_MATCH_WEIGHT = 20.0
ADDRESS_LEVEL_TYPE_BONUS = 12.0
STREET_LEVEL_TYPE_BONUS = 5.0
ROAD_MATCH_PRECISION_THRESHOLD = 0.5
ADDRESS_PRECISION_BONUS = 30.0
STREET_PRECISION_BONUS = 10.0
CITY_PRECISION_PENALTY = -10.0
APPROXIMATE_PRECISION_PENALTY = -20.0

NUMERIC_STREET_CONTINUATIONS = {
    "de",
    "del",
    "la",
    "las",
    "los",
    "norte",
    "sur",
    "oriente",
    "poniente",
    "este",
    "oeste",
}

ROAD_TYPE_WORDS = {
    "avenida",
    "av",
    "avda",
    "calle",
    "pasaje",
    "pje",
    "camino",
    "ruta",
    "sector",
}

UNIT_WORDS = {
    "depto",
    "departamento",
    "dpto",
    "piso",
    "oficina",
    "torre",
    "block",
    "bloque",
}

_nominatim_lock = threading.Lock()
_nominatim_last_call_at = 0.0


def _throttle_nominatim() -> None:
    """Respect Nominatim's public-service rate limit within this Python process."""
    global _nominatim_last_call_at
    with _nominatim_lock:
        now = time.monotonic()
        elapsed = now - _nominatim_last_call_at
        wait = max(0.0, NOMINATIM_MIN_INTERVAL_SECONDS - elapsed)
        if wait > 0:
            time.sleep(wait)
        _nominatim_last_call_at = time.monotonic()


def _clean_house_number(value: str) -> str:
    """Normalize a house number for comparison."""
    return re.sub(r"[^0-9A-Z]", "", str(value or "").upper())


def _extract_requested_house_number(address: str) -> str:
    """
    Extract a likely house number from the first address segment.

    Conservative examples:
    - "1020 Avenida Errázuriz, Valparaíso" -> "1020"
    - "Avenida Errázuriz 1020, Valparaíso" -> "1020"
    - "21 de Mayo 1020, Valparaíso" -> "1020"
    - "21 de Mayo, Valparaíso" -> ""
    - "5 Norte, Viña del Mar" -> ""
    """
    first_part = str(address or "").split(",", 1)[0].strip()
    if not first_part:
        return ""

    matches = list(re.finditer(r"\b\d{1,6}[A-Za-z]?\b", first_part))
    if not matches:
        return ""

    usable_matches = []

    for match in matches:
        before = first_part[:match.start()].strip()
        after = first_part[match.end():].strip()

        before_tokens = normalize_geo_key(before).split()
        after_tokens = normalize_geo_key(after).split()

        previous_token = before_tokens[-1] if before_tokens else ""
        next_token = after_tokens[0] if after_tokens else ""

        # Apartment/unit numbers should not replace the house number:
        # "Av Errázuriz 1020 depto 5" should keep 1020, not 5.
        if previous_token in UNIT_WORDS:
            continue

        # A single number followed by "de", "norte", "oriente", etc. is often
        # part of the street name: "21 de Mayo", "5 Norte", "7 Oriente".
        if len(matches) == 1 and next_token in NUMERIC_STREET_CONTINUATIONS:
            return ""

        # "Calle 5" / "Pasaje 7" can be a street name rather than a house number.
        # Be conservative when the only number is immediately after a road-type word.
        if len(matches) == 1 and previous_token in ROAD_TYPE_WORDS and not after_tokens:
            return ""

        usable_matches.append(match)

    if not usable_matches:
        return ""

    # When several numbers appear, the last usable one is usually the house
    # number: "21 de Mayo 1020", "Avenida 5 Norte 1020".
    if len(usable_matches) >= 2:
        return _clean_house_number(usable_matches[-1].group(0))

    only = usable_matches[0]
    before = first_part[:only.start()].strip()
    after = first_part[only.end():].strip()

    before_tokens = normalize_geo_key(before).split()
    after_tokens = normalize_geo_key(after).split()
    next_token = after_tokens[0] if after_tokens else ""

    # Leading-number form: "1020 Avenida Errázuriz" or "1020 Errázuriz".
    # Reject only the common numeric-street pattern, such as "21 de Mayo".
    if only.start() == 0:
        if next_token in NUMERIC_STREET_CONTINUATIONS:
            return ""
        return _clean_house_number(only.group(0))

    # Trailing-number form: "Avenida Errázuriz 1020".
    if not after:
        return _clean_house_number(only.group(0))

    # Trailing house number followed by unit details: "Errázuriz 1020 depto 5".
    if after_tokens and after_tokens[0] in UNIT_WORDS:
        return _clean_house_number(only.group(0))

    return ""


def _expected_street_key(address: str, requested_house_number: str) -> str:
    """Return the likely street name, with the requested house number removed."""
    first_part = str(address or "").split(",", 1)[0]

    if requested_house_number:
        first_part = re.sub(
            r"\b\d{1,6}[A-Za-z]?\b",
            lambda match: " " if _clean_house_number(match.group(0)) == requested_house_number else match.group(0),
            first_part,
            flags=re.IGNORECASE,
        )

    return normalize_geo_key(first_part)


def _nominatim_address(result: dict) -> dict:
    """Safely return Nominatim's address-details dictionary."""
    address = result.get("address", {})
    return address if isinstance(address, dict) else {}


def _first_address_value(address: dict, keys: tuple[str, ...]) -> str:
    """Return the first non-empty address field among several possible keys."""
    for key in keys:
        value = str(address.get(key, "")).strip()
        if value:
            return value
    return ""


def _token_match_strength(expected: str, candidate: str) -> float:
    """
    Simple textual match score between 0 and 1.

    This avoids an extra dependency and handles cases like "Avenida Errázuriz"
    vs "Av. Errázuriz" or display names containing the road.
    """
    expected_key = normalize_geo_key(expected)
    candidate_key = normalize_geo_key(candidate)

    if not expected_key or not candidate_key:
        return 0.0

    if expected_key == candidate_key:
        return 1.0

    if expected_key in candidate_key or candidate_key in expected_key:
        return 0.9

    expected_tokens = set(expected_key.split())
    candidate_tokens = set(candidate_key.split())

    if not expected_tokens or not candidate_tokens:
        return 0.0

    return len(expected_tokens & candidate_tokens) / max(
        len(expected_tokens),
        len(candidate_tokens),
    )


def _score_nominatim_result(result: dict, original_address: str) -> dict:
    """
    Score one Nominatim result.

    The score prioritizes matching house number, street, city/municipality, and
    address-level results. Precision is intentionally conservative: the UI should
    warn families whenever the location is only street/city/approximate-level.
    """
    address = _nominatim_address(result)
    query_key = normalize_geo_key(original_address)

    requested_house_number = _extract_requested_house_number(original_address)
    returned_house_number = _clean_house_number(address.get("house_number", ""))

    expected_street = _expected_street_key(original_address, requested_house_number)
    returned_road = _first_address_value(address, ROAD_ADDRESS_KEYS)
    display_name = str(result.get("display_name", "")).strip()

    addresstype = normalize_geo_key(result.get("addresstype", ""))
    result_type = normalize_geo_key(result.get("type", ""))
    result_class = normalize_geo_key(result.get("class", result.get("category", "")))

    score = 0.0

    house_number_confirmed = bool(
        requested_house_number and returned_house_number == requested_house_number
    )

    if requested_house_number:
        if house_number_confirmed:
            score += HOUSE_NUMBER_MATCH_BONUS
        elif returned_house_number:
            # A different returned house number is worse than a street-level result.
            score += HOUSE_NUMBER_MISMATCH_PENALTY
        else:
            # Street-level result: usable, but not exact.
            score += HOUSE_NUMBER_MISSING_PENALTY

    road_match = max(
        _token_match_strength(expected_street, returned_road),
        _token_match_strength(expected_street, display_name),
    )
    score += ROAD_MATCH_WEIGHT * road_match

    city_match = 0.0
    for key in CITY_ADDRESS_KEYS:
        value = normalize_geo_key(address.get(key, ""))
        if value and value in query_key:
            city_match = 1.0
            break

    score += CITY_MATCH_WEIGHT * city_match

    is_address_level = (
        addresstype in ADDRESS_LEVEL_ADDRESSTYPES
        or result_type in ADDRESS_LEVEL_ADDRESSTYPES
        or result_class in ADDRESS_LEVEL_ADDRESSTYPES
    )
    is_street_level = addresstype in STREET_LEVEL_ADDRESSTYPES or result_type in STREET_LEVEL_ADDRESSTYPES

    if is_address_level:
        score += ADDRESS_LEVEL_TYPE_BONUS
    elif is_street_level:
        score += STREET_LEVEL_TYPE_BONUS

    if requested_house_number and house_number_confirmed and road_match >= ROAD_MATCH_PRECISION_THRESHOLD:
        precision = "address"
    elif not requested_house_number and returned_house_number and road_match >= ROAD_MATCH_PRECISION_THRESHOLD:
        precision = "address"
    elif road_match >= ROAD_MATCH_PRECISION_THRESHOLD or is_street_level:
        precision = "street"
    elif city_match:
        precision = "city"
    else:
        precision = "approximate"

    if precision == "address":
        score += ADDRESS_PRECISION_BONUS
    elif precision == "street":
        score += STREET_PRECISION_BONUS
    elif precision == "city":
        score += CITY_PRECISION_PENALTY
    else:
        score += APPROXIMATE_PRECISION_PENALTY

    return {
        "score": score,
        "precision": precision,
        "house_number_requested": requested_house_number,
        "house_number_found": returned_house_number,
        "house_number_confirmed": house_number_confirmed,
        "road_match": road_match,
        "city_match": city_match,
    }


def geocoding_precision_warning_key(geo_result: dict) -> str:
    """Return the untranslated warning key for a non-exact geocoding result."""
    precision = str((geo_result or {}).get("precision", "approximate"))
    requested_house_number = str((geo_result or {}).get("house_number_requested", "")).strip()

    if precision == "address":
        return ""

    if requested_house_number and precision == "street":
        return (
            "The geocoder found the street, but could not confirm the exact street number. "
            "Distances are computed from an approximate street-level location."
        )

    if precision == "street":
        return (
            "The geocoder found the street, but not an exact address point. "
            "Distances are computed from an approximate street-level location."
        )

    if precision == "city":
        return (
            "The geocoder could only identify the city or municipality. "
            "Distances are approximate."
        )

    return (
        "The geocoder returned only an approximate location. "
        "Distances should be interpreted carefully."
    )


def _geocoding_error(address: str, error_key: str, **error_kwargs) -> dict:
    """Return a cache-safe, language-neutral geocoding error payload.

    Translation belongs to the UI layer. Keeping only the translation key and
    formatting arguments in the cached result prevents one session's language
    from leaking into another session through Streamlit's shared data cache.
    """
    return {
        "ok": False,
        "address": address,
        "error_key": error_key,
        "error_kwargs": dict(error_kwargs),
    }


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def geocode_chilean_address(address: str) -> dict:
    """Geocode a family-entered address using OpenStreetMap/Nominatim.

    The app does not store the address permanently; Streamlit only keeps the
    returned coordinates in session/cache to avoid repeated calls on reruns.

    This version asks Nominatim for several candidates and selects the best one
    with a conservative precision score. It avoids treating a street-level match
    as if it were an exact house-number match.
    """
    original_address = " ".join(str(address or "").strip().split())
    query = normalize_address_for_geocoding(original_address)
    if not query:
        return _geocoding_error(original_address, "No address entered.")

    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "limit": GEOCODING_RESULT_LIMIT,
        "addressdetails": 1,
        "countrycodes": "cl",
        "dedupe": 0,
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
                return _geocoding_error(
                    original_address,
                    "Geocoding service returned status {status}.",
                    status=status,
                )
            payload = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        return _geocoding_error(
            original_address,
            "Could not reach the geocoding service: {error}",
            error=str(exc),
        )
    except Exception as exc:
        return _geocoding_error(
            original_address,
            "Could not reach the geocoding service: {error}",
            error=str(exc),
        )

    try:
        results = json.loads(payload)
    except Exception as exc:
        return _geocoding_error(
            original_address,
            "Could not read the geocoding response: {error}",
            error=str(exc),
        )

    if not results:
        return _geocoding_error(
            original_address,
            "No result found for this address in Chile.",
        )

    scored_results = []

    for rank, result in enumerate(results, start=1):
        lat = parse_coordinate(result.get("lat"))
        lon = parse_coordinate(result.get("lon"))

        if not valid_lat_lon(lat, lon):
            continue

        score_info = _score_nominatim_result(result, original_address)
        scored_results.append({
            "rank": rank,
            "result": result,
            "lat": float(lat),
            "lon": float(lon),
            **score_info,
        })

    if not scored_results:
        return _geocoding_error(
            original_address,
            "The geocoded result is outside Chile or has invalid coordinates.",
        )

    best = max(scored_results, key=lambda item: (item["score"], -item["rank"]))
    best_result = best["result"]

    return {
        "ok": True,
        "address": original_address,
        "query": query,
        "lat": best["lat"],
        "lon": best["lon"],
        "display_name": str(best_result.get("display_name", query)),
        "precision": best["precision"],
        "score": float(best["score"]),
        "candidate_count": len(results),
        "selected_candidate_rank": int(best["rank"]),
        "house_number_requested": best.get("house_number_requested", ""),
        "house_number_found": best.get("house_number_found", ""),
        "house_number_confirmed": bool(best.get("house_number_confirmed", False)),
        "road_match": float(best.get("road_match", 0.0)),
        "city_match": float(best.get("city_match", 0.0)),
    }
