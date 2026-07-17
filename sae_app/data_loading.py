"""Reading and validating the CSV data files -- backend dispatcher.

This module has no logic of its own. It re-exports the exact same public API
from one of two interchangeable implementations, selected by
sae_app.constants.USE_DUCKDB:

- USE_DUCKDB = False (default): sae_app.data_loading_pandas, the original
  pandas implementation, unchanged from before DuckDB was introduced. Does
  not require the duckdb package to be installed.
- USE_DUCKDB = True: sae_app.data_loading_duckdb, which runs the same joins
  and validation as SQL through DuckDB (see sae_app/db.py).

Only the selected branch is ever imported, so running with USE_DUCKDB=False
has zero dependency on duckdb being installed. Every other module in this
package imports from here (`from sae_app.data_loading import ...`), never
from the two backend modules directly, so the mode switch is transparent to
the rest of the app.
"""

from __future__ import annotations

import io
import re
import numpy as np
import pandas as pd
import streamlit as st

from sae_app.constants import (
    CHILE_COORDINATE_ZONES,
    PROGRAM_DISPLAY_NAME,
    PROGRAM_ENROLLMENT_FEE,
    PROGRAM_FILTERS_PATH,
    PROGRAM_GENDER,
    PROGRAM_LATITUDE,
    PROGRAM_LONGITUDE,
    PROGRAM_MONTHLY_FEE,
    PROGRAM_NAMES_PATH,
    PROGRAM_PACE,
    PROGRAM_PIE,
    PROGRAM_RELIGIOUS_ORIENTATION,
    PROGRAM_RURALITY,
    PROGRAM_SCHOOL_DAY,
    PROGRAM_SPECIALTY_NAME,
    PROGRAM_SPECIALTY_SECTOR,
    PROGRAM_TRACK,
    RBD_REGION_PATH,
    REGION,
    REGION_ORDER,
    SCHOOL_COMMUNE,
    SCHOOL_NAME,
    TIERS,
    TRACK_SPECIALIZED,
    UNKNOWN_FILTER_VALUE,
    UNKNOWN_PROGRAM_NAME,
    UNKNOWN_REGION,
    UNKNOWN_SCHOOL_NAME,
    CAPACITY,
    POP,
    PRIORITY_STUDENT_SEATS,
    TRUE_APP,
)
from sae_app.text_utils import clean_optional_value, clean_text, norm_code, parse_coordinate


class DataSchemaError(ValueError):
    """Raised when an input CSV is readable but structurally unsafe to use."""


_REQUIRED_CODE_PATTERN = re.compile(r"^([0-9]+)(?:[.,]0+)?$")


def _normalize_required_code_series(
    values: pd.Series,
    *,
    field_name: str,
    source_name: str,
) -> pd.Series:
    """Normalize a required positive-integer identifier or raise a schema error.

    RBD and program codes are identifiers rather than free-form labels. Common
    spreadsheet exports such as ``123.0``, ``42,00`` and ``="456"`` are
    accepted only when the decimal part is zero. Empty, non-numeric,
    fractional, zero and negative values are rejected before they can
    participate in joins or become program keys.

    Leading zeroes are removed as text rather than through ``int()`` so even an
    unexpectedly long malformed identifier is handled deterministically and
    can only produce a ``DataSchemaError``.
    """
    normalized: list[str] = []
    invalid_examples: list[str] = []

    for csv_row, value in enumerate(values.tolist(), start=2):
        text = "" if pd.isna(value) else str(value).strip()

        if text.startswith('="') and text.endswith('"'):
            text = text[2:-1].strip()

        match = _REQUIRED_CODE_PATTERN.fullmatch(text)
        if match is None:
            invalid_examples.append(f"row {csv_row}: {value!r}")
            normalized.append("")
            continue

        digits = match.group(1).lstrip("0")
        if not digits:
            invalid_examples.append(f"row {csv_row}: {value!r}")
            normalized.append("")
            continue

        normalized.append(digits)

    if invalid_examples:
        shown = "; ".join(invalid_examples[:5])
        remaining = len(invalid_examples) - 5
        suffix = f"; and {remaining} more" if remaining > 0 else ""
        raise DataSchemaError(
            f"{source_name} contains invalid {field_name} value(s): {shown}{suffix}. "
            "Expected a positive integer identifier."
        )

    return pd.Series(normalized, index=values.index, dtype="object")


# ---------------------------------------------------------------------------
# CSV reading utilities
# ---------------------------------------------------------------------------

def read_csv(file_bytes: bytes, sep: str = "auto") -> pd.DataFrame:
    kwargs: dict = {"dtype": str, "encoding": "utf-8-sig"}
    if sep == "auto":
        kwargs |= {"sep": None, "engine": "python"}
    else:
        kwargs["sep"] = sep
    df = pd.read_csv(io.BytesIO(file_bytes), **kwargs)
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    return df


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first existing column from a candidate list."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _drop_exact_duplicates_or_raise_conflicts(
    df: pd.DataFrame,
    key_columns: list[str],
    *,
    source_name: str,
) -> pd.DataFrame:
    """Drop exact duplicate keys, but reject keys with conflicting content.

    The previous loaders called drop_duplicates() directly, which could hide a
    future source-data conflict by keeping whichever row happened to appear
    first. Whitespace-only differences are ignored; any meaningful difference
    in the retained columns is reported before deduplication.
    """
    duplicate_mask = df.duplicated(key_columns, keep=False)
    if not duplicate_mask.any():
        return df

    duplicate_rows = df.loc[duplicate_mask].copy()
    value_columns = [col for col in df.columns if col not in key_columns]
    conflicting_keys: list[tuple] = []

    groupby_keys = key_columns[0] if len(key_columns) == 1 else key_columns
    for key, group in duplicate_rows.groupby(groupby_keys, dropna=False, sort=False):
        comparable = group[value_columns].copy()
        for col in value_columns:
            comparable[col] = comparable[col].map(
                lambda value: ""
                if pd.isna(value)
                else " ".join(str(value).strip().split())
            )
        if len(comparable.drop_duplicates()) > 1:
            if not isinstance(key, tuple):
                key = (key,)
            conflicting_keys.append(key)

    if conflicting_keys:
        examples = ", ".join(
            "/".join(str(part) for part in key)
            for key in conflicting_keys[:5]
        )
        raise DataSchemaError(
            f"{source_name} contains conflicting duplicate rows for key(s) "
            f"{', '.join(key_columns)}: {examples}"
        )

    return df.drop_duplicates(key_columns, keep="first").copy()


# ---------------------------------------------------------------------------
# Region lookup
# ---------------------------------------------------------------------------

def region_sort_index(region: str) -> int:
    try:
        return REGION_ORDER.index(str(region).strip())
    except ValueError:
        return len(REGION_ORDER)


def load_rbd_region_map(file_bytes: bytes | None = None) -> pd.DataFrame:
    """Load the RBD-region map, caching by the actual file contents."""
    if file_bytes is None:
        file_bytes = RBD_REGION_PATH.read_bytes()
    return _load_rbd_region_map(file_bytes)


@st.cache_data(show_spinner=False)
def _load_rbd_region_map(file_bytes: bytes) -> pd.DataFrame:
    df = read_csv(file_bytes, sep=",")

    required = {"rbd", REGION}
    missing = required - set(df.columns)
    if missing:
        raise DataSchemaError(f"{RBD_REGION_PATH.name} is missing columns: " + ", ".join(sorted(missing)))

    out = df[["rbd", REGION]].copy()
    out["rbd"] = _normalize_required_code_series(
        out["rbd"],
        field_name="rbd",
        source_name=RBD_REGION_PATH.name,
    )
    out[REGION] = out[REGION].astype(str).str.strip()
    out = _drop_exact_duplicates_or_raise_conflicts(
        out, ["rbd"], source_name=RBD_REGION_PATH.name
    )
    return out


def attach_regions(
    calib: pd.DataFrame,
    region_file_bytes: bytes | None = None,
) -> pd.DataFrame:
    """Attach region labels loaded from data/rbd_region_map.csv.

    This keeps every program from the capacities file. If an RBD is not found in
    the lookup, it is still available under Unknown region.
    """
    out = calib.copy()
    out["rbd"] = norm_code(out["rbd"])

    regions = load_rbd_region_map(region_file_bytes)
    out = out.merge(regions, on="rbd", how="left")
    out[REGION] = out[REGION].fillna(UNKNOWN_REGION)
    return out


def available_regions(calib: pd.DataFrame) -> list[str]:
    """Return regions present in the capacities file, in the official north-to-south order."""
    if REGION not in calib.columns:
        return [UNKNOWN_REGION]

    present = {str(x).strip() or UNKNOWN_REGION for x in calib[REGION].dropna().unique()}
    if not present:
        return [UNKNOWN_REGION]

    ordered = [r for r in REGION_ORDER if r in present]
    extra = sorted(r for r in present if r not in ordered)
    return ordered + extra


# ---------------------------------------------------------------------------
# Program-characteristic filters (data/program_filters.csv)
# ---------------------------------------------------------------------------

def load_program_filters(file_bytes: bytes | None = None) -> pd.DataFrame:
    """Load program filters, caching by the actual file contents."""
    if file_bytes is None:
        file_bytes = PROGRAM_FILTERS_PATH.read_bytes()
    return _load_program_filters(file_bytes)


@st.cache_data(show_spinner=False)
def _load_program_filters(file_bytes: bytes) -> pd.DataFrame:
    df = read_csv(file_bytes, sep=",")

    required = {
        "rbd",
        "program_code",
        PROGRAM_TRACK,
        PROGRAM_SPECIALTY_SECTOR,
        PROGRAM_SPECIALTY_NAME,
        PROGRAM_GENDER,
        PROGRAM_SCHOOL_DAY,
    }
    missing = required - set(df.columns)
    if missing:
        raise DataSchemaError(f"{PROGRAM_FILTERS_PATH.name} is missing columns: " + ", ".join(sorted(missing)))

    out = df[list(required)].copy()
    out["rbd"] = _normalize_required_code_series(
        out["rbd"],
        field_name="rbd",
        source_name=PROGRAM_FILTERS_PATH.name,
    )
    out["program_code"] = _normalize_required_code_series(
        out["program_code"],
        field_name="program_code",
        source_name=PROGRAM_FILTERS_PATH.name,
    )
    out = _drop_exact_duplicates_or_raise_conflicts(
        out, ["rbd", "program_code"], source_name=PROGRAM_FILTERS_PATH.name
    )
    return out


def attach_program_filters(
    calib: pd.DataFrame,
    filter_file_bytes: bytes | None = None,
) -> pd.DataFrame:
    """Attach program characteristics used by the sidebar filters.

    The capacities/calibration file remains the source of the available programs.
    The metadata file only adds filtering fields. Programs with missing metadata
    remain available when no filter is active.
    """
    out = calib.copy()
    out["rbd"] = norm_code(out["rbd"])
    out["program_code"] = norm_code(out["program_code"])

    filters = load_program_filters(filter_file_bytes)
    out = out.merge(filters, on=["rbd", "program_code"], how="left")

    for col in [
        PROGRAM_TRACK,
        PROGRAM_SPECIALTY_SECTOR,
        PROGRAM_SPECIALTY_NAME,
        PROGRAM_GENDER,
        PROGRAM_SCHOOL_DAY,
    ]:
        out[col] = out[col].fillna(UNKNOWN_FILTER_VALUE)

    return out


PROGRAM_FILTER_FIELD_CONFIG = [
    ("genders", PROGRAM_GENDER),
    ("school_days", PROGRAM_SCHOOL_DAY),
    ("rurality", PROGRAM_RURALITY),
    ("pie", PROGRAM_PIE),
    ("pace", PROGRAM_PACE),
    ("enrollment_fee", PROGRAM_ENROLLMENT_FEE),
    ("monthly_fee", PROGRAM_MONTHLY_FEE),
    ("religious_orientation", PROGRAM_RELIGIOUS_ORIENTATION),
]

PROGRAM_FILTER_KEYS = [
    "tracks",
    "specialty_sectors",
    *[filter_key for filter_key, _ in PROGRAM_FILTER_FIELD_CONFIG],
]


def program_matches_filters(row: pd.Series, filters: dict | None) -> bool:
    """Return True if a program row satisfies the sidebar filters.

    Empty filters mean no restriction. Selected existing wishes are preserved
    separately in filter_program_options().
    """
    if not filters:
        return True

    track = str(row.get(PROGRAM_TRACK, UNKNOWN_FILTER_VALUE)).strip()

    selected_tracks = filters.get("tracks") or []
    if selected_tracks and track not in selected_tracks:
        return False

    selected_specialties = filters.get("specialty_sectors") or []
    if selected_specialties:
        specialty_sector = str(row.get(PROGRAM_SPECIALTY_SECTOR, UNKNOWN_FILTER_VALUE)).strip()
        # Specialty-area filters only apply to specialized/technical programs.
        # General academic programs should remain visible when the family has
        # explicitly included them through the track filter.
        if track == TRACK_SPECIALIZED and specialty_sector not in selected_specialties:
            return False

    for filter_key, column_name in PROGRAM_FILTER_FIELD_CONFIG:
        selected_values = filters.get(filter_key) or []
        if not selected_values:
            continue
        value = str(row.get(column_name, UNKNOWN_FILTER_VALUE)).strip()
        if value not in selected_values:
            return False

    return True

def filters_are_active(filters: dict | None) -> bool:
    if not filters:
        return False
    return any(bool(filters.get(k)) for k in PROGRAM_FILTER_KEYS)



# ---------------------------------------------------------------------------
# Program names, school names, and additional recommendation criteria
# (data/programmes_chili_criteres_recommandation.csv)
# ---------------------------------------------------------------------------

def _program_descriptor_key(value: str) -> str:
    """Normalize program-name fragments before matching known descriptors."""
    key = clean_text(value, default="", lower=True, strip_accents=True)
    key = key.replace("º", "o").replace("°", "o")
    key = re.sub(r"[^a-z0-9]+", " ", key)
    return " ".join(key.split())


def _is_first_grade_secondary_fragment(value: str) -> bool:
    """Return True for common variants of the repeated 1º medio prefix."""
    key = _program_descriptor_key(value)
    return key in {"1 medio", "1o medio", "primero medio", "1st grade secondary"}


def compact_program_name(name: str) -> str:
    """Convert the reconstructed program name into a short English label.

    The source file contains concise reconstructed names such as
    "1º medio — Général H-C — Mixte — jornada completa". The dropdown is
    easier to scan if the repeated grade is removed and stable descriptors are
    translated. Matching is intentionally tolerant to accents, case, and the
    common º/° variants found in spreadsheet exports.
    """
    text = " ".join(str(name or "").strip().split())
    if not text:
        return UNKNOWN_PROGRAM_NAME

    descriptor_translations = {
        "general h c": "General H-C",
        "general hc": "General H-C",
        "specialite tp": "Technical-vocational",
        "speciality tp": "Technical-vocational",
        "especialidad tp": "Technical-vocational",
        "mixte": "Mixed",
        "mixto": "Mixed",
        "mixed": "Mixed",
        "garcons": "Boys",
        "hombres": "Boys",
        "varones": "Boys",
        "boys": "Boys",
        "filles": "Girls",
        "mujeres": "Girls",
        "girls": "Girls",
        "jornada completa": "Full day",
        "full day": "Full day",
        "jornada manana": "Morning",
        "morning": "Morning",
        "jornada tarde": "Afternoon",
        "afternoon": "Afternoon",
    }

    raw_parts = [
        part.strip()
        for part in re.split(r"\s*(?:—|–)\s*|\s+-\s+", text)
        if part.strip()
    ]

    parts = []
    for part in raw_parts:
        if _is_first_grade_secondary_fragment(part):
            continue
        key = _program_descriptor_key(part)
        parts.append(descriptor_translations.get(key, part))

    if not parts:
        return UNKNOWN_PROGRAM_NAME

    if parts[0] == "Technical-vocational" and len(parts) >= 2:
        main = f"Technical-vocational: {parts[1]}"
        rest = parts[2:]
        return " · ".join([main] + rest)

    return " · ".join(parts)

def compact_school_name(name: str) -> str:
    """Return a readable school name for the program dropdown."""
    text = " ".join(str(name or "").strip().split())
    if not text:
        return UNKNOWN_SCHOOL_NAME

    # The source file is mostly uppercase. Title case is easier to scan in a dropdown.
    if text.upper() == text:
        text = text.title()
        for old, new in {
            " De ": " de ",
            " Del ": " del ",
            " La ": " la ",
            " Las ": " las ",
            " Los ": " los ",
            " Y ": " y ",
        }.items():
            text = text.replace(old, new)

    return text


VALUE_TRANSLATIONS = {
    PROGRAM_RURALITY: {
        "Urbain": "Urban",
        "Rural": "Rural",
    },
    PROGRAM_PIE: {
        "Avec PIE": "With PIE",
        "Sans PIE": "Without PIE",
    },
    PROGRAM_PACE: {
        "Avec PACE": "With PACE",
        "Sans PACE": "Without PACE",
    },
    PROGRAM_ENROLLMENT_FEE: {
        "Gratuit": "Free",
        "$1.000 A $10.000": "$1,000–$10,000",
        "$10.001 A $25.000": "$10,001–$25,000",
        "$25.001 A $50.000": "$25,001–$50,000",
        "$50.001 A $100.000": "$50,001–$100,000",
        "MAS DE $100.000": "More than $100,000",
        "Sans information": "No information",
    },
    PROGRAM_MONTHLY_FEE: {
        "Gratuit": "Free",
        "$1.000 A $10.000": "$1,000–$10,000",
        "$10.001 A $25.000": "$10,001–$25,000",
        "$25.001 A $50.000": "$25,001–$50,000",
        "$50.001 A $100.000": "$50,001–$100,000",
        "MAS DE $100.000": "More than $100,000",
        "Sans information": "No information",
    },
    PROGRAM_RELIGIOUS_ORIENTATION: {
        "Laïque": "Secular",
        "Catholique": "Catholic",
        "Évangélique": "Evangelical",
        "Autre": "Other",
        "Sans information": "No information",
    },
}


def translate_filter_value(value, target_column: str, *, default: str = "No information") -> str:
    text = clean_optional_value(value, default=default)
    return VALUE_TRANSLATIONS.get(target_column, {}).get(text, text)


PROGRAM_COORDINATE_SOURCE = "program_coordinate_source"
PROGRAM_GEO_MATCH_LEVEL = "program_geo_match_level"
COORDINATE_DISCREPANCY_KM = "coordinate_discrepancy_km"
RBD_COORDINATE_SPREAD_KM = "rbd_coordinate_spread_km"

# Coordinate columns are evaluated as coherent pairs. Physical school
# coordinates are preferred to coordinates inherited from a matching step. A
# later pair is used only when no earlier pair is valid for that row.
PROGRAM_COORDINATE_COLUMN_PAIRS = [
    ("school_latitude", "school_longitude", "school coordinate"),
    ("lat_lycee", "lon_lycee", "school coordinate"),
    ("establecimiento_latitude", "establecimiento_longitude", "school coordinate"),
    (PROGRAM_LATITUDE, PROGRAM_LONGITUDE, "matched program coordinate"),
    ("school_capacity_lat", "school_capacity_lon", "matched program coordinate"),
    ("latitude", "longitude", "generic coordinate"),
    ("lat", "lon", "generic coordinate"),
    ("lat", "lng", "generic coordinate"),
    ("latitud", "longitud", "generic coordinate"),
    ("commune_latitude", "commune_longitude", "commune coordinate"),
]


def _coordinates_inside_chile_zones(
    latitudes: pd.Series,
    longitudes: pd.Series,
) -> pd.Series:
    """Vectorized mainland/insular Chile validation for coordinate ingestion."""
    valid = pd.Series(False, index=latitudes.index, dtype=bool)
    for _, min_lat, max_lat, min_lon, max_lon in CHILE_COORDINATE_ZONES:
        valid |= (
            latitudes.between(min_lat, max_lat, inclusive="both")
            & longitudes.between(min_lon, max_lon, inclusive="both")
        )
    return valid & latitudes.notna() & longitudes.notna()


def _coalesce_program_coordinates(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, list[str]]:
    """Select the first complete valid coordinate pair for each row.

    Latitude and longitude are never selected independently, which prevents
    hybrid coordinates such as a school latitude combined with a commune
    longitude. Rapa Nui and Juan Fernández are accepted through the same shared
    Chile-zone definition used by geo.valid_lat_lon().
    """
    latitudes = pd.Series(np.nan, index=df.index, dtype=float)
    longitudes = pd.Series(np.nan, index=df.index, dtype=float)
    sources = pd.Series("", index=df.index, dtype=object)
    source_columns: list[str] = []

    for lat_col, lon_col, source in PROGRAM_COORDINATE_COLUMN_PAIRS:
        if lat_col not in df.columns or lon_col not in df.columns:
            continue

        for col in (lat_col, lon_col):
            if col not in source_columns:
                source_columns.append(col)

        pair_latitudes = df[lat_col].map(parse_coordinate)
        pair_longitudes = df[lon_col].map(parse_coordinate)
        valid_pair = _coordinates_inside_chile_zones(pair_latitudes, pair_longitudes)
        use_pair = latitudes.isna() & longitudes.isna() & valid_pair

        latitudes.loc[use_pair] = pair_latitudes.loc[use_pair]
        longitudes.loc[use_pair] = pair_longitudes.loc[use_pair]
        sources.loc[use_pair] = source

    return latitudes, longitudes, sources, source_columns


def _parse_nonnegative_float(value) -> float:
    """Parse a non-negative numeric quality metric, otherwise return NaN."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", ".")
    if not text:
        return np.nan
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(parsed) or parsed < 0:
        return np.nan
    return parsed


def _great_circle_distance_between_points(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    """Return a haversine distance used only for ingestion quality checks."""
    radius_km = 6371.0
    lat1, lon1, lat2, lon2 = np.radians(
        [latitude_a, longitude_a, latitude_b, longitude_b]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return float(2.0 * radius_km * np.arcsin(np.sqrt(a)))


def _rbd_coordinate_spread_km(
    rbd_values: pd.Series,
    latitudes: pd.Series,
    longitudes: pd.Series,
    coordinate_sources: pd.Series,
) -> pd.Series:
    """Return the maximum pairwise school-coordinate distance for each RBD."""
    spread = pd.Series(np.nan, index=rbd_values.index, dtype=float)
    valid = (
        coordinate_sources.eq("school coordinate")
        & latitudes.notna()
        & longitudes.notna()
    )
    if not valid.any():
        return spread

    normalized_rbd = norm_code(rbd_values)
    quality_frame = pd.DataFrame(
        {
            "rbd": normalized_rbd,
            "lat": latitudes,
            "lon": longitudes,
        }
    ).loc[valid]

    for rbd, group in quality_frame.groupby("rbd", sort=False):
        coordinates = (
            group[["lat", "lon"]]
            .dropna()
            .drop_duplicates()
            .to_numpy(dtype=float)
        )
        group_spread = 0.0
        for first in range(len(coordinates)):
            for second in range(first + 1, len(coordinates)):
                group_spread = max(
                    group_spread,
                    _great_circle_distance_between_points(
                        coordinates[first, 0],
                        coordinates[first, 1],
                        coordinates[second, 0],
                        coordinates[second, 1],
                    ),
                )
        spread.loc[normalized_rbd.eq(rbd)] = group_spread

    return spread


@st.cache_data(show_spinner=False)
def load_program_names(file_bytes: bytes) -> pd.DataFrame:
    df = read_csv(file_bytes, sep="auto")
    df.columns = [str(c).lstrip("﻿").strip() for c in df.columns]

    required = {"rbd", "program_code", "nom_programme_reconstruit", "nom_lycee"}
    missing = required - set(df.columns)
    if missing:
        raise DataSchemaError(f"{PROGRAM_NAMES_PATH.name} is missing columns: " + ", ".join(sorted(missing)))

    optional_source_cols = [
        "commune",
        "ruralite",
        "convenio_pie",
        "pace",
        "paiement_matricula",
        "paiement_mensualite",
        "orientation_religieuse",
from sae_app.constants import USE_DUCKDB
from sae_app.errors import DataSchemaError

if USE_DUCKDB:
    from sae_app.data_loading_duckdb import (
        COORDINATE_DISCREPANCY_KM,
        PROGRAM_COORDINATE_COLUMN_PAIRS,
        PROGRAM_COORDINATE_SOURCE,
        PROGRAM_FILTER_FIELD_CONFIG,
        PROGRAM_FILTER_KEYS,
        PROGRAM_GEO_MATCH_LEVEL,
        RBD_COORDINATE_SPREAD_KM,
        SEAT_COMPONENT_COLUMNS,
        TOTAL_CAPACITY_COLUMN,
        VALUE_TRANSLATIONS,
        attach_program_filters,
        attach_program_names,
        attach_regions,
        available_regions,
        compact_program_name,
        compact_school_name,
        filters_are_active,
        first_existing_column,
        load_calibration,
        load_program_filters,
        load_program_names,
        load_rbd_region_map,
        program_matches_filters,
        read_csv,
        read_csv_path,
        region_sort_index,
        required_cols,
        translate_filter_value,
        validate_core_numeric_columns,
        validate_cumulative_share_columns,
    )
else:
    from sae_app.data_loading_pandas import (
        COORDINATE_DISCREPANCY_KM,
        PROGRAM_COORDINATE_COLUMN_PAIRS,
        PROGRAM_COORDINATE_SOURCE,
        PROGRAM_FILTER_FIELD_CONFIG,
        PROGRAM_FILTER_KEYS,
        PROGRAM_GEO_MATCH_LEVEL,
        RBD_COORDINATE_SPREAD_KM,
        SEAT_COMPONENT_COLUMNS,
        TOTAL_CAPACITY_COLUMN,
        VALUE_TRANSLATIONS,
        attach_program_filters,
        attach_program_names,
        attach_regions,
        available_regions,
        compact_program_name,
        compact_school_name,
        filters_are_active,
        first_existing_column,
        load_calibration,
        load_program_filters,
        load_program_names,
        load_rbd_region_map,
        program_matches_filters,
        read_csv,
        read_csv_path,
        region_sort_index,
        required_cols,
        translate_filter_value,
        validate_core_numeric_columns,
        validate_cumulative_share_columns,
    )

__all__ = [
    "COORDINATE_DISCREPANCY_KM",
    "DataSchemaError",
    "PROGRAM_COORDINATE_COLUMN_PAIRS",
    "PROGRAM_COORDINATE_SOURCE",
    "PROGRAM_FILTER_FIELD_CONFIG",
    "PROGRAM_FILTER_KEYS",
    "PROGRAM_GEO_MATCH_LEVEL",
    "RBD_COORDINATE_SPREAD_KM",
    "SEAT_COMPONENT_COLUMNS",
    "TOTAL_CAPACITY_COLUMN",
    "VALUE_TRANSLATIONS",
    "attach_program_filters",
    "attach_program_names",
    "attach_regions",
    "available_regions",
    "compact_program_name",
    "compact_school_name",
    "filters_are_active",
    "first_existing_column",
    "load_calibration",
    "load_program_filters",
    "load_program_names",
    "load_rbd_region_map",
    "program_matches_filters",
    "read_csv",
    "read_csv_path",
    "region_sort_index",
    "required_cols",
    "translate_filter_value",
    "validate_core_numeric_columns",
    "validate_cumulative_share_columns",
]
