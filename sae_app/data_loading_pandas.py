"""Reading and validating the CSV data files -- original pandas implementation.

This module owns "how to interpret the raw CSV files on disk" — encoding
quirks, column-name variants, and source-language value translation. It does
not know anything about Streamlit widgets or the recommendation engine.

This is the pre-DuckDB implementation, kept verbatim as the default backend.
sae_app/data_loading.py dispatches to this module or to
sae_app/data_loading_duckdb.py based on sae_app.constants.USE_DUCKDB. Do not
import this module directly -- import sae_app.data_loading instead, so the
mode switch keeps working.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

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
from sae_app.errors import DataSchemaError
from sae_app.text_utils import clean_optional_value, clean_text, norm_code, parse_coordinate

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


def read_csv_path(path: Path, sep: str = "auto") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    return read_csv(path.read_bytes(), sep=sep)


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
        PROGRAM_GEO_MATCH_LEVEL,
        COORDINATE_DISCREPANCY_KM,
    ]
    coordinate_latitudes, coordinate_longitudes, coordinate_sources, coordinate_source_cols = (
        _coalesce_program_coordinates(df)
    )

    keep_cols = ["rbd", "program_code", "nom_programme_reconstruit", "nom_lycee"]
    keep_cols += [c for c in optional_source_cols if c in df.columns]
    keep_cols += [c for c in coordinate_source_cols if c not in keep_cols]

    out = df[keep_cols].copy()
    out["rbd"] = _normalize_required_code_series(
        out["rbd"],
        field_name="rbd",
        source_name=PROGRAM_NAMES_PATH.name,
    )
    out["program_code"] = _normalize_required_code_series(
        out["program_code"],
        field_name="program_code",
        source_name=PROGRAM_NAMES_PATH.name,
    )
    out[PROGRAM_DISPLAY_NAME] = out["nom_programme_reconstruit"].map(compact_program_name)
    out[SCHOOL_NAME] = out["nom_lycee"].map(compact_school_name)
    out[SCHOOL_COMMUNE] = (
        out["commune"].astype(str).str.strip().str.title()
        if "commune" in out.columns else ""
    )
    out[PROGRAM_LATITUDE] = coordinate_latitudes
    out[PROGRAM_LONGITUDE] = coordinate_longitudes
    out[PROGRAM_COORDINATE_SOURCE] = coordinate_sources
    out[PROGRAM_GEO_MATCH_LEVEL] = (
        out[PROGRAM_GEO_MATCH_LEVEL].fillna("").astype(str).str.strip()
        if PROGRAM_GEO_MATCH_LEVEL in out.columns
        else ""
    )
    out[COORDINATE_DISCREPANCY_KM] = (
        out[COORDINATE_DISCREPANCY_KM].map(_parse_nonnegative_float)
        if COORDINATE_DISCREPANCY_KM in out.columns
        else np.nan
    )
    out[RBD_COORDINATE_SPREAD_KM] = _rbd_coordinate_spread_km(
        out["rbd"],
        coordinate_latitudes,
        coordinate_longitudes,
        coordinate_sources,
    )

    criteria_sources = {
        PROGRAM_RURALITY: "ruralite",
        PROGRAM_PIE: "convenio_pie",
        PROGRAM_PACE: "pace",
        PROGRAM_ENROLLMENT_FEE: "paiement_matricula",
        PROGRAM_MONTHLY_FEE: "paiement_mensualite",
        PROGRAM_RELIGIOUS_ORIENTATION: "orientation_religieuse",
    }
    for target_col, source_col in criteria_sources.items():
        if source_col in out.columns:
            out[target_col] = out[source_col].map(lambda x, c=target_col: translate_filter_value(x, c))
        else:
            out[target_col] = "No information"

    source_cols_to_drop = [
        "nom_programme_reconstruit",
        "nom_lycee",
        "commune",
        "ruralite",
        "convenio_pie",
        "pace",
        "paiement_matricula",
        "paiement_mensualite",
        "orientation_religieuse",
    ]
    source_cols_to_drop += [
        c for c in coordinate_source_cols if c not in {PROGRAM_LATITUDE, PROGRAM_LONGITUDE}
    ]
    out = out.drop(columns=[c for c in source_cols_to_drop if c in out.columns])
    out = _drop_exact_duplicates_or_raise_conflicts(
        out, ["rbd", "program_code"], source_name=PROGRAM_NAMES_PATH.name
    )
    return out


def attach_program_names(
    calib: pd.DataFrame,
    program_names_file_bytes: bytes | None = None,
) -> pd.DataFrame:
    """Attach display names, school names, and additional choice criteria."""
    out = calib.copy()
    out["rbd"] = norm_code(out["rbd"])
    out["program_code"] = norm_code(out["program_code"])

    if program_names_file_bytes is None:
        program_names_file_bytes = PROGRAM_NAMES_PATH.read_bytes()
    names = load_program_names(program_names_file_bytes)
    out = out.merge(names, on=["rbd", "program_code"], how="left")
    out[PROGRAM_DISPLAY_NAME] = out[PROGRAM_DISPLAY_NAME].fillna(UNKNOWN_PROGRAM_NAME)
    out[SCHOOL_NAME] = out[SCHOOL_NAME].fillna("")
    out[SCHOOL_COMMUNE] = out[SCHOOL_COMMUNE].fillna("")
    if PROGRAM_LATITUDE not in out.columns:
        out[PROGRAM_LATITUDE] = np.nan
    if PROGRAM_LONGITUDE not in out.columns:
        out[PROGRAM_LONGITUDE] = np.nan
    if PROGRAM_COORDINATE_SOURCE not in out.columns:
        out[PROGRAM_COORDINATE_SOURCE] = ""
    else:
        out[PROGRAM_COORDINATE_SOURCE] = out[PROGRAM_COORDINATE_SOURCE].fillna("")
    if PROGRAM_GEO_MATCH_LEVEL not in out.columns:
        out[PROGRAM_GEO_MATCH_LEVEL] = ""
    else:
        out[PROGRAM_GEO_MATCH_LEVEL] = out[PROGRAM_GEO_MATCH_LEVEL].fillna("")
    for col in (COORDINATE_DISCREPANCY_KM, RBD_COORDINATE_SPREAD_KM):
        if col not in out.columns:
            out[col] = np.nan

    for col in [
        PROGRAM_RURALITY,
        PROGRAM_PIE,
        PROGRAM_PACE,
        PROGRAM_ENROLLMENT_FEE,
        PROGRAM_MONTHLY_FEE,
        PROGRAM_RELIGIOUS_ORIENTATION,
    ]:
        out[col] = out[col].fillna("No information")
    return out


# ---------------------------------------------------------------------------
# Calibration file loading and validation
# (data/capacities_2025_wta_with_2024_calibration.csv)
# ---------------------------------------------------------------------------

def load_calibration(file_bytes: bytes) -> pd.DataFrame:
    """Load all calibration inputs, caching by every file's actual contents."""
    return _load_calibration(
        file_bytes,
        RBD_REGION_PATH.read_bytes(),
        PROGRAM_FILTERS_PATH.read_bytes(),
        PROGRAM_NAMES_PATH.read_bytes(),
    )


@st.cache_data(show_spinner=False)
def _load_calibration(
    file_bytes: bytes,
    region_file_bytes: bytes,
    filter_file_bytes: bytes,
    program_names_file_bytes: bytes,
) -> pd.DataFrame:
    """Parse and merge calibration inputs after minimum-schema validation."""
    df = read_csv(file_bytes, sep=";")
    if len(df.columns) == 1:
        df = read_csv(file_bytes, sep=",")

    base_required = {"rbd", "program_code"}
    missing = base_required - set(df.columns)
    if missing:
        raise DataSchemaError(
            "Calibration CSV is missing required column(s): "
            + ", ".join(sorted(missing))
        )

    # Validate and normalize only after the minimum schema is known to be
    # present. This avoids a raw KeyError and prevents malformed identifiers
    # from silently participating in joins.
    df["rbd"] = _normalize_required_code_series(
        df["rbd"],
        field_name="rbd",
        source_name="Calibration CSV",
    )
    df["program_code"] = _normalize_required_code_series(
        df["program_code"],
        field_name="program_code",
        source_name="Calibration CSV",
    )
    df = _drop_exact_duplicates_or_raise_conflicts(
        df, ["rbd", "program_code"], source_name="Calibration CSV"
    )

    df = attach_regions(df, region_file_bytes)
    df = attach_program_filters(df, filter_file_bytes)
    df = attach_program_names(df, program_names_file_bytes)
    return df


def required_cols() -> list[str]:
    cols = ["rbd", "program_code", CAPACITY, PRIORITY_STUDENT_SEATS, TRUE_APP, POP]
    for tier in TIERS:
        cols += [
            f"priority_share_{tier}_2024",
            f"cum_share_before_{tier}_2024",
            f"cum_share_through_{tier}_2024",
        ]
    return cols


def validate_cumulative_share_columns(
    calib: pd.DataFrame,
    *,
    tolerance: float = 1e-6,
) -> list[str]:
    """Validate the full priority-share distribution for every program.

    Checks include missing/non-numeric values, [0, 1] bounds,
    before + share == through, continuity between adjacent tiers, a zero start,
    and a final cumulative share of one.
    """
    errors: list[str] = []
    numeric_by_tier: dict[str, tuple[pd.Series, pd.Series, pd.Series]] = {}

    for tier in TIERS:
        share_col = f"priority_share_{tier}_2024"
        before_col = f"cum_share_before_{tier}_2024"
        through_col = f"cum_share_through_{tier}_2024"
        needed = [share_col, before_col, through_col]
        if any(col not in calib.columns for col in needed):
            # Missing columns are reported by required_cols() in app.py.
            continue

        share = pd.to_numeric(calib[share_col], errors="coerce")
        before = pd.to_numeric(calib[before_col], errors="coerce")
        through = pd.to_numeric(calib[through_col], errors="coerce")
        numeric_by_tier[tier] = (share, before, through)

        missing_or_invalid = share.isna() | before.isna() | through.isna()
        if missing_or_invalid.any():
            errors.append(
                f"{tier}: {int(missing_or_invalid.sum())} row(s) with missing or "
                f"non-numeric calibration share values in {share_col}, "
                f"{before_col}, or {through_col}."
            )

        valid = ~missing_or_invalid
        out_of_range = valid & (
            (share < -tolerance)
            | (share > 1 + tolerance)
            | (before < -tolerance)
            | (before > 1 + tolerance)
            | (through < -tolerance)
            | (through > 1 + tolerance)
        )
        if out_of_range.any():
            errors.append(
                f"{tier}: {int(out_of_range.sum())} row(s) with calibration "
                "shares or cumulative shares outside [0, 1]."
            )

        expected_through = before + share
        diff = (expected_through - through).abs()
        inconsistent = valid & (diff > tolerance)
        if inconsistent.any():
            errors.append(
                f"{through_col}: {int(inconsistent.sum())} row(s) where "
                "cum_share_before + priority_share differs from "
                f"cum_share_through (max diff {float(diff[inconsistent].max()):.6g})."
            )

    if TIERS and TIERS[0] in numeric_by_tier:
        first_before = numeric_by_tier[TIERS[0]][1]
        invalid_start = first_before.notna() & (first_before.abs() > tolerance)
        if invalid_start.any():
            errors.append(
                f"{int(invalid_start.sum())} row(s) where the first priority "
                "tier does not start at cumulative share 0."
            )

    for previous_tier, current_tier in zip(TIERS, TIERS[1:]):
        if previous_tier not in numeric_by_tier or current_tier not in numeric_by_tier:
            continue
        previous_through = numeric_by_tier[previous_tier][2]
        current_before = numeric_by_tier[current_tier][1]
        valid = previous_through.notna() & current_before.notna()
        gap = (previous_through - current_before).abs()
        discontinuous = valid & (gap > tolerance)
        if discontinuous.any():
            errors.append(
                f"{previous_tier} -> {current_tier}: "
                f"{int(discontinuous.sum())} row(s) with a discontinuity between "
                "the previous cumulative end and the next cumulative start "
                f"(max gap {float(gap[discontinuous].max()):.6g})."
            )

    if TIERS and TIERS[-1] in numeric_by_tier:
        final_through = numeric_by_tier[TIERS[-1]][2]
        invalid_end = final_through.notna() & ((final_through - 1.0).abs() > tolerance)
        if invalid_end.any():
            errors.append(
                f"{int(invalid_end.sum())} row(s) where the final cumulative "
                "priority share does not equal 1."
            )

    return errors


SEAT_COMPONENT_COLUMNS = [
    "integration_student_seats",
    PRIORITY_STUDENT_SEATS,
    "high_selectivity_seats_transitional",
    "high_selectivity_seats_ranking",
    "regular_seats",
]
TOTAL_CAPACITY_COLUMN = "total_capacity"


def validate_core_numeric_columns(
    calib: pd.DataFrame,
    *,
    tolerance: float = 1e-6,
) -> list[str]:
    """Validate core counts and cross-column calibration relationships.

    Counts used by the model must be finite non-negative integers, lottery
    population must be positive, true applicants cannot exceed that population,
    and the admission-seat components must reconcile to the admission-seat total.
    """
    errors: list[str] = []
    numeric: dict[str, pd.Series] = {}

    integer_columns = [
        CAPACITY,
        TRUE_APP,
        POP,
        TOTAL_CAPACITY_COLUMN,
        *SEAT_COMPONENT_COLUMNS,
    ]
    for col in integer_columns:
        if col not in calib.columns:
            continue

        values = pd.to_numeric(calib[col], errors="coerce")
        numeric[col] = values
        invalid_number = values.isna()
        if invalid_number.any():
            errors.append(
                f"{col}: {int(invalid_number.sum())} row(s) with missing or "
                "non-numeric values."
            )

        valid = ~invalid_number
        minimum_invalid = valid & (values <= 0) if col == POP else valid & (values < 0)
        if minimum_invalid.any():
            requirement = "positive" if col == POP else "non-negative"
            errors.append(
                f"{col}: {int(minimum_invalid.sum())} row(s) that are not {requirement}."
            )

        non_integer = valid & ((values - values.round()).abs() > tolerance)
        if non_integer.any():
            errors.append(
                f"{col}: {int(non_integer.sum())} row(s) with non-integer count values."
            )

    if TRUE_APP in numeric and POP in numeric:
        valid = numeric[TRUE_APP].notna() & numeric[POP].notna()
        impossible = valid & (numeric[TRUE_APP] > numeric[POP] + tolerance)
        if impossible.any():
            errors.append(
                f"{TRUE_APP}: {int(impossible.sum())} row(s) where true applicants "
                f"exceed {POP}."
            )

    if CAPACITY in numeric and PRIORITY_STUDENT_SEATS in numeric:
        valid = numeric[CAPACITY].notna() & numeric[PRIORITY_STUDENT_SEATS].notna()
        exceeds_capacity = valid & (
            numeric[PRIORITY_STUDENT_SEATS] > numeric[CAPACITY] + tolerance
        )
        if exceeds_capacity.any():
            errors.append(
                f"{PRIORITY_STUDENT_SEATS}: {int(exceeds_capacity.sum())} row(s) "
                f"exceed {CAPACITY}."
            )

    if CAPACITY in numeric and TOTAL_CAPACITY_COLUMN in numeric:
        valid = numeric[CAPACITY].notna() & numeric[TOTAL_CAPACITY_COLUMN].notna()
        impossible = valid & (numeric[CAPACITY] > numeric[TOTAL_CAPACITY_COLUMN] + tolerance)
        if impossible.any():
            errors.append(
                f"{CAPACITY}: {int(impossible.sum())} row(s) where admission seats "
                f"exceed {TOTAL_CAPACITY_COLUMN}."
            )

    if CAPACITY in numeric and all(col in numeric for col in SEAT_COMPONENT_COLUMNS):
        components = sum((numeric[col] for col in SEAT_COMPONENT_COLUMNS), start=pd.Series(0.0, index=calib.index))
        valid = numeric[CAPACITY].notna()
        for col in SEAT_COMPONENT_COLUMNS:
            valid &= numeric[col].notna()
        mismatch = valid & ((components - numeric[CAPACITY]).abs() > tolerance)
        if mismatch.any():
            errors.append(
                f"{CAPACITY}: {int(mismatch.sum())} row(s) where seat components "
                "do not sum to total admission seats "
                f"(max diff {float((components[mismatch] - numeric[CAPACITY][mismatch]).abs().max()):.6g})."
            )

    return errors
