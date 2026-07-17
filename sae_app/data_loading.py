"""Reading and validating the CSV data files.

This module owns "how to interpret the raw CSV files on disk" — encoding
quirks, column-name variants, and source-language value translation. It does
not know anything about Streamlit widgets or the recommendation engine.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sae_app import db
from sae_app.constants import (
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
from sae_app.text_utils import clean_optional_value, clean_text, norm_code


class DataSchemaError(ValueError):
    """Raised when an input CSV is readable but structurally unsafe to use."""


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

    Backed by sae_app.db.normalize_required_code_column: leading zeroes are
    stripped as text (via SQL ltrim) rather than through int(), so an
    unexpectedly long malformed identifier is handled deterministically and
    can only produce a DataSchemaError.
    """
    frame = pd.DataFrame({"value": values.reset_index(drop=True)})
    con = db.connect()
    try:
        normalized = db.normalize_required_code_column(
            con, frame, "value", field_name=field_name, source_name=source_name,
        )
    finally:
        con.close()
    return pd.Series(normalized.tolist(), index=values.index, dtype="object")


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

    Backed by sae_app.db.drop_exact_duplicates_or_raise_conflicts. Dropping
    duplicates directly could hide a future source-data conflict by keeping
    whichever row happened to appear first. Whitespace-only differences are
    ignored; any meaningful difference in the retained columns is reported
    before deduplication.
    """
    con = db.connect()
    try:
        return db.drop_exact_duplicates_or_raise_conflicts(
            con, df, key_columns, source_name=source_name,
        )
    finally:
        con.close()


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
    con = db.connect()
    try:
        out = db.left_join_preserving_order(con, out, regions, on=["rbd"])
    finally:
        con.close()
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
    con = db.connect()
    try:
        out = db.left_join_preserving_order(con, out, filters, on=["rbd", "program_code"])
    finally:
        con.close()

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


def _parse_nonnegative_float_expr(column: str) -> str:
    """SQL expression parsing a possibly comma-decimal, non-negative numeric quality metric."""
    parsed = db.parse_coordinate_expr(column)  # finite double or NULL
    return f"(CASE WHEN {parsed} >= 0 THEN {parsed} ELSE NULL END)"


@st.cache_data(show_spinner=False)
def load_program_names(file_bytes: bytes) -> pd.DataFrame:
    df = read_csv(file_bytes, sep="auto")
    df.columns = [str(c).lstrip("﻿").strip() for c in df.columns]

    required = {"rbd", "program_code", "nom_programme_reconstruit", "nom_lycee"}
    missing = required - set(df.columns)
    if missing:
        raise DataSchemaError(f"{PROGRAM_NAMES_PATH.name} is missing columns: " + ", ".join(sorted(missing)))

    con = db.connect()
    db.register_text_udf(con, "compact_program_name_udf", compact_program_name, arg_types=["VARCHAR"])
    db.register_text_udf(con, "compact_school_name_udf", compact_school_name, arg_types=["VARCHAR"])
    db.register_text_udf(
        con, "title_case_udf",
        lambda x: "" if x is None else str(x).strip().title(),
        arg_types=["VARCHAR"],
    )
    db.register_text_udf(
        con, "translate_filter_value_udf",
        lambda value, target_column: translate_filter_value(value, target_column),
        arg_types=["VARCHAR", "VARCHAR"],
    )

    working = df.reset_index(drop=True).copy()
    working["rbd"] = db.normalize_required_code_column(
        con, working, "rbd", field_name="rbd", source_name=PROGRAM_NAMES_PATH.name,
    )
    working["program_code"] = db.normalize_required_code_column(
        con, working, "program_code", field_name="program_code", source_name=PROGRAM_NAMES_PATH.name,
    )

    # Distinct internal names: several PROGRAM_COORDINATE_COLUMN_PAIRS candidates
    # (e.g. program_latitude/program_longitude) share a name with the final
    # output column, which the closing explicit SELECT resolves by never
    # passing raw columns through with `SELECT *`.
    working = db.coalesce_program_coordinates(
        con, working, PROGRAM_COORDINATE_COLUMN_PAIRS,
        latitude_out="_coalesced_latitude",
        longitude_out="_coalesced_longitude",
        source_out="_coalesced_source",
    )
    working = db.rbd_coordinate_spread_km(
        con, working,
        rbd_column="rbd",
        latitude_column="_coalesced_latitude",
        longitude_column="_coalesced_longitude",
        source_column="_coalesced_source",
        school_coordinate_label="school coordinate",
        output_column="_rbd_spread_km",
    )

    con.register("_names_src", working)

    criteria_sources = {
        PROGRAM_RURALITY: "ruralite",
        PROGRAM_PIE: "convenio_pie",
        PROGRAM_PACE: "pace",
        PROGRAM_ENROLLMENT_FEE: "paiement_matricula",
        PROGRAM_MONTHLY_FEE: "paiement_mensualite",
        PROGRAM_RELIGIOUS_ORIENTATION: "orientation_religieuse",
    }
    criteria_select = [
        f"""translate_filter_value_udf("{source_col}", '{target_col}') AS "{target_col}\""""
        if source_col in df.columns
        else f"""'No information' AS "{target_col}\""""
        for target_col, source_col in criteria_sources.items()
    ]

    school_commune_expr = "title_case_udf(commune)" if "commune" in df.columns else "''"
    empty_quoted = "''"
    geo_match_level_expr = (
        f'coalesce(trim(CAST("{PROGRAM_GEO_MATCH_LEVEL}" AS VARCHAR)), {empty_quoted})'
        if PROGRAM_GEO_MATCH_LEVEL in df.columns
        else empty_quoted
    )
    discrepancy_expr = (
        _parse_nonnegative_float_expr(COORDINATE_DISCREPANCY_KM)
        if COORDINATE_DISCREPANCY_KM in df.columns
        else "CAST(NULL AS DOUBLE)"
    )

    query = f"""
        SELECT
            rbd,
            program_code,
            compact_program_name_udf(nom_programme_reconstruit) AS "{PROGRAM_DISPLAY_NAME}",
            compact_school_name_udf(nom_lycee) AS "{SCHOOL_NAME}",
            {school_commune_expr} AS "{SCHOOL_COMMUNE}",
            _coalesced_latitude AS "{PROGRAM_LATITUDE}",
            _coalesced_longitude AS "{PROGRAM_LONGITUDE}",
            _coalesced_source AS "{PROGRAM_COORDINATE_SOURCE}",
            {geo_match_level_expr} AS "{PROGRAM_GEO_MATCH_LEVEL}",
            {discrepancy_expr} AS "{COORDINATE_DISCREPANCY_KM}",
            _rbd_spread_km AS "{RBD_COORDINATE_SPREAD_KM}",
            {", ".join(criteria_select)}
        FROM _names_src
    """
    out = db.as_object_dtype(con.sql(query).df())
    con.unregister("_names_src")

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
    con = db.connect()
    try:
        out = db.left_join_preserving_order(con, out, names, on=["rbd", "program_code"])
    finally:
        con.close()
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


def _numeric_cast_expr(column: str) -> str:
    """SQL expression mirroring pandas.to_numeric(..., errors='coerce') (no comma tolerance)."""
    return f'TRY_CAST(trim("{column}") AS DOUBLE)'


def validate_cumulative_share_columns(
    calib: pd.DataFrame,
    *,
    tolerance: float = 1e-6,
) -> list[str]:
    """Validate the full priority-share distribution for every program.

    Checks include missing/non-numeric values, [0, 1] bounds,
    before + share == through, continuity between adjacent tiers, a zero start,
    and a final cumulative share of one. Every count/max is computed in a
    single SQL aggregate query; only English message composition stays here.
    """
    available_tiers = [
        tier for tier in TIERS
        if all(
            col in calib.columns
            for col in (
                f"priority_share_{tier}_2024",
                f"cum_share_before_{tier}_2024",
                f"cum_share_through_{tier}_2024",
            )
        )
    ]
    if not available_tiers:
        return []

    con = db.connect()
    con.register("_share_src", calib)

    aggregates: list[str] = []
    for tier in available_tiers:
        share = _numeric_cast_expr(f"priority_share_{tier}_2024")
        before = _numeric_cast_expr(f"cum_share_before_{tier}_2024")
        through = _numeric_cast_expr(f"cum_share_through_{tier}_2024")
        valid = f"({share} IS NOT NULL AND {before} IS NOT NULL AND {through} IS NOT NULL)"
        out_of_range = (
            f"({share} < {-tolerance} OR {share} > {1 + tolerance} "
            f"OR {before} < {-tolerance} OR {before} > {1 + tolerance} "
            f"OR {through} < {-tolerance} OR {through} > {1 + tolerance})"
        )
        diff = f"ABS(({before} + {share}) - {through})"
        aggregates += [
            f'COUNT(*) FILTER (WHERE {share} IS NULL OR {before} IS NULL OR {through} IS NULL) AS "{tier}__missing"',
            f'COUNT(*) FILTER (WHERE {valid} AND {out_of_range}) AS "{tier}__out_of_range"',
            f'COUNT(*) FILTER (WHERE {valid} AND {diff} > {tolerance}) AS "{tier}__inconsistent"',
            f'MAX({diff}) FILTER (WHERE {valid} AND {diff} > {tolerance}) AS "{tier}__inconsistent_max_diff"',
        ]

    first_tier = available_tiers[0] if available_tiers[0] == TIERS[0] else None
    if first_tier:
        first_before = _numeric_cast_expr(f"cum_share_before_{first_tier}_2024")
        aggregates.append(
            f'COUNT(*) FILTER (WHERE {first_before} IS NOT NULL AND ABS({first_before}) > {tolerance}) '
            f'AS "start__invalid"'
        )

    adjacency_pairs = [
        (previous_tier, current_tier)
        for previous_tier, current_tier in zip(TIERS, TIERS[1:])
        if previous_tier in available_tiers and current_tier in available_tiers
    ]
    for previous_tier, current_tier in adjacency_pairs:
        previous_through = _numeric_cast_expr(f"cum_share_through_{previous_tier}_2024")
        current_before = _numeric_cast_expr(f"cum_share_before_{current_tier}_2024")
        both_valid = f"({previous_through} IS NOT NULL AND {current_before} IS NOT NULL)"
        gap = f"ABS({previous_through} - {current_before})"
        key = f"{previous_tier}__{current_tier}"
        aggregates += [
            f'COUNT(*) FILTER (WHERE {both_valid} AND {gap} > {tolerance}) AS "{key}__discontinuous"',
            f'MAX({gap}) FILTER (WHERE {both_valid} AND {gap} > {tolerance}) AS "{key}__discontinuous_max_gap"',
        ]

    last_tier = available_tiers[-1] if available_tiers[-1] == TIERS[-1] else None
    if last_tier:
        final_through = _numeric_cast_expr(f"cum_share_through_{last_tier}_2024")
        aggregates.append(
            f'COUNT(*) FILTER (WHERE {final_through} IS NOT NULL AND ABS({final_through} - 1.0) > {tolerance}) '
            f'AS "end__invalid"'
        )

    row = con.sql(f"SELECT {', '.join(aggregates)} FROM _share_src").df().iloc[0]
    con.unregister("_share_src")

    errors: list[str] = []
    for tier in available_tiers:
        share_col = f"priority_share_{tier}_2024"
        before_col = f"cum_share_before_{tier}_2024"
        through_col = f"cum_share_through_{tier}_2024"

        missing = int(row[f"{tier}__missing"])
        if missing:
            errors.append(
                f"{tier}: {missing} row(s) with missing or "
                f"non-numeric calibration share values in {share_col}, "
                f"{before_col}, or {through_col}."
            )

        out_of_range = int(row[f"{tier}__out_of_range"])
        if out_of_range:
            errors.append(
                f"{tier}: {out_of_range} row(s) with calibration "
                "shares or cumulative shares outside [0, 1]."
            )

        inconsistent = int(row[f"{tier}__inconsistent"])
        if inconsistent:
            errors.append(
                f"{through_col}: {inconsistent} row(s) where "
                "cum_share_before + priority_share differs from "
                f"cum_share_through (max diff {float(row[f'{tier}__inconsistent_max_diff']):.6g})."
            )

    if first_tier and int(row["start__invalid"]):
        errors.append(
            f"{int(row['start__invalid'])} row(s) where the first priority "
            "tier does not start at cumulative share 0."
        )

    for previous_tier, current_tier in adjacency_pairs:
        key = f"{previous_tier}__{current_tier}"
        discontinuous = int(row[f"{key}__discontinuous"])
        if discontinuous:
            errors.append(
                f"{previous_tier} -> {current_tier}: "
                f"{discontinuous} row(s) with a discontinuity between "
                "the previous cumulative end and the next cumulative start "
                f"(max gap {float(row[f'{key}__discontinuous_max_gap']):.6g})."
            )

    if last_tier and int(row["end__invalid"]):
        errors.append(
            f"{int(row['end__invalid'])} row(s) where the final cumulative "
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
    integer_columns = [
        col for col in (CAPACITY, TRUE_APP, POP, TOTAL_CAPACITY_COLUMN, *SEAT_COMPONENT_COLUMNS)
        if col in calib.columns
    ]
    if not integer_columns:
        return []

    con = db.connect()
    con.register("_numeric_src", calib)

    exprs = {col: _numeric_cast_expr(col) for col in integer_columns}
    aggregates: list[str] = []
    for col in integer_columns:
        v = exprs[col]
        minimum = "0" if col == POP else "-1e18"
        # POP must be strictly positive (<=0 invalid); every other column is
        # non-negative (<0 invalid). Using a very low bound plus `< 0` filter
        # keeps a single template while preserving each column's own rule.
        minimum_invalid = f"{v} <= 0" if col == POP else f"{v} < 0"
        aggregates += [
            f'COUNT(*) FILTER (WHERE {v} IS NULL) AS "{col}__invalid_number"',
            f'COUNT(*) FILTER (WHERE {v} IS NOT NULL AND ({minimum_invalid})) AS "{col}__minimum_invalid"',
            f'COUNT(*) FILTER (WHERE {v} IS NOT NULL AND ABS({v} - ROUND({v})) > {tolerance}) '
            f'AS "{col}__non_integer"',
        ]

    def _pair_valid(col_a: str, col_b: str) -> str:
        return f"({exprs[col_a]} IS NOT NULL AND {exprs[col_b]} IS NOT NULL)"

    has_true_app_pop = TRUE_APP in integer_columns and POP in integer_columns
    if has_true_app_pop:
        aggregates.append(
            f'COUNT(*) FILTER (WHERE {_pair_valid(TRUE_APP, POP)} '
            f'AND {exprs[TRUE_APP]} > {exprs[POP]} + {tolerance}) AS "true_app_exceeds_pop"'
        )

    has_priority_seats = CAPACITY in integer_columns and PRIORITY_STUDENT_SEATS in integer_columns
    if has_priority_seats:
        aggregates.append(
            f'COUNT(*) FILTER (WHERE {_pair_valid(CAPACITY, PRIORITY_STUDENT_SEATS)} '
            f'AND {exprs[PRIORITY_STUDENT_SEATS]} > {exprs[CAPACITY]} + {tolerance}) '
            f'AS "priority_seats_exceed_capacity"'
        )

    has_total_capacity = CAPACITY in integer_columns and TOTAL_CAPACITY_COLUMN in integer_columns
    if has_total_capacity:
        aggregates.append(
            f'COUNT(*) FILTER (WHERE {_pair_valid(CAPACITY, TOTAL_CAPACITY_COLUMN)} '
            f'AND {exprs[CAPACITY]} > {exprs[TOTAL_CAPACITY_COLUMN]} + {tolerance}) '
            f'AS "capacity_exceeds_total_capacity"'
        )

    has_seat_components = CAPACITY in integer_columns and all(col in integer_columns for col in SEAT_COMPONENT_COLUMNS)
    if has_seat_components:
        components_sum = " + ".join(exprs[col] for col in SEAT_COMPONENT_COLUMNS)
        components_valid = " AND ".join(f"{exprs[col]} IS NOT NULL" for col in SEAT_COMPONENT_COLUMNS)
        seat_diff = f"ABS(({components_sum}) - {exprs[CAPACITY]})"
        aggregates += [
            f'COUNT(*) FILTER (WHERE {exprs[CAPACITY]} IS NOT NULL AND {components_valid} '
            f'AND {seat_diff} > {tolerance}) AS "seat_components_mismatch"',
            f'MAX({seat_diff}) FILTER (WHERE {exprs[CAPACITY]} IS NOT NULL AND {components_valid} '
            f'AND {seat_diff} > {tolerance}) AS "seat_components_mismatch_max_diff"',
        ]

    row = con.sql(f"SELECT {', '.join(aggregates)} FROM _numeric_src").df().iloc[0]
    con.unregister("_numeric_src")

    errors: list[str] = []
    for col in integer_columns:
        invalid_number = int(row[f"{col}__invalid_number"])
        if invalid_number:
            errors.append(f"{col}: {invalid_number} row(s) with missing or non-numeric values.")

        minimum_invalid = int(row[f"{col}__minimum_invalid"])
        if minimum_invalid:
            requirement = "positive" if col == POP else "non-negative"
            errors.append(f"{col}: {minimum_invalid} row(s) that are not {requirement}.")

        non_integer = int(row[f"{col}__non_integer"])
        if non_integer:
            errors.append(f"{col}: {non_integer} row(s) with non-integer count values.")

    if has_true_app_pop and int(row["true_app_exceeds_pop"]):
        errors.append(
            f"{TRUE_APP}: {int(row['true_app_exceeds_pop'])} row(s) where true applicants exceed {POP}."
        )

    if has_priority_seats and int(row["priority_seats_exceed_capacity"]):
        errors.append(
            f"{PRIORITY_STUDENT_SEATS}: {int(row['priority_seats_exceed_capacity'])} row(s) exceed {CAPACITY}."
        )

    if has_total_capacity and int(row["capacity_exceeds_total_capacity"]):
        errors.append(
            f"{CAPACITY}: {int(row['capacity_exceeds_total_capacity'])} row(s) where admission seats "
            f"exceed {TOTAL_CAPACITY_COLUMN}."
        )

    if has_seat_components and int(row["seat_components_mismatch"]):
        errors.append(
            f"{CAPACITY}: {int(row['seat_components_mismatch'])} row(s) where seat components "
            "do not sum to total admission seats "
            f"(max diff {float(row['seat_components_mismatch_max_diff']):.6g})."
        )

    return errors
