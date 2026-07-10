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
    PROGRAM_RECONSTRUCTED_NAME,
    PROGRAM_RELIGIOUS_DETAIL,
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
    TRUE_APP,
)
from sae_app.text_utils import clean_optional_value, clean_text, norm_code, parse_coordinate

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


# ---------------------------------------------------------------------------
# Region lookup
# ---------------------------------------------------------------------------

def region_sort_index(region: str) -> int:
    try:
        return REGION_ORDER.index(str(region).strip())
    except ValueError:
        return len(REGION_ORDER)


@st.cache_data(show_spinner=False)
def load_rbd_region_map() -> pd.DataFrame:
    df = read_csv_path(RBD_REGION_PATH, sep=",")

    required = {"rbd", REGION}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{RBD_REGION_PATH.name} is missing columns: " + ", ".join(sorted(missing)))

    out = df[["rbd", REGION]].copy()
    out["rbd"] = norm_code(out["rbd"])
    out[REGION] = out[REGION].astype(str).str.strip()
    out = out.drop_duplicates("rbd")
    return out


def attach_regions(calib: pd.DataFrame) -> pd.DataFrame:
    """Attach region labels loaded from data/rbd_region_map.csv.

    This keeps every program from the capacities file. If an RBD is not found in
    the lookup, it is still available under Unknown region.
    """
    out = calib.copy()
    out["rbd"] = norm_code(out["rbd"])

    regions = load_rbd_region_map()
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

@st.cache_data(show_spinner=False)
def load_program_filters() -> pd.DataFrame:
    df = read_csv_path(PROGRAM_FILTERS_PATH, sep=",")

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
        raise ValueError(f"{PROGRAM_FILTERS_PATH.name} is missing columns: " + ", ".join(sorted(missing)))

    out = df[list(required)].copy()
    out["rbd"] = norm_code(out["rbd"])
    out["program_code"] = norm_code(out["program_code"])
    out = out.drop_duplicates(["rbd", "program_code"])
    return out


def attach_program_filters(calib: pd.DataFrame) -> pd.DataFrame:
    """Attach program characteristics used by the sidebar filters.

    The capacities/calibration file remains the source of the available programs.
    The metadata file only adds filtering fields. Programs with missing metadata
    remain available when no filter is active.
    """
    out = calib.copy()
    out["rbd"] = norm_code(out["rbd"])
    out["program_code"] = norm_code(out["program_code"])

    filters = load_program_filters()
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


@st.cache_data(show_spinner=False)
def load_program_names(file_bytes: bytes) -> pd.DataFrame:
    df = read_csv(file_bytes, sep="auto")
    df.columns = [str(c).lstrip("﻿").strip() for c in df.columns]

    required = {"rbd", "program_code", "nom_programme_reconstruit", "nom_lycee"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{PROGRAM_NAMES_PATH.name} is missing columns: " + ", ".join(sorted(missing)))

    optional_source_cols = [
        "commune",
        "ruralite",
        "convenio_pie",
        "pace",
        "paiement_matricula",
        "paiement_mensualite",
        "orientation_religieuse",
        "orientation_religieuse_autre_detail",
    ]
    lat_source_col = first_existing_column(df, [
        "program_latitude", "school_latitude", "lat_lycee",
        "latitude", "lat", "latitud", "commune_latitude",
        "establecimiento_latitude",
    ])
    lon_source_col = first_existing_column(df, [
        "program_longitude", "school_longitude", "lon_lycee",
        "longitude", "lon", "lng", "longitud", "commune_longitude",
        "establecimiento_longitude",
    ])

    keep_cols = ["rbd", "program_code", "nom_programme_reconstruit", "nom_lycee"]
    keep_cols += [c for c in optional_source_cols if c in df.columns]
    keep_cols += [c for c in [lat_source_col, lon_source_col] if c and c not in keep_cols]

    out = df[keep_cols].copy()
    out["rbd"] = norm_code(out["rbd"])
    out["program_code"] = norm_code(out["program_code"])
    out[PROGRAM_RECONSTRUCTED_NAME] = out["nom_programme_reconstruit"].astype(str).str.strip()
    out[PROGRAM_DISPLAY_NAME] = out[PROGRAM_RECONSTRUCTED_NAME].map(compact_program_name)
    out[SCHOOL_NAME] = out["nom_lycee"].map(compact_school_name)
    out[SCHOOL_COMMUNE] = (
        out["commune"].astype(str).str.strip().str.title()
        if "commune" in out.columns else ""
    )
    out[PROGRAM_LATITUDE] = out[lat_source_col].map(parse_coordinate) if lat_source_col else np.nan
    out[PROGRAM_LONGITUDE] = out[lon_source_col].map(parse_coordinate) if lon_source_col else np.nan

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

    if "orientation_religieuse_autre_detail" in out.columns:
        out[PROGRAM_RELIGIOUS_DETAIL] = out["orientation_religieuse_autre_detail"].map(lambda x: clean_optional_value(x, default=""))
    else:
        out[PROGRAM_RELIGIOUS_DETAIL] = ""

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
        "orientation_religieuse_autre_detail",
    ]
    source_cols_to_drop += [
        c for c in [lat_source_col, lon_source_col]
        if c and c not in {PROGRAM_LATITUDE, PROGRAM_LONGITUDE}
    ]
    out = out.drop(columns=[c for c in source_cols_to_drop if c in out.columns])
    out = out.drop_duplicates(["rbd", "program_code"])
    return out


def attach_program_names(calib: pd.DataFrame) -> pd.DataFrame:
    """Attach reconstructed program names, real school names, and additional choice criteria."""
    out = calib.copy()
    out["rbd"] = norm_code(out["rbd"])
    out["program_code"] = norm_code(out["program_code"])

    names = load_program_names(PROGRAM_NAMES_PATH.read_bytes())
    out = out.merge(names, on=["rbd", "program_code"], how="left")
    out[PROGRAM_RECONSTRUCTED_NAME] = out[PROGRAM_RECONSTRUCTED_NAME].fillna("")
    out[PROGRAM_DISPLAY_NAME] = out[PROGRAM_DISPLAY_NAME].fillna(UNKNOWN_PROGRAM_NAME)
    out[SCHOOL_NAME] = out[SCHOOL_NAME].fillna("")
    out[SCHOOL_COMMUNE] = out[SCHOOL_COMMUNE].fillna("")
    if PROGRAM_LATITUDE not in out.columns:
        out[PROGRAM_LATITUDE] = np.nan
    if PROGRAM_LONGITUDE not in out.columns:
        out[PROGRAM_LONGITUDE] = np.nan

    for col in [
        PROGRAM_RURALITY,
        PROGRAM_PIE,
        PROGRAM_PACE,
        PROGRAM_ENROLLMENT_FEE,
        PROGRAM_MONTHLY_FEE,
        PROGRAM_RELIGIOUS_ORIENTATION,
    ]:
        out[col] = out[col].fillna("No information")
    out[PROGRAM_RELIGIOUS_DETAIL] = out[PROGRAM_RELIGIOUS_DETAIL].fillna("")
    return out


# ---------------------------------------------------------------------------
# Calibration file loading and validation
# (data/capacities_2025_wta_with_2024_calibration.csv)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_calibration(file_bytes: bytes) -> pd.DataFrame:
    df = read_csv(file_bytes, sep=";")
    if len(df.columns) == 1:
        df = read_csv(file_bytes, sep=",")
    df["program_code"] = norm_code(df["program_code"])
    df["rbd"] = norm_code(df["rbd"])
    df = attach_regions(df)
    df = attach_program_filters(df)
    df = attach_program_names(df)
    return df


def required_cols() -> list[str]:
    cols = ["rbd", "program_code", CAPACITY, TRUE_APP, POP]
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
    """Check before + share == through for every priority tier.

    The calculation itself uses cum_share_before_* and priority_share_*.
    cum_share_through_* is kept as a calibration integrity check so a malformed
    calibration file is caught before the app silently computes with
    inconsistent cumulative shares.
    """
    errors: list[str] = []

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
        expected_through = before + share

        missing_or_invalid = share.isna() | before.isna() | through.isna()
        if missing_or_invalid.any():
            errors.append(
                f"{tier}: {int(missing_or_invalid.sum())} row(s) with missing or "
                f"non-numeric calibration share values in {share_col}, "
                f"{before_col}, or {through_col}."
            )

        valid = ~missing_or_invalid
        diff = (expected_through - through).abs()
        inconsistent = valid & (diff > tolerance)

        if inconsistent.any():
            errors.append(
                f"{through_col}: {int(inconsistent.sum())} row(s) where "
                f"cum_share_before + priority_share differs from cum_share_through "
                f"(max diff {float(diff[inconsistent].max()):.6g})."
            )

    return errors


def validate_core_numeric_columns(calib: pd.DataFrame) -> list[str]:
    """Check core numeric calibration fields before any simulation is run.

    Missing capacity or true-applicant values would otherwise be converted to
    0.0 by as_float() inside the calculation engine. That fallback is useful
    for defensive parsing, but malformed calibration data should be reported at
    startup instead of silently producing optimistic or pessimistic estimates.
    """
    checks = [
        (CAPACITY, "missing, non-numeric, or negative"),
        (TRUE_APP, "missing, non-numeric, or negative"),
    ]
    errors: list[str] = []

    for col, description in checks:
        if col not in calib.columns:
            # Missing columns are reported by required_cols() in app.py.
            continue

        values = pd.to_numeric(calib[col], errors="coerce")
        invalid = values.isna() | (values < 0)
        if invalid.any():
            errors.append(f"{col}: {int(invalid.sum())} row(s) with {description} values.")

    return errors
