from __future__ import annotations

import hashlib
import io
import math
import re
from itertools import permutations, product
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import hypergeom

# ---------------------------------------------------------------------------
# Data columns
# ---------------------------------------------------------------------------
WISH_RANK    = "wish_rank"
PROGRAM      = "program"
EQUIV_GROUP  = "preference_group"
LOTTERY      = "lottery_number"
HASH_INPUT   = "lottery_hash_input"
HASH_HEX     = "lottery_hash_hex"
HASH_PCT     = "lottery_hash_percentile"

CAPACITY     = "total_admission_seats"
TRUE_APP     = "true_applicants_last_year"
POP          = "program_lottery_population_2024"
IMPUTED      = "calibration_2024_imputed"
IMPUT_METHOD = "calibration_2024_imputation_method"

PRIORITY_STUDENT_QUOTA = 0.15   # 15% reserved for priority students
DEFAULT_THRESHOLD_MTB  = 0.025
MAX_EXACT_EQUIV_PERMUTATIONS = 10000

PRIORITIES = [
    "priority_sibling",
    "priority_student",
    "priority_parent_civil_servant",
    "priority_ex_student",
]
SAFETY     = "priority_already_registered"
NO_PRIORITY = "no_priority"
TIERS       = PRIORITIES + [NO_PRIORITY]

MAX_SHA256 = 2 ** 256 - 1

REGION = "Region"
UNKNOWN_REGION = "Unknown region"

# ---------------------------------------------------------------------------
# External data files
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CAPACITIES_PATH = DATA_DIR / "capacities_2025_wta_with_2024_calibration.csv"
PROGRAM_NAMES_PATH = DATA_DIR / "programmes_chili_criteres_recommandation.csv"
RBD_REGION_PATH = DATA_DIR / "rbd_region_map.csv"
PROGRAM_FILTERS_PATH = DATA_DIR / "program_filters.csv"

# Embedded RBD -> Region lookup built from the 2025 individual-level preferences file.
# This lets the app sort the program dropdown by region without asking users to upload
# the large individual-level file.
REGION_ORDER = [
    "Región de Arica y Parinacota",
    "Región de Tarapacá",
    "Región de Antofagasta",
    "Región de Atacama",
    "Región de Coquimbo",
    "Región de Valparaíso",
    "Región Metropolitana de Santiago",
    "Región del Libertador Bernardo O'Higgins",
    "Región del Maule",
    "Región de Ñuble",
    "Región del Bío-Bío",
    "Región de La Araucanía",
    "Región de Los Ríos",
    "Región de Los Lagos",
    "Región de Aysén del Gral.Ibañez del Campo",
    "Región de Magallanes y Antártica Chilena",
    UNKNOWN_REGION,
]

# RBD -> Region lookup is loaded from data/rbd_region_map.csv.


# Embedded program-characteristic lookup built from chile_programs_sorted_by_specialty.csv.
# Key: "rbd|program_code". Values: (track, specialty sector, specialty name, gender, school day).
PROGRAM_TRACK = "program_track"
PROGRAM_SPECIALTY_SECTOR = "program_specialty_sector"
PROGRAM_SPECIALTY_NAME = "program_specialty_name"
PROGRAM_GENDER = "program_gender"
PROGRAM_SCHOOL_DAY = "program_school_day"
UNKNOWN_FILTER_VALUE = "Unknown"

PROGRAM_RECONSTRUCTED_NAME = "program_reconstructed_name"
PROGRAM_DISPLAY_NAME = "program_display_name"
SCHOOL_NAME = "school_name"
SCHOOL_COMMUNE = "school_commune"
UNKNOWN_PROGRAM_NAME = "Program details unavailable"
UNKNOWN_SCHOOL_NAME = "School name unavailable"

PROGRAM_RURALITY = "program_rurality"
PROGRAM_PIE = "program_pie"
PROGRAM_PACE = "program_pace"
PROGRAM_ENROLLMENT_FEE = "program_enrollment_fee"
PROGRAM_MONTHLY_FEE = "program_monthly_fee"
PROGRAM_RELIGIOUS_ORIENTATION = "program_religious_orientation"
PROGRAM_RELIGIOUS_DETAIL = "program_religious_detail"

TRACK_GENERAL = "General"
TRACK_SPECIALIZED = "Specialized"

SPECIALTY_FILTER_OPTIONS = [
    "Agriculture",
    "Metalworking and mechanics",
    "Electricity",
    "Food services",
    "Construction",
    "Technology and communications",
]
GENDER_FILTER_OPTIONS = ["Mixed", "Boys", "Girls"]
SCHOOL_DAY_FILTER_OPTIONS = ["Full day", "Morning", "Afternoon"]
RURALITY_FILTER_OPTIONS = ["Urban", "Rural"]
PIE_FILTER_OPTIONS = ["With PIE", "Without PIE"]
PACE_FILTER_OPTIONS = ["With PACE", "Without PACE"]
PAYMENT_FILTER_OPTIONS = [
    "Free",
    "$1,000–$10,000",
    "$10,001–$25,000",
    "$25,001–$50,000",
    "$50,001–$100,000",
    "More than $100,000",
    "No information",
]
RELIGIOUS_FILTER_OPTIONS = ["Secular", "Catholic", "Evangelical", "Other", "No information"]

# Program-characteristic lookup is loaded from data/program_filters.csv.



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


def program_matches_filters(row: pd.Series, filters: dict | None) -> bool:
    """Return True if a program row satisfies the sidebar filters.

    Empty filters mean no restriction. Selected existing wishes are preserved
    separately in filter_program_options().
    """
    if not filters:
        return True

    selected_tracks = filters.get("tracks") or []
    selected_specialties = filters.get("specialty_sectors") or []
    selected_genders = filters.get("genders") or []
    selected_school_days = filters.get("school_days") or []
    selected_rurality = filters.get("rurality") or []
    selected_pie = filters.get("pie") or []
    selected_pace = filters.get("pace") or []
    selected_enrollment_fee = filters.get("enrollment_fee") or []
    selected_monthly_fee = filters.get("monthly_fee") or []
    selected_religious_orientation = filters.get("religious_orientation") or []

    track = str(row.get(PROGRAM_TRACK, UNKNOWN_FILTER_VALUE)).strip()
    specialty_sector = str(row.get(PROGRAM_SPECIALTY_SECTOR, UNKNOWN_FILTER_VALUE)).strip()
    gender = str(row.get(PROGRAM_GENDER, UNKNOWN_FILTER_VALUE)).strip()
    school_day = str(row.get(PROGRAM_SCHOOL_DAY, UNKNOWN_FILTER_VALUE)).strip()
    rurality = str(row.get(PROGRAM_RURALITY, UNKNOWN_FILTER_VALUE)).strip()
    pie = str(row.get(PROGRAM_PIE, UNKNOWN_FILTER_VALUE)).strip()
    pace = str(row.get(PROGRAM_PACE, UNKNOWN_FILTER_VALUE)).strip()
    enrollment_fee = str(row.get(PROGRAM_ENROLLMENT_FEE, UNKNOWN_FILTER_VALUE)).strip()
    monthly_fee = str(row.get(PROGRAM_MONTHLY_FEE, UNKNOWN_FILTER_VALUE)).strip()
    religious_orientation = str(row.get(PROGRAM_RELIGIOUS_ORIENTATION, UNKNOWN_FILTER_VALUE)).strip()

    if selected_tracks and track not in selected_tracks:
        return False

    if selected_specialties:
        if track != TRACK_SPECIALIZED or specialty_sector not in selected_specialties:
            return False

    if selected_genders and gender not in selected_genders:
        return False

    if selected_school_days and school_day not in selected_school_days:
        return False

    if selected_rurality and rurality not in selected_rurality:
        return False

    if selected_pie and pie not in selected_pie:
        return False

    if selected_pace and pace not in selected_pace:
        return False

    if selected_enrollment_fee and enrollment_fee not in selected_enrollment_fee:
        return False

    if selected_monthly_fee and monthly_fee not in selected_monthly_fee:
        return False

    if selected_religious_orientation and religious_orientation not in selected_religious_orientation:
        return False

    return True


def filters_are_active(filters: dict | None) -> bool:
    if not filters:
        return False
    return any(bool(filters.get(k)) for k in [
        "tracks",
        "specialty_sectors",
        "genders",
        "school_days",
        "rurality",
        "pie",
        "pace",
        "enrollment_fee",
        "monthly_fee",
        "religious_orientation",
    ])


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


def norm_code_value(x) -> str:
    x = str(x).strip()
    if x.startswith('="') and x.endswith('"'):
        x = x[2:-1].strip()
    try:
        return str(int(float(x.replace(",", "."))))
    except Exception:
        return x


def norm_code(s: pd.Series) -> pd.Series:
    return s.map(norm_code_value)


def as_bool(x) -> bool:
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"1", "true", "yes", "y", "x", "oui"}


def as_float(x, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(str(x).strip().replace(",", "."))
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Hash MTB (SHA-256 RUN/IPE + RBD)
# ---------------------------------------------------------------------------

def normalize_run(student_id: str) -> str:
    """Normalize a Chilean RUN/IPE before hashing.

    Removes dots and spaces, uppercases K, and keeps the hyphen.
    Raises ValueError if the identifier is empty or contains invalid characters.
    """
    cleaned = str(student_id).strip().upper().replace(".", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    if not cleaned:
        raise ValueError("Enter the student RUN/IPE before running the MTB calculation.")
    if not re.fullmatch(r"[0-9K\-]+", cleaned):
        raise ValueError(
            "The RUN/IPE may contain only digits, one optional hyphen, and the letter K."
        )
    return cleaned


def mtb_hash(student_id: str, rbd) -> dict:
    """Compute the deterministic lottery percentile for a (student, school) pair.

    SHA-256 returns a value between 0 and MAX_SHA256.
    The official priority direction is larger = better; it is converted into a
    0-best/1-worst percentile to match the model convention.

    Returns a dict with HASH_INPUT, HASH_HEX, HASH_PCT, and priority_percentile.
    """
    norm_id  = normalize_run(student_id)
    norm_rbd = norm_code_value(rbd)
    hash_input = f"{norm_id}{norm_rbd}"
    hex_digest  = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
    decimal     = int(hex_digest, 16)

    priority_pct = decimal / MAX_SHA256          # 1 = best
    lottery_pct  = 1.0 - priority_pct            # 0 = best

    return {
        HASH_INPUT: hash_input,
        HASH_HEX:   hex_digest,
        HASH_PCT:   float(np.clip(lottery_pct, 0, 1)),
        "priority_percentile": float(np.clip(priority_pct, 0, 1)),
    }


def pct_to_rank(percentile: float, n: int) -> int:
    """Convert a 0-best/1-worst percentile into an integer rank among n candidates."""
    n = max(int(n), 1)
    return int(1 + np.floor(np.clip(percentile, 0, 1) * max(n - 1, 0)))


def attach_mtb_hashes(
    wishes: pd.DataFrame,
    mapping: dict[str, pd.Series],
    student_id: str,
) -> pd.DataFrame:
    """Compute and attach the MTB percentile to each valid wish."""
    out = wishes.copy()
    for col in (HASH_INPUT, HASH_HEX, HASH_PCT):
        if col not in out.columns:
            out[col] = np.nan if col == HASH_PCT else ""

    for idx, wish in out.iterrows():
        label = str(wish.get(PROGRAM, "")).strip()
        if not label or label not in mapping:
            continue
        program    = mapping[label]
        population = max(round(as_float(program[POP])), 1)
        h          = mtb_hash(student_id, program["rbd"])

        out.at[idx, HASH_INPUT] = h[HASH_INPUT]
        out.at[idx, HASH_HEX]   = h[HASH_HEX]
        out.at[idx, HASH_PCT]   = h[HASH_PCT]
        # Theory-consistent equivalent lottery rank within the program-level
        # reference population N_s = program_lottery_population_2024.
        out.at[idx, LOTTERY]    = pct_to_rank(h[HASH_PCT], population)

    return out


# ---------------------------------------------------------------------------
# Capacities + 2024 calibration file
# ---------------------------------------------------------------------------
# Loaded from data/capacities_2025_wta_with_2024_calibration.csv.


# Program-name, school-name, and recommendation-criteria lookup is loaded from
# data/programmes_chili_criteres_recommandation.csv.

def compact_program_name(name: str) -> str:
    """Convert the reconstructed program name into a short English label.

    The source file contains concise reconstructed names such as
    "1º medio — Général H-C — Mixte — jornada completa". The dropdown is
    easier to scan if the repeated grade is removed and the stable descriptors
    are translated.
    """
    text = str(name or "").strip()
    if not text:
        return UNKNOWN_PROGRAM_NAME

    replacements = {
        "1º medio": "1st grade secondary",
        "Général H-C": "General H-C",
        "Spécialité TP": "Technical-vocational",
        "Mixte": "Mixed",
        "garçons": "Boys",
        "filles": "Girls",
        "jornada completa": "Full day",
        "jornada mañana": "Morning",
        "jornada tarde": "Afternoon",
    }
    for old_value, new_value in replacements.items():
        text = text.replace(old_value, new_value)

    parts = [part.strip() for part in text.split("—") if part.strip()]
    if parts and parts[0].lower().startswith("1st grade"):
        parts = parts[1:]

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


def clean_optional_value(value, *, default: str = "No information") -> str:
    text = " ".join(str(value or "").strip().split())
    if not text or text.lower() == "nan":
        return default
    return text


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
    keep_cols = ["rbd", "program_code", "nom_programme_reconstruit", "nom_lycee"]
    keep_cols += [c for c in optional_source_cols if c in df.columns]

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


# ---------------------------------------------------------------------------
# Build program options
# ---------------------------------------------------------------------------

def make_program_option_label(row: pd.Series, duplicate_count: int = 1) -> str:
    """Build a readable but still uniquely identifiable dropdown label."""
    rbd = str(row["rbd"]).strip()
    code = str(row["program_code"]).strip()
    school_name = str(row.get(SCHOOL_NAME, "")).strip()
    commune = str(row.get(SCHOOL_COMMUNE, "")).strip()
    display_name = str(row.get(PROGRAM_DISPLAY_NAME, "")).strip()

    if not school_name or school_name == UNKNOWN_SCHOOL_NAME:
        school_part = f"RBD {rbd}"
    elif commune and commune.lower() != "nan":
        school_part = f"{school_name} ({commune})"
    else:
        school_part = school_name

    if not display_name or display_name == UNKNOWN_PROGRAM_NAME:
        display_name = f"Program code {code}"

    label = f"{school_part} — {display_name} · RBD {rbd}"
    if duplicate_count > 1:
        label = f"{label} · code {code}"
    return label


def build_options(calib: pd.DataFrame) -> tuple[list[str], dict[str, pd.Series]]:
    options, mapping = [], {}

    unique_programs = calib.drop_duplicates(["rbd", "program_code"]).copy()
    unique_programs["_region_sort"] = unique_programs[REGION].map(region_sort_index)
    unique_programs["_rbd_sort"] = pd.to_numeric(unique_programs["rbd"], errors="coerce")
    unique_programs["_program_sort"] = pd.to_numeric(unique_programs["program_code"], errors="coerce")

    # A few schools can have multiple distinct program codes with the same readable
    # reconstructed name. In those cases only, append the code to keep labels unique.
    unique_programs["_base_display_label"] = unique_programs.apply(
        lambda row: make_program_option_label(row, duplicate_count=1),
        axis=1,
    )
    duplicate_counts = unique_programs["_base_display_label"].value_counts().to_dict()

    unique_programs = unique_programs.sort_values(
        ["_region_sort", "_rbd_sort", "_program_sort", REGION, "rbd", "program_code"]
    )

    for _, row in unique_programs.iterrows():
        base_label = row["_base_display_label"]
        label = make_program_option_label(row, duplicate_counts.get(base_label, 1))
        options.append(label)
        mapping[label] = row

    return options, mapping

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


def filter_program_options(
    program_mapping: dict[str, pd.Series],
    selected_region: str,
    active_filters: dict | None = None,
    current_values: list[str] | None = None,
) -> list[str]:
    """Filter program options by region and characteristics while preserving existing values."""
    options = []
    for label, row in program_mapping.items():
        if selected_region != "All regions" and str(row.get(REGION, UNKNOWN_REGION)).strip() != selected_region:
            continue
        if not program_matches_filters(row, active_filters):
            continue
        options.append(label)

    for value in current_values or []:
        value = str(value).strip()
        if value and value in program_mapping and value not in options:
            options.append(value)

    return options


# ---------------------------------------------------------------------------
# Wish list handling (empty table + CSV import)
# ---------------------------------------------------------------------------

def empty_wishes() -> pd.DataFrame:
    df = pd.DataFrame({
        WISH_RANK: [1, 2, 3],
        EQUIV_GROUP: [1, 2, 3],
        PROGRAM: ["", "", ""],
        LOTTERY: [1, 1, 1],
    })
    for col in PRIORITIES + [SAFETY]:
        df[col] = False
    return df


def clean_wish_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only meaningful wish rows, preserve optional equivalence groups,
    then pad back to 3 default rows.
    """
    out = df.copy()

    for col in [WISH_RANK, EQUIV_GROUP, PROGRAM, LOTTERY] + PRIORITIES + [SAFETY]:
        if col not in out.columns:
            if col in PRIORITIES + [SAFETY]:
                out[col] = False
            elif col == LOTTERY:
                out[col] = 1
            elif col == EQUIV_GROUP:
                out[col] = np.nan
            else:
                out[col] = ""

    out[PROGRAM] = out[PROGRAM].fillna("").astype(str).str.strip()

    priority_cols = PRIORITIES + [SAFETY]
    for col in priority_cols:
        if col not in out.columns:
            out[col] = False
        out[col] = out[col].map(as_bool).fillna(False).astype(bool)

    has_priority = out[priority_cols].any(axis=1)
    has_program = out[PROGRAM] != ""

    out = out[has_program | has_priority].copy().reset_index(drop=True)

    out[WISH_RANK] = pd.to_numeric(out[WISH_RANK], errors="coerce")
    out[EQUIV_GROUP] = pd.to_numeric(out[EQUIV_GROUP], errors="coerce")

    if len(out) > 0:
        fallback = pd.Series(range(1, len(out) + 1), index=out.index)
        out[WISH_RANK] = out[WISH_RANK].where(out[WISH_RANK].notna(), fallback).astype(int)
        out[EQUIV_GROUP] = out[EQUIV_GROUP].where(out[EQUIV_GROUP].notna(), out[WISH_RANK]).astype(int)

    while len(out) < 3:
        next_rank = len(out) + 1
        new_row = {
            WISH_RANK: next_rank,
            EQUIV_GROUP: next_rank,
            PROGRAM: "",
            LOTTERY: 1,
        }
        for col in priority_cols:
            new_row[col] = False
        out = pd.concat([out, pd.DataFrame([new_row])], ignore_index=True)

    return out.reset_index(drop=True)

def parse_wishes(file_bytes: bytes, mapping: dict[str, pd.Series]) -> pd.DataFrame:
    df = read_csv(file_bytes, sep="auto")
    base_to_label = {
        f"{r['rbd']} || {r['program_code']}": label
        for label, r in mapping.items()
    }

    # Automatic column-format detection
    if {WISH_RANK, PROGRAM, LOTTERY}.issubset(df.columns):
        out = pd.DataFrame({WISH_RANK: df[WISH_RANK], PROGRAM: df[PROGRAM], LOTTERY: df[LOTTERY]})
    elif {WISH_RANK, PROGRAM}.issubset(df.columns):
        out = pd.DataFrame({WISH_RANK: df[WISH_RANK], PROGRAM: df[PROGRAM], LOTTERY: 1})
    elif {"rang_du_voeu", "programme", "numero_loterie"}.issubset(df.columns):
        out = pd.DataFrame({WISH_RANK: df["rang_du_voeu"], PROGRAM: df["programme"], LOTTERY: df["numero_loterie"]})
    elif {"rang_du_voeu", "programme"}.issubset(df.columns):
        out = pd.DataFrame({WISH_RANK: df["rang_du_voeu"], PROGRAM: df["programme"], LOTTERY: 1})
    elif {"rbd", "program_code", "preference_number"}.issubset(df.columns):
        labels = df["rbd"].astype(str).str.strip() + " || " + norm_code(df["program_code"])
        lottery_col = df["lottery"] if "lottery" in df.columns else 1
        out = pd.DataFrame({WISH_RANK: df["preference_number"], PROGRAM: labels, LOTTERY: lottery_col})
    else:
        raise ValueError(
            "Expected columns: wish_rank/program, rang_du_voeu/programme, "
            "or rbd/program_code/preference_number."
        )

    out[WISH_RANK] = pd.to_numeric(out[WISH_RANK], errors="coerce").fillna(1).astype(int)
    group_source = None
    for candidate in (EQUIV_GROUP, "equivalence_group", "equivalence_class", "preference_class"):
        if candidate in df.columns:
            group_source = df[candidate]
            break
    if group_source is not None:
        out[EQUIV_GROUP] = pd.to_numeric(group_source, errors="coerce").fillna(out[WISH_RANK]).astype(int)
    else:
        out[EQUIV_GROUP] = out[WISH_RANK]
    out[LOTTERY]   = pd.to_numeric(out[LOTTERY], errors="coerce").fillna(1).astype(int)
    out[PROGRAM]   = (
        out[PROGRAM].astype(str).str.strip()
        .map(lambda x: x if x in mapping else base_to_label.get(x, ""))
    )
    for col in PRIORITIES + [SAFETY]:
        out[col] = df[col].map(as_bool) if col in df.columns else False

    return out.sort_values(WISH_RANK).reset_index(drop=True) if not out.empty else empty_wishes()



# ---------------------------------------------------------------------------
# Similar-program recommendation engine
# ---------------------------------------------------------------------------

RECOMMENDATION_CRITERIA = [
    (PROGRAM_TRACK, "Program type", 1.00),
    (PROGRAM_SPECIALTY_SECTOR, "Specialty area", 1.00),
    (PROGRAM_GENDER, "Gender composition", 0.50),
    (PROGRAM_SCHOOL_DAY, "School day", 0.50),
    (PROGRAM_RURALITY, "Rurality", 0.50),
    (PROGRAM_PIE, "PIE", 0.40),
    (PROGRAM_PACE, "PACE", 0.40),
    (PROGRAM_ENROLLMENT_FEE, "Enrollment fee", 1.00),
    (PROGRAM_MONTHLY_FEE, "Monthly fee", 1.25),
    (PROGRAM_RELIGIOUS_ORIENTATION, "Religious orientation", 1.00),
]


def clean_recommendation_value(value) -> str:
    """Return a usable categorical value for recommendation scoring."""
    if pd.isna(value):
        return ""
    text = " ".join(str(value).strip().split())
    if not text or text.lower() in {"nan", "unknown", "no information"}:
        return ""
    return text


def wish_rank_weight(rank, rank_sensitive: bool = True) -> float:
    """
    Weight higher-ranked wishes slightly more.

    If rank_sensitive=False, every wish has the same weight.
    """
    if not rank_sensitive:
        return 1.0
    try:
        r = max(int(float(rank)), 1)
    except Exception:
        r = 1
    return 1.0 / np.sqrt(r)


def recommendation_rank_value(wish: pd.Series):
    """Use the equivalence group when available, otherwise use the strict rank."""
    group = wish.get(EQUIV_GROUP, np.nan)
    if not pd.isna(group):
        return group
    return wish.get(WISH_RANK, 1)


def build_wish_profile(
    wishes: pd.DataFrame,
    program_mapping: dict[str, pd.Series],
    *,
    rank_sensitive: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """
    Build the student's revealed-preference profile from the current wish list.

    The profile is a weighted distribution of program characteristics.
    Example: if most listed programs are Catholic and free, those values receive
    high shares and will drive recommendations.
    """
    valid_wishes = wishes.copy()
    valid_wishes[PROGRAM] = valid_wishes[PROGRAM].fillna("").astype(str).str.strip()
    valid_wishes = valid_wishes[valid_wishes[PROGRAM].isin(program_mapping)].copy()

    if valid_wishes.empty:
        return {}, pd.DataFrame()

    profile = {
        "selected_programs": set(valid_wishes[PROGRAM].tolist()),
        "regions": {},
        "criteria": {},
    }

    for _, wish in valid_wishes.iterrows():
        label = str(wish[PROGRAM]).strip()
        row = program_mapping[label]
        weight = wish_rank_weight(recommendation_rank_value(wish), rank_sensitive=rank_sensitive)

        region = clean_recommendation_value(row.get(REGION, UNKNOWN_REGION)) or UNKNOWN_REGION
        profile["regions"][region] = profile["regions"].get(region, 0.0) + weight

        for col, _, _ in RECOMMENDATION_CRITERIA:
            value = clean_recommendation_value(row.get(col, ""))
            if not value:
                continue
            profile["criteria"].setdefault(col, {})
            profile["criteria"][col][value] = profile["criteria"][col].get(value, 0.0) + weight

    total_region_weight = sum(profile["regions"].values())
    if total_region_weight > 0:
        profile["regions"] = {
            k: v / total_region_weight
            for k, v in profile["regions"].items()
        }

    dominant_rows = []
    for col, label, _ in RECOMMENDATION_CRITERIA:
        dist = profile["criteria"].get(col, {})
        total = sum(dist.values())
        if total <= 0:
            continue

        normalized = {k: v / total for k, v in dist.items()}
        profile["criteria"][col] = normalized

        dominant_value, dominant_share = max(normalized.items(), key=lambda x: x[1])
        dominant_rows.append({
            "Criterion": label,
            "Dominant value in current list": dominant_value,
            "Share": f"{dominant_share:.0%}",
        })

    profile_table = pd.DataFrame(dominant_rows)
    return profile, profile_table


def recommend_similar_programs(
    wishes: pd.DataFrame,
    program_mapping: dict[str, pd.Series],
    criterion_weights: dict[str, float],
    *,
    max_recommendations: int = 15,
    rank_sensitive: bool = True,
    competition_weight: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recommend programs similar to the current wish list.

    Hard rule:
    - candidates must be in one of the regions already present in the wish list.
      If the student listed programs in two regions, both regions are allowed.
      No outside-region program is recommended.

    Similarity rule:
    - for each criterion, the candidate receives points when its value matches
      frequent values in the student's list.
    - user-defined criterion weights decide how much each criterion matters.

    Optional competition rule:
    - competition_weight can give a small bonus to programs with lower
      true_applicants_last_year / capacity ratios.
    """
    profile, profile_table = build_wish_profile(
        wishes,
        program_mapping,
        rank_sensitive=rank_sensitive,
    )

    if not profile:
        return pd.DataFrame(), profile_table

    allowed_regions = set(profile["regions"].keys())
    selected_programs = profile["selected_programs"]

    active_weight_total = 0.0
    for col, _, _ in RECOMMENDATION_CRITERIA:
        if profile["criteria"].get(col):
            active_weight_total += max(float(criterion_weights.get(col, 0.0)), 0.0)

    if active_weight_total <= 0:
        return pd.DataFrame(), profile_table

    rows = []
    competition_weight = max(float(competition_weight), 0.0)

    for candidate_label, row in program_mapping.items():
        if candidate_label in selected_programs:
            continue

        candidate_region = clean_recommendation_value(row.get(REGION, UNKNOWN_REGION)) or UNKNOWN_REGION

        # Exclusive regional priority: never recommend outside the listed region(s).
        if candidate_region not in allowed_regions:
            continue

        raw_similarity = 0.0
        reason_parts = []

        for col, human_label, _ in RECOMMENDATION_CRITERIA:
            user_weight = max(float(criterion_weights.get(col, 0.0)), 0.0)
            if user_weight <= 0:
                continue

            wish_distribution = profile["criteria"].get(col, {})
            if not wish_distribution:
                continue

            candidate_value = clean_recommendation_value(row.get(col, ""))
            if not candidate_value:
                continue

            match_share = wish_distribution.get(candidate_value, 0.0)
            if match_share <= 0:
                continue

            raw_similarity += user_weight * match_share

            if match_share >= 0.40:
                reason_parts.append(f"{human_label}: {candidate_value} ({match_share:.0%})")

        similarity_score = raw_similarity / active_weight_total

        capacity = max(as_float(row.get(CAPACITY, 0), 0.0), 0.0)
        true_applicants = max(as_float(row.get(TRUE_APP, 0), 0.0), 0.0)

        if capacity > 0 and true_applicants > 0:
            competition_ratio = true_applicants / capacity
            accessibility_score = min(capacity / true_applicants, 1.0)
        else:
            competition_ratio = np.nan
            accessibility_score = 0.0

        final_score = (
            similarity_score
            if competition_weight == 0
            else (similarity_score + competition_weight * accessibility_score) / (1.0 + competition_weight)
        )

        rows.append({
            PROGRAM: candidate_label,
            "School": clean_recommendation_value(row.get(SCHOOL_NAME, "")) or "School name unavailable",
            "Commune": clean_recommendation_value(row.get(SCHOOL_COMMUNE, "")),
            "Region": candidate_region,
            "Program details": clean_recommendation_value(row.get(PROGRAM_DISPLAY_NAME, "")),
            "Similarity score": round(100 * similarity_score, 1),
            "Recommendation score": round(100 * final_score, 1),
            "Capacity": int(capacity) if capacity == int(capacity) else capacity,
            "True applicants last year": int(true_applicants) if true_applicants == int(true_applicants) else true_applicants,
            "Applicants / seat": round(competition_ratio, 2) if not pd.isna(competition_ratio) else "",
            "Why recommended": "; ".join(reason_parts[:4]) if reason_parts else "Partial similarity across weighted criteria",
        })

    if not rows:
        return pd.DataFrame(), profile_table

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["Recommendation score", "Similarity score", "Region", "School"],
        ascending=[False, False, True, True],
    ).head(max_recommendations)

    return out.reset_index(drop=True), profile_table


def clear_wish_editor_widget_state(editor_widget_key_base: str) -> None:
    """Clear Streamlit data-editor widget keys so added recommendations appear immediately."""
    for key in list(st.session_state.keys()):
        if str(key).startswith(editor_widget_key_base):
            del st.session_state[key]


def render_similar_program_recommendations(
    edited: pd.DataFrame,
    program_mapping: dict[str, pd.Series],
    *,
    editor_state_key: str,
    editor_widget_key_base: str,
    use_equivalence_classes: bool = False,
    simulation_done_key: str | None = None,
    simulation_result_key: str | None = None,
) -> None:
    """Render the recommendation UI and optionally append selected programs to the wish list."""
    st.subheader("4. Recommended similar programs")

    with st.expander("Find additional programs similar to the current wish list", expanded=True):
        current_selected_programs = [
            p for p in edited[PROGRAM].dropna().astype(str).str.strip()
            if p and p in program_mapping
        ]

        if not current_selected_programs:
            st.info("Enter at least one valid program in the wish list to get recommendations.")
            return

        st.caption(
            "Recommendations are restricted to the region(s) already present in the wish list. "
            "If the list contains programs from two regions, recommendations may come from both regions."
        )

        rec_rank_sensitive = st.checkbox(
            "Give slightly more importance to higher-ranked wishes",
            value=True,
            help="If unchecked, every listed wish counts equally in the preference profile.",
        )

        rec_max = st.slider(
            "Number of recommendations",
            min_value=2,
            max_value=10,
            value=5,
            step=1,
        )

        rec_competition_weight = st.slider(
            "Bonus for less oversubscribed programs",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.25,
            help=(
                "0 means recommendations are based only on similarity. "
                "Higher values give a bonus to programs with fewer true applicants per seat last year."
            ),
        )

        st.markdown("#### Criterion weights")

        criterion_weights = {}
        weight_cols = st.columns(3)

        for i, (criterion_col, criterion_label, default_weight) in enumerate(RECOMMENDATION_CRITERIA):
            with weight_cols[i % 3]:
                criterion_weights[criterion_col] = st.slider(
                    criterion_label,
                    min_value=0.0,
                    max_value=3.0,
                    value=float(default_weight),
                    step=0.25,
                    key=f"recommendation_weight_{criterion_col}",
                )

        recommendations, profile_table = recommend_similar_programs(
            edited,
            program_mapping,
            criterion_weights,
            max_recommendations=rec_max,
            rank_sensitive=rec_rank_sensitive,
            competition_weight=rec_competition_weight,
        )

        if not profile_table.empty:
            st.markdown("#### Main criteria inferred from the wish list")
            st.dataframe(profile_table, width="stretch", hide_index=True)

        if recommendations.empty:
            st.warning(
                "No similar program was found in the same region(s) with the current weights."
            )
            return

        st.markdown("#### Suggested programs")
        st.dataframe(
            recommendations[
                [
                    "Recommendation score",
                    "Similarity score",
                    "School",
                    "Commune",
                    "Region",
                    "Program details",
                    "Capacity",
                    "True applicants last year",
                    "Applicants / seat",
                    "Why recommended",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

        programs_to_add = st.multiselect(
            "Add recommended programs to the wish list",
            options=recommendations[PROGRAM].tolist(),
            default=[],
        )

        if st.button("Add selected recommendations", disabled=not programs_to_add):
            non_empty = edited.copy()
            non_empty[PROGRAM] = non_empty[PROGRAM].fillna("").astype(str).str.strip()
            non_empty = non_empty[non_empty[PROGRAM] != ""].copy()

            existing = set(non_empty[PROGRAM].tolist())
            new_programs = [p for p in programs_to_add if p not in existing]

            rows_to_add = []
            if len(non_empty) > 0:
                existing_ranks = pd.to_numeric(
                    non_empty.get(WISH_RANK, pd.Series(dtype=float)),
                    errors="coerce",
                ).dropna()
                existing_groups = pd.to_numeric(
                    non_empty.get(EQUIV_GROUP, pd.Series(dtype=float)),
                    errors="coerce",
                ).dropna()

                next_rank = int(existing_ranks.max()) + 1 if not existing_ranks.empty else len(non_empty) + 1
                next_group = int(existing_groups.max()) + 1 if not existing_groups.empty else next_rank
            else:
                next_rank = 1
                next_group = 1

            for i, program_label in enumerate(new_programs):
                if use_equivalence_classes:
                    # In equivalence-class mode, recommended programs are appended to
                    # the next preference group, not to the next strict row number.
                    wish_rank_value = next_rank + i
                    equivalence_group_value = next_group
                else:
                    wish_rank_value = next_rank + i
                    equivalence_group_value = next_rank + i

                new_row = {
                    WISH_RANK: wish_rank_value,
                    EQUIV_GROUP: equivalence_group_value,
                    PROGRAM: program_label,
                    LOTTERY: 1,
                }
                for col in PRIORITIES + [SAFETY]:
                    new_row[col] = False
                rows_to_add.append(new_row)

            if rows_to_add:
                updated_wishes = pd.concat(
                    [non_empty, pd.DataFrame(rows_to_add)],
                    ignore_index=True,
                )
                st.session_state[editor_state_key] = clean_wish_rows(updated_wishes)
                if simulation_done_key:
                    st.session_state[simulation_done_key] = False
                if simulation_result_key:
                    st.session_state.pop(simulation_result_key, None)
                clear_wish_editor_widget_state(editor_widget_key_base)
                st.rerun()
            else:
                st.info("All selected recommendations are already in the wish list.")

# ---------------------------------------------------------------------------
# Priority logic
# ---------------------------------------------------------------------------

def resolve_priority_tier(wish: pd.Series, program: pd.Series) -> str:
    """Determine the priority tier for a wish.

    This version reuses the older MTB app rule for priority_student:
    the student keeps the priority_student tier only if their lottery number
    is within floor(15% * program capacity). Otherwise, the student falls back
    to the next active priority tier.
    """
    if as_bool(wish.get("priority_sibling")):
        return "priority_sibling"

    if as_bool(wish.get("priority_student")):
        capacity = max(round(as_float(program[CAPACITY])), 0)
        quota_count = int(np.floor(PRIORITY_STUDENT_QUOTA * capacity))
        lottery = max(round(as_float(wish.get(LOTTERY, 1), 1)), 1)
        if lottery <= quota_count:
            return "priority_student"

    if as_bool(wish.get("priority_parent_civil_servant")):
        return "priority_parent_civil_servant"
    if as_bool(wish.get("priority_ex_student")):
        return "priority_ex_student"
    return NO_PRIORITY


# ---------------------------------------------------------------------------
# Availability calculation for one wish
# ---------------------------------------------------------------------------

def availability(wish: pd.Series, program: pd.Series) -> dict:
    capacity = max(round(as_float(program[CAPACITY])), 0)
    true_app = max(round(as_float(program[TRUE_APP])), 0)

    # Theory-consistent MTB mode:
    # SHA-256 gives a percentile, which is converted into an equivalent
    # lottery rank inside the program-level reference population N_s.
    population = max(round(as_float(program[POP])), 1)
    pop_label  = POP

    lottery  = max(round(as_float(wish.get(LOTTERY, 1), 1)), 1)
    raw_rank = min(lottery, population)

    # Reference-theory step: raw lottery number -> within-program percentile.
    # The hash percentile is used upstream to generate the equivalent rank,
    # but the availability calculation itself follows the rank-based formula.
    percentile = float(np.clip((raw_rank - 1) / max(population - 1, 1), 0, 1))

    tier = resolve_priority_tier(wish, program)
    share  = as_float(program[f"priority_share_{tier}_2024"])
    before = as_float(program[f"cum_share_before_{tier}_2024"])
    eff_pct  = float(np.clip(before + share * percentile, 0, 1))
    eff_rank = pct_to_rank(eff_pct, population)

    if as_bool(wish.get(SAFETY)):
        p_avail = 1.0
    elif capacity <= 0:
        p_avail = 0.0
    else:
        # Reference-theory model:
        # X ~ Hypergeometric(N_s - 1, T_s - 1, r_e - 1).
        M = max(population - 1, 0)
        draws = min(max(eff_rank - 1, 0), M)
        successes = min(max(true_app - 1, 0), M)
        p_avail = (
            1.0
            if draws == 0 or successes == 0
            else float(hypergeom.cdf(capacity - 1, M, successes, draws))
        )

    return {
        "wish_rank":                        int(wish[WISH_RANK]),
        "program":                          wish[PROGRAM],
        "lottery_number":                   lottery,
        "priority_tier":                    tier,
        "capacity":                         capacity,
        "true_applicants_last_year":        true_app,
        "lottery_population_used":          population,
        "lottery_population_source":        pop_label,
        "raw_lottery_rank":                 raw_rank,
        "lottery_percentile_used":          percentile,
        "priority_effective_percentile":    eff_pct,
        "priority_effective_rank":          eff_rank,
        "lottery_hash_input":               str(wish.get(HASH_INPUT, "")),
        "lottery_hash_hex":                 str(wish.get(HASH_HEX, "")),
        "lottery_hash_percentile":          as_float(wish.get(HASH_PCT), np.nan),
        "availability_probability":         float(np.clip(p_avail, 0, 1)),
        "calibration_2024_imputed":         as_bool(program.get(IMPUTED, False)),
        "calibration_2024_imputation_method": str(program.get(IMPUT_METHOD, "")),
    }


# ---------------------------------------------------------------------------
# Global calculation (wish list -> results DataFrame)
# ---------------------------------------------------------------------------

def compute(
    wishes: pd.DataFrame,
    mapping: dict[str, pd.Series],
) -> pd.DataFrame:
    clean = wishes[wishes[PROGRAM].astype(str).str.strip() != ""].sort_values(WISH_RANK)
    if clean.empty:
        raise ValueError("Add at least one valid wish.")

    rows = [
        availability(wish, mapping[wish[PROGRAM]])
        for _, wish in clean.iterrows()
        if wish[PROGRAM] in mapping
    ]

    choices = pd.DataFrame(rows)
    choices["cumulative_unavailable_before_choice"] = (
        (1 - choices["availability_probability"]).cumprod().shift(1).fillna(1)
    )
    choices["choice_assignment_probability"] = (
        choices["cumulative_unavailable_before_choice"] * choices["availability_probability"]
    )
    choices["cumulative_unavailable_after_choice"] = (
        (1 - choices["availability_probability"]).cumprod()
    )
    return choices




# ---------------------------------------------------------------------------
# Equivalence-class handling and display helpers
# ---------------------------------------------------------------------------

def prepare_ordered_wishes(wishes: pd.DataFrame, use_equivalence_classes: bool) -> pd.DataFrame:
    """Return the reference strict order used for preview and the first simulation.

    The calculation model is unchanged. This function only converts the user's
    interface input into a strict list. If equivalence classes are enabled, group
    order is respected and the current row order is used inside each group.
    """
    clean = clean_wish_rows(wishes)
    clean = clean[clean[PROGRAM].astype(str).str.strip() != ""].copy().reset_index(drop=True)
    if clean.empty:
        return clean

    clean["_row_order"] = range(len(clean))
    clean[WISH_RANK] = pd.to_numeric(clean[WISH_RANK], errors="coerce").fillna(clean["_row_order"] + 1).astype(int)
    clean[EQUIV_GROUP] = pd.to_numeric(clean[EQUIV_GROUP], errors="coerce").fillna(clean[WISH_RANK]).astype(int)

    if use_equivalence_classes:
        clean = clean.sort_values([EQUIV_GROUP, "_row_order"], kind="stable")
    else:
        clean = clean.sort_values([WISH_RANK, "_row_order"], kind="stable")
        clean[EQUIV_GROUP] = range(1, len(clean) + 1)

    clean = clean.drop(columns=["_row_order"], errors="ignore").reset_index(drop=True)
    clean[WISH_RANK] = range(1, len(clean) + 1)
    return clean


def count_equivalence_orders(wishes: pd.DataFrame) -> int:
    clean = prepare_ordered_wishes(wishes, use_equivalence_classes=True)
    if clean.empty:
        return 0
    total = 1
    for size in clean.groupby(EQUIV_GROUP, sort=True).size().tolist():
        total *= math.factorial(int(size))
    return int(total)


def iter_equivalence_orders(wishes: pd.DataFrame):
    """Yield every strict ranking compatible with the equivalence classes."""
    clean = prepare_ordered_wishes(wishes, use_equivalence_classes=True)
    if clean.empty:
        return

    groups = [g.copy() for _, g in clean.groupby(EQUIV_GROUP, sort=True)]
    index_blocks = [list(permutations(g.index.tolist())) for g in groups]

    for combo in product(*index_blocks):
        ordered_indices = [idx for block in combo for idx in block]
        out = clean.loc[ordered_indices].copy().reset_index(drop=True)
        out[WISH_RANK] = range(1, len(out) + 1)
        yield out


def predicted_outcome_from_choices(choices: pd.DataFrame, threshold: float) -> tuple[str, float, bool]:
    """Apply the app's existing threshold rule to summarize the predicted outcome."""
    p_unmatched = float(choices["cumulative_unavailable_after_choice"].iloc[-1])
    at_risk = p_unmatched >= threshold
    if at_risk:
        return "Unmatched", p_unmatched, True

    positive = (
        choices[choices["choice_assignment_probability"] > 0]
        .sort_values("choice_assignment_probability", ascending=False)
        .reset_index(drop=True)
    )
    if positive.empty:
        return "Unmatched", p_unmatched, True
    return str(positive.iloc[0]["program"]), p_unmatched, False


def format_choices_table(choices: pd.DataFrame) -> pd.DataFrame:
    display_cols = [
        "wish_rank",
        "program",
        "lottery_number",
        "priority_tier",
        "capacity",
        "true_applicants_last_year",
        "availability_probability",
        "choice_assignment_probability",
    ]

    table = choices[display_cols].copy()
    for prob_col in ("availability_probability", "choice_assignment_probability"):
        table[prob_col] = table[prob_col].astype(float).map(lambda x: f"{x:.1%}")

    return table.rename(columns={
        "wish_rank": "Wish rank",
        "program": "Program",
        "lottery_number": "Lottery number",
        "priority_tier": "Priority tier",
        "capacity": "Seats",
        "true_applicants_last_year": "True applicants last year",
        "availability_probability": "Chance if considered",
        "choice_assignment_probability": "Final chance of assignment",
    })


def compact_order_label(order_df: pd.DataFrame, max_items: int = 5) -> str:
    programs = order_df[PROGRAM].astype(str).str.strip().tolist()
    if len(programs) <= max_items:
        return " → ".join(programs)
    return " → ".join(programs[:max_items]) + f" → … (+{len(programs) - max_items})"


def render_single_summary(choices: pd.DataFrame, threshold: float) -> None:
    p_unmatched = float(choices["cumulative_unavailable_after_choice"].iloc[-1])
    at_risk = p_unmatched >= threshold

    st.subheader("Summary")
    st.metric("Unmatched risk", f"{p_unmatched:.1%}")

    positive = (
        choices[choices["choice_assignment_probability"] > 0]
        .sort_values("choice_assignment_probability", ascending=False)
        .reset_index(drop=True)
    )

    if at_risk:
        st.error(
            "The student is at risk of remaining unmatched. "
            "The list appears risky; adding safer options is recommended."
        )
        if positive.empty:
            st.markdown("**Most likely outcome:**")
            st.write("1. Unmatched")
        else:
            st.markdown("**Most likely outcomes:**")
            st.write("1. Unmatched")
            for i, row in positive.head(2).iterrows():
                st.write(f"{i + 2}. {row['program']}")
    else:
        if positive.empty:
            st.error("No listed school appears realistically accessible.")
        else:
            best = positive.iloc[0]
            st.success(
                f"The student is not flagged as at risk. "
                f"The most likely assignment is: **{best['program']}**."
            )
            st.markdown("**Top 3 most likely schools:**")
            for i, row in positive.head(3).iterrows():
                st.write(f"{i + 1}. {row['program']}")


def render_simulation_result(result: dict) -> None:
    """Render the last simulation, including equivalence-class sensitivity.

    The result is stored in session_state so it remains visible after the user
    interacts with recommendation sliders or add-program controls.
    """
    if not result:
        return

    threshold_used = float(result.get("threshold", DEFAULT_THRESHOLD_MTB))
    mode = result.get("mode", "strict")

    if mode == "equivalence":
        reference_choices = result.get("reference_choices")
        variants_df = result.get("variants_df")
        distinct_outcomes = result.get("distinct_outcomes", [])

        if reference_choices is None or variants_df is None or len(variants_df) == 0:
            return

        st.subheader("Reference strict-order details")
        st.caption(
            "This table uses the current row order inside each equivalence group as the reference order. "
            "The sensitivity test below evaluates every strict order compatible with the groups."
        )
        st.dataframe(format_choices_table(reference_choices), width="stretch", hide_index=True)
        render_single_summary(reference_choices, threshold_used)

        st.subheader("Equivalence-class sensitivity")
        if len(distinct_outcomes) == 1:
            st.success(
                "The strict ordering inside the equivalence classes does not change the predicted final outcome. "
                f"All {len(variants_df):,} compatible strict order(s) lead to: **{distinct_outcomes[0]}**."
            )
        else:
            st.warning(
                "The strict ordering inside at least one equivalence class can change the predicted final outcome. "
                "The user should choose a strict order carefully for the tied programs."
            )

        outcome_summary = (
            variants_df
            .groupby("Predicted outcome", as_index=False)
            .agg(
                strict_orders=("Strict order #", "count"),
                min_unmatched_risk=("Unmatched risk", "min"),
                max_unmatched_risk=("Unmatched risk", "max"),
            )
            .sort_values(["strict_orders", "Predicted outcome"], ascending=[False, True])
        )
        outcome_summary["Share of strict orders"] = outcome_summary["strict_orders"] / len(variants_df)
        for col in ["min_unmatched_risk", "max_unmatched_risk", "Share of strict orders"]:
            outcome_summary[col] = outcome_summary[col].map(lambda x: f"{x:.1%}")
        outcome_summary = outcome_summary.rename(columns={
            "strict_orders": "Strict orders",
            "min_unmatched_risk": "Min unmatched risk",
            "max_unmatched_risk": "Max unmatched risk",
        })
        st.dataframe(outcome_summary, width="stretch", hide_index=True)

        with st.expander("All strict orders tested", expanded=False):
            variants_display = variants_df.copy()
            variants_display["Unmatched risk"] = variants_display["Unmatched risk"].map(lambda x: f"{x:.1%}")
            st.dataframe(variants_display, width="stretch", hide_index=True)
        return

    choices = result.get("choices")
    if choices is None:
        return

    st.subheader("Wish-level details")
    st.caption(
        "Chance if considered is the chance of getting that program if the student reaches that wish. "
        "Final chance of assignment also accounts for all higher-ranked wishes."
    )
    st.dataframe(format_choices_table(choices), width="stretch", hide_index=True)
    render_single_summary(choices, threshold_used)
# ===========================================================================
# Interface Streamlit
# ===========================================================================

st.set_page_config(
    page_title="SAE simulation – unmatched risk",
    page_icon="🎓",
    layout="wide",
)
st.title("SAE admission-risk simulation")
st.caption(
    "MTB mode (admission 2026): SHA-256(RUN/IPE+RBD) percentile by school."
)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.caption("Capacities + 2024 calibration data are loaded from data/.")

    threshold = st.slider(
        "Alert threshold – unmatched risk",
        0.01, 1.0,
        DEFAULT_THRESHOLD_MTB,
        0.005,
        key="threshold_mtb",
    )

    national_student_id = st.text_input(
        "Student RUN/IPE",
        value="",
        placeholder="12.345.678-9",
        help=(
            "Used to compute the SHA-256 percentile specific to each "
            "school. RUN format: 12.345.678-9. Dots are optional. "
            "For foreign students, enter the IPE."
        ),
    )

# ── Built-in capacities/calibration data ─────────────────────────────
calib = load_calibration(CAPACITIES_PATH.read_bytes())
missing = [c for c in required_cols() if c not in calib.columns]
if missing:
    st.error("Missing columns: " + ", ".join(missing[:20]))
    st.stop()

invalid_population = pd.to_numeric(calib[POP], errors="coerce").isna() | (pd.to_numeric(calib[POP], errors="coerce") <= 0)
if invalid_population.any():
    st.error(
        f"Invalid {POP}: {int(invalid_population.sum())} program(s) have missing or non-positive lottery population."
    )
    st.stop()

program_options, program_mapping = build_options(calib)

# ── Section 1: pathway ───────────────────────────────────────────────
st.subheader("1. Start with the student's preferences")

list_status = st.radio(
    "Is the student's wish list already established?",
    [
        "Yes — I already have the list",
        "No — help me build it with filters",
    ],
    horizontal=True,
)
needs_builder = list_status.startswith("No")

ranking_mode = st.radio(
    "How should preferences be entered?",
    [
        "Strict ranking",
        "Equivalence classes",
    ],
    horizontal=True,
    help=(
        "Strict ranking means every program has a precise rank. Equivalence classes "
        "allow several programs to share the same preference group."
    ),
)
use_equivalence_classes = ranking_mode == "Equivalence classes"

if use_equivalence_classes:
    st.info(
        "Use the same preference-group number for programs the student considers tied. "
        "Lower group numbers are preferred. The app will test every strict order within each group."
    )
else:
    st.info("Enter programs in strict order. The first row is the highest-ranked choice.")

wish_file = st.file_uploader(
    "Optional: import a wish-list CSV to pre-fill the table",
    type=["csv"],
)

if wish_file is not None:
    try:
        base_rows = parse_wishes(wish_file.getvalue(), program_mapping)
        st.success(f"Table pre-filled with {len(base_rows)} wish(es).")
    except Exception as exc:
        st.error(f"Could not import the CSV: {exc}")
        base_rows = empty_wishes()
else:
    base_rows = empty_wishes()

# ── Optional program-building filters ─────────────────────────────────
empty_filters = {
    "tracks": [],
    "specialty_sectors": [],
    "genders": [],
    "school_days": [],
    "rurality": [],
    "pie": [],
    "pace": [],
    "enrollment_fee": [],
    "monthly_fee": [],
    "religious_orientation": [],
}
program_filters = empty_filters.copy()
selected_program_region = "All regions"

if needs_builder:
    st.subheader("2. Find programs")
    with st.expander("Program search filters", expanded=True):
        st.caption("Leave every filter empty to include all programs.")

        region_options = ["All regions"] + available_regions(calib)
        selected_program_region = st.selectbox(
            "Program region",
            region_options,
            index=0,
            help=(
                "Choose a region to make the program list shorter. Already selected "
                "programs from other regions are kept in the table."
            ),
        )

        c1, c2 = st.columns(2)
        with c1:
            filter_general = st.checkbox("General academic programs", value=False)
        with c2:
            filter_specialized = st.checkbox("Specialized / technical programs", value=False)

        selected_specialty_sectors = []
        if filter_specialized:
            selected_specialty_sectors = st.multiselect(
                "Specialized area",
                SPECIALTY_FILTER_OPTIONS,
                default=[],
                help="Leave empty to include all specialized areas.",
            )

        c1, c2 = st.columns(2)
        with c1:
            selected_genders = st.multiselect(
                "Gender composition",
                GENDER_FILTER_OPTIONS,
                default=[],
                help="Leave empty to include mixed, boys-only, and girls-only programs.",
            )
            selected_rurality = st.multiselect(
                "Rurality",
                RURALITY_FILTER_OPTIONS,
                default=[],
                help="Leave empty to include both urban and rural schools.",
            )
            selected_pie = st.multiselect(
                "PIE integration program",
                PIE_FILTER_OPTIONS,
                default=[],
                help="Leave empty to include schools with and without PIE.",
            )
            selected_enrollment_fee = st.multiselect(
                "Enrollment fee",
                PAYMENT_FILTER_OPTIONS,
                default=[],
                help="Leave empty to include every enrollment-fee category.",
            )
        with c2:
            selected_school_days = st.multiselect(
                "School day",
                SCHOOL_DAY_FILTER_OPTIONS,
                default=[],
                help="Leave empty to include full-day, morning, and afternoon programs.",
            )
            selected_pace = st.multiselect(
                "PACE program",
                PACE_FILTER_OPTIONS,
                default=[],
                help="Leave empty to include schools with and without PACE.",
            )
            selected_monthly_fee = st.multiselect(
                "Monthly fee",
                PAYMENT_FILTER_OPTIONS,
                default=[],
                help="Leave empty to include every monthly-fee category.",
            )
            selected_religious_orientation = st.multiselect(
                "Religious orientation",
                RELIGIOUS_FILTER_OPTIONS,
                default=[],
                help="Leave empty to include every orientation.",
            )

        program_filters = {
            "tracks": ([TRACK_GENERAL] if filter_general else []) + ([TRACK_SPECIALIZED] if filter_specialized else []),
            "specialty_sectors": selected_specialty_sectors,
            "genders": selected_genders,
            "school_days": selected_school_days,
            "rurality": selected_rurality,
            "pie": selected_pie,
            "pace": selected_pace,
            "enrollment_fee": selected_enrollment_fee,
            "monthly_fee": selected_monthly_fee,
            "religious_orientation": selected_religious_orientation,
        }
else:
    st.subheader("2. Enter the list")
    st.caption("Use the table below to enter the existing wish list directly.")

# ── Wish-list editor ──────────────────────────────────────────────────
table_key_parts = [
    hashlib.md5(wish_file.getvalue()).hexdigest()[:8] if wish_file else "empty",
    "builder" if needs_builder else "direct",
    "equiv" if use_equivalence_classes else "strict",
]
table_key = "_".join(table_key_parts)
editor_state_key = f"wish_rows_{table_key}_mtb"
editor_source_key = f"wish_rows_source_{table_key}_mtb"
editor_widget_key_base = f"wishes_editor_{table_key}_mtb"
simulation_done_key = f"simulation_done_{table_key}_mtb"
simulation_result_key = f"simulation_result_{table_key}_mtb"

if st.session_state.get(editor_source_key) != table_key or editor_state_key not in st.session_state:
    st.session_state[editor_source_key] = table_key
    st.session_state[editor_state_key] = clean_wish_rows(base_rows)

editor_rows = st.session_state[editor_state_key].copy()
if PROGRAM in editor_rows.columns:
    editor_rows[PROGRAM] = editor_rows[PROGRAM].map(
        lambda x: x if str(x).strip() in program_mapping or str(x).strip() == "" else ""
    )

current_program_values = (
    editor_rows.get(PROGRAM, pd.Series(dtype=str))
    .dropna()
    .astype(str)
    .str.strip()
    .tolist()
)
program_options_for_editor = filter_program_options(
    program_mapping,
    selected_program_region,
    active_filters=program_filters,
    current_values=current_program_values,
)

options_signature = hashlib.md5(
    "|".join(program_options_for_editor).encode("utf-8")
).hexdigest()[:8]
editor_widget_key = f"{editor_widget_key_base}_{options_signature}"

if needs_builder and (selected_program_region != "All regions" or filters_are_active(program_filters)):
    preserved = [
        p for p in current_program_values
        if p in program_mapping
        and not (
            (selected_program_region == "All regions" or str(program_mapping[p].get(REGION, UNKNOWN_REGION)).strip() == selected_program_region)
            and program_matches_filters(program_mapping[p], program_filters)
        )
    ]
    matching_count = max(len(program_options_for_editor) - len(preserved), 0)
    extra_note = (
        f" Existing selected program(s) outside the current filters are also kept available: "
        f"{len(preserved)}."
        if preserved else ""
    )
    region_text = selected_program_region if selected_program_region != "All regions" else "all regions"
    st.caption(
        f"Showing {matching_count} matching program option(s) for {region_text}."
        f"{extra_note}"
    )

col_config: dict = {
    WISH_RANK: st.column_config.NumberColumn("Wish rank", min_value=1, step=1, width=95),
    EQUIV_GROUP: st.column_config.NumberColumn(
        "Preference group",
        min_value=1,
        step=1,
        width=130,
        help=(
            "Programs with the same number are treated as equivalent. "
            "Group 1 is preferred to group 2, group 2 to group 3, etc."
        ),
    ),
    PROGRAM: st.column_config.SelectboxColumn(
        "Program",
        options=[""] + program_options_for_editor,
        width="large",
        help=(
            "Each option shows the school name, program name, and RBD. "
            "In builder mode, use filters to shorten the list."
        ),
    ),
    "priority_sibling": st.column_config.CheckboxColumn("Sibling priority", width="small"),
    "priority_student": st.column_config.CheckboxColumn(
        "Priority student",
        width="medium",
        help=(
            "RSH means Registro Social de Hogares. Check this when the student "
            "is eligible for the Chilean priority-student criterion."
        ),
    ),
    "priority_parent_civil_servant": st.column_config.CheckboxColumn("Civil-servant parent priority", width="medium"),
    "priority_ex_student": st.column_config.CheckboxColumn("Former-student priority", width="medium"),
    SAFETY: st.column_config.CheckboxColumn("Already enrolled", width="medium"),
}

editor_rows = editor_rows.drop(columns=[LOTTERY], errors="ignore")
column_order = [
    EQUIV_GROUP if use_equivalence_classes else WISH_RANK,
    PROGRAM,
    "priority_sibling",
    "priority_student",
    "priority_parent_civil_servant",
    "priority_ex_student",
    SAFETY,
]

edited = st.data_editor(
    editor_rows,
    num_rows="dynamic",
    width="stretch",
    hide_index=True,
    key=editor_widget_key,
    column_config=col_config,
    column_order=column_order,
)

cleaned_edited = clean_wish_rows(edited)
old_state = clean_wish_rows(st.session_state[editor_state_key])
if not cleaned_edited.astype(str).equals(old_state.astype(str)):
    st.session_state[editor_state_key] = cleaned_edited
    st.session_state[simulation_done_key] = False
    st.session_state.pop(simulation_result_key, None)
    st.rerun()
edited = cleaned_edited

selected = [p for p in edited[PROGRAM].dropna().astype(str).str.strip() if p]
imputed = [
    p for p in selected
    if p in program_mapping and as_bool(program_mapping[p].get(IMPUTED, False))
]
if imputed:
    st.warning(
        "Less reliable estimate: at least one selected program uses "
        "mean-imputed 2024 calibration values."
    )

# ── MTB percentile preview ────────────────────────────────────────────
reference_order = prepare_ordered_wishes(edited, use_equivalence_classes)
if not reference_order.empty and national_student_id.strip():
    try:
        preview_w = attach_mtb_hashes(reference_order, program_mapping, national_student_id)
        preview_cols = [WISH_RANK, PROGRAM, LOTTERY, HASH_PCT]
        if use_equivalence_classes:
            preview_cols.insert(1, EQUIV_GROUP)
        preview = preview_w[preview_cols].copy()
        preview[HASH_PCT] = (
            pd.to_numeric(preview[HASH_PCT], errors="coerce")
            .map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        )
        preview = preview.rename(columns={
            WISH_RANK: "Reference rank",
            EQUIV_GROUP: "Preference group",
            PROGRAM: "Program",
            LOTTERY: "Lottery number",
            HASH_PCT: "MTB hash percentile",
        })
        with st.expander("Calculated MTB percentiles (RUN + RBD)", expanded=False):
            st.dataframe(preview, width="stretch", hide_index=True)
    except Exception as exc:
        st.warning(f"MTB preview unavailable: {exc}")

# ── Section 3: simulation ─────────────────────────────────────────────
st.subheader("3. Run the simulation")

if use_equivalence_classes:
    total_orders = count_equivalence_orders(edited)
    if total_orders:
        st.caption(
            f"The current equivalence classes generate {total_orders:,} compatible strict order(s)."
        )

calculated_now = False

if st.button("Calculate unmatched risk", type="primary"):
    if not national_student_id.strip():
        st.error("Please enter the student's RUN/IPE before running the simulation.")
        st.stop()

    try:
        reference_order = prepare_ordered_wishes(edited, use_equivalence_classes)
        if reference_order.empty:
            st.error("Add at least one valid program before running the simulation.")
            st.stop()

        simulation_result = None

        if use_equivalence_classes:
            total_orders = count_equivalence_orders(reference_order)
            if total_orders > MAX_EXACT_EQUIV_PERMUTATIONS:
                st.error(
                    f"The equivalence classes generate {total_orders:,} strict orders. "
                    f"This is above the exact-evaluation limit of {MAX_EXACT_EQUIV_PERMUTATIONS:,}. "
                    "Split large equivalence groups into smaller groups, then run the simulation again."
                )
                st.stop()

            variants = []
            reference_choices = None
            reference_order_used = None

            for idx, strict_order in enumerate(iter_equivalence_orders(reference_order), start=1):
                wishes_for_compute = attach_mtb_hashes(strict_order, program_mapping, national_student_id)
                choices = compute(wishes_for_compute, program_mapping)
                outcome, p_unmatched, at_risk = predicted_outcome_from_choices(choices, threshold)

                if idx == 1:
                    reference_choices = choices
                    reference_order_used = strict_order

                variants.append({
                    "Strict order #": idx,
                    "Predicted outcome": outcome,
                    "Unmatched risk": p_unmatched,
                    "Flagged at risk": at_risk,
                    "Strict order": compact_order_label(strict_order),
                })

            variants_df = pd.DataFrame(variants)
            distinct_outcomes = sorted(variants_df["Predicted outcome"].unique().tolist())
            simulation_result = {
                "mode": "equivalence",
                "threshold": threshold,
                "reference_choices": reference_choices,
                "variants_df": variants_df,
                "distinct_outcomes": distinct_outcomes,
            }

            st.subheader("Reference strict-order details")
            st.caption(
                "This table uses the current row order inside each equivalence group as the reference order. "
                "The sensitivity test below evaluates every strict order compatible with the groups."
            )
            st.dataframe(format_choices_table(reference_choices), width="stretch", hide_index=True)
            render_single_summary(reference_choices, threshold)

            st.subheader("Equivalence-class sensitivity")
            if len(distinct_outcomes) == 1:
                st.success(
                    "The strict ordering inside the equivalence classes does not change the predicted final outcome. "
                    f"All {len(variants_df):,} compatible strict order(s) lead to: **{distinct_outcomes[0]}**."
                )
            else:
                st.warning(
                    "The strict ordering inside at least one equivalence class can change the predicted final outcome. "
                    "The user should choose a strict order carefully for the tied programs."
                )

            outcome_summary = (
                variants_df
                .groupby("Predicted outcome", as_index=False)
                .agg(
                    strict_orders=("Strict order #", "count"),
                    min_unmatched_risk=("Unmatched risk", "min"),
                    max_unmatched_risk=("Unmatched risk", "max"),
                )
                .sort_values(["strict_orders", "Predicted outcome"], ascending=[False, True])
            )
            outcome_summary["Share of strict orders"] = outcome_summary["strict_orders"] / len(variants_df)
            for col in ["min_unmatched_risk", "max_unmatched_risk", "Share of strict orders"]:
                outcome_summary[col] = outcome_summary[col].map(lambda x: f"{x:.1%}")
            outcome_summary = outcome_summary.rename(columns={
                "strict_orders": "Strict orders",
                "min_unmatched_risk": "Min unmatched risk",
                "max_unmatched_risk": "Max unmatched risk",
            })
            st.dataframe(outcome_summary, width="stretch", hide_index=True)

            with st.expander("All strict orders tested", expanded=False):
                variants_display = variants_df.copy()
                variants_display["Unmatched risk"] = variants_display["Unmatched risk"].map(lambda x: f"{x:.1%}")
                st.dataframe(variants_display, width="stretch", hide_index=True)

        else:
            strict_order = prepare_ordered_wishes(edited, use_equivalence_classes=False)
            wishes_for_compute = attach_mtb_hashes(strict_order, program_mapping, national_student_id)
            choices = compute(wishes_for_compute, program_mapping)
            simulation_result = {
                "mode": "strict",
                "threshold": threshold,
                "choices": choices,
            }

            st.subheader("Wish-level details")
            st.caption(
                "Chance if considered is the chance of getting that program if the student reaches that wish. "
                "Final chance of assignment also accounts for all higher-ranked wishes."
            )
            st.dataframe(format_choices_table(choices), width="stretch", hide_index=True)
            render_single_summary(choices, threshold)

        st.session_state[simulation_result_key] = simulation_result
        st.session_state[simulation_done_key] = True
        calculated_now = True

    except ValueError as exc:
        st.error(str(exc))

    except Exception as exc:
        st.error("Unexpected error during the simulation.")
        st.exception(exc)

if st.session_state.get(simulation_done_key, False):
    if not calculated_now:
        render_simulation_result(st.session_state.get(simulation_result_key, {}))

    render_similar_program_recommendations(
        edited,
        program_mapping,
        editor_state_key=editor_state_key,
        editor_widget_key_base=editor_widget_key_base,
        use_equivalence_classes=use_equivalence_classes,
        simulation_done_key=simulation_done_key,
        simulation_result_key=simulation_result_key,
    )
else:
    st.subheader("4. Recommended similar programs")
    st.info("Run the simulation first to unlock similar-program recommendations.")
