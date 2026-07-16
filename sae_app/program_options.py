"""ProgramRecord + building and filtering the program dropdown.

ProgramRecord is a typed, read-only view of one program row. The core
simulation (mtb_engine.py) still receives the original pandas row for
backwards compatibility, but the recommendation engine and geo code use
ProgramRecord's explicit attributes instead of repeated row.get(...) calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sae_app.constants import (
    CAPACITY,
    PROGRAM_DISPLAY_NAME,
    PROGRAM_ENROLLMENT_FEE,
    PROGRAM_GENDER,
    PROGRAM_LATITUDE,
    PROGRAM_LONGITUDE,
    PROGRAM_MONTHLY_FEE,
    PROGRAM_PACE,
    PROGRAM_PIE,
    PROGRAM_RELIGIOUS_ORIENTATION,
    PROGRAM_RURALITY,
    PROGRAM_SCHOOL_DAY,
    PROGRAM_SPECIALTY_NAME,
    PROGRAM_SPECIALTY_SECTOR,
    PROGRAM_TRACK,
    REGION,
    SCHOOL_COMMUNE,
    SCHOOL_NAME,
    TRUE_APP,
    UNKNOWN_PROGRAM_NAME,
    UNKNOWN_REGION,
    UNKNOWN_SCHOOL_NAME,
)
from sae_app.data_loading import (
    COORDINATE_DISCREPANCY_KM,
    PROGRAM_COORDINATE_SOURCE,
    PROGRAM_GEO_MATCH_LEVEL,
    RBD_COORDINATE_SPREAD_KM,
    program_matches_filters,
    region_sort_index,
)
from sae_app.errors import CandidateEvaluationError
from sae_app.text_utils import clean_recommendation_value, clean_text, parse_coordinate, series_value


def _optional_nonnegative_metric(value) -> float:
    """Parse an optional non-negative quality metric, otherwise return NaN."""
    if pd.isna(value):
        return np.nan
    try:
        parsed = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(parsed) or parsed < 0:
        return np.nan
    return parsed


def _required_nonnegative_candidate_number(
    row: pd.Series,
    column: str,
    *,
    label: str,
) -> float:
    """Read one required candidate number without silently coercing bad data."""
    raw_value = series_value(row, column, np.nan)
    try:
        value = float(str(raw_value).strip().replace(",", "."))
    except (TypeError, ValueError) as exc:
        raise CandidateEvaluationError(
            f"{label or '<unknown program>'}: invalid {column} value {raw_value!r}."
        ) from exc

    if not np.isfinite(value) or value < 0:
        raise CandidateEvaluationError(
            f"{label or '<unknown program>'}: {column} must be finite and non-negative."
        )
    return value


@dataclass(frozen=True)
class ProgramRecord:
    """Typed view of one program row used by the recommendation engine.

    The core simulation still receives the original pandas row for backwards
    compatibility, but recommendation code uses explicit attributes instead of
    repeated row.get(...) calls.
    """
    label: str
    raw: pd.Series
    rbd: str
    program_code: str
    region: str
    school_name: str
    school_commune: str
    program_display_name: str
    program_track: str
    program_specialty_sector: str
    program_specialty_name: str
    program_gender: str
    program_school_day: str
    program_rurality: str
    program_pie: str
    program_pace: str
    program_enrollment_fee: str
    program_monthly_fee: str
    program_religious_orientation: str
    program_latitude: float
    program_longitude: float
    program_coordinate_source: str
    program_geo_match_level: str
    coordinate_discrepancy_km: float
    rbd_coordinate_spread_km: float
    capacity: float
    true_applicants: float

    @classmethod
    def from_series(cls, row: pd.Series, *, label: str = "") -> "ProgramRecord":
        rbd = str(series_value(row, "rbd", "")).strip()
        program_code = str(series_value(row, "program_code", "")).strip()
        if not rbd or not program_code:
            raise CandidateEvaluationError(
                f"{label or '<unknown program>'}: missing rbd or program_code."
            )

        capacity = _required_nonnegative_candidate_number(row, CAPACITY, label=label)
        true_applicants = _required_nonnegative_candidate_number(row, TRUE_APP, label=label)

        return cls(
            label=str(label),
            raw=row,
            rbd=rbd,
            program_code=program_code,
            region=clean_text(series_value(row, REGION, UNKNOWN_REGION), default=UNKNOWN_REGION),
            school_name=clean_text(series_value(row, SCHOOL_NAME, "")),
            school_commune=clean_text(series_value(row, SCHOOL_COMMUNE, "")),
            program_display_name=clean_text(series_value(row, PROGRAM_DISPLAY_NAME, "")),
            program_track=clean_text(series_value(row, PROGRAM_TRACK, "")),
            program_specialty_sector=clean_text(series_value(row, PROGRAM_SPECIALTY_SECTOR, "")),
            program_specialty_name=clean_text(series_value(row, PROGRAM_SPECIALTY_NAME, "")),
            program_gender=clean_text(series_value(row, PROGRAM_GENDER, "")),
            program_school_day=clean_text(series_value(row, PROGRAM_SCHOOL_DAY, "")),
            program_rurality=clean_text(series_value(row, PROGRAM_RURALITY, "")),
            program_pie=clean_text(series_value(row, PROGRAM_PIE, "")),
            program_pace=clean_text(series_value(row, PROGRAM_PACE, "")),
            program_enrollment_fee=clean_text(series_value(row, PROGRAM_ENROLLMENT_FEE, "")),
            program_monthly_fee=clean_text(series_value(row, PROGRAM_MONTHLY_FEE, "")),
            program_religious_orientation=clean_text(series_value(row, PROGRAM_RELIGIOUS_ORIENTATION, "")),
            program_latitude=parse_coordinate(series_value(row, PROGRAM_LATITUDE, np.nan)),
            program_longitude=parse_coordinate(series_value(row, PROGRAM_LONGITUDE, np.nan)),
            program_coordinate_source=clean_text(
                series_value(row, PROGRAM_COORDINATE_SOURCE, "")
            ),
            program_geo_match_level=clean_text(
                series_value(row, PROGRAM_GEO_MATCH_LEVEL, "")
            ),
            coordinate_discrepancy_km=_optional_nonnegative_metric(
                series_value(row, COORDINATE_DISCREPANCY_KM, np.nan)
            ),
            rbd_coordinate_spread_km=_optional_nonnegative_metric(
                series_value(row, RBD_COORDINATE_SPREAD_KM, np.nan)
            ),
            capacity=capacity,
            true_applicants=true_applicants,
        )

    def criterion_value(self, col: str) -> str:
        values = {
            PROGRAM_TRACK: self.program_track,
            PROGRAM_SPECIALTY_SECTOR: self.program_specialty_sector,
            PROGRAM_SPECIALTY_NAME: self.program_specialty_name,
            PROGRAM_GENDER: self.program_gender,
            PROGRAM_SCHOOL_DAY: self.program_school_day,
            PROGRAM_RURALITY: self.program_rurality,
            PROGRAM_PIE: self.program_pie,
            PROGRAM_PACE: self.program_pace,
            PROGRAM_ENROLLMENT_FEE: self.program_enrollment_fee,
            PROGRAM_MONTHLY_FEE: self.program_monthly_fee,
            PROGRAM_RELIGIOUS_ORIENTATION: self.program_religious_orientation,
        }
        if col not in values:
            raise KeyError(f"Unknown recommendation criterion: {col}")
        return clean_recommendation_value(values[col])


# ---------------------------------------------------------------------------
# Build program options
# ---------------------------------------------------------------------------

def _clean_label_part(value) -> str:
    """Return a family-facing label part, without placeholder/null text."""
    text = clean_text(value)
    if not text or text.lower() == "nan":
        return ""
    return text


def _school_label_base(row: pd.Series) -> str:
    """Return the shortest useful school name for a program row."""
    rbd = str(row["rbd"]).strip()
    school_name = _clean_label_part(row.get(SCHOOL_NAME, ""))
    if school_name and school_name != UNKNOWN_SCHOOL_NAME:
        return school_name
    return f"RBD {rbd}"


def _program_detail_label(row: pd.Series) -> str:
    """Return the shortest useful program detail for disambiguation."""
    code = str(row["program_code"]).strip()
    display_name = _clean_label_part(row.get(PROGRAM_DISPLAY_NAME, ""))
    if display_name and display_name != UNKNOWN_PROGRAM_NAME:
        return display_name
    return f"Program code {code}"


def compact_program_label(label: str) -> str:
    """Simplify legacy full labels for family-facing display.

    New labels are already compact and only include program details when needed
    to distinguish two options. Older labels always included program details and
    RBD; those are shortened here when they reach result summaries or the wish
    cards.
    """
    text = str(label or "").strip()
    if not text:
        return ""
    if " · RBD " in text:
        before_rbd = text.split(" · RBD ", 1)[0].strip()
        if " — " in before_rbd:
            school_part, detail_part = before_rbd.split(" — ", 1)
            # Legacy labels packed several metadata fields before the RBD
            # suffix, e.g. "General H-C · Mixed · Full day". New compact
            # labels may legitimately keep "· RBD" to disambiguate schools,
            # so only strip legacy multi-field detail blocks.
            if " · " in detail_part:
                return school_part.strip()
    return text


def make_program_option_label(
    row: pd.Series,
    *,
    school_name_count: int = 1,
    program_count_for_school: int = 1,
    duplicate_count: int = 1,
) -> str:
    """Build a compact but still uniquely identifiable dropdown label.

    Families do not need to see the program track, commune, and RBD repeated on
    every line. The label starts with the school name and adds details only when
    they are needed to distinguish otherwise similar options.
    """
    rbd = str(row["rbd"]).strip()
    code = str(row["program_code"]).strip()
    commune = _clean_label_part(row.get(SCHOOL_COMMUNE, ""))

    label = _school_label_base(row)

    if school_name_count > 1 and commune:
        label = f"{label} ({commune})"

    if program_count_for_school > 1:
        label = f"{label} — {_program_detail_label(row)}"

    if duplicate_count > 1:
        label = f"{label} · RBD {rbd}"

    # Extremely rare final safeguard: two rows can share school, commune, RBD,
    # and display name while still being distinct program codes.
    if duplicate_count > 1 and code:
        label = f"{label} · code {code}"

    return label


def build_options(calib: pd.DataFrame) -> tuple[list[str], dict[str, pd.Series]]:
    options, mapping = [], {}

    unique_programs = calib.drop_duplicates(["rbd", "program_code"]).copy()
    unique_programs["_region_sort"] = unique_programs[REGION].map(region_sort_index)
    unique_programs["_rbd_sort"] = pd.to_numeric(unique_programs["rbd"], errors="coerce")
    unique_programs["_program_sort"] = pd.to_numeric(unique_programs["program_code"], errors="coerce")
    unique_programs["_school_label_base"] = unique_programs.apply(_school_label_base, axis=1)

    school_name_counts = unique_programs["_school_label_base"].value_counts().to_dict()
    program_counts_by_school = unique_programs.groupby("_school_label_base")["program_code"].nunique().to_dict()

    unique_programs["_base_display_label"] = unique_programs.apply(
        lambda row: make_program_option_label(
            row,
            school_name_count=school_name_counts.get(row["_school_label_base"], 1),
            program_count_for_school=program_counts_by_school.get(row["_school_label_base"], 1),
            duplicate_count=1,
        ),
        axis=1,
    )
    duplicate_counts = unique_programs["_base_display_label"].value_counts().to_dict()

    unique_programs = unique_programs.sort_values(
        ["_region_sort", "_rbd_sort", "_program_sort", REGION, "rbd", "program_code"]
    )

    seen_labels: set[str] = set()
    for _, row in unique_programs.iterrows():
        base_label = row["_base_display_label"]
        label = make_program_option_label(
            row,
            school_name_count=school_name_counts.get(row["_school_label_base"], 1),
            program_count_for_school=program_counts_by_school.get(row["_school_label_base"], 1),
            duplicate_count=duplicate_counts.get(base_label, 1),
        )
        if label in seen_labels:
            label = f"{label} · code {str(row['program_code']).strip()}"
        seen_labels.add(label)
        options.append(label)
        mapping[label] = row

    return options, mapping


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
