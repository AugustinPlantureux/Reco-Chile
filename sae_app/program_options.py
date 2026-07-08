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
from sae_app.data_loading import program_matches_filters, region_sort_index
from sae_app.text_utils import as_float, clean_recommendation_value, clean_text, parse_coordinate, series_value


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
    capacity: float
    true_applicants: float

    @classmethod
    def from_series(cls, row: pd.Series, *, label: str = "") -> "ProgramRecord":
        return cls(
            label=str(label),
            raw=row,
            rbd=str(series_value(row, "rbd", "")).strip(),
            program_code=str(series_value(row, "program_code", "")).strip(),
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
            capacity=as_float(series_value(row, CAPACITY, 0), 0.0),
            true_applicants=as_float(series_value(row, TRUE_APP, 0), 0.0),
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
