"""Static configuration: data columns, thresholds, file paths, and dropdown options.

Nothing in this module reads a file, calls Streamlit, or does any computation.
It only defines names and values that the rest of the app agrees on.
"""

from __future__ import annotations

import os
from pathlib import Path

# Set to True during local development to display full Python tracebacks.
APP_DEBUG = False

# Selects the data-loading/validation backend in sae_app/data_loading.py:
# False (default) -> sae_app.data_loading_pandas, the original pandas
#   implementation. Byte-for-byte the same behavior as before DuckDB was
#   introduced, and does not require the duckdb package to be installed.
# True -> sae_app.data_loading_duckdb, which runs the same joins/validation
#   as SQL through DuckDB.
# Overridable via the SAE_USE_DUCKDB environment variable so this can be
# toggled per-deployment without a code change.
USE_DUCKDB = os.environ.get("SAE_USE_DUCKDB", "false").strip().lower() in {"1", "true", "yes"}

# ---------------------------------------------------------------------------
# Data columns
# ---------------------------------------------------------------------------
WISH_RANK    = "wish_rank"
PROGRAM      = "program"
EQUIV_GROUP  = "preference_group"
LOTTERY      = "lottery_number"
HASH_PCT     = "lottery_hash_percentile"

CAPACITY     = "total_admission_seats"
TRUE_APP     = "true_applicants_last_year"
POP          = "program_lottery_population_2024"
IMPUTED      = "calibration_2024_imputed"
IMPUT_METHOD = "calibration_2024_imputation_method"

PRIORITY_STUDENT_SEATS = "priority_student_seats"

# Hard-coded unmatched-risk thresholds.
# Change these values directly in the code if you want another calibration.
# Hard threshold: if the unmatched risk reaches this level, Unmatched is shown
# as the first predicted outcome.
# Soft threshold: if the unmatched risk is between the soft and hard thresholds,
# Unmatched is shown in the podium as a warning signal, but not forced first.
HARD_UNMATCHED_THRESHOLD = 0.027   # 2.7%: strong unmatched-risk alert
SOFT_UNMATCHED_THRESHOLD = 0.004   # 0.4%: lighter podium warning
MAX_EXACT_EQUIV_PERMUTATIONS = 10000
# If compatible strict orders keep the same predicted school but change its
# final chance by at least 0.5 percentage point, show an intermediate warning.
EQUIV_PROBABILITY_CHANGE_WARNING_THRESHOLD = 0.005

PRIORITIES = [
    "priority_sibling",
    "priority_student",
    "priority_parent_civil_servant",
    "priority_ex_student",
]
SAFETY      = "priority_already_registered"
NO_PRIORITY = "no_priority"
TIERS       = PRIORITIES + [NO_PRIORITY]

MAX_SHA256 = 2 ** 256 - 1

REGION = "Region"
UNKNOWN_REGION = "Unknown region"

# ---------------------------------------------------------------------------
# External data files
# ---------------------------------------------------------------------------
# constants.py lives in <project_root>/sae_app/, so the project root is one
# level up from this file's parent directory.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CAPACITIES_PATH = DATA_DIR / "capacities_2025_wta_with_2024_calibration.csv"
PROGRAM_NAMES_PATH = DATA_DIR / "programmes_chili_criteres_recommandation.csv"
RBD_REGION_PATH = DATA_DIR / "rbd_region_map.csv"
PROGRAM_FILTERS_PATH = DATA_DIR / "program_filters.csv"
COMMUNE_COORDINATES_PATH = DATA_DIR / "commune_coordinates.csv"

GEOCODING_TIMEOUT_SECONDS = 8
GEOCODING_USER_AGENT = "sae-admission-risk-simulator/1.0"
# Nominatim's usage policy caps free usage at 1 request/second. geo.py enforces
# this within a single Python process, regardless of how many Streamlit sessions
# request geocoding concurrently in that process. Multi-worker deployments need
# shared throttling or a dedicated geocoding service.
NOMINATIM_MIN_INTERVAL_SECONDS = 1.0

# Bounding boxes used to accept coordinates from Chilean territory represented
# in the program data. Keeping these zones in one place prevents the ingestion
# and distance layers from disagreeing about mainland and insular coordinates.
CHILE_COORDINATE_ZONES = (
    ("mainland_chile", -56.0, -17.0, -76.0, -66.0),
    ("rapa_nui", -27.3, -27.0, -109.6, -109.1),
    ("juan_fernandez", -34.0, -33.4, -81.0, -78.5),
)

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

# Approximate region centroids used only as a fallback when commune/program
# coordinates are unavailable. Add data/commune_coordinates.csv with columns
# commune, region, latitude, longitude to get a much better proximity score.
REGION_CENTROIDS = {
    "Región de Arica y Parinacota": (-18.4783, -70.3126),
    "Región de Tarapacá": (-20.2133, -70.1528),
    "Región de Antofagasta": (-23.6509, -70.3975),
    "Región de Atacama": (-27.3668, -70.3323),
    "Región de Coquimbo": (-29.9533, -71.3436),
    "Región de Valparaíso": (-33.0472, -71.6127),
    "Región Metropolitana de Santiago": (-33.4489, -70.6693),
    "Región del Libertador Bernardo O'Higgins": (-34.1708, -70.7444),
    "Región del Maule": (-35.4264, -71.6554),
    "Región de Ñuble": (-36.6063, -72.1034),
    "Región del Bío-Bío": (-36.8201, -73.0444),
    "Región de La Araucanía": (-38.7359, -72.5904),
    "Región de Los Ríos": (-39.8142, -73.2459),
    "Región de Los Lagos": (-41.4693, -72.9424),
    "Región de Aysén del Gral.Ibañez del Campo": (-45.5712, -72.0685),
    "Región de Magallanes y Antártica Chilena": (-53.1638, -70.9171),
}

# RBD -> Region lookup is loaded from data/rbd_region_map.csv.

# Program-characteristic column names.
# Values are loaded from data/program_filters.csv and
# data/programmes_chili_criteres_recommandation.csv.
PROGRAM_TRACK = "program_track"
PROGRAM_SPECIALTY_SECTOR = "program_specialty_sector"
PROGRAM_SPECIALTY_NAME = "program_specialty_name"
PROGRAM_GENDER = "program_gender"
PROGRAM_SCHOOL_DAY = "program_school_day"
UNKNOWN_FILTER_VALUE = "Unknown"

PROGRAM_DISPLAY_NAME = "program_display_name"
SCHOOL_NAME = "school_name"
SCHOOL_COMMUNE = "school_commune"
PROGRAM_LATITUDE = "program_latitude"
PROGRAM_LONGITUDE = "program_longitude"
UNKNOWN_PROGRAM_NAME = "Program details unavailable"
UNKNOWN_SCHOOL_NAME = "School name unavailable"

PROGRAM_RURALITY = "program_rurality"
PROGRAM_PIE = "program_pie"
PROGRAM_PACE = "program_pace"
PROGRAM_ENROLLMENT_FEE = "program_enrollment_fee"
PROGRAM_MONTHLY_FEE = "program_monthly_fee"
PROGRAM_RELIGIOUS_ORIENTATION = "program_religious_orientation"

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
