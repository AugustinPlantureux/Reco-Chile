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
    "region_sort_index",
    "required_cols",
    "translate_filter_value",
    "validate_core_numeric_columns",
    "validate_cumulative_share_columns",
]
