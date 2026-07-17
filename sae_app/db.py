"""DuckDB connection and reusable SQL-side data-loading/validation helpers.

Row-level operations that used to live entirely in pandas -- required-code
normalization, duplicate-key conflict detection, table joins, coordinate
coalescing/quality checks, and the calibration validation queries -- run as
SQL through the connection built here.

Pure per-string display formatting (accent stripping, descriptor
dictionaries, title-casing exceptions) stays as plain Python, because it is
lookup-table/branching logic over free text, not a table-shaped operation --
forcing it into SQL would mean a large, more fragile CASE/replace chain for
no real benefit. Callers register that logic as DuckDB scalar UDFs (see
register_text_udf), so it still executes as part of the same SQL query
instead of a separate pandas post-processing pass.
"""

from __future__ import annotations

from typing import Callable

import duckdb
import pandas as pd


def connect() -> duckdb.DuckDBPyConnection:
    """Return a fresh in-memory DuckDB connection.

    A new connection per load keeps this safe to call from multiple Streamlit
    sessions/threads without shared mutable state; the calibration data is
    small enough that connection setup cost is negligible next to CSV parsing.
    """
    return duckdb.connect(":memory:")


def as_object_dtype(df: pd.DataFrame) -> pd.DataFrame:
    """Cast pandas StringDtype columns from a DuckDB .df() result back to 'object'.

    DuckDB's Python client returns VARCHAR columns using pandas' newer
    StringDtype by default. The rest of this codebase -- and pandas'
    Series.equals(), used in caching/testing -- expects plain 'object' dtype
    for text columns, matching pre-DuckDB pandas behavior; values are
    unaffected, only the column's dtype label.
    """
    for col in df.columns:
        if isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].astype(object)
    return df


def register_text_udf(
    con: duckdb.DuckDBPyConnection,
    name: str,
    func: Callable[..., str],
    *,
    arg_types: list[str],
) -> None:
    """Register a Python string-formatting function as a scalar SQL function."""
    con.create_function(name, func, arg_types, "VARCHAR")


# ---------------------------------------------------------------------------
# Required positive-integer identifier normalization (rbd / program_code)
# ---------------------------------------------------------------------------
# Mirrors the previous _normalize_required_code_series: '="123"' Excel-formula
# quoting is unwrapped, values are matched against ^[0-9]+(?:[.,]0+)?$, and
# leading zeroes are stripped as text so an unexpectedly long malformed
# identifier still resolves deterministically instead of overflowing int().

_CODE_MATCH_CASE = """
    CASE
        WHEN regexp_matches(trim("{col}"), '^="[0-9]+([.,]0+)?"$')
            THEN regexp_extract(trim("{col}"), '^="([0-9]+)([.,]0+)?"$', 1)
        WHEN regexp_matches(trim("{col}"), '^[0-9]+([.,]0+)?$')
            THEN regexp_extract(trim("{col}"), '^([0-9]+)([.,]0+)?$', 1)
        ELSE NULL
    END
"""


def normalized_code_expr(column: str) -> str:
    """Return a SQL expression yielding the normalized code or NULL if invalid."""
    digits = _CODE_MATCH_CASE.format(col=column)
    return f"CASE WHEN ({digits}) IS NULL THEN NULL ELSE NULLIF(ltrim({digits}, '0'), '') END"


def normalize_required_code_column(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    column: str,
    *,
    field_name: str,
    source_name: str,
) -> pd.Series:
    """Normalize a required positive-integer identifier column, or raise DataSchemaError.

    Deferred-imports DataSchemaError from data_loading to avoid a circular
    import (data_loading imports this module at the top level).
    """
    from sae_app.data_loading import DataSchemaError

    con.register("_normalize_src", df[[column]].reset_index(drop=True))
    query = f"""
        SELECT
            ROW_NUMBER() OVER () + 1 AS csv_row,
            "{column}" AS raw_value,
            {normalized_code_expr(column)} AS normalized_code
        FROM _normalize_src
    """
    result = con.sql(query).df()
    con.unregister("_normalize_src")

    invalid = result[result["normalized_code"].isna()]
    if not invalid.empty:
        examples = [
            f"row {int(r['csv_row'])}: {r['raw_value']!r}"
            for _, r in invalid.head(5).iterrows()
        ]
        shown = "; ".join(examples)
        remaining = len(invalid) - 5
        suffix = f"; and {remaining} more" if remaining > 0 else ""
        raise DataSchemaError(
            f"{source_name} contains invalid {field_name} value(s): {shown}{suffix}. "
            "Expected a positive integer identifier."
        )

    return pd.Series(result["normalized_code"].tolist(), index=df.index, dtype="object")


# ---------------------------------------------------------------------------
# Duplicate-key conflict detection
# ---------------------------------------------------------------------------
# Mirrors the previous _drop_exact_duplicates_or_raise_conflicts: rows sharing
# a key are allowed only if every other column is identical after whitespace
# normalization; otherwise the conflicting keys are reported before any
# deduplication happens.

def drop_exact_duplicates_or_raise_conflicts(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    key_columns: list[str],
    *,
    source_name: str,
) -> pd.DataFrame:
    from sae_app.data_loading import DataSchemaError

    if not df.duplicated(key_columns, keep=False).any():
        return as_object_dtype(df)

    value_columns = [c for c in df.columns if c not in key_columns]
    working = df.reset_index(drop=True).copy()
    working["_orig_order"] = range(len(working))
    con.register("_dedupe_src", working)

    key_list = ", ".join(f'"{c}"' for c in key_columns)
    norm_value_cols = ", ".join(
        f"""regexp_replace(trim(coalesce(CAST("{c}" AS VARCHAR), '')), '\\s+', ' ', 'g') AS "{c}_norm\""""
        for c in value_columns
    )
    norm_value_col_names = ", ".join(f'"{c}_norm"' for c in value_columns) if value_columns else "NULL"

    conflict_sql = f"""
        WITH normalized AS (
            SELECT {key_list}{"," if value_columns else ""} {norm_value_cols}
            FROM _dedupe_src
        ),
        distinct_combos AS (
            SELECT DISTINCT {key_list}{"," if value_columns else ""} {norm_value_col_names}
            FROM normalized
        )
        SELECT {key_list}, COUNT(*) AS distinct_count
        FROM distinct_combos
        GROUP BY {key_list}
        HAVING COUNT(*) > 1
    """
    conflicts = con.sql(conflict_sql).df()

    if not conflicts.empty:
        con.unregister("_dedupe_src")
        examples = ", ".join(
            "/".join(str(row[c]) for c in key_columns)
            for _, row in conflicts.head(5).iterrows()
        )
        raise DataSchemaError(
            f"{source_name} contains conflicting duplicate rows for key(s) "
            f"{', '.join(key_columns)}: {examples}"
        )

    dedup_sql = f"""
        SELECT * EXCLUDE (_orig_order, _rn)
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY {key_list} ORDER BY _orig_order) AS _rn
            FROM _dedupe_src
        )
        WHERE _rn = 1
        ORDER BY _orig_order
    """
    deduped = con.sql(dedup_sql).df()
    con.unregister("_dedupe_src")
    deduped = deduped.drop(columns=["_orig_order"], errors="ignore").reset_index(drop=True)
    return as_object_dtype(deduped)


# ---------------------------------------------------------------------------
# Left join, preserving the left table's row order and column order
# ---------------------------------------------------------------------------

def left_join_preserving_order(
    con: duckdb.DuckDBPyConnection,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: list[str],
) -> pd.DataFrame:
    """LEFT JOIN two DataFrames on `on`, preserving left_df's row order/columns.

    Mirrors left_df.merge(right_df, on=on, how="left"): every left row is kept
    exactly once (right_df's join keys must already be unique -- callers run
    drop_exact_duplicates_or_raise_conflicts upstream), left's column order
    comes first, and only right_df's non-key columns are appended.
    """
    left = left_df.reset_index(drop=True).copy()
    left["_row_order"] = range(len(left))
    con.register("_ljl", left)
    con.register("_ljr", right_df)

    on_list = ", ".join(f'"{c}"' for c in on)
    right_extra_cols = [c for c in right_df.columns if c not in on]
    left_cols = [c for c in left.columns if c != "_row_order"]
    select_list = ", ".join(f'_ljl."{c}"' for c in left_cols)
    if right_extra_cols:
        select_list += ", " + ", ".join(f'_ljr."{c}"' for c in right_extra_cols)

    query = f"""
        SELECT {select_list}
        FROM _ljl
        LEFT JOIN _ljr USING ({on_list})
        ORDER BY _ljl._row_order
    """
    result = con.sql(query).df()
    con.unregister("_ljl")
    con.unregister("_ljr")
    return as_object_dtype(result.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Haversine distance, used both for coordinate-quality checks and the
# recommendation engine's proximity scoring.
# ---------------------------------------------------------------------------

def register_haversine_km(con: duckdb.DuckDBPyConnection) -> None:
    """Register haversine_km(lat1, lon1, lat2, lon2) as a native SQL macro."""
    con.sql("""
        CREATE OR REPLACE MACRO haversine_km(lat1, lon1, lat2, lon2) AS
            2 * 6371.0 * asin(sqrt(
                pow(sin(radians(lat2 - lat1) / 2), 2)
                + cos(radians(lat1)) * cos(radians(lat2))
                  * pow(sin(radians(lon2 - lon1) / 2), 2)
            ))
    """)


# ---------------------------------------------------------------------------
# Coordinate coalescing and per-RBD coordinate-quality spread
# ---------------------------------------------------------------------------

def parse_coordinate_expr(column: str) -> str:
    """SQL expression parsing a possibly comma-decimal coordinate to a finite DOUBLE or NULL."""
    cleaned = f"""try_cast(replace(trim(CAST("{column}" AS VARCHAR)), ',', '.') AS DOUBLE)"""
    return f"(CASE WHEN {cleaned} IS NOT NULL AND isfinite({cleaned}) THEN {cleaned} ELSE NULL END)"


def _chile_zone_sql(lat_expr: str, lon_expr: str) -> str:
    """SQL predicate: true iff (lat_expr, lon_expr) fall inside a Chile coordinate zone."""
    from sae_app.constants import CHILE_COORDINATE_ZONES

    zones = " OR ".join(
        f"({lat_expr} BETWEEN {min_lat} AND {max_lat} AND {lon_expr} BETWEEN {min_lon} AND {max_lon})"
        for _, min_lat, max_lat, min_lon, max_lon in CHILE_COORDINATE_ZONES
    )
    return f"(({zones}) AND {lat_expr} IS NOT NULL AND {lon_expr} IS NOT NULL)"


def coalesce_program_coordinates(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    column_pairs: list[tuple[str, str, str]],
    *,
    latitude_out: str,
    longitude_out: str,
    source_out: str,
) -> pd.DataFrame:
    """Select the first Chile-zone-valid coordinate pair per row, by priority order.

    Mirrors the previous _coalesce_program_coordinates: latitude and longitude
    are never chosen independently (a whole pair must be jointly valid), and
    the first valid pair in `column_pairs` order wins -- a plain top-to-bottom
    SQL CASE WHEN has exactly that "first match wins" semantics. Returns df's
    original columns plus latitude_out/longitude_out/source_out.
    """
    available = [
        (lat, lon, source) for lat, lon, source in column_pairs
        if lat in df.columns and lon in df.columns
    ]

    working = df.reset_index(drop=True).copy()
    working["_row_order"] = range(len(working))
    con.register("_coord_src", working)

    if not available:
        query = f"""
            SELECT *, CAST(NULL AS DOUBLE) AS "{latitude_out}",
                   CAST(NULL AS DOUBLE) AS "{longitude_out}", '' AS "{source_out}"
            FROM _coord_src ORDER BY _row_order
        """
    else:
        lat_case, lon_case, source_case = "CASE", "CASE", "CASE"
        for lat_col, lon_col, source in available:
            lat_expr = parse_coordinate_expr(lat_col)
            lon_expr = parse_coordinate_expr(lon_col)
            cond = _chile_zone_sql(lat_expr, lon_expr)
            lat_case += f" WHEN {cond} THEN {lat_expr}"
            lon_case += f" WHEN {cond} THEN {lon_expr}"
            source_case += f" WHEN {cond} THEN '{source}'"
        lat_case += " ELSE NULL END"
        lon_case += " ELSE NULL END"
        source_case += " ELSE '' END"

        query = f"""
            SELECT *, {lat_case} AS "{latitude_out}", {lon_case} AS "{longitude_out}",
                   {source_case} AS "{source_out}"
            FROM _coord_src
            ORDER BY _row_order
        """

    result = con.sql(query).df()
    con.unregister("_coord_src")
    return as_object_dtype(result.drop(columns=["_row_order"]).reset_index(drop=True))


def rbd_coordinate_spread_km(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    *,
    rbd_column: str,
    latitude_column: str,
    longitude_column: str,
    source_column: str,
    school_coordinate_label: str,
    output_column: str,
) -> pd.DataFrame:
    """Add a column: max pairwise school-coordinate distance among an RBD's points.

    Mirrors the previous _rbd_coordinate_spread_km: for each RBD, the maximum
    great-circle distance between any two distinct valid
    school_coordinate_label-sourced points sharing that RBD. 0.0 for an RBD
    with 0 or 1 distinct points; NULL for an RBD with no such points at all.
    """
    register_haversine_km(con)
    working = df.reset_index(drop=True).copy()
    working["_row_order"] = range(len(working))
    con.register("_spread_src", working)

    query = f"""
        WITH valid_points AS (
            SELECT DISTINCT "{rbd_column}" AS rbd,
                "{latitude_column}" AS lat, "{longitude_column}" AS lon
            FROM _spread_src
            WHERE "{source_column}" = '{school_coordinate_label}'
              AND "{latitude_column}" IS NOT NULL AND "{longitude_column}" IS NOT NULL
        ),
        pairwise AS (
            SELECT a.rbd AS rbd, haversine_km(a.lat, a.lon, b.lat, b.lon) AS distance_km
            FROM valid_points a
            JOIN valid_points b
                ON a.rbd = b.rbd AND (a.lat, a.lon) < (b.lat, b.lon)
        ),
        spread_by_rbd AS (
            SELECT valid_points.rbd AS rbd, COALESCE(MAX(distance_km), 0.0) AS spread_km
            FROM valid_points
            LEFT JOIN pairwise USING (rbd)
            GROUP BY valid_points.rbd
        )
        SELECT s.*, spread_by_rbd.spread_km AS "{output_column}"
        FROM _spread_src s
        LEFT JOIN spread_by_rbd ON s."{rbd_column}" = spread_by_rbd.rbd
        ORDER BY s._row_order
    """
    result = con.sql(query).df()
    con.unregister("_spread_src")
    return as_object_dtype(result.drop(columns=["_row_order"]).reset_index(drop=True))
