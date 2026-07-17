"""Small, dependency-free text and number cleaning helpers.

Every function here only depends on pandas/numpy, never on another sae_app
module. This keeps them safe to import from anywhere (data loading, the MTB
engine, geo code, the recommendation engine) without any risk of a circular
import.
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd


def clean_text(
    value,
    *,
    default: str = "",
    lower: bool = False,
    strip_accents: bool = False,
    missing_values: set[str] | None = None,
) -> str:
    """Normalize scalar text values consistently across UI, geo, and recommendation code."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    text = " ".join(str(value).strip().split())
    if lower:
        text = text.lower()
    if strip_accents:
        text = "".join(
            character
            for character in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(character)
        )
    missing = {""} | {str(x).lower() for x in (missing_values or set())}
    if text.lower() in missing:
        return default
    return text


def series_value(row: pd.Series, col: str, default=""):
    """Read one column explicitly from a pandas row, preserving a default for optional metadata."""
    return row[col] if col in row.index else default


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
    return str(x).strip().lower() in {"1", "true", "yes", "y", "x", "oui", "si", "sí", "s"}


def as_float(x, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(str(x).strip().replace(",", "."))
    except Exception:
        return default


def parse_coordinate(value) -> float:
    """Parse latitude/longitude values that may use commas as decimal separators."""
    try:
        if pd.isna(value):
            return np.nan
        parsed = float(str(value).strip().replace(",", "."))
    except Exception:
        return np.nan
    return parsed if np.isfinite(parsed) else np.nan


def clean_optional_value(value, *, default: str = "No information") -> str:
    return clean_text(value, default=default, missing_values={"nan"})


def normalize_geo_key(value) -> str:
    """Normalize commune/region names for approximate coordinate lookup."""
    return clean_text(value, default="", lower=True, strip_accents=True, missing_values={"nan"})


def clean_recommendation_value(value) -> str:
    """Return a usable categorical value for recommendation scoring."""
    return clean_text(value, default="", missing_values={"nan", "unknown", "no information"})
