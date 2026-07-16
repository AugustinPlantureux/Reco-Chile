"""Shared display/formatting helpers for Streamlit tables.

format_display_table() is used by every part of the UI that shows a pandas
DataFrame to the family: it translates column headers and selected categorical
values without touching free-text columns such as school names or communes.
"""

from __future__ import annotations

import pandas as pd

from sae_app.i18n import display_outcome_label, t


def format_display_table(df: pd.DataFrame) -> pd.DataFrame:
    """Translate display-only DataFrame headers and selected categorical values.

    Free-text columns such as school names, communes, program labels, and
    program details are intentionally left untouched. Only known categorical
    fields are translated, then column headers are translated.
    """
    out = df.copy()

    distance_cols = [
        "Straight-line distance from home (km)",
        "Straight-line distance from current list (km)",
    ]
    one_decimal_cols = [
        "Recommendation score",
    ]
    two_decimal_cols = [
        "Applicants / seat",
    ]
    integer_cols = [
        "Capacity",
        "Estimated MTB rank",
    ]

    for col in distance_cols:
        if col in out.columns:
            out[col] = out[col].map(
                lambda x: "" if pd.isna(x) or str(x).strip() == "" else f"{float(x):.1f} km"
            )

    for col in one_decimal_cols:
        if col in out.columns:
            out[col] = out[col].map(
                lambda x: "" if pd.isna(x) or str(x).strip() == "" else f"{float(x):.1f}"
            )

    for col in two_decimal_cols:
        if col in out.columns:
            out[col] = out[col].map(
                lambda x: "" if pd.isna(x) or str(x).strip() == "" else f"{float(x):.2f}"
            )

    for col in integer_cols:
        if col in out.columns:
            out[col] = out[col].map(
                lambda x: "" if pd.isna(x) or str(x).strip() == "" else f"{int(round(float(x)))}"
            )

    categorical_translation_cols = {
        "Criterion",
        "Dominant value in current list",
    }

    for col in out.columns:
        if col in {"Program", "Predicted outcome"}:
            out[col] = out[col].map(display_outcome_label)
        elif col == "Flagged at risk":
            out[col] = out[col].map(lambda x: t("Yes") if bool(x) else t("No"))
        elif col in categorical_translation_cols:
            out[col] = out[col].map(lambda x: t(x) if isinstance(x, str) else x)

    return out.rename(columns={col: t(col) for col in out.columns})
