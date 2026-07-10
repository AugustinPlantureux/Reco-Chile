"""Wish-list parsing, cleaning, and equivalence-class handling.

This module owns the shape of the wish-list DataFrame: default rows, CSV
import, padding/cleaning, and turning equivalence classes into every
compatible strict order.
"""

from __future__ import annotations

import math
from itertools import permutations, product

import numpy as np
import pandas as pd

from sae_app.constants import EQUIV_GROUP, LOTTERY, PRIORITIES, PROGRAM, SAFETY, WISH_RANK
from sae_app.data_loading import read_csv
from sae_app.i18n import display_outcome_label, t
from sae_app.text_utils import as_bool, norm_code

# ---------------------------------------------------------------------------
# Wish-list handling (default rows + CSV import)
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

    has_program = out[PROGRAM] != ""

    # Rows without a selected program cannot be simulated and are dropped.
    # Duplicate programs can happen through CSV import; keep the first one so
    # Streamlit widget keys remain unique and the wish list stays valid.
    out = out[has_program].copy().reset_index(drop=True)
    out = out.drop_duplicates(subset=[PROGRAM], keep="first").reset_index(drop=True)

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

    # Automatic column-format detection.
    # In MTB mode, lottery_number / numero_loterie columns are intentionally
    # ignored: the effective lottery rank is computed deterministically from
    # RUN/IPE + RBD in attach_mtb_hashes() before every calculation.
    if {WISH_RANK, PROGRAM}.issubset(df.columns):
        out = pd.DataFrame({WISH_RANK: df[WISH_RANK], PROGRAM: df[PROGRAM]})
    elif {"rang_du_voeu", "programme"}.issubset(df.columns):
        out = pd.DataFrame({WISH_RANK: df["rang_du_voeu"], PROGRAM: df["programme"]})
    elif {"rbd", "program_code", "preference_number"}.issubset(df.columns):
        labels = norm_code(df["rbd"]) + " || " + norm_code(df["program_code"])
        out = pd.DataFrame({WISH_RANK: df["preference_number"], PROGRAM: labels})
    else:
        raise ValueError(
            t("Expected columns: wish_rank/program, rang_du_voeu/programme, or rbd/program_code/preference_number.")
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
    out[LOTTERY] = 1
    out[PROGRAM] = (
        out[PROGRAM].astype(str).str.strip()
        .map(lambda x: x if x in mapping else base_to_label.get(x, ""))
    )
    for col in PRIORITIES + [SAFETY]:
        out[col] = df[col].map(as_bool) if col in df.columns else False

    return out.sort_values(WISH_RANK).reset_index(drop=True) if not out.empty else empty_wishes()


def uploaded_lottery_columns(file_bytes: bytes) -> list[str]:
    """Return user-supplied lottery columns that will be ignored in MTB mode."""
    try:
        df = read_csv(file_bytes, sep="auto")
    except Exception:
        return []
    candidates = [LOTTERY, "numero_loterie", "lottery"]
    return [col for col in candidates if col in df.columns]


# ---------------------------------------------------------------------------
# Wish-list builder helpers (used by the UI to add/normalize rows)
# ---------------------------------------------------------------------------

def non_empty_wish_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows with a selected program, with priority columns normalized."""
    out = clean_wish_rows(df).copy()
    out[PROGRAM] = out[PROGRAM].fillna("").astype(str).str.strip()
    out = out[out[PROGRAM] != ""].copy().reset_index(drop=True)

    for col in PRIORITIES + [SAFETY]:
        if col not in out.columns:
            out[col] = False
        out[col] = out[col].map(as_bool).fillna(False).astype(bool)

    return out


def make_builder_wish_row(program_label: str, wish_rank: int, preference_group: int) -> dict:
    """Create one wish row in the same format expected by the simulation engine."""
    row = {
        WISH_RANK: int(wish_rank),
        EQUIV_GROUP: int(preference_group),
        PROGRAM: str(program_label).strip(),
        LOTTERY: 1,
    }
    for col in PRIORITIES + [SAFETY]:
        row[col] = False
    return row


def normalize_builder_wishes(
    df: pd.DataFrame,
    use_equivalence_classes: bool,
) -> pd.DataFrame:
    """
    Normalize builder rows into the same DataFrame structure expected by the
    simulation engine.
    """
    out = non_empty_wish_rows(df)

    if out.empty:
        return clean_wish_rows(out)

    out[WISH_RANK] = pd.to_numeric(out[WISH_RANK], errors="coerce")
    out[EQUIV_GROUP] = pd.to_numeric(out[EQUIV_GROUP], errors="coerce")

    fallback_rank = pd.Series(range(1, len(out) + 1), index=out.index)
    out[WISH_RANK] = out[WISH_RANK].where(out[WISH_RANK].notna(), fallback_rank)
    out[EQUIV_GROUP] = out[EQUIV_GROUP].where(out[EQUIV_GROUP].notna(), out[WISH_RANK])

    out[WISH_RANK] = out[WISH_RANK].astype(int).clip(lower=1)
    out[EQUIV_GROUP] = out[EQUIV_GROUP].astype(int).clip(lower=1)

    if use_equivalence_classes:
        # Preference groups determine the true preference order. Compact group
        # labels so 1, 1, 5 behaves like 1, 1, 2 for downstream weighting.
        # wish_rank is only the reference strict order used for preview/testing.
        out = out.sort_values([EQUIV_GROUP, WISH_RANK], kind="stable").reset_index(drop=True)
        group_map = {old_group: i + 1 for i, old_group in enumerate(pd.unique(out[EQUIV_GROUP]))}
        out[EQUIV_GROUP] = out[EQUIV_GROUP].map(group_map).astype(int)
        out[WISH_RANK] = np.arange(1, len(out) + 1)
    else:
        # In strict mode, the card order is the ranking.
        out = out.reset_index(drop=True)
        out[WISH_RANK] = np.arange(1, len(out) + 1)
        out[EQUIV_GROUP] = out[WISH_RANK]

    if LOTTERY not in out.columns:
        out[LOTTERY] = 1
    out[LOTTERY] = pd.to_numeric(out[LOTTERY], errors="coerce").fillna(1).astype(int)

    return clean_wish_rows(out)


# ---------------------------------------------------------------------------
# Equivalence-class handling
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
        group_map = {old_group: i + 1 for i, old_group in enumerate(pd.unique(clean[EQUIV_GROUP]))}
        clean[EQUIV_GROUP] = clean[EQUIV_GROUP].map(group_map).astype(int)
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


def predicted_outcome_from_choices(choices: pd.DataFrame, hard_threshold: float) -> tuple[str, float, bool]:
    """Apply the hard unmatched-risk threshold to summarize the top-1 prediction."""
    p_unmatched = float(choices["cumulative_unavailable_after_choice"].iloc[-1])
    at_risk = p_unmatched >= hard_threshold
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



def predicted_outcome_final_chance(choices: pd.DataFrame, outcome: str) -> float:
    """Return the final probability attached to the predicted outcome.

    In equivalence-class sensitivity, the total unmatched risk is invariant to
    the order of tied programs. What varies is the final assignment probability
    carried by the school that becomes the predicted outcome under each strict
    order.
    """
    if choices.empty:
        return np.nan

    if str(outcome) == "Unmatched":
        return float(choices["cumulative_unavailable_after_choice"].iloc[-1])

    match = choices[choices["program"].astype(str) == str(outcome)]
    if match.empty:
        return np.nan
    return float(match["choice_assignment_probability"].iloc[0])

def compact_order_label(order_df: pd.DataFrame, max_items: int = 5) -> str:
    programs = [display_outcome_label(p) for p in order_df[PROGRAM].astype(str).str.strip().tolist()]
    if len(programs) <= max_items:
        return " → ".join(programs)
    return " → ".join(programs[:max_items]) + f" → … (+{len(programs) - max_items})"
