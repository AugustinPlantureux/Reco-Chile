"""The SHA-256 lottery hash + hypergeometric availability model.

This is the pure calculation core of the app: given a wish list and a
student RUN/IPE, it computes the deterministic lottery rank for each school
and the resulting availability/assignment probabilities. Nothing here touches
Streamlit widgets, so this module can be unit-tested on its own.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from sae_app.constants import (
    CAPACITY,
    HASH_HEX,
    HASH_INPUT,
    HASH_PCT,
    IMPUT_METHOD,
    IMPUTED,
    LOTTERY,
    MAX_SHA256,
    NO_PRIORITY,
    POP,
    PRIORITIES,
    PRIORITY_STUDENT_QUOTA,
    PROGRAM,
    SAFETY,
    TRUE_APP,
    WISH_RANK,
)
from sae_app.i18n import t
from sae_app.text_utils import as_bool, as_float, norm_code_value

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
        raise ValueError(t("Enter the student RUN/IPE before running the MTB calculation."))
    if not re.fullmatch(r"[0-9K\-]+", cleaned):
        raise ValueError(
            t("The RUN/IPE may contain only digits, one optional hyphen, and the letter K.")
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

def compute_from_availability_rows(rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """Aggregate already-computed wish availabilities into assignment chances."""
    choices = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
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


def compute(
    wishes: pd.DataFrame,
    mapping: dict[str, pd.Series],
) -> pd.DataFrame:
    clean = wishes[wishes[PROGRAM].astype(str).str.strip() != ""].sort_values(WISH_RANK)
    if clean.empty:
        raise ValueError(t("Add at least one valid wish."))

    rows = [
        availability(wish, mapping[wish[PROGRAM]])
        for _, wish in clean.iterrows()
        if wish[PROGRAM] in mapping
    ]

    return compute_from_availability_rows(rows)


def wish_availability_cache_key(wish: pd.Series) -> tuple:
    """Key for availability values that do not depend on strict-list position."""
    return (
        str(wish.get(PROGRAM, "")).strip(),
        tuple(as_bool(wish.get(col, False)) for col in PRIORITIES + [SAFETY]),
    )


def precompute_equivalence_availability(
    reference_order: pd.DataFrame,
    mapping: dict[str, pd.Series],
    student_id: str,
) -> dict[tuple, dict]:
    """Compute each wish availability once for all equivalence permutations."""
    hashed = attach_mtb_hashes(reference_order, mapping, student_id)
    lookup: dict[tuple, dict] = {}
    for _, wish in hashed.iterrows():
        label = str(wish.get(PROGRAM, "")).strip()
        if label and label in mapping:
            lookup[wish_availability_cache_key(wish)] = availability(wish, mapping[label])
    return lookup


def compute_equivalence_order_from_precomputed(
    strict_order: pd.DataFrame,
    availability_lookup: dict[tuple, dict],
) -> pd.DataFrame:
    """Recompute only cumulative assignment probabilities for one permutation."""
    clean = strict_order[strict_order[PROGRAM].astype(str).str.strip() != ""].sort_values(WISH_RANK)
    rows = []
    for _, wish in clean.iterrows():
        cached = availability_lookup[wish_availability_cache_key(wish)].copy()
        cached["wish_rank"] = int(wish[WISH_RANK])
        rows.append(cached)
    return compute_from_availability_rows(rows)
