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
    HASH_PCT,
    IMPUT_METHOD,
    IMPUTED,
    LOTTERY,
    MAX_SHA256,
    NO_PRIORITY,
    POP,
    PRIORITIES,
    PRIORITY_STUDENT_SEATS,
    PROGRAM,
    SAFETY,
    TRUE_APP,
    WISH_RANK,
)
from sae_app.errors import EmptyWishList, InvalidStudentIdentifier, UnknownProgram
from sae_app.text_utils import as_bool, as_float, norm_code_value

# ---------------------------------------------------------------------------
# Hash MTB (SHA-256 RUN/IPE + RBD)
# ---------------------------------------------------------------------------

def _clean_identifier_input(student_id: str) -> str:
    """Return a whitespace-free, uppercase identifier or raise if it is empty."""
    raw = re.sub(r"\s+", "", str(student_id).strip().upper())
    if not raw:
        raise InvalidStudentIdentifier("Enter the student RUN/IPE before running the MTB calculation.")
    return raw


def _run_check_digit(body: str) -> str:
    """Return the modulo-11 check digit for a RUN numeric body."""
    total = 0
    multiplier = 2
    for digit in reversed(body):
        total += int(digit) * multiplier
        multiplier = 2 if multiplier == 7 else multiplier + 1

    result = 11 - (total % 11)
    if result == 11:
        return "0"
    if result == 10:
        return "K"
    return str(result)


def normalize_run(student_id: str) -> str:
    """Validate and canonicalize a Chilean RUN before hashing.

    Dots and the hyphen are accepted as optional input formatting. The returned
    value always uses the canonical ``numeric_body-check_digit`` representation.
    """
    raw = _clean_identifier_input(student_id)
    match = re.fullmatch(
        r"(?:(?P<plain>\d{1,8})|(?P<dotted>\d{1,2}(?:\.\d{3}){1,2}))"
        r"-?(?P<check>[0-9K])",
        raw,
    )
    if match is None:
        raise InvalidStudentIdentifier(
            "Invalid RUN format. Enter the numeric body plus its check digit, "
            "for example 12.345.678-5. Dots and the hyphen are optional."
        )

    body = (match.group("plain") or match.group("dotted")).replace(".", "")
    if int(body) == 0:
        raise InvalidStudentIdentifier(
            "Invalid RUN format. Enter the numeric body plus its check digit, "
            "for example 12.345.678-5. Dots and the hyphen are optional."
        )

    check_digit = match.group("check")
    expected = _run_check_digit(body)
    if check_digit != expected:
        raise InvalidStudentIdentifier("The RUN check digit is invalid.")

    return f"{int(body)}-{check_digit}"


def normalize_ipe(student_id: str) -> str:
    """Validate and canonicalize an IPE before hashing.

    An IPE uses a nine-digit numeric body from the 100-million series followed
    by a numeric verifier digit. Dots and the hyphen are accepted as optional
    input formatting; the returned value always uses ``body-verifier``.
    """
    raw = _clean_identifier_input(student_id)
    match = re.fullmatch(
        r"(?:(?P<plain>\d{9})|(?P<dotted>\d{3}(?:\.\d{3}){2}))"
        r"-?(?P<check>\d)",
        raw,
    )
    if match is None:
        raise InvalidStudentIdentifier(
            "Invalid IPE format. Enter the nine-digit IPE plus its numeric "
            "verifier digit, for example 111222333-4. Dots and the hyphen are optional."
        )

    body = (match.group("plain") or match.group("dotted")).replace(".", "")
    return f"{body}-{match.group('check')}"


def normalize_student_identifier(student_id: str) -> str:
    """Dispatch to the strict RUN or IPE parser and return a canonical value."""
    raw = _clean_identifier_input(student_id)
    compact = raw.replace(".", "").replace("-", "")
    if len(compact) == 10:
        return normalize_ipe(raw)
    return normalize_run(raw)


def mtb_hash(student_id: str, rbd) -> float:
    """Return the deterministic 0-best/1-worst MTB lottery percentile.

    The SHA-256 input and digest exist only as local temporaries and are never
    returned or attached to a DataFrame.
    """
    norm_id = normalize_student_identifier(student_id)
    norm_rbd = norm_code_value(rbd)
    digest = hashlib.sha256(f"{norm_id}{norm_rbd}".encode("utf-8")).digest()
    priority_pct = int.from_bytes(digest, byteorder="big") / MAX_SHA256
    return float(np.clip(1.0 - priority_pct, 0, 1))


def pct_to_rank(percentile: float, n: int) -> int:
    """Convert a 0-best/1-worst percentile into one of the ranks 1..n.

    The unit interval is divided into ``n`` equal-width bins:
    [0, 1/n) -> rank 1, ..., [(n - 1)/n, 1] -> rank n.

    Multiplying by ``n - 1`` would make the final rank reachable only at an
    exact percentile of 1, systematically biasing ranks in the student's favor.
    """
    n = max(int(n), 1)
    percentile = float(np.clip(percentile, 0, 1))
    return min(int(np.floor(percentile * n)) + 1, n)


def attach_mtb_hashes(
    wishes: pd.DataFrame,
    mapping: dict[str, pd.Series],
    student_id: str,
) -> pd.DataFrame:
    """Compute and attach the MTB percentile to each valid wish."""
    # Drop legacy sensitive columns if an old in-memory DataFrame is reused.
    out = wishes.drop(
        columns=["lottery_hash_input", "lottery_hash_hex"],
        errors="ignore",
    ).copy()
    if HASH_PCT not in out.columns:
        out[HASH_PCT] = np.nan

    for idx, wish in out.iterrows():
        label = str(wish.get(PROGRAM, "")).strip()
        if not label or label not in mapping:
            continue
        program = mapping[label]
        population = max(round(as_float(program[POP])), 1)
        lottery_percentile = mtb_hash(student_id, program["rbd"])

        out.at[idx, HASH_PCT] = lottery_percentile
        # Theory-consistent equivalent lottery rank within the program-level
        # reference population N_s = program_lottery_population_2024.
        out.at[idx, LOTTERY] = pct_to_rank(lottery_percentile, population)

    return out


# ---------------------------------------------------------------------------
# Priority logic
# ---------------------------------------------------------------------------

def resolve_priority_tier(wish: pd.Series, program: pd.Series) -> str:
    """Determine the priority tier for a wish.

    A student flagged as priority_student keeps that tier only when their
    lottery rank falls within the program's actual priority_student_seats
    allocation. Otherwise, the student falls back to the next active tier.
    """
    if as_bool(wish.get("priority_sibling")):
        return "priority_sibling"

    if as_bool(wish.get("priority_student")):
        quota_count = max(round(as_float(program[PRIORITY_STUDENT_SEATS])), 0)
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
        "raw_lottery_rank":                 raw_rank,
        "lottery_percentile_used":          percentile,
        "priority_effective_percentile":    eff_pct,
        "priority_effective_rank":          eff_rank,
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
    if choices.empty or "availability_probability" not in choices.columns:
        raise UnknownProgram(
            "No valid wish could be matched to the program data. Check the imported wish list."
        )
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
    clean = wishes[wishes[PROGRAM].astype(str).str.strip() != ""].sort_values(WISH_RANK, kind="stable")
    if clean.empty:
        raise EmptyWishList("Add at least one valid wish.")

    rows = []
    for _, wish in clean.iterrows():
        label = str(wish.get(PROGRAM, "")).strip()
        if label in mapping:
            rows.append(availability(wish, mapping[label]))

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

    if not lookup:
        raise UnknownProgram(
            "No valid wish could be matched to the program data. Check the imported wish list."
        )
    return lookup


def compute_equivalence_order_from_precomputed(
    strict_order: pd.DataFrame,
    availability_lookup: dict[tuple, dict],
) -> pd.DataFrame:
    """Recompute only cumulative assignment probabilities for one permutation."""
    clean = strict_order[strict_order[PROGRAM].astype(str).str.strip() != ""].sort_values(WISH_RANK, kind="stable")
    if clean.empty:
        raise EmptyWishList("Add at least one valid wish.")

    rows = []
    for _, wish in clean.iterrows():
        key = wish_availability_cache_key(wish)
        if key not in availability_lookup:
            raise UnknownProgram(
                "A wish in the equivalence-class test could not be matched to the precomputed availability values. Check the imported wish list."
            )
        cached = availability_lookup[key].copy()
        cached["wish_rank"] = int(wish[WISH_RANK])
        rows.append(cached)
    return compute_from_availability_rows(rows)
