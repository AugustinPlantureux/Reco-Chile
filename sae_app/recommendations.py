"""The "similar programs" portfolio-risk recommendation engine.

Given the current wish list, this module infers a revealed-preference profile,
scores every other program on similarity/proximity/competition/portfolio-risk
improvement, and picks a diverse top-N with Maximal Marginal Relevance.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import streamlit as st

from sae_app.constants import (
    CAPACITY,
    EQUIV_GROUP,
    HARD_UNMATCHED_THRESHOLD,
    NO_PRIORITY,
    POP,
    PROGRAM,
    PROGRAM_ENROLLMENT_FEE,
    PROGRAM_GENDER,
    PROGRAM_MONTHLY_FEE,
    PROGRAM_PACE,
    PROGRAM_PIE,
    PROGRAM_RELIGIOUS_ORIENTATION,
    PROGRAM_RURALITY,
    PROGRAM_SCHOOL_DAY,
    PROGRAM_SPECIALTY_SECTOR,
    PROGRAM_TRACK,
    SOFT_UNMATCHED_THRESHOLD,
    TRUE_APP,
    UNKNOWN_REGION,
    WISH_RANK,
)
from sae_app.geo import (
    commune_coordinate_lookup,
    haversine_km,
    home_distance_filter_is_reliable,
    program_coordinate_reference_priority,
    program_coordinates,
    proximity_from_distance,
    valid_lat_lon,
)
from sae_app.errors import CandidateEvaluationError
from sae_app.i18n import t
from sae_app.mtb_engine import attach_mtb_hashes, availability
from sae_app.program_options import ProgramRecord
from sae_app.text_utils import as_float, clean_recommendation_value, normalize_geo_key
from sae_app.wish_list import make_builder_wish_row

LOGGER = logging.getLogger(__name__)

RECOMMENDATION_CRITERIA = [
    (PROGRAM_TRACK, "Program type", 1.00),
    (PROGRAM_SPECIALTY_SECTOR, "Specialty area", 1.00),
    (PROGRAM_GENDER, "Gender composition", 0.50),
    (PROGRAM_SCHOOL_DAY, "School day", 0.50),
    (PROGRAM_RURALITY, "Rurality", 0.50),
    (PROGRAM_PIE, "PIE", 0.40),
    (PROGRAM_PACE, "PACE", 0.40),
    (PROGRAM_ENROLLMENT_FEE, "Enrollment fee", 1.00),
    (PROGRAM_MONTHLY_FEE, "Monthly fee", 1.25),
    (PROGRAM_RELIGIOUS_ORIENTATION, "Religious orientation", 1.00),
]

# ---------------------------------------------------------------------------
# Hidden recommendation settings
# ---------------------------------------------------------------------------
# These values replace the former family-facing recommendation controls.
# Families only see the recommendations; you can still tune the portfolio engine
# directly here if you want another behavior.
RECOMMENDATION_HARD_CONSTRAINT_COLS = []
# Examples:
# RECOMMENDATION_HARD_CONSTRAINT_COLS = [PROGRAM_RELIGIOUS_ORIENTATION, PROGRAM_GENDER]

RECOMMENDATION_FAVOR_LESS_OVERSUBSCRIBED = True
RECOMMENDATION_COMPETITION_WEIGHT = 0.25
RECOMMENDATION_RISK_OPTIMIZATION_WEIGHT = 2.25
# Minimum revealed-preference similarity required in normal mode.
# If the current wish list contains no usable similarity signal, this threshold
# is not applied and the UI explicitly warns that recommendations are based
# mainly on proximity and admission-risk reduction.
RECOMMENDATION_MIN_SIMILARITY_SCORE = 0.12
RECOMMENDATION_PROXIMITY_WEIGHT = 0.75
RECOMMENDATION_DISTANCE_SCALE_KM = 50.0
# When a home address is provided, exclude schools farther than this
# straight-line distance only when both endpoints are reliable enough:
# address/street-level home geocoding and program/commune-level coordinates.
# Regional and city-level approximations affect scoring but never exclusion.
RECOMMENDATION_MAX_HOME_DISTANCE_KM = 100.0
RECOMMENDATION_DIVERSIFY = True
RECOMMENDATION_DIVERSITY_STRENGTH = 0.35
# Number of valid wishes after which the revealed-preference profile is treated
# as fully reliable. Shorter lists still work, but similarity has less influence
# on candidate filtering and final ranking.
RECOMMENDATION_FULL_RELIABILITY_WISH_COUNT = 4.0

# Recommendation row colors use the same unmatched-risk thresholds as the
# simulation summary, applied to the projected risk after appending a program.
# This makes the color depend on the real marginal portfolio effect rather than
# only on the conditional chance of admission.
CANDIDATE_RISK_CACHE_SESSION_KEY = "candidate_risk_cache_v2"
LEGACY_CANDIDATE_RISK_CACHE_SESSION_KEYS = ("candidate_risk_cache_v1",)


def wish_rank_weight(rank, rank_sensitive: bool = True) -> float:
    """
    Weight higher-ranked wishes slightly more.

    If rank_sensitive=False, every wish has the same weight.
    """
    if not rank_sensitive:
        return 1.0
    try:
        r = max(int(float(rank)), 1)
    except Exception:
        r = 1
    return 1.0 / np.sqrt(r)


def recommendation_rank_value(wish: pd.Series):
    """Use the equivalence group when available, otherwise use the strict rank."""
    group = wish.get(EQUIV_GROUP, np.nan)
    if not pd.isna(group):
        return group
    return wish.get(WISH_RANK, 1)


def build_wish_profile(
    wishes: pd.DataFrame,
    program_mapping: dict[str, pd.Series],
    *,
    rank_sensitive: bool = True,
    commune_lookup: dict[tuple[str, str], tuple[float, float]] | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Build the student's revealed-preference profile from the current wish list.

    The automatic profile tracks both dominance and coverage. A criterion is not
    treated as strongly informative if it is observed for only one wish.
    """
    valid_wishes = wishes.copy()
    valid_wishes[PROGRAM] = valid_wishes[PROGRAM].fillna("").astype(str).str.strip()
    valid_wishes = valid_wishes[valid_wishes[PROGRAM].isin(program_mapping)].copy()

    if valid_wishes.empty:
        return {}, pd.DataFrame()

    profile = {
        "selected_programs": set(valid_wishes[PROGRAM].tolist()),
        "valid_wish_count": int(len(valid_wishes)),
        "total_wish_weight": 0.0,
        "criteria": {},
        "criterion_coverage": {},
        "geo_reference": {},
    }

    geo_rows = []

    for _, wish in valid_wishes.iterrows():
        label = str(wish[PROGRAM]).strip()
        row = program_mapping[label]
        program = ProgramRecord.from_series(row, label=label)
        weight = wish_rank_weight(recommendation_rank_value(wish), rank_sensitive=rank_sensitive)
        profile["total_wish_weight"] += weight

        lat, lon, coordinate_source = program_coordinates(
            program,
            commune_lookup=commune_lookup,
        )
        if valid_lat_lon(lat, lon):
            geo_rows.append({
                "lat": lat,
                "lon": lon,
                "weight": weight,
                "coordinate_source": coordinate_source,
                "reference_priority": program_coordinate_reference_priority(
                    coordinate_source
                ),
            })

        for col, _, _ in RECOMMENDATION_CRITERIA:
            value = program.criterion_value(col)
            if not value:
                continue
            profile["criteria"].setdefault(col, {})
            profile["criteria"][col][value] = profile["criteria"][col].get(value, 0.0) + weight
            profile["criterion_coverage"][col] = profile["criterion_coverage"].get(col, 0.0) + weight

    total_wish_weight = max(float(profile["total_wish_weight"]), 1e-9)

    if geo_rows:
        best_reference_priority = max(g["reference_priority"] for g in geo_rows)
        reference_geo_rows = [
            g for g in geo_rows
            if g["reference_priority"] == best_reference_priority
        ]
        geo_weight = sum(g["weight"] for g in reference_geo_rows)
        profile["geo_reference"] = {
            "lat": sum(g["lat"] * g["weight"] for g in reference_geo_rows) / geo_weight,
            "lon": sum(g["lon"] * g["weight"] for g in reference_geo_rows) / geo_weight,
            "coordinate_reference_priority": best_reference_priority,
        }

    for col, _, _ in RECOMMENDATION_CRITERIA:
        dist = profile["criteria"].get(col, {})
        total = sum(dist.values())
        coverage = float(profile["criterion_coverage"].get(col, 0.0)) / total_wish_weight
        profile["criterion_coverage"][col] = float(np.clip(coverage, 0.0, 1.0))

        if total > 0:
            profile["criteria"][col] = {k: v / total for k, v in dist.items()}

    auto_weights = automatic_recommendation_weights(profile)
    profile_reliability = recommendation_profile_reliability(profile)
    dominant_rows = []

    for col, label, _ in RECOMMENDATION_CRITERIA:
        normalized = profile["criteria"].get(col, {})
        if not normalized:
            continue

        coverage = float(profile["criterion_coverage"].get(col, 0.0))
        dominant_value, dominant_share = max(normalized.items(), key=lambda x: x[1])
        dominant_rows.append({
            "Criterion": label,
            "Dominant value in current list": dominant_value,
            "Share": f"{dominant_share:.0%}",
            "Coverage": f"{coverage:.0%}",
            "Automatic weight": round(
                auto_weights.get(col, 0.0) * profile_reliability,
                2,
            ),
        })

    profile_table = pd.DataFrame(dominant_rows)
    return profile, profile_table


def recommendation_profile_reliability(profile: dict) -> float:
    """Return the confidence assigned to similarity inferred from the wish list."""
    valid_wish_count = max(int(profile.get("valid_wish_count", 0)), 0)
    full_reliability_count = max(
        float(RECOMMENDATION_FULL_RELIABILITY_WISH_COUNT),
        1.0,
    )
    return float(np.clip(valid_wish_count / full_reliability_count, 0.0, 1.0))


def automatic_recommendation_weights(profile: dict) -> dict[str, float]:
    """Infer relative criterion weights from revealed preferences only.

    Automatic signal = base_weight × dominance × coverage.

    These weights determine the relative importance of the criteria. Profile
    reliability is applied after similarity normalization so it cannot cancel
    out as a common multiplicative factor. Coverage prevents a sparsely observed
    criterion from dominating the profile.
    """
    weights: dict[str, float] = {}
    for criterion_col, _, base_weight in RECOMMENDATION_CRITERIA:
        distribution = profile.get("criteria", {}).get(criterion_col, {})
        coverage = float(np.clip(profile.get("criterion_coverage", {}).get(criterion_col, 0.0), 0.0, 1.0))

        if not distribution:
            auto_signal = 0.0
        else:
            dominance = max(float(v) for v in distribution.values())
            auto_signal = dominance * coverage

        weights[criterion_col] = float(base_weight) * auto_signal

    return weights


def candidate_satisfies_hard_constraints(
    program: ProgramRecord,
    profile: dict,
    hard_constraint_cols: list[str] | None,
) -> bool:
    """Apply optional deal-breaker constraints based on values in the wish list."""
    for col in hard_constraint_cols or []:
        accepted_values = set(profile.get("criteria", {}).get(col, {}).keys())
        if not accepted_values:
            continue
        if program.criterion_value(col) not in accepted_values:
            return False
    return True


def current_unmatched_risk_from_simulation_result(simulation_result: dict | None) -> float:
    """Return the current unmatched risk from the last simulation result."""
    if not simulation_result:
        return np.nan

    choices = simulation_result.get("choices")
    if isinstance(choices, pd.DataFrame) and not choices.empty:
        return float(choices["cumulative_unavailable_after_choice"].iloc[-1])

    reference_choices = simulation_result.get("reference_choices")
    if isinstance(reference_choices, pd.DataFrame) and not reference_choices.empty:
        return float(reference_choices["cumulative_unavailable_after_choice"].iloc[-1])

    variants_df = simulation_result.get("variants_df")
    if isinstance(variants_df, pd.DataFrame) and not variants_df.empty and "Unmatched risk" in variants_df.columns:
        # In equivalence-class mode, the same set of programs is tested in every
        # compatible strict order. Therefore the total unmatched risk is invariant
        # to the internal order; use the first valid value rather than a misleading
        # average over identical values.
        risks = pd.to_numeric(variants_df["Unmatched risk"], errors="coerce").dropna()
        if not risks.empty:
            return float(risks.iloc[0])

    return np.nan


def risk_color_from_projected_unmatched_risk(projected_unmatched_risk: float) -> str:
    """Color a recommendation from the risk remaining after it is appended."""
    try:
        risk = float(projected_unmatched_risk)
    except (TypeError, ValueError):
        return "gray"

    if not np.isfinite(risk):
        return "gray"
    if risk < SOFT_UNMATCHED_THRESHOLD:
        return "green"
    if risk < HARD_UNMATCHED_THRESHOLD:
        return "orange"
    return "red"


def format_probability(value: float) -> str:
    try:
        value = float(value)
    except Exception:
        return ""
    return "" if not np.isfinite(value) else f"{value:.1%}"


def candidate_risk_cache_key(
    candidate_label: str,
    candidate_program: pd.Series,
) -> tuple:
    """Cache key for candidate metrics within the current student session.

    The student identifier is deliberately excluded. The whole cache is cleared
    by the UI whenever the RUN/IPE widget changes.
    """
    return (
        str(candidate_label),
        str(candidate_program.get("rbd", "")).strip(),
        str(candidate_program.get("program_code", "")).strip(),
        round(as_float(candidate_program.get(CAPACITY, 0), 0.0), 6),
        round(as_float(candidate_program.get(TRUE_APP, 0), 0.0), 6),
        round(as_float(candidate_program.get(POP, 0), 0.0), 6),
        round(as_float(candidate_program.get(f"priority_share_{NO_PRIORITY}_2024", 0), 0.0), 8),
        round(as_float(candidate_program.get(f"cum_share_before_{NO_PRIORITY}_2024", 0), 0.0), 8),
    )


def clear_candidate_risk_cache() -> None:
    """Remove all student-derived recommendation metrics from session state."""
    st.session_state.pop(CANDIDATE_RISK_CACHE_SESSION_KEY, None)
    for legacy_key in LEGACY_CANDIDATE_RISK_CACHE_SESSION_KEYS:
        st.session_state.pop(legacy_key, None)


def cached_candidate_base_metrics(
    candidate_label: str,
    candidate_program: pd.Series,
    program_mapping: dict[str, pd.Series],
    student_id: str,
) -> dict:
    """Compute/cache the candidate's chance-if-considered and MTB rank.

    current_unmatched_risk is intentionally not part of this cache: it only
    scales the final appended chance and can be applied cheaply after lookup.
    """
    cache = st.session_state.setdefault(CANDIDATE_RISK_CACHE_SESSION_KEY, {})
    key = candidate_risk_cache_key(candidate_label, candidate_program)
    if key in cache:
        return cache[key]

    candidate_wish = pd.DataFrame([make_builder_wish_row(candidate_label, 1, 1)])

    hashed = attach_mtb_hashes(candidate_wish, program_mapping, student_id)
    avail = availability(hashed.iloc[0], candidate_program)
    chance_if_considered = float(avail.get("availability_probability", np.nan))
    lottery_rank = int(avail.get("lottery_number", 1))

    out = {
        "chance_if_considered_raw": chance_if_considered,
        "estimated_lottery_rank": lottery_rank,
    }
    cache[key] = out
    return out


def candidate_portfolio_metrics(
    candidate_label: str,
    candidate_program: pd.Series,
    program_mapping: dict[str, pd.Series],
    *,
    student_id: str,
    current_unmatched_risk: float,
) -> dict:
    """Estimate one appended recommendation's marginal assignment probability.

    Because the candidate is appended at the end, its final assignment
    probability is exactly equal to the marginal reduction in unmatched risk:

        final_chance_if_appended = current_unmatched_risk * chance_if_considered

    We keep only the final assignment probability in the UI to avoid displaying
    two algebraically identical values under different names.
    """
    base_metrics = cached_candidate_base_metrics(
        candidate_label,
        candidate_program,
        program_mapping,
        student_id,
    )
    chance_if_considered = float(base_metrics.get("chance_if_considered_raw", np.nan))
    lottery_rank = base_metrics.get("estimated_lottery_rank", np.nan)

    base_unmatched = float(current_unmatched_risk) if np.isfinite(current_unmatched_risk) else np.nan
    if np.isfinite(base_unmatched) and np.isfinite(chance_if_considered):
        final_chance_if_appended = base_unmatched * chance_if_considered
        projected_unmatched_risk = float(
            np.clip(base_unmatched - final_chance_if_appended, 0.0, 1.0)
        )
    else:
        final_chance_if_appended = np.nan
        projected_unmatched_risk = np.nan

    return {
        "chance_if_considered_raw": chance_if_considered,
        "final_chance_if_appended_raw": final_chance_if_appended,
        "projected_unmatched_risk_raw": projected_unmatched_risk,
        "risk_color": risk_color_from_projected_unmatched_risk(projected_unmatched_risk),
        "estimated_lottery_rank": lottery_rank,
    }


def candidate_feature_values(program: ProgramRecord) -> dict[str, str]:
    values = {col: program.criterion_value(col) for col, _, _ in RECOMMENDATION_CRITERIA}
    values["school"] = normalize_geo_key(program.school_name)
    values["commune"] = normalize_geo_key(program.school_commune)
    values["region"] = normalize_geo_key(program.region)
    return values


def candidate_similarity(features_a: dict, features_b: dict) -> float:
    """Similarity used by MMR to avoid near-duplicate recommendations."""
    total = 0.0
    matched = 0.0

    for key, weight in [("school", 2.0), ("commune", 1.25), ("region", 0.25)]:
        va = features_a.get(key, "")
        vb = features_b.get(key, "")
        if va and vb:
            total += weight
            if va == vb:
                matched += weight

    for col, _, base_weight in RECOMMENDATION_CRITERIA:
        va = features_a.get(col, "")
        vb = features_b.get(col, "")
        if va and vb:
            total += float(base_weight)
            if va == vb:
                matched += float(base_weight)

    return float(matched / total) if total > 0 else 0.0


def select_diverse_recommendations(
    candidates: pd.DataFrame,
    max_recommendations: int,
    diversity_strength: float = 0.35,
) -> pd.DataFrame:
    """Select top recommendations with Maximal Marginal Relevance."""
    if candidates.empty or max_recommendations <= 0:
        return candidates.head(0)

    diversity_strength = float(np.clip(diversity_strength, 0.0, 0.9))
    if diversity_strength <= 0:
        return candidates.sort_values(
            ["_recommendation_score_raw", "_chance_if_considered_raw", "_similarity_score_raw", "School"],
            ascending=[False, False, False, True],
        ).head(max_recommendations).copy()

    pool = candidates.sort_values("_recommendation_score_raw", ascending=False).head(max(50, max_recommendations * 8)).copy()
    selected_indices: list[int] = []
    remaining_indices = list(pool.index)

    while remaining_indices and len(selected_indices) < max_recommendations:
        best_idx = None
        best_score = -np.inf

        for idx in remaining_indices:
            row = pool.loc[idx]
            relevance = float(row.get("_recommendation_score_raw", 0.0))
            redundancy = 0.0
            if selected_indices:
                redundancy = max(
                    candidate_similarity(row["_features"], pool.loc[selected_idx]["_features"])
                    for selected_idx in selected_indices
                )
            mmr_score = (1.0 - diversity_strength) * relevance - diversity_strength * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return pool.loc[selected_indices].copy()


def recommend_similar_programs(
    wishes: pd.DataFrame,
    program_mapping: dict[str, pd.Series],
    *,
    student_id: str,
    current_unmatched_risk: float,
    max_recommendations: int = 15,
    rank_sensitive: bool = True,
    competition_weight: float = RECOMMENDATION_COMPETITION_WEIGHT,
    hard_constraint_cols: list[str] | None = None,
    proximity_weight: float = RECOMMENDATION_PROXIMITY_WEIGHT,
    distance_scale_km: float = RECOMMENDATION_DISTANCE_SCALE_KM,
    home_geo_reference: dict | None = None,
    max_home_distance_km: float | None = RECOMMENDATION_MAX_HOME_DISTANCE_KM,
    risk_optimization_weight: float = RECOMMENDATION_RISK_OPTIMIZATION_WEIGHT,
    min_similarity_score: float = RECOMMENDATION_MIN_SIMILARITY_SCORE,
    diversify: bool = RECOMMENDATION_DIVERSIFY,
    diversity_strength: float = RECOMMENDATION_DIVERSITY_STRENGTH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recommend programs as a portfolio-risk optimization problem.

    Each candidate is evaluated as if it were appended after the current wish
    list. The model uses the student's real MTB hash for the candidate school,
    estimates the chance of obtaining that school if reached, then converts it
    into a marginal improvement in the matched outcome.
    """
    commune_lookup = commune_coordinate_lookup()
    profile, profile_table = build_wish_profile(
        wishes,
        program_mapping,
        rank_sensitive=rank_sensitive,
        commune_lookup=commune_lookup,
    )

    if not profile:
        return pd.DataFrame(), profile_table

    selected_programs = profile["selected_programs"]
    criterion_weights = automatic_recommendation_weights(profile)
    profile_reliability = recommendation_profile_reliability(profile)
    active_weight_total = sum(max(float(v), 0.0) for v in criterion_weights.values())
    similarity_signal_available = active_weight_total > 0
    similarity_weight = profile_reliability if similarity_signal_available else 0.0

    # A short wish list should not impose an overconfident similarity filter.
    # Reliability therefore scales both the final contribution of similarity
    # and the minimum similarity required to retain a candidate.
    base_min_similarity_score = max(float(min_similarity_score), 0.0)
    effective_min_similarity_score = (
        base_min_similarity_score * profile_reliability
        if similarity_signal_available
        else 0.0
    )

    proximity_weight = max(float(proximity_weight), 0.0)
    competition_weight = max(float(competition_weight), 0.0)
    risk_optimization_weight = max(float(risk_optimization_weight), 0.0)

    if similarity_weight <= 0 and proximity_weight <= 0 and competition_weight <= 0 and risk_optimization_weight <= 0:
        return pd.DataFrame(), profile_table

    home_geo_reference = home_geo_reference or {}
    use_home_reference = valid_lat_lon(
        home_geo_reference.get("lat", np.nan),
        home_geo_reference.get("lon", np.nan),
    )
    if use_home_reference:
        ref_lat = float(home_geo_reference.get("lat"))
        ref_lon = float(home_geo_reference.get("lon"))
    else:
        geo_reference = profile.get("geo_reference", {})
        ref_lat = geo_reference.get("lat", np.nan)
        ref_lon = geo_reference.get("lon", np.nan)

    rows = []
    failed_candidates = 0
    failed_candidate_examples: list[tuple[str, str]] = []

    for candidate_label, row in program_mapping.items():
        if candidate_label in selected_programs:
            continue

        try:
            program = ProgramRecord.from_series(row, label=candidate_label)
            if not candidate_satisfies_hard_constraints(program, profile, hard_constraint_cols):
                continue

            capacity = max(program.capacity, 0.0)
            if capacity <= 0:
                continue

            raw_similarity = 0.0
            for col, _, _ in RECOMMENDATION_CRITERIA:
                user_weight = max(float(criterion_weights.get(col, 0.0)), 0.0)
                if user_weight <= 0:
                    continue

                wish_distribution = profile["criteria"].get(col, {})
                if not wish_distribution:
                    continue

                candidate_value = program.criterion_value(col)
                if not candidate_value:
                    continue

                match_share = wish_distribution.get(candidate_value, 0.0)
                if match_share <= 0:
                    continue

                raw_similarity += user_weight * match_share

            normalized_similarity = (
                raw_similarity / active_weight_total
                if active_weight_total > 0
                else 0.0
            )
            if (
                similarity_signal_available
                and normalized_similarity < effective_min_similarity_score
            ):
                continue
            similarity_score = similarity_weight * normalized_similarity

            lat, lon, coordinate_source = program_coordinates(
                program,
                commune_lookup=commune_lookup,
            )
            distance_km = haversine_km(ref_lat, ref_lon, lat, lon)
            apply_hard_distance_filter = (
                use_home_reference
                and max_home_distance_km is not None
                and home_distance_filter_is_reliable(
                    home_geo_reference,
                    coordinate_source,
                    program,
                )
            )
            if apply_hard_distance_filter:
                if not np.isfinite(distance_km) or distance_km > float(max_home_distance_km):
                    continue

            # Approximate coordinates still influence ranking through this soft
            # score, but they are never used to exclude a candidate outright.
            proximity_score = proximity_from_distance(
                distance_km,
                distance_scale_km=distance_scale_km,
            )

            true_applicants = max(program.true_applicants, 0.0)
            if true_applicants <= 0:
                # If a program had available seats and no true applicants last year,
                # it should be treated as highly accessible in the recommendation
                # score, consistently with availability().
                competition_ratio = 0.0
                accessibility_score = 1.0
            else:
                competition_ratio = true_applicants / capacity
                accessibility_score = min(capacity / true_applicants, 1.0)

            portfolio = candidate_portfolio_metrics(
                candidate_label,
                row,
                program_mapping,
                student_id=student_id,
                current_unmatched_risk=current_unmatched_risk,
            )
            chance_if_considered = float(portfolio["chance_if_considered_raw"])
            risk_score = chance_if_considered if np.isfinite(chance_if_considered) else 0.0

            denominator = similarity_weight + proximity_weight + competition_weight + risk_optimization_weight
            final_score = (
                similarity_score
                + proximity_weight * proximity_score
                + competition_weight * accessibility_score
                + risk_optimization_weight * risk_score
            ) / denominator

            rows.append({
                PROGRAM: candidate_label,
                "School": clean_recommendation_value(program.school_name) or t("School name unavailable"),
                "Commune": clean_recommendation_value(program.school_commune),
                "Region": clean_recommendation_value(program.region) or UNKNOWN_REGION,
                "Program details": clean_recommendation_value(program.program_display_name),
                "Chance if considered": format_probability(portfolio["chance_if_considered_raw"]),
                "Marginal unmatched-risk reduction": format_probability(portfolio["final_chance_if_appended_raw"]),
                "Projected unmatched risk after append": format_probability(portfolio["projected_unmatched_risk_raw"]),
                "Estimated MTB rank": portfolio["estimated_lottery_rank"] if np.isfinite(portfolio["estimated_lottery_rank"]) else "",
                "Recommendation score": round(100 * final_score, 1),
                "Straight-line distance from home (km)": round(distance_km, 1) if use_home_reference and not pd.isna(distance_km) else "",
                "Straight-line distance from current list (km)": round(distance_km, 1) if (not use_home_reference) and not pd.isna(distance_km) else "",
                "Capacity": int(capacity) if capacity == int(capacity) else capacity,
                "Applicants / seat": round(competition_ratio, 2) if not pd.isna(competition_ratio) else "",
                "_recommendation_score_raw": float(final_score),
                "_similarity_score_raw": float(similarity_score),
                "_chance_if_considered_raw": float(risk_score),
                "_proximity_score_raw": float(proximity_score),
                "_projected_unmatched_risk_raw": portfolio["projected_unmatched_risk_raw"],
                "_risk_color": portfolio["risk_color"],
                "_similarity_fallback_mode": not similarity_signal_available,
                "_features": candidate_feature_values(program),
            })
        except CandidateEvaluationError as exc:
            failed_candidates += 1
            if len(failed_candidate_examples) < 5:
                failed_candidate_examples.append((candidate_label, str(exc)))
            LOGGER.warning(
                "Skipping malformed recommendation candidate %s: %s",
                candidate_label,
                exc,
            )
            continue

    diagnostics = {
        "failed_candidates": failed_candidates,
        "failed_candidate_examples": tuple(failed_candidate_examples),
    }

    if not rows:
        empty = pd.DataFrame()
        empty.attrs["recommendation_diagnostics"] = diagnostics
        return empty, profile_table

    candidates = pd.DataFrame(rows)
    if diversify:
        out = select_diverse_recommendations(
            candidates,
            max_recommendations=max_recommendations,
            diversity_strength=diversity_strength,
        )
    else:
        out = candidates.sort_values(
            ["_recommendation_score_raw", "_chance_if_considered_raw", "_similarity_score_raw", "_proximity_score_raw", "School"],
            ascending=[False, False, False, False, True],
        ).head(max_recommendations).copy()

    out = out.drop(columns=["_features"], errors="ignore").reset_index(drop=True)
    out.attrs["recommendation_diagnostics"] = diagnostics
    return out, profile_table
