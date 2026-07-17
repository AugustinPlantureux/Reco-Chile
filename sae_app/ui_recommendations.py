"""Rendering the "recommended similar programs" section.

render_similar_program_recommendations() draws the whole section 4: the home
address / geocoding controls, the recommendation table, and the "add selected
recommendations to the wish list" button.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd
import streamlit as st

from sae_app.constants import (
    APP_DEBUG,
    EQUIV_GROUP,
    PROGRAM,
    WISH_RANK,
)
from sae_app.errors import CandidateEvaluationError, MtbEngineError
from sae_app.geo import (
    geocode_chilean_address,
    geocoding_precision_warning_key,
    home_geocoding_supports_hard_filter,
    valid_lat_lon,
)
from sae_app.i18n import t
from sae_app.recommendations import (
    RECOMMENDATION_COMPETITION_WEIGHT,
    RECOMMENDATION_DISTANCE_SCALE_KM,
    RECOMMENDATION_DIVERSIFY,
    RECOMMENDATION_DIVERSITY_STRENGTH,
    RECOMMENDATION_FAVOR_LESS_OVERSUBSCRIBED,
    RECOMMENDATION_HARD_CONSTRAINT_COLS,
    RECOMMENDATION_PROXIMITY_WEIGHT,
    RECOMMENDATION_RISK_OPTIMIZATION_WEIGHT,
    RECOMMENDATION_MIN_SIMILARITY_SCORE,
    RECOMMENDATION_MAX_HOME_DISTANCE_KM,
    current_unmatched_risk_from_simulation_result,
    recommend_similar_programs,
)
from sae_app.session_state import clear_wish_editor_widget_state, invalidate_simulation_state
from sae_app.wish_list import clean_wish_rows, make_appended_recommendation_rows


LOGGER = logging.getLogger(__name__)

def render_similar_program_recommendations(
    edited: pd.DataFrame,
    program_mapping: dict[str, pd.Series],
    *,
    student_id: str,
    editor_state_key: str,
    editor_widget_key_base: str,
    use_equivalence_classes: bool = False,
    simulation_done_key: str | None = None,
    simulation_result_key: str | None = None,
) -> None:
    """Render the recommendation UI and optionally append selected programs to the wish list."""
    st.subheader(t("4. Improve the preference list"))

    with st.expander(t("Find acceptable additional programs"), expanded=True):
        current_selected_programs = [
            p for p in edited[PROGRAM].dropna().astype(str).str.strip()
            if p and p in program_mapping
        ]

        if not current_selected_programs:
            st.info(t("Enter at least one valid program in the wish list to get recommendations."))
            return

        simulation_result = st.session_state.get(simulation_result_key, {}) if simulation_result_key else {}
        current_unmatched_risk = current_unmatched_risk_from_simulation_result(simulation_result)

        if np.isfinite(current_unmatched_risk):
            st.metric(t("Current unmatched risk"), f"{current_unmatched_risk:.1%}")
        st.info(
            t("Adding an acceptable program at the end does not reduce the chance of receiving a higher-ranked preference. After adding programs, verify priorities and analyze the list again.")
        )

        with st.expander(t("How are these programs selected?"), expanded=False):
            st.write(
                t(
                    "Suggestions combine similarity to the current list, approximate proximity, historical demand, estimated admission access and diversity between recommendations."
                )
            )
            st.write(
                t(
                    "Each suggestion is evaluated as if it were added at the end of the list and without any special priority. Adding it higher or declaring a real priority can change the result."
                )
            )
            st.caption(
                t(
                    "Distances are straight-line estimates. School coordinates are preferred; commune and regional locations are approximate fallbacks."
                )
            )

        st.markdown(t("#### Improve distance estimates — optional"))
        st.caption(
            t("Enter a home address to measure distance from home. Otherwise, distance is estimated from the current preference list.")
        )
        with st.popover(t("Address privacy and precision")):
            st.write(
                t(
                    "The address is sent to OpenStreetMap/Nominatim only after the family clicks the geocoding button. Distances remain approximate and do not represent travel time."
                )
            )
        # Keep recommendation/address state outside the wish-editor namespace.
        # clear_wish_editor_widget_state(editor_widget_key_base) deletes every
        # key starting with editor_widget_key_base after wish-list edits; using
        # a separate prefix prevents the family home address and geocoded point
        # from being erased when wishes are added, removed, or reordered.
        recommendation_widget_key_base = f"recommendations_{editor_widget_key_base}"
        address_key = f"{recommendation_widget_key_base}_home_address"
        geo_key = f"{recommendation_widget_key_base}_home_geo"
        home_address = st.text_input(
            t("Student home address"),
            key=address_key,
            help=t("Optional. Used only to compute distance/proximity to recommended programs. If left empty, proximity is estimated from the current wish list. The address is geocoded with OpenStreetMap/Nominatim when you click the button."),
        )
        address_cols = st.columns([1, 1, 3])
        with address_cols[0]:
            geocode_clicked = st.button(
                t("Use this address for distance"),
                disabled=not bool(str(home_address or "").strip()),
                key=f"{recommendation_widget_key_base}_geocode_address",
            )
        with address_cols[1]:
            clear_address_clicked = st.button(
                t("Clear address"),
                key=f"{recommendation_widget_key_base}_clear_address",
            )

        if geocode_clicked:
            st.session_state[geo_key] = geocode_chilean_address(home_address)
        if clear_address_clicked:
            st.session_state.pop(address_key, None)
            st.session_state.pop(geo_key, None)
            st.rerun()

        home_geo_reference = None
        stored_geo = st.session_state.get(geo_key)
        normalized_current_address = " ".join(str(home_address or "").strip().split())
        if stored_geo and stored_geo.get("address") == normalized_current_address:
            if stored_geo.get("ok") and valid_lat_lon(stored_geo.get("lat"), stored_geo.get("lon")):
                home_geo_reference = stored_geo
                display_geocoded_address = stored_geo.get("display_name", normalized_current_address)
                precision = stored_geo.get("precision", "approximate")

                if precision == "address":
                    st.success(
                        t(
                            "Distances will be computed from the confirmed address: {address}",
                            address=display_geocoded_address,
                        )
                    )
                else:
                    warning_key = geocoding_precision_warning_key(stored_geo)
                    st.warning(
                        t(
                            "{warning} Location used: {address}",
                            warning=t(warning_key)
                            if warning_key
                            else t("The geocoded location is approximate. Distances should be interpreted carefully."),
                            address=display_geocoded_address,
                        )
                    )
            elif normalized_current_address:
                error_key = str(stored_geo.get("error_key", "")).strip()
                error_kwargs = stored_geo.get("error_kwargs", {})
                if not isinstance(error_kwargs, dict):
                    error_kwargs = {}

                # Backward-compatible fallback for a result already stored in
                # session state before language-neutral geocoding errors were
                # introduced.
                error_text = (
                    t(error_key, **error_kwargs)
                    if error_key
                    else str(stored_geo.get("error", ""))
                )
                st.warning(
                    t("Address could not be geocoded: {error}", error=error_text)
                )
        elif normalized_current_address and stored_geo:
            st.info(t("Address changed. Click the button to update the coordinates."))

        hard_home_distance_filter_available = bool(
            home_geo_reference
            and home_geocoding_supports_hard_filter(home_geo_reference)
        )
        if home_geo_reference:
            if hard_home_distance_filter_available:
                st.caption(
                    t(
                        "The {max_distance:.0f} km straight-line limit is applied only when the program has reliable school-level coordinates. Commune and regional approximations are never used for hard exclusion.",
                        max_distance=RECOMMENDATION_MAX_HOME_DISTANCE_KM,
                    )
                )
            else:
                st.caption(
                    t(
                        "Because the home location is only city-level or approximate, no hard distance cutoff is applied. Straight-line distance only affects the recommendation score."
                    )
                )

        with st.expander(t("Recommendation display settings"), expanded=False):
            rec_max = st.slider(
                t("Number of recommendations"),
                min_value=2,
                max_value=10,
                value=5,
                step=1,
            )

        # Recommendation behavior is intentionally hidden from families.
        # Tune these constants in sae_app/recommendations.py if needed:
        # RECOMMENDATION_HARD_CONSTRAINT_COLS, RECOMMENDATION_COMPETITION_WEIGHT,
        # RECOMMENDATION_RISK_OPTIMIZATION_WEIGHT, RECOMMENDATION_PROXIMITY_WEIGHT,
        # RECOMMENDATION_DISTANCE_SCALE_KM, RECOMMENDATION_DIVERSIFY,
        # RECOMMENDATION_DIVERSITY_STRENGTH.
        hard_constraint_cols = RECOMMENDATION_HARD_CONSTRAINT_COLS
        favor_less_oversubscribed = RECOMMENDATION_FAVOR_LESS_OVERSUBSCRIBED
        competition_weight = RECOMMENDATION_COMPETITION_WEIGHT if favor_less_oversubscribed else 0.0
        risk_optimization_weight = RECOMMENDATION_RISK_OPTIMIZATION_WEIGHT
        proximity_weight = RECOMMENDATION_PROXIMITY_WEIGHT
        distance_scale_km = RECOMMENDATION_DISTANCE_SCALE_KM
        diversify = RECOMMENDATION_DIVERSIFY
        diversity_strength = RECOMMENDATION_DIVERSITY_STRENGTH if diversify else 0.0

        try:
            recommendations, _profile_table = recommend_similar_programs(
                edited,
                program_mapping,
                student_id=student_id,
                current_unmatched_risk=current_unmatched_risk,
                max_recommendations=rec_max,
                rank_sensitive=True,
                competition_weight=competition_weight,
                hard_constraint_cols=hard_constraint_cols,
                proximity_weight=proximity_weight,
                distance_scale_km=float(distance_scale_km),
                home_geo_reference=home_geo_reference,
                max_home_distance_km=(
                    RECOMMENDATION_MAX_HOME_DISTANCE_KM
                    if home_geo_reference
                    else None
                ),
                risk_optimization_weight=risk_optimization_weight,
                min_similarity_score=RECOMMENDATION_MIN_SIMILARITY_SCORE,
                diversify=diversify,
                diversity_strength=diversity_strength,
            )
        except MtbEngineError as exc:
            LOGGER.warning("Expected recommendation-engine error: %s", exc)
            st.error(t(exc.message_key, **exc.message_kwargs))
            if APP_DEBUG:
                st.exception(exc)
            return
        except CandidateEvaluationError as exc:
            LOGGER.warning(
                "Recommendation calculation stopped because of malformed data: %s",
                exc,
            )
            st.error(
                t(
                    "Recommendations could not be computed because some program data are invalid."
                )
            )
            if APP_DEBUG:
                st.exception(exc)
            return
        except Exception as exc:
            LOGGER.exception("Unexpected recommendation calculation error")
            st.error(
                t(
                    "Recommendations could not be computed because of an unexpected internal error."
                )
            )
            if APP_DEBUG:
                st.exception(exc)
            return

        recommendation_diagnostics = recommendations.attrs.get(
            "recommendation_diagnostics",
            {},
        )
        failed_candidates = int(
            recommendation_diagnostics.get("failed_candidates", 0) or 0
        )

        if recommendations.empty:
            if failed_candidates:
                st.warning(
                    t(
                        "Some programs could not be evaluated because of invalid source data, and no recommendation was produced."
                    )
                )
                if APP_DEBUG:
                    for candidate_label, error in recommendation_diagnostics.get(
                        "failed_candidate_examples",
                        (),
                    ):
                        st.caption(f"{candidate_label}: {error}")
            elif hard_home_distance_filter_available:
                st.warning(
                    t(
                        "No recommended program matched the current scoring and reliable straight-line distance rules. You can still add programs manually."
                    )
                )
            else:
                st.warning(
                    t("No similar program was found under the current proximity/scoring rules.")
                )
            return

        if APP_DEBUG and failed_candidates:
            for candidate_label, error in recommendation_diagnostics.get(
                "failed_candidate_examples",
                (),
            ):
                st.caption(f"Skipped candidate — {candidate_label}: {error}")

        if (
            "_similarity_fallback_mode" in recommendations.columns
            and recommendations["_similarity_fallback_mode"].fillna(False).astype(bool).any()
        ):
            st.info(
                t("The current list does not contain enough usable information to infer clear similar-program preferences. The suggestions below are therefore based mainly on distance and estimated admission safety.")
            )

        risk_values_missing = (
            "Chance if considered" in recommendations.columns
            and recommendations["Chance if considered"].astype(str).str.strip().eq("").all()
        )
        if risk_values_missing:
            st.warning(
                t("Portfolio-risk estimates could not be computed. Check that the student's RUN/IPE is still entered, then rerun the simulation.")
            )

        st.markdown(t("#### Suggested programs"))
        distance_column = (
            "Straight-line distance from home (km)"
            if home_geo_reference
            else "Straight-line distance from current list (km)"
        )
        st.caption(
            t("Select only programs the family would genuinely accept. Recommendations are options to investigate, not automatic choices.")
        )

        programs_to_add = []
        for _, recommendation in recommendations.iterrows():
            program_label = str(recommendation.get(PROGRAM, "")).strip()
            row_key = hashlib.md5(program_label.encode("utf-8")).hexdigest()[:10]
            school = str(recommendation.get("School", "")).strip()
            commune = str(recommendation.get("Commune", "")).strip()
            region = str(recommendation.get("Region", "")).strip()
            program_details = str(recommendation.get("Program details", "")).strip()
            distance = recommendation.get(distance_column, "")
            chance_if_considered = str(recommendation.get("Chance if considered", "")).strip()
            projected_risk = str(
                recommendation.get("Projected unmatched risk after append", "")
            ).strip()
            risk_color = str(recommendation.get("_risk_color", "gray")).strip()

            with st.container(border=True):
                st.markdown(f"### {school}")
                location_parts = [part for part in [commune, region] if part]
                if location_parts:
                    st.caption(" · ".join(location_parts))
                if program_details:
                    st.write(program_details)

                if distance != "" and not pd.isna(distance):
                    st.caption(
                        t(
                            "Approximate straight-line distance: {distance:.1f} km",
                            distance=float(distance),
                        )
                    )

                if np.isfinite(current_unmatched_risk) and projected_risk:
                    impact_message = t(
                        "If appended at the end: estimated unmatched risk {current:.1%} → {projected}",
                        current=current_unmatched_risk,
                        projected=projected_risk,
                    )
                    if risk_color == "green":
                        st.success(impact_message)
                    elif risk_color == "orange":
                        st.warning(impact_message)
                    elif risk_color == "red":
                        st.error(impact_message)
                    else:
                        st.info(impact_message)

                if chance_if_considered:
                    st.write(
                        t(
                            "Estimated chance if the student reaches this preference: **{chance}**",
                            chance=chance_if_considered,
                        )
                    )

                st.caption(
                    t(
                        "Why it appears: it balances similarity to the current list, proximity and estimated admission safety."
                    )
                )

                with st.popover(t("View calculation details")):
                    st.markdown(
                        f"**{t('Capacity')}:** {recommendation.get('Capacity', t('No information'))}"
                    )
                    st.markdown(
                        f"**{t('Applicants / seat')}:** {recommendation.get('Applicants / seat', t('No information'))}"
                    )
                    st.markdown(
                        f"**{t('Estimated MTB rank')}:** {recommendation.get('Estimated MTB rank', t('No information'))}"
                    )
                    st.caption(
                        t(
                            "This estimate assumes the program is appended at the end and no special priority is declared."
                        )
                    )

                if st.checkbox(
                    t("Add this program to the end of my list"),
                    key=f"{recommendation_widget_key_base}_select_{row_key}",
                ):
                    programs_to_add.append(program_label)

        if st.button(
            t("Add selected programs and review my list"),
            disabled=not programs_to_add,
            type="primary",
            use_container_width=True,
        ):
            non_empty = edited.copy()
            non_empty[PROGRAM] = non_empty[PROGRAM].fillna("").astype(str).str.strip()
            non_empty = non_empty[non_empty[PROGRAM] != ""].copy()

            existing = set(non_empty[PROGRAM].tolist())
            selected_programs = set(programs_to_add)

            # Preserve the recommendation ranking. A multiselect records which
            # programs were selected, not an explicit preference order.
            new_programs = [
                program_label
                for program_label in recommendations[PROGRAM].tolist()
                if program_label in selected_programs and program_label not in existing
            ]

            if len(non_empty) > 0:
                existing_ranks = pd.to_numeric(
                    non_empty.get(WISH_RANK, pd.Series(dtype=float)),
                    errors="coerce",
                ).dropna()
                existing_groups = pd.to_numeric(
                    non_empty.get(EQUIV_GROUP, pd.Series(dtype=float)),
                    errors="coerce",
                ).dropna()

                next_rank = int(existing_ranks.max()) + 1 if not existing_ranks.empty else len(non_empty) + 1
                next_group = int(existing_groups.max()) + 1 if not existing_groups.empty else next_rank
            else:
                next_rank = 1
                next_group = 1

            rows_to_add = make_appended_recommendation_rows(
                new_programs,
                next_rank=next_rank,
                next_group=next_group,
                use_equivalence_classes=use_equivalence_classes,
            )

            if rows_to_add:
                updated_wishes = pd.concat(
                    [non_empty, pd.DataFrame(rows_to_add)],
                    ignore_index=True,
                )
                st.session_state[editor_state_key] = clean_wish_rows(updated_wishes)
                st.session_state["recommendations_added_notice"] = len(rows_to_add)
                invalidate_simulation_state(
                    simulation_done_key=simulation_done_key,
                    simulation_result_key=simulation_result_key,
                )
                clear_wish_editor_widget_state(editor_widget_key_base)
                st.rerun()
            else:
                st.info(t("All selected recommendations are already in the wish list."))
