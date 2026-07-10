"""Rendering the "recommended similar programs" section.

render_similar_program_recommendations() draws the whole section 4: the home
address / geocoding controls, the recommendation table, and the "add selected
recommendations to the wish list" button.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from sae_app.constants import (
    EQUIV_GROUP,
    HARD_UNMATCHED_THRESHOLD,
    PROGRAM,
    SOFT_UNMATCHED_THRESHOLD,
    WISH_RANK,
)
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
from sae_app.ui_common import format_display_table
from sae_app.wish_list import clean_wish_rows, make_builder_wish_row


def format_recommendation_display(recommendations: pd.DataFrame, visible_columns: list[str]):
    """Return a color-coded Styler for recommendation rows."""
    display = format_display_table(recommendations[visible_columns].reset_index(drop=True))
    colors = recommendations.get("_risk_color", pd.Series(["gray"] * len(recommendations))).reset_index(drop=True)

    def style_row(row):
        color = colors.iloc[row.name] if row.name < len(colors) else "gray"
        if color == "green":
            style = "background-color: #e8f5e9"
        elif color == "orange":
            style = "background-color: #fff3e0"
        elif color == "red":
            style = "background-color: #ffebee"
        else:
            style = ""
        return [style for _ in row]

    return display.style.apply(style_row, axis=1)


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
    st.subheader(t("4. Recommended similar programs"))

    with st.expander(t("Find additional programs similar to the current wish list"), expanded=True):
        current_selected_programs = [
            p for p in edited[PROGRAM].dropna().astype(str).str.strip()
            if p and p in program_mapping
        ]

        if not current_selected_programs:
            st.info(t("Enter at least one valid program in the wish list to get recommendations."))
            return

        simulation_result = st.session_state.get(simulation_result_key, {}) if simulation_result_key else {}
        current_unmatched_risk = current_unmatched_risk_from_simulation_result(simulation_result)

        st.caption(
            t("Recommendations combine revealed preferences, proximity, portfolio-risk improvement, and a diversity step to avoid near-duplicates.")
        )
        st.caption(
            t("Proximity uses program/commune coordinates when available. Regional centroids are only a soft fallback and are never used for hard distance exclusion. To improve precision, add data/commune_coordinates.csv with commune, region, latitude, longitude.")
        )

        st.markdown(t("#### Portfolio-risk optimization"))
        if np.isfinite(current_unmatched_risk):
            st.metric(t("Current unmatched risk"), f"{current_unmatched_risk:.1%}")
        st.caption(
            t(
                'Each recommended program is evaluated as if it were appended after the current wish list. "Chance if considered" is conditional on the student reaching that wish. The marginal unmatched-risk reduction is the current unmatched risk multiplied by that conditional chance; it is also the estimated final assignment chance for a program appended at the end. Projected unmatched risk is the current risk minus that reduction.'
            )
        )
        st.caption(
            t("Recommended programs assume no special priority flags for the newly added school. If the student has a sibling, priority-student quota, civil-servant, former-student, or already-enrolled priority for that school, add the program to the list and mark the priority before rerunning the simulation.")
        )
        st.info(
            t("Strategic note: adding additional acceptable programs at the end of the wish list does not reduce the student's chance of getting higher-ranked choices. The assignment process considers the list in order and keeps the best available option. Families should therefore add every acceptable backup program, then mark any applicable priority for those added schools and rerun the simulation.")
        )

        st.markdown(t("#### Home address for distance calculation"))
        st.caption(
            t("Enter an address, then click the button to compute recommendation distances from home instead of the current wish-list centroid.")
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
                st.warning(t("Address could not be geocoded: {error}", error=stored_geo.get("error", "")))
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
                        "The {max_distance:.0f} km straight-line limit is applied only when the program has reliable school/commune coordinates. Regional approximations are never used for hard exclusion.",
                        max_distance=RECOMMENDATION_MAX_HOME_DISTANCE_KM,
                    )
                )
            else:
                st.caption(
                    t(
                        "Because the home location is only city-level or approximate, no hard distance cutoff is applied. Straight-line distance only affects the recommendation score."
                    )
                )

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
            max_home_distance_km=RECOMMENDATION_MAX_HOME_DISTANCE_KM if home_geo_reference else None,
            risk_optimization_weight=risk_optimization_weight,
            min_similarity_score=RECOMMENDATION_MIN_SIMILARITY_SCORE,
            diversify=diversify,
            diversity_strength=diversity_strength,
        )

        if recommendations.empty:
            if hard_home_distance_filter_available:
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

        if (
            "_similarity_fallback_mode" in recommendations.columns
            and recommendations["_similarity_fallback_mode"].fillna(False).astype(bool).any()
        ):
            st.info(
                t("The current list does not contain enough usable information to infer clear similar-program preferences. The suggestions below are therefore based mainly on distance and admission-risk reduction.")
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
        visible_columns = [
            "School",
            "Commune",
            "Region",
            "Program details",
            distance_column,
            "Chance if considered",
            "Recommendation score",
            "Capacity",
            "Applicants / seat",
            "Estimated MTB rank",
        ]
        st.caption(
            t("Distances are straight-line estimates. They do not represent road distance, travel time, or actual accessibility.")
        )
        st.dataframe(
            format_recommendation_display(recommendations, visible_columns),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            t(
                "Row colors reflect the projected unmatched risk after appending the program: green below the soft threshold ({soft:.1%}), orange between the soft and hard thresholds, and red at or above the hard threshold ({hard:.1%}).",
                soft=SOFT_UNMATCHED_THRESHOLD,
                hard=HARD_UNMATCHED_THRESHOLD,
            )
        )
        st.caption(
            t("Portfolio-risk estimates are marginal: they assume the program is appended after the current list. Reordering it higher would change final probabilities and should be tested by adding it to the wish list and rerunning the simulation.")
        )

        programs_to_add = st.multiselect(
            t("Add recommended programs to the wish list"),
            options=recommendations[PROGRAM].tolist(),
            default=[],
        )

        if st.button(t("Add selected recommendations"), disabled=not programs_to_add):
            non_empty = edited.copy()
            non_empty[PROGRAM] = non_empty[PROGRAM].fillna("").astype(str).str.strip()
            non_empty = non_empty[non_empty[PROGRAM] != ""].copy()

            existing = set(non_empty[PROGRAM].tolist())
            new_programs = [p for p in programs_to_add if p not in existing]

            rows_to_add = []
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

            for i, program_label in enumerate(new_programs):
                if use_equivalence_classes:
                    # In equivalence-class mode, recommended programs are appended to
                    # the next preference group, not to the next strict row number.
                    wish_rank_value = next_rank + i
                    equivalence_group_value = next_group
                else:
                    wish_rank_value = next_rank + i
                    equivalence_group_value = next_rank + i

                rows_to_add.append(
                    make_builder_wish_row(program_label, wish_rank_value, equivalence_group_value)
                )

            if rows_to_add:
                updated_wishes = pd.concat(
                    [non_empty, pd.DataFrame(rows_to_add)],
                    ignore_index=True,
                )
                st.session_state[editor_state_key] = clean_wish_rows(updated_wishes)
                invalidate_simulation_state(
                    simulation_done_key=simulation_done_key,
                    simulation_result_key=simulation_result_key,
                )
                clear_wish_editor_widget_state(editor_widget_key_base)
                st.rerun()
            else:
                st.info(t("All selected recommendations are already in the wish list."))
