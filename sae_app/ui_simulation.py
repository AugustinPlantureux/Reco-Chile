"""Rendering the simulation results: summary, podium, and equivalence-class
sensitivity table.

render_simulation_result() is the single entry point: both the "Calculate
unmatched risk" button handler and the "still showing a past result" path in
app.py call it, so there is exactly one place that draws this UI.
"""

from __future__ import annotations

import streamlit as st

from sae_app.constants import (
    EQUIV_PROBABILITY_CHANGE_WARNING_THRESHOLD,
    HARD_UNMATCHED_THRESHOLD,
    SOFT_UNMATCHED_THRESHOLD,
)
from sae_app.i18n import display_outcome_label, t
from sae_app.ui_common import format_display_table



def format_choices_table(choices):
    display_cols = [
        "wish_rank",
        "program",
        "lottery_number",
        "priority_tier",
        "capacity",
        "true_applicants_last_year",
        "availability_probability",
        "choice_assignment_probability",
    ]

    table = choices[display_cols].copy()
    table["program"] = table["program"].map(display_outcome_label)
    table["priority_tier"] = table["priority_tier"].map(t)
    for prob_col in ("availability_probability", "choice_assignment_probability"):
        table[prob_col] = table[prob_col].astype(float).map(lambda x: f"{x:.1%}")

    return table.rename(columns={
        "wish_rank": t("Wish rank"),
        "program": t("Program"),
        "lottery_number": t("Calculated MTB lottery rank"),
        "priority_tier": t("Priority tier"),
        "capacity": t("Seats"),
        "true_applicants_last_year": t("Applicants in the historical calibration"),
        "availability_probability": t("Chance if considered"),
        "choice_assignment_probability": t("Final chance of assignment"),
    })


def format_family_choices_table(choices):
    """Return the short table needed for a first reading of the result."""
    table = choices[["wish_rank", "program", "choice_assignment_probability"]].copy()
    table["program"] = table["program"].map(display_outcome_label)
    table["choice_assignment_probability"] = table[
        "choice_assignment_probability"
    ].astype(float).map(lambda value: f"{value:.1%}")
    return table.rename(
        columns={
            "wish_rank": t("Preference"),
            "program": t("Establishment"),
            "choice_assignment_probability": t("Estimated final chance"),
        }
    )


def _split_tied_group_orders(value) -> list[list[str]]:
    """Parse the stored tied-group order into independently renderable groups."""
    text = str(value or "").strip()
    if not text:
        return []
    return [
        [program.strip() for program in block.split(" → ") if program.strip()]
        for block in text.split(" | ")
        if block.strip()
    ]


def _render_tied_order(value) -> None:
    """Render one or more tied groups as short numbered rankings."""
    groups = _split_tied_group_orders(value)
    if not groups:
        st.write(t("No tied-program order was recorded for this option."))
        return

    for group_index, programs in enumerate(groups, start=1):
        if len(groups) > 1:
            st.markdown(t("**Tied group {group}:**", group=group_index))
        for rank, program in enumerate(programs, start=1):
            st.write(f"{rank}. {display_outcome_label(program)}")


def _family_order_view(variants_df) -> None:
    """Show families which tied-program order leads to which predicted result."""
    order_column = (
        "Order inside tied programs"
        if "Order inside tied programs" in variants_df.columns
        else "Strict order"
    )
    rows = variants_df.sort_values("Strict order #", kind="stable").reset_index(drop=True)

    st.markdown(t("#### What each order inside the tied programs leads to"))
    st.caption(
        t(
            "Only programs tied within the same preference group are shown below. "
            "Programs whose position never changes are omitted."
        )
    )

    if len(rows) <= 12:
        for option_number, (_, row) in enumerate(rows.iterrows(), start=1):
            with st.container(border=True):
                st.markdown(t("### Option {number}", number=option_number))
                st.markdown(t("**Place the tied programs in this order:**"))
                _render_tied_order(row.get(order_column, ""))

                outcome = display_outcome_label(row.get("Predicted outcome", ""))
                chance = float(row.get("Predicted outcome final chance", 0.0))
                st.success(t("Most likely outcome: **{outcome}**", outcome=outcome))
                st.caption(
                    t(
                        "Estimated final chance for this outcome: {chance:.1%}",
                        chance=chance,
                    )
                )
    else:
        st.caption(
            t(
                "Because there are {n:,} compatible orders, they are grouped below "
                "by their most likely outcome.",
                n=len(rows),
            )
        )
        grouped = rows.groupby("Predicted outcome", sort=False, dropna=False)
        for outcome, outcome_rows in grouped:
            display_outcome = display_outcome_label(outcome)
            with st.expander(
                t(
                    "{outcome} — {n:,} compatible order(s)",
                    outcome=display_outcome,
                    n=len(outcome_rows),
                ),
                expanded=False,
            ):
                family_table = outcome_rows[[order_column, "Predicted outcome final chance"]].copy()
                family_table["Predicted outcome final chance"] = family_table[
                    "Predicted outcome final chance"
                ].map(lambda value: f"{float(value):.1%}")
                family_table = family_table.rename(
                    columns={
                        order_column: "Order inside tied programs",
                        "Predicted outcome final chance": "Final chance for predicted outcome",
                    }
                )
                st.dataframe(
                    format_display_table(family_table),
                    width="stretch",
                    hide_index=True,
                )

    st.info(
        t(
            "Inside each tied group, place first the program the family genuinely "
            "prefers. The overall unmatched risk does not change, but the most "
            "likely school can change."
        )
    )


def ordered_estimated_outcomes(choices) -> list[dict[str, str | float]]:
    """Return all outcomes ordered only by their modeled probability."""
    p_unmatched = float(choices["cumulative_unavailable_after_choice"].iloc[-1])
    outcomes = [
        {
            "label": display_outcome_label(row["program"]),
            "probability": float(row["choice_assignment_probability"]),
        }
        for _, row in choices.iterrows()
        if float(row["choice_assignment_probability"]) > 0
    ]
    outcomes.append({"label": t("Unmatched"), "probability": p_unmatched})
    return sorted(outcomes, key=lambda item: item["probability"], reverse=True)


def render_single_summary(
    choices,
    hard_threshold: float,
    soft_threshold: float = SOFT_UNMATCHED_THRESHOLD,
) -> None:
    """Render the decision first and keep alert severity separate from likelihood."""
    p_unmatched = float(choices["cumulative_unavailable_after_choice"].iloc[-1])
    hard_at_risk = p_unmatched >= hard_threshold
    soft_at_risk = soft_threshold <= p_unmatched < hard_threshold

    st.subheader(t("Result of the preference list"))
    st.metric(t("Estimated risk of remaining without an assignment"), f"{p_unmatched:.1%}")
    if hard_at_risk:
        st.error(
            t("High attention level: consider adding more acceptable programs at the end of the list.")
        )
    elif soft_at_risk:
        st.warning(
            t("Moderate attention level: review the list and consider acceptable backup options.")
        )
    else:
        st.success(t("Low attention level under the tool's current alert settings."))

    st.caption(
        t(
            "This percentage is an estimate based on historical data and model assumptions; it is not an official SAE result or guarantee."
        )
    )

    outcomes = ordered_estimated_outcomes(choices)

    st.markdown(t("#### Most likely estimated outcomes"))
    for position, item in enumerate(outcomes[:4], start=1):
        st.markdown(f"{position}. **{item['label']}** — {item['probability']:.1%}")

    if len(outcomes) > 4:
        with st.expander(t("Show all estimated outcomes"), expanded=False):
            for position, item in enumerate(outcomes, start=1):
                st.write(f"{position}. {item['label']} — {item['probability']:.1%}")

    with st.popover(t("How should I interpret these percentages?")):
        st.write(
            t(
                "Outcomes are always ordered by their estimated probability. The attention alert is displayed separately and never changes this ranking."
            )
        )
        st.write(
            t(
                "A program's final chance accounts for every program placed above it. The unmatched risk is the estimated chance that none of the listed programs is available."
            )
        )

    with st.expander(t("How are the attention levels defined?"), expanded=False):
        st.write(
            t(
                "These are presentation thresholds defined by this research tool, not official SAE thresholds. Low is below {soft:.1%}; moderate is from {soft:.1%} to below {hard:.1%}; high is {hard:.1%} or above.",
                soft=soft_threshold,
                hard=hard_threshold,
            )
        )


def render_simulation_result(result: dict) -> None:
    """Render the last simulation, including equivalence-class sensitivity.

    The result is stored in session_state so it remains visible after the user
    interacts with recommendation sliders or add-program controls.
    """
    if not result:
        return

    hard_threshold_used = float(result.get("hard_threshold", result.get("threshold", HARD_UNMATCHED_THRESHOLD)))
    soft_threshold_used = float(result.get("soft_threshold", SOFT_UNMATCHED_THRESHOLD))
    mode = result.get("mode", "strict")

    if mode == "equivalence":
        reference_choices = result.get("reference_choices")
        variants_df = result.get("variants_df")
        distinct_outcomes = result.get("distinct_outcomes", [])

        if reference_choices is None or variants_df is None or len(variants_df) == 0:
            return

        render_single_summary(reference_choices, hard_threshold_used, soft_threshold_used)

        st.subheader(t("Does the undecided internal order matter?"))

        predicted_chance_values = []
        if "Predicted outcome final chance" in variants_df.columns:
            for value in variants_df["Predicted outcome final chance"].dropna().tolist():
                try:
                    predicted_chance_values.append(float(value))
                except (TypeError, ValueError):
                    continue

        predicted_chance_min = min(predicted_chance_values) if predicted_chance_values else None
        predicted_chance_max = max(predicted_chance_values) if predicted_chance_values else None
        predicted_chance_range = (
            predicted_chance_max - predicted_chance_min
            if predicted_chance_min is not None and predicted_chance_max is not None
            else 0.0
        )
        same_outcome_but_probability_changes = (
            len(distinct_outcomes) == 1
            and predicted_chance_range >= EQUIV_PROBABILITY_CHANGE_WARNING_THRESHOLD
        )

        if same_outcome_but_probability_changes:
            st.warning(
                t(
                    "The strict ordering inside the equivalence classes does not change the most likely school: **{outcome}**. However, it changes the final assignment probability for that school, from {min_chance:.1%} to {max_chance:.1%} across compatible strict order(s).",
                    outcome=display_outcome_label(distinct_outcomes[0]),
                    min_chance=predicted_chance_min,
                    max_chance=predicted_chance_max,
                )
            )
            st.caption(
                t("Changing the internal order does not change the main predicted school, but it can still affect the chances of receiving other options. The overall unmatched risk remains unchanged, so the family should still choose the internal order carefully.")
            )
        elif len(distinct_outcomes) == 1:
            st.success(
                t(
                    "The strict ordering inside the equivalence classes does not change the predicted final outcome. All {n:,} compatible strict order(s) lead to: **{outcome}**.",
                    n=len(variants_df),
                    outcome=display_outcome_label(distinct_outcomes[0]),
                )
            )
            st.caption(
                t("Changing the internal order does not change the predicted school. The overall unmatched risk also remains unchanged across compatible orders.")
            )
        else:
            st.warning(
                t("The strict ordering inside at least one equivalence class can change the predicted final outcome. The user should choose a strict order carefully for the tied programs.")
            )
            st.caption(
                t("Changing the internal order can lead to different predicted schools. The overall unmatched risk remains unchanged; only the distribution of assignment probabilities across schools changes.")
            )

        if len(distinct_outcomes) > 1 or same_outcome_but_probability_changes:
            _family_order_view(variants_df)

        with st.expander(t("Detailed calculation for the reference order"), expanded=False):
            st.caption(
                t("This reference uses the current row order inside each preference group.")
            )
            st.dataframe(format_choices_table(reference_choices), width="stretch", hide_index=True)

        with st.expander(t("Technical details of all tested orders"), expanded=False):
            st.caption(
                t(
                    "This technical table contains the complete strict ranking for every "
                    "tested permutation. It is not needed to choose the order inside tied groups."
                )
            )
            variants_display = variants_df.copy()
            if "Predicted outcome final chance" in variants_display.columns:
                variants_display["Predicted outcome final chance"] = variants_display[
                    "Predicted outcome final chance"
                ].map(lambda x: f"{x:.1%}")
            variants_display = variants_display.drop(
                columns=["Unmatched risk", "Order inside tied programs"],
                errors="ignore",
            )
            st.dataframe(format_display_table(variants_display), width="stretch", hide_index=True)
        return

    choices = result.get("choices")
    if choices is None:
        return

    render_single_summary(choices, hard_threshold_used, soft_threshold_used)

    st.subheader(t("Estimated final chance by preference"))
    st.caption(
        t("The final chance accounts for every program placed above each preference.")
    )
    st.dataframe(format_family_choices_table(choices), width="stretch", hide_index=True)

    with st.popover(t("Chance if considered vs. final chance")):
        st.write(
            t(
                "Chance if considered estimates access to a program if the student reaches that preference. Final chance also accounts for the possibility of receiving a higher-ranked program first."
            )
        )

    with st.expander(t("See the detailed calculation for each preference"), expanded=False):
        st.dataframe(format_choices_table(choices), width="stretch", hide_index=True)
        st.caption(
            t(
                "MTB ranks, priority tiers, seats and historical applicant counts are calculation details. They should not be interpreted as official SAE results."
            )
        )
