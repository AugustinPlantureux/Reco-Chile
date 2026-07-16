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
        "true_applicants_last_year": t("True applicants last year"),
        "availability_probability": t("Chance if considered"),
        "choice_assignment_probability": t("Final chance of assignment"),
    })


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


def render_single_summary(
    choices,
    hard_threshold: float,
    soft_threshold: float = SOFT_UNMATCHED_THRESHOLD,
) -> None:
    p_unmatched = float(choices["cumulative_unavailable_after_choice"].iloc[-1])
    hard_at_risk = p_unmatched >= hard_threshold
    soft_at_risk = soft_threshold <= p_unmatched < hard_threshold

    st.subheader(t("Summary"))
    st.metric(t("Unmatched risk"), f"{p_unmatched:.1%}")
    st.caption(
        t(
            "How to read this: above {hard:.1%}, Unmatched is shown first. Between {soft:.1%} and {hard:.1%}, it is shown in the podium as a warning. Below {soft:.1%}, only schools are shown in the podium.",
            hard=hard_threshold,
            soft=soft_threshold,
        )
    )

    positive = (
        choices[choices["choice_assignment_probability"] > 0]
        .sort_values("choice_assignment_probability", ascending=False)
        .reset_index(drop=True)
    )

    if hard_at_risk:
        st.error(
            t("Strong unmatched-risk alert: the risk is above the hard threshold. Unmatched is therefore shown as the first outcome. Adding safer options is recommended.")
        )
        if positive.empty:
            st.markdown(t("**Most likely outcome:**"))
            st.write(f"1. {t('Unmatched')}")
        else:
            st.markdown(t("**Most likely outcomes:**"))
            st.write(f"1. {t('Unmatched')}")
            for i, row in positive.head(2).iterrows():
                st.write(f"{i + 2}. {display_outcome_label(row['program'])}")
            st.caption(
                t("The schools listed below Unmatched are still the most likely school assignments, but the unmatched risk is high enough to be treated as the main warning.")
            )
    elif soft_at_risk:
        if positive.empty:
            st.error(t("No listed school appears realistically accessible."))
            return

        best = positive.iloc[0]
        st.warning(
            t(
                "Moderate unmatched-risk warning: the most likely assignment is **{program}**, but the unmatched risk is high enough to appear in the podium.",
                program=display_outcome_label(best["program"]),
            )
        )

        podium = [
            {
                "label": display_outcome_label(row["program"]),
                "probability": float(row["choice_assignment_probability"]),
                "is_unmatched": False,
            }
            for _, row in positive.iterrows()
        ]
        podium.append({
            "label": "Unmatched",
            "probability": p_unmatched,
            "is_unmatched": True,
        })
        podium = sorted(podium, key=lambda x: x["probability"], reverse=True)
        top3 = podium[:3]

        if not any(item["is_unmatched"] for item in top3):
            top3 = top3[:2] + [
                {
                    "label": "Unmatched",
                    "probability": p_unmatched,
                    "is_unmatched": True,
                }
            ]

        st.markdown(t("**Top 3 most likely outcomes:**"))
        for i, item in enumerate(top3, start=1):
            st.write(f"{i}. {display_outcome_label(item['label'])}")
        st.caption(
            t("Unmatched is included here as a warning signal because the risk is above the soft threshold; it is not forced into first place unless the hard threshold is reached.")
        )
    else:
        if positive.empty:
            st.error(t("No listed school appears realistically accessible."))
        else:
            best = positive.iloc[0]
            st.success(
                t(
                    "The student is not flagged as at risk. The most likely assignment is: **{program}**.",
                    program=display_outcome_label(best["program"]),
                )
            )
            st.caption(
                t("The unmatched risk is below the soft threshold, so the podium focuses on school assignments only.")
            )
            st.markdown(t("**Top 3 most likely schools:**"))
            for i, row in positive.head(3).iterrows():
                st.write(f"{i + 1}. {display_outcome_label(row['program'])}")


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

        st.subheader(t("Reference strict-order details"))
        st.caption(
            t("This table shows one reference order: the current row order inside each preference group. The sensitivity test below then checks every strict order that is compatible with the groups. This matters because tied programs can still lead to different predicted schools.")
        )
        st.dataframe(format_choices_table(reference_choices), width="stretch", hide_index=True)
        render_single_summary(reference_choices, hard_threshold_used, soft_threshold_used)

        st.subheader(t("Equivalence-class sensitivity"))

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

        if len(distinct_outcomes) == 1 and same_outcome_but_probability_changes:
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

    st.subheader(t("Wish-level details"))
    st.caption(
        t("Chance if considered is the chance of getting that program if the student reaches that wish. Final chance of assignment also accounts for all higher-ranked wishes. For example, a school can be accessible if considered, but have a lower final chance if the student is likely to get a higher-ranked option first.")
    )
    st.dataframe(format_choices_table(choices), width="stretch", hide_index=True)
    render_single_summary(choices, hard_threshold_used, soft_threshold_used)
