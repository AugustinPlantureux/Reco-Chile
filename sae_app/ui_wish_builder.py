"""Rendering the wish-list builder widget.

render_wish_list_builder() draws the "add a program" selector plus the
current list of wishes (with reorder arrows in strict mode, or a group
number input in equivalence-class mode), and returns the same cleaned
DataFrame the simulation engine expects.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from sae_app.constants import (
    EQUIV_GROUP,
    LOTTERY,
    PRIORITIES,
    PROGRAM,
    PROGRAM_DISPLAY_NAME,
    PROGRAM_ENROLLMENT_FEE,
    PROGRAM_MONTHLY_FEE,
    PROGRAM_PACE,
    PROGRAM_PIE,
    PROGRAM_RELIGIOUS_ORIENTATION,
    PROGRAM_SCHOOL_DAY,
    PROGRAM_TRACK,
    REGION,
    SAFETY,
    SCHOOL_COMMUNE,
    WISH_RANK,
)
from sae_app.i18n import t
from sae_app.program_options import compact_program_label
from sae_app.session_state import invalidate_simulation_state, update_builder_state
from sae_app.text_utils import as_bool
from sae_app.wish_list import make_builder_wish_row, non_empty_wish_rows, normalize_builder_wishes


def _family_display_value(value) -> str:
    """Return a translated, non-empty value for a family-facing program card."""
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return t("No information")
    return t(text)


def _render_program_details(program_row: pd.Series) -> None:
    """Show the characteristics families most often need before adding a program."""
    detail_rows = [
        ("Program details", PROGRAM_DISPLAY_NAME),
        ("Commune", SCHOOL_COMMUNE),
        ("Region", REGION),
        ("Program type", PROGRAM_TRACK),
        ("School day", PROGRAM_SCHOOL_DAY),
        ("PIE", PROGRAM_PIE),
        ("PACE", PROGRAM_PACE),
        ("Enrollment fee", PROGRAM_ENROLLMENT_FEE),
        ("Monthly fee", PROGRAM_MONTHLY_FEE),
        ("Religious orientation", PROGRAM_RELIGIOUS_ORIENTATION),
    ]
    for label, column in detail_rows:
        st.markdown(f"**{t(label)}:** {_family_display_value(program_row.get(column, ''))}")


def _active_priority_labels(row: pd.Series) -> list[str]:
    labels = []
    if as_bool(row.get("priority_sibling", False)):
        labels.append(t("Sibling priority"))
    if as_bool(row.get("priority_student", False)):
        labels.append(t("Priority-student quota"))
    if as_bool(row.get("priority_parent_civil_servant", False)):
        labels.append(t("Civil-servant child priority"))
    if as_bool(row.get("priority_ex_student", False)):
        labels.append(t("Former-student priority"))
    return labels


def render_wish_list_builder(
    *,
    editor_state_key: str,
    editor_widget_key_base: str,
    program_options_for_editor: list[str],
    program_mapping: dict[str, pd.Series],
    use_equivalence_classes: bool,
    simulation_done_key: str | None = None,
    simulation_result_key: str | None = None,
) -> pd.DataFrame:
    """
    Friendlier wish-list input UI.

    Returns the same cleaned DataFrame that the simulation engine already expects.
    """
    current = normalize_builder_wishes(
        st.session_state[editor_state_key],
        use_equivalence_classes,
    )
    current_non_empty = non_empty_wish_rows(current)

    selected_programs = set(current_non_empty[PROGRAM].astype(str).str.strip())
    add_options = [
        p for p in program_options_for_editor
        if str(p).strip() and p in program_mapping and p not in selected_programs
    ]

    st.markdown(t("#### Search and add programs"))

    add_cols = st.columns([5, 1])
    with add_cols[0]:
        program_to_add = st.selectbox(
            t("Search for a program"),
            options=[""] + add_options,
            key=f"{editor_widget_key_base}_add_program",
            help=t("Start typing the school name, commune, or program details."),
        )

    with add_cols[1]:
        st.write("")
        st.write("")
        add_clicked = st.button(
            t("Add"),
            disabled=not bool(program_to_add),
            key=f"{editor_widget_key_base}_add_button",
        )

    if add_clicked and program_to_add:
        next_rank = len(current_non_empty) + 1

        if use_equivalence_classes:
            existing_groups = pd.to_numeric(
                current_non_empty.get(EQUIV_GROUP, pd.Series(dtype=float)),
                errors="coerce",
            ).dropna()
            next_group = int(existing_groups.max()) + 1 if not existing_groups.empty else next_rank
        else:
            next_group = next_rank

        new_row = make_builder_wish_row(program_to_add, next_rank, next_group)
        updated = pd.concat(
            [current_non_empty, pd.DataFrame([new_row])],
            ignore_index=True,
        )

        update_builder_state(
            updated,
            editor_state_key=editor_state_key,
            editor_widget_key_base=editor_widget_key_base,
            use_equivalence_classes=use_equivalence_classes,
            simulation_done_key=simulation_done_key,
            simulation_result_key=simulation_result_key,
        )

    if current_non_empty.empty:
        st.info(t("No program selected yet. Add the student's first wish above."))
        return current

    st.markdown(t("#### Current preference list"))
    st.caption(t("{n} program(s) selected", n=len(current_non_empty)))

    if use_equivalence_classes:
        st.caption(
            t("Set the same preference-group number for programs considered equivalent. Group 1 is preferred to group 2, etc.")
        )
        display_rows = current_non_empty.sort_values(
            [EQUIV_GROUP, WISH_RANK],
            kind="stable",
        ).reset_index(drop=True)
    else:
        st.caption(t("Use ↑ and ↓ to reorder the student's strict ranking."))
        display_rows = current_non_empty.reset_index(drop=True)

    edited_rows = []

    for i, row in display_rows.iterrows():
        program_label = str(row[PROGRAM]).strip()
        row_key = hashlib.md5(program_label.encode("utf-8")).hexdigest()[:10]

        with st.container(border=True):
            top_cols = st.columns([0.8, 5, 1.2])

            with top_cols[0]:
                if use_equivalence_classes:
                    group_value = st.number_input(
                        t("Group"),
                        min_value=1,
                        value=int(row[EQUIV_GROUP]),
                        step=1,
                        key=f"{editor_widget_key_base}_group_{row_key}",
                    )
                    wish_rank_value = i + 1
                else:
                    st.markdown(f"**#{i + 1}**")
                    wish_rank_value = i + 1
                    group_value = i + 1

            with top_cols[1]:
                st.markdown(f"**{compact_program_label(program_label)}**")

                if program_label in program_mapping:
                    program_row = program_mapping[program_label]
                    program_details = str(program_row.get(PROGRAM_DISPLAY_NAME, "")).strip()
                    commune = str(program_row.get(SCHOOL_COMMUNE, "")).strip()
                    region = str(program_row.get(REGION, "")).strip()

                    details = " · ".join(
                        part for part in [program_details, commune, region]
                        if part and part.lower() != "nan"
                    )
                    if details:
                        st.caption(details)

                    with st.popover(t("View program details")):
                        _render_program_details(program_row)

                active_priorities = _active_priority_labels(row)
                if active_priorities:
                    st.caption(
                        t(
                            "Declared priorities: {priorities}",
                            priorities=", ".join(active_priorities),
                        )
                    )
                else:
                    st.caption(t("No priority declared for this program"))

            with top_cols[2]:
                if st.button(
                    t("Remove"),
                    key=f"{editor_widget_key_base}_remove_{row_key}",
                    use_container_width=True,
                ):
                    updated = display_rows.drop(index=i).reset_index(drop=True)

                    update_builder_state(
                        updated,
                        editor_state_key=editor_state_key,
                        editor_widget_key_base=editor_widget_key_base,
                        use_equivalence_classes=use_equivalence_classes,
                        simulation_done_key=simulation_done_key,
                        simulation_result_key=simulation_result_key,
                    )

            if not use_equivalence_classes:
                move_cols = st.columns(2)
                with move_cols[0]:
                    if st.button(
                        t("Move up"),
                        disabled=i == 0,
                        key=f"{editor_widget_key_base}_up_{row_key}",
                        use_container_width=True,
                    ):
                        updated = display_rows.copy()
                        order = list(range(len(updated)))
                        order[i - 1], order[i] = order[i], order[i - 1]
                        updated = updated.iloc[order].reset_index(drop=True)

                        update_builder_state(
                            updated,
                            editor_state_key=editor_state_key,
                            editor_widget_key_base=editor_widget_key_base,
                            use_equivalence_classes=use_equivalence_classes,
                            simulation_done_key=simulation_done_key,
                            simulation_result_key=simulation_result_key,
                        )
                with move_cols[1]:
                    if st.button(
                        t("Move down"),
                        disabled=i == len(display_rows) - 1,
                        key=f"{editor_widget_key_base}_down_{row_key}",
                        use_container_width=True,
                    ):
                        updated = display_rows.copy()
                        order = list(range(len(updated)))
                        order[i], order[i + 1] = order[i + 1], order[i]
                        updated = updated.iloc[order].reset_index(drop=True)

                        update_builder_state(
                            updated,
                            editor_state_key=editor_state_key,
                            editor_widget_key_base=editor_widget_key_base,
                            use_equivalence_classes=use_equivalence_classes,
                            simulation_done_key=simulation_done_key,
                            simulation_result_key=simulation_result_key,
                        )

            with st.expander(
                t("Does the student have priority at this establishment?"),
                expanded=False,
            ):
                st.caption(
                    t(
                        "Mark only the situations that apply to this specific establishment. SAE recognizes the four priority criteria below."
                    )
                )
                priority_sibling = st.checkbox(
                    t("Has a sibling enrolled at the establishment"),
                    value=as_bool(row.get("priority_sibling", False)),
                    key=f"{editor_widget_key_base}_sib_{row_key}",
                )
                priority_student = st.checkbox(
                    t("Belongs to the 15% priority-student quota"),
                    value=as_bool(row.get("priority_student", False)),
                    key=f"{editor_widget_key_base}_student_{row_key}",
                )
                priority_parent = st.checkbox(
                    t("Is the child of an employee of the establishment"),
                    value=as_bool(row.get("priority_parent_civil_servant", False)),
                    key=f"{editor_widget_key_base}_parent_{row_key}",
                )
                priority_ex_student = st.checkbox(
                    t("Previously attended the establishment and was not expelled"),
                    value=as_bool(row.get("priority_ex_student", False)),
                    key=f"{editor_widget_key_base}_ex_{row_key}",
                )

                st.divider()
                st.caption(
                    t(
                        "Current enrollment is shown separately because it is not one of the four SAE priority criteria."
                    )
                )
                safety = st.checkbox(
                    t("The student is already enrolled in this establishment"),
                    value=as_bool(row.get(SAFETY, False)),
                    key=f"{editor_widget_key_base}_safety_{row_key}",
                )

            edited_rows.append({
                WISH_RANK: wish_rank_value,
                EQUIV_GROUP: group_value,
                PROGRAM: program_label,
                LOTTERY: 1,
                "priority_sibling": priority_sibling,
                "priority_student": priority_student,
                "priority_parent_civil_servant": priority_parent,
                "priority_ex_student": priority_ex_student,
                SAFETY: safety,
            })

    edited = normalize_builder_wishes(
        pd.DataFrame(edited_rows),
        use_equivalence_classes,
    )

    old = normalize_builder_wishes(
        st.session_state[editor_state_key],
        use_equivalence_classes,
    )

    compare_cols = [WISH_RANK, EQUIV_GROUP, PROGRAM] + PRIORITIES + [SAFETY]

    if not edited[compare_cols].astype(str).equals(old[compare_cols].astype(str)):
        st.session_state[editor_state_key] = edited

        invalidate_simulation_state(
            simulation_done_key=simulation_done_key,
            simulation_result_key=simulation_result_key,
        )

        st.rerun()

    return edited
