"""SAE admission-risk simulation — Streamlit entry point.

This file only wires the sae_app modules together in the order the page is
built. All calculation, data loading, and widget-rendering logic lives in the
sae_app package — see sae_app/__init__.py for a map of what lives where.

Run with: streamlit run app.py
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from sae_app.constants import (
    CAPACITIES_PATH,
    CAPACITY,
    EQUIV_GROUP,
    HARD_UNMATCHED_THRESHOLD,
    HASH_PCT,
    IMPUTED,
    LOTTERY,
    MAX_EXACT_EQUIV_PERMUTATIONS,
    PACE_FILTER_OPTIONS,
    PAYMENT_FILTER_OPTIONS,
    PIE_FILTER_OPTIONS,
    POP,
    PROGRAM,
    TRUE_APP,
    REGION,
    RELIGIOUS_FILTER_OPTIONS,
    RURALITY_FILTER_OPTIONS,
    GENDER_FILTER_OPTIONS,
    SCHOOL_DAY_FILTER_OPTIONS,
    SOFT_UNMATCHED_THRESHOLD,
    SPECIALTY_FILTER_OPTIONS,
    TRACK_GENERAL,
    TRACK_SPECIALIZED,
    UNKNOWN_REGION,
    WISH_RANK,
)
from sae_app.data_loading import (
    available_regions,
    filters_are_active,
    load_calibration,
    program_matches_filters,
    required_cols,
    validate_core_numeric_columns,
    validate_cumulative_share_columns,
)
from sae_app.i18n import format_option_label, initialize_language_selector, t
from sae_app.mtb_engine import (
    attach_mtb_hashes,
    compute,
    compute_equivalence_order_from_precomputed,
    precompute_equivalence_availability,
)
from sae_app.program_options import build_options, compact_program_label, filter_program_options
from sae_app.session_state import clear_wish_editor_widget_state, invalidate_simulation_state
from sae_app.text_utils import as_bool
from sae_app.ui_common import format_display_table
from sae_app.ui_recommendations import render_similar_program_recommendations
from sae_app.ui_simulation import render_simulation_result
from sae_app.ui_wish_builder import render_wish_list_builder
from sae_app.wish_list import (
    clean_wish_rows,
    compact_order_label,
    count_equivalence_orders,
    empty_wishes,
    iter_equivalence_orders,
    non_empty_wish_rows,
    parse_wishes,
    predicted_outcome_final_chance,
    predicted_outcome_from_choices,
    prepare_ordered_wishes,
    uploaded_lottery_columns,
)

# ===========================================================================
# Page setup
# ===========================================================================

st.set_page_config(
    page_title="SAE simulation – unmatched risk",
    page_icon="🎓",
    layout="wide",
)
initialize_language_selector()
st.title(t("SAE admission-risk simulation"))
st.caption(
    t("MTB mode (admission 2026): SHA-256(RUN/IPE+RBD) percentile by school. Results are estimates based on last year's calibration data, not official admission guarantees.")
)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.caption(t("Capacities + 2024 calibration data are loaded from data/."))

    hard_threshold = HARD_UNMATCHED_THRESHOLD
    soft_threshold = SOFT_UNMATCHED_THRESHOLD
    if soft_threshold > hard_threshold:
        st.error(t("SOFT_UNMATCHED_THRESHOLD must be lower than or equal to HARD_UNMATCHED_THRESHOLD."))
        st.stop()
    st.caption(
        t("Unmatched thresholds are fixed in code: hard {hard:.1%}, soft {soft:.1%}.", hard=hard_threshold, soft=soft_threshold)
    )
    st.caption(
        t("Hard = Unmatched is shown first. Soft = Unmatched appears in the podium as a warning.")
    )

    national_student_id = st.text_input(
        t("Student RUN/IPE"),
        key="national_student_id_mtb",
        placeholder="12.345.678-9",
        help=t("Used to compute the SHA-256 percentile specific to each school. RUN format: 12.345.678-9. Dots are optional. For foreign students, enter the IPE."),
    )

# ── Built-in capacities/calibration data ─────────────────────────────
calib = load_calibration(CAPACITIES_PATH.read_bytes())
missing = [c for c in required_cols() if c not in calib.columns]
if missing:
    st.error(t("Missing columns: ") + ", ".join(missing[:20]))
    st.stop()

calibration_numeric_errors = validate_core_numeric_columns(calib)
if calibration_numeric_errors:
    st.error(t("Calibration numeric columns contain invalid values. Check the calibration CSV before running the app."))
    st.code("\n".join(calibration_numeric_errors[:10]))
    st.stop()

cumulative_share_errors = validate_cumulative_share_columns(calib)
if cumulative_share_errors:
    st.error(t("Calibration cumulative-share columns are inconsistent or incomplete. Check the calibration CSV before running the app."))
    st.code("\n".join(cumulative_share_errors[:10]))
    st.stop()

invalid_population = pd.to_numeric(calib[POP], errors="coerce").isna() | (pd.to_numeric(calib[POP], errors="coerce") <= 0)
if invalid_population.any():
    st.error(
        t("Invalid {pop}: {n} program(s) have missing or non-positive lottery population.", pop=POP, n=int(invalid_population.sum()))
    )
    st.stop()

program_options, program_mapping = build_options(calib)

# ── Section 1: pathway ───────────────────────────────────────────────
st.subheader(t("1. Start with the student's preferences"))

list_status = st.radio(
    t("Is the student's wish list already established?"),
    [
        "Yes — I already have the list",
        "No — help me build it with filters",
    ],
    horizontal=True,
    format_func=format_option_label,
    key="list_status_mtb",
)
needs_builder = list_status.startswith("No")

ranking_mode = st.radio(
    t("How should preferences be entered?"),
    [
        "Strict ranking",
        "Equivalence classes",
    ],
    horizontal=True,
    format_func=format_option_label,
    help=t("Strict ranking means every program has a precise rank. Equivalence classes allow several programs to share the same preference group when the family sees them as tied."),
    key="ranking_mode_mtb",
)
use_equivalence_classes = ranking_mode == "Equivalence classes"

if use_equivalence_classes:
    st.info(
        t("Use the same preference-group number for programs the student considers tied. Lower group numbers are preferred. The app will test every possible order inside each tied group, so families can see whether the exact internal order changes the predicted outcome.")
    )
else:
    st.info(
        t("Enter programs in strict order. The first program is the highest-ranked choice, and the final chance of each lower option depends on not getting the options above it.")
    )

wish_file = st.file_uploader(
    t("Optional: import a wish-list CSV to pre-fill the list"),
    type=["csv"],
)

uploaded_wish_hash = None
uploaded_wish_rows = None
uploaded_ignored_lottery_cols: list[str] = []
base_rows = empty_wishes()

if wish_file is not None:
    try:
        wish_file_bytes = wish_file.getvalue()
        uploaded_wish_hash = hashlib.md5(wish_file_bytes).hexdigest()[:8]
        uploaded_wish_rows = parse_wishes(wish_file_bytes, program_mapping)
        uploaded_ignored_lottery_cols = uploaded_lottery_columns(wish_file_bytes)
        base_rows = uploaded_wish_rows
    except Exception as exc:
        st.error(t("Could not import the CSV: {error}", error=exc))
        uploaded_wish_hash = None
        uploaded_wish_rows = None
        base_rows = empty_wishes()

# ── Optional program-building filters ─────────────────────────────────
empty_filters = {
    "tracks": [],
    "specialty_sectors": [],
    "genders": [],
    "school_days": [],
    "rurality": [],
    "pie": [],
    "pace": [],
    "enrollment_fee": [],
    "monthly_fee": [],
    "religious_orientation": [],
}
program_filters = empty_filters.copy()
selected_program_region = "All regions"

if needs_builder:
    st.subheader(t("2. Find programs"))
    with st.expander(t("Program search filters"), expanded=True):
        st.caption(t("Leave every filter empty to include all programs."))

        region_options = ["All regions"] + available_regions(calib)
        selected_program_region = st.selectbox(
            t("Program region"),
            region_options,
            index=0,
            format_func=format_option_label,
            help=t("Choose a region to make the program list shorter. Already selected programs from other regions are kept in the list."),
            key="program_region_filter_mtb",
        )

        c1, c2 = st.columns(2)
        with c1:
            filter_general = st.checkbox(t("General academic programs"), value=False, key="filter_general_mtb")
        with c2:
            filter_specialized = st.checkbox(t("Specialized / technical programs"), value=False, key="filter_specialized_mtb")

        selected_specialty_sectors = []
        if filter_specialized:
            selected_specialty_sectors = st.multiselect(
                t("Specialized area"),
                SPECIALTY_FILTER_OPTIONS,
                default=[],
                format_func=format_option_label,
                help=t("Leave empty to include all specialized areas."),
                key="filter_specialty_sectors_mtb",
            )

        c1, c2 = st.columns(2)
        with c1:
            selected_genders = st.multiselect(
                t("Gender composition"),
                GENDER_FILTER_OPTIONS,
                default=[],
                format_func=format_option_label,
                help=t("Leave empty to include mixed, boys-only, and girls-only programs."),
                key="filter_genders_mtb",
            )
            selected_rurality = st.multiselect(
                t("Rurality"),
                RURALITY_FILTER_OPTIONS,
                default=[],
                format_func=format_option_label,
                help=t("Leave empty to include both urban and rural schools."),
                key="filter_rurality_mtb",
            )
            selected_pie = st.multiselect(
                t("PIE integration program"),
                PIE_FILTER_OPTIONS,
                default=[],
                format_func=format_option_label,
                help=t("Leave empty to include schools with and without PIE."),
                key="filter_pie_mtb",
            )
            selected_enrollment_fee = st.multiselect(
                t("Enrollment fee"),
                PAYMENT_FILTER_OPTIONS,
                default=[],
                format_func=format_option_label,
                help=t("Leave empty to include every enrollment-fee category."),
                key="filter_enrollment_fee_mtb",
            )
        with c2:
            selected_school_days = st.multiselect(
                t("School day"),
                SCHOOL_DAY_FILTER_OPTIONS,
                default=[],
                format_func=format_option_label,
                help=t("Leave empty to include full-day, morning, and afternoon programs."),
                key="filter_school_days_mtb",
            )
            selected_pace = st.multiselect(
                t("PACE program"),
                PACE_FILTER_OPTIONS,
                default=[],
                format_func=format_option_label,
                help=t("Leave empty to include schools with and without PACE."),
                key="filter_pace_mtb",
            )
            selected_monthly_fee = st.multiselect(
                t("Monthly fee"),
                PAYMENT_FILTER_OPTIONS,
                default=[],
                format_func=format_option_label,
                help=t("Leave empty to include every monthly-fee category."),
                key="filter_monthly_fee_mtb",
            )
            selected_religious_orientation = st.multiselect(
                t("Religious orientation"),
                RELIGIOUS_FILTER_OPTIONS,
                default=[],
                format_func=format_option_label,
                help=t("Leave empty to include every orientation."),
                key="filter_religious_orientation_mtb",
            )

        program_filters = {
            "tracks": ([TRACK_GENERAL] if filter_general else []) + ([TRACK_SPECIALIZED] if filter_specialized else []),
            "specialty_sectors": selected_specialty_sectors,
            "genders": selected_genders,
            "school_days": selected_school_days,
            "rurality": selected_rurality,
            "pie": selected_pie,
            "pace": selected_pace,
            "enrollment_fee": selected_enrollment_fee,
            "monthly_fee": selected_monthly_fee,
            "religious_orientation": selected_religious_orientation,
        }
else:
    st.subheader(t("2. Enter the list"))
    st.caption(t("Use the builder below to enter the existing wish list directly."))

# ── Wish-list builder ─────────────────────────────────────────────────
# Keep the wish-list state key stable across UI mode changes. Switching between
# direct entry / guided filters or strict ranking / equivalence classes should
# never silently reset the family's current list.
editor_state_key = "wish_rows_mtb"
editor_import_hash_key = "wish_rows_import_hash_mtb"
editor_mode_key = "wish_rows_ranking_mode_mtb"
editor_widget_key_base = "wishes_builder_mtb"
simulation_done_key = "simulation_done_mtb"
simulation_result_key = "simulation_result_mtb"
simulation_student_id_key = "simulation_student_id_mtb"

# Keep simulation outputs tied to the RUN/IPE that produced them. Without this,
# a language change or a cleared RUN/IPE can leave an old simulation visible while
# candidate-level portfolio-risk estimates fail silently.
if (
    st.session_state.get(simulation_done_key, False)
    and st.session_state.get(simulation_student_id_key, "").strip() != str(national_student_id).strip()
):
    invalidate_simulation_state(
        simulation_done_key=simulation_done_key,
        simulation_result_key=simulation_result_key,
        simulation_student_id_key=simulation_student_id_key,
    )

if editor_state_key not in st.session_state:
    st.session_state[editor_state_key] = clean_wish_rows(base_rows)
    if uploaded_wish_hash:
        st.session_state[editor_import_hash_key] = uploaded_wish_hash

current_ranking_mode_key = "equiv" if use_equivalence_classes else "strict"
if st.session_state.get(editor_mode_key) != current_ranking_mode_key:
    # The list is preserved, but any previously displayed simulation is no
    # longer guaranteed to match the current interpretation of the preferences.
    st.session_state[editor_mode_key] = current_ranking_mode_key
    invalidate_simulation_state(
        simulation_done_key=simulation_done_key,
        simulation_result_key=simulation_result_key,
        simulation_student_id_key=simulation_student_id_key,
    )

if uploaded_ignored_lottery_cols:
    st.info(
        t(
            "The uploaded CSV contains lottery-number column(s) that are ignored in MTB mode: {cols}. The app computes the lottery rank from RUN/IPE + RBD.",
            cols=", ".join(uploaded_ignored_lottery_cols),
        )
    )

if uploaded_wish_rows is not None and uploaded_wish_hash:
    already_imported = st.session_state.get(editor_import_hash_key) == uploaded_wish_hash
    current_non_empty_wishes = non_empty_wish_rows(st.session_state[editor_state_key])
    uploaded_non_empty_wishes = non_empty_wish_rows(uploaded_wish_rows)

    if not already_imported and current_non_empty_wishes.empty:
        st.session_state[editor_state_key] = clean_wish_rows(uploaded_wish_rows)
        st.session_state[editor_import_hash_key] = uploaded_wish_hash
        invalidate_simulation_state(
            simulation_done_key=simulation_done_key,
            simulation_result_key=simulation_result_key,
            simulation_student_id_key=simulation_student_id_key,
        )
        clear_wish_editor_widget_state(editor_widget_key_base)
        st.success(t("Wish list imported with {n} wish(es).", n=len(uploaded_non_empty_wishes)))
    elif not already_imported:
        st.warning(
            t("A CSV has been uploaded, but the current wish list was not replaced automatically. Use the button below if you want to replace the current list with the uploaded file.")
        )
        if st.button(t("Replace current wish list with uploaded CSV")):
            st.session_state[editor_state_key] = clean_wish_rows(uploaded_wish_rows)
            st.session_state[editor_import_hash_key] = uploaded_wish_hash
            invalidate_simulation_state(
                simulation_done_key=simulation_done_key,
                simulation_result_key=simulation_result_key,
                simulation_student_id_key=simulation_student_id_key,
            )
            clear_wish_editor_widget_state(editor_widget_key_base)
            st.rerun()
    elif uploaded_non_empty_wishes.empty:
        st.info(t("The uploaded CSV does not contain any valid wish."))

editor_rows = st.session_state[editor_state_key].copy()
if PROGRAM in editor_rows.columns:
    program_values = editor_rows[PROGRAM].fillna("").astype(str).str.strip()
    unavailable_programs = sorted({p for p in program_values if p and p not in program_mapping})

    if unavailable_programs:
        shown_programs = [compact_program_label(p) for p in unavailable_programs[:5]]
        if len(unavailable_programs) > len(shown_programs):
            shown_programs.append("...")

        st.warning(
            t(
                "Some programs in the current wish list are no longer available in the loaded data and were removed: {programs}",
                programs=", ".join(shown_programs),
            )
        )

        editor_rows[PROGRAM] = program_values.map(
            lambda x: x if x in program_mapping or x == "" else ""
        )
        st.session_state[editor_state_key] = clean_wish_rows(editor_rows)
        invalidate_simulation_state(
            simulation_done_key=simulation_done_key,
            simulation_result_key=simulation_result_key,
            simulation_student_id_key=simulation_student_id_key,
        )
        editor_rows = st.session_state[editor_state_key].copy()

current_program_values = (
    editor_rows.get(PROGRAM, pd.Series(dtype=str))
    .dropna()
    .astype(str)
    .str.strip()
    .tolist()
)
program_options_for_editor = filter_program_options(
    program_mapping,
    selected_program_region,
    active_filters=program_filters,
    current_values=current_program_values,
)

if needs_builder and (selected_program_region != "All regions" or filters_are_active(program_filters)):
    preserved = [
        p for p in current_program_values
        if p in program_mapping
        and not (
            (selected_program_region == "All regions" or str(program_mapping[p].get(REGION, UNKNOWN_REGION)).strip() == selected_program_region)
            and program_matches_filters(program_mapping[p], program_filters)
        )
    ]
    matching_count = max(len(program_options_for_editor) - len(preserved), 0)
    extra_note = (
        t(" Existing selected program(s) outside the current filters are also kept available: {n}.", n=len(preserved))
        if preserved else ""
    )
    region_text = selected_program_region if selected_program_region != "All regions" else t("all regions")
    st.caption(
        t("Showing {n} matching program option(s) for {region}.", n=matching_count, region=region_text)
        + extra_note
    )

edited = render_wish_list_builder(
    editor_state_key=editor_state_key,
    editor_widget_key_base=editor_widget_key_base,
    program_options_for_editor=program_options_for_editor,
    program_mapping=program_mapping,
    use_equivalence_classes=use_equivalence_classes,
    simulation_done_key=simulation_done_key,
    simulation_result_key=simulation_result_key,
    simulation_student_id_key=simulation_student_id_key,
)

selected = [p for p in edited[PROGRAM].dropna().astype(str).str.strip() if p]
imputed = [
    p for p in selected
    if p in program_mapping and as_bool(program_mapping[p].get(IMPUTED, False))
]
if imputed:
    st.warning(
        t("Less reliable estimate: at least one selected program uses mean-imputed 2024 calibration values.")
    )

# ── MTB percentile preview ────────────────────────────────────────────
reference_order = prepare_ordered_wishes(edited, use_equivalence_classes)
if not reference_order.empty and national_student_id.strip():
    try:
        preview_w = attach_mtb_hashes(reference_order, program_mapping, national_student_id)
        preview_cols = [WISH_RANK, PROGRAM, LOTTERY, HASH_PCT]
        if use_equivalence_classes:
            preview_cols.insert(1, EQUIV_GROUP)
        preview = preview_w[preview_cols].copy()
        preview[HASH_PCT] = (
            pd.to_numeric(preview[HASH_PCT], errors="coerce")
            .map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        )
        preview = preview.rename(columns={
            WISH_RANK: "Reference rank",
            EQUIV_GROUP: "Preference group",
            PROGRAM: "Program",
            LOTTERY: "Calculated MTB lottery rank",
            HASH_PCT: "MTB hash percentile",
        })
        with st.expander(t("Calculated MTB percentiles (RUN + RBD)"), expanded=False):
            st.dataframe(format_display_table(preview), width="stretch", hide_index=True)
    except Exception as exc:
        st.warning(t("MTB preview unavailable: {error}", error=exc))

# ── Section 3: simulation ─────────────────────────────────────────────
st.subheader(t("3. Run the simulation"))

if use_equivalence_classes:
    total_orders = count_equivalence_orders(edited)
    if total_orders:
        st.caption(
            t("The current equivalence classes generate {n:,} compatible strict order(s). The app will test them to see whether tied programs lead to the same or different predicted outcomes.", n=total_orders)
        )

calculated_now = False

if st.button(t("Calculate unmatched risk"), type="primary"):
    if not national_student_id.strip():
        st.error(t("Please enter the student's RUN/IPE before running the simulation."))
        st.stop()

    try:
        reference_order = prepare_ordered_wishes(edited, use_equivalence_classes)
        if reference_order.empty:
            st.error(t("Add at least one valid program before running the simulation."))
            st.stop()

        if use_equivalence_classes:
            total_orders = count_equivalence_orders(reference_order)
            if total_orders > MAX_EXACT_EQUIV_PERMUTATIONS:
                st.error(
                    t(
                        "The equivalence classes generate {n:,} strict orders. This is above the exact-evaluation limit of {limit:,}. Split large equivalence groups into smaller groups, then run the simulation again.",
                        n=total_orders,
                        limit=MAX_EXACT_EQUIV_PERMUTATIONS,
                    )
                )
                st.stop()

            variants = []
            reference_choices = None
            availability_lookup = precompute_equivalence_availability(
                reference_order,
                program_mapping,
                national_student_id,
            )

            for idx, strict_order in enumerate(iter_equivalence_orders(reference_order), start=1):
                choices = compute_equivalence_order_from_precomputed(strict_order, availability_lookup)
                outcome, p_unmatched, at_risk = predicted_outcome_from_choices(choices, hard_threshold)

                if idx == 1:
                    reference_choices = choices

                variants.append({
                    "Strict order #": idx,
                    "Predicted outcome": outcome,
                    "Predicted outcome final chance": predicted_outcome_final_chance(choices, outcome),
                    "Unmatched risk": p_unmatched,
                    "Flagged at risk": at_risk,
                    "Strict order": compact_order_label(strict_order),
                })

            variants_df = pd.DataFrame(variants)
            distinct_outcomes = sorted(variants_df["Predicted outcome"].unique().tolist())
            simulation_result = {
                "mode": "equivalence",
                "hard_threshold": hard_threshold,
                "soft_threshold": soft_threshold,
                "reference_choices": reference_choices,
                "variants_df": variants_df,
                "distinct_outcomes": distinct_outcomes,
            }

        else:
            strict_order = prepare_ordered_wishes(edited, use_equivalence_classes=False)
            wishes_for_compute = attach_mtb_hashes(strict_order, program_mapping, national_student_id)
            choices = compute(wishes_for_compute, program_mapping)
            simulation_result = {
                "mode": "strict",
                "hard_threshold": hard_threshold,
                "soft_threshold": soft_threshold,
                "choices": choices,
            }

        st.session_state[simulation_result_key] = simulation_result
        st.session_state[simulation_student_id_key] = str(national_student_id).strip()
        st.session_state[simulation_done_key] = True
        calculated_now = True
        render_simulation_result(simulation_result)

    except ValueError as exc:
        st.error(str(exc))

    except Exception as exc:
        st.error(t("Unexpected error during the simulation."))
        st.exception(exc)

if st.session_state.get(simulation_done_key, False):
    if not calculated_now:
        render_simulation_result(st.session_state.get(simulation_result_key, {}))

    render_similar_program_recommendations(
        edited,
        program_mapping,
        student_id=national_student_id,
        editor_state_key=editor_state_key,
        editor_widget_key_base=editor_widget_key_base,
        use_equivalence_classes=use_equivalence_classes,
        simulation_done_key=simulation_done_key,
        simulation_result_key=simulation_result_key,
        simulation_student_id_key=simulation_student_id_key,
    )
else:
    st.subheader(t("4. Recommended similar programs"))
    st.info(t("Run the simulation first to unlock similar-program recommendations."))
