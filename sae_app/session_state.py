"""Small Streamlit session-state helpers shared by the UI modules.

Centralizing "invalidate the cached simulation" and "clear the wish-list
widget keys" in one place means every UI entry point that changes the wish
list invalidates state the same way.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from sae_app.wish_list import normalize_builder_wishes


def invalidate_simulation_state(
    *,
    simulation_done_key: str | None = None,
    simulation_result_key: str | None = None,
    simulation_student_id_key: str | None = None,
) -> None:
    """Invalidate cached simulation output after any input that affects results changes."""
    if simulation_done_key:
        st.session_state[simulation_done_key] = False
    if simulation_result_key:
        st.session_state.pop(simulation_result_key, None)
    if simulation_student_id_key:
        st.session_state.pop(simulation_student_id_key, None)


def clear_wish_editor_widget_state(editor_widget_key_base: str) -> None:
    """Clear Streamlit wish-list widget keys so added recommendations appear immediately."""
    for key in list(st.session_state.keys()):
        if str(key).startswith(editor_widget_key_base):
            del st.session_state[key]


def update_builder_state(
    updated: pd.DataFrame,
    *,
    editor_state_key: str,
    editor_widget_key_base: str,
    use_equivalence_classes: bool,
    simulation_done_key: str | None = None,
    simulation_result_key: str | None = None,
    simulation_student_id_key: str | None = None,
) -> None:
    """Save builder state and invalidate previous simulation output."""
    st.session_state[editor_state_key] = normalize_builder_wishes(
        updated,
        use_equivalence_classes,
    )

    invalidate_simulation_state(
        simulation_done_key=simulation_done_key,
        simulation_result_key=simulation_result_key,
        simulation_student_id_key=simulation_student_id_key,
    )

    clear_wish_editor_widget_state(editor_widget_key_base)
    st.rerun()
