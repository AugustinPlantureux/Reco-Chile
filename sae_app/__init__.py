"""SAE admission-risk simulator.

This package splits what used to be a single monolithic Streamlit script into
focused modules:

- constants:          static configuration (columns, thresholds, file paths, dropdown options)
- i18n:                the ES/EN translation dictionary and the t() helper
- text_utils:          small, dependency-free text/number cleaning helpers
- data_loading:        reading and validating the CSV data files
- program_options:     ProgramRecord + building/filtering the program dropdown
- errors:              typed calculation errors translated only by the UI layer
- mtb_engine:          the SHA-256 lottery hash + hypergeometric availability model
                       (pure calculation, no Streamlit dependency)
- wish_list:           wish-list parsing, cleaning, and equivalence-class handling
- geo:                 coordinates, distance, and address geocoding
- recommendations:     the "similar programs" portfolio-risk recommendation engine
- session_state:       small Streamlit session-state helpers shared by the UI
- ui_common:           shared display/formatting helpers for Streamlit tables
- ui_simulation:       rendering the simulation results (summary, sensitivity tables)
- ui_wish_builder:     rendering the wish-list builder widget
- ui_recommendations:  rendering the "recommended similar programs" section

`app.py`, at the project root, only wires these modules together in the order
the Streamlit page is built.
"""
