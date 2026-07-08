# SAE admission-risk simulator

A Streamlit application to simulate admission risk in the Chilean SAE school-admission system.

The tool lets families build a preference list, estimate admission probabilities for selected programs, and identify similar programs that may be worth considering.

## Features

- Build a ranked preference list of school programs.
- Support strict rankings and equivalence classes.
- Estimate admission probabilities using the SAE lottery logic.
- Display risk indicators for each selected program.
- Recommend similar programs based on school characteristics.
- Filter programs by region, commune, education level, gender composition, school day, and other available criteria.
- Run locally without sending student identifiers or preference lists to an external server.

## Run the application

Install the required packages:

```bash
pip install -r requirements.txt
```

Launch the app:

```bash
streamlit run app.py
```

The application should then open in your browser.

## Required data files

The app expects the following files in a `data/` folder located next to `app.py`:

```text
data/
    capacities_2025_wta_with_2024_calibration.csv
    programmes_chili_criteres_recommandation.csv
    rbd_region_map.csv
    program_filters.csv
```

## Project structure

```text
app.py                          # Streamlit entry point
requirements.txt
README.md

data/                           # Input data files

sae_app/
    __init__.py
    constants.py                # Constants, column names, thresholds, file paths
    i18n.py                     # Spanish/English translation dictionary
    text_utils.py               # Text and number cleaning utilities
    data_loading.py             # CSV loading and validation
    program_options.py          # Program records and dropdown-menu construction
    mtb_engine.py               # Lottery number, priority, and admission-risk calculations
    wish_list.py                # Preference-list parsing and equivalence-class handling
    geo.py                      # Geographic coordinates, distances, and address geocoding
    recommendations.py          # Similar-program recommendation engine
    session_state.py            # Streamlit session-state helpers
    ui_common.py                # Shared UI formatting helpers
    ui_simulation.py            # Simulation-result display
    ui_wish_builder.py          # Preference-list builder UI
    ui_recommendations.py       # Similar-program recommendation UI
```

## How the app works

The user enters a student identifier and builds a list of preferred school programs.

For each selected program, the app uses historical calibration data, program capacities, priority groups, and the student lottery number to estimate the probability of admission.

The app then displays:

- the estimated probability of assignment to each selected program;
- risk indicators for the preference list;
- sensitivity information when relevant;
- recommended similar programs based on the selected preferences.

## Local use and privacy

The application is designed to run locally.

When used locally, student identifiers and preference lists remain on the user’s machine. The app uses the identifier only to compute the lottery-based admission-risk estimates during the local session.

## Dependencies

The main dependencies are listed in `requirements.txt`:

```text
streamlit
pandas
numpy
scipy
```

Install them with:

```bash
pip install -r requirements.txt
```

## Notes

This tool is intended as an admission-risk simulator and decision-support interface. It does not replace official SAE results or guarantee admission outcomes.

Admission probabilities are estimates based on the available data and assumptions encoded in the model.