# Reco Chile — SAE admission-risk simulator

Reco Chile is a bilingual Streamlit application that helps families explore school choices in Chile's *Sistema de Admisión Escolar* (SAE).

The app estimates the probability of assignment to each program in a student's wish list, measures the risk of remaining unmatched, tests whether ties between preferences are consequential, and recommends additional programs that can improve the portfolio while remaining similar to the family's revealed preferences.

> [!IMPORTANT]
> This is a research and decision-support tool. It is not an official SAE service, does not reproduce every operational detail of the centralized assignment process, and cannot guarantee an admission outcome.

## Main features

- Build and reorder a wish list directly in the interface.
- Search programs by region and school characteristics.
- Use either a strict ranking or preference equivalence classes.
- Compute a deterministic MTB lottery percentile for each school from `SHA-256(RUN/IPE + RBD)`.
- Account for sibling, priority-student, civil-servant, former-student, and already-enrolled priority flags.
- Estimate each program's availability and final assignment probability.
- Display the overall unmatched risk and the most likely outcomes.
- Test every strict ordering compatible with the selected equivalence classes, up to a configurable limit.
- Recommend similar programs using revealed preferences, geographic proximity, competition, estimated admission safety, and diversity.
- Geocode an optional Chilean home address and restrict suggestions to a realistic radius.
- Switch between Spanish and English from the interface.
- Use progressive, family-facing explanations while keeping MTB ranks, calibration details, and recommendation methodology available on demand.

## Application workflow

1. Enter the student's RUN or IPE and indicate whether the wish list already exists.
2. Add programs in the family's genuine order of preference. An optional planning toggle can compare undecided internal orders.
3. Mark every applicable priority for each establishment and analyze the list.
4. Review the unmatched-risk estimate, the outcomes ordered by probability, and the short family-facing wish table.
5. Open the optional detail panels to inspect MTB ranks, calibration inputs, assumptions, or equivalence-order sensitivity.
6. Inspect suggested backup programs, compare the projected risk after appending each one, and add only acceptable options.
7. Verify priorities for newly added programs and rerun the analysis.

The guided program search can filter by:

- region;
- general or technical-vocational track;
- technical-vocational specialty;
- gender composition;
- school day;
- urban or rural location;
- PIE and PACE participation;
- enrollment and monthly fees;
- religious orientation.

## How the risk model works

### 1. MTB lottery rank

For every selected program, the app normalizes the student identifier, concatenates it with the school's RBD, and computes:

```text
SHA-256(normalized RUN/IPE + normalized RBD)
```

The hexadecimal digest is mapped to a percentile. Because the official priority direction is larger-is-better while the model uses zero-as-best, the percentile is reversed and converted into an integer rank within the program-level reference population.

The same student can therefore receive a different deterministic rank at different schools, while repeated simulations with the same RUN/IPE and RBD remain reproducible.

### 2. Priority-adjusted rank

The active priority tier is resolved in this order:

1. sibling;
2. priority student;
3. parent or guardian employed as a civil servant at the school;
4. former student;
5. no priority.

The current implementation retains the priority-student tier only when the calculated lottery rank falls within the program's recorded `priority_student_seats` allocation; otherwise it falls back to the next applicable tier.

The raw percentile is then mapped into the relevant part of the calibrated 2024 priority distribution using the tier's share and cumulative share.

### 3. Program availability

For a program `s`, let:

- `N_s` be its 2024 lottery reference population;
- `T_s` be the number of true applicants in the calibration data;
- `C_s` be its admission capacity;
- `r_e` be the student's priority-adjusted effective rank.

The number of competing true applicants appearing before the student is modeled as:

```text
X ~ Hypergeometric(N_s - 1, T_s - 1, r_e - 1)
```

The probability that the program is available if the student reaches that wish is:

```text
P(program available) = P(X <= C_s - 1)
```

An option marked as already enrolled is treated as certain to be available. A program with no capacity is treated as unavailable.

### 4. Final assignment probabilities

For a strict list with wish-level availability probabilities `a_1, ..., a_k`, the final probability of assignment to wish `i` is:

```text
P(assigned to i) = a_i × product(1 - a_j) for every j < i
```

The estimated unmatched risk is:

```text
P(unmatched) = product(1 - a_i) for i = 1, ..., k
```

The interface distinguishes between:

- **Chance if considered:** availability conditional on reaching that wish.
- **Final chance of assignment:** availability after accounting for every higher-ranked wish.

The current attention thresholds are defined in `sae_app/constants.py`:

- `2.7%` or above: high attention;
- `0.4%` to below `2.7%`: moderate attention;
- below `0.4%`: low attention.

These are presentation thresholds, not official SAE cutoffs. They control only the alert message. Estimated outcomes are always ordered by their actual modeled probability, so the alert never changes the likelihood ranking.

## Equivalence classes

Equivalence-class mode allows several programs to share the same preference group. Lower group numbers remain preferred, but programs inside the same group are treated as tied.

The app enumerates every compatible strict order inside the tied groups. If group sizes are `m_1, ..., m_g`, the number of variants is:

```text
m_1! × m_2! × ... × m_g!
```

Availability is computed once per program and reused across permutations. The app then reports whether internal ordering changes:

- the most likely assigned school;
- the final probability attached to that predicted school;
- the distribution of outcomes across compatible strict orders.

Because the same set of programs is used in every variant, the overall unmatched risk is invariant to internal ordering under the current model. Exact enumeration is capped at `10,000` compatible orders by default.

## Recommendation engine

Recommendations are treated as a portfolio-risk problem rather than a simple nearest-school lookup.

The engine first infers a revealed-preference profile from the current wish list. Higher-ranked wishes receive more weight, using `1 / sqrt(rank)`; in equivalence-class mode, the preference-group number is used instead of the reference row order.

Candidates are evaluated on four components:

| Component | What it captures |
| --- | --- |
| Preference similarity | Track, specialty, gender composition, school day, rurality, PIE, PACE, fees, and religious orientation |
| Proximity | Straight-line distance from the geocoded home address or the weighted centroid of the current wish list |
| Accessibility | Historical applicants per seat |
| Portfolio-risk improvement | The candidate's estimated availability using the student's actual MTB hash for that school |

Similarity weights adapt to the dominance, coverage, and reliability of each criterion in the current list. Candidates below the configured similarity floor are excluded when the list provides a usable preference signal.

For a candidate appended after the current list:

```text
estimated final chance if appended
    = current unmatched risk × candidate availability
```

This value is also the marginal reduction in unmatched risk under the model. Newly recommended programs are initially evaluated without special priority flags; the family should add the program, mark any real priority, and rerun the full simulation.

The final selection uses Maximal Marginal Relevance to reduce near-duplicate recommendations. Default weights, distance limits, similarity thresholds, and diversity settings are centralized in `sae_app/recommendations.py`.

## Geographic data

Coordinates are resolved through the following cascade:

1. program or school coordinates from the program metadata;
2. commune coordinates from the optional `data/commune_coordinates.csv` file;
3. an approximate regional centroid.

Distances use the haversine formula and therefore represent straight-line distance, not road distance or travel time.

If the user explicitly submits a home address, the app queries OpenStreetMap's Nominatim service, ranks returned locations for address quality, and warns when the result is only street-, city-, or region-level. With a valid home location, recommendations are limited to `100 km` by default.

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/AugustinPlantureux/Reco-Chile.git
cd Reco-Chile
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the application:

```bash
python3 -m streamlit run app.py
```

The main dependencies are Streamlit, pandas, NumPy, and SciPy.

## Data files

The application expects a `data/` directory next to `app.py`.

### Required files

| File | Purpose |
| --- | --- |
| `capacities_2025_wta_with_2024_calibration.csv` | Capacities, true applicants, lottery reference populations, priority shares, and calibration fields used by the risk model |
| `programmes_chili_criteres_recommandation.csv` | School and program names, communes, recommendation criteria, and program/school coordinates when available |
| `rbd_region_map.csv` | RBD-to-region mapping |
| `program_filters.csv` | Program track, specialty, gender composition, and school-day metadata used by the search filters |

### Optional file

| File | Expected columns | Purpose |
| --- | --- | --- |
| `commune_coordinates.csv` | `commune`, optional `region`, latitude (`latitude`, `lat`, or `latitud`), longitude (`longitude`, `lon`, `lng`, or `longitud`) | Improves distance estimates when program-level coordinates are unavailable |

At startup, the app checks required columns, core numeric fields, positive lottery populations, and the internal consistency of cumulative priority shares. Programs with mean-imputed 2024 calibration values remain usable but are flagged as less reliable.

## Project structure

```text
Reco-Chile/
├── app.py                         # Streamlit entry point and page orchestration
├── requirements.txt
├── README.md
├── data/                          # Calibration and program metadata
└── sae_app/
    ├── __init__.py                # Package overview
    ├── constants.py               # Columns, thresholds, paths, and configuration
    ├── data_loading.py            # CSV loading, joins, translation, and validation
    ├── geo.py                     # Coordinates, distance, and Nominatim geocoding
    ├── i18n.py                    # Spanish/English translations
    ├── mtb_engine.py              # SHA-256 MTB and hypergeometric risk model
    ├── program_options.py         # Typed program records and dropdown options
    ├── recommendations.py         # Similarity and portfolio-risk recommendations
    ├── session_state.py           # Shared Streamlit state helpers
    ├── text_utils.py              # Text, number, code, and coordinate cleaning
    ├── ui_common.py               # Shared table formatting
    ├── ui_recommendations.py      # Recommendation interface
    ├── ui_simulation.py           # Simulation results and sensitivity display
    ├── ui_wish_builder.py         # Interactive wish-list editor
    └── wish_list.py               # Wish-list state and equivalence-class handling
```

`app.py` intentionally contains only application orchestration. Calculation logic lives in the focused modules under `sae_app/`, with the core MTB engine kept independent from Streamlit.

## Privacy and external services

When the app is run locally:

- RUN/IPE normalization, hashing, wish-list processing, and risk calculations occur on the local machine;
- the student identifier and wish list are not intentionally sent to an external API;
- a home address is sent to OpenStreetMap's Nominatim service only after the user clicks the geocoding button.

If the app is deployed on a remote server, inputs are necessarily processed by that server instance. Deployment operators are responsible for access control, logging, retention, and compliance with applicable privacy requirements. Multi-worker deployments should also use shared Nominatim throttling or a dedicated geocoding service.

## Model limitations

- Results are estimates based on historical calibration data and encoded assumptions, not official admission probabilities.
- Changes in demand, capacities, priorities, or SAE rules can make past calibration less predictive.
- The current priority-student treatment and warning thresholds are explicit modeling choices.
- The final-list calculation combines wish-level availability estimates according to the model implemented in `mtb_engine.py`.
- Imputed calibration values are less reliable than directly observed values.
- Geographic distances are approximate and do not represent routes, travel times, mountain crossings, or public-transport accessibility.
- A geocoder can return an approximate or incorrect location; users should read the precision warning shown by the app.
- Recommendations optimize the encoded score and should be treated as options to investigate, not automatic choices.

For research, auditing, or policy use, review the current data sources, calibration assumptions, and model code before interpreting the outputs.
