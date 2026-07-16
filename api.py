"""FastAPI entry point.

A thin HTTP adapter around the same Streamlit-free engine used by app.py
This module only translates HTTP requests into calls against that engine
and formats the results as JSON.

Note /simulate always runs the equivalence-class pipeline. A wish without an
explicit equivalence_group is its own singleton group (equal to its position),
which is mathematically identical to strict ranking, so this covers both
modes without a separate code path.

Run with: uvicorn api:app --reload

"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sae_app.constants import (
    CAPACITIES_PATH,
    CAPACITY,
    EQUIV_GROUP,
    HARD_UNMATCHED_THRESHOLD,
    LOTTERY,
    MAX_EXACT_EQUIV_PERMUTATIONS, # this cap exists to avoid combinatorial explosion of permutations
    PRIORITIES,
    PROGRAM,
    PROGRAM_DISPLAY_NAME,
    PROGRAM_TRACK,
    REGION,
    SAFETY,
    SCHOOL_COMMUNE,
    SCHOOL_NAME,
    TRUE_APP,
    WISH_RANK,
)
from sae_app.data_loading import available_regions, load_calibration
from sae_app.errors import MtbEngineError
from sae_app.i18n import t
from sae_app.mtb_engine import (
    compute_equivalence_order_from_precomputed,
    normalize_student_identifier,
    precompute_equivalence_availability,
)
from sae_app.program_options import build_options
from sae_app.wish_list import (
    count_equivalence_orders,
    iter_equivalence_orders,
    predicted_outcome_final_chance,
    predicted_outcome_from_choices,
    prepare_ordered_wishes,
)

STATE: dict = {}


def _error_detail(error_key: str, message: str) -> dict:
    return {"error_key": error_key, "message": message}


def _engine_error(exc: MtbEngineError) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail=_error_detail(exc.message_key, t(exc.message_key, **exc.message_kwargs)),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    calib = load_calibration(CAPACITIES_PATH.read_bytes())
    program_options, program_mapping = build_options(calib)

    id_to_label: dict[str, str] = {}
    label_to_id: dict[str, str] = {}
    for label, row in program_mapping.items():
        program_id = f"{row['rbd']}:{row['program_code']}"
        id_to_label[program_id] = label
        label_to_id[label] = program_id

    STATE["calib"] = calib
    STATE["program_mapping"] = program_mapping
    STATE["id_to_label"] = id_to_label
    STATE["label_to_id"] = label_to_id
    yield
    STATE.clear()


app = FastAPI(
    title="SAE admission-risk simulation API",
    version="0.1.0",
    lifespan=lifespan,
)

# Open for now so a separately hosted frontend can call this during
# development. Restrict allow_origins to the deployed frontend's origin
# before this goes live for real users.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProgramSummary(BaseModel):
    program_id: str
    school_name: str
    school_commune: str
    region: str
    program_display_name: str
    program_track: str
    capacity: int
    true_applicants_last_year: int


class WishItem(BaseModel):
    program_id: str
    equivalence_group: int | None = Field(
        default=None,
        description=(
            "Wishes sharing the same group are treated as tied; lower numbers "
            "are preferred over higher ones. Omit for strict ranking, where "
            "each wish defaults to its own group equal to its position in the "
            "list."
        ),
    )
    priority_sibling: bool = False
    priority_student: bool = False
    priority_parent_civil_servant: bool = False
    priority_ex_student: bool = False
    priority_already_registered: bool = False


class SimulationRequest(BaseModel):
    student_id: str = Field(..., description="Student RUN/IPE, e.g. 12.345.678-9")
    wishes: list[WishItem] = Field(..., min_length=1, max_length=20)


class WishResult(BaseModel):
    wish_rank: int
    program_id: str
    program_label: str
    priority_tier: str
    capacity: int
    true_applicants_last_year: int
    availability_probability: float
    choice_assignment_probability: float


class SimulationVariant(BaseModel):
    order_index: int
    program_order: list[str]
    predicted_outcome: str
    predicted_outcome_program_id: str | None
    predicted_outcome_final_chance: float
    unmatched_risk: float
    at_risk: bool


class EquivalenceSensitivity(BaseModel):
    total_orders: int
    distinct_outcome_count: int
    outcome_stable: bool
    variants: list[SimulationVariant]


class SimulationResponse(BaseModel):
    unmatched_risk: float
    at_risk: bool
    predicted_outcome: str
    predicted_outcome_program_id: str | None
    wishes: list[WishResult]
    equivalence_sensitivity: EquivalenceSensitivity | None = None

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/regions", response_model=list[str])
def get_regions() -> list[str]:
    return available_regions(STATE["calib"])


@app.get("/programs", response_model=list[ProgramSummary])
def get_programs(
    region: str | None = Query(None, description="Exact region name, as returned by /regions"),
    q: str | None = Query(None, description="Free-text search over school name, commune, and program name"),
    limit: int = Query(200, ge=1, le=1000),
) -> list[ProgramSummary]:
    program_mapping = STATE["program_mapping"]
    label_to_id = STATE["label_to_id"]
    needle = q.strip().lower() if q else None

    results: list[ProgramSummary] = []
    for label, row in program_mapping.items():
        if region and str(row.get(REGION, "")).strip() != region:
            continue

        school_name = str(row.get(SCHOOL_NAME, "")).strip()
        school_commune = str(row.get(SCHOOL_COMMUNE, "")).strip()
        program_display_name = str(row.get(PROGRAM_DISPLAY_NAME, "")).strip()

        if needle:
            haystack = f"{school_name} {school_commune} {program_display_name}".lower()
            if needle not in haystack:
                continue

        results.append(ProgramSummary(
            program_id=label_to_id[label],
            school_name=school_name,
            school_commune=school_commune,
            region=str(row.get(REGION, "")).strip(),
            program_display_name=program_display_name,
            program_track=str(row.get(PROGRAM_TRACK, "")).strip(),
            capacity=int(row.get(CAPACITY, 0) or 0),
            true_applicants_last_year=int(row.get(TRUE_APP, 0) or 0),
        ))
        if len(results) >= limit:
            break

    return results


@app.post("/simulate", response_model=SimulationResponse)
def simulate(payload: SimulationRequest) -> SimulationResponse:
    try:
        normalize_student_identifier(payload.student_id)
    except MtbEngineError as exc:
        raise _engine_error(exc) from exc

    id_to_label = STATE["id_to_label"]
    program_mapping = STATE["program_mapping"]
    label_to_id = STATE["label_to_id"]

    rows = []
    for rank, wish in enumerate(payload.wishes, start=1):
        label = id_to_label.get(wish.program_id)
        if label is None:
            raise HTTPException(
                status_code=422,
                detail=_error_detail("unknown_program_id", f"Unknown program_id: {wish.program_id}"),
            )
        rows.append({
            WISH_RANK: rank,
            EQUIV_GROUP: wish.equivalence_group if wish.equivalence_group is not None else rank,
            PROGRAM: label,
            LOTTERY: 1,
            "priority_sibling": wish.priority_sibling,
            "priority_student": wish.priority_student,
            "priority_parent_civil_servant": wish.priority_parent_civil_servant,
            "priority_ex_student": wish.priority_ex_student,
            SAFETY: wish.priority_already_registered,
        })

    wishes_df = pd.DataFrame(
        rows, columns=[WISH_RANK, EQUIV_GROUP, PROGRAM, LOTTERY] + PRIORITIES + [SAFETY]
    )

    reference_order = prepare_ordered_wishes(wishes_df, use_equivalence_classes=True)
    if reference_order.empty:
        raise HTTPException(
            status_code=422,
            detail=_error_detail("empty_wish_list", "Add at least one valid wish."),
        )

    total_orders = count_equivalence_orders(reference_order)
    if total_orders > MAX_EXACT_EQUIV_PERMUTATIONS:
        raise HTTPException(
            status_code=422,
            detail=_error_detail(
                "too_many_equivalence_orders",
                f"The equivalence classes generate {total_orders:,} strict orders, "
                f"above the exact-evaluation limit of {MAX_EXACT_EQUIV_PERMUTATIONS:,}. "
                "Split large equivalence groups into smaller groups and try again.",
            ),
        )

    try:
        availability_lookup = precompute_equivalence_availability(
            reference_order, program_mapping, payload.student_id
        )

        variants: list[SimulationVariant] = []
        reference_choices = None

        for idx, strict_order in enumerate(iter_equivalence_orders(reference_order), start=1):
            choices = compute_equivalence_order_from_precomputed(strict_order, availability_lookup)
            outcome, p_unmatched, at_risk = predicted_outcome_from_choices(choices, HARD_UNMATCHED_THRESHOLD)

            if idx == 1:
                reference_choices = choices

            variants.append(SimulationVariant(
                order_index=idx,
                program_order=[label_to_id.get(str(p), "") for p in strict_order[PROGRAM]],
                predicted_outcome=outcome,
                predicted_outcome_program_id=label_to_id.get(outcome),
                predicted_outcome_final_chance=predicted_outcome_final_chance(choices, outcome),
                unmatched_risk=p_unmatched,
                at_risk=at_risk,
            ))
    except MtbEngineError as exc:
        raise _engine_error(exc) from exc

    reference_outcome, reference_p_unmatched, reference_at_risk = predicted_outcome_from_choices(
        reference_choices, HARD_UNMATCHED_THRESHOLD
    )

    equivalence_sensitivity = None
    if total_orders > 1:
        distinct_outcomes = {v.predicted_outcome for v in variants}
        equivalence_sensitivity = EquivalenceSensitivity(
            total_orders=total_orders,
            distinct_outcome_count=len(distinct_outcomes),
            outcome_stable=len(distinct_outcomes) == 1,
            variants=variants,
        )

    return SimulationResponse(
        unmatched_risk=reference_p_unmatched,
        at_risk=reference_at_risk,
        predicted_outcome=reference_outcome,
        predicted_outcome_program_id=label_to_id.get(reference_outcome),
        wishes=[
            WishResult(
                wish_rank=int(row["wish_rank"]),
                program_id=label_to_id.get(str(row["program"]), ""),
                program_label=str(row["program"]),
                priority_tier=str(row["priority_tier"]),
                capacity=int(row["capacity"]),
                true_applicants_last_year=int(row["true_applicants_last_year"]),
                availability_probability=float(row["availability_probability"]),
                choice_assignment_probability=float(row["choice_assignment_probability"]),
            )
            for _, row in reference_choices.iterrows()
        ],
        equivalence_sensitivity=equivalence_sensitivity,
    )
