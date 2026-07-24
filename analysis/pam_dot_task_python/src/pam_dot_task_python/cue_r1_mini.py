"""Frozen small-scale diagnostic preceding the formal cue-locus Gate R1."""

import hashlib
import json
from dataclasses import asdict
from typing import Sequence, Tuple

import numpy as np

from .cue_r1 import (
    ARCHITECTURE_MODELS,
    R1_HGF_CACHE_SIZE,
    CueR1Cell,
    cue_r1_cells,
)
from .prior_predictive import CalibrationDesign, calibration_design_digest


MINI_VERSION = "cue-r1-mini-screen-0.1.0"
MINI_SEED = 202607240
MINI_SUBJECTS_PER_CONDITION = 2
MINI_REPETITIONS = 2
MINI_BMS_SAMPLES = 100_000


def cue_r1_mini_cells() -> Tuple[CueR1Cell, ...]:
    """Keep only null, medium-w, and medium-v0 generation in each architecture."""

    wanted = {
        (architecture, locus, effect)
        for architecture in ("parallel", "integrated")
        for locus, effect in (("null", "zero"), ("w", "medium"), ("v0", "medium"))
    }
    return tuple(
        cell
        for cell in cue_r1_cells()
        if (cell.architecture, cell.locus, cell.effect_level) in wanted
    )


def select_mini_subject_indices(subjects: Sequence) -> Tuple[int, ...]:
    """Select two design-representative subjects per counterbalance condition.

    Selection uses only condition and the number of response-likelihood trials.
    Choice and RT values are never inspected. Within each condition, the subjects
    nearest the one-third and two-thirds missingness ranks are selected.
    """

    if len(subjects) != 37:
        raise ValueError("The mini screen must be selected from the frozen 37 subjects.")
    selected = []
    for condition in ("normal", "normal_cb", "reverse", "reverse_cb"):
        candidates = [
            (index, int(len(subject.likelihood_trials)), subject.subject_id)
            for index, subject in enumerate(subjects)
            if subject.condition.name == condition
        ]
        if len(candidates) < MINI_SUBJECTS_PER_CONDITION:
            raise ValueError("Condition %s has fewer than two subjects." % condition)
        candidates.sort(key=lambda item: (item[1], item[2]))
        ranks = (
            int(round((len(candidates) - 1) / 3.0)),
            int(round(2.0 * (len(candidates) - 1) / 3.0)),
        )
        if ranks[0] == ranks[1]:
            ranks = (0, len(candidates) - 1)
        selected.extend(candidates[rank][0] for rank in ranks)
    return tuple(sorted(selected))


def cue_r1_mini_manifest(
    parent_manifest_digest: str,
    subjects: Sequence,
    designs: Sequence[CalibrationDesign],
) -> dict:
    indices = select_mini_subject_indices(subjects)
    selection = []
    for index in indices:
        subject = subjects[index]
        selection.append(
            {
                "full_cohort_index": index,
                "subject_id": subject.subject_id,
                "condition": subject.condition.name,
                "response_likelihood_trials": int(len(subject.likelihood_trials)),
            }
        )
    payload = {
        "status": "frozen_before_generation",
        "screen_version": MINI_VERSION,
        "screening_only": True,
        "formal_gate_status": "not_evaluable",
        "parent_gate_manifest_digest": str(parent_manifest_digest),
        "full_design_digest": calibration_design_digest(designs),
        "selection_rule": (
            "two subjects per condition nearest the one-third and two-thirds "
            "ranks of response-likelihood trial count; outcome values unused"
        ),
        "selected_subjects": selection,
        "subjects": len(selection),
        "repetitions_per_cell": MINI_REPETITIONS,
        "cells": [
            asdict(cell) | {"identifier": cell.identifier}
            for cell in cue_r1_mini_cells()
        ],
        "candidate_sets": {
            key: list(value) for key, value in ARCHITECTURE_MODELS.items()
        },
        "optimizer": {
            "algorithm": "ported_TAPAS_Ridders_BFGS",
            "initial_values": 3,
            "max_iterations": 100,
            "ridders_min_steps": 10,
            "laplace_lme": True,
            "hgf_cache_size": R1_HGF_CACHE_SIZE,
        },
        "bms": {
            "algorithm": "SPM_style_random_effects_Gibbs",
            "samples": MINI_BMS_SAMPLES,
        },
        "seed": MINI_SEED,
        "declared_subject_tasks": (
            len(selection) * MINI_REPETITIONS * len(cue_r1_mini_cells())
        ),
        "declared_lme_fits": (
            len(selection)
            * MINI_REPETITIONS
            * sum(
                len(ARCHITECTURE_MODELS[cell.architecture])
                for cell in cue_r1_mini_cells()
            )
        ),
        "interpretation": (
            "A quick no-go diagnostic for gross non-recoverability. It cannot "
            "pass or replace formal Gate R1 and does not validate weak, strong, "
            "or combined-locus generation."
        ),
    }
    payload["manifest_digest"] = cue_r1_mini_manifest_digest(payload)
    return payload


def cue_r1_mini_manifest_digest(manifest: dict) -> str:
    payload = dict(manifest)
    payload.pop("manifest_digest", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
