"""Frozen-design utilities for cue-locus Gate R1 model recovery."""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .bms import random_effects_bms
from .config import (
    CUE_FORMULATION_VERSION,
    CUE_PRIOR_CALIBRATION_DESIGN_DIGEST,
    CUE_PRIOR_VERSION,
    cue_registry_digest,
)
from .data import SubjectData
from .numerics import QuasiNewtonOptions, RiddersOptions, laplace_evidence
from .objective import JointModel
from .prior_predictive import CalibrationDesign, calibration_design_digest
from .recovery import RecoveryDataset, simulate_recovery_dataset


FloatArray = NDArray[np.float64]

R1_VERSION = "cue-r1-model-recovery-0.1.0"
R1_CRITERIA_VERSION = "cue-r1-criteria-0.1.0"
R1_SEED = 202607230
R1_REPETITIONS = 20
R1_BMS_SAMPLES = 1_000_000
R1_DECISION_TIME_STEP = 0.01
R1_TER_REFERENCE_RT = 0.60
R1_INITIAL_OFFSET_SD = 0.25
R1_HGF_CACHE_SIZE = 256
R1_HGF_CACHE_ADDENDUM_VERSION = "cue-r1-hgf-cache-addendum-0.1.0"

ARCHITECTURE_MODELS = {
    "parallel": (
        "cue_history_w",
        "cue_parallel_w",
        "cue_parallel_vbias",
        "cue_parallel_w_vbias",
    ),
    "integrated": (
        "cue_history_w",
        "cue_integrated_w",
        "cue_integrated_vbias",
        "cue_integrated_w_vbias",
    ),
}

GENERATING_MODELS = {
    "parallel": {
        "null": "cue_history_w",
        "w": "cue_parallel_w",
        "v0": "cue_parallel_vbias",
        "w_v0": "cue_parallel_w_vbias",
    },
    "integrated": {
        "null": "cue_history_w",
        "w": "cue_integrated_w",
        "v0": "cue_integrated_vbias",
        "w_v0": "cue_integrated_w_vbias",
    },
}

EFFECT_TRANSFORMED_VALUES = {
    "parallel": {
        "w": {
            "weak": 0.020000666706669477,
            "medium": 0.06001800972625305,
            "strong": 0.1000834585569826,
        },
        "v0": {
            "weak": 0.025431265192825007,
            "medium": 0.07630542572272696,
            "strong": 0.1272149128342861,
        },
    },
    "integrated": {
        "w": {
            "weak": 0.3284540568952013,
            "medium": 1.0676488381920615,
            "strong": 2.2765461945358165,
        },
        "v0": {
            "weak": 0.7309251850195388,
            "medium": 2.2028557468758128,
            "strong": 3.7505622404216705,
        },
    },
}

NUISANCE_RANGES = {
    "hgf.omega_2": (-4.0, -2.0),
    "ddm.log_a_a": (-0.45, 0.45),
    "ddm.log_a_v": (-0.45, 0.45),
    "ddm.b_H_w": (-0.80, 0.80),
    "ddm.b_c": (-0.60, 0.60),
    "ddm.Ter_logit": (-1.0, 1.0),
}


@dataclass(frozen=True)
class CueR1Cell:
    architecture: str
    locus: str
    effect_level: str
    generating_model: str
    primary_gate: bool

    @property
    def identifier(self) -> str:
        return "%s__%s__%s" % (
            self.architecture,
            self.locus,
            self.effect_level,
        )


@dataclass(frozen=True)
class CueR1Fit:
    model_id: str
    status: str
    free_parameter_names: tuple
    starts: tuple
    selected_start: Optional[int]
    estimate: Optional[tuple]
    negative_log_joint: Optional[float]
    negative_log_likelihood: Optional[float]
    log_prior: Optional[float]
    lme: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    iterations: Optional[int]
    reset_count: Optional[int]
    convergence_reason: Optional[str]
    hessian_minimum_eigenvalue: Optional[float]
    hessian_condition_number: Optional[float]
    used_bfgs_fallback: Optional[bool]
    hessian_fallback_reason: Optional[str]
    hessian: Optional[tuple]
    covariance: Optional[tuple]
    correlation: Optional[tuple]
    hgf_cache_diagnostics: dict
    error_message: Optional[str]


def cue_r1_cells() -> Tuple[CueR1Cell, ...]:
    cells = []
    for architecture in ("parallel", "integrated"):
        cells.append(
            CueR1Cell(
                architecture=architecture,
                locus="null",
                effect_level="zero",
                generating_model=GENERATING_MODELS[architecture]["null"],
                primary_gate=True,
            )
        )
        for locus in ("w", "v0", "w_v0"):
            for level in ("weak", "medium", "strong"):
                cells.append(
                    CueR1Cell(
                        architecture=architecture,
                        locus=locus,
                        effect_level=level,
                        generating_model=GENERATING_MODELS[architecture][locus],
                        primary_gate=level in {"medium", "strong"},
                    )
                )
    return tuple(cells)


def cue_r1_manifest(prior_manifest_digest: str) -> dict:
    """Return the complete pre-generation Gate R1 contract."""

    payload = {
        "status": "frozen_before_generation",
        "gate_version": R1_VERSION,
        "criteria_version": R1_CRITERIA_VERSION,
        "formulation_version": CUE_FORMULATION_VERSION,
        "prior_version": CUE_PRIOR_VERSION,
        "prior_manifest_digest": str(prior_manifest_digest),
        "registry_digest": cue_registry_digest(),
        "design_digest": CUE_PRIOR_CALIBRATION_DESIGN_DIGEST,
        "outcome_values_used": False,
        "subjects": 37,
        "trials_per_subject": 380,
        "hgf_update_trials": [1, 380],
        "response_likelihood_trials": [101, 380],
        "response_deadline_seconds": 3.0,
        "decision_time_step": R1_DECISION_TIME_STEP,
        "ter_reference_rt_seconds": R1_TER_REFERENCE_RT,
        "repetitions_per_cell": R1_REPETITIONS,
        "seed": R1_SEED,
        "candidate_sets": {
            key: list(value) for key, value in ARCHITECTURE_MODELS.items()
        },
        "cells": [asdict(cell) | {"identifier": cell.identifier} for cell in cue_r1_cells()],
        "effect_transformed_values": EFFECT_TRANSFORMED_VALUES,
        "truth_variation": {
            "method": "independently_permuted_latin_hypercube_by_subject",
            "nuisance_ranges": NUISANCE_RANGES,
            "effect_multiplier_range": [0.70, 1.30],
        },
        "optimizer": {
            "algorithm": "ported_TAPAS_Ridders_BFGS",
            "initial_values": 3,
            "initial_scheme": "prior_mean_and_symmetric_0.25_prior_sd_offsets",
            "max_iterations": 100,
            "ridders_min_steps": 10,
            "laplace_lme": True,
        },
        "bms": {
            "algorithm": "SPM_style_random_effects_Gibbs",
            "samples": R1_BMS_SAMPLES,
            "model_prior": "equal_within_architecture",
        },
        "success_criteria": {
            "primary_effect_levels": ["medium", "strong"],
            "generating_locus_expected_frequency_winner_rate_min": 0.80,
            "null_false_positive_rate_max": 0.10,
            "w_to_v0_misclassification_rate_max": 0.10,
            "v0_to_w_misclassification_rate_max": 0.10,
            "parameter_correlation_min": 0.70,
            "parameter_absolute_bias_max": 0.50,
            "parameter_rmse_over_prior_sd_max": 1.0,
            "parameter_cases_min": 20,
        },
        "note": (
            "Weak effects are descriptive. Medium and strong effects are the "
            "primary Gate. Results may not alter this manifest in place."
        ),
    }
    payload["manifest_digest"] = cue_r1_manifest_digest(payload)
    return payload


def cue_r1_manifest_digest(manifest: dict) -> str:
    payload = dict(manifest)
    payload.pop("manifest_digest", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cue_r1_execution_addendum_digest(addendum: dict) -> str:
    payload = dict(addendum)
    payload.pop("addendum_digest", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_r1_designs(designs: Sequence[CalibrationDesign]) -> None:
    if len(designs) != 37:
        raise ValueError("Gate R1 requires exactly 37 subject designs.")
    if calibration_design_digest(designs) != CUE_PRIOR_CALIBRATION_DESIGN_DIGEST:
        raise ValueError("Gate R1 design digest differs from the frozen contract.")


def redacted_template(subject: SubjectData, model_id: str) -> JointModel:
    """Build a response-redacted model while preserving the real mask."""

    design = CalibrationDesign.from_subject(subject)
    y = np.full((380, 2), np.nan)
    y[design.likelihood_mask, 0] = R1_TER_REFERENCE_RT
    y[design.likelihood_mask, 1] = design.stimulus[design.likelihood_mask]
    return JointModel(
        u=design.u,
        y=y,
        model_id=model_id,
        tie=design.tie,
        cue_evidence=design.cue_evidence,
    )


def r1_truth_vector(
    model: JointModel,
    cell: CueR1Cell,
    repetition: int,
    subject_index: int,
    subject_count: int = 37,
) -> FloatArray:
    """Deterministic independently permuted Latin-hypercube subject truth."""

    if not 0 <= subject_index < subject_count:
        raise ValueError("Subject index lies outside the declared group.")
    if not 0 <= repetition < R1_REPETITIONS:
        raise ValueError("Repetition lies outside the frozen Gate range.")
    truth = model.initial_free_parameters.copy()
    names = model.free_parameter_names
    for parameter, limits in NUISANCE_RANGES.items():
        if parameter in names:
            truth[names.index(parameter)] = _latin_value(
                limits,
                cell.identifier,
                repetition,
                parameter,
                subject_index,
                subject_count,
            )
    effects = _cell_effect_parameters(cell)
    for parameter, base_value in effects.items():
        multiplier = _latin_value(
            (0.70, 1.30),
            cell.identifier,
            repetition,
            parameter + ".multiplier",
            subject_index,
            subject_count,
        )
        truth[names.index(parameter)] = base_value * multiplier
    return truth


def simulate_r1_subject(
    subject: SubjectData,
    cell: CueR1Cell,
    repetition: int,
    subject_index: int,
) -> RecoveryDataset:
    template = redacted_template(subject, cell.generating_model)
    truth = r1_truth_vector(template, cell, repetition, subject_index)
    seed = _stable_seed(cell.identifier, repetition, subject_index, "generation")
    return simulate_recovery_dataset(
        template,
        truth,
        seed=seed,
        decision_time_step=R1_DECISION_TIME_STEP,
    )


def fit_r1_candidate(
    dataset: RecoveryDataset,
    model_id: str,
    optimizer_options: Optional[QuasiNewtonOptions] = None,
    compute_lme: bool = True,
    hgf_cache_size: int = 0,
) -> CueR1Fit:
    """Fit three predeclared starts and compute LME only for the best MAP."""

    source = dataset.model
    model = JointModel(
        u=source.u.copy(),
        y=source.y.copy(),
        model_id=model_id,
        tie=source.tie.copy(),
        cue_evidence=source.cue_evidence.copy(),
        hgf_cache_size=hgf_cache_size,
    )
    options = optimizer_options or QuasiNewtonOptions(record_trace=False)
    starts = _initial_values(model)
    attempted = []
    successful = []
    for index, initial in enumerate(starts):
        try:
            fit = model.fit_map(initial=initial, options=options, compute_lme=False)
            record = {
                "index": index,
                "status": "ok",
                "initial": initial.tolist(),
                "estimate": fit.optimization.argument_minimum.tolist(),
                "negative_log_joint": float(fit.optimization.value_minimum),
                "iterations": int(fit.optimization.iterations),
                "reset_count": int(fit.optimization.reset_count),
                "convergence_reason": fit.optimization.convergence_reason,
            }
            if np.isfinite(fit.optimization.value_minimum):
                successful.append((index, fit))
        except Exception as error:  # noqa: BLE001 - every failed start is audited
            record = {
                "index": index,
                "status": "failed",
                "initial": initial.tolist(),
                "error_message": "%s: %s" % (type(error).__name__, str(error)),
            }
        attempted.append(record)
    if not successful:
        return CueR1Fit(
            model_id=model_id,
            status="failed",
            free_parameter_names=model.free_parameter_names,
            starts=tuple(attempted),
            selected_start=None,
            estimate=None,
            negative_log_joint=None,
            negative_log_likelihood=None,
            log_prior=None,
            lme=None,
            aic=None,
            bic=None,
            iterations=None,
            reset_count=None,
            convergence_reason=None,
            hessian_minimum_eigenvalue=None,
            hessian_condition_number=None,
            used_bfgs_fallback=None,
            hessian_fallback_reason=None,
            hessian=None,
            covariance=None,
            correlation=None,
            hgf_cache_diagnostics=model.hgf_cache_diagnostics,
            error_message="No finite MAP start.",
        )
    selected_index, selected = min(
        successful, key=lambda item: item[1].optimization.value_minimum
    )
    evaluation = selected.evaluation
    observations = int(np.sum(np.all(np.isfinite(model.y), axis=1)))
    dimension = selected.optimization.argument_minimum.size
    aic = 2.0 * evaluation.negative_log_likelihood + 2.0 * dimension
    bic = 2.0 * evaluation.negative_log_likelihood + dimension * np.log(observations)
    laplace = None
    try:
        if compute_lme:
            laplace = laplace_evidence(model.objective, selected.optimization)
    except Exception as error:  # noqa: BLE001 - retain MAP but mark LME failure
        return CueR1Fit(
            model_id=model_id,
            status="lme_failed",
            free_parameter_names=model.free_parameter_names,
            starts=tuple(attempted),
            selected_start=selected_index,
            estimate=tuple(map(float, selected.optimization.argument_minimum)),
            negative_log_joint=float(selected.optimization.value_minimum),
            negative_log_likelihood=float(evaluation.negative_log_likelihood),
            log_prior=float(evaluation.log_prior),
            lme=None,
            aic=float(aic),
            bic=float(bic),
            iterations=int(selected.optimization.iterations),
            reset_count=int(selected.optimization.reset_count),
            convergence_reason=selected.optimization.convergence_reason,
            hessian_minimum_eigenvalue=None,
            hessian_condition_number=None,
            used_bfgs_fallback=None,
            hessian_fallback_reason=None,
            hessian=None,
            covariance=None,
            correlation=None,
            hgf_cache_diagnostics=model.hgf_cache_diagnostics,
            error_message="%s: %s" % (type(error).__name__, str(error)),
        )
    return CueR1Fit(
        model_id=model_id,
        status="ok" if compute_lme else "map_only",
        free_parameter_names=model.free_parameter_names,
        starts=tuple(attempted),
        selected_start=selected_index,
        estimate=tuple(map(float, selected.optimization.argument_minimum)),
        negative_log_joint=float(selected.optimization.value_minimum),
        negative_log_likelihood=float(evaluation.negative_log_likelihood),
        log_prior=float(evaluation.log_prior),
        lme=None if laplace is None else float(laplace.lme),
        aic=float(aic),
        bic=float(bic),
        iterations=int(selected.optimization.iterations),
        reset_count=int(selected.optimization.reset_count),
        convergence_reason=selected.optimization.convergence_reason,
        hessian_minimum_eigenvalue=(
            None if laplace is None else laplace.numerical_minimum_eigenvalue
        ),
        hessian_condition_number=(
            None if laplace is None else laplace.numerical_condition_number
        ),
        used_bfgs_fallback=(
            None if laplace is None else bool(laplace.used_bfgs_fallback)
        ),
        hessian_fallback_reason=(
            None if laplace is None else laplace.fallback_reason
        ),
        hessian=(
            None
            if laplace is None
            else tuple(tuple(map(float, row)) for row in laplace.hessian)
        ),
        covariance=(
            None
            if laplace is None
            else tuple(tuple(map(float, row)) for row in laplace.covariance)
        ),
        correlation=(
            None
            if laplace is None
            else tuple(tuple(map(float, row)) for row in laplace.correlation)
        ),
        hgf_cache_diagnostics=model.hgf_cache_diagnostics,
        error_message=None,
    )


def r1_bms(
    lme: FloatArray,
    model_ids: Sequence[str],
    subject_ids: Sequence[str],
    seed: int,
    samples: int = R1_BMS_SAMPLES,
) -> dict:
    result = random_effects_bms(
        lme,
        model_ids,
        subject_ids,
        samples=samples,
        seed=seed,
        run_sensitivity=False,
    )
    return {
        "model_ids": list(result.model_ids),
        "expected_frequency": result.primary.expected_frequency.tolist(),
        "exceedance_probability": result.primary.exceedance_probability.tolist(),
        "winner": result.model_ids[int(np.argmax(result.primary.expected_frequency))],
        "samples": samples,
        "seed": seed,
    }


def _cell_effect_parameters(cell: CueR1Cell) -> Dict[str, float]:
    if cell.locus == "null":
        return {}
    parameters = {}
    if cell.locus in {"w", "w_v0"}:
        name = "ddm.gamma_w" if cell.architecture == "parallel" else "ddm.b_w"
        parameters[name] = EFFECT_TRANSFORMED_VALUES[cell.architecture]["w"][
            cell.effect_level
        ]
    if cell.locus in {"v0", "w_v0"}:
        name = "ddm.gamma_v0" if cell.architecture == "parallel" else "ddm.b_v"
        parameters[name] = EFFECT_TRANSFORMED_VALUES[cell.architecture]["v0"][
            cell.effect_level
        ]
    return parameters


def _latin_value(
    limits: Tuple[float, float],
    cell_id: str,
    repetition: int,
    parameter: str,
    subject_index: int,
    subject_count: int,
) -> float:
    seed = _stable_seed(cell_id, repetition, parameter)
    rng = np.random.Generator(np.random.MT19937(seed))
    order = rng.permutation(subject_count)
    quantile = (float(order[subject_index]) + 0.5) / subject_count
    return float(limits[0] + quantile * (limits[1] - limits[0]))


def _stable_seed(*parts) -> int:
    text = "|".join(map(str, (R1_SEED,) + parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _initial_values(model: JointModel) -> Tuple[FloatArray, ...]:
    mean = model.initial_free_parameters.copy()
    variance = np.concatenate(
        (
            model.hgf_prior.variances[model.hgf_prior.free_mask],
            model.response_prior.variances[model.response_prior.free_mask],
        )
    )
    offset = R1_INITIAL_OFFSET_SD * np.sqrt(variance)
    signs = np.where(np.arange(mean.size) % 2 == 0, 1.0, -1.0)
    return (mean, mean + signs * offset, mean - signs * offset)


def optimizer_from_manifest(
    max_iterations: int = 100,
    ridders_min_steps: int = 10,
) -> QuasiNewtonOptions:
    return QuasiNewtonOptions(
        max_iterations=max_iterations,
        record_trace=False,
        gradient_options=RiddersOptions(min_steps=ridders_min_steps),
    )
