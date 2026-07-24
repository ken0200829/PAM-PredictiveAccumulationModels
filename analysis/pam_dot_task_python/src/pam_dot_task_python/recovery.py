"""Parameter-recovery generation, fitting, and summary utilities."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .numerics import QuasiNewtonOptions
from .objective import JointModel, TapasMapFit
from .ppc import SimulationBatch, simulate_ddm


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RecoveryDataset:
    model: JointModel
    simulation: SimulationBatch
    generative_free_parameters: FloatArray
    estimation_scale_truth: FloatArray
    likelihood_mask: NDArray[np.bool_]
    seed: int


@dataclass(frozen=True)
class RecoveryCaseResult:
    dataset: RecoveryDataset
    fit: TapasMapFit
    estimated_free_parameters: FloatArray
    error: FloatArray


@dataclass(frozen=True)
class RecoveryResult:
    parameter_names: tuple
    cases: Tuple[RecoveryCaseResult, ...]
    summary: pd.DataFrame


def simulate_recovery_dataset(
    template_model: JointModel,
    true_free_parameters: FloatArray,
    seed: int,
    decision_time_step: float = 0.001,
) -> RecoveryDataset:
    """Generate one masked dataset while preserving the original trial design."""

    truth = np.asarray(true_free_parameters, dtype=float)
    if truth.shape != template_model.initial_free_parameters.shape:
        raise ValueError("Recovery truth does not match the model dimension.")
    if np.any(~np.isfinite(truth)):
        raise ValueError("Recovery truth must contain only finite values.")
    evaluation = template_model.evaluate(truth)
    simulation = simulate_ddm(
        evaluation.trialwise,
        replicates=1,
        seed=seed,
        decision_time_step=decision_time_step,
    )
    likelihood_mask = np.all(np.isfinite(template_model.y), axis=1)
    if not np.any(likelihood_mask):
        raise ValueError("Recovery requires at least one likelihood trial.")
    generated_y = np.full_like(template_model.y, np.nan, dtype=float)
    generated_y[likelihood_mask, 0] = simulation.rt[0, likelihood_mask]
    generated_y[likelihood_mask, 1] = simulation.choice[0, likelihood_mask]
    generated_model = JointModel(
        u=template_model.u.copy(),
        y=generated_y,
        model_id=template_model.model_id,
        tie=template_model.tie.copy(),
        cue_evidence=(
            None
            if template_model.cue_evidence is None
            else template_model.cue_evidence.copy()
        ),
        hgf_prior=template_model.hgf_prior,
    )

    estimation_truth = truth.copy()
    ter_name = "ddm.Ter_logit"
    if ter_name in template_model.free_parameter_names:
        ter_index = template_model.free_parameter_names.index(ter_name)
        generated_minimum_rt = float(np.min(generated_y[likelihood_mask, 0]))
        ter_ratio = evaluation.ddm_parameters.Ter / generated_minimum_rt
        if not 0 < ter_ratio < 1:
            raise FloatingPointError("Generated RT does not admit the true Ter transform.")
        estimation_truth[ter_index] = np.log(ter_ratio / (1.0 - ter_ratio))

    return RecoveryDataset(
        model=generated_model,
        simulation=simulation,
        generative_free_parameters=truth.copy(),
        estimation_scale_truth=estimation_truth,
        likelihood_mask=likelihood_mask,
        seed=seed,
    )


def run_recovery_case(
    template_model: JointModel,
    true_free_parameters: FloatArray,
    seed: int,
    optimizer_options: Optional[QuasiNewtonOptions] = None,
    decision_time_step: float = 0.001,
    compute_lme: bool = False,
) -> RecoveryCaseResult:
    """Generate and refit one parameter-recovery dataset."""

    dataset = simulate_recovery_dataset(
        template_model,
        true_free_parameters,
        seed,
        decision_time_step,
    )
    fit = dataset.model.fit_map(
        options=optimizer_options,
        compute_lme=compute_lme,
    )
    estimate = fit.optimization.argument_minimum.copy()
    return RecoveryCaseResult(
        dataset=dataset,
        fit=fit,
        estimated_free_parameters=estimate,
        error=estimate - dataset.estimation_scale_truth,
    )


def run_parameter_recovery(
    template_model: JointModel,
    true_parameter_sets: Sequence[FloatArray],
    seeds: Sequence[int],
    optimizer_options: Optional[QuasiNewtonOptions] = None,
    decision_time_step: float = 0.001,
    compute_lme: bool = False,
) -> RecoveryResult:
    """Run a declared recovery grid and summarize transformed-space recovery."""

    truths = tuple(true_parameter_sets)
    run_seeds = tuple(seeds)
    if not truths or len(truths) != len(run_seeds):
        raise ValueError("Recovery truth sets and seeds must have equal non-zero length.")
    cases = tuple(
        run_recovery_case(
            template_model,
            truth,
            seed,
            optimizer_options,
            decision_time_step,
            compute_lme,
        )
        for truth, seed in zip(truths, run_seeds)
    )
    true_matrix = np.vstack(
        [case.dataset.estimation_scale_truth for case in cases]
    )
    estimate_matrix = np.vstack(
        [case.estimated_free_parameters for case in cases]
    )
    summary = summarize_recovery(
        true_matrix,
        estimate_matrix,
        template_model.free_parameter_names,
    )
    return RecoveryResult(
        parameter_names=template_model.free_parameter_names,
        cases=cases,
        summary=summary,
    )


def summarize_recovery(
    true_parameters: FloatArray,
    estimated_parameters: FloatArray,
    parameter_names: Sequence[str],
) -> pd.DataFrame:
    """Return bias, absolute error, RMSE, and across-case correlation."""

    truth = np.asarray(true_parameters, dtype=float)
    estimate = np.asarray(estimated_parameters, dtype=float)
    names = tuple(parameter_names)
    if truth.ndim != 2 or estimate.shape != truth.shape:
        raise ValueError("Recovery arrays must be equal case-by-parameter matrices.")
    if truth.shape[1] != len(names):
        raise ValueError("Parameter names do not match recovery matrix columns.")
    rows = []
    for index, name in enumerate(names):
        valid = np.isfinite(truth[:, index]) & np.isfinite(estimate[:, index])
        true_values = truth[valid, index]
        estimated_values = estimate[valid, index]
        error = estimated_values - true_values
        correlation = np.nan
        if (
            valid.sum() >= 3
            and np.std(true_values, ddof=1) > 0
            and np.std(estimated_values, ddof=1) > 0
        ):
            correlation = float(np.corrcoef(true_values, estimated_values)[0, 1])
        rows.append(
            {
                "parameter": name,
                "cases": int(valid.sum()),
                "bias": float(np.mean(error)) if error.size else np.nan,
                "mean_absolute_error": (
                    float(np.mean(np.abs(error))) if error.size else np.nan
                ),
                "rmse": (
                    float(np.sqrt(np.mean(error**2))) if error.size else np.nan
                ),
                "correlation": correlation,
            }
        )
    return pd.DataFrame(rows)
