"""MAP/Laplace DDM simulation, aggregate PPC, and time-resolved PPC."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .objective import JointModel, TapasMapFit
from .response import TrialwiseDDM
from .wfpt import wfpt_density


FloatArray = NDArray[np.float64]


STATISTICS = (
    "choice_rate",
    "accuracy",
    "rt_q10",
    "rt_q50",
    "rt_q90",
    "rt_choice0_q50",
    "rt_choice1_q50",
)
RESPONSE_DEADLINE_SECONDS = 3.0


@dataclass(frozen=True)
class PPCWindow:
    identifier: str
    family: str
    indices: NDArray[np.int64]


@dataclass(frozen=True)
class PPCSpec:
    version: str
    windows: Tuple[PPCWindow, ...]
    statistics: tuple = STATISTICS
    min_valid_trials: int = 5
    point_interval: Tuple[float, float] = (0.025, 0.975)
    simultaneous_level: float = 0.95
    global_discrepancy: str = "max_absolute_standardized_deviation"


@dataclass(frozen=True)
class SimulationBatch:
    rt: FloatArray
    choice: FloatArray
    replicates: int
    seed: int
    algorithm: str
    decision_time_step: float
    max_decision_time: float
    captured_mass: FloatArray
    parameter_mode: str = "MAP_fixed"
    parameter_draw_id: Optional[NDArray[np.int64]] = None
    free_parameter_draws: Optional[FloatArray] = None
    draw_diagnostics: Optional["PosteriorDrawDiagnostics"] = None


@dataclass(frozen=True)
class PosteriorDrawPolicy:
    """Configurable Gate-PPC criteria; defaults are not a frozen analysis decision."""

    max_condition_number: float = 1e8
    max_rejection_rate: float = 0.20

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_condition_number) or self.max_condition_number <= 1:
            raise ValueError("max_condition_number must be finite and greater than one.")
        if not 0 <= self.max_rejection_rate < 1:
            raise ValueError("max_rejection_rate must lie in [0,1).")


@dataclass(frozen=True)
class PosteriorDrawDiagnostics:
    requested_mode: str
    used_mode: str
    fallback_reason: Optional[str]
    proposal_count: int
    rejected_draws: int
    rejection_rate: float
    numerical_condition_number: float
    max_condition_number: float
    max_rejection_rate: float


@dataclass(frozen=True)
class PPCResult:
    spec: PPCSpec
    summary: pd.DataFrame
    replicated_statistics: FloatArray
    global_observed: float
    global_replicated: FloatArray
    global_tail_probability: float
    simultaneous_threshold: float
    replicates: int


def simulate_ddm(
    trialwise: TrialwiseDDM,
    replicates: int = 2000,
    seed: int = 20260720,
    decision_time_step: float = 0.001,
    max_decision_time: float = RESPONSE_DEADLINE_SECONDS,
) -> SimulationBatch:
    """Generate one replicate-by-trial batch using PAM's WFPT mapping."""

    if not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be a positive integer.")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    _validate_deadline(max_decision_time)
    if decision_time_step <= 0:
        raise ValueError("decision_time_step must be positive.")
    w = np.asarray(trialwise.w, dtype=float)
    boundary = np.asarray(trialwise.a, dtype=float)
    drift = np.asarray(trialwise.v, dtype=float)
    nondecision = np.asarray(trialwise.Ter, dtype=float)
    if not (w.shape == boundary.shape == drift.shape == nondecision.shape):
        raise ValueError("Trial-wise DDM parameter arrays must share one shape.")
    if w.ndim != 1 or w.size == 0:
        raise ValueError("Trial-wise DDM parameters must be non-empty vectors.")
    if np.any(~np.isfinite(w)) or np.any((w <= 0) | (w >= 1)):
        raise ValueError("Starting points must lie strictly inside (0,1).")
    if np.any(~np.isfinite(boundary)) or np.any(boundary <= 0):
        raise ValueError("Boundary separations must be finite and positive.")
    if np.any(~np.isfinite(drift)):
        raise ValueError("Drifts must be finite.")
    if np.any(~np.isfinite(nondecision)) or np.any(nondecision <= 0):
        raise ValueError("Non-decision times must be finite and positive.")

    rng = np.random.Generator(np.random.MT19937(seed))
    rt, choice, captured_mass = _simulate_ddm_with_rng(
        trialwise,
        replicates,
        rng,
        decision_time_step,
        max_decision_time,
    )
    return SimulationBatch(
        rt=rt,
        choice=choice,
        replicates=replicates,
        seed=seed,
        algorithm="MT19937",
        decision_time_step=decision_time_step,
        max_decision_time=max_decision_time,
        captured_mass=captured_mass,
        parameter_draw_id=np.zeros(replicates, dtype=np.int64),
    )


def simulate_posterior_ddm(
    model: JointModel,
    fit: TapasMapFit,
    replicates: int = 2000,
    seed: int = 20260720,
    decision_time_step: float = 0.001,
    max_decision_time: float = RESPONSE_DEADLINE_SECONDS,
    policy: PosteriorDrawPolicy = PosteriorDrawPolicy(),
) -> SimulationBatch:
    """Draw transformed parameters from a valid Laplace fit or record MAP fallback."""

    if not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be a positive integer.")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    _validate_deadline(max_decision_time)
    if decision_time_step <= 0:
        raise ValueError("decision_time_step must be positive.")
    map_parameters = np.asarray(fit.optimization.argument_minimum, dtype=float)
    if map_parameters.shape != model.initial_free_parameters.shape:
        raise ValueError("Fit and model free-parameter dimensions differ.")
    fallback_reason = _laplace_fallback_reason(fit, map_parameters.size, policy)
    condition = (
        np.inf
        if fit.laplace is None
        else float(fit.laplace.numerical_condition_number)
    )
    rng = np.random.Generator(np.random.MT19937(seed))
    if fallback_reason is not None:
        return _map_fallback_batch(
            fit.evaluation.trialwise,
            map_parameters,
            replicates,
            seed,
            decision_time_step,
            max_decision_time,
            policy,
            fallback_reason,
            condition,
            proposal_count=0,
            rejected_draws=0,
        )

    covariance = np.asarray(fit.laplace.covariance, dtype=float)
    accepted_parameters = []
    accepted_rt = []
    accepted_choice = []
    accepted_mass = []
    proposal_count = 0
    max_proposals = int(
        np.floor(replicates / (1.0 - policy.max_rejection_rate))
    )
    max_proposals = max(max_proposals, replicates)
    while len(accepted_parameters) < replicates and proposal_count < max_proposals:
        proposal_count += 1
        proposal = rng.multivariate_normal(map_parameters, covariance)
        try:
            evaluation = model.evaluate(proposal)
            _validate_trialwise_support(evaluation.trialwise)
            rt, choice, mass = _simulate_ddm_with_rng(
                evaluation.trialwise,
                1,
                rng,
                decision_time_step,
                max_decision_time,
            )
        except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
            continue
        accepted_parameters.append(proposal)
        accepted_rt.append(rt[0])
        accepted_choice.append(choice[0])
        accepted_mass.append(mass)

    rejected = proposal_count - len(accepted_parameters)
    rejection_rate = rejected / proposal_count if proposal_count else 0.0
    if len(accepted_parameters) < replicates or rejection_rate > policy.max_rejection_rate:
        return _map_fallback_batch(
            fit.evaluation.trialwise,
            map_parameters,
            replicates,
            seed,
            decision_time_step,
            max_decision_time,
            policy,
            "posterior_draw_rejection_rate_exceeded",
            condition,
            proposal_count,
            rejected,
        )

    diagnostics = PosteriorDrawDiagnostics(
        requested_mode="Laplace_draw",
        used_mode="Laplace_draw",
        fallback_reason=None,
        proposal_count=proposal_count,
        rejected_draws=rejected,
        rejection_rate=rejection_rate,
        numerical_condition_number=condition,
        max_condition_number=policy.max_condition_number,
        max_rejection_rate=policy.max_rejection_rate,
    )
    return SimulationBatch(
        rt=np.asarray(accepted_rt),
        choice=np.asarray(accepted_choice),
        replicates=replicates,
        seed=seed,
        algorithm="MT19937",
        decision_time_step=decision_time_step,
        max_decision_time=max_decision_time,
        captured_mass=np.asarray(accepted_mass),
        parameter_mode="Laplace_draw",
        parameter_draw_id=np.arange(replicates, dtype=np.int64),
        free_parameter_draws=np.asarray(accepted_parameters),
        draw_diagnostics=diagnostics,
    )


def _simulate_ddm_with_rng(
    trialwise: TrialwiseDDM,
    replicates: int,
    rng: np.random.Generator,
    decision_time_step: float,
    max_decision_time: float,
) -> tuple:
    """Internal simulator that permits one shared random stream across draws."""

    w = np.asarray(trialwise.w, dtype=float)
    boundary = np.asarray(trialwise.a, dtype=float)
    drift = np.asarray(trialwise.v, dtype=float)
    nondecision = np.asarray(trialwise.Ter, dtype=float)
    _validate_trialwise_support(trialwise)
    grid = np.arange(
        decision_time_step,
        max_decision_time + decision_time_step * 0.5,
        decision_time_step,
    )
    trial_count = w.size
    grid_count = grid.size
    rt = np.full((replicates, trial_count), np.nan)
    choice = np.full((replicates, trial_count), np.nan)
    captured_mass = np.full(trial_count, np.nan)

    for trial in range(trial_count):
        choice_one_density = np.asarray(
            wfpt_density(grid, -drift[trial], boundary[trial], 1.0 - w[trial])
        )
        choice_zero_density = np.asarray(
            wfpt_density(grid, drift[trial], boundary[trial], w[trial])
        )
        weights = np.concatenate((choice_zero_density[::-1], choice_one_density))
        if np.any(~np.isfinite(weights)) or np.any(weights < 0) or np.sum(weights) <= 0:
            raise FloatingPointError(
                "WFPT simulation weights are invalid on trial %d." % (trial + 1)
            )
        captured_mass[trial] = np.sum(weights) * decision_time_step
        positive_indices = np.flatnonzero(weights > 0)
        positive_weights = weights[positive_indices]
        cumulative = np.cumsum(positive_weights) / np.sum(positive_weights)
        cumulative[-1] = 1.0
        positive_bins = np.searchsorted(
            cumulative, rng.random(replicates), side="left"
        )
        sampled_indices = positive_indices[positive_bins]
        choice_one = sampled_indices >= grid_count
        decision_time = np.full(replicates, np.nan)
        decision_time[choice_one] = grid[sampled_indices[choice_one] - grid_count]
        zero_indices = sampled_indices[~choice_one]
        decision_time[~choice_one] = grid[grid_count - zero_indices - 1]
        choice[:, trial] = choice_one.astype(float)
        rt[:, trial] = decision_time + nondecision[trial]

    return rt, choice, captured_mass


def _laplace_fallback_reason(
    fit: TapasMapFit,
    dimension: int,
    policy: PosteriorDrawPolicy,
) -> Optional[str]:
    if fit.laplace is None:
        return "laplace_not_computed"
    if fit.laplace.used_bfgs_fallback:
        return "numerical_hessian_fallback_used"
    if not np.isfinite(fit.laplace.numerical_minimum_eigenvalue):
        return "non_finite_numerical_hessian"
    if fit.laplace.numerical_minimum_eigenvalue <= 0:
        return "non_positive_definite_numerical_hessian"
    if (
        not np.isfinite(fit.laplace.numerical_condition_number)
        or fit.laplace.numerical_condition_number > policy.max_condition_number
    ):
        return "numerical_hessian_condition_exceeded"
    covariance = np.asarray(fit.laplace.covariance, dtype=float)
    if covariance.shape != (dimension, dimension):
        return "covariance_dimension_mismatch"
    if np.any(~np.isfinite(covariance)):
        return "non_finite_covariance"
    if np.min(np.linalg.eigvalsh((covariance + covariance.T) / 2.0)) <= 0:
        return "non_positive_definite_covariance"
    return None


def _map_fallback_batch(
    trialwise: TrialwiseDDM,
    map_parameters: FloatArray,
    replicates: int,
    seed: int,
    decision_time_step: float,
    max_decision_time: float,
    policy: PosteriorDrawPolicy,
    reason: str,
    condition: float,
    proposal_count: int,
    rejected_draws: int,
) -> SimulationBatch:
    rng = np.random.Generator(np.random.MT19937(seed))
    rt, choice, captured_mass = _simulate_ddm_with_rng(
        trialwise,
        replicates,
        rng,
        decision_time_step,
        max_decision_time,
    )
    rejection_rate = (
        rejected_draws / proposal_count if proposal_count else 0.0
    )
    diagnostics = PosteriorDrawDiagnostics(
        requested_mode="Laplace_draw",
        used_mode="MAP_fixed",
        fallback_reason=reason,
        proposal_count=proposal_count,
        rejected_draws=rejected_draws,
        rejection_rate=rejection_rate,
        numerical_condition_number=condition,
        max_condition_number=policy.max_condition_number,
        max_rejection_rate=policy.max_rejection_rate,
    )
    return SimulationBatch(
        rt=rt,
        choice=choice,
        replicates=replicates,
        seed=seed,
        algorithm="MT19937",
        decision_time_step=decision_time_step,
        max_decision_time=max_decision_time,
        captured_mass=captured_mass,
        parameter_mode="MAP_fixed",
        parameter_draw_id=np.zeros(replicates, dtype=np.int64),
        free_parameter_draws=np.repeat(map_parameters[None, :], replicates, axis=0),
        draw_diagnostics=diagnostics,
    )


def _validate_trialwise_support(trialwise: TrialwiseDDM) -> None:
    arrays = (
        np.asarray(trialwise.w, dtype=float),
        np.asarray(trialwise.a, dtype=float),
        np.asarray(trialwise.v, dtype=float),
        np.asarray(trialwise.Ter, dtype=float),
    )
    if any(array.ndim != 1 or array.size == 0 for array in arrays):
        raise ValueError("Trial-wise DDM parameters must be non-empty vectors.")
    if not all(array.shape == arrays[0].shape for array in arrays):
        raise ValueError("Trial-wise DDM parameter arrays must share one shape.")
    w, boundary, drift, nondecision = arrays
    if np.any(~np.isfinite(w)) or np.any((w <= 0) | (w >= 1)):
        raise ValueError("Starting points must lie strictly inside (0,1).")
    if np.any(~np.isfinite(boundary)) or np.any(boundary <= 0):
        raise ValueError("Boundary separations must be finite and positive.")
    if np.any(~np.isfinite(drift)):
        raise ValueError("Drifts must be finite.")
    if np.any(~np.isfinite(nondecision)) or np.any(nondecision <= 0):
        raise ValueError("Non-decision times must be finite and positive.")


def _validate_deadline(max_decision_time: float) -> None:
    if max_decision_time != RESPONSE_DEADLINE_SECONDS:
        raise ValueError(
            "The dot task has a fixed 3-second response deadline; "
            "max_decision_time must equal 3.0."
        )


def make_sequential_spec(audit: pd.DataFrame) -> PPCSpec:
    """Port of the pre-outcome window definitions in ``pam_dot_task_ppc_spec``."""

    _validate_audit_design(audit)
    windows = []
    test_indices = np.flatnonzero(audit["phase"].to_numpy() == "test")
    for block in range(10):
        take = test_indices[block * 28 : (block + 1) * 28]
        windows.append(PPCWindow("test_global_%02d" % (block + 1), "test_global", take))

    cue_values = audit["cue_white"].to_numpy()
    phase = audit["phase"].to_numpy()
    for cue_value, cue_name in ((1.0, "white"), (0.0, "red")):
        cue_indices = np.flatnonzero((phase == "test") & (cue_values == cue_value))
        if cue_indices.size != 140:
            raise ValueError(
                "Expected 140 test trials for cue %s, found %d."
                % (cue_name, cue_indices.size)
            )
        for block in range(10):
            take = cue_indices[block * 14 : (block + 1) * 14]
            windows.append(
                PPCWindow(
                    "test_%s_%02d" % (cue_name, block + 1),
                    "test_cue_presentation",
                    take,
                )
            )

    coherence = np.round(np.abs(audit["signed_coherence"].to_numpy()), 10)
    levels = np.array([0.0, 0.1, 0.2, 0.3])
    observed_levels = np.unique(coherence[phase == "test"])
    if not np.array_equal(observed_levels, levels):
        raise ValueError("Expected test |signed coherence| levels [0, 0.1, 0.2, 0.3].")
    stimulus = audit["stimulus_category"].to_numpy()
    for level in levels:
        for stimulus_value in (0.0, 1.0):
            for cue_value, cue_name in ((0.0, "red"), (1.0, "white")):
                take = np.flatnonzero(
                    (phase == "test")
                    & (coherence == level)
                    & (stimulus == stimulus_value)
                    & (cue_values == cue_value)
                )
                if take.size:
                    windows.append(
                        PPCWindow(
                            "coh_%.1f_stim%d_%s"
                            % (level, int(stimulus_value), cue_name),
                            "test_coherence",
                            take,
                        )
                    )

    learning_indices = np.flatnonzero(phase == "learning")
    for block in range(5):
        take = learning_indices[block * 20 : (block + 1) * 20]
        windows.append(
            PPCWindow(
                "learning_global_%02d" % (block + 1),
                "learning_conditional_holdout",
                take,
            )
        )
    return PPCSpec(version="1.0.0", windows=tuple(windows))


def make_aggregate_spec(audit: pd.DataFrame) -> PPCSpec:
    """Pre-outcome aggregate test-phase views using the same simulation batch."""

    _validate_audit_design(audit)
    phase = audit["phase"].to_numpy()
    cue = audit["cue_white"].to_numpy()
    coherence = np.round(np.abs(audit["signed_coherence"].to_numpy()), 10)
    windows = [PPCWindow("test_all", "test_aggregate", np.flatnonzero(phase == "test"))]
    for cue_value, label in ((1.0, "white"), (0.0, "red")):
        windows.append(
            PPCWindow(
                "test_cue_%s" % label,
                "test_aggregate_cue",
                np.flatnonzero((phase == "test") & (cue == cue_value)),
            )
        )
    for level in (0.0, 0.1, 0.2, 0.3):
        windows.append(
            PPCWindow(
                "test_coherence_%.1f" % level,
                "test_aggregate_coherence",
                np.flatnonzero((phase == "test") & (coherence == level)),
            )
        )
    return PPCSpec(version="aggregate-1.0.0", windows=tuple(windows))


def sequential_ppc(
    audit: pd.DataFrame,
    simulation: SimulationBatch,
    spec: PPCSpec = None,
) -> PPCResult:
    """Reuse one prediction batch across all time-resolved windows."""

    active_spec = make_sequential_spec(audit) if spec is None else spec
    return _evaluate_ppc(audit, simulation, active_spec)


def aggregate_ppc(
    audit: pd.DataFrame,
    simulation: SimulationBatch,
    spec: PPCSpec = None,
) -> PPCResult:
    """Evaluate aggregate views without generating another response batch."""

    active_spec = make_aggregate_spec(audit) if spec is None else spec
    return _evaluate_ppc(audit, simulation, active_spec)


def _evaluate_ppc(
    audit: pd.DataFrame, simulation: SimulationBatch, spec: PPCSpec
) -> PPCResult:
    required = {"rt_seconds_raw", "choice_white", "stimulus_category"}
    if not required.issubset(audit.columns):
        raise ValueError("Audit table lacks observed response columns.")
    if simulation.rt.shape != simulation.choice.shape or simulation.rt.shape[1] != len(audit):
        raise ValueError("Simulation arrays must be replicate-by-trial and match audit rows.")
    replicate_count = simulation.rt.shape[0]
    window_count = len(spec.windows)
    statistic_count = len(spec.statistics)
    observed = np.full((window_count, statistic_count), np.nan)
    replicated = np.full((replicate_count, window_count, statistic_count), np.nan)
    valid_count = np.zeros(window_count, dtype=int)

    raw_rt = audit["rt_seconds_raw"].to_numpy(dtype=float)
    raw_choice = audit["choice_white"].to_numpy(dtype=float)
    stimulus = audit["stimulus_category"].to_numpy(dtype=float)
    physical_valid = (
        np.isfinite(raw_rt)
        & (raw_rt >= 0.15)
        & (raw_rt <= 3.0)
        & np.isfinite(raw_choice)
    )
    for window_index, window in enumerate(spec.windows):
        indices = np.asarray(window.indices, dtype=int)
        indices = indices[physical_valid[indices]]
        valid_count[window_index] = indices.size
        if indices.size < spec.min_valid_trials:
            continue
        observed[window_index] = response_statistics(
            raw_rt[indices], raw_choice[indices], stimulus[indices]
        )
        for replicate in range(replicate_count):
            replicated[replicate, window_index] = response_statistics(
                simulation.rt[replicate, indices],
                simulation.choice[replicate, indices],
                stimulus[indices],
            )

    row_count = window_count * statistic_count
    replicate_matrix = replicated.reshape(replicate_count, row_count)
    observed_vector = observed.reshape(row_count)
    center = np.full(row_count, np.nan)
    scale = np.full(row_count, np.nan)
    predictive_median = np.full(row_count, np.nan)
    predictive_lower = np.full(row_count, np.nan)
    predictive_upper = np.full(row_count, np.nan)
    predictive_percentile = np.full(row_count, np.nan)
    tail_probability = np.full(row_count, np.nan)

    for row in range(row_count):
        draws = replicate_matrix[:, row]
        draws = draws[np.isfinite(draws)]
        if draws.size == 0 or not np.isfinite(observed_vector[row]):
            continue
        predictive_median[row] = empirical_quantile(draws, 0.5)
        predictive_lower[row] = empirical_quantile(draws, spec.point_interval[0])
        predictive_upper[row] = empirical_quantile(draws, spec.point_interval[1])
        predictive_percentile[row] = np.mean(draws <= observed_vector[row])
        tail_probability[row] = min(
            1.0,
            2.0
            * min(predictive_percentile[row], 1.0 - predictive_percentile[row]),
        )
        center[row] = predictive_median[row]
        scale[row] = np.std(draws, ddof=1) if draws.size > 1 else np.nan

    usable = (
        np.isfinite(observed_vector)
        & np.isfinite(center)
        & np.isfinite(scale)
        & (scale > 0)
    )
    if not np.any(usable):
        raise ValueError("No window statistic has finite non-zero predictive variation.")
    observed_z = np.full(row_count, np.nan)
    observed_z[usable] = (observed_vector[usable] - center[usable]) / scale[usable]
    replicate_z = np.full_like(replicate_matrix, np.nan)
    replicate_z[:, usable] = (
        replicate_matrix[:, usable] - center[usable]
    ) / scale[usable]
    global_replicated = _row_max_abs(replicate_z[:, usable])
    global_observed = float(np.max(np.abs(observed_z[usable])))
    global_tail_probability = float(np.mean(global_replicated >= global_observed))
    simultaneous_threshold = empirical_quantile(
        global_replicated, spec.simultaneous_level
    )
    outside_simultaneous = np.abs(observed_z) > simultaneous_threshold

    window_ids = np.repeat([window.identifier for window in spec.windows], statistic_count)
    families = np.repeat([window.family for window in spec.windows], statistic_count)
    statistics = np.tile(np.asarray(spec.statistics), window_count)
    summary = pd.DataFrame(
        {
            "window_id": window_ids,
            "family": families,
            "statistic": statistics,
            "valid_trials": np.repeat(valid_count, statistic_count),
            "observed_value": observed_vector,
            "predictive_median": predictive_median,
            "predictive_lower": predictive_lower,
            "predictive_upper": predictive_upper,
            "predictive_percentile": predictive_percentile,
            "tail_probability_two_sided": tail_probability,
            "observed_z": observed_z,
            "outside_simultaneous": outside_simultaneous,
        }
    )
    return PPCResult(
        spec=spec,
        summary=summary,
        replicated_statistics=replicated,
        global_observed=global_observed,
        global_replicated=global_replicated,
        global_tail_probability=global_tail_probability,
        simultaneous_threshold=simultaneous_threshold,
        replicates=replicate_count,
    )


def response_statistics(
    rt: FloatArray, choice: FloatArray, stimulus: FloatArray
) -> FloatArray:
    valid = np.isfinite(rt) & np.isfinite(choice) & np.isfinite(stimulus)
    rt = np.asarray(rt)[valid]
    choice = np.asarray(choice)[valid]
    stimulus = np.asarray(stimulus)[valid]
    if rt.size == 0:
        return np.full(len(STATISTICS), np.nan)
    return np.array(
        [
            np.mean(choice),
            np.mean(choice == stimulus),
            empirical_quantile(rt, 0.10),
            empirical_quantile(rt, 0.50),
            empirical_quantile(rt, 0.90),
            _conditional_median(rt, choice, 0.0),
            _conditional_median(rt, choice, 1.0),
        ]
    )


def empirical_quantile(values: FloatArray, probability: float) -> float:
    finite = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if finite.size == 0:
        return np.nan
    if finite.size == 1:
        return float(finite[0])
    position = (finite.size - 1) * probability
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    weight = position - lower
    return float(finite[lower] * (1.0 - weight) + finite[upper] * weight)


def _conditional_median(rt: FloatArray, choice: FloatArray, target: float) -> float:
    selected = rt[choice == target]
    return empirical_quantile(selected, 0.5) if selected.size else np.nan


def _row_max_abs(values: FloatArray) -> FloatArray:
    maxima = []
    for row in values:
        finite = np.abs(row[np.isfinite(row)])
        if finite.size:
            maxima.append(np.max(finite))
    return np.asarray(maxima, dtype=float)


def _validate_audit_design(audit: pd.DataFrame) -> None:
    required = {"trial", "phase", "cue_white", "signed_coherence", "stimulus_category"}
    if not required.issubset(audit.columns):
        raise ValueError("Audit table is missing PPC design columns.")
    if len(audit) != 380 or not np.array_equal(
        audit["trial"].to_numpy(), np.arange(1, 381)
    ):
        raise ValueError("Sequential PPC requires the original 380-row trial index.")
