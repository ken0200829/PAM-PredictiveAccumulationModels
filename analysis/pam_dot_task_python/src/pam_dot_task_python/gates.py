"""Frozen Gate-PPC and parameter-recovery decisions.

This module exists so that the decisions the integration plan requires to be
fixed *before* results are inspected are stored as versioned, hashable data
rather than as defaults scattered through call sites.  Nothing here reads a
fit, a simulation, or a PPC summary; every value is a declaration.

Two separate freezes live here.

``GATE_PPC_V1``
    Integration plan section "Gate PPC": window definitions, generation size,
    seeds, the Laplace-versus-MAP adoption rule, the reported statistics, the
    pointwise tail definition, the global discrepancy standardization, the
    simultaneous band level, and the quantitative "systematic deviation"
    criterion used to decide whether the Gate-B coherence extension stays in
    the final model set.

``RECOVERY_GRID_V1`` / ``RECOVERY_CRITERIA_V1``
    Integration plan section 12.2: the declared generating parameter sets,
    their seeds, and the pass criteria applied to the recovery summary.

Both are hashed by :func:`freeze_digest` so a run manifest can record exactly
which declaration produced a result.
"""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Dict, Optional, Tuple

import json

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .ppc import PPCResult, PPCSpec, STATISTICS
from .recovery import RecoveryResult


FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Gate PPC
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GatePPCFreeze:
    """Pre-outcome Gate-PPC declaration.

    Threshold rationale
    -------------------
    ``max_condition_number = 1e8``
        The Laplace covariance is obtained by inverting the numerical Hessian
        and is then factorized to draw parameters.  In IEEE double precision
        the relative round-off amplification of that inversion is bounded by
        ``kappa * eps`` with ``eps = 2.22e-16``.  At ``kappa = 1e8`` this is
        ``2.2e-8``, i.e. round-off still leaves roughly eight significant
        digits in the covariance, which is far below any tolerance that
        matters for drawing parameters.  The threshold therefore rejects
        matrices whose inversion is numerically meaningless while remaining
        loose enough not to reject merely ill-scaled but usable Hessians.

        This criterion bounds *round-off* only.  It does not bound the
        finite-difference error of the Ridders Hessian itself, which is
        recorded separately as a diagnostic (see
        ``record_hessian_relative_error``) because the two error sources are
        independent and the finite-difference term is the larger one in
        practice.

    ``max_rejection_rate = 0.20``
        A proposal is rejected when the Gaussian Laplace approximation places
        a draw outside the model's valid support.  If a fraction ``r`` of the
        approximating Gaussian's mass lies outside that support, then the
        total variation distance between the Laplace approximation and any
        distribution actually supported on the valid region is at least ``r``,
        and renormalizing the truncated Gaussian inflates its density by
        ``1/(1-r)``.  Capping ``r`` at 0.20 caps that density distortion at
        25%.  The specific value is a declared convention, not a derivation;
        what the plan requires is that it is fixed before results are seen.

    ``systematic_deviation_subject_fraction = 0.20``
        Under a correctly specified model the observed global discrepancy
        exceeds its own 95% simultaneous threshold for about 5% of subjects by
        construction.  With 37 subjects that is an expected 1.85 flagged
        subjects.  Requiring more than 20% of subjects (at least 8 of 37) to
        be flagged puts the decision boundary at four times the nominal rate;
        under the null the binomial probability of reaching it is below 1e-3.
        This gives a clear signal-to-noise margin for the decision described
        in plan section 8.2.
    """

    version: str = "gate-ppc-1.0.0"

    # Window definitions (plan 12.3.2).  Referenced by spec version; the
    # structural digest of the realized windows is recorded per subject.
    sequential_spec_version: str = "1.0.0"
    aggregate_spec_version: str = "aggregate-1.0.0"
    sequential_window_count: int = 49
    aggregate_window_count: int = 7
    min_valid_trials: int = 5

    # Generation (plan 12.3.1 and 12.3.3).
    replicates: int = 2000
    simulation_seed: int = 20260720
    rng_algorithm: str = "MT19937"
    decision_time_step: float = 0.001
    response_deadline_seconds: float = 3.0

    # Laplace adoption and MAP fallback (plan 12.3.3).
    max_condition_number: float = 1e8
    max_rejection_rate: float = 0.20
    require_positive_definite_numerical_hessian: bool = True
    reject_bfgs_fallback_hessian: bool = True
    record_hessian_relative_error: bool = True

    # Reported statistics and discrepancy definitions (plan 12.3.3).
    statistics: Tuple[str, ...] = STATISTICS
    point_interval: Tuple[float, float] = (0.025, 0.975)
    tail_probability_definition: str = "two_sided_2x_min_percentile"
    global_discrepancy: str = "max_absolute_standardized_deviation"
    standardization: str = "center_predictive_median_scale_predictive_sd"
    simultaneous_level: float = 0.95

    # Model-adoption rule (plan 8.2 and 12.3.5).
    systematic_deviation_window_family: str = "test_coherence"
    systematic_deviation_subject_fraction: float = 0.20

    def __post_init__(self) -> None:
        if not 0 < self.simultaneous_level < 1:
            raise ValueError("simultaneous_level must lie strictly in (0,1).")
        if not 0 <= self.max_rejection_rate < 1:
            raise ValueError("max_rejection_rate must lie in [0,1).")
        if self.max_condition_number <= 1:
            raise ValueError("max_condition_number must exceed one.")
        if self.replicates <= 0:
            raise ValueError("replicates must be positive.")
        if self.response_deadline_seconds != 3.0:
            raise ValueError("The dot task response deadline is fixed at 3 seconds.")
        if not 0 < self.systematic_deviation_subject_fraction < 1:
            raise ValueError("systematic_deviation_subject_fraction must be in (0,1).")

    def draw_policy(self):
        """Return the :class:`PosteriorDrawPolicy` implied by this freeze."""

        from .ppc import PosteriorDrawPolicy

        return PosteriorDrawPolicy(
            max_condition_number=self.max_condition_number,
            max_rejection_rate=self.max_rejection_rate,
        )

    def validate_spec(self, spec: PPCSpec, kind: str = "sequential") -> None:
        """Fail if a realized PPC spec departs from the frozen declaration."""

        if kind == "sequential":
            expected_version = self.sequential_spec_version
            expected_count = self.sequential_window_count
        elif kind == "aggregate":
            expected_version = self.aggregate_spec_version
            expected_count = self.aggregate_window_count
        else:
            raise ValueError("kind must be 'sequential' or 'aggregate'.")
        if spec.version != expected_version:
            raise ValueError(
                "PPC spec version %r does not match the frozen %s version %r."
                % (spec.version, kind, expected_version)
            )
        if len(spec.windows) != expected_count:
            raise ValueError(
                "Frozen %s freeze declares %d windows; spec realized %d."
                % (kind, expected_count, len(spec.windows))
            )
        if tuple(spec.statistics) != tuple(self.statistics):
            raise ValueError("PPC statistics differ from the frozen declaration.")
        if tuple(spec.point_interval) != tuple(self.point_interval):
            raise ValueError("PPC point interval differs from the frozen declaration.")
        if spec.simultaneous_level != self.simultaneous_level:
            raise ValueError("Simultaneous level differs from the frozen declaration.")
        if spec.global_discrepancy != self.global_discrepancy:
            raise ValueError("Global discrepancy differs from the frozen declaration.")
        if spec.min_valid_trials != self.min_valid_trials:
            raise ValueError("min_valid_trials differs from the frozen declaration.")


GATE_PPC_V1 = GatePPCFreeze()


def window_structure_digest(spec: PPCSpec) -> str:
    """Hash a realized spec's window structure, excluding subject-specific rows."""

    payload = [
        {
            "identifier": window.identifier,
            "family": window.family,
            "size": int(np.asarray(window.indices).size),
        }
        for window in spec.windows
    ]
    return _digest(
        {
            "version": spec.version,
            "statistics": list(spec.statistics),
            "windows": payload,
        }
    )


def subject_systematic_deviation(
    result: PPCResult,
    freeze: GatePPCFreeze = GATE_PPC_V1,
    family: Optional[str] = None,
) -> Dict[str, float]:
    """Apply the frozen per-subject systematic-deviation rule.

    With ``family=None`` the rule uses the whole predeclared window set, which
    is the plan's global discrepancy.  Passing a family restricts the maximum
    standardized deviation to that family, which is what plan section 8.2 uses
    to ask whether the official binary DDM misfits coherence specifically.
    """

    summary = result.summary
    if family is None:
        observed = result.global_observed
        threshold = result.simultaneous_threshold
        tail = result.global_tail_probability
    else:
        selected = summary[summary["family"] == family]
        if selected.empty:
            raise ValueError("No window belongs to family %r." % family)
        values = selected["observed_z"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("Family %r has no finite standardized deviation." % family)
        observed = float(np.max(np.abs(values)))
        family_mask = (
            np.repeat(
                [window.family for window in result.spec.windows],
                len(result.spec.statistics),
            )
            == family
        )
        replicate_z = _replicate_standardized(result)[:, family_mask]
        family_replicated = _row_max_abs(replicate_z)
        if family_replicated.size == 0:
            raise ValueError("Family %r has no replicated discrepancy." % family)
        threshold = float(
            np.quantile(family_replicated, freeze.simultaneous_level)
        )
        tail = float(np.mean(family_replicated >= observed))
    return {
        "family": "all" if family is None else family,
        "observed_discrepancy": float(observed),
        "simultaneous_threshold": float(threshold),
        "tail_probability": float(tail),
        "flagged": bool(observed > threshold),
    }


def group_systematic_deviation(
    flags: Tuple[bool, ...], freeze: GatePPCFreeze = GATE_PPC_V1
) -> Dict[str, float]:
    """Apply the frozen across-subject rule to per-subject flags."""

    values = np.asarray(flags, dtype=bool)
    if values.size == 0:
        raise ValueError("Group rule requires at least one subject flag.")
    fraction = float(np.mean(values))
    return {
        "subjects": int(values.size),
        "flagged_subjects": int(np.sum(values)),
        "flagged_fraction": fraction,
        "threshold_fraction": freeze.systematic_deviation_subject_fraction,
        "systematic_deviation": bool(
            fraction > freeze.systematic_deviation_subject_fraction
        ),
    }


# ---------------------------------------------------------------------------
# Parameter recovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryCriteria:
    """Pass criteria applied to a recovery summary.

    ``max_rmse_over_prior_sd = 1.0`` is the substantive bar: the prior standard
    deviation is what an estimator that ignores the data would achieve, so a
    parameter whose recovery RMSE reaches the prior SD has not been identified
    by the data at all.  The correlation and bias bounds are conventional
    recovery reporting thresholds and are declared, not derived.
    """

    version: str = "recovery-criteria-1.0.0"
    min_correlation: float = 0.70
    max_absolute_bias: float = 0.50
    max_rmse_over_prior_sd: float = 1.0
    min_cases_per_parameter: int = 20

    def __post_init__(self) -> None:
        if not -1 <= self.min_correlation <= 1:
            raise ValueError("min_correlation must lie in [-1,1].")
        if self.max_absolute_bias <= 0:
            raise ValueError("max_absolute_bias must be positive.")
        if self.max_rmse_over_prior_sd <= 0:
            raise ValueError("max_rmse_over_prior_sd must be positive.")


RECOVERY_CRITERIA_V1 = RecoveryCriteria()


@dataclass(frozen=True)
class RecoveryGrid:
    """Declared generating parameter sets for one model.

    The ``hgf.omega_2`` direction is only identified when at least one
    belief-to-DDM slope is non-zero.  At the prior mean every slope is zero, so
    the response likelihood is exactly independent of the HGF and the gradient
    in the ``omega_2`` direction vanishes.  Every generating set below
    therefore uses a non-zero slope; a grid containing slope-zero cases would
    report a spurious recovery failure for ``omega_2`` that reflects the
    model's structure rather than the estimator.
    """

    version: str
    model_id: str
    parameter_names: Tuple[str, ...]
    truths: Tuple[Tuple[float, ...], ...]
    seeds: Tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.truths) != len(self.seeds):
            raise ValueError("Each generating set needs exactly one seed.")
        if not self.truths:
            raise ValueError("A recovery grid needs at least one generating set.")
        width = len(self.parameter_names)
        for row in self.truths:
            if len(row) != width:
                raise ValueError("Generating sets must match the parameter names.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Recovery seeds must be unique.")

    @property
    def truth_array(self) -> FloatArray:
        return np.asarray(self.truths, dtype=float)


def _ddm_v_grid() -> RecoveryGrid:
    """24-case declared grid for the official ``ddm_v`` reduction.

    Ranges are centred on the prior means with spreads inside one prior
    standard deviation, except ``b_v`` which is deliberately kept away from
    zero for the identifiability reason documented on :class:`RecoveryGrid`.
    """

    omega_values = (-4.0, -3.0, -2.0)
    slope_values = (-2.0, -1.0, 1.0, 2.0)
    log_a_a, log_a_v, ter = 0.6, 0.2, 1.9
    truths = []
    seeds = []
    seed = 20260721
    for omega in omega_values:
        for slope in slope_values:
            truths.append((omega, log_a_a, log_a_v, slope, ter))
            seeds.append(seed)
            seed += 1
    # Two extra spreads in the nuisance directions so that log_a_a, log_a_v and
    # Ter_logit also vary across cases; a constant column would make their
    # across-case correlation undefined.
    for index, (delta_a, delta_v, delta_ter) in enumerate(
        ((-0.4, 0.5, -0.5), (0.5, -0.4, 0.6))
    ):
        for omega in omega_values:
            for slope in (-1.5, 1.5):
                truths.append(
                    (
                        omega,
                        log_a_a + delta_a,
                        log_a_v + delta_v,
                        slope,
                        ter + delta_ter,
                    )
                )
                seeds.append(seed)
                seed += 1
    return RecoveryGrid(
        version="recovery-grid-ddm_v-1.0.0",
        model_id="ddm_v",
        parameter_names=(
            "hgf.omega_2",
            "ddm.log_a_a",
            "ddm.log_a_v",
            "ddm.b_v",
            "ddm.Ter_logit",
        ),
        truths=tuple(truths),
        seeds=tuple(seeds),
    )


RECOVERY_GRID_V1 = _ddm_v_grid()


def _ddm_v_grid_v2() -> RecoveryGrid:
    """24-case ``ddm_v`` grid relocated to the Bayes-optimal operating range.

    V1 centred ``omega_2`` on the hardcoded default of -3 and spanned only
    {-4, -3, -2}. The per-subject Bayes-optimal fits in
    ``priors/bayes_optimal_omega.json`` showed the actual operating range is
    ``[-5.46, -3.15]`` with mean -4.44; not one of the 37 subjects reached -3.
    V1's truths therefore barely overlapped where the model really lives, and
    its ``omega_2`` correlation of 0.30 was partly an artifact of a truth range
    sitting off to one side.

    V2 fixes three things measured on the V1 run:

    - ``omega_2`` spans ``[-5.6, -3.1]`` at six levels (SD 0.87), covering the
      observed Bayes-optimal range instead of sitting above it.
    - The nuisance columns ``log_a_a``, ``log_a_v`` and ``Ter_logit`` are
      balanced 8/8/8 and assigned so every pairwise truth correlation is below
      0.09 in absolute value; V1's cycled offsets left them mutually
      correlated at -0.5, which biases their individual recovery statistics.
    - ``Ter_logit`` is capped at 2.0. On the designated subject (minimum RT
      0.455 s) ``Ter_logit = 2.5`` leaves only 3.5% of the minimum RT for the
      decision and produced the six non-finite failures in the V1 run; every
      V2 case stays in a region where the response likelihood is finite.

    ``b_v`` remains non-zero throughout for the identifiability reason on
    :class:`RecoveryGrid`. The literal assignment below was produced by an
    orthogonalizing search and is frozen here so the grid carries no runtime
    seed dependence.
    """

    return RecoveryGrid(
        version="recovery-grid-ddm_v-2.0.0",
        model_id="ddm_v",
        parameter_names=(
            "hgf.omega_2",
            "ddm.log_a_a",
            "ddm.log_a_v",
            "ddm.b_v",
            "ddm.Ter_logit",
        ),
        truths=(
            (-5.6, 0.6, 0.2, -2.0, 2.0),
            (-5.6, 0.4, 0.4, -1.2, 1.8),
            (-5.6, 0.4, 0.0, 1.2, 1.6),
            (-5.6, 0.6, 0.0, 2.0, 2.0),
            (-5.1, 0.8, 0.2, -2.0, 1.8),
            (-5.1, 0.8, 0.4, -1.2, 1.6),
            (-5.1, 0.8, 0.2, 1.2, 1.6),
            (-5.1, 0.6, 0.4, 2.0, 1.6),
            (-4.6, 0.8, 0.0, -2.0, 1.8),
            (-4.6, 0.4, 0.0, -1.2, 1.6),
            (-4.6, 0.4, 0.4, 1.2, 2.0),
            (-4.6, 0.4, 0.2, 2.0, 1.8),
            (-4.1, 0.8, 0.2, -2.0, 2.0),
            (-4.1, 0.6, 0.4, -1.2, 1.8),
            (-4.1, 0.8, 0.0, 1.2, 2.0),
            (-4.1, 0.8, 0.0, 2.0, 1.8),
            (-3.6, 0.4, 0.0, -2.0, 2.0),
            (-3.6, 0.6, 0.0, -1.2, 1.6),
            (-3.6, 0.4, 0.2, 1.2, 2.0),
            (-3.6, 0.8, 0.4, 2.0, 2.0),
            (-3.1, 0.4, 0.4, -2.0, 1.8),
            (-3.1, 0.6, 0.2, -1.2, 1.6),
            (-3.1, 0.6, 0.2, 1.2, 1.8),
            (-3.1, 0.6, 0.4, 2.0, 1.6),
        ),
        seeds=tuple(range(20260751, 20260775)),
    )


RECOVERY_GRID_V2 = _ddm_v_grid_v2()


def _walsh_sign(row: int, column: int) -> float:
    """Element of a fixed order-32 Walsh/Hadamard design."""

    return -1.0 if bin(row & column).count("1") % 2 else 1.0


def _recovery_grid_w_v3() -> RecoveryGrid:
    """32-case Gate for the finalized tie-driven starting-point hypothesis.

    Four equally frequent omega levels span the Bayes-optimal operating range.
    The remaining columns are balanced two-level Walsh contrasts.  All truth
    columns are exactly uncorrelated, and the 8 unique parameter combinations
    are independently generated four times to expose response-sampling noise.
    """

    truths = []
    for row in range(32):
        h = lambda column: _walsh_sign(row, column)
        truths.append(
            (
                -4.3 + 0.9 * h(1) + 0.3 * h(2),
                0.4 + 0.2 * h(3),
                0.2 + 0.2 * h(4),
                1.2 * h(5),
                1.8 + 0.2 * h(6),
            )
        )
    return RecoveryGrid(
        version="recovery-grid-ddm_w-tie_v0-3.1.0",
        model_id="ddm_w",
        parameter_names=(
            "hgf.omega_2",
            "ddm.log_a_a",
            "ddm.log_a_v",
            "ddm.b_w",
            "ddm.Ter_logit",
        ),
        truths=tuple(truths),
        seeds=tuple(range(20260801, 20260833)),
    )


def _recovery_grid_full_c_v3() -> RecoveryGrid:
    """32-case Gate for the finalized full coherence model under tie v=0.

    The eight free parameters use mutually orthogonal Walsh contrasts.  Omega
    has four levels; every other parameter has two balanced levels.  Sixteen
    unique parameter combinations are independently generated twice.  Slope
    magnitudes stay inside DDM support on the designated real trial design.
    """

    truths = []
    for row in range(32):
        h = lambda column: _walsh_sign(row, column)
        truths.append(
            (
                -4.3 + 0.9 * h(1) + 0.3 * h(2),
                0.4 + 0.2 * h(3),
                0.2 + 0.2 * h(4),
                1.2 * h(5),
                1.2 * h(6),
                1.5 * h(7),
                1.0 * h(8),
                1.8 + 0.2 * h(9),
            )
        )
    return RecoveryGrid(
        version="recovery-grid-ddm_full_c-tie_v0-3.1.0",
        model_id="ddm_full_c",
        parameter_names=(
            "hgf.omega_2",
            "ddm.log_a_a",
            "ddm.log_a_v",
            "ddm.b_w",
            "ddm.b_a",
            "ddm.b_v",
            "ddm.b_c",
            "ddm.Ter_logit",
        ),
        truths=tuple(truths),
        seeds=tuple(range(20260851, 20260883)),
    )


RECOVERY_GRID_W_V3 = _recovery_grid_w_v3()
RECOVERY_GRID_FULL_C_V3 = _recovery_grid_full_c_v3()
FINAL_RECOVERY_GRIDS_V3 = (RECOVERY_GRID_W_V3, RECOVERY_GRID_FULL_C_V3)


def prior_standard_deviations(model) -> Dict[str, float]:
    """Return the prior SD of each free parameter of a :class:`JointModel`."""

    variances = np.concatenate(
        (
            model.hgf_prior.variances[model.hgf_prior.free_mask],
            model.response_prior.variances[model.response_prior.free_mask],
        )
    )
    return {
        name: float(np.sqrt(variance))
        for name, variance in zip(model.free_parameter_names, variances)
    }


def evaluate_recovery(
    result: RecoveryResult,
    prior_sd: Dict[str, float],
    criteria: RecoveryCriteria = RECOVERY_CRITERIA_V1,
) -> pd.DataFrame:
    """Apply the frozen criteria to a recovery summary, one row per parameter."""

    summary = result.summary.copy()
    missing = set(summary["parameter"]) - set(prior_sd)
    if missing:
        raise ValueError("Missing prior SD for %s." % sorted(missing))
    sd = summary["parameter"].map(prior_sd).to_numpy(dtype=float)
    summary["prior_sd"] = sd
    summary["rmse_over_prior_sd"] = summary["rmse"].to_numpy(dtype=float) / sd
    summary["passes_cases"] = (
        summary["cases"].to_numpy(dtype=int) >= criteria.min_cases_per_parameter
    )
    summary["passes_correlation"] = (
        summary["correlation"].to_numpy(dtype=float) >= criteria.min_correlation
    )
    summary["passes_bias"] = (
        np.abs(summary["bias"].to_numpy(dtype=float)) <= criteria.max_absolute_bias
    )
    summary["passes_rmse"] = (
        summary["rmse_over_prior_sd"].to_numpy(dtype=float)
        <= criteria.max_rmse_over_prior_sd
    )
    summary["passes"] = (
        summary["passes_cases"]
        & summary["passes_correlation"]
        & summary["passes_bias"]
        & summary["passes_rmse"]
    )
    return summary


def recovery_verdict(evaluated: pd.DataFrame) -> Dict[str, object]:
    """Reduce an evaluated recovery table to a single gate outcome."""

    passes = evaluated["passes"].to_numpy(dtype=bool)
    failed = evaluated.loc[~passes, "parameter"].tolist()
    return {
        "parameters": int(passes.size),
        "passed": int(np.sum(passes)),
        "failed_parameters": failed,
        "gate_passed": bool(np.all(passes)),
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def freeze_digest(freeze) -> str:
    """Return the SHA-256 of a frozen declaration's canonical JSON form."""

    return _digest(asdict(freeze))


def gate_manifest(
    freeze: GatePPCFreeze = GATE_PPC_V1,
    grids: Tuple[RecoveryGrid, ...] = FINAL_RECOVERY_GRIDS_V3,
    criteria: RecoveryCriteria = RECOVERY_CRITERIA_V1,
) -> Dict[str, object]:
    """Return the serializable record a run manifest must store."""

    return {
        "gate_ppc": {
            "declaration": asdict(freeze),
            "digest": freeze_digest(freeze),
        },
        "recovery_grids": [
            {"declaration": asdict(grid), "digest": freeze_digest(grid)}
            for grid in grids
        ],
        "recovery_criteria": {
            "declaration": asdict(criteria),
            "digest": freeze_digest(criteria),
        },
    }


def _digest(payload) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError("Unserializable value in a frozen declaration: %r" % type(value))


def _replicate_standardized(result: PPCResult) -> FloatArray:
    """Recompute replicate standardized deviations from a PPC result."""

    replicates = result.replicated_statistics.reshape(result.replicates, -1)
    center = result.summary["predictive_median"].to_numpy(dtype=float)
    observed_z = result.summary["observed_z"].to_numpy(dtype=float)
    observed = result.summary["observed_value"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = (observed - center) / observed_z
    standardized = np.full_like(replicates, np.nan)
    usable = np.isfinite(center) & np.isfinite(scale) & (scale > 0)
    standardized[:, usable] = (replicates[:, usable] - center[usable]) / scale[usable]
    return standardized


def _row_max_abs(values: FloatArray) -> FloatArray:
    maxima = []
    for row in values:
        finite = np.abs(row[np.isfinite(row)])
        if finite.size:
            maxima.append(np.max(finite))
    return np.asarray(maxima, dtype=float)
