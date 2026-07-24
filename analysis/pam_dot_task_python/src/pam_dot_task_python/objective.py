"""Joint HGF+DDM MAP objective in TAPAS transformed parameter space."""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from .config import (
    CueModelSpec,
    ParameterPrior,
    cue_hgf_prior,
    cue_model_spec,
    ddm_prior,
)
from .hgf import (
    CueHGFResult,
    HGFResult,
    cue_binary_hgf,
    cue_blind_binary_hgf,
    transform_ehgf_binary,
)
from .numerics import (
    LaplaceResult,
    QuasiNewtonOptions,
    QuasiNewtonResult,
    laplace_evidence,
    tapas_quasi_newton,
)
from .response import (
    CueDDMParameters,
    DDMParameters,
    TrialwiseDDM,
    cue_ddm_loglik,
    ddm_loglik,
    transform_cue_ddm,
    transform_ddm,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class JointEvaluation:
    negative_log_joint: float
    negative_log_likelihood: float
    log_prior: float
    hgf: Union[CueHGFResult, HGFResult]
    ddm_parameters: Union[DDMParameters, CueDDMParameters]
    trialwise: TrialwiseDDM
    trial_log_likelihood: FloatArray


@dataclass(frozen=True)
class TerDiagnostic:
    """Audit the response-time support used by the Ter transformation."""

    minimum_fitted_rt: float
    ter: float
    ter_fraction_of_minimum_rt: float
    decision_time_slack: float
    decision_time_fraction: float
    transformed_ter: float


@dataclass(frozen=True)
class ScipyMapFit:
    """Provisional SciPy MAP result; Laplace LME is deliberately not claimed."""

    optimization: OptimizeResult
    evaluation: JointEvaluation
    hgf_transformed: FloatArray
    ddm_transformed: FloatArray


@dataclass(frozen=True)
class TapasMapFit:
    """MAP result from the ported TAPAS optimizer and optional Laplace step."""

    optimization: QuasiNewtonResult
    evaluation: JointEvaluation
    hgf_transformed: FloatArray
    ddm_transformed: FloatArray
    laplace: Optional[LaplaceResult]
    aic: float
    bic: float


@dataclass
class JointModel:
    """Joint eHGF+DDM objective for legacy and cue-locus model IDs."""

    u: FloatArray
    y: FloatArray
    model_id: str
    tie: Optional[NDArray[np.bool_]] = None
    cue_evidence: Optional[FloatArray] = None
    hgf_prior: ParameterPrior = field(default_factory=cue_hgf_prior)
    hgf_cache_size: int = 0
    response_prior: ParameterPrior = field(init=False)
    cue_spec: Optional[CueModelSpec] = field(init=False)
    _hgf_cache: OrderedDict = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _hgf_cache_hits: int = field(default=0, init=False, repr=False)
    _hgf_cache_misses: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.hgf_cache_size, int) or self.hgf_cache_size < 0:
            raise ValueError("HGF cache size must be a non-negative integer.")
        self.u = np.asarray(self.u, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.u.ndim != 2 or self.u.shape[1] < 3:
            raise ValueError("Joint dot-task model requires three input columns.")
        if self.y.shape != (self.u.shape[0], 2):
            raise ValueError("Responses must be N x 2 and match the inputs.")
        if np.any(~np.isfinite(self.u)):
            raise ValueError("All stimulus inputs must remain finite.")
        # Tie trials carry no objective category (plan section 5.2.1).  Default
        # to deriving them from signed coherence so existing callers keep
        # working, but let SubjectData pass its own mask explicitly.
        if self.tie is None:
            self.tie = self.u[:, 2] == 0.0
        else:
            self.tie = np.asarray(self.tie, dtype=bool)
            if self.tie.shape != (self.u.shape[0],):
                raise ValueError("Tie mask must match the trial count.")
        self.model_id = self.model_id.lower()
        self.cue_spec = cue_model_spec(self.model_id)
        if self.cue_spec is not None:
            if self.cue_evidence is None:
                raise ValueError("Cue-locus models require signed cue evidence.")
            self.cue_evidence = np.asarray(self.cue_evidence, dtype=float)
            if self.cue_evidence.shape != (self.u.shape[0],):
                raise ValueError("Cue evidence must match the trial count.")
            if np.any(~np.isfinite(self.cue_evidence)) or np.any(
                ~np.isin(self.cue_evidence, (-1.0, 0.0, 1.0))
            ):
                raise ValueError("Cue evidence must be finite and coded -1, 0, or 1.")
            cue_white = self.u[:, 1] == 1.0
            cue_red = self.u[:, 1] == 0.0
            if np.any(self.cue_evidence[cue_white] != 0.0) or np.any(
                np.abs(self.cue_evidence[cue_red]) != 1.0
            ):
                raise ValueError("Cue evidence must be zero for white and signed for red.")
            if np.unique(self.cue_evidence[cue_red]).size != 1:
                raise ValueError("Red cue evidence must have one subject-level direction.")
        self.response_prior = ddm_prior(self.model_id)

    @property
    def initial_free_parameters(self) -> FloatArray:
        return np.concatenate(
            (
                self.hgf_prior.means[self.hgf_prior.free_mask],
                self.response_prior.means[self.response_prior.free_mask],
            )
        )

    @property
    def free_parameter_names(self) -> tuple:
        return tuple("hgf." + name for name in self.hgf_prior.free_names) + tuple(
            "ddm." + name for name in self.response_prior.free_names
        )

    def expand_free(self, free_parameters: FloatArray) -> tuple:
        values = np.asarray(free_parameters, dtype=float)
        hgf_count = int(np.sum(self.hgf_prior.free_mask))
        expected = hgf_count + int(np.sum(self.response_prior.free_mask))
        if values.shape != (expected,):
            raise ValueError("Expected %d free parameters, received %d." % (expected, values.size))
        hgf_full = self.hgf_prior.means.copy()
        ddm_full = self.response_prior.means.copy()
        hgf_full[self.hgf_prior.free_mask] = values[:hgf_count]
        ddm_full[self.response_prior.free_mask] = values[hgf_count:]
        return hgf_full, ddm_full

    def evaluate(self, free_parameters: FloatArray) -> JointEvaluation:
        hgf_transformed, ddm_transformed = self.expand_free(free_parameters)
        regular_rt = self.y[:, 0]
        hgf, prediction = self._hgf_prediction(hgf_transformed)
        if self.cue_spec is None:
            ddm_parameters = transform_ddm(ddm_transformed, regular_rt)
            trial_log_likelihood, trialwise = ddm_loglik(
                self.y,
                self.u[:, 0],
                prediction,
                ddm_parameters,
                self.u[:, 2],
                self.tie,
            )
        else:
            ddm_parameters = transform_cue_ddm(
                ddm_transformed, regular_rt, self.response_prior.names
            )
            trial_log_likelihood, trialwise = cue_ddm_loglik(
                self.y,
                self.cue_spec.architecture,
                self.u[:, 0],
                prediction,
                self.cue_evidence,
                ddm_parameters,
                self.u[:, 2],
                self.tie,
            )
        regular = np.all(np.isfinite(self.y), axis=1)
        log_likelihood = float(np.sum(trial_log_likelihood[regular]))
        if not np.isfinite(log_likelihood):
            raise FloatingPointError("The response log likelihood is non-finite.")
        log_prior = self.hgf_prior.log_density(hgf_transformed) + self.response_prior.log_density(
            ddm_transformed
        )
        negative_log_joint = -(log_likelihood + log_prior)
        if not np.isfinite(negative_log_joint):
            raise FloatingPointError("The negative log joint is non-finite.")
        return JointEvaluation(
            negative_log_joint=negative_log_joint,
            negative_log_likelihood=-log_likelihood,
            log_prior=log_prior,
            hgf=hgf,
            ddm_parameters=ddm_parameters,
            trialwise=trialwise,
            trial_log_likelihood=trial_log_likelihood,
        )

    def clear_hgf_cache(self) -> None:
        """Clear cached deterministic HGF trajectories and audit counters."""

        self._hgf_cache.clear()
        self._hgf_cache_hits = 0
        self._hgf_cache_misses = 0

    @property
    def hgf_cache_diagnostics(self) -> dict:
        return {
            "enabled": self.hgf_cache_size > 0,
            "capacity": self.hgf_cache_size,
            "entries": len(self._hgf_cache),
            "hits": self._hgf_cache_hits,
            "misses": self._hgf_cache_misses,
        }

    def _hgf_prediction(self, hgf_transformed: FloatArray):
        key = tuple(
            map(float, hgf_transformed[self.hgf_prior.free_mask])
        )
        if self.hgf_cache_size > 0 and key in self._hgf_cache:
            self._hgf_cache_hits += 1
            self._hgf_cache.move_to_end(key)
            return self._hgf_cache[key]

        self._hgf_cache_misses += 1
        hgf_parameters = transform_ehgf_binary(hgf_transformed)
        if self.cue_spec is None:
            hgf = cue_binary_hgf(self.u, hgf_parameters, self.tie)
            prediction = hgf.active.muhat[:, 0]
        elif self.cue_spec.perceptual_model == "cue_blind":
            hgf = cue_blind_binary_hgf(
                self.u[:, 0], hgf_parameters, self.tie
            )
            prediction = hgf.muhat[:, 0]
        else:
            hgf = cue_binary_hgf(self.u, hgf_parameters, self.tie)
            prediction = hgf.active.muhat[:, 0]
        payload = (hgf, prediction)
        if self.hgf_cache_size > 0:
            self._hgf_cache[key] = payload
            self._hgf_cache.move_to_end(key)
            while len(self._hgf_cache) > self.hgf_cache_size:
                self._hgf_cache.popitem(last=False)
        return payload

    def objective(self, free_parameters: FloatArray) -> float:
        """TAPAS-compatible failure behavior for numerical optimization."""

        try:
            return self.evaluate(free_parameters).negative_log_joint
        except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
            return float(np.finfo(float).max)

    def ter_diagnostic(self, free_parameters: FloatArray) -> TerDiagnostic:
        """Report the fitted Ter relative to the minimum included RT.

        ``Ter`` is parameterized as ``min(RT_valid) * sigmoid(theta_Ter)``.
        The quantities returned here make a boundary-hugging transform visible
        without imposing an outcome-dependent pass/fail threshold.
        """
        parameters = np.asarray(free_parameters, dtype=float)
        _, response_transformed = self.expand_free(parameters)
        evaluation = self.evaluate(parameters)
        fitted_rt = self.y[np.all(np.isfinite(self.y), axis=1), 0]
        minimum_rt = float(np.min(fitted_rt))
        ter = float(evaluation.ddm_parameters.Ter)
        fraction = ter / minimum_rt
        return TerDiagnostic(
            minimum_fitted_rt=minimum_rt,
            ter=ter,
            ter_fraction_of_minimum_rt=fraction,
            decision_time_slack=minimum_rt - ter,
            decision_time_fraction=1.0 - fraction,
            transformed_ter=float(response_transformed[-1]),
        )

    def fit_map(
        self,
        initial: Optional[FloatArray] = None,
        options: Optional[QuasiNewtonOptions] = None,
        compute_lme: bool = True,
    ) -> TapasMapFit:
        """Run the ported TAPAS Ridders-gradient BFGS and optional LME step."""

        start = self.initial_free_parameters if initial is None else np.asarray(initial, dtype=float)
        optimizer_options = QuasiNewtonOptions() if options is None else options
        optimization = tapas_quasi_newton(self.objective, start, optimizer_options)
        evaluation = self.evaluate(optimization.argument_minimum)
        hgf_transformed, ddm_transformed = self.expand_free(
            optimization.argument_minimum
        )
        laplace = (
            laplace_evidence(self.objective, optimization)
            if compute_lme
            else None
        )
        dimension = optimization.argument_minimum.size
        observations = int(np.sum(np.all(np.isfinite(self.y), axis=1)))
        aic = 2.0 * evaluation.negative_log_likelihood + 2.0 * dimension
        bic = (
            2.0 * evaluation.negative_log_likelihood
            + dimension * np.log(observations)
        )
        return TapasMapFit(
            optimization=optimization,
            evaluation=evaluation,
            hgf_transformed=hgf_transformed,
            ddm_transformed=ddm_transformed,
            laplace=laplace,
            aic=float(aic),
            bic=float(bic),
        )

    def fit_map_scipy(
        self,
        initial: Optional[FloatArray] = None,
        max_iterations: int = 1000,
    ) -> ScipyMapFit:
        """Run SciPy BFGS as a non-parity diagnostic comparison only.

        SciPy uses different numerical gradients and line-search behavior.
        It must not silently replace :meth:`fit_map` in reported analyses.
        """

        start = self.initial_free_parameters if initial is None else np.asarray(initial, dtype=float)
        result = minimize(
            self.objective,
            start,
            method="BFGS",
            options={"maxiter": int(max_iterations), "gtol": 1e-3},
        )
        evaluation = self.evaluate(result.x)
        hgf_transformed, ddm_transformed = self.expand_free(result.x)
        return ScipyMapFit(
            optimization=result,
            evaluation=evaluation,
            hgf_transformed=hgf_transformed,
            ddm_transformed=ddm_transformed,
        )
