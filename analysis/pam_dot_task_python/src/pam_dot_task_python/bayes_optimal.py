"""Per-subject Bayes-optimal perceptual parameters used as prior means.

The official PAM demo does not fit the perceptual model with an arbitrary
default prior. ``Examples/HGF_examples/DDM_HGF_example.m`` first fits the
perceptual model to the input sequence alone::

    bo = tapas_fitModel([], u, prc_model, 'tapas_bayes_optimal_binary_config');
    prc_model.ommu = bo.p_prc.om;

and then uses the result as the *prior mean* for the joint fit. This module
ports that step.

The observation model being ported is ``tapas_bayes_optimal_binary``, whose
trial-wise log probability is

    logp_t = u_t * log(muhat_t) + (1 - u_t) * log(1 - muhat_t)

with no observation parameters at all. It therefore scores how well an agent
with a given set of perceptual parameters predicts the stimulus sequence it
actually received. Only ``u`` enters; responses are never used, so this does
not feed behavioural data back into the prior. It calibrates the prior to the
experimental design, which is why the official workflow accepts it.

Every one of the 37 subjects received an individually randomized sequence, so
this is computed per subject and yields a different prior mean for each.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import ParameterPrior
from .hgf import binary_hgf, cue_binary_hgf, transform_ehgf_binary
from .numerics import QuasiNewtonOptions, tapas_quasi_newton


FloatArray = NDArray[np.float64]

OMEGA_PARAMETER = "omega_2"


@dataclass(frozen=True)
class BayesOptimalFit:
    """Result of one Bayes-optimal perceptual fit to an input sequence."""

    stream: str
    transformed_parameters: FloatArray
    free_parameters: FloatArray
    free_names: Tuple[str, ...]
    omega_2: float
    input_log_likelihood: float
    negative_log_joint: float
    scored_trials: int
    iterations: int
    reset_count: int
    convergence_reason: str


def bayes_optimal_log_likelihood(
    stimulus: FloatArray, muhat: FloatArray
) -> Tuple[float, FloatArray]:
    """Port ``tapas_bayes_optimal_binary``'s trial-wise log probability."""

    stimulus = np.asarray(stimulus, dtype=float)
    prediction = np.asarray(muhat, dtype=float)
    if stimulus.shape != prediction.shape:
        raise ValueError("Stimulus and prediction vectors must share one shape.")
    if np.any(~np.isfinite(prediction)) or np.any(
        (prediction <= 0.0) | (prediction >= 1.0)
    ):
        raise FloatingPointError("Predictions left the open unit interval.")
    trial_log_likelihood = stimulus * np.log(prediction) + (1.0 - stimulus) * np.log(
        1.0 - prediction
    )
    total = float(np.sum(trial_log_likelihood))
    if not np.isfinite(total):
        raise FloatingPointError("The Bayes-optimal log likelihood is non-finite.")
    return total, trial_log_likelihood


def fit_bayes_optimal(
    u: FloatArray,
    prior: ParameterPrior,
    stream: str = "both_cues",
    options: Optional[QuasiNewtonOptions] = None,
) -> BayesOptimalFit:
    """Fit the perceptual model to the inputs alone, exactly as PAM's demo does.

    ``stream`` selects what the agent is asked to predict:

    ``"both_cues"``
        The frozen Plan-D two-stream model with shared parameters, predicting
        each trial from the active cue's belief. This is the model the joint
        fit actually uses, so it is the consistent choice for a prior mean.
    ``"white"`` / ``"red"``
        A single-stream HGF over one cue's presentations only. These are
        diagnostics: comparing them shows how much the shared-parameter
        assumption frozen in Gate A is forcing a compromise.
    """

    inputs = np.asarray(u, dtype=float)
    if inputs.ndim != 2 or inputs.shape[1] < 2:
        raise ValueError("Bayes-optimal fitting requires stimulus and cue columns.")
    if np.any(~np.isfinite(inputs[:, :2])):
        raise ValueError("Stimulus and cue inputs must all be finite.")
    if stream not in ("both_cues", "white", "red"):
        raise ValueError("stream must be 'both_cues', 'white', or 'red'.")

    mask = prior.free_mask
    if not np.any(mask):
        raise ValueError("The perceptual prior has no free parameter to fit.")

    def predictions(transformed: FloatArray) -> Tuple[FloatArray, FloatArray]:
        parameters = transform_ehgf_binary(transformed)
        if stream == "both_cues":
            result = cue_binary_hgf(inputs, parameters)
            return inputs[:, 0], result.active.muhat[:, 0]
        selector = 1.0 if stream == "white" else 0.0
        take = np.flatnonzero(inputs[:, 1] == selector)
        if take.size == 0:
            raise ValueError("Stream %r has no trials." % stream)
        stimulus = inputs[take, 0]
        result = binary_hgf(stimulus, parameters)
        return stimulus, result.muhat[:, 0]

    def objective(free: FloatArray) -> float:
        full = prior.means.copy()
        full[mask] = np.asarray(free, dtype=float)
        try:
            stimulus, muhat = predictions(full)
            log_likelihood, _ = bayes_optimal_log_likelihood(stimulus, muhat)
        except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
            return float(np.finfo(float).max)
        value = -(log_likelihood + prior.log_density(full))
        if not np.isfinite(value):
            return float(np.finfo(float).max)
        return value

    start = prior.means[mask].astype(float)
    optimization = tapas_quasi_newton(
        objective, start, QuasiNewtonOptions() if options is None else options
    )

    full = prior.means.copy()
    full[mask] = optimization.argument_minimum
    stimulus, muhat = predictions(full)
    log_likelihood, _ = bayes_optimal_log_likelihood(stimulus, muhat)
    names = tuple(prior.names)
    return BayesOptimalFit(
        stream=stream,
        transformed_parameters=full,
        free_parameters=optimization.argument_minimum.copy(),
        free_names=prior.free_names,
        omega_2=float(full[names.index(OMEGA_PARAMETER)]),
        input_log_likelihood=log_likelihood,
        negative_log_joint=float(optimization.value_minimum),
        scored_trials=int(stimulus.size),
        iterations=int(optimization.iterations),
        reset_count=int(optimization.reset_count),
        convergence_reason=optimization.convergence_reason,
    )


def bayes_optimal_prior(
    u: FloatArray,
    base_prior: ParameterPrior,
    options: Optional[QuasiNewtonOptions] = None,
    variance: Optional[float] = None,
) -> Tuple[ParameterPrior, BayesOptimalFit]:
    """Return ``base_prior`` with ``omega_2``'s mean set to its Bayes-optimal value.

    Only the mean is replaced, matching the official demo, which overrides
    ``prc_model.ommu`` and leaves ``omsa`` untouched. Pass ``variance`` to also
    set the prior variance; the current configuration uses 2 where TAPAS and
    the PAM demo both use 4, so that choice is worth making explicit rather
    than inheriting silently.
    """

    fit = fit_bayes_optimal(u, base_prior, "both_cues", options)
    names = tuple(base_prior.names)
    index = names.index(OMEGA_PARAMETER)
    means = base_prior.means.copy()
    means[index] = fit.omega_2
    variances = base_prior.variances.copy()
    if variance is not None:
        if not np.isfinite(variance) or variance < 0:
            raise ValueError("Prior variance must be finite and non-negative.")
        variances[index] = float(variance)
    updated = ParameterPrior(means=means, variances=variances, names=names)
    return updated, fit


def per_cue_bayes_optimal(
    u: FloatArray,
    prior: ParameterPrior,
    options: Optional[QuasiNewtonOptions] = None,
) -> Dict[str, BayesOptimalFit]:
    """Fit each cue stream separately to test the shared-parameter assumption.

    This uses only ``u``, so it is cheap and can be run before any joint fit.
    A small gap between the white and red optima supports sharing one
    perceptual parameter; a large gap quantifies what sharing costs.
    """

    return {
        "both_cues": fit_bayes_optimal(u, prior, "both_cues", options),
        "white": fit_bayes_optimal(u, prior, "white", options),
        "red": fit_bayes_optimal(u, prior, "red", options),
    }
