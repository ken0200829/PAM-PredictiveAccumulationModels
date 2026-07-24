"""Official and Gate-B coherence PAM DDM response models."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .wfpt import wfpt_density


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DDMParameters:
    """Native-space PAM response parameters."""

    a_a: float
    a_v: float
    b_w: float
    b_a: float
    b_v: float
    Ter: float
    b_c: Optional[float] = None


@dataclass(frozen=True)
class CueDDMParameters:
    """Native-space parameters for the recovery-first cue-locus models."""

    a_a: float
    a_v: float
    b_c: float
    Ter: float
    b_H_w: float = 0.0
    b_w: float = 0.0
    b_v: float = 0.0
    gamma_w: float = 0.0
    gamma_v0: float = 0.0


@dataclass(frozen=True)
class TrialwiseDDM:
    precision_modulator: FloatArray
    w: FloatArray
    a: FloatArray
    v: FloatArray
    Ter: FloatArray
    coherence_magnitude: FloatArray
    belief_presented: FloatArray
    sensory_drift: Optional[FloatArray] = None
    cue_drift_bias: Optional[FloatArray] = None
    belief_drift_bias: Optional[FloatArray] = None
    cue_evidence: Optional[FloatArray] = None
    history_prediction: Optional[FloatArray] = None
    integrated_prediction: Optional[FloatArray] = None


def transform_ddm(transformed: FloatArray, regular_rt: FloatArray) -> DDMParameters:
    """Match ``ddm_hgf_transp`` or ``ddm_hgf_coherence_transp``."""

    p = np.asarray(transformed, dtype=float)
    rt = np.asarray(regular_rt, dtype=float)
    rt = rt[np.isfinite(rt)]
    if rt.size == 0:
        raise ValueError("At least one finite fitted RT is required to transform Ter.")
    if p.shape not in ((6,), (7,)):
        raise ValueError("Official DDM needs 6 and coherence DDM needs 7 parameters.")
    a_a = float(np.exp(p[0]))
    a_v = float(np.exp(p[1]))
    b_w = float(2.0 / (1.0 + np.exp(-p[2])) - 1.0)
    if p.size == 6:
        return DDMParameters(
            a_a=a_a,
            a_v=a_v,
            b_w=b_w,
            b_a=float(p[3]),
            b_v=float(p[4]),
            Ter=float(np.min(rt) / (1.0 + np.exp(-p[5]))),
        )
    return DDMParameters(
        a_a=a_a,
        a_v=a_v,
        b_w=b_w,
        b_a=float(p[3]),
        b_v=float(p[4]),
        b_c=float(p[5]),
        Ter=float(np.min(rt) / (1.0 + np.exp(-p[6]))),
    )


def transform_cue_ddm(
    transformed: FloatArray,
    regular_rt: FloatArray,
    parameter_names: tuple,
) -> CueDDMParameters:
    """Transform one registry-declared cue model without positional ambiguity."""

    values = np.asarray(transformed, dtype=float)
    names = tuple(parameter_names)
    if values.shape != (len(names),) or len(set(names)) != len(names):
        raise ValueError("Cue DDM values and unique parameter names must align.")
    required = {"log_a_a", "log_a_v", "b_c", "Ter_logit"}
    allowed = required | {"b_H_w", "b_w", "b_v", "gamma_w", "gamma_v0"}
    if not required.issubset(names) or not set(names).issubset(allowed):
        raise ValueError("Cue DDM parameter names do not match the registry contract.")
    rt = np.asarray(regular_rt, dtype=float)
    rt = rt[np.isfinite(rt)]
    if rt.size == 0:
        raise ValueError("At least one finite fitted RT is required to transform Ter.")
    raw = dict(zip(names, values))
    return CueDDMParameters(
        a_a=float(np.exp(raw["log_a_a"])),
        a_v=float(np.exp(raw["log_a_v"])),
        b_c=float(raw["b_c"]),
        Ter=float(np.min(rt) * _sigmoid_scalar(raw["Ter_logit"])),
        b_H_w=_unit_slope(raw.get("b_H_w", 0.0)),
        b_w=_unit_slope(raw.get("b_w", 0.0)),
        b_v=float(raw.get("b_v", 0.0)),
        gamma_w=float(raw.get("gamma_w", 0.0)),
        gamma_v0=float(raw.get("gamma_v0", 0.0)),
    )


def trialwise_ddm(
    stimulus: FloatArray,
    muhat: FloatArray,
    parameters: DDMParameters,
    signed_coherence: Optional[FloatArray] = None,
    tie: Optional[NDArray[np.bool_]] = None,
) -> TrialwiseDDM:
    """Reconstruct trial-wise PAM DDM parameters before the WFPT call.

    ``tie`` marks trials with no objective category (ratio 0.5).  PAM's drift
    is ``direction * (...)`` with ``direction = 2*stimulus - 1``, which is
    undefined when no category exists.  The plan resolves this by setting
    ``direction = 0``, hence ``v = 0``: with 100 white and 100 black dots the
    sensory evidence favours neither boundary, so zero drift is the principled
    value rather than a convention.  The decision is then driven by the
    starting point ``w`` and diffusion noise alone, which is what makes tie
    trials a clean probe of the belief-driven starting-point bias
    (plan section 5.2.1).  Every non-tie trial keeps the official PAM drift
    bit-for-bit.
    """

    stimulus = np.asarray(stimulus, dtype=float)
    muhat = np.asarray(muhat, dtype=float)
    if stimulus.ndim != 1 or muhat.shape != stimulus.shape:
        raise ValueError("Stimulus and muhat must be equal-length vectors.")
    if np.any(~np.isfinite(stimulus)) or np.any(~np.isin(stimulus, (0.0, 1.0))):
        raise ValueError("Stimulus category must be finite and binary.")
    if np.any(~np.isfinite(muhat)) or np.any((muhat <= 0) | (muhat >= 1)):
        raise ValueError("Predicted level-one beliefs must lie inside (0,1).")

    if signed_coherence is None:
        coherence_magnitude = np.zeros_like(stimulus)
        if parameters.b_c is not None:
            raise ValueError("The coherence DDM requires signed coherence.")
    else:
        coherence = np.asarray(signed_coherence, dtype=float)
        if coherence.shape != stimulus.shape:
            raise ValueError("Signed coherence must match the trial count.")
        if np.any(~np.isfinite(coherence)) or np.any(np.abs(coherence) > 1):
            raise ValueError("Signed coherence must be finite and within [-1,1].")
        coherence_magnitude = np.abs(coherence)

    if tie is None:
        tie_mask = np.zeros_like(stimulus, dtype=bool)
    else:
        tie_mask = np.asarray(tie, dtype=bool)
        if tie_mask.shape != stimulus.shape:
            raise ValueError("Tie mask must match the trial count.")
        if signed_coherence is not None and np.any(
            coherence_magnitude[tie_mask] != 0.0
        ):
            raise ValueError("Tie trials must carry zero signed coherence.")

    precision_modulator = _sigmoid(1.0 / (muhat * (1.0 - muhat)) - 4.0) - 0.5
    w = 0.5 + parameters.b_w * (muhat - 0.5)
    a = parameters.a_a + parameters.b_a * precision_modulator
    direction = np.where(tie_mask, 0.0, 2.0 * stimulus - 1.0)
    belief_presented = stimulus * muhat + (1.0 - stimulus) * (1.0 - muhat)
    coherence_slope = 0.0 if parameters.b_c is None else parameters.b_c
    v = direction * (
        parameters.a_v
        + coherence_slope * coherence_magnitude
        + parameters.b_v * (belief_presented - 0.5)
    )
    return TrialwiseDDM(
        precision_modulator=precision_modulator,
        w=w,
        a=a,
        v=v,
        Ter=np.full_like(muhat, parameters.Ter),
        coherence_magnitude=coherence_magnitude,
        belief_presented=belief_presented,
    )


def trialwise_cue_ddm(
    architecture: str,
    stimulus: FloatArray,
    prediction: FloatArray,
    cue_evidence: FloatArray,
    parameters: CueDDMParameters,
    signed_coherence: FloatArray,
    tie: NDArray[np.bool_],
) -> TrialwiseDDM:
    """Compute the history, parallel, or integrated cue-locus DDM arrays."""

    architecture = str(architecture).lower()
    if architecture not in {"history", "parallel", "integrated"}:
        raise ValueError("Unknown cue DDM architecture: %s" % architecture)
    stimulus = np.asarray(stimulus, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    cue = np.asarray(cue_evidence, dtype=float)
    coherence = np.asarray(signed_coherence, dtype=float)
    tie_mask = np.asarray(tie, dtype=bool)
    if stimulus.ndim != 1 or not (
        prediction.shape == cue.shape == coherence.shape == tie_mask.shape == stimulus.shape
    ):
        raise ValueError("Cue DDM trialwise inputs must be equal-length vectors.")
    if np.any(~np.isfinite(stimulus)) or np.any(~np.isin(stimulus, (0.0, 1.0))):
        raise ValueError("Stimulus category must be finite and coded 0/1.")
    if np.any(~np.isfinite(prediction)) or np.any(
        (prediction <= 0.0) | (prediction >= 1.0)
    ):
        raise ValueError("HGF predictions must lie strictly inside (0,1).")
    if np.any(~np.isfinite(cue)) or np.any(~np.isin(cue, (-1.0, 0.0, 1.0))):
        raise ValueError("Cue evidence must be coded -1, 0, or 1.")
    if np.any(~np.isfinite(coherence)) or np.any(np.abs(coherence) > 1.0):
        raise ValueError("Signed coherence must be finite and within [-1,1].")
    if np.any(np.abs(coherence[tie_mask]) != 0.0):
        raise ValueError("Tie trials must carry zero signed coherence.")

    coherence_magnitude = np.abs(coherence)
    direction = np.where(tie_mask, 0.0, 2.0 * stimulus - 1.0)
    sensory_drift = direction * (
        parameters.a_v + parameters.b_c * coherence_magnitude
    )
    cue_drift_bias = np.zeros_like(prediction)
    belief_drift_bias = np.zeros_like(prediction)

    if architecture in {"history", "parallel"}:
        history_w = 0.5 + parameters.b_H_w * (prediction - 0.5)
        if np.any((history_w <= 0.0) | (history_w >= 1.0)):
            raise ValueError("History starting point must lie strictly inside (0,1).")
        if architecture == "parallel" and parameters.gamma_w != 0.0:
            w = history_w.copy()
            cue_active = cue != 0.0
            w[cue_active] = _sigmoid(
                _logit(history_w[cue_active])
                + parameters.gamma_w * cue[cue_active]
            )
        else:
            # Preserve exact zero-effect nesting instead of round-tripping
            # history_w through logit and sigmoid.
            w = history_w.copy()
        if architecture == "parallel":
            cue_drift_bias = parameters.gamma_v0 * cue
        history_prediction = prediction.copy()
        integrated_prediction = None
    else:
        w = 0.5 + parameters.b_w * (prediction - 0.5)
        belief_drift_bias = parameters.b_v * (prediction - 0.5)
        history_prediction = None
        integrated_prediction = prediction.copy()

    if np.any(~np.isfinite(w)) or np.any((w <= 0.0) | (w >= 1.0)):
        raise ValueError("Starting point must lie strictly inside (0,1).")

    v = sensory_drift + cue_drift_bias + belief_drift_bias
    v = np.where(tie_mask, 0.0, v)
    cue_drift_bias = np.where(tie_mask, 0.0, cue_drift_bias)
    belief_drift_bias = np.where(tie_mask, 0.0, belief_drift_bias)
    precision_modulator = (
        _sigmoid(1.0 / (prediction * (1.0 - prediction)) - 4.0) - 0.5
    )
    belief_presented = (
        stimulus * prediction + (1.0 - stimulus) * (1.0 - prediction)
    )
    return TrialwiseDDM(
        precision_modulator=precision_modulator,
        w=w,
        a=np.full_like(prediction, parameters.a_a),
        v=v,
        Ter=np.full_like(prediction, parameters.Ter),
        coherence_magnitude=coherence_magnitude,
        belief_presented=belief_presented,
        sensory_drift=sensory_drift,
        cue_drift_bias=cue_drift_bias,
        belief_drift_bias=belief_drift_bias,
        cue_evidence=cue.copy(),
        history_prediction=history_prediction,
        integrated_prediction=integrated_prediction,
    )


def ddm_loglik(
    y: FloatArray,
    stimulus: FloatArray,
    muhat: FloatArray,
    parameters: DDMParameters,
    signed_coherence: Optional[FloatArray] = None,
    tie: Optional[NDArray[np.bool_]] = None,
) -> Tuple[FloatArray, TrialwiseDDM]:
    """Return trial log likelihoods with excluded response rows left NaN."""

    trialwise = trialwise_ddm(stimulus, muhat, parameters, signed_coherence, tie)
    return _trialwise_loglik(y, trialwise), trialwise


def cue_ddm_loglik(
    y: FloatArray,
    architecture: str,
    stimulus: FloatArray,
    prediction: FloatArray,
    cue_evidence: FloatArray,
    parameters: CueDDMParameters,
    signed_coherence: FloatArray,
    tie: NDArray[np.bool_],
) -> Tuple[FloatArray, TrialwiseDDM]:
    """Return cue-model trial log likelihoods with masked rows left NaN."""

    trialwise = trialwise_cue_ddm(
        architecture,
        stimulus,
        prediction,
        cue_evidence,
        parameters,
        signed_coherence,
        tie,
    )
    return _trialwise_loglik(y, trialwise), trialwise


def choice_one_probability(trialwise: TrialwiseDDM) -> FloatArray:
    """Return the eventual upper/white-boundary probability for each trial.

    This is the closed-form boundary-hitting probability without a response
    deadline.  It is used to calibrate cue-effect sizes on an interpretable
    probability scale; three-second captured mass remains a separate
    prior-predictive diagnostic.
    """

    w = np.asarray(trialwise.w, dtype=float)
    boundary = np.asarray(trialwise.a, dtype=float)
    drift = np.asarray(trialwise.v, dtype=float)
    if not (w.shape == boundary.shape == drift.shape) or w.ndim != 1:
        raise ValueError("Choice-probability DDM arrays must be equal vectors.")
    if np.any(~np.isfinite(w)) or np.any((w <= 0.0) | (w >= 1.0)):
        raise ValueError("Starting point must lie strictly inside (0,1).")
    if np.any(~np.isfinite(boundary)) or np.any(boundary <= 0.0):
        raise ValueError("Boundary separation must be finite and positive.")
    if np.any(~np.isfinite(drift)):
        raise ValueError("Drift must be finite.")

    probability = np.empty_like(w)
    nearly_zero = np.abs(drift * boundary) < 1e-8
    probability[nearly_zero] = w[nearly_zero]
    positive = (~nearly_zero) & (drift > 0.0)
    if np.any(positive):
        probability[positive] = _positive_drift_choice_probability(
            drift[positive], boundary[positive], w[positive]
        )
    negative = (~nearly_zero) & (drift < 0.0)
    if np.any(negative):
        probability[negative] = 1.0 - _positive_drift_choice_probability(
            -drift[negative], boundary[negative], 1.0 - w[negative]
        )
    return probability


def _trialwise_loglik(y: FloatArray, trialwise: TrialwiseDDM) -> FloatArray:
    responses = np.asarray(y, dtype=float)
    if responses.ndim != 2 or responses.shape[1] != 2:
        raise ValueError("Responses must be an N x 2 [RT, choice] array.")
    if responses.shape[0] != trialwise.w.size:
        raise ValueError("Response and input trial counts differ.")
    regular = np.all(np.isfinite(responses), axis=1)
    if not np.any(regular):
        raise ValueError("At least one response trial must be included.")
    choice = responses[regular, 1]
    if np.any(~np.isin(choice, (0.0, 1.0))):
        raise ValueError("Regular choices must be coded 0/1.")

    if np.any(trialwise.w[regular] <= 0) or np.any(trialwise.w[regular] >= 1):
        raise ValueError("Starting point must lie strictly inside (0,1).")
    if np.any(trialwise.a[regular] <= 0):
        raise ValueError("Boundary separation must be positive.")

    logp = np.full(responses.shape[0], np.nan)
    decision_time = np.maximum(
        np.finfo(float).eps, responses[regular, 0] - trialwise.Ter[regular]
    )
    regular_indices = np.flatnonzero(regular)
    response = responses[regular, 1]
    choice_one_density = wfpt_density(
        decision_time,
        -trialwise.v[regular],
        trialwise.a[regular],
        1.0 - trialwise.w[regular],
    )
    choice_zero_density = wfpt_density(
        decision_time,
        trialwise.v[regular],
        trialwise.a[regular],
        trialwise.w[regular],
    )
    probability = choice_one_density * response + choice_zero_density * (1.0 - response)
    positive = probability > 0
    logp[regular_indices[positive]] = np.log(probability[positive] + np.finfo(float).eps)
    return logp


def _sigmoid(value: FloatArray) -> FloatArray:
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-value))


def _sigmoid_scalar(value: float) -> float:
    if value >= 0:
        return float(1.0 / (1.0 + np.exp(-value)))
    exponential = np.exp(value)
    return float(exponential / (1.0 + exponential))


def _unit_slope(value: float) -> float:
    return 2.0 * _sigmoid_scalar(float(value)) - 1.0


def _logit(probability: FloatArray) -> FloatArray:
    return np.log(probability) - np.log1p(-probability)


def _positive_drift_choice_probability(
    drift: FloatArray, boundary: FloatArray, w: FloatArray
) -> FloatArray:
    numerator = -np.expm1(-2.0 * drift * boundary * w)
    denominator = -np.expm1(-2.0 * drift * boundary)
    return numerator / denominator
