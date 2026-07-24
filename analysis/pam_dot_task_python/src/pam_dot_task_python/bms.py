"""SPM12-style random-effects Bayesian model selection."""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import betainc, digamma, expit, gammaln


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GibbsBMSResult:
    expected_frequency: FloatArray
    exceedance_probability: FloatArray
    frequency_samples: FloatArray
    subject_model_probability: FloatArray


@dataclass(frozen=True)
class SensitivityBMSResult:
    alpha: FloatArray
    expected_frequency: FloatArray
    exceedance_probability: FloatArray
    protected_exceedance_probability: FloatArray
    bayes_omnibus_risk: float
    iterations: int


@dataclass(frozen=True)
class BMSResult:
    model_ids: tuple
    subject_ids: tuple
    lme: FloatArray
    alpha0: FloatArray
    samples: int
    seed: int
    algorithm: str
    primary: GibbsBMSResult
    sensitivity: Optional[SensitivityBMSResult]


def random_effects_bms(
    lme: FloatArray,
    model_ids: Sequence[str],
    subject_ids: Sequence[str],
    alpha0: Optional[FloatArray] = None,
    samples: int = 1_000_000,
    seed: int = 20260720,
    run_sensitivity: bool = False,
) -> BMSResult:
    """Named wrapper matching ``pam_dot_task_bms.m`` policy."""

    evidence, models, subjects, prior = _validate_inputs(
        lme, model_ids, subject_ids, alpha0, samples, seed
    )
    primary_rng = _mt19937(seed)
    primary = gibbs_bms(evidence, prior, samples, primary_rng)
    sensitivity = None
    if run_sensitivity:
        sensitivity_rng = _mt19937(seed)
        sensitivity = protected_bms(evidence, prior, samples, sensitivity_rng)
    return BMSResult(
        model_ids=models,
        subject_ids=subjects,
        lme=evidence,
        alpha0=prior,
        samples=samples,
        seed=seed,
        algorithm="MT19937",
        primary=primary,
        sensitivity=sensitivity,
    )


def gibbs_bms(
    lme: FloatArray,
    alpha0: FloatArray,
    samples: int,
    rng: np.random.Generator,
) -> GibbsBMSResult:
    """Port of ``spm_BMS_gibbs`` with subjects in rows and models in columns."""

    evidence = np.asarray(lme, dtype=float)
    prior = np.asarray(alpha0, dtype=float)
    subject_count, model_count = evidence.shape
    frequency = rng.gamma(shape=prior, scale=1.0)
    frequency /= np.sum(frequency)
    centered_evidence = evidence - np.max(evidence, axis=1, keepdims=True)
    frequency_samples = np.empty((samples, model_count), dtype=float)
    subject_probability_sum = np.zeros((subject_count, model_count), dtype=float)
    epsilon = np.finfo(float).eps

    for draw_index in range(2 * samples):
        unnormalized = np.exp(centered_evidence + np.log(frequency)) + epsilon
        subject_probability = unnormalized / np.sum(
            unnormalized, axis=1, keepdims=True
        )
        uniforms = rng.random(subject_count)
        cumulative = np.cumsum(subject_probability, axis=1)
        assignments = np.sum(uniforms[:, None] > cumulative, axis=1)
        counts = np.bincount(assignments, minlength=model_count)
        frequency = rng.gamma(shape=prior + counts, scale=1.0)
        frequency /= np.sum(frequency)
        if draw_index >= samples:
            retained = draw_index - samples
            frequency_samples[retained] = frequency
            subject_probability_sum += subject_probability

    winners = np.argmax(frequency_samples, axis=1)
    exceedance = np.bincount(winners, minlength=model_count) / samples
    return GibbsBMSResult(
        expected_frequency=np.mean(frequency_samples, axis=0),
        exceedance_probability=exceedance,
        frequency_samples=frequency_samples,
        subject_model_probability=subject_probability_sum / samples,
    )


def protected_bms(
    lme: FloatArray,
    alpha0: FloatArray,
    samples: int,
    rng: np.random.Generator,
) -> SensitivityBMSResult:
    """Port the non-family, non-plotting branch of ``spm_BMS``."""

    evidence = np.asarray(lme, dtype=float)
    prior = np.asarray(alpha0, dtype=float)
    subject_count, model_count = evidence.shape
    alpha = prior.copy()
    convergence = np.inf
    iterations = 0
    subject_probability = np.full((subject_count, model_count), np.nan)
    while convergence > 1e-3:
        iterations += 1
        expected_log_frequency = digamma(alpha) - digamma(np.sum(alpha))
        log_unnormalized = evidence + expected_log_frequency
        unnormalized = np.exp(
            log_unnormalized - np.max(log_unnormalized, axis=1, keepdims=True)
        )
        subject_probability = unnormalized / np.sum(
            unnormalized, axis=1, keepdims=True
        )
        previous = alpha.copy()
        alpha = prior + np.sum(subject_probability, axis=0)
        convergence = float(np.linalg.norm(alpha - previous))
        if iterations > 100_000:
            raise RuntimeError("SPM variational BMS did not converge.")

    expected_frequency = alpha / np.sum(alpha)
    if model_count == 2:
        exceedance = np.array(
            [
                betainc(alpha[1], alpha[0], 0.5),
                betainc(alpha[0], alpha[1], 0.5),
            ]
        )
    else:
        exceedance = dirichlet_exceedance(alpha, samples, rng)
    bor = bayes_omnibus_risk(
        evidence.T,
        alpha,
        subject_probability.T,
        prior,
    )
    protected = (1.0 - bor) * exceedance + bor / model_count
    return SensitivityBMSResult(
        alpha=alpha,
        expected_frequency=expected_frequency,
        exceedance_probability=exceedance,
        protected_exceedance_probability=protected,
        bayes_omnibus_risk=bor,
        iterations=iterations,
    )


def dirichlet_exceedance(
    alpha: FloatArray, samples: int, rng: np.random.Generator
) -> FloatArray:
    """Monte Carlo exceedance probabilities from ``spm_dirichlet_exceedance``."""

    parameters = np.asarray(alpha, dtype=float)
    model_count = parameters.size
    bytes_limit = 2**28
    block_count = int(np.ceil(samples * model_count * 8 / bytes_limit))
    block_count = max(block_count, 1)
    base = samples // block_count
    blocks = [base] * block_count
    blocks[-1] = samples - sum(blocks[:-1])
    winner_count = np.zeros(model_count, dtype=float)
    for block_size in blocks:
        draws = rng.gamma(shape=parameters, scale=1.0, size=(block_size, model_count))
        draws /= np.sum(draws, axis=1, keepdims=True)
        winner_count += np.bincount(
            np.argmax(draws, axis=1), minlength=model_count
        )
    return winner_count / samples


def bayes_omnibus_risk(
    model_by_subject_lme: FloatArray,
    posterior_alpha: FloatArray,
    posterior_model_probability: FloatArray,
    prior_alpha: FloatArray,
) -> float:
    """Model-prior branch of ``spm_BMS_bor``."""

    evidence = np.asarray(model_by_subject_lme, dtype=float)
    posterior_alpha = np.asarray(posterior_alpha, dtype=float)
    posterior_probability = np.asarray(posterior_model_probability, dtype=float)
    prior_alpha = np.asarray(prior_alpha, dtype=float)
    model_count, subject_count = evidence.shape
    epsilon = np.finfo(float).eps

    null_evidence = 0.0
    for subject in range(subject_count):
        centered = evidence[:, subject] - np.max(evidence[:, subject])
        probability = np.exp(centered)
        probability /= np.sum(probability)
        null_evidence += np.sum(
            probability
            * (
                evidence[:, subject]
                - np.log(model_count)
                - np.log(probability + epsilon)
            )
        )

    expected_log_frequency = digamma(posterior_alpha) - digamma(
        np.sum(posterior_alpha)
    )
    entropy_frequency = (
        np.sum(gammaln(posterior_alpha))
        - gammaln(np.sum(posterior_alpha))
        - np.sum((posterior_alpha - 1.0) * expected_log_frequency)
    )
    entropy_model = -np.sum(
        posterior_probability * np.log(posterior_probability + epsilon)
    )
    expected_log_joint = (
        gammaln(np.sum(prior_alpha))
        - np.sum(gammaln(prior_alpha))
        + np.sum((prior_alpha - 1.0) * expected_log_frequency)
        + np.sum(
            posterior_probability
            * (expected_log_frequency[:, None] + evidence)
        )
    )
    alternative_evidence = expected_log_joint + entropy_frequency + entropy_model
    return float(expit(null_evidence - alternative_evidence))


def _validate_inputs(
    lme: FloatArray,
    model_ids: Sequence[str],
    subject_ids: Sequence[str],
    alpha0: Optional[FloatArray],
    samples: int,
    seed: int,
) -> tuple:
    evidence = np.asarray(lme, dtype=float)
    models = tuple(str(value) for value in model_ids)
    subjects = tuple(str(value) for value in subject_ids)
    if evidence.ndim != 2 or evidence.shape != (len(subjects), len(models)):
        raise ValueError("LME must be subjects-by-models and match both ID vectors.")
    if len(set(models)) != len(models) or len(set(subjects)) != len(subjects):
        raise ValueError("Subject and model IDs must be unique.")
    if np.any(~np.isfinite(evidence)):
        raise ValueError("BMS requires finite LME on a common subject set.")
    if not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer.")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    prior = (
        np.ones(len(models), dtype=float)
        if alpha0 is None
        else np.asarray(alpha0, dtype=float)
    )
    if prior.shape != (len(models),) or np.any(~np.isfinite(prior)) or np.any(prior <= 0):
        raise ValueError("alpha0 must contain one finite positive value per model.")
    return evidence, models, subjects, prior


def _mt19937(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.MT19937(seed))

