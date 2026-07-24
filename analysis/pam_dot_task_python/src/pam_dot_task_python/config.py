"""Legacy frozen priors and the versioned candidate cue-locus registry."""

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

CUE_FORMULATION_VERSION = "cue-locus-0.2.0"
CUE_PRIOR_VERSION = "cue-prior-candidate-0.2.0"
CUE_PRIOR_CALIBRATION_DESIGN_DIGEST = (
    "9a326c820b03d5829ceb7dbd075248896ed6cb77db5fe72c46083c5cade76a28"
)
CUE_EFFECT_PRIOR_COVERAGE_Z = 1.959963984540054
CUE_EFFECT_STRONG_TRANSFORMED = MappingProxyType(
    {
        "gamma_w": 0.1000834585569826,
        "gamma_v0": 0.1272149128342861,
        "b_w": 2.2765461945358165,
        "b_v": 3.7505622404216705,
    }
)
CUE_EFFECT_PRIOR_VARIANCES = MappingProxyType(
    {
        parameter: (value / CUE_EFFECT_PRIOR_COVERAGE_Z) ** 2
        for parameter, value in CUE_EFFECT_STRONG_TRANSFORMED.items()
    }
)


@dataclass(frozen=True)
class ParameterPrior:
    """Gaussian priors in TAPAS' transformed parameter space."""

    means: FloatArray
    variances: FloatArray
    names: tuple

    def __post_init__(self) -> None:
        means = np.asarray(self.means, dtype=float)
        variances = np.asarray(self.variances, dtype=float)
        if means.ndim != 1 or variances.ndim != 1:
            raise ValueError("Prior means and variances must be one-dimensional.")
        if means.shape != variances.shape or means.size != len(self.names):
            raise ValueError("Prior arrays and names must have the same length.")
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "variances", variances)

    @property
    def free_mask(self) -> NDArray[np.bool_]:
        return np.isfinite(self.variances) & (self.variances > 0)

    @property
    def free_names(self) -> tuple:
        return tuple(name for name, free in zip(self.names, self.free_mask) if free)

    def log_density(self, transformed: FloatArray) -> float:
        values = np.asarray(transformed, dtype=float)
        if values.shape != self.means.shape:
            raise ValueError("Parameter vector does not match the prior.")
        mask = self.free_mask
        delta = values[mask] - self.means[mask]
        variance = self.variances[mask]
        terms = -0.5 * np.log(2.0 * np.pi * variance) - 0.5 * delta**2 / variance
        return float(np.sum(terms))


@dataclass(frozen=True)
class CueModelSpec:
    """One cue-locus model's perceptual and response-layer contract."""

    model_id: str
    architecture: str
    perceptual_model: str
    response_effects: tuple

    def __post_init__(self) -> None:
        if self.architecture not in {"history", "parallel", "integrated"}:
            raise ValueError("Unknown cue architecture: %s" % self.architecture)
        if self.perceptual_model not in {"cue_blind", "two_cue"}:
            raise ValueError("Unknown cue perceptual model: %s" % self.perceptual_model)
        allowed = {"b_H_w", "b_w", "b_v", "gamma_w", "gamma_v0"}
        if len(set(self.response_effects)) != len(self.response_effects):
            raise ValueError("Cue response effects must be unique.")
        if not set(self.response_effects).issubset(allowed):
            raise ValueError("Cue response effects contain an unknown parameter.")


CUE_MODEL_SPECS = MappingProxyType({
    "cue_history_w": CueModelSpec(
        "cue_history_w", "history", "cue_blind", ("b_H_w",)
    ),
    "cue_parallel_w": CueModelSpec(
        "cue_parallel_w", "parallel", "cue_blind", ("b_H_w", "gamma_w")
    ),
    "cue_parallel_vbias": CueModelSpec(
        "cue_parallel_vbias",
        "parallel",
        "cue_blind",
        ("b_H_w", "gamma_v0"),
    ),
    "cue_parallel_w_vbias": CueModelSpec(
        "cue_parallel_w_vbias",
        "parallel",
        "cue_blind",
        ("b_H_w", "gamma_w", "gamma_v0"),
    ),
    "cue_integrated_w": CueModelSpec(
        "cue_integrated_w", "integrated", "two_cue", ("b_w",)
    ),
    "cue_integrated_vbias": CueModelSpec(
        "cue_integrated_vbias", "integrated", "two_cue", ("b_v",)
    ),
    "cue_integrated_w_vbias": CueModelSpec(
        "cue_integrated_w_vbias",
        "integrated",
        "two_cue",
        ("b_w", "b_v"),
    ),
})


def cue_model_spec(model_id: str):
    """Return a cue-locus model contract, or ``None`` for legacy models."""

    return CUE_MODEL_SPECS.get(str(model_id).lower())


def cue_hgf_prior() -> ParameterPrior:
    """Shared effective-two-level eHGF prior used by the two cue streams."""

    names = (
        "mu_0_1",
        "mu_0_2",
        "mu_0_3",
        "logsa_0_1",
        "logsa_0_2",
        "logsa_0_3",
        "rho_1",
        "rho_2",
        "rho_3",
        "logkappa_1",
        "logkappa_2",
        "omega_1",
        "omega_2",
        "logtheta",
    )
    means = np.array(
        [
            np.nan,
            0.0,
            1.0,
            np.nan,
            np.log(0.1),
            np.log(1.0),
            np.nan,
            0.0,
            0.0,
            np.log(1.0),
            -np.inf,
            np.nan,
            -3.0,
            2.0,
        ],
        dtype=float,
    )
    variances = np.array(
        [
            np.nan,
            0.0,
            0.0,
            np.nan,
            0.0,
            0.0,
            np.nan,
            0.0,
            0.0,
            0.0,
            0.0,
            np.nan,
            2.0,
            0.0,
        ],
        dtype=float,
    )
    return ParameterPrior(means=means, variances=variances, names=names)


def ddm_prior(model_id: str) -> ParameterPrior:
    """Return an exact official or coherence-extended DDM reduction."""

    model_id = model_id.lower()
    official_models = {
        "ddm_null": (),
        "ddm_w": ("b_w",),
        "ddm_a": ("b_a",),
        "ddm_v": ("b_v",),
        "ddm_full": ("b_w", "b_a", "b_v"),
    }
    coherence_models = {
        "ddm_c": (),
        "ddm_w_c": ("b_w",),
        "ddm_a_c": ("b_a",),
        "ddm_v_c": ("b_v",),
        "ddm_full_c": ("b_w", "b_a", "b_v"),
    }
    cue_spec = cue_model_spec(model_id)
    if cue_spec is not None:
        common = ("log_a_a", "log_a_v")
        trailing = ("b_c", "Ter_logit")
        names = common + cue_spec.response_effects + trailing
        variances = np.full(len(names), 4.0, dtype=float)
        for parameter, variance in CUE_EFFECT_PRIOR_VARIANCES.items():
            if parameter in names:
                variances[names.index(parameter)] = variance
        return ParameterPrior(
            means=np.zeros(len(names), dtype=float),
            variances=variances,
            names=names,
        )
    if model_id in official_models:
        names = ("log_a_a", "log_a_v", "b_w", "b_a", "b_v", "Ter_logit")
        free_slopes = official_models[model_id]
        variances = np.array([4.0, 4.0, 0.0, 0.0, 0.0, 4.0])
    elif model_id in coherence_models:
        names = (
            "log_a_a",
            "log_a_v",
            "b_w",
            "b_a",
            "b_v",
            "b_c",
            "Ter_logit",
        )
        free_slopes = coherence_models[model_id]
        variances = np.array([4.0, 4.0, 0.0, 0.0, 0.0, 4.0, 4.0])
    else:
        raise ValueError("Unknown DDM model ID: %s" % model_id)

    for slope in free_slopes:
        variances[names.index(slope)] = 4.0
    return ParameterPrior(
        means=np.zeros(len(names), dtype=float),
        variances=variances,
        names=names,
    )


def cue_registry_manifest() -> dict:
    """Return the deterministic formulation/prior payload used for hashing."""

    models = {}
    for model_id, spec in sorted(CUE_MODEL_SPECS.items()):
        prior = ddm_prior(model_id)
        models[model_id] = {
            "architecture": spec.architecture,
            "perceptual_model": spec.perceptual_model,
            "response_effects": list(spec.response_effects),
            "parameter_names": list(prior.names),
            "prior_means": prior.means.tolist(),
            "prior_variances": prior.variances.tolist(),
        }
    return {
        "formulation_version": CUE_FORMULATION_VERSION,
        "prior_version": CUE_PRIOR_VERSION,
        "effect_prior_calibration": {
            "design_digest": CUE_PRIOR_CALIBRATION_DESIGN_DIGEST,
            "choice_probability_targets": {
                "weak": 0.005,
                "medium": 0.015,
                "strong": 0.025,
            },
            "central_prior_mass": 0.95,
            "coverage_z": CUE_EFFECT_PRIOR_COVERAGE_Z,
            "strong_transformed_values": dict(CUE_EFFECT_STRONG_TRANSFORMED),
            "variances": dict(CUE_EFFECT_PRIOR_VARIANCES),
            "status": "candidate_not_frozen",
        },
        "models": models,
    }


def cue_registry_digest() -> str:
    """Hash the cue model registry so incompatible runs cannot be mixed."""

    payload = json.dumps(
        cue_registry_manifest(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
