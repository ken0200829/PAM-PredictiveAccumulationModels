"""Pre-outcome cue-effect calibration and prior-predictive diagnostics."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .config import (
    CUE_FORMULATION_VERSION,
    CUE_PRIOR_VERSION,
    cue_hgf_prior,
    cue_registry_digest,
    ddm_prior,
)
from .data import SubjectData
from .hgf import cue_binary_hgf, cue_blind_binary_hgf, transform_ehgf_binary
from .objective import JointModel
from .ppc import RESPONSE_DEADLINE_SECONDS, simulate_ddm
from .response import (
    TrialwiseDDM,
    choice_one_probability,
    transform_cue_ddm,
    trialwise_cue_ddm,
)


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class CalibrationDesign:
    """Outcome-redacted subject design used before fitting real responses."""

    subject_id: str
    condition: str
    stimulus: FloatArray
    cue_white: FloatArray
    signed_coherence: FloatArray
    tie: BoolArray
    cue_evidence: FloatArray
    likelihood_mask: BoolArray

    def __post_init__(self) -> None:
        if not self.subject_id:
            raise ValueError("Calibration design requires a subject identifier.")
        expected_signs = {
            "normal": 1.0,
            "normal_cb": 1.0,
            "reverse": -1.0,
            "reverse_cb": -1.0,
        }
        if self.condition not in expected_signs:
            raise ValueError("Calibration design has an unknown condition.")
        values = {
            "stimulus": (self.stimulus, float),
            "cue_white": (self.cue_white, float),
            "signed_coherence": (self.signed_coherence, float),
            "tie": (self.tie, bool),
            "cue_evidence": (self.cue_evidence, float),
            "likelihood_mask": (self.likelihood_mask, bool),
        }
        arrays = {}
        for name, (raw, dtype) in values.items():
            array = np.asarray(raw, dtype=dtype).copy()
            if array.shape != (380,):
                raise ValueError("Calibration design %s must have 380 rows." % name)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
            arrays[name] = array
        stimulus = arrays["stimulus"]
        cue_white = arrays["cue_white"]
        coherence = arrays["signed_coherence"]
        tie = arrays["tie"]
        cue_evidence = arrays["cue_evidence"]
        likelihood = arrays["likelihood_mask"]
        if np.any(~np.isfinite(stimulus)) or np.any(~np.isin(stimulus, (0.0, 1.0))):
            raise ValueError("Calibration stimulus must be finite and binary.")
        if np.any(~np.isfinite(cue_white)) or np.any(~np.isin(cue_white, (0.0, 1.0))):
            raise ValueError("Calibration cue must be finite and binary.")
        if np.any(~np.isfinite(coherence)) or np.any(np.abs(coherence) > 1.0):
            raise ValueError("Calibration coherence must be finite and within [-1,1].")
        if np.any(tie != (coherence == 0.0)):
            raise ValueError("Calibration tie mask must equal zero coherence.")
        non_tie = ~tie
        if np.any((coherence[non_tie] > 0.0) != (stimulus[non_tie] == 1.0)):
            raise ValueError("Calibration coherence sign and stimulus disagree.")
        red = cue_white == 0.0
        expected_cue = red.astype(float) * expected_signs[self.condition]
        if np.any(cue_evidence != expected_cue):
            raise ValueError("Calibration cue evidence and condition disagree.")
        if np.any(likelihood[:100]):
            raise ValueError("Learning trials cannot enter the response likelihood mask.")

    @classmethod
    def from_subject(cls, subject: SubjectData) -> "CalibrationDesign":
        audit = subject.audit
        return cls(
            subject_id=subject.subject_id,
            condition=subject.condition.name,
            stimulus=subject.u[:, 0].copy(),
            cue_white=subject.u[:, 1].copy(),
            signed_coherence=subject.u[:, 2].copy(),
            tie=subject.is_tie.copy(),
            cue_evidence=subject.cue_evidence.copy(),
            likelihood_mask=audit["likelihood_included"].to_numpy(dtype=bool),
        )

    @property
    def u(self) -> FloatArray:
        return np.column_stack(
            (self.stimulus, self.cue_white, self.signed_coherence)
        )


@dataclass(frozen=True)
class EffectTargetSpec:
    """Choice-probability targets defining weak, medium, and strong effects."""

    version: str = "cue-effect-targets-0.1.0"
    weak: float = 0.005
    medium: float = 0.015
    strong: float = 0.025
    w_metric: str = "mean_signed_red_tie_eventual_choice_change_vs_zero_effect"
    v0_metric: str = "mean_signed_red_nontie_eventual_choice_change_vs_zero_effect"

    def __post_init__(self) -> None:
        values = (self.weak, self.medium, self.strong)
        if not 0 < values[0] < values[1] < values[2] < 0.5:
            raise ValueError("Effect targets must increase strictly inside (0,0.5).")

    @property
    def levels(self) -> Tuple[Tuple[str, float], ...]:
        return (
            ("weak", self.weak),
            ("medium", self.medium),
            ("strong", self.strong),
        )


@dataclass(frozen=True)
class EffectCalibration:
    model_id: str
    parameter: str
    locus: str
    level: str
    target_choice_change: float
    transformed_value: Optional[float]
    native_value: Optional[float]
    achieved_choice_change: float
    reachable: bool


@dataclass(frozen=True)
class PriorPredictivePolicy:
    """Predeclared engineering checks for a candidate prior."""

    version: str = "cue-prior-predictive-policy-0.1.0"
    draws: int = 16
    replicates_per_draw: int = 4
    seed: int = 202607221
    sampling_scope: str = "cue_response_effects_only"
    decision_time_step: float = 0.01
    response_deadline_seconds: float = RESPONSE_DEADLINE_SECONDS
    ter_reference_rt_seconds: float = 0.60
    extreme_w_threshold: float = 0.01
    max_invalid_draw_rate: float = 0.10
    max_extreme_w_fraction: float = 0.10
    min_median_captured_mass: float = 0.95
    max_maximum_captured_mass: float = 1.02
    min_median_choice_rate: float = 0.05
    max_median_choice_rate: float = 0.95
    min_median_rt_q10: float = 0.15
    min_median_rt_q50: float = 0.20
    max_median_rt_q50: float = 2.50
    max_median_rt_q90: float = RESPONSE_DEADLINE_SECONDS

    def __post_init__(self) -> None:
        if self.draws <= 0 or self.replicates_per_draw <= 0:
            raise ValueError("Prior-predictive draw counts must be positive.")
        if self.sampling_scope not in {
            "cue_response_effects_only",
            "all_free_parameters",
        }:
            raise ValueError("Unknown prior-predictive sampling scope.")
        if self.response_deadline_seconds != RESPONSE_DEADLINE_SECONDS:
            raise ValueError("The response deadline is fixed at three seconds.")
        if self.decision_time_step <= 0:
            raise ValueError("Decision-time step must be positive.")
        if not 0 < self.ter_reference_rt_seconds < self.response_deadline_seconds:
            raise ValueError("Ter reference RT must lie inside the response window.")
        if not 0 < self.extreme_w_threshold < 0.5:
            raise ValueError("Extreme-w threshold must lie inside (0,0.5).")
        if not 0.0 <= self.max_invalid_draw_rate < 1.0:
            raise ValueError("Invalid-draw limit must lie in [0,1).")
        if not 0.0 <= self.max_extreme_w_fraction <= 1.0:
            raise ValueError("Extreme-w fraction limit must lie in [0,1].")
        if not 0.0 < self.min_median_captured_mass <= 1.0:
            raise ValueError("Median captured-mass floor must lie in (0,1].")
        if not 1.0 <= self.max_maximum_captured_mass <= 1.10:
            raise ValueError("Captured-mass ceiling must be a small tolerance above one.")
        if not (
            0.0
            <= self.min_median_choice_rate
            < self.max_median_choice_rate
            <= 1.0
        ):
            raise ValueError("Choice-rate limits must form a valid interval.")
        if not (
            0.0
            < self.min_median_rt_q10
            <= self.min_median_rt_q50
            < self.max_median_rt_q50
            <= self.max_median_rt_q90
            <= self.response_deadline_seconds
        ):
            raise ValueError("RT quantile limits must be ordered inside the deadline.")


@dataclass(frozen=True)
class PriorPredictiveAudit:
    model_id: str
    condition: str
    design_digest: str
    sampling_scope: str
    sampled_parameter_names: tuple
    requested_draws: int
    valid_draws: int
    invalid_draws: int
    invalid_draw_rate: float
    extreme_w_fraction: float
    median_captured_mass: float
    minimum_captured_mass: float
    maximum_captured_mass: float
    median_choice_rate: Optional[float]
    median_rt_q10: Optional[float]
    median_rt_q50: Optional[float]
    median_rt_q90: Optional[float]
    invalid_reason_counts: Dict[str, int]
    passed: bool


def calibration_design_digest(designs: Sequence[CalibrationDesign]) -> str:
    """Hash only stimulus/cue/tie/missingness structure, never observed outcomes."""

    records = []
    for design in designs:
        record = sha256()
        record.update(design.condition.encode("utf-8"))
        for values in (
            design.stimulus,
            design.cue_white,
            design.signed_coherence,
            design.tie.astype(np.uint8),
            design.cue_evidence,
            design.likelihood_mask.astype(np.uint8),
        ):
            array = np.ascontiguousarray(values)
            record.update(str(array.dtype).encode("ascii"))
            record.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            record.update(array.tobytes())
        records.append(record.digest())
    digest = sha256()
    for record in sorted(records):
        digest.update(record)
    return digest.hexdigest()


def calibrate_primary_effects(
    designs: Sequence[CalibrationDesign],
    targets: EffectTargetSpec = EffectTargetSpec(),
    maximum_transformed_value: float = 12.0,
) -> Tuple[EffectCalibration, ...]:
    """Calibrate w/v0 coefficients to signed red-trial choice changes."""

    prepared = _prepare_predictions(designs)
    declarations = (
        ("cue_parallel_w", "gamma_w", "w", "parallel"),
        ("cue_parallel_vbias", "gamma_v0", "v0", "parallel"),
        ("cue_integrated_w", "b_w", "w", "integrated"),
        ("cue_integrated_vbias", "b_v", "v0", "integrated"),
    )
    results = []
    for model_id, parameter, locus, architecture in declarations:
        metric = lambda value: _effect_metric(  # noqa: E731 - scalar root function
            prepared[architecture], model_id, parameter, locus, value
        )
        maximum = float(metric(maximum_transformed_value))
        for level, target in targets.levels:
            if maximum < target:
                transformed = None
                native = None
                achieved = maximum
                reachable = False
            else:
                transformed = float(
                    brentq(
                        lambda value: metric(value) - target,
                        0.0,
                        maximum_transformed_value,
                        xtol=1e-12,
                        rtol=1e-12,
                    )
                )
                native = _native_effect_value(model_id, parameter, transformed)
                achieved = float(metric(transformed))
                reachable = True
            results.append(
                EffectCalibration(
                    model_id=model_id,
                    parameter=parameter,
                    locus=locus,
                    level=level,
                    target_choice_change=float(target),
                    transformed_value=transformed,
                    native_value=native,
                    achieved_choice_change=achieved,
                    reachable=reachable,
                )
            )
    return tuple(results)


def run_prior_predictive_audit(
    model: JointModel,
    condition: str,
    design_digest: str,
    included_mask: Optional[BoolArray] = None,
    policy: PriorPredictivePolicy = PriorPredictivePolicy(),
) -> PriorPredictiveAudit:
    """Sample a candidate prior without reading observed choice/RT values."""

    if model.cue_spec is None:
        raise ValueError("Prior-predictive cue audit requires a cue-locus model.")
    if included_mask is None:
        selected = np.ones(model.u.shape[0], dtype=bool)
    else:
        selected = np.asarray(included_mask, dtype=bool)
        if selected.shape != (model.u.shape[0],):
            raise ValueError("Included mask must match the model trial count.")
    if not np.any(selected):
        raise ValueError("At least one trial must be selected for the audit.")

    rng = np.random.Generator(np.random.MT19937(policy.seed))
    means = model.initial_free_parameters
    all_variances = np.concatenate(
        (
            model.hgf_prior.variances[model.hgf_prior.free_mask],
            model.response_prior.variances[model.response_prior.free_mask],
        )
    )
    if policy.sampling_scope == "all_free_parameters":
        sampled_mask = np.ones(means.size, dtype=bool)
    else:
        sampled_mask = np.asarray(
            [
                name.startswith("ddm.")
                and name.removeprefix("ddm.") in model.cue_spec.response_effects
                for name in model.free_parameter_names
            ],
            dtype=bool,
        )
    if not np.any(sampled_mask):
        raise ValueError("Prior-predictive scope selected no free parameters.")
    variances = np.where(sampled_mask, all_variances, 0.0)
    sampled_names = tuple(
        name for name, sampled in zip(model.free_parameter_names, sampled_mask) if sampled
    )
    extreme_count = 0
    trial_count = 0
    captured = []
    choice_rates = []
    rt_q10 = []
    rt_q50 = []
    rt_q90 = []
    invalid = 0
    invalid_reasons: Dict[str, int] = {}
    for draw in range(policy.draws):
        parameters = means + np.sqrt(variances) * rng.standard_normal(means.size)
        try:
            trialwise = _prior_trialwise(
                model, parameters, policy.ter_reference_rt_seconds
            )
            extreme_count += int(
                np.sum(
                    (trialwise.w[selected] < policy.extreme_w_threshold)
                    | (
                        trialwise.w[selected]
                        > 1.0 - policy.extreme_w_threshold
                    )
                )
            )
            trial_count += int(np.sum(selected))
            batch = simulate_ddm(
                trialwise,
                replicates=policy.replicates_per_draw,
                seed=policy.seed + draw + 1,
                decision_time_step=policy.decision_time_step,
                max_decision_time=policy.response_deadline_seconds,
            )
        except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError) as error:
            invalid += 1
            reason = "%s: %s" % (type(error).__name__, str(error))
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
            continue
        captured.extend(batch.captured_mass[selected].tolist())
        selected_choice = batch.choice[:, selected]
        selected_rt = batch.rt[:, selected]
        choice_rates.append(float(np.mean(selected_choice)))
        rt_q10.append(float(np.quantile(selected_rt, 0.10)))
        rt_q50.append(float(np.quantile(selected_rt, 0.50)))
        rt_q90.append(float(np.quantile(selected_rt, 0.90)))

    valid = policy.draws - invalid
    invalid_rate = invalid / policy.draws
    extreme_fraction = extreme_count / trial_count if trial_count else 1.0
    median_mass = float(np.median(captured)) if captured else 0.0
    minimum_mass = float(np.min(captured)) if captured else 0.0
    maximum_mass = float(np.max(captured)) if captured else 0.0
    median_choice = float(np.median(choice_rates)) if choice_rates else None
    median_q10 = float(np.median(rt_q10)) if rt_q10 else None
    median_q50 = float(np.median(rt_q50)) if rt_q50 else None
    median_q90 = float(np.median(rt_q90)) if rt_q90 else None
    passed = bool(
        valid > 0
        and invalid_rate <= policy.max_invalid_draw_rate
        and extreme_fraction <= policy.max_extreme_w_fraction
        and median_mass >= policy.min_median_captured_mass
        and maximum_mass <= policy.max_maximum_captured_mass
        and median_choice is not None
        and policy.min_median_choice_rate
        <= median_choice
        <= policy.max_median_choice_rate
        and median_q10 is not None
        and median_q10 >= policy.min_median_rt_q10
        and median_q50 is not None
        and policy.min_median_rt_q50
        <= median_q50
        <= policy.max_median_rt_q50
        and median_q90 is not None
        and median_q90 <= policy.max_median_rt_q90
    )
    return PriorPredictiveAudit(
        model_id=model.model_id,
        condition=str(condition),
        design_digest=str(design_digest),
        sampling_scope=policy.sampling_scope,
        sampled_parameter_names=sampled_names,
        requested_draws=policy.draws,
        valid_draws=valid,
        invalid_draws=invalid,
        invalid_draw_rate=float(invalid_rate),
        extreme_w_fraction=float(extreme_fraction),
        median_captured_mass=median_mass,
        minimum_captured_mass=minimum_mass,
        maximum_captured_mass=maximum_mass,
        median_choice_rate=median_choice,
        median_rt_q10=median_q10,
        median_rt_q50=median_q50,
        median_rt_q90=median_q90,
        invalid_reason_counts=dict(sorted(invalid_reasons.items())),
        passed=passed,
    )


def candidate_manifest(
    designs: Sequence[CalibrationDesign],
    calibrations: Sequence[EffectCalibration],
    audits: Sequence[PriorPredictiveAudit],
    targets: EffectTargetSpec,
    policy: PriorPredictivePolicy,
) -> dict:
    """Build a candidate manifest that cannot be mistaken for a frozen Gate."""

    condition_counts: Dict[str, int] = {}
    for design in designs:
        condition_counts[design.condition] = condition_counts.get(design.condition, 0) + 1
    return {
        "status": "candidate_not_frozen",
        "frozen_before_real_data_model_fit": False,
        "formulation_version": CUE_FORMULATION_VERSION,
        "candidate_prior_version": CUE_PRIOR_VERSION,
        "registry_digest": cue_registry_digest(),
        "design_contract": {
            "subjects": len(designs),
            "condition_counts": dict(sorted(condition_counts.items())),
            "outcomes_in_digest": False,
            "observed_choice_rt_used_for_calibration": False,
            "digest": calibration_design_digest(designs),
        },
        "effect_targets": asdict(targets),
        "calibration_anchor": {
            "hgf_free_transformed": {"omega_2": -3.0},
            "log_a_a": 0.0,
            "log_a_v": 0.0,
            "b_c": 0.0,
            "Ter_logit": 0.0,
            "ter_reference_rt_seconds": policy.ter_reference_rt_seconds,
            "test_trials_only": True,
            "red_trials_only": True,
        },
        "calibrations": [asdict(result) for result in calibrations],
        "prior_predictive_policy": asdict(policy),
        "prior_predictive_audits": [asdict(audit) for audit in audits],
        "all_effect_targets_reachable": bool(calibrations)
        and all(item.reachable for item in calibrations),
        "all_prior_predictive_audits_passed": bool(audits)
        and all(item.passed for item in audits),
        "note": (
            "This file is a pre-freeze candidate. Review and version the prior "
            "before creating a recovery manifest; do not use it as a Gate pass."
        ),
    }


def manifest_digest(manifest: dict) -> str:
    digest_payload = dict(manifest)
    digest_payload.pop("manifest_digest", None)
    payload = json.dumps(
        digest_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _prepare_predictions(
    designs: Sequence[CalibrationDesign],
) -> Dict[str, Tuple[Tuple[CalibrationDesign, FloatArray], ...]]:
    if not designs:
        raise ValueError("At least one calibration design is required.")
    hgf_parameters = transform_ehgf_binary(cue_hgf_prior().means)
    parallel = []
    integrated = []
    for design in designs:
        if design.stimulus.shape != (380,):
            raise ValueError("Calibration designs must contain 380 trials.")
        global_hgf = cue_blind_binary_hgf(
            design.stimulus, hgf_parameters, design.tie
        )
        cue_hgf = cue_binary_hgf(design.u, hgf_parameters, design.tie)
        parallel.append((design, global_hgf.muhat[:, 0]))
        integrated.append((design, cue_hgf.active.muhat[:, 0]))
    return {"parallel": tuple(parallel), "integrated": tuple(integrated)}


def _effect_metric(
    prepared: Sequence[Tuple[CalibrationDesign, FloatArray]],
    model_id: str,
    parameter: str,
    locus: str,
    transformed_value: float,
) -> float:
    changes = []
    for design, prediction in prepared:
        base = _trialwise_for_effect(
            design, prediction, model_id, parameter, 0.0
        )
        effect = _trialwise_for_effect(
            design, prediction, model_id, parameter, transformed_value
        )
        red = design.cue_evidence != 0.0
        test = np.arange(design.stimulus.size) >= 100
        if locus == "w":
            selected = red & design.tie & test & design.likelihood_mask
        else:
            selected = red & ~design.tie & test & design.likelihood_mask
        if not np.any(selected):
            raise ValueError("Calibration design has no selected %s trials." % locus)
        base_probability = choice_one_probability(base)
        effect_probability = choice_one_probability(effect)
        delta = design.cue_evidence[selected] * (
            effect_probability[selected] - base_probability[selected]
        )
        changes.extend(delta.tolist())
    return float(np.mean(changes))


def _trialwise_for_effect(
    design: CalibrationDesign,
    prediction: FloatArray,
    model_id: str,
    parameter: str,
    transformed_value: float,
):
    prior = ddm_prior(model_id)
    transformed = prior.means.copy()
    transformed[prior.names.index(parameter)] = transformed_value
    native = transform_cue_ddm(transformed, np.array([0.6]), prior.names)
    architecture = "parallel" if model_id.startswith("cue_parallel") else "integrated"
    return trialwise_cue_ddm(
        architecture,
        design.stimulus,
        prediction,
        design.cue_evidence,
        native,
        design.signed_coherence,
        design.tie,
    )


def _native_effect_value(model_id: str, parameter: str, value: float) -> float:
    prior = ddm_prior(model_id)
    transformed = prior.means.copy()
    transformed[prior.names.index(parameter)] = value
    native = transform_cue_ddm(transformed, np.array([0.6]), prior.names)
    return float(getattr(native, parameter))


def _prior_trialwise(
    model: JointModel,
    free_parameters: FloatArray,
    ter_reference_rt_seconds: float,
) -> TrialwiseDDM:
    """Build generative arrays without evaluating any response likelihood."""

    hgf_transformed, response_transformed = model.expand_free(free_parameters)
    hgf_parameters = transform_ehgf_binary(hgf_transformed)
    if model.cue_spec.perceptual_model == "cue_blind":
        hgf = cue_blind_binary_hgf(model.u[:, 0], hgf_parameters, model.tie)
        prediction = hgf.muhat[:, 0]
    else:
        hgf = cue_binary_hgf(model.u, hgf_parameters, model.tie)
        prediction = hgf.active.muhat[:, 0]
    native = transform_cue_ddm(
        response_transformed,
        np.array([ter_reference_rt_seconds]),
        model.response_prior.names,
    )
    return trialwise_cue_ddm(
        model.cue_spec.architecture,
        model.u[:, 0],
        prediction,
        model.cue_evidence,
        native,
        model.u[:, 2],
        model.tie,
    )
