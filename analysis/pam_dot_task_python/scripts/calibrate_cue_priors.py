"""Create an outcome-redacted cue-effect/prior-predictive candidate manifest.

Only stimulus, cue, coherence, tie, condition, and the pre-existing
missingness mask enter calibration. Observed choices and RT values are never
passed to the generative audit.

Usage::

    PYTHONPATH=src python3 -B scripts/calibrate_cue_priors.py DATA_DIR OUTPUT.json
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from pam_dot_task_python import (
    CalibrationDesign,
    EffectTargetSpec,
    JointModel,
    PriorPredictivePolicy,
    calibrate_primary_effects,
    calibration_design_digest,
    candidate_manifest,
    load_subject,
    manifest_digest,
    run_prior_predictive_audit,
)
from pam_dot_task_python.config import (
    CUE_EFFECT_STRONG_TRANSFORMED,
    CUE_PRIOR_CALIBRATION_DESIGN_DIGEST,
)


EXPECTED_SUBJECTS = 37
AUDIT_MODELS = ("cue_parallel_w_vbias", "cue_integrated_w_vbias")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--decision-time-step", type=float, default=0.01)
    arguments = parser.parse_args()

    paths = sorted(arguments.data_directory.glob("*_dot_task_*.csv"))
    if len(paths) != EXPECTED_SUBJECTS:
        raise SystemExit(
            "Expected %d top-level dot-task CSVs, found %d."
            % (EXPECTED_SUBJECTS, len(paths))
        )
    subjects = [load_subject(path) for path in paths]
    designs = tuple(CalibrationDesign.from_subject(subject) for subject in subjects)
    all_designs_digest = calibration_design_digest(designs)
    if all_designs_digest != CUE_PRIOR_CALIBRATION_DESIGN_DIGEST:
        raise SystemExit(
            "The outcome-redacted design digest differs from the candidate prior calibration."
        )
    conditions = sorted({design.condition for design in designs})
    if conditions != ["normal", "normal_cb", "reverse", "reverse_cb"]:
        raise SystemExit("The four counterbalanced conditions are not all present.")

    targets = EffectTargetSpec()
    calibrations = calibrate_primary_effects(designs, targets)
    strong_values = {
        result.parameter: result.transformed_value
        for result in calibrations
        if result.level == "strong"
    }
    for parameter, expected in CUE_EFFECT_STRONG_TRANSFORMED.items():
        if not np.isclose(strong_values.get(parameter), expected, rtol=0.0, atol=1e-12):
            raise SystemExit(
                "Configured %s prior no longer matches the outcome-redacted calibration."
                % parameter
            )
    policy = replace(
        PriorPredictivePolicy(),
        draws=arguments.draws,
        replicates_per_draw=arguments.replicates,
        decision_time_step=arguments.decision_time_step,
    )

    representatives = {}
    for design in designs:
        representatives.setdefault(design.condition, design)
    audits = []
    for condition in conditions:
        design = representatives[condition]
        redacted_y = np.full((design.stimulus.size, 2), np.nan)
        design_hash = calibration_design_digest((design,))
        for model_id in AUDIT_MODELS:
            model = JointModel(
                design.u,
                redacted_y,
                model_id=model_id,
                tie=design.tie,
                cue_evidence=design.cue_evidence,
            )
            audits.append(
                run_prior_predictive_audit(
                    model,
                    condition=condition,
                    design_digest=design_hash,
                    included_mask=design.likelihood_mask,
                    policy=policy,
                )
            )

    payload = candidate_manifest(designs, calibrations, audits, targets, policy)
    payload["prior_predictive_design_selection"] = {
        "rule": "lexicographically_first_csv_within_each_condition",
        "identifiers_recorded": False,
        "models": list(AUDIT_MODELS),
    }
    payload["configured_effect_priors_match_calibration"] = True
    payload["manifest_digest"] = manifest_digest(payload)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    print(
        "candidate written: %d designs, %d calibrations, %d audits"
        % (len(designs), len(calibrations), len(audits))
    )
    print(
        "reachable=%s prior_predictive_pass=%s"
        % (
            payload["all_effect_targets_reachable"],
            payload["all_prior_predictive_audits_passed"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
