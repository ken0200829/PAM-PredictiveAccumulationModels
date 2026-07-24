"""Run the declared recovery grid incrementally, writing after every case.

Each finished case is written to disk immediately and the run can be resumed:
rerunning skips cases whose seed already appears in the output file.

The template model is built with the subject's own Bayes-optimal ``omega_2``
prior mean, matching the joint-fit workflow, so recovery is assessed under the
same prior the main analysis will use rather than the hardcoded default.

Usage::

    PYTHONPATH=src python3 scripts/run_recovery_pilot.py OUTPUT.json [CASES] [GRID]

``CASES`` limits how many grid rows to attempt in this session.
``GRID`` is one of ``ddm_v_v2``, ``ddm_w_v3``, or ``ddm_full_c_v3``.
"""

import json
import os
import sys
import time

import numpy as np

from pam_dot_task_python import JointModel, cue_hgf_prior, load_subject
from pam_dot_task_python.bayes_optimal import bayes_optimal_prior
from pam_dot_task_python.gates import (
    RECOVERY_CRITERIA_V1,
    RECOVERY_GRID_V2,
    RECOVERY_GRID_W_V3,
    RECOVERY_GRID_FULL_C_V3,
    evaluate_recovery,
    prior_standard_deviations,
    recovery_verdict,
)
from pam_dot_task_python.recovery import RecoveryResult, run_recovery_case, summarize_recovery


FORMULATION = "tie-hold-hgf__tie-direction-zero-v-zero__1.0.0"
GRIDS = {
    "ddm_v_v2": RECOVERY_GRID_V2,
    "ddm_w_v3": RECOVERY_GRID_W_V3,
    "ddm_full_c_v3": RECOVERY_GRID_FULL_C_V3,
}
SUBJECT = (
    "/Users/utsumikensuke/Research/dot_task/analysis/real_data/"
    "normal_cb_dot_task_20260526_013329_6717f0ac88d9f27d9b79af31.csv"
)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    output_path = sys.argv[1]
    grid_key = sys.argv[3] if len(sys.argv) > 3 else "ddm_v_v2"
    if grid_key not in GRIDS:
        raise SystemExit("Unknown grid %r; choose %s." % (grid_key, sorted(GRIDS)))
    grid = GRIDS[grid_key]
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else len(grid.seeds)

    completed = {}
    if os.path.exists(output_path):
        with open(output_path, "r") as handle:
            previous = json.load(handle)
        if previous.get("grid_version") != grid.version:
            raise SystemExit("Existing output belongs to a different recovery grid.")
        if previous.get("formulation_version") != FORMULATION:
            raise SystemExit("Existing output belongs to a different formulation.")
        completed = {
            int(case["seed"]): case for case in previous["cases"]
        }
        print("resuming with %d completed cases" % len(completed), flush=True)

    subject = load_subject(SUBJECT)
    hgf_prior, bo_fit = bayes_optimal_prior(subject.u, cue_hgf_prior())
    print(
        "subject Bayes-optimal omega_2 prior mean = %.4f" % bo_fit.omega_2,
        flush=True,
    )
    template = JointModel(
        subject.u, subject.y, model_id=grid.model_id, hgf_prior=hgf_prior
    )
    if tuple(template.free_parameter_names) != grid.parameter_names:
        raise SystemExit(
            "Model free parameters %s do not match the declared grid %s."
            % (template.free_parameter_names, grid.parameter_names)
        )
    prior_sd = prior_standard_deviations(template)

    attempted = 0
    for truth, seed in zip(grid.truths, grid.seeds):
        if seed in completed:
            continue
        if attempted >= budget:
            break
        attempted += 1
        start = time.time()
        record = {"seed": int(seed), "truth": list(map(float, truth))}
        try:
            case = run_recovery_case(
                template, np.asarray(truth, dtype=float), int(seed)
            )
            ter = case.dataset.model.ter_diagnostic(
                case.estimated_free_parameters
            )
            record.update(
                {
                    "status": "ok",
                    "estimation_scale_truth": [
                        float(v) for v in case.dataset.estimation_scale_truth
                    ],
                    "estimate": [float(v) for v in case.estimated_free_parameters],
                    "error": [float(v) for v in case.error],
                    "neg_log_joint": float(case.fit.optimization.value_minimum),
                    "iterations": int(case.fit.optimization.iterations),
                    "resets": int(case.fit.optimization.reset_count),
                    "convergence_reason": case.fit.optimization.convergence_reason,
                    "ter_diagnostic": {
                        "minimum_fitted_rt": ter.minimum_fitted_rt,
                        "ter": ter.ter,
                        "ter_fraction_of_minimum_rt": ter.ter_fraction_of_minimum_rt,
                        "decision_time_slack": ter.decision_time_slack,
                        "decision_time_fraction": ter.decision_time_fraction,
                        "transformed_ter": ter.transformed_ter,
                    },
                }
            )
        except Exception as error:  # noqa: BLE001 - recorded, never swallowed
            record.update({"status": "failed", "error_message": repr(error)})
        record["seconds"] = time.time() - start
        completed[seed] = record
        _write(output_path, completed, template, prior_sd, grid)
        print(
            "seed %d %s in %.0f s (%d/%d done)"
            % (
                seed,
                record["status"],
                record["seconds"],
                len(completed),
                len(grid.seeds),
            ),
            flush=True,
        )
    return 0


def _write(output_path, completed, template, prior_sd, grid) -> None:
    cases = [completed[seed] for seed in sorted(completed)]
    payload = {
        "grid_version": grid.version,
        "criteria_version": RECOVERY_CRITERIA_V1.version,
        "formulation_version": FORMULATION,
        "model_id": grid.model_id,
        "parameter_names": list(grid.parameter_names),
        "declared_cases": len(grid.seeds),
        "completed_cases": len(cases),
        "cases": cases,
    }
    successful = [case for case in cases if case["status"] == "ok"]
    if len(successful) >= 3:
        truth = np.vstack([case["estimation_scale_truth"] for case in successful])
        estimate = np.vstack([case["estimate"] for case in successful])
        summary = summarize_recovery(
            truth, estimate, grid.parameter_names
        )
        evaluated = evaluate_recovery(
            RecoveryResult(
                parameter_names=grid.parameter_names,
                cases=(),
                summary=summary,
            ),
            prior_sd,
        )
        serialized_summary = json.loads(evaluated.to_json(orient="records"))
        verdict = recovery_verdict(evaluated)
        if len(cases) == len(grid.seeds) and len(successful) == len(grid.seeds):
            payload["summary"] = serialized_summary
            payload["verdict"] = verdict
        else:
            payload["interim_summary"] = serialized_summary
            payload["interim_verdict"] = verdict
            payload["interim_warning"] = (
                "Interim only. The declared grid is not complete, so this verdict "
                "does not satisfy the recovery gate."
            )
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
