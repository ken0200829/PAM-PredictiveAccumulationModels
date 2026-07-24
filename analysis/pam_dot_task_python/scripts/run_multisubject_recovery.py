"""Run one recovery grid across several subjects' input designs.

Codex diagnostic 1: is ``omega_2``'s weak recovery specific to one subject's
stimulus/cue sequence, or structural across designs? This runner refits the
same declared grid on each subject, every subject using its OWN Bayes-optimal
``omega_2`` prior mean and its own tie/missing masks, and reports the per-
subject recovery correlation so the two explanations can be told apart.

Only ``u`` (and the frozen grid truths) drive generation; each subject's
generated responses are simulated from the model, so this isolates
identifiability from real-response quality exactly as the single-subject run
does -- just repeated across designs.

Each (subject, case) result is written immediately and the run resumes by
skipping (subject_id, seed) pairs already present.

Usage::

    PYTHONPATH=src python3 scripts/run_multisubject_recovery.py \\
        OUTPUT.json GRID_NAME SUBJECT_CSV [SUBJECT_CSV ...]

GRID_NAME is one of: ddm_w_v3, ddm_full_c_v3, ddm_v_v2.
"""

import json
import os
import sys
import time

import numpy as np

from pam_dot_task_python import JointModel, cue_hgf_prior, load_subject
from pam_dot_task_python.bayes_optimal import bayes_optimal_prior
from pam_dot_task_python.gates import (
    RECOVERY_GRID_V2,
    RECOVERY_GRID_W_V3,
    RECOVERY_GRID_FULL_C_V3,
)
from pam_dot_task_python.recovery import run_recovery_case


GRIDS = {
    "ddm_w_v3": RECOVERY_GRID_W_V3,
    "ddm_full_c_v3": RECOVERY_GRID_FULL_C_V3,
    "ddm_v_v2": RECOVERY_GRID_V2,
}


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    output_path = sys.argv[1]
    grid_name = sys.argv[2]
    subject_paths = sys.argv[3:]
    if grid_name not in GRIDS:
        raise SystemExit("Unknown grid %r; choose from %s." % (grid_name, list(GRIDS)))
    grid = GRIDS[grid_name]

    done = set()
    subjects_out = {}
    if os.path.exists(output_path):
        with open(output_path, "r") as handle:
            prior_state = json.load(handle)
        for entry in prior_state["subjects"]:
            subjects_out[entry["subject_id"]] = entry
            for case in entry["cases"]:
                done.add((entry["subject_id"], int(case["seed"])))
        print("resuming with %d (subject, case) results" % len(done), flush=True)

    for path in subject_paths:
        subject_id = os.path.splitext(os.path.basename(path))[0]
        subject = load_subject(path)
        hgf_prior, bo_fit = bayes_optimal_prior(subject.u, cue_hgf_prior())
        template = JointModel(
            subject.u, subject.y, model_id=grid.model_id, hgf_prior=hgf_prior
        )
        if tuple(template.free_parameter_names) != grid.parameter_names:
            raise SystemExit(
                "Model free parameters %s do not match grid %s."
                % (template.free_parameter_names, grid.parameter_names)
            )
        entry = subjects_out.setdefault(
            subject_id,
            {
                "subject_id": subject_id,
                "condition": str(subject.audit["condition"].iloc[0])
                if "condition" in subject.audit.columns
                else None,
                "bayes_optimal_omega_2": float(bo_fit.omega_2),
                "cases": [],
            },
        )
        for truth, seed in zip(grid.truths, grid.seeds):
            if (subject_id, int(seed)) in done:
                continue
            start = time.time()
            record = {"seed": int(seed)}
            try:
                case = run_recovery_case(
                    template, np.asarray(truth, dtype=float), int(seed)
                )
                record.update(
                    {
                        "status": "ok",
                        "estimation_scale_truth": [
                            float(v) for v in case.dataset.estimation_scale_truth
                        ],
                        "estimate": [
                            float(v) for v in case.estimated_free_parameters
                        ],
                    }
                )
            except Exception as error:  # noqa: BLE001 - recorded, never swallowed
                record.update({"status": "failed", "error_message": repr(error)})
            record["seconds"] = time.time() - start
            entry["cases"].append(record)
            done.add((subject_id, int(seed)))
            _write(output_path, grid, grid_name, subjects_out)
            print(
                "%-12s seed %d %s %.0fs (%d cases)"
                % (
                    subject_id.split("_dot_task")[0],
                    seed,
                    record["status"],
                    record["seconds"],
                    len(entry["cases"]),
                ),
                flush=True,
            )
    _write(output_path, grid, grid_name, subjects_out)
    _print_summary(grid, subjects_out)
    return 0


def _omega_correlation(grid, entry):
    omega_index = grid.parameter_names.index("hgf.omega_2")
    ok = [c for c in entry["cases"] if c["status"] == "ok"]
    if len(ok) < 3:
        return None, len(ok)
    truth = np.array([c["estimation_scale_truth"][omega_index] for c in ok])
    estimate = np.array([c["estimate"][omega_index] for c in ok])
    if truth.std() == 0 or estimate.std() == 0:
        return None, len(ok)
    return float(np.corrcoef(truth, estimate)[0, 1]), len(ok)


def _write(output_path, grid, grid_name, subjects_out):
    subjects = list(subjects_out.values())
    per_subject = []
    for entry in subjects:
        correlation, n = _omega_correlation(grid, entry)
        per_subject.append(
            {
                "subject_id": entry["subject_id"],
                "condition": entry["condition"],
                "bayes_optimal_omega_2": entry["bayes_optimal_omega_2"],
                "omega_2_correlation": correlation,
                "successful_cases": n,
            }
        )
    payload = {
        "grid_name": grid_name,
        "grid_version": grid.version,
        "model_id": grid.model_id,
        "parameter_names": list(grid.parameter_names),
        "n_subjects": len(subjects),
        "per_subject_omega_2": per_subject,
        "subjects": subjects,
    }
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)


def _print_summary(grid, subjects_out):
    print("\n=== per-subject omega_2 recovery ===", flush=True)
    correlations = []
    for entry in subjects_out.values():
        correlation, n = _omega_correlation(grid, entry)
        label = entry["condition"] or "?"
        text = "n/a" if correlation is None else "%.3f" % correlation
        print(
            "  %-12s omega_2 corr=%-6s (n=%d, BO prior=%.2f)"
            % (label, text, n, entry["bayes_optimal_omega_2"]),
            flush=True,
        )
        if correlation is not None:
            correlations.append(correlation)
    if correlations:
        arr = np.array(correlations)
        print(
            "  across %d subjects: mean %.3f  min %.3f  max %.3f  SD %.3f"
            % (arr.size, arr.mean(), arr.min(), arr.max(), arr.std(ddof=1)),
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
