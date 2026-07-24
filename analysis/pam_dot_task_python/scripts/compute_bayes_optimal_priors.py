"""Compute each subject's Bayes-optimal perceptual prior mean.

Writes one JSON table holding, per subject, the shared two-stream Bayes-optimal
``omega_2`` that becomes that subject's prior mean, plus the white-only and
red-only optima used to judge how much the Gate-A shared-parameter assumption
is compromising.

Only stimulus and cue inputs are read. Responses never enter, so this can be
run before any joint fit and does not feed behaviour back into the prior.

Usage::

    PYTHONPATH=src python3 scripts/compute_bayes_optimal_priors.py OUTPUT.json [DATA_GLOB]
"""

import glob
import json
import os
import sys
import time

import numpy as np

from pam_dot_task_python import cue_hgf_prior, load_subject
from pam_dot_task_python.bayes_optimal import per_cue_bayes_optimal


DEFAULT_GLOB = "/Users/utsumikensuke/Research/dot_task/analysis/real_data/*.csv"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    output_path = sys.argv[1]
    pattern = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_GLOB

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit("No subject files matched %r." % pattern)
    prior = cue_hgf_prior()
    omega_index = list(prior.names).index("omega_2")

    records = []
    for path in paths:
        subject_id = os.path.splitext(os.path.basename(path))[0]
        start = time.time()
        subject = load_subject(path)
        try:
            fits = per_cue_bayes_optimal(subject.u, prior)
            record = {
                "subject_id": subject_id,
                "status": "ok",
                "condition": str(subject.audit["condition"].iloc[0])
                if "condition" in subject.audit.columns
                else None,
                "omega_2_shared": fits["both_cues"].omega_2,
                "omega_2_white": fits["white"].omega_2,
                "omega_2_red": fits["red"].omega_2,
                "cue_gap": fits["white"].omega_2 - fits["red"].omega_2,
                "input_log_likelihood_shared": fits["both_cues"].input_log_likelihood,
                "convergence": {
                    name: fit.convergence_reason for name, fit in fits.items()
                },
                "iterations": {name: fit.iterations for name, fit in fits.items()},
            }
        except Exception as error:  # noqa: BLE001 - recorded, never swallowed
            record = {
                "subject_id": subject_id,
                "status": "failed",
                "error_message": repr(error),
            }
        record["seconds"] = time.time() - start
        records.append(record)
        print(
            "%-60s %s" % (subject_id[:60], record.get("omega_2_shared", record["status"])),
            flush=True,
        )

    successful = [r for r in records if r["status"] == "ok"]
    shared = np.array([r["omega_2_shared"] for r in successful])
    gap = np.array([r["cue_gap"] for r in successful])
    payload = {
        "prior_parameter": "omega_2",
        "prior_index": omega_index,
        "default_prior_mean": float(prior.means[omega_index]),
        "prior_variance": float(prior.variances[omega_index]),
        "subjects": len(records),
        "successful": len(successful),
        "summary": {
            "shared_mean": float(np.mean(shared)),
            "shared_sd": float(np.std(shared, ddof=1)),
            "shared_min": float(np.min(shared)),
            "shared_max": float(np.max(shared)),
            "cue_gap_mean": float(np.mean(gap)),
            "cue_gap_sd": float(np.std(gap, ddof=1)),
            "cue_gap_min": float(np.min(gap)),
            "cue_gap_max": float(np.max(gap)),
            "subjects_with_white_below_red": int(np.sum(gap < 0)),
        },
        "records": records,
    }
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print("\nwrote %s" % output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
