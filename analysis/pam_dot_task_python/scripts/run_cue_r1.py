"""Freeze, run, resume, and summarize cue-locus Gate R1.

Commands::

    PYTHONPATH=src python3 -B scripts/run_cue_r1.py freeze DATA_DIR RUN_DIR
    PYTHONPATH=src python3 -B scripts/run_cue_r1.py run DATA_DIR RUN_DIR [options]
    PYTHONPATH=src python3 -B scripts/run_cue_r1.py summarize RUN_DIR
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pam_dot_task_python import CalibrationDesign, calibration_design_digest, load_subject
from pam_dot_task_python.config import cue_hgf_prior, ddm_prior
from pam_dot_task_python.cue_r1 import (
    ARCHITECTURE_MODELS,
    R1_HGF_CACHE_ADDENDUM_VERSION,
    R1_HGF_CACHE_SIZE,
    R1_BMS_SAMPLES,
    R1_REPETITIONS,
    cue_r1_cells,
    cue_r1_manifest,
    cue_r1_manifest_digest,
    cue_r1_execution_addendum_digest,
    fit_r1_candidate,
    optimizer_from_manifest,
    r1_bms,
    simulate_r1_subject,
    validate_r1_designs,
)
from pam_dot_task_python.prior_predictive import manifest_digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("data_directory", type=Path)
    freeze.add_argument("run_directory", type=Path)
    run = commands.add_parser("run")
    run.add_argument("data_directory", type=Path)
    run.add_argument("run_directory", type=Path)
    run.add_argument("--phase", choices=("primary", "weak", "all"), default="primary")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--max-tasks", type=int)
    run.add_argument("--max-iterations", type=int, default=100)
    run.add_argument("--ridders-min-steps", type=int, default=10)
    run.add_argument("--hgf-cache-size", type=int, default=0)
    run.add_argument("--map-only", action="store_true")
    summarize = commands.add_parser("summarize")
    summarize.add_argument("run_directory", type=Path)
    summarize.add_argument("--bms-samples", type=int, default=R1_BMS_SAMPLES)
    arguments = parser.parse_args()
    if arguments.command == "freeze":
        return freeze_manifest(arguments.data_directory, arguments.run_directory)
    if arguments.command == "run":
        return run_gate(arguments)
    return summarize_gate(arguments.run_directory, arguments.bms_samples)


def _subjects(data_directory):
    paths = sorted(data_directory.glob("*_dot_task_*.csv"))
    subjects = [load_subject(path) for path in paths]
    designs = tuple(CalibrationDesign.from_subject(subject) for subject in subjects)
    validate_r1_designs(designs)
    return paths, subjects, designs


def freeze_manifest(data_directory, run_directory):
    _, _, designs = _subjects(data_directory)
    prior_path = Path(__file__).resolve().parents[1] / "manifests" / "cue_prior_candidate_0.2.0.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior["manifest_digest"] != manifest_digest(prior):
        raise SystemExit("Candidate prior manifest digest is invalid.")
    manifest = cue_r1_manifest(prior["manifest_digest"])
    if manifest["design_digest"] != calibration_design_digest(designs):
        raise SystemExit("Frozen design digest does not match the 37 inputs.")
    run_directory.mkdir(parents=True, exist_ok=True)
    destination = run_directory / "manifest.json"
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous != manifest:
            raise SystemExit("A different manifest already exists in the run directory.")
    else:
        _atomic_json(destination, manifest)
    print(manifest["manifest_digest"])
    return 0


def run_gate(arguments):
    paths, _, designs = _subjects(arguments.data_directory)
    manifest = _read_manifest(arguments.run_directory)
    if manifest["design_digest"] != calibration_design_digest(designs):
        raise SystemExit("Run inputs differ from the frozen Gate manifest.")
    addendum_digest = _validated_cache_addendum(
        arguments.run_directory,
        manifest,
        arguments.hgf_cache_size,
    )
    frozen_optimizer = manifest["optimizer"]
    nonstandard = (
        arguments.max_iterations != frozen_optimizer["max_iterations"]
        or arguments.ridders_min_steps != frozen_optimizer["ridders_min_steps"]
        or arguments.map_only
    )
    output_root = (
        arguments.run_directory / _smoke_directory(arguments)
        if nonstandard
        else arguments.run_directory / "tasks"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    cells = _selected_cells(arguments.phase)
    pending = []
    for cell in cells:
        for repetition in range(R1_REPETITIONS):
            for subject_index, path in enumerate(paths):
                destination = output_root / _task_filename(cell.identifier, repetition, subject_index)
                if not destination.exists():
                    pending.append((cell, repetition, subject_index, path, destination))
    if arguments.max_tasks is not None:
        pending = pending[: arguments.max_tasks]
    print(
        "pending=%d output=%s manifest=%s"
        % (len(pending), output_root.name, manifest["manifest_digest"]),
        flush=True,
    )
    if arguments.workers <= 1:
        for task in pending:
            payload = _run_task(
                task[:-1],
                arguments.max_iterations,
                arguments.ridders_min_steps,
                not arguments.map_only,
                manifest["manifest_digest"],
                arguments.hgf_cache_size,
                addendum_digest,
            )
            _atomic_json(task[-1], payload)
            _print_task(payload)
    else:
        with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
            futures = {
                executor.submit(
                    _run_task,
                    task[:-1],
                    arguments.max_iterations,
                    arguments.ridders_min_steps,
                    not arguments.map_only,
                    manifest["manifest_digest"],
                    arguments.hgf_cache_size,
                    addendum_digest,
                ): task[-1]
                for task in pending
            }
            for future in as_completed(futures):
                payload = future.result()
                _atomic_json(futures[future], payload)
                _print_task(payload)
    return 0


def _run_task(
    task,
    max_iterations,
    ridders_min_steps,
    compute_lme,
    manifest_digest_value,
    hgf_cache_size,
    execution_addendum_digest,
):
    cell, repetition, subject_index, path = task
    start = time.time()
    subject = load_subject(path)
    dataset = simulate_r1_subject(subject, cell, repetition, subject_index)
    options = optimizer_from_manifest(max_iterations, ridders_min_steps)
    fits = []
    for model_id in ARCHITECTURE_MODELS[cell.architecture]:
        fits.append(
            asdict(
                fit_r1_candidate(
                    dataset,
                    model_id,
                    optimizer_options=options,
                    compute_lme=compute_lme,
                    hgf_cache_size=hgf_cache_size,
                )
            )
        )
    design = CalibrationDesign.from_subject(subject)
    return {
        "manifest_digest": manifest_digest_value,
        "execution_addendum_digest": execution_addendum_digest,
        "hgf_cache_size": hgf_cache_size,
        "mode": (
            "formal"
            if compute_lme and max_iterations == 100 and ridders_min_steps == 10
            else "smoke"
        ),
        "cell": asdict(cell) | {"identifier": cell.identifier},
        "repetition": repetition,
        "subject_index": subject_index,
        "subject_id": subject.subject_id,
        "condition": subject.condition.name,
        "design_digest": calibration_design_digest((design,)),
        "generation_seed": dataset.seed,
        "generating_parameter_names": list(dataset.model.free_parameter_names),
        "generative_truth": dataset.generative_free_parameters.tolist(),
        "estimation_scale_truth": dataset.estimation_scale_truth.tolist(),
        "generated_minimum_rt": float(np.nanmin(dataset.model.y[:, 0])),
        "generated_captured_mass_min": float(np.min(dataset.simulation.captured_mass)),
        "generated_captured_mass_median": float(np.median(dataset.simulation.captured_mass)),
        "fits": fits,
        "seconds": time.time() - start,
    }


def summarize_gate(run_directory, bms_samples):
    manifest = _read_manifest(run_directory)
    task_root = run_directory / "tasks"
    summaries = []
    for cell in cue_r1_cells():
        for repetition in range(R1_REPETITIONS):
            records = []
            for subject_index in range(37):
                path = task_root / _task_filename(cell.identifier, repetition, subject_index)
                if path.exists():
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if record.get("manifest_digest") != manifest["manifest_digest"]:
                        raise SystemExit("Task manifest digest mismatch: %s" % path)
                    records.append(record)
            if len(records) != 37:
                continue
            models = ARCHITECTURE_MODELS[cell.architecture]
            lme = np.full((37, len(models)), np.nan)
            for row, record in enumerate(sorted(records, key=lambda item: item["subject_index"])):
                by_model = {fit["model_id"]: fit for fit in record["fits"]}
                for column, model_id in enumerate(models):
                    lme[row, column] = by_model[model_id]["lme"]
            if np.any(~np.isfinite(lme)):
                summaries.append({
                    "cell": asdict(cell) | {"identifier": cell.identifier},
                    "repetition": repetition,
                    "status": "incomplete_finite_lme",
                })
                continue
            seed = int(manifest["seed"] + repetition + 1000 * cue_r1_cells().index(cell))
            result = r1_bms(
                lme,
                models,
                [record["subject_id"] for record in sorted(records, key=lambda item: item["subject_index"])],
                seed=seed,
                samples=bms_samples,
            )
            summaries.append({
                "cell": asdict(cell) | {"identifier": cell.identifier},
                "repetition": repetition,
                "status": "ok",
                "bms": result,
                "winner_locus": _model_locus(result["winner"]),
                "subject_exact_model_winner_rate": float(
                    np.mean(np.asarray(models)[np.argmax(lme, axis=1)] == cell.generating_model)
                ),
            })
    cell_summaries = _summarize_cells(summaries)
    parameter_recovery = _summarize_parameter_recovery(
        task_root, manifest["success_criteria"]
    )
    formal_bms = bms_samples == manifest["bms"]["samples"]
    verdict = (
        _gate_verdict(manifest, cell_summaries, parameter_recovery)
        if formal_bms
        else {
            "status": "not_evaluable",
            "gate_passed": None,
            "reason": "A non-frozen BMS sample count was used for sensitivity only.",
        }
    )
    payload = {
        "manifest_digest": manifest["manifest_digest"],
        "bms_samples": bms_samples,
        "formal_bms_samples": formal_bms,
        "complete_group_repetitions": sum(item["status"] == "ok" for item in summaries),
        "declared_group_repetitions": len(cue_r1_cells()) * R1_REPETITIONS,
        "repetitions": summaries,
        "cells": cell_summaries,
        "model_recovery_confusion": _confusion_payload(cell_summaries),
        "parameter_recovery": parameter_recovery,
        "gate_verdict": verdict,
    }
    destination = (
        run_directory / "summary.json"
        if formal_bms
        else run_directory / "smoke" / ("summary_bms_%d.json" % bms_samples)
    )
    _atomic_json(destination, payload)
    print(
        "complete_group_repetitions=%d/%d verdict=%s"
        % (
            payload["complete_group_repetitions"],
            payload["declared_group_repetitions"],
            verdict["status"],
        )
    )
    return 0


def _selected_cells(phase):
    cells = cue_r1_cells()
    if phase == "all":
        return cells
    if phase == "weak":
        return tuple(cell for cell in cells if cell.effect_level == "weak")
    return tuple(cell for cell in cells if cell.primary_gate)


def _smoke_directory(arguments):
    mode = "map_only" if arguments.map_only else "lme"
    cache = (
        "_hgf%d" % arguments.hgf_cache_size
        if arguments.hgf_cache_size > 0
        else ""
    )
    return Path("smoke") / (
        "%s_i%d_r%d%s"
        % (
            mode,
            arguments.max_iterations,
            arguments.ridders_min_steps,
            cache,
        )
    )


def _model_locus(model_id):
    if model_id == "cue_history_w":
        return "null"
    has_w = model_id.endswith("_w") or "_w_vbias" in model_id
    has_v = "vbias" in model_id
    if has_w and has_v:
        return "w_v0"
    if has_w:
        return "w"
    if has_v:
        return "v0"
    raise ValueError("Unknown Gate R1 model: %s" % model_id)


def _summarize_cells(repetitions):
    rows = []
    for cell in cue_r1_cells():
        selected = [
            item
            for item in repetitions
            if item["status"] == "ok"
            and item["cell"]["identifier"] == cell.identifier
        ]
        winner_counts = {locus: 0 for locus in ("null", "w", "v0", "w_v0")}
        for item in selected:
            winner_counts[item["winner_locus"]] += 1
        complete = len(selected)
        rows.append(
            {
                "cell": asdict(cell) | {"identifier": cell.identifier},
                "complete_repetitions": complete,
                "declared_repetitions": R1_REPETITIONS,
                "winner_counts": winner_counts,
                "winner_rates": {
                    locus: (count / complete if complete else None)
                    for locus, count in winner_counts.items()
                },
                "generating_locus_winner_rate": (
                    winner_counts[cell.locus] / complete if complete else None
                ),
                "mean_subject_exact_model_winner_rate": (
                    float(
                        np.mean(
                            [item["subject_exact_model_winner_rate"] for item in selected]
                        )
                    )
                    if complete
                    else None
                ),
                "w_v0_single_locus_collapse_rate": (
                    (
                        winner_counts["w"] + winner_counts["v0"]
                    )
                    / complete
                    if complete and cell.locus == "w_v0"
                    else None
                ),
            }
        )
    return rows


def _confusion_payload(cell_rows):
    return [
        {
            "architecture": item["cell"]["architecture"],
            "effect_level": item["cell"]["effect_level"],
            "generating_locus": item["cell"]["locus"],
            "complete_repetitions": item["complete_repetitions"],
            "winner_counts": item["winner_counts"],
            "winner_rates": item["winner_rates"],
        }
        for item in cell_rows
    ]


def _summarize_parameter_recovery(task_root, criteria):
    rows = []
    for cell in cue_r1_cells():
        truth_rows = []
        estimate_rows = []
        parameter_names = None
        condition_numbers = []
        named_posterior_correlations = []
        for repetition in range(R1_REPETITIONS):
            for subject_index in range(37):
                path = task_root / _task_filename(
                    cell.identifier, repetition, subject_index
                )
                if not path.exists():
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                exact = next(
                    (
                        fit
                        for fit in record["fits"]
                        if fit["model_id"] == cell.generating_model
                    ),
                    None,
                )
                if exact is None or exact["estimate"] is None:
                    continue
                names = tuple(record["generating_parameter_names"])
                if tuple(exact["free_parameter_names"]) != names:
                    raise SystemExit("Exact-model parameter names do not align: %s" % path)
                if parameter_names is None:
                    parameter_names = names
                elif parameter_names != names:
                    raise SystemExit("Generating parameter names changed within a cell.")
                truth_rows.append(record["estimation_scale_truth"])
                estimate_rows.append(exact["estimate"])
                value = exact.get("hessian_condition_number")
                if value is not None and np.isfinite(value):
                    condition_numbers.append(float(value))
                if exact.get("correlation") is not None:
                    named_posterior_correlations.extend(
                        _named_correlations(
                            names, np.asarray(exact["correlation"], dtype=float)
                        )
                    )
        parameters = []
        if truth_rows:
            truth = np.asarray(truth_rows, dtype=float)
            estimate = np.asarray(estimate_rows, dtype=float)
            prior_sd = _prior_sd(cell.generating_model)
            for column, name in enumerate(parameter_names):
                true_values = truth[:, column]
                estimated_values = estimate[:, column]
                error = estimated_values - true_values
                correlation = None
                if (
                    true_values.size >= 3
                    and np.std(true_values, ddof=1) > 0.0
                    and np.std(estimated_values, ddof=1) > 0.0
                ):
                    correlation = float(
                        np.corrcoef(true_values, estimated_values)[0, 1]
                    )
                bias = float(np.mean(error))
                rmse = float(np.sqrt(np.mean(error**2)))
                ratio = rmse / prior_sd[name]
                parameters.append(
                    {
                        "parameter": name,
                        "cases": int(true_values.size),
                        "correlation": correlation,
                        "bias": bias,
                        "absolute_bias": abs(bias),
                        "rmse": rmse,
                        "prior_sd": prior_sd[name],
                        "rmse_over_prior_sd": ratio,
                        "passes": bool(
                            true_values.size >= criteria["parameter_cases_min"]
                            and correlation is not None
                            and correlation >= criteria["parameter_correlation_min"]
                            and abs(bias)
                            <= criteria["parameter_absolute_bias_max"]
                            and ratio <= criteria["parameter_rmse_over_prior_sd_max"]
                        ),
                    }
                )
        rows.append(
            {
                "cell": asdict(cell) | {"identifier": cell.identifier},
                "cases": len(truth_rows),
                "parameters": parameters,
                "all_parameters_pass": bool(parameters)
                and all(item["passes"] for item in parameters),
                "hessian_condition_number_median": (
                    float(np.median(condition_numbers)) if condition_numbers else None
                ),
                "hessian_condition_number_max": (
                    float(np.max(condition_numbers)) if condition_numbers else None
                ),
                "posterior_correlations": _summarize_named_correlations(
                    named_posterior_correlations
                ),
            }
        )
    return rows


def _prior_sd(model_id):
    hgf = cue_hgf_prior()
    response = ddm_prior(model_id)
    names = tuple("hgf." + name for name in hgf.free_names) + tuple(
        "ddm." + name for name in response.free_names
    )
    variances = np.concatenate(
        (hgf.variances[hgf.free_mask], response.variances[response.free_mask])
    )
    return {name: float(np.sqrt(value)) for name, value in zip(names, variances)}


def _named_correlations(names, matrix):
    wanted = (
        ("ddm.gamma_w", "ddm.gamma_v0"),
        ("ddm.b_w", "ddm.b_v"),
    )
    wanted += tuple(
        ("hgf.omega_2", name)
        for name in names
        if name.startswith("ddm.")
        and name not in {"ddm.log_a_a", "ddm.Ter_logit"}
    )
    rows = []
    for left, right in wanted:
        if left in names and right in names:
            rows.append(
                {
                    "pair": [left, right],
                    "value": float(matrix[names.index(left), names.index(right)]),
                }
            )
    return rows


def _summarize_named_correlations(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row["pair"]), []).append(row["value"])
    return [
        {
            "pair": list(pair),
            "cases": len(values),
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
        for pair, values in sorted(grouped.items())
    ]


def _gate_verdict(manifest, cell_rows, parameter_rows):
    cell_by_id = {item["cell"]["identifier"]: item for item in cell_rows}
    parameter_by_id = {
        item["cell"]["identifier"]: item for item in parameter_rows
    }
    required = [cell for cell in cue_r1_cells() if cell.primary_gate]
    if any(
        cell_by_id[cell.identifier]["complete_repetitions"] < R1_REPETITIONS
        for cell in required
    ):
        return {
            "status": "not_evaluable",
            "gate_passed": None,
            "reason": "Not all 20 repetitions of every primary cell are complete.",
        }
    criteria = manifest["success_criteria"]
    checks = []
    for cell in required:
        row = cell_by_id[cell.identifier]
        if cell.locus == "null":
            cue_false_positive = 1.0 - row["winner_rates"]["null"]
            checks.append(
                {
                    "check": cell.identifier + ".null_false_positive",
                    "value": cue_false_positive,
                    "limit": criteria["null_false_positive_rate_max"],
                    "passes": cue_false_positive
                    <= criteria["null_false_positive_rate_max"],
                }
            )
        else:
            value = row["generating_locus_winner_rate"]
            checks.append(
                {
                    "check": cell.identifier + ".generating_locus",
                    "value": value,
                    "limit": criteria[
                        "generating_locus_expected_frequency_winner_rate_min"
                    ],
                    "passes": value
                    >= criteria[
                        "generating_locus_expected_frequency_winner_rate_min"
                    ],
                }
            )
        recovery = parameter_by_id[cell.identifier]
        checks.append(
            {
                "check": cell.identifier + ".parameter_recovery",
                "value": recovery["all_parameters_pass"],
                "limit": True,
                "passes": recovery["all_parameters_pass"],
            }
        )
    for architecture in ("parallel", "integrated"):
        for source, target, key in (
            ("w", "v0", "w_to_v0_misclassification_rate_max"),
            ("v0", "w", "v0_to_w_misclassification_rate_max"),
        ):
            source_rows = [
                cell_by_id[cell.identifier]
                for cell in required
                if cell.architecture == architecture and cell.locus == source
            ]
            count = sum(row["winner_counts"][target] for row in source_rows)
            total = sum(row["complete_repetitions"] for row in source_rows)
            value = count / total
            checks.append(
                {
                    "check": "%s.%s_to_%s" % (architecture, source, target),
                    "value": value,
                    "limit": criteria[key],
                    "passes": value <= criteria[key],
                }
            )
    passed = all(item["passes"] for item in checks)
    return {
        "status": "passed" if passed else "failed",
        "gate_passed": passed,
        "checks": checks,
    }


def _read_manifest(run_directory):
    path = run_directory / "manifest.json"
    if not path.exists():
        raise SystemExit("Freeze the Gate manifest before running recovery.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_digest") != cue_r1_manifest_digest(manifest):
        raise SystemExit("Gate R1 manifest digest is invalid.")
    return manifest


def _validated_cache_addendum(run_directory, manifest, cache_size):
    if cache_size == 0:
        return None
    if cache_size != R1_HGF_CACHE_SIZE:
        raise SystemExit(
            "Formal cached R1 requires hgf_cache_size=%d." % R1_HGF_CACHE_SIZE
        )
    path = run_directory / "execution_addendum_hgf_cache_0.1.0.json"
    if not path.exists():
        raise SystemExit("The frozen HGF-cache execution addendum is missing.")
    addendum = json.loads(path.read_text(encoding="utf-8"))
    if addendum.get("addendum_version") != R1_HGF_CACHE_ADDENDUM_VERSION:
        raise SystemExit("HGF-cache addendum version mismatch.")
    if addendum.get("parent_manifest_digest") != manifest["manifest_digest"]:
        raise SystemExit("HGF-cache addendum belongs to a different manifest.")
    if addendum.get("hgf_cache_size") != cache_size:
        raise SystemExit("HGF-cache addendum capacity mismatch.")
    if addendum.get("addendum_digest") != cue_r1_execution_addendum_digest(
        addendum
    ):
        raise SystemExit("HGF-cache execution addendum digest is invalid.")
    return addendum["addendum_digest"]


def _task_filename(cell_id, repetition, subject_index):
    return "%s__r%02d__s%02d.json" % (cell_id, repetition + 1, subject_index + 1)


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.%d" % os.getpid())
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _print_task(payload):
    ok = sum(fit["status"] in {"ok", "map_only"} for fit in payload["fits"])
    print(
        "%s r%02d s%02d fits=%d/4 %.1fs"
        % (
            payload["cell"]["identifier"],
            payload["repetition"] + 1,
            payload["subject_index"] + 1,
            ok,
            payload["seconds"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
