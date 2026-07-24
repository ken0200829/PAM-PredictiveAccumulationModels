"""Freeze, run, resume, and summarize the small cue-locus R1 diagnostic.

Commands::

    PYTHONPATH=src python3 -B scripts/run_cue_r1_mini.py freeze DATA_DIR RUN_DIR
    PYTHONPATH=src python3 -B scripts/run_cue_r1_mini.py run DATA_DIR RUN_DIR --workers 4
    PYTHONPATH=src python3 -B scripts/run_cue_r1_mini.py summarize RUN_DIR
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pam_dot_task_python.cue_r1 import (
    ARCHITECTURE_MODELS,
    R1_HGF_CACHE_SIZE,
    cue_r1_manifest_digest,
    r1_bms,
)
from pam_dot_task_python.cue_r1_mini import (
    MINI_BMS_SAMPLES,
    MINI_REPETITIONS,
    cue_r1_mini_cells,
    cue_r1_mini_manifest,
    cue_r1_mini_manifest_digest,
    select_mini_subject_indices,
)

from run_cue_r1 import (
    _atomic_json,
    _run_task,
    _subjects,
    _task_filename,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("data_directory", type=Path)
    freeze.add_argument("run_directory", type=Path)
    freeze.add_argument("--parent-manifest", type=Path)
    run = commands.add_parser("run")
    run.add_argument("data_directory", type=Path)
    run.add_argument("run_directory", type=Path)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--max-tasks", type=int)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("run_directory", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "freeze":
        return freeze_manifest(
            arguments.data_directory,
            arguments.run_directory,
            arguments.parent_manifest,
        )
    if arguments.command == "run":
        return run_screen(arguments)
    return summarize_screen(arguments.run_directory)


def freeze_manifest(data_directory, run_directory, parent_manifest_path=None):
    _, subjects, designs = _subjects(data_directory)
    parent_path = parent_manifest_path or (
        Path(__file__).resolve().parents[1]
        / "recovery_runs"
        / "cue_r1_0_1_0"
        / "manifest.json"
    )
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("manifest_digest") != cue_r1_manifest_digest(parent):
        raise SystemExit("The parent Gate R1 manifest digest is invalid.")
    manifest = cue_r1_mini_manifest(parent["manifest_digest"], subjects, designs)
    destination = run_directory / "manifest.json"
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous != manifest:
            raise SystemExit("A different mini-screen manifest already exists.")
    else:
        _atomic_json(destination, manifest)
    print(manifest["manifest_digest"])
    return 0


def run_screen(arguments):
    paths, subjects, designs = _subjects(arguments.data_directory)
    manifest = _read_manifest(arguments.run_directory)
    expected = cue_r1_mini_manifest(
        manifest["parent_gate_manifest_digest"], subjects, designs
    )
    if expected != manifest:
        raise SystemExit("Current inputs or mini-screen selection differ from the manifest.")
    selected = select_mini_subject_indices(subjects)
    output_root = arguments.run_directory / "tasks"
    output_root.mkdir(parents=True, exist_ok=True)
    pending = []
    for cell in cue_r1_mini_cells():
        for repetition in range(MINI_REPETITIONS):
            for subject_index in selected:
                destination = output_root / _task_filename(
                    cell.identifier, repetition, subject_index
                )
                if not destination.exists():
                    pending.append(
                        (
                            cell,
                            repetition,
                            subject_index,
                            paths[subject_index],
                            destination,
                        )
                    )
    if arguments.max_tasks is not None:
        pending = pending[: arguments.max_tasks]
    print(
        "pending=%d/96 output=%s manifest=%s"
        % (len(pending), output_root.name, manifest["manifest_digest"]),
        flush=True,
    )
    if arguments.workers <= 1:
        for task in pending:
            _write_task(task, manifest)
    else:
        with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
            futures = {
                executor.submit(
                    _run_task,
                    task[:-1],
                    100,
                    10,
                    True,
                    manifest["manifest_digest"],
                    R1_HGF_CACHE_SIZE,
                    manifest["manifest_digest"],
                ): task[-1]
                for task in pending
            }
            for future in as_completed(futures):
                payload = future.result()
                _atomic_json(futures[future], payload)
                _print_mini_task(payload)
    return 0


def _write_task(task, manifest):
    payload = _run_task(
        task[:-1],
        100,
        10,
        True,
        manifest["manifest_digest"],
        R1_HGF_CACHE_SIZE,
        manifest["manifest_digest"],
    )
    _atomic_json(task[-1], payload)
    _print_mini_task(payload)


def summarize_screen(run_directory):
    manifest = _read_manifest(run_directory)
    task_root = run_directory / "tasks"
    selected = [
        int(item["full_cohort_index"]) for item in manifest["selected_subjects"]
    ]
    repetitions = []
    failed_fits = 0
    total_fits = 0
    for cell in cue_r1_mini_cells():
        models = ARCHITECTURE_MODELS[cell.architecture]
        for repetition in range(MINI_REPETITIONS):
            records = []
            for subject_index in selected:
                path = task_root / _task_filename(
                    cell.identifier, repetition, subject_index
                )
                if path.exists():
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if record.get("manifest_digest") != manifest["manifest_digest"]:
                        raise SystemExit("Task manifest digest mismatch: %s" % path)
                    records.append(record)
            if len(records) != len(selected):
                repetitions.append(
                    {
                        "cell": cell.identifier,
                        "repetition": repetition,
                        "status": "incomplete",
                        "subjects": len(records),
                    }
                )
                continue
            lme = np.full((len(selected), len(models)), np.nan)
            ordered = sorted(records, key=lambda item: item["subject_index"])
            for row, record in enumerate(ordered):
                by_model = {fit["model_id"]: fit for fit in record["fits"]}
                for column, model_id in enumerate(models):
                    fit = by_model[model_id]
                    total_fits += 1
                    failed_fits += fit["status"] != "ok"
                    if fit["lme"] is not None:
                        lme[row, column] = fit["lme"]
            if np.any(~np.isfinite(lme)):
                repetitions.append(
                    {
                        "cell": cell.identifier,
                        "repetition": repetition,
                        "status": "nonfinite_lme",
                    }
                )
                continue
            result = r1_bms(
                lme,
                models,
                [record["subject_id"] for record in ordered],
                seed=manifest["seed"] + 100 * cue_r1_mini_cells().index(cell) + repetition,
                samples=MINI_BMS_SAMPLES,
            )
            repetitions.append(
                {
                    "cell": cell.identifier,
                    "architecture": cell.architecture,
                    "generating_locus": cell.locus,
                    "repetition": repetition,
                    "status": "ok",
                    "bms": result,
                    "winner_locus": _model_locus(result["winner"]),
                    "subject_exact_model_winner_rate": float(
                        np.mean(
                            np.asarray(models)[np.argmax(lme, axis=1)]
                            == cell.generating_model
                        )
                    ),
                }
            )
    reasons = []
    warnings = []
    if any(item["status"] != "ok" for item in repetitions):
        reasons.append("Not all 12 group-cell repetitions have finite LME.")
    if failed_fits:
        reasons.append("%d of %d candidate fits failed." % (failed_fits, total_fits))
    for cell in cue_r1_mini_cells():
        rows = [
            item
            for item in repetitions
            if item["status"] == "ok" and item["cell"] == cell.identifier
        ]
        if len(rows) == MINI_REPETITIONS and not any(
            item["winner_locus"] == cell.locus for item in rows
        ):
            reasons.append(
                "%s never recovered its generating locus in two repetitions."
                % cell.identifier
            )
        if (
            cell.locus == "null"
            and len(rows) == MINI_REPETITIONS
            and any(item["winner_locus"] != "null" for item in rows)
        ):
            warnings.append(
                "%s produced a cue-effect false positive in at least one repetition."
                % cell.identifier
            )
    payload = {
        "manifest_digest": manifest["manifest_digest"],
        "screening_only": True,
        "formal_gate_status": "not_evaluable",
        "status": (
            "incomplete"
            if any(item["status"] != "ok" for item in repetitions)
            else ("promising" if not reasons else "concerning")
        ),
        "interpretation": (
            "Promising means only that no gross no-go pattern appeared in this "
            "small diagnostic. It is not evidence that formal Gate R1 passed."
        ),
        "failed_fits": failed_fits,
        "total_fits": total_fits,
        "reasons": reasons,
        "warnings": warnings,
        "repetitions": repetitions,
    }
    _atomic_json(run_directory / "summary.json", payload)
    print(
        "status=%s repetitions=%d/12 failed_fits=%d/%d"
        % (
            payload["status"],
            sum(item["status"] == "ok" for item in repetitions),
            failed_fits,
            total_fits,
        )
    )
    return 0


def _read_manifest(run_directory):
    path = run_directory / "manifest.json"
    if not path.exists():
        raise SystemExit("Freeze the mini-screen manifest before running it.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_digest") != cue_r1_mini_manifest_digest(manifest):
        raise SystemExit("The mini-screen manifest digest is invalid.")
    return manifest


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


def _print_mini_task(payload):
    ok = sum(fit["status"] == "ok" for fit in payload["fits"])
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
