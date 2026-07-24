"""Read-only dot-task adapter matching ``pam_dot_task_load_subject.m``."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


PathLike = Union[str, Path]


@dataclass(frozen=True)
class Condition:
    name: str
    stimulus_reversed: bool
    white_key: str
    black_key: str

    @property
    def red_prediction_sign(self) -> int:
        """Direction predicted by red, aligned to the white/black boundaries."""

        return -1 if self.stimulus_reversed else 1


@dataclass(frozen=True)
class SubjectData:
    subject_id: str
    csv_path: Path
    condition: Condition
    audit: pd.DataFrame
    u: NDArray[np.float64]
    y: NDArray[np.float64]
    is_tie: NDArray[np.bool_]
    cue_red: NDArray[np.float64]
    cue_evidence: NDArray[np.float64]
    irregular_trials: NDArray[np.int64]
    likelihood_trials: NDArray[np.int64]


def resolve_condition(filename: PathLike) -> Condition:
    """Resolve counterbalancing exclusively from the filename prefix."""

    stem = Path(filename).stem
    definitions = (
        ("normal_cb", False, "f"),
        ("reverse_cb", True, "j"),
        ("normal", False, "j"),
        ("reverse", True, "f"),
    )
    for name, reversed_stimulus, white_key in definitions:
        if stem.startswith(name + "_dot_task_"):
            return Condition(
                name=name,
                stimulus_reversed=reversed_stimulus,
                white_key=white_key,
                black_key="j" if white_key == "f" else "f",
            )
    raise ValueError("Unknown condition prefix in filename: %s" % stem)


def load_subject(csv_path: PathLike) -> SubjectData:
    """Build 380-row PAM inputs without modifying or deleting raw trials."""

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError("CSV does not exist: %s" % path)
    condition = resolve_condition(path)
    raw = pd.read_csv(path)
    required = {"main_trial_number", "rt", "response", "ratio", "cross_color"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError("CSV is missing required columns: %s" % ", ".join(missing))

    main_trial_all = pd.to_numeric(raw["main_trial_number"], errors="coerce")
    main_mask = main_trial_all.notna().to_numpy()
    main = raw.loc[main_mask].copy()
    main["__raw_row"] = np.flatnonzero(main_mask) + 1
    main["__trial"] = main_trial_all.loc[main_mask].to_numpy(dtype=float)
    if np.any(main["__trial"].to_numpy() != np.trunc(main["__trial"].to_numpy())):
        raise ValueError("main_trial_number contains non-integer values: %s" % path)
    main = main.sort_values("__trial", kind="stable").reset_index(drop=True)
    trial = main["__trial"].to_numpy(dtype=int)
    if not np.array_equal(trial, np.arange(1, 381)):
        raise ValueError("Expected each main trial 1:380 exactly once: %s" % path)

    phase = np.where(trial <= 100, "learning", "test")
    rt_seconds_raw = pd.to_numeric(main["rt"], errors="coerce").to_numpy() / 1000.0
    rt_valid = np.isfinite(rt_seconds_raw) & (rt_seconds_raw >= 0.15) & (
        rt_seconds_raw <= 3.0
    )

    response = main["response"].astype("string").str.strip().str.lower()
    response_missing = (
        response.isna().to_numpy(dtype=bool)
        | response.fillna("").eq("").to_numpy(dtype=bool)
    )
    response_key = response.fillna("").to_numpy(dtype=str)
    response_valid = (response_key == condition.white_key) | (
        response_key == condition.black_key
    )
    choice_white = np.full(380, np.nan)
    choice_white[response_key == condition.white_key] = 1.0
    choice_white[response_key == condition.black_key] = 0.0

    ratio_raw = pd.to_numeric(main["ratio"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(ratio_raw) | (ratio_raw < 0) | (ratio_raw > 1)):
        raise ValueError("ratio must be finite and within [0,1]: %s" % path)
    ratio_corrected = 1.0 - ratio_raw if condition.stimulus_reversed else ratio_raw
    signed_coherence = 2.0 * ratio_corrected - 1.0
    stimulus_category = (ratio_corrected > 0.5).astype(float)
    # Tie trials (ratio 0.5) show 100 white and 100 black dots, so no category
    # is objectively correct.  The stimulus_category value stored for them is a
    # placeholder with no semantics; consumers must branch on is_tie instead
    # (plan section 5.2.1).  Coding ties as "black" -- the retracted
    # convention -- pulled P(white) from 0.500 to 0.429 purely as an artefact.
    # Ratio 0.5 is exactly representable and 2*0.5 - 1 == 0 exactly, in the
    # normal and the reversed (1 - ratio) branch alike, so this test is safe.
    is_tie = signed_coherence == 0.0

    cue = main["cross_color"].astype("string").str.strip().str.lower()
    cue_key = cue.fillna("").to_numpy(dtype=str)
    cue_valid = (cue_key == "white") | (cue_key == "red")
    if not np.all(cue_valid):
        raise ValueError("cross_color must be white or red: %s" % path)
    cue_white = (cue_key == "white").astype(float)
    cue_red = (cue_key == "red").astype(float)
    red_prediction_sign = np.full(380, condition.red_prediction_sign, dtype=int)
    cue_evidence = cue_red * red_prediction_sign

    likelihood_included = (phase == "test") & rt_valid & response_valid
    rt_for_pam = rt_seconds_raw.copy()
    choice_for_pam = choice_white.copy()
    rt_for_pam[~likelihood_included] = np.nan
    choice_for_pam[~likelihood_included] = np.nan

    reasons = [[] for _ in range(380)]
    _append_reason(reasons, phase == "learning", "learning_likelihood_mask")
    _append_reason(reasons, ~np.isfinite(rt_seconds_raw), "rt_missing")
    _append_reason(
        reasons, np.isfinite(rt_seconds_raw) & (rt_seconds_raw < 0.15), "rt_low"
    )
    _append_reason(
        reasons, np.isfinite(rt_seconds_raw) & (rt_seconds_raw > 3.0), "rt_high"
    )
    _append_reason(reasons, response_missing, "choice_missing")
    _append_reason(reasons, ~response_missing & ~response_valid, "invalid_key")
    exclude_reason = np.array([";".join(parts) for parts in reasons], dtype=object)
    exclude_reason[likelihood_included] = "included"

    subject_id = path.stem
    audit = pd.DataFrame(
        {
            "subject_id": subject_id,
            "condition": condition.name,
            "stimulus_reversed": condition.stimulus_reversed,
            "white_key": condition.white_key,
            "raw_row": main["__raw_row"].to_numpy(dtype=int),
            "trial": trial,
            "phase": phase,
            "rt_seconds_raw": rt_seconds_raw,
            "rt_for_pam": rt_for_pam,
            "response_key": response_key,
            "choice_white": choice_white,
            "choice_for_pam": choice_for_pam,
            "ratio_raw": ratio_raw,
            "ratio_corrected": ratio_corrected,
            "signed_coherence": signed_coherence,
            "stimulus_category": stimulus_category,
            "is_tie": is_tie,
            "cue": cue_key,
            "cue_white": cue_white,
            "cue_red": cue_red,
            "red_prediction_sign": red_prediction_sign,
            "cue_evidence": cue_evidence,
            "likelihood_included": likelihood_included,
            "exclude_reason": exclude_reason,
        }
    )
    u = np.column_stack((stimulus_category, cue_white, signed_coherence))
    y = np.column_stack((rt_for_pam, choice_for_pam))
    if u.shape != (380, 3) or y.shape != (380, 2) or not np.all(np.isfinite(u)):
        raise RuntimeError("The adapter changed the required input shape or trial history.")
    if not np.all(np.isnan(y[~likelihood_included])):
        raise RuntimeError("Every excluded response row must be all NaN.")
    if not np.all(np.isfinite(y[likelihood_included])):
        raise RuntimeError("Every included response row must be finite.")

    return SubjectData(
        subject_id=subject_id,
        csv_path=path,
        condition=condition,
        audit=audit,
        u=u,
        y=y,
        is_tie=is_tie,
        cue_red=cue_red,
        cue_evidence=cue_evidence,
        irregular_trials=np.flatnonzero(~likelihood_included),
        likelihood_trials=np.flatnonzero(likelihood_included),
    )


def _append_reason(reasons: list, mask: NDArray[np.bool_], reason: str) -> None:
    for index in np.flatnonzero(mask):
        reasons[int(index)].append(reason)
