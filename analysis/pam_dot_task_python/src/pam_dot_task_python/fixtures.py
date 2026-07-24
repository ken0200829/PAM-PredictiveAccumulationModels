"""Reconstruct the MATLAB fixture design and read exported reference files.

The MATLAB exporter (``pam_dot_task_export_fixtures.m``) writes JSON built
from a design that contains no participant data: every value comes from the
Lehmer generator ``state <- 16807 * state mod (2**31 - 1)`` seeded at
20260721.  :func:`fixture_design` reproduces that design here in integer
arithmetic, so both languages construct bit-identical inputs and the fixtures
can be produced on a machine that never holds the private CSVs.

If this reconstruction and the exported ``design.json`` ever disagree, the
comparison is invalid regardless of what the other fixtures say; use
:func:`assert_design_matches` before trusting any parity result.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import json

import numpy as np
import pandas as pd
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

LEHMER_MULTIPLIER = 16807
LEHMER_MODULUS = 2147483647
FIXTURE_SEED = 20260721
INVALID_TEST_TRIALS = (131, 187, 244, 301, 365)


class _Lehmer:
    """Integer Lehmer stream matching ``lehmer_next`` in the MATLAB sources."""

    def __init__(self, seed: int = FIXTURE_SEED) -> None:
        self.state = int(seed)

    def next(self) -> float:
        self.state = (LEHMER_MULTIPLIER * self.state) % LEHMER_MODULUS
        return self.state / LEHMER_MODULUS

    def shuffle(self, values: List[float]) -> List[float]:
        """Fisher-Yates with the exact index convention used in MATLAB."""

        items = list(values)
        for i in range(len(items), 1, -1):
            u = self.next()
            j = int(np.floor(u * i)) + 1
            items[i - 1], items[j - 1] = items[j - 1], items[i - 1]
        return items


@dataclass(frozen=True)
class FixtureDesign:
    u: FloatArray
    y: FloatArray
    trial: NDArray[np.int64]
    phase: Tuple[str, ...]
    ratio_corrected: FloatArray
    muhat_reference: FloatArray
    seed: int = FIXTURE_SEED

    @property
    def audit(self) -> pd.DataFrame:
        """Audit table in the column layout the PPC spec builders expect."""

        return pd.DataFrame(
            {
                "trial": self.trial,
                "phase": list(self.phase),
                "stimulus_category": self.u[:, 0],
                "cue_white": self.u[:, 1],
                "signed_coherence": self.u[:, 2],
                "ratio_corrected": self.ratio_corrected,
                "rt_seconds_raw": self.y[:, 0],
                "choice_white": self.y[:, 1],
            }
        )


def fixture_design() -> FixtureDesign:
    """Rebuild the deterministic 380-trial fixture design."""

    stream = _Lehmer(FIXTURE_SEED)
    n_learning, n_test = 100, 280
    n_total = n_learning + n_test

    cue_learning = stream.shuffle([1.0] * 70 + [0.0] * 30)
    cue_test = stream.shuffle([1.0] * 140 + [0.0] * 140)

    per_cue_ratios = (
        [0.5] * 20
        + [0.45] * 20
        + [0.55] * 20
        + [0.40] * 20
        + [0.60] * 20
        + [0.35] * 20
        + [0.65] * 20
    )
    ratio_test = [0.0] * n_test
    white_positions = [i for i, value in enumerate(cue_test) if value == 1.0]
    red_positions = [i for i, value in enumerate(cue_test) if value == 0.0]
    white_ratios = stream.shuffle(per_cue_ratios)
    red_ratios = stream.shuffle(per_cue_ratios)
    for position, value in zip(white_positions, white_ratios):
        ratio_test[position] = value
    for position, value in zip(red_positions, red_ratios):
        ratio_test[position] = value

    ratio_learning = []
    for cue_value in cue_learning:
        value = stream.next()
        if cue_value == 1.0:
            ratio_learning.append(0.2 if value < 0.5 else 0.8)
        else:
            ratio_learning.append(0.8 if value < 0.85 else 0.2)

    ratio = np.asarray(ratio_learning + ratio_test, dtype=float)
    cue = np.asarray(list(cue_learning) + list(cue_test), dtype=float)
    stimulus = (ratio > 0.5).astype(float)
    signed_coherence = 2.0 * ratio - 1.0

    y = np.full((n_total, 2), np.nan)
    for index in range(n_learning, n_total):
        u_rt = stream.next()
        u_choice = stream.next()
        rt = 0.35 + 1.10 * u_rt
        probability = 0.5 + 0.35 * (2.0 * stimulus[index] - 1.0) * abs(
            signed_coherence[index]
        ) / 0.3
        probability = min(max(probability, 0.02), 0.98)
        y[index, 0] = rt
        y[index, 1] = float(u_choice < probability)

    for trial_number in INVALID_TEST_TRIALS:
        y[trial_number - 1, :] = np.nan

    muhat_reference = np.asarray(
        [0.15 + 0.70 * stream.next() for _ in range(n_total)], dtype=float
    )

    return FixtureDesign(
        u=np.column_stack((stimulus, cue, signed_coherence)),
        y=y,
        trial=np.arange(1, n_total + 1, dtype=np.int64),
        phase=tuple(["learning"] * n_learning + ["test"] * n_test),
        ratio_corrected=ratio,
        muhat_reference=muhat_reference,
    )


def load_fixture(path: str) -> Dict[str, Any]:
    """Read one exported fixture, restoring NaN and infinities."""

    with open(path, "r") as handle:
        return _restore(json.load(handle))


def assert_design_matches(
    exported: Dict[str, Any], design: FixtureDesign = None, tolerance: float = 0.0
) -> None:
    """Fail unless the exported design equals the Python reconstruction.

    The default tolerance is exact equality: both sides run the same integer
    recurrence and the same arithmetic, so any difference indicates a genuine
    divergence rather than accumulated round-off.
    """

    local = fixture_design() if design is None else design
    exported_u = np.asarray(exported["u"], dtype=float)
    exported_y = np.asarray(exported["y"], dtype=float)
    if exported_u.shape != local.u.shape:
        raise AssertionError(
            "Exported u has shape %s; reconstruction has %s."
            % (exported_u.shape, local.u.shape)
        )
    if not np.allclose(exported_u, local.u, rtol=tolerance, atol=tolerance):
        raise AssertionError("Exported and reconstructed inputs differ.")
    finite = np.isfinite(exported_y) & np.isfinite(local.y)
    if not np.array_equal(np.isfinite(exported_y), np.isfinite(local.y)):
        raise AssertionError("Exported and reconstructed response masks differ.")
    if not np.allclose(
        exported_y[finite], local.y[finite], rtol=tolerance, atol=tolerance
    ):
        raise AssertionError("Exported and reconstructed responses differ.")


def _restore(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore(item) for item in value]
    if isinstance(value, str):
        if value == "NaN":
            return float("nan")
        if value == "Infinity":
            return float("inf")
        if value == "-Infinity":
            return float("-inf")
    return value
