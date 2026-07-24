"""Regression checks against MATLAB Online HGF and WFPT fixtures."""

from pathlib import Path
import unittest

import numpy as np

from pam_dot_task_python.fixtures import fixture_design, load_fixture
from pam_dot_task_python.hgf import (
    binary_hgf,
    cue_binary_hgf,
    transform_ehgf_binary,
)
from pam_dot_task_python.wfpt import wfpt_density


FIXTURE_DIRECTORY = Path(__file__).parents[1] / "fixtures" / "matlab"
TRAJECTORY_FIELDS = ("muhat", "sahat", "mu", "sa")


class MatlabHGFFixtureTests(unittest.TestCase):
    """Check TAPAS eHGF trajectories at three fixed volatility settings."""

    @classmethod
    def setUpClass(cls):
        cls.design = fixture_design()
        cls.fixture = load_fixture(str(FIXTURE_DIRECTORY / "hgf.json"))

    def test_two_cue_trajectories_match_matlab(self):
        for index, case in enumerate(self.fixture["cue_cases"]):
            result = cue_binary_hgf(
                self.design.u[:, :2],
                transform_ehgf_binary(np.asarray(case["ptrans"], dtype=float)),
            )
            for field in TRAJECTORY_FIELDS:
                np.testing.assert_allclose(
                    getattr(result.active, field),
                    np.asarray(case[field], dtype=float),
                    rtol=0.0,
                    atol=1e-12,
                    equal_nan=True,
                    err_msg="Two-cue HGF %s differs in case %d" % (field, index),
                )
            np.testing.assert_array_equal(
                result.white_indices + 1,
                np.asarray(case["white_index"], dtype=int),
            )
            np.testing.assert_array_equal(
                result.red_indices + 1,
                np.asarray(case["red_index"], dtype=int),
            )

    def test_single_stream_trajectories_match_matlab(self):
        omega_index = int(self.fixture["omega_index"]) - 1
        base = np.asarray(self.fixture["prior_mus"], dtype=float)
        for index, case in enumerate(self.fixture["single_stream_cases"]):
            transformed = base.copy()
            transformed[omega_index] = float(case["omega_2"])
            result = binary_hgf(
                self.design.u[:, 0], transform_ehgf_binary(transformed)
            )
            for field in TRAJECTORY_FIELDS:
                np.testing.assert_allclose(
                    getattr(result, field),
                    np.asarray(case[field], dtype=float),
                    rtol=0.0,
                    atol=1e-12,
                    equal_nan=True,
                    err_msg="Single-stream HGF %s differs in case %d"
                    % (field, index),
                )


class MatlabWFPTFixtureTests(unittest.TestCase):
    """Check PAM's Gondan-series density grid against MATLAB output."""

    def test_wfpt_grid_matches_matlab(self):
        fixture = load_fixture(str(FIXTURE_DIRECTORY / "wfpt.json"))
        grid = np.asarray(fixture["grid"], dtype=float)
        density = np.asarray(
            [
                wfpt_density(
                    time=row[0],
                    drift=row[1],
                    boundary=row[2],
                    start=row[3],
                    precision=row[4],
                )
                for row in grid
            ],
            dtype=float,
        )
        np.testing.assert_allclose(
            density,
            grid[:, 5],
            rtol=0.0,
            atol=1e-12,
            err_msg="WFPT density grid differs from MATLAB.",
        )


if __name__ == "__main__":
    unittest.main()
