"""Regression checks against MATLAB Online DDM reference fixtures."""

from pathlib import Path
import unittest

import numpy as np

from pam_dot_task_python.fixtures import (
    assert_design_matches,
    fixture_design,
    load_fixture,
)
from pam_dot_task_python.response import ddm_loglik, transform_ddm


FIXTURE_DIRECTORY = Path(__file__).parents[1] / "fixtures" / "matlab"


class MatlabDDMFixtureTests(unittest.TestCase):
    """Compare trial-wise MATLAB/Python DDM likelihoods on fixed inputs."""

    @classmethod
    def setUpClass(cls):
        cls.design = fixture_design()
        cls.exported_design = load_fixture(str(FIXTURE_DIRECTORY / "design.json"))
        cls.fixture = load_fixture(str(FIXTURE_DIRECTORY / "ddm.json"))

    def test_exported_design_matches_exactly_before_likelihood_comparison(self):
        assert_design_matches(self.exported_design, self.design)

    def test_official_ddm_trial_log_likelihoods_match_matlab(self):
        self._assert_cases_match("official_cases")

    def test_coherence_ddm_trial_log_likelihoods_match_matlab(self):
        self._assert_cases_match("coherence_cases")

    def _assert_cases_match(self, case_name: str):
        muhat = np.asarray(self.fixture["muhat_active"], dtype=float)
        for index, case in enumerate(self.fixture[case_name]):
            parameters = transform_ddm(
                np.asarray(case["ptrans_obs"], dtype=float), self.design.y[:, 0]
            )
            observed, _ = ddm_loglik(
                self.design.y,
                self.design.u[:, 0],
                muhat,
                parameters,
                self.design.u[:, 2],
            )
            expected = np.asarray(case["logp"], dtype=float)
            finite = np.isfinite(expected)
            self.assertTrue(
                np.array_equal(np.isfinite(observed), finite),
                "MATLAB/Python response mask differs in %s case %d"
                % (case_name, index),
            )
            np.testing.assert_allclose(
                observed[finite], expected[finite], rtol=0.0, atol=1e-12,
                err_msg="Trial likelihood differs in %s case %d" % (case_name, index),
            )
            self.assertAlmostEqual(
                float(np.sum(observed[finite])),
                float(case["sum_logp"]),
                delta=1e-12,
                msg="Total likelihood differs in %s case %d" % (case_name, index),
            )


if __name__ == "__main__":
    unittest.main()
