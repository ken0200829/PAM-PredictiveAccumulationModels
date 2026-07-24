"""Regression checks against MATLAB Online SPM BMS fixture summaries."""

from pathlib import Path
import unittest

import numpy as np

from pam_dot_task_python.bms import random_effects_bms
from pam_dot_task_python.fixtures import load_fixture


FIXTURE_DIRECTORY = Path(__file__).parents[1] / "fixtures" / "matlab"


class MatlabBMSFixtureTests(unittest.TestCase):
    """SPM BMS parity, allowing for independent MATLAB/NumPy gamma streams."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = load_fixture(str(FIXTURE_DIRECTORY / "bms.json"))
        lme = np.asarray(cls.fixture["lme"], dtype=float)
        cls.result = random_effects_bms(
            lme,
            model_ids=("model_1", "model_2", "model_3"),
            subject_ids=tuple("subject_%02d" % index for index in range(lme.shape[0])),
            alpha0=np.asarray(cls.fixture["alpha0"], dtype=float),
            samples=int(cls.fixture["Nsamp"]),
            seed=int(cls.fixture["gibbs_seed"]),
            run_sensitivity=True,
        )

    def test_gibbs_summaries_match_within_declared_monte_carlo_tolerance(self):
        matlab = self.fixture
        # MATLAB and NumPy both use MT19937 for uniforms but do not share a
        # Gamma sampler. These are independent Markov chains, so the declared
        # 0.02 bound is intentionally larger than iid binomial sampling error.
        tolerance = 0.02
        np.testing.assert_allclose(
            self.result.primary.expected_frequency,
            np.asarray(matlab["gibbs_exp_r"], dtype=float),
            rtol=0.0,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            self.result.primary.exceedance_probability,
            np.asarray(matlab["gibbs_xp"], dtype=float),
            rtol=0.0,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            np.mean(self.result.primary.subject_model_probability, axis=0),
            np.asarray(matlab["gibbs_g_post_mean"], dtype=float),
            rtol=0.0,
            atol=tolerance,
        )

    def test_protected_bms_variational_quantities_match_matlab(self):
        matlab = self.fixture
        sensitivity = self.result.sensitivity
        self.assertIsNotNone(sensitivity)
        np.testing.assert_allclose(
            sensitivity.alpha,
            np.asarray(matlab["bms_alpha"], dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            sensitivity.expected_frequency,
            np.asarray(matlab["bms_exp_r"], dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertAlmostEqual(
            sensitivity.bayes_omnibus_risk,
            float(matlab["bms_bor"]),
            delta=1e-12,
        )

    def test_protected_exceedance_summaries_match_within_monte_carlo_tolerance(self):
        matlab = self.fixture
        sensitivity = self.result.sensitivity
        self.assertIsNotNone(sensitivity)
        np.testing.assert_allclose(
            sensitivity.exceedance_probability,
            np.asarray(matlab["bms_xp"], dtype=float),
            rtol=0.0,
            atol=0.005,
        )
        np.testing.assert_allclose(
            sensitivity.protected_exceedance_probability,
            np.asarray(matlab["bms_pxp"], dtype=float),
            rtol=0.0,
            atol=0.005,
        )


if __name__ == "__main__":
    unittest.main()
