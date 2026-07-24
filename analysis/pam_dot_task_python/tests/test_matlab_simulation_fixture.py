"""Distributional simulation checks against a MATLAB Online fixture."""

from pathlib import Path
import unittest

import numpy as np

from pam_dot_task_python.fixtures import load_fixture
from pam_dot_task_python.ppc import simulate_ddm
from pam_dot_task_python.response import TrialwiseDDM


FIXTURE_DIRECTORY = Path(__file__).parents[1] / "fixtures" / "matlab"


class MatlabSimulationFixtureTests(unittest.TestCase):
    """Compare the fixed three-second DDM simulator without matching draws."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = load_fixture(str(FIXTURE_DIRECTORY / "simulation.json"))
        trial_count = len(cls.fixture["trialwise_w"])
        placeholder = np.full(trial_count, np.nan)
        cls.trialwise = TrialwiseDDM(
            precision_modulator=placeholder,
            w=np.asarray(cls.fixture["trialwise_w"], dtype=float),
            a=np.asarray(cls.fixture["trialwise_a"], dtype=float),
            v=np.asarray(cls.fixture["trialwise_v"], dtype=float),
            Ter=np.asarray(cls.fixture["trialwise_Ter"], dtype=float),
            coherence_magnitude=placeholder,
            belief_presented=placeholder,
        )
        cls.simulation = simulate_ddm(
            cls.trialwise,
            replicates=int(cls.fixture["replicates"]),
            seed=20260721,
            decision_time_step=float(cls.fixture["decision_time_step"]),
        )

    def test_captured_probability_mass_matches_matlab(self):
        np.testing.assert_allclose(
            self.simulation.captured_mass,
            np.asarray(self.fixture["captured_mass"], dtype=float),
            rtol=0.0,
            atol=1e-12,
        )

    def test_independent_random_streams_have_matching_summary_distributions(self):
        # MATLAB's twister/Gamma implementation and NumPy's MT19937 do not
        # produce matching draw sequences. With 200 responses per trial, an
        # RMSE of 0.06 accommodates ordinary independent-sample variation
        # while catching a parameterization or response-coding mismatch.
        summaries = (
            (np.mean(self.simulation.rt, axis=0), "rt_mean"),
            (np.median(self.simulation.rt, axis=0), "rt_median"),
            (np.mean(self.simulation.choice, axis=0), "choice_rate"),
        )
        for observed, fixture_name in summaries:
            expected = np.asarray(self.fixture[fixture_name], dtype=float)
            rmse = float(np.sqrt(np.mean((observed - expected) ** 2)))
            self.assertLessEqual(
                rmse,
                0.06,
                "%s RMSE %.6f exceeds independent-stream tolerance."
                % (fixture_name, rmse),
            )


if __name__ == "__main__":
    unittest.main()
