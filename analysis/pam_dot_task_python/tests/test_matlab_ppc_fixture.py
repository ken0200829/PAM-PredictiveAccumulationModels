"""Aggregate and sequential PPC checks against the MATLAB Online fixture."""

from pathlib import Path
import unittest

import numpy as np

from pam_dot_task_python.fixtures import fixture_design, load_fixture
from pam_dot_task_python.ppc import aggregate_ppc, sequential_ppc, simulate_ddm
from pam_dot_task_python.response import TrialwiseDDM


FIXTURE_DIRECTORY = Path(__file__).parents[1] / "fixtures" / "matlab"
STATISTIC_COUNT = 7


class MatlabPPCFixtureTests(unittest.TestCase):
    """Check PPC windowing exactly and predictive summaries distributionally."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = load_fixture(str(FIXTURE_DIRECTORY / "ppc.json"))
        simulation_fixture = load_fixture(str(FIXTURE_DIRECTORY / "simulation.json"))
        trial_count = len(simulation_fixture["trialwise_w"])
        placeholder = np.full(trial_count, np.nan)
        trialwise = TrialwiseDDM(
            precision_modulator=placeholder,
            w=np.asarray(simulation_fixture["trialwise_w"], dtype=float),
            a=np.asarray(simulation_fixture["trialwise_a"], dtype=float),
            v=np.asarray(simulation_fixture["trialwise_v"], dtype=float),
            Ter=np.asarray(simulation_fixture["trialwise_Ter"], dtype=float),
            coherence_magnitude=placeholder,
            belief_presented=placeholder,
        )
        cls.simulation = simulate_ddm(
            trialwise,
            replicates=int(cls.fixture["replicates"]),
            seed=20260721,
            decision_time_step=float(cls.fixture["decision_time_step"]),
        )
        cls.audit = fixture_design().audit

    def test_window_metadata_and_observed_statistics_match_exactly(self):
        for name, evaluator, expected_windows, expected_version in (
            ("sequential", sequential_ppc, 49, "1.0.0"),
            ("aggregate", aggregate_ppc, 7, "aggregate-1.0.0"),
        ):
            result = evaluator(self.audit, self.simulation)
            expected = self.fixture[name]
            self.assertEqual(result.spec.version, expected_version)
            self.assertEqual(len(result.spec.windows), expected_windows)
            self.assertEqual(
                [window.identifier for window in result.spec.windows],
                expected["window_ids"],
            )
            self.assertEqual(
                [window.family for window in result.spec.windows], expected["families"]
            )
            self.assertEqual(list(result.spec.statistics), expected["statistics"])
            np.testing.assert_array_equal(
                result.summary["valid_trials"].to_numpy().reshape(expected_windows, STATISTIC_COUNT),
                np.asarray(expected["valid_trials"], dtype=int),
            )
            np.testing.assert_allclose(
                result.summary["observed_value"].to_numpy().reshape(
                    expected_windows, STATISTIC_COUNT
                ),
                np.asarray(expected["observed_statistics"], dtype=float),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )

    def test_predictive_summaries_match_independent_stream_tolerances(self):
        # MATLAB Twister and NumPy's generator deliberately use independent
        # streams. These RMSE limits cover ordinary 200-replicate variation
        # while still detecting an incorrect batch reuse, window, or statistic.
        tolerances = {
            "predictive_median": 0.025,
            "predictive_lower": 0.030,
            "predictive_upper": 0.060,
            "predictive_percentile": 0.035,
            "tail_probability_two_sided": 0.065,
        }
        for name, evaluator in (("sequential", sequential_ppc), ("aggregate", aggregate_ppc)):
            result = evaluator(self.audit, self.simulation)
            expected = self.fixture[name]
            window_count = len(result.spec.windows)
            for column, tolerance in tolerances.items():
                observed = result.summary[column].to_numpy().reshape(
                    window_count, STATISTIC_COUNT
                )
                reference = np.asarray(expected[column], dtype=float)
                rmse = float(np.sqrt(np.nanmean((observed - reference) ** 2)))
                self.assertLessEqual(
                    rmse,
                    tolerance,
                    "%s %s RMSE %.6f exceeds independent-stream tolerance %.6f."
                    % (name, column, rmse, tolerance),
                )


if __name__ == "__main__":
    unittest.main()
