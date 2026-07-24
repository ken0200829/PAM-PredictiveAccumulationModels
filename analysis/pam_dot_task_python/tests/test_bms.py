import unittest

import numpy as np

from pam_dot_task_python.bms import random_effects_bms


class BMSTests(unittest.TestCase):
    def test_gibbs_is_reproducible_with_fixed_seed(self):
        lme = np.zeros((8, 2))
        arguments = dict(
            lme=lme,
            model_ids=("m1", "m2"),
            subject_ids=tuple("s%d" % value for value in range(8)),
            samples=2000,
            seed=42,
        )
        first = random_effects_bms(**arguments)
        second = random_effects_bms(**arguments)
        np.testing.assert_array_equal(
            first.primary.frequency_samples, second.primary.frequency_samples
        )
        np.testing.assert_array_equal(
            first.primary.exceedance_probability,
            second.primary.exceedance_probability,
        )
        self.assertAlmostEqual(np.sum(first.primary.expected_frequency), 1.0)
        self.assertTrue(
            np.allclose(first.primary.subject_model_probability, 0.5, atol=0.03)
        )

    def test_strong_common_evidence_favors_first_model(self):
        lme = np.column_stack((np.full(12, 5.0), np.zeros(12)))
        result = random_effects_bms(
            lme,
            ("winner", "other"),
            tuple("s%d" % value for value in range(12)),
            samples=3000,
            seed=7,
            run_sensitivity=True,
        )
        self.assertGreater(result.primary.exceedance_probability[0], 0.99)
        self.assertGreater(result.sensitivity.expected_frequency[0], 0.85)
        self.assertGreater(
            result.sensitivity.protected_exceedance_probability[0], 0.95
        )
        self.assertGreaterEqual(result.sensitivity.bayes_omnibus_risk, 0.0)
        self.assertLessEqual(result.sensitivity.bayes_omnibus_risk, 1.0)

    def test_symmetric_sensitivity_returns_half_probabilities(self):
        result = random_effects_bms(
            np.zeros((6, 2)),
            ("m1", "m2"),
            tuple("s%d" % value for value in range(6)),
            samples=1000,
            seed=1,
            run_sensitivity=True,
        )
        np.testing.assert_allclose(
            result.sensitivity.expected_frequency, [0.5, 0.5], rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            result.sensitivity.protected_exceedance_probability,
            [0.5, 0.5],
            rtol=0,
            atol=1e-12,
        )

    def test_rejects_incomplete_lme_matrix(self):
        with self.assertRaises(ValueError):
            random_effects_bms(
                np.array([[1.0, np.nan]]),
                ("m1", "m2"),
                ("s1",),
                samples=10,
            )


if __name__ == "__main__":
    unittest.main()
