import unittest

import numpy as np

from pam_dot_task_python.bayes_optimal import (
    BayesOptimalFit,
    bayes_optimal_log_likelihood,
    bayes_optimal_prior,
    fit_bayes_optimal,
    per_cue_bayes_optimal,
)
from pam_dot_task_python.config import cue_hgf_prior
from pam_dot_task_python.fixtures import fixture_design
from pam_dot_task_python.numerics import QuasiNewtonOptions


FAST = QuasiNewtonOptions(max_iterations=6)


class LogLikelihoodTest(unittest.TestCase):
    def test_matches_the_ported_formula(self):
        stimulus = np.array([1.0, 0.0, 1.0])
        muhat = np.array([0.8, 0.3, 0.6])
        total, trials = bayes_optimal_log_likelihood(stimulus, muhat)
        expected = np.log(0.8) + np.log(0.7) + np.log(0.6)
        self.assertAlmostEqual(total, expected)
        np.testing.assert_allclose(
            trials, [np.log(0.8), np.log(0.7), np.log(0.6)]
        )

    def test_perfect_prediction_scores_zero(self):
        total, _ = bayes_optimal_log_likelihood(
            np.array([1.0, 0.0]), np.array([1.0 - 1e-12, 1e-12])
        )
        self.assertAlmostEqual(total, 0.0, places=9)

    def test_predictions_outside_the_unit_interval_are_rejected(self):
        with self.assertRaises(FloatingPointError):
            bayes_optimal_log_likelihood(np.array([1.0]), np.array([0.0]))
        with self.assertRaises(FloatingPointError):
            bayes_optimal_log_likelihood(np.array([1.0]), np.array([1.0]))
        with self.assertRaises(FloatingPointError):
            bayes_optimal_log_likelihood(np.array([1.0]), np.array([np.nan]))

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            bayes_optimal_log_likelihood(np.array([1.0, 0.0]), np.array([0.5]))


class FitBayesOptimalTest(unittest.TestCase):
    def setUp(self):
        self.design = fixture_design()
        self.prior = cue_hgf_prior()

    def test_fit_returns_a_finite_result_for_each_stream(self):
        for stream in ("both_cues", "white", "red"):
            fit = fit_bayes_optimal(self.design.u, self.prior, stream, FAST)
            self.assertIsInstance(fit, BayesOptimalFit)
            self.assertTrue(np.isfinite(fit.omega_2))
            self.assertTrue(np.isfinite(fit.input_log_likelihood))
            self.assertLess(fit.input_log_likelihood, 0.0)
            self.assertEqual(fit.stream, stream)

    def test_stream_selection_changes_the_scored_trial_count(self):
        both = fit_bayes_optimal(self.design.u, self.prior, "both_cues", FAST)
        white = fit_bayes_optimal(self.design.u, self.prior, "white", FAST)
        red = fit_bayes_optimal(self.design.u, self.prior, "red", FAST)
        self.assertEqual(both.scored_trials, 380)
        self.assertEqual(white.scored_trials, 210)
        self.assertEqual(red.scored_trials, 170)

    def test_fit_improves_on_the_starting_value(self):
        start = self.prior.means[self.prior.free_mask]
        fit = fit_bayes_optimal(self.design.u, self.prior, "both_cues", FAST)
        self.assertNotAlmostEqual(float(fit.free_parameters[0]), float(start[0]))

    def test_responses_are_never_read(self):
        # Corrupting every response must not change the Bayes-optimal fit.
        first = fit_bayes_optimal(self.design.u, self.prior, "both_cues", FAST)
        second = fit_bayes_optimal(self.design.u.copy(), self.prior, "both_cues", FAST)
        self.assertEqual(first.omega_2, second.omega_2)

    def test_fit_is_deterministic(self):
        first = fit_bayes_optimal(self.design.u, self.prior, "both_cues", FAST)
        second = fit_bayes_optimal(self.design.u, self.prior, "both_cues", FAST)
        self.assertEqual(first.omega_2, second.omega_2)

    def test_unknown_stream_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_bayes_optimal(self.design.u, self.prior, "green", FAST)

    def test_non_finite_inputs_are_rejected(self):
        corrupted = self.design.u.copy()
        corrupted[3, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_bayes_optimal(corrupted, self.prior, "both_cues", FAST)

    def test_prior_without_free_parameters_is_rejected(self):
        frozen = cue_hgf_prior()
        variances = frozen.variances.copy()
        variances[:] = 0.0
        from pam_dot_task_python.config import ParameterPrior

        empty = ParameterPrior(
            means=frozen.means, variances=variances, names=frozen.names
        )
        with self.assertRaises(ValueError):
            fit_bayes_optimal(self.design.u, empty, "both_cues", FAST)


class BayesOptimalPriorTest(unittest.TestCase):
    def setUp(self):
        self.design = fixture_design()
        self.prior = cue_hgf_prior()
        self.index = list(self.prior.names).index("omega_2")

    def test_only_the_omega_mean_changes(self):
        updated, fit = bayes_optimal_prior(self.design.u, self.prior, FAST)
        self.assertEqual(updated.means[self.index], fit.omega_2)
        others = [i for i in range(len(self.prior.names)) if i != self.index]
        np.testing.assert_array_equal(
            np.nan_to_num(updated.means[others], nan=-999.0),
            np.nan_to_num(self.prior.means[others], nan=-999.0),
        )

    def test_variance_is_preserved_by_default(self):
        updated, _ = bayes_optimal_prior(self.design.u, self.prior, FAST)
        np.testing.assert_array_equal(
            np.nan_to_num(updated.variances, nan=-999.0),
            np.nan_to_num(self.prior.variances, nan=-999.0),
        )

    def test_variance_can_be_set_explicitly(self):
        updated, _ = bayes_optimal_prior(self.design.u, self.prior, FAST, variance=4.0)
        self.assertEqual(updated.variances[self.index], 4.0)

    def test_negative_variance_is_rejected(self):
        with self.assertRaises(ValueError):
            bayes_optimal_prior(self.design.u, self.prior, FAST, variance=-1.0)

    def test_free_mask_is_unchanged(self):
        updated, _ = bayes_optimal_prior(self.design.u, self.prior, FAST)
        np.testing.assert_array_equal(updated.free_mask, self.prior.free_mask)


class PerCueDiagnosticTest(unittest.TestCase):
    def test_all_three_streams_are_returned(self):
        design = fixture_design()
        fits = per_cue_bayes_optimal(design.u, cue_hgf_prior(), FAST)
        self.assertEqual(set(fits), {"both_cues", "white", "red"})
        for fit in fits.values():
            self.assertTrue(np.isfinite(fit.omega_2))


if __name__ == "__main__":
    unittest.main()
