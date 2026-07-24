import unittest

import numpy as np
from scipy.integrate import quad

from pam_dot_task_python.response import (
    CueDDMParameters,
    DDMParameters,
    choice_one_probability,
    cue_ddm_loglik,
    ddm_loglik,
    trialwise_cue_ddm,
    trialwise_ddm,
)
from pam_dot_task_python.wfpt import wfpt_density


class WFPTTests(unittest.TestCase):
    def test_zero_drift_centered_boundaries_integrate_to_one(self):
        lower_mass = 0.0
        for start, stop in ((1e-8, 0.05), (0.05, 0.2), (0.2, 1.0), (1.0, 20.0), (20.0, 100.0)):
            mass, _ = quad(
                lambda t: wfpt_density(t, 0.0, 1.0, 0.5, precision=1e-8),
                start,
                stop,
                limit=500,
            )
            lower_mass += mass
        self.assertAlmostEqual(2.0 * lower_mass, 1.0, places=6)

    def test_vector_broadcast_matches_scalar_calls(self):
        time = np.array([0.2, 0.5, 0.9])
        vector = wfpt_density(time, 0.7, 1.3, 0.4)
        scalar = np.array([wfpt_density(value, 0.7, 1.3, 0.4) for value in time])
        np.testing.assert_allclose(vector, scalar, rtol=0, atol=0)


class ResponseModelTests(unittest.TestCase):
    def test_zero_coherence_slope_is_exactly_nested(self):
        stimulus = np.array([1, 0, 1, 0], dtype=float)
        coherence = np.array([0.1, -0.1, 0.3, -0.3])
        muhat = np.array([0.55, 0.45, 0.60, 0.40])
        y = np.array([[0.7, 1], [0.85, 0], [0.65, 1], [0.9, 0]], dtype=float)
        official = DDMParameters(1.2, 2.0, 0.1, -0.2, 0.4, 0.2)
        extended = DDMParameters(1.2, 2.0, 0.1, -0.2, 0.4, 0.2, b_c=0.0)
        official_logp, _ = ddm_loglik(y, stimulus, muhat, official)
        extended_logp, _ = ddm_loglik(y, stimulus, muhat, extended, coherence)
        np.testing.assert_allclose(extended_logp, official_logp, rtol=0, atol=0)

    def test_coherence_slope_changes_drift_magnitude(self):
        stimulus = np.array([1, 0], dtype=float)
        muhat = np.array([0.55, 0.45])
        parameters = DDMParameters(1.2, 2.0, 0.0, 0.0, 0.0, 0.2, b_c=1.5)
        result = trialwise_ddm(stimulus, muhat, parameters, np.array([0.1, -0.3]))
        self.assertAlmostEqual(abs(result.v[1]) - abs(result.v[0]), 0.3, places=15)


class TieTrialDriftTests(unittest.TestCase):
    """Drift is zero on tie trials (plan section 5.2.1)."""

    def setUp(self):
        self.stimulus = np.array([1.0, 0.0, 1.0, 0.0])
        self.muhat = np.array([0.62, 0.41, 0.55, 0.48])
        self.coherence = np.array([0.3, -0.3, 0.0, 0.0])
        self.parameters = DDMParameters(
            a_a=1.2, a_v=2.0, b_w=0.4, b_a=-0.2, b_v=0.5, b_c=0.8, Ter=0.2
        )

    def test_drift_is_zero_on_tie_trials(self):
        tie = self.coherence == 0.0
        trialwise = trialwise_ddm(
            self.stimulus, self.muhat, self.parameters, self.coherence, tie
        )
        np.testing.assert_array_equal(trialwise.v[tie], 0.0)

    def test_non_tie_drift_is_bit_identical_to_the_official_path(self):
        tie = self.coherence == 0.0
        tied = trialwise_ddm(
            self.stimulus, self.muhat, self.parameters, self.coherence, tie
        )
        untied = trialwise_ddm(
            self.stimulus, self.muhat, self.parameters, self.coherence
        )
        np.testing.assert_array_equal(tied.v[~tie], untied.v[~tie])

    def test_starting_point_and_boundary_are_untouched_by_the_tie_rule(self):
        """Only drift loses its direction; belief still drives w and a."""

        tie = self.coherence == 0.0
        tied = trialwise_ddm(
            self.stimulus, self.muhat, self.parameters, self.coherence, tie
        )
        untied = trialwise_ddm(
            self.stimulus, self.muhat, self.parameters, self.coherence
        )
        np.testing.assert_array_equal(tied.w, untied.w)
        np.testing.assert_array_equal(tied.a, untied.a)
        self.assertTrue(np.any(tied.w[tie] != 0.5))

    def test_tie_with_non_zero_coherence_is_rejected(self):
        bad = np.array([True, False, False, False])
        with self.assertRaises(ValueError):
            trialwise_ddm(
                self.stimulus, self.muhat, self.parameters, self.coherence, bad
            )

    def test_tie_mask_length_is_validated(self):
        with self.assertRaises(ValueError):
            trialwise_ddm(
                self.stimulus,
                self.muhat,
                self.parameters,
                self.coherence,
                np.zeros(2, dtype=bool),
            )


class CueLocusResponseTests(unittest.TestCase):
    def setUp(self):
        self.stimulus = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        self.prediction = np.array([0.70, 0.30, 0.60, 0.40, 0.80, 0.20])
        self.coherence = np.array([0.30, -0.30, 0.0, 0.0, 0.10, -0.10])
        self.tie = self.coherence == 0.0
        self.cue_evidence = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        self.parameters = CueDDMParameters(
            a_a=1.2,
            a_v=2.0,
            b_c=0.8,
            Ter=0.2,
            b_H_w=0.4,
            b_w=0.4,
            b_v=0.7,
            gamma_w=1.0,
            gamma_v0=0.7,
        )

    def evaluate(self, architecture, parameters=None):
        return trialwise_cue_ddm(
            architecture,
            self.stimulus,
            self.prediction,
            self.cue_evidence,
            self.parameters if parameters is None else parameters,
            self.coherence,
            self.tie,
        )

    def test_parallel_zero_effect_is_exactly_nested_in_history_model(self):
        parameters = CueDDMParameters(
            a_a=1.2,
            a_v=2.0,
            b_c=0.8,
            Ter=0.2,
            b_H_w=0.4,
        )
        history = self.evaluate("history", parameters)
        parallel = self.evaluate("parallel", parameters)
        np.testing.assert_array_equal(parallel.w, history.w)
        np.testing.assert_array_equal(parallel.v, history.v)

    def test_parallel_zero_effect_has_identical_trial_log_likelihood(self):
        parameters = CueDDMParameters(
            a_a=1.2,
            a_v=2.0,
            b_c=0.8,
            Ter=0.2,
            b_H_w=0.4,
        )
        y = np.column_stack((np.full(self.stimulus.size, 0.8), self.stimulus))
        history_logp, _ = cue_ddm_loglik(
            y,
            "history",
            self.stimulus,
            self.prediction,
            self.cue_evidence,
            parameters,
            self.coherence,
            self.tie,
        )
        parallel_logp, _ = cue_ddm_loglik(
            y,
            "parallel",
            self.stimulus,
            self.prediction,
            self.cue_evidence,
            parameters,
            self.coherence,
            self.tie,
        )
        np.testing.assert_array_equal(parallel_logp, history_logp)

    def test_parallel_starting_point_uses_logit_addition_without_clipping(self):
        result = self.evaluate("parallel")
        self.assertTrue(np.all(result.w > 0.0))
        self.assertTrue(np.all(result.w < 1.0))
        white_cue = self.cue_evidence == 0.0
        expected_history = 0.5 + self.parameters.b_H_w * (
            self.prediction - 0.5
        )
        np.testing.assert_array_equal(result.w[white_cue], expected_history[white_cue])

    def test_all_architectures_force_zero_drift_on_ties(self):
        for architecture in ("history", "parallel", "integrated"):
            with self.subTest(architecture=architecture):
                result = self.evaluate(architecture)
                np.testing.assert_array_equal(result.v[self.tie], 0.0)
                np.testing.assert_array_equal(result.cue_drift_bias[self.tie], 0.0)
                np.testing.assert_array_equal(result.belief_drift_bias[self.tie], 0.0)

    def test_parallel_v0_is_an_additive_non_tie_cue_bias(self):
        result = self.evaluate("parallel")
        direction = 2.0 * self.stimulus - 1.0
        expected = direction * (
            self.parameters.a_v + self.parameters.b_c * np.abs(self.coherence)
        ) + self.parameters.gamma_v0 * self.cue_evidence
        expected[self.tie] = 0.0
        np.testing.assert_allclose(result.v, expected, rtol=0, atol=0)

    def test_integrated_vbias_is_the_existing_pam_bias_algebra(self):
        integrated = self.evaluate("integrated")
        legacy_parameters = DDMParameters(
            a_a=self.parameters.a_a,
            a_v=self.parameters.a_v,
            b_w=self.parameters.b_w,
            b_a=0.0,
            b_v=self.parameters.b_v,
            b_c=self.parameters.b_c,
            Ter=self.parameters.Ter,
        )
        legacy = trialwise_ddm(
            self.stimulus,
            self.prediction,
            legacy_parameters,
            self.coherence,
            self.tie,
        )
        np.testing.assert_array_equal(integrated.w, legacy.w)
        np.testing.assert_allclose(integrated.v, legacy.v, rtol=0, atol=2e-16)

    def test_integrated_response_does_not_add_raw_cue_evidence(self):
        original = self.evaluate("integrated")
        changed_cue = trialwise_cue_ddm(
            "integrated",
            self.stimulus,
            self.prediction,
            -self.cue_evidence,
            self.parameters,
            self.coherence,
            self.tie,
        )
        np.testing.assert_array_equal(changed_cue.w, original.w)
        np.testing.assert_array_equal(changed_cue.v, original.v)

    def test_boundary_aligned_category_reversal_is_symmetric(self):
        for architecture in ("parallel", "integrated"):
            original = self.evaluate(architecture)
            reversed_result = trialwise_cue_ddm(
                architecture,
                1.0 - self.stimulus,
                1.0 - self.prediction,
                -self.cue_evidence,
                self.parameters,
                -self.coherence,
                self.tie,
            )
            with self.subTest(architecture=architecture):
                np.testing.assert_allclose(
                    reversed_result.w, 1.0 - original.w, rtol=0, atol=2e-16
                )
                np.testing.assert_allclose(
                    reversed_result.v, -original.v, rtol=0, atol=2e-16
                )

    def test_closed_form_choice_probability_matches_wfpt_mass(self):
        result = self.evaluate("parallel")
        trial = 0
        expected = 0.0
        for start, stop in (
            (1e-8, 0.05),
            (0.05, 0.2),
            (0.2, 1.0),
            (1.0, 5.0),
            (5.0, 20.0),
        ):
            mass, _ = quad(
                lambda time: wfpt_density(
                    time,
                    -result.v[trial],
                    result.a[trial],
                    1.0 - result.w[trial],
                    precision=1e-8,
                ),
                start,
                stop,
                limit=500,
            )
            expected += mass
        actual = choice_one_probability(result)[trial]
        self.assertAlmostEqual(actual, expected, places=6)

    def test_zero_drift_choice_probability_is_the_starting_point(self):
        result = self.evaluate("parallel")
        probability = choice_one_probability(result)
        np.testing.assert_array_equal(probability[self.tie], result.w[self.tie])


if __name__ == "__main__":
    unittest.main()
