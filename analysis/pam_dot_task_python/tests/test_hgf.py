import unittest

import numpy as np

from pam_dot_task_python.config import cue_hgf_prior
from pam_dot_task_python.hgf import (
    binary_hgf,
    cue_binary_hgf,
    cue_blind_binary_hgf,
    transform_ehgf_binary,
)


class HGFTests(unittest.TestCase):
    def setUp(self):
        self.parameters = transform_ehgf_binary(cue_hgf_prior().means)

    def test_neutral_first_prediction_and_finite_trajectory(self):
        result = binary_hgf(np.array([1.0, 0.0, 1.0, 1.0]), self.parameters)
        self.assertAlmostEqual(result.muhat[0, 0], 0.5, places=15)
        self.assertTrue(np.all(np.isfinite(result.muhat[:, 0])))
        self.assertTrue(np.allclose(result.mu[:, 0], [1.0, 0.0, 1.0, 1.0]))

    def test_two_stream_stitch_matches_independent_runs(self):
        stimulus = np.array([1, 0, 0, 1, 1, 0, 1, 0], dtype=float)
        cue = np.array([1, 0, 1, 0, 1, 0, 0, 1], dtype=float)
        combined = cue_binary_hgf(np.column_stack((stimulus, cue)), self.parameters)
        white = binary_hgf(stimulus[cue == 1], self.parameters)
        red = binary_hgf(stimulus[cue == 0], self.parameters)
        np.testing.assert_allclose(
            combined.active.muhat[cue == 1], white.muhat, rtol=0, atol=0
        )
        np.testing.assert_allclose(
            combined.active.muhat[cue == 0], red.muhat, rtol=0, atol=0
        )

    def test_cue_blind_filter_uses_one_global_trial_axis(self):
        stimulus = np.array([1, 0, 0, 1, 1, 0, 1, 0], dtype=float)
        result = cue_blind_binary_hgf(stimulus, self.parameters)
        expected = binary_hgf(stimulus, self.parameters)
        np.testing.assert_array_equal(result.muhat, expected.muhat)

    def test_cue_blind_ties_do_not_update_the_global_history(self):
        stimulus = np.array([1, 0, 0, 1, 1, 0, 1, 0], dtype=float)
        tie = np.zeros(stimulus.size, dtype=bool)
        tie[[2, 5]] = True
        tied = cue_blind_binary_hgf(stimulus, self.parameters, tie)
        informative = ~tie
        dropped = cue_blind_binary_hgf(stimulus[informative], self.parameters)
        np.testing.assert_array_equal(
            tied.muhat[informative], dropped.muhat
        )
        np.testing.assert_array_equal(tied.da[tie], 0.0)


if __name__ == "__main__":
    unittest.main()



class TieTrialTests(unittest.TestCase):
    """Analytic checks for the tie rule of plan section 5.2.1.

    TAPAS has no concept of a tie trial, so there is no MATLAB reference to
    parity-check against.  These properties define the rule instead.
    """

    def setUp(self):
        self.parameters = transform_ehgf_binary(cue_hgf_prior().means)
        self.stimulus = np.array([1, 0, 0, 1, 1, 0, 1, 0], dtype=float)
        self.cue = np.array([1, 0, 1, 0, 1, 0, 0, 1], dtype=float)
        self.u = np.column_stack((self.stimulus, self.cue))

    def test_tie_mask_none_reproduces_untied_run_exactly(self):
        without = cue_binary_hgf(self.u, self.parameters)
        explicit = cue_binary_hgf(
            self.u, self.parameters, np.zeros(self.stimulus.size, dtype=bool)
        )
        np.testing.assert_array_equal(
            without.active.muhat, explicit.active.muhat
        )

    def test_ties_do_not_change_what_informative_trials_learn(self):
        """The belief path over informative trials must ignore ties entirely."""

        tie = np.zeros(self.stimulus.size, dtype=bool)
        tie[[2, 5]] = True
        tied = cue_binary_hgf(self.u, self.parameters, tie)

        informative = ~tie
        dropped = cue_binary_hgf(
            self.u[informative], self.parameters, np.zeros(int(informative.sum()), bool)
        )
        np.testing.assert_allclose(
            tied.active.muhat[informative],
            dropped.active.muhat,
            rtol=0,
            atol=0,
        )

    def test_tie_prediction_equals_next_informative_prediction(self):
        tie = np.zeros(self.stimulus.size, dtype=bool)
        tie[2] = True  # white-cue stream: positions 0, 2, 4, 7 -> tie at 2
        result = cue_binary_hgf(self.u, self.parameters, tie)
        white = np.flatnonzero(self.cue == 1.0)
        tie_position = int(np.flatnonzero(white == 2)[0])
        following = white[tie_position + 1]
        self.assertEqual(
            result.active.muhat[2, 0], result.active.muhat[following, 0]
        )

    def test_tie_produces_no_prediction_error(self):
        tie = np.zeros(self.stimulus.size, dtype=bool)
        tie[[2, 5]] = True
        result = cue_binary_hgf(self.u, self.parameters, tie)
        np.testing.assert_array_equal(result.active.da[tie], 0.0)
        np.testing.assert_array_equal(result.active.wt[tie], 0.0)

    def test_tie_holds_the_posterior_belief(self):
        tie = np.zeros(self.stimulus.size, dtype=bool)
        tie[4] = True  # white stream positions 0, 2, 4, 7
        result = cue_binary_hgf(self.u, self.parameters, tie)
        np.testing.assert_allclose(
            result.active.mu[4], result.active.mu[2], rtol=0, atol=0
        )

    def test_leading_tie_holds_the_prior_belief(self):
        tie = np.zeros(self.stimulus.size, dtype=bool)
        tie[0] = True  # first trial of the white stream
        result = cue_binary_hgf(self.u, self.parameters, tie)
        self.assertAlmostEqual(result.active.muhat[0, 0], 0.5, places=15)
        np.testing.assert_array_equal(result.active.da[0], 0.0)

    def test_trailing_ties_still_receive_a_finite_prediction(self):
        """A tie with no later informative trial needs the padded prediction.

        White-cue trials are 0, 2, 4, 7.  Making trial 7 a tie leaves no
        informative trial after it, so its prediction is the one that follows
        all three earlier white observations -- not the prediction made at
        trial 4, which precedes trial 4's own update.
        """

        tie = np.zeros(self.stimulus.size, dtype=bool)
        tie[[6, 7]] = True  # last trial of each stream
        result = cue_binary_hgf(self.u, self.parameters, tie)
        self.assertTrue(np.all(np.isfinite(result.active.muhat[:, 0])))

        white_informative = self.stimulus[[0, 2, 4]]
        padded = binary_hgf(
            np.concatenate((white_informative, np.zeros(1))), self.parameters
        )
        self.assertEqual(result.active.muhat[7, 0], padded.muhat[-1, 0])
        self.assertNotEqual(result.active.muhat[7, 0], result.active.muhat[4, 0])

    def test_all_tie_stream_stays_at_the_prior(self):
        tie = self.cue == 1.0  # every white-cue trial is a tie
        result = cue_binary_hgf(self.u, self.parameters, tie)
        white = np.flatnonzero(self.cue == 1.0)
        self.assertTrue(np.all(np.isfinite(result.active.muhat[white, 0])))
        np.testing.assert_allclose(
            result.active.muhat[white, 0], 0.5, rtol=0, atol=1e-15
        )

    def test_tie_mask_length_is_validated(self):
        with self.assertRaises(ValueError):
            cue_binary_hgf(self.u, self.parameters, np.zeros(3, dtype=bool))
