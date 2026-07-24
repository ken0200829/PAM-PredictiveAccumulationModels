import json
import os
import tempfile
import unittest

import numpy as np

from pam_dot_task_python.fixtures import (
    FIXTURE_SEED,
    INVALID_TEST_TRIALS,
    _Lehmer,
    assert_design_matches,
    fixture_design,
    load_fixture,
)
from pam_dot_task_python.gates import GATE_PPC_V1
from pam_dot_task_python.ppc import make_aggregate_spec, make_sequential_spec


class LehmerStreamTest(unittest.TestCase):
    def test_first_values_match_the_integer_recurrence(self):
        stream = _Lehmer(FIXTURE_SEED)
        state = FIXTURE_SEED
        for _ in range(5):
            state = (16807 * state) % 2147483647
            self.assertEqual(stream.next(), state / 2147483647)

    def test_stream_is_deterministic(self):
        first = [_Lehmer().next() for _ in range(10)]
        second = [_Lehmer().next() for _ in range(10)]
        self.assertEqual(first, second)

    def test_shuffle_is_a_permutation(self):
        values = list(range(50))
        shuffled = _Lehmer().shuffle(values)
        self.assertEqual(sorted(shuffled), values)
        self.assertNotEqual(shuffled, values)

    def test_shuffle_does_not_mutate_its_input(self):
        values = list(range(20))
        _Lehmer().shuffle(values)
        self.assertEqual(values, list(range(20)))


class FixtureDesignTest(unittest.TestCase):
    def setUp(self):
        self.design = fixture_design()

    def test_design_is_reproducible(self):
        again = fixture_design()
        np.testing.assert_array_equal(self.design.u, again.u)
        np.testing.assert_array_equal(
            np.isfinite(self.design.y), np.isfinite(again.y)
        )

    def test_trial_and_phase_layout(self):
        self.assertEqual(self.design.u.shape, (380, 3))
        self.assertEqual(self.design.y.shape, (380, 2))
        self.assertEqual(self.design.phase.count("learning"), 100)
        self.assertEqual(self.design.phase.count("test"), 280)

    def test_cue_counts_match_the_real_task(self):
        cue = self.design.u[:, 1]
        self.assertEqual(int(np.sum(cue == 1)), 210)
        self.assertEqual(int(np.sum(cue == 0)), 170)
        test_cue = cue[100:]
        self.assertEqual(int(np.sum(test_cue == 1)), 140)
        self.assertEqual(int(np.sum(test_cue == 0)), 140)

    def test_learning_responses_are_masked(self):
        self.assertTrue(np.all(~np.isfinite(self.design.y[:100])))

    def test_invalid_test_trials_clear_both_columns(self):
        for trial in INVALID_TEST_TRIALS:
            row = self.design.y[trial - 1]
            self.assertTrue(np.all(~np.isfinite(row)))
        valid = np.isfinite(self.design.y[100:, 0]).sum()
        self.assertEqual(int(valid), 280 - len(INVALID_TEST_TRIALS))

    def test_response_mask_never_keeps_only_one_column(self):
        finite = np.isfinite(self.design.y)
        self.assertTrue(np.all(finite[:, 0] == finite[:, 1]))

    def test_test_coherence_levels_are_the_declared_four(self):
        coherence = np.round(np.abs(self.design.u[100:, 2]), 10)
        np.testing.assert_array_equal(
            np.unique(coherence), np.array([0.0, 0.1, 0.2, 0.3])
        )

    def test_inputs_are_finite_everywhere(self):
        self.assertTrue(np.all(np.isfinite(self.design.u)))

    def test_design_realizes_the_frozen_window_counts(self):
        audit = self.design.audit
        sequential = make_sequential_spec(audit)
        aggregate = make_aggregate_spec(audit)
        self.assertEqual(len(sequential.windows), 49)
        self.assertEqual(len(aggregate.windows), 7)
        GATE_PPC_V1.validate_spec(sequential, "sequential")
        GATE_PPC_V1.validate_spec(aggregate, "aggregate")


class FixtureIOTest(unittest.TestCase):
    def test_non_finite_markers_round_trip(self):
        payload = {"a": ["NaN", "Infinity", "-Infinity", 1.5], "b": {"c": "NaN"}}
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "f.json")
            with open(path, "w") as handle:
                json.dump(payload, handle)
            restored = load_fixture(path)
        self.assertTrue(np.isnan(restored["a"][0]))
        self.assertEqual(restored["a"][1], float("inf"))
        self.assertEqual(restored["a"][2], float("-inf"))
        self.assertEqual(restored["a"][3], 1.5)
        self.assertTrue(np.isnan(restored["b"]["c"]))

    def test_matching_design_passes(self):
        design = fixture_design()
        exported = {"u": design.u.tolist(), "y": design.y.tolist()}
        assert_design_matches(exported, design)

    def test_changed_input_is_detected(self):
        design = fixture_design()
        altered = design.u.copy()
        altered[7, 0] = 1.0 - altered[7, 0]
        with self.assertRaises(AssertionError):
            assert_design_matches(
                {"u": altered.tolist(), "y": design.y.tolist()}, design
            )

    def test_changed_response_mask_is_detected(self):
        design = fixture_design()
        altered = design.y.copy()
        altered[130, :] = [1.0, 1.0]
        with self.assertRaises(AssertionError):
            assert_design_matches(
                {"u": design.u.tolist(), "y": altered.tolist()}, design
            )


if __name__ == "__main__":
    unittest.main()
