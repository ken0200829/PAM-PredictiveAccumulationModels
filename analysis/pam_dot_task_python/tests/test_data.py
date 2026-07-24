import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pam_dot_task_python.data import load_subject, resolve_condition


class DataAdapterTests(unittest.TestCase):
    def test_condition_registry(self):
        normal = resolve_condition("normal_dot_task_example.csv")
        reverse_cb = resolve_condition("reverse_cb_dot_task_example.csv")
        self.assertEqual(normal.white_key, "j")
        self.assertFalse(normal.stimulus_reversed)
        self.assertEqual(reverse_cb.white_key, "j")
        self.assertTrue(reverse_cb.stimulus_reversed)

    def test_adapter_keeps_all_trials_and_masks_only_y(self):
        frame = _fixture_frame()
        frame.loc[149, "rt"] = 100
        frame.loc[150, "response"] = ""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normal_dot_task_fixture.csv"
            frame.to_csv(path, index=False)
            subject = load_subject(path)
        self.assertEqual(subject.u.shape, (380, 3))
        self.assertEqual(subject.y.shape, (380, 2))
        self.assertTrue(np.all(np.isfinite(subject.u)))
        self.assertTrue(np.all(np.isnan(subject.y[:100])))
        self.assertTrue(np.all(np.isnan(subject.y[149:151])))
        self.assertEqual(int(subject.audit["likelihood_included"].sum()), 278)

    def test_cue_evidence_is_boundary_aligned_and_audited(self):
        expected = {
            "normal": 1.0,
            "normal_cb": 1.0,
            "reverse": -1.0,
            "reverse_cb": -1.0,
        }
        for condition, red_sign in expected.items():
            frame = _fixture_frame()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / (condition + "_dot_task_fixture.csv")
                frame.to_csv(path, index=False)
                subject = load_subject(path)
            red = subject.audit["cue_red"].to_numpy(dtype=bool)
            with self.subTest(condition=condition):
                self.assertEqual(subject.condition.red_prediction_sign, red_sign)
                np.testing.assert_array_equal(subject.cue_red, red.astype(float))
                np.testing.assert_array_equal(subject.cue_evidence[~red], 0.0)
                np.testing.assert_array_equal(subject.cue_evidence[red], red_sign)
                np.testing.assert_array_equal(
                    subject.audit["cue_evidence"].to_numpy(), subject.cue_evidence
                )

    def test_key_counterbalance_does_not_change_cue_evidence(self):
        subjects = {}
        for condition in ("normal", "normal_cb", "reverse", "reverse_cb"):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / (condition + "_dot_task_fixture.csv")
                _fixture_frame().to_csv(path, index=False)
                subjects[condition] = load_subject(path)
        np.testing.assert_array_equal(
            subjects["normal"].cue_evidence, subjects["normal_cb"].cue_evidence
        )
        np.testing.assert_array_equal(
            subjects["reverse"].cue_evidence, subjects["reverse_cb"].cue_evidence
        )


def _fixture_frame():
    trial = np.arange(1, 381)
    return pd.DataFrame(
        {
            "main_trial_number": trial,
            "rt": np.full(380, 700.0),
            "response": np.where(trial % 2 == 0, "j", "f"),
            "ratio": np.where(trial % 2 == 0, 0.7, 0.3),
            "cross_color": np.where(trial % 2 == 0, "white", "red"),
        }
    )


class TieFlagTests(unittest.TestCase):
    """is_tie must agree exactly with signed_coherence == 0 in both branches."""

    def test_tie_flag_matches_zero_signed_coherence(self):
        for condition in ("normal", "reverse"):
            frame = _fixture_frame()
            frame["ratio"] = 0.5
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / ("%s_dot_task_fixture.csv" % condition)
                frame.to_csv(path, index=False)
                subject = load_subject(path)
            with self.subTest(condition=condition):
                # 1 - 0.5 == 0.5 exactly, so reversal cannot break the flag.
                np.testing.assert_array_equal(subject.audit["signed_coherence"], 0.0)
                self.assertTrue(np.all(subject.is_tie))
                np.testing.assert_array_equal(
                    subject.is_tie, subject.audit["signed_coherence"].to_numpy() == 0.0
                )

    def test_no_tie_when_ratio_is_off_centre(self):
        frame = _fixture_frame()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normal_dot_task_fixture.csv"
            frame.to_csv(path, index=False)
            subject = load_subject(path)
        self.assertFalse(np.any(subject.is_tie))


class ConditionGroundTruthTests(unittest.TestCase):
    """Bind the condition table to the experiment source, not a hand copy.

    A hand-maintained condition table silently inverted the choice labels of
    17 of 37 subjects once already.  The jsPsych task ships the authoritative
    values, so assert against them instead of trusting the copy.
    """

    #: Repo layout: <research root>/PAM-PredictiveAccumulationModels and
    #: <research root>/dot_task are siblings.
    TASK_ROOT = Path(__file__).resolve().parents[4] / "dot_task"

    def test_condition_table_matches_task_config(self):
        if not self.TASK_ROOT.is_dir():
            self.skipTest("dot_task source not available at %s" % self.TASK_ROOT)
        for name in ("normal", "normal_cb", "reverse", "reverse_cb"):
            config = self.TASK_ROOT / name / "src" / "config.js"
            if not config.is_file():
                self.skipTest("Missing task config: %s" % config)
            text = config.read_text(encoding="utf-8")
            reversed_match = re.search(
                r"STIMULUS_REVERSED\s*:\s*(true|false)", text
            )
            white_match = re.search(r"WHITE_KEY\s*:\s*['\"](f|j)['\"]", text)
            self.assertIsNotNone(reversed_match, "STIMULUS_REVERSED not found in %s" % config)
            self.assertIsNotNone(white_match, "WHITE_KEY not found in %s" % config)

            condition = resolve_condition("%s_dot_task_subject.csv" % name)
            with self.subTest(condition=name):
                self.assertEqual(
                    condition.stimulus_reversed,
                    reversed_match.group(1) == "true",
                )
                self.assertEqual(condition.white_key, white_match.group(1))
                self.assertNotEqual(condition.white_key, condition.black_key)


if __name__ == "__main__":
    unittest.main()
