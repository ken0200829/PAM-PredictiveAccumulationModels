import unittest
from types import SimpleNamespace

import numpy as np

from pam_dot_task_python.cue_r1_mini import (
    MINI_REPETITIONS,
    cue_r1_mini_cells,
    cue_r1_mini_manifest_digest,
    select_mini_subject_indices,
)


class CueR1MiniContractTests(unittest.TestCase):
    def test_grid_has_six_cells_and_two_repetitions(self):
        cells = cue_r1_mini_cells()
        self.assertEqual(len(cells), 6)
        self.assertEqual(MINI_REPETITIONS, 2)
        self.assertEqual(
            {(cell.locus, cell.effect_level) for cell in cells},
            {("null", "zero"), ("w", "medium"), ("v0", "medium")},
        )

    def test_selection_balances_four_conditions_without_using_outcomes(self):
        subjects = []
        counts = {"normal": 10, "normal_cb": 10, "reverse": 7, "reverse_cb": 10}
        for condition, count in counts.items():
            for index in range(count):
                subjects.append(
                    SimpleNamespace(
                        subject_id="%s_%02d" % (condition, index),
                        condition=SimpleNamespace(name=condition),
                        likelihood_trials=np.arange(240 + index),
                    )
                )
        selected = select_mini_subject_indices(subjects)
        self.assertEqual(len(selected), 8)
        selected_conditions = [subjects[index].condition.name for index in selected]
        for condition in counts:
            self.assertEqual(selected_conditions.count(condition), 2)

    def test_manifest_digest_ignores_only_digest_field(self):
        manifest = {"screen_version": "test", "subjects": 8}
        digest = cue_r1_mini_manifest_digest(manifest)
        manifest["manifest_digest"] = digest
        self.assertEqual(cue_r1_mini_manifest_digest(manifest), digest)


if __name__ == "__main__":
    unittest.main()

