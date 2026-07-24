import unittest

import numpy as np

from pam_dot_task_python.cue_r1 import (
    ARCHITECTURE_MODELS,
    R1_REPETITIONS,
    cue_r1_cells,
    cue_r1_execution_addendum_digest,
    cue_r1_manifest,
    cue_r1_manifest_digest,
    r1_truth_vector,
)
from pam_dot_task_python.fixtures import fixture_design
from pam_dot_task_python.objective import JointModel


class CueR1ContractTests(unittest.TestCase):
    def test_grid_contains_twenty_cells_and_twenty_repetitions(self):
        cells = cue_r1_cells()
        self.assertEqual(len(cells), 20)
        self.assertEqual(len({cell.identifier for cell in cells}), 20)
        self.assertEqual(R1_REPETITIONS, 20)
        self.assertEqual(sum(cell.primary_gate for cell in cells), 14)

    def test_manifest_is_self_hashing_and_declares_59200_lmes(self):
        manifest = cue_r1_manifest("a" * 64)
        self.assertEqual(manifest["manifest_digest"], cue_r1_manifest_digest(manifest))
        declared = len(manifest["cells"]) * 20 * 37 * 4
        self.assertEqual(declared, 59200)
        self.assertFalse(manifest["outcome_values_used"])

    def test_truth_grid_is_deterministic_and_varies_nuisance_and_effect(self):
        fixture = fixture_design()
        cue = fixture.u[:, 1]
        evidence = (cue == 0.0).astype(float)
        model = JointModel(
            fixture.u,
            fixture.y,
            model_id="cue_parallel_w_vbias",
            tie=fixture.u[:, 2] == 0.0,
            cue_evidence=evidence,
        )
        cell = next(
            item
            for item in cue_r1_cells()
            if item.identifier == "parallel__w_v0__medium"
        )
        first = r1_truth_vector(model, cell, 0, 0)
        again = r1_truth_vector(model, cell, 0, 0)
        other = r1_truth_vector(model, cell, 0, 1)
        np.testing.assert_array_equal(first, again)
        self.assertFalse(np.array_equal(first, other))
        gamma_w = model.free_parameter_names.index("ddm.gamma_w")
        gamma_v0 = model.free_parameter_names.index("ddm.gamma_v0")
        self.assertGreater(first[gamma_w], 0.0)
        self.assertGreater(first[gamma_v0], 0.0)

    def test_each_architecture_has_four_distinct_candidates(self):
        for models in ARCHITECTURE_MODELS.values():
            self.assertEqual(len(models), 4)
            self.assertEqual(len(set(models)), 4)

    def test_execution_addendum_digest_ignores_only_its_digest_field(self):
        addendum = {
            "addendum_version": "test",
            "parent_manifest_digest": "a" * 64,
            "hgf_cache_size": 256,
        }
        digest = cue_r1_execution_addendum_digest(addendum)
        addendum["addendum_digest"] = digest
        self.assertEqual(cue_r1_execution_addendum_digest(addendum), digest)


if __name__ == "__main__":
    unittest.main()
