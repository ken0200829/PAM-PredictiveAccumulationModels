import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from pam_dot_task_python.gates import (
    GATE_PPC_V1,
    RECOVERY_CRITERIA_V1,
    RECOVERY_GRID_V1,
    RECOVERY_GRID_V2,
    RECOVERY_GRID_W_V3,
    RECOVERY_GRID_FULL_C_V3,
    FINAL_RECOVERY_GRIDS_V3,
    GatePPCFreeze,
    RecoveryCriteria,
    RecoveryGrid,
    evaluate_recovery,
    freeze_digest,
    gate_manifest,
    group_systematic_deviation,
    recovery_verdict,
    window_structure_digest,
)
from pam_dot_task_python.ppc import PPCSpec, PPCWindow
from pam_dot_task_python.recovery import RecoveryResult


class GateFreezeDigestTest(unittest.TestCase):
    def test_digest_is_stable_across_calls(self):
        self.assertEqual(freeze_digest(GATE_PPC_V1), freeze_digest(GATE_PPC_V1))

    def test_digest_changes_when_a_threshold_changes(self):
        altered = replace(GATE_PPC_V1, max_rejection_rate=0.25)
        self.assertNotEqual(freeze_digest(GATE_PPC_V1), freeze_digest(altered))

    def test_manifest_carries_all_declarations(self):
        manifest = gate_manifest()
        self.assertEqual(
            set(manifest), {"gate_ppc", "recovery_grids", "recovery_criteria"}
        )
        for section in (manifest["gate_ppc"], manifest["recovery_criteria"]):
            self.assertIn("declaration", section)
            self.assertEqual(len(section["digest"]), 64)
        self.assertEqual(len(manifest["recovery_grids"]), 2)
        for section in manifest["recovery_grids"]:
            self.assertIn("declaration", section)
            self.assertEqual(len(section["digest"]), 64)

    def test_frozen_values_match_the_documented_gate(self):
        self.assertEqual(GATE_PPC_V1.max_condition_number, 1e8)
        self.assertEqual(GATE_PPC_V1.max_rejection_rate, 0.20)
        self.assertEqual(GATE_PPC_V1.replicates, 2000)
        self.assertEqual(GATE_PPC_V1.response_deadline_seconds, 3.0)
        self.assertEqual(GATE_PPC_V1.sequential_window_count, 49)
        self.assertEqual(GATE_PPC_V1.simultaneous_level, 0.95)

    def test_invalid_freezes_are_rejected(self):
        with self.assertRaises(ValueError):
            GatePPCFreeze(max_rejection_rate=1.0)
        with self.assertRaises(ValueError):
            GatePPCFreeze(simultaneous_level=1.0)
        with self.assertRaises(ValueError):
            GatePPCFreeze(response_deadline_seconds=2.5)

    def test_draw_policy_matches_the_freeze(self):
        policy = GATE_PPC_V1.draw_policy()
        self.assertEqual(policy.max_condition_number, GATE_PPC_V1.max_condition_number)
        self.assertEqual(policy.max_rejection_rate, GATE_PPC_V1.max_rejection_rate)


class SpecValidationTest(unittest.TestCase):
    def _spec(self, version="1.0.0", count=49):
        windows = tuple(
            PPCWindow("w%02d" % index, "test_global", np.arange(3))
            for index in range(count)
        )
        return PPCSpec(version=version, windows=windows)

    def test_matching_spec_passes(self):
        GATE_PPC_V1.validate_spec(self._spec(), "sequential")

    def test_wrong_window_count_fails(self):
        with self.assertRaises(ValueError):
            GATE_PPC_V1.validate_spec(self._spec(count=48), "sequential")

    def test_wrong_version_fails(self):
        with self.assertRaises(ValueError):
            GATE_PPC_V1.validate_spec(self._spec(version="2.0.0"), "sequential")

    def test_unknown_kind_fails(self):
        with self.assertRaises(ValueError):
            GATE_PPC_V1.validate_spec(self._spec(), "other")

    def test_window_digest_detects_a_structural_change(self):
        first = window_structure_digest(self._spec())
        second = window_structure_digest(self._spec(count=48))
        self.assertNotEqual(first, second)


class GroupDeviationRuleTest(unittest.TestCase):
    def test_fraction_above_threshold_is_systematic(self):
        flags = tuple([True] * 8 + [False] * 29)
        outcome = group_systematic_deviation(flags)
        self.assertEqual(outcome["flagged_subjects"], 8)
        self.assertTrue(outcome["systematic_deviation"])

    def test_nominal_rate_is_not_systematic(self):
        flags = tuple([True] * 2 + [False] * 35)
        self.assertFalse(group_systematic_deviation(flags)["systematic_deviation"])

    def test_empty_group_is_rejected(self):
        with self.assertRaises(ValueError):
            group_systematic_deviation(())


class RecoveryGridTest(unittest.TestCase):
    def test_declared_grid_shape_and_unique_seeds(self):
        self.assertEqual(RECOVERY_GRID_V1.model_id, "ddm_v")
        self.assertEqual(len(RECOVERY_GRID_V1.truths), 24)
        self.assertEqual(len(set(RECOVERY_GRID_V1.seeds)), 24)
        self.assertEqual(RECOVERY_GRID_V1.truth_array.shape, (24, 5))

    def test_no_generating_set_switches_off_the_belief_coupling(self):
        slope_index = RECOVERY_GRID_V1.parameter_names.index("ddm.b_v")
        slopes = RECOVERY_GRID_V1.truth_array[:, slope_index]
        self.assertTrue(np.all(np.abs(slopes) > 0))

    def test_every_parameter_varies_across_cases(self):
        spread = np.std(RECOVERY_GRID_V1.truth_array, axis=0)
        self.assertTrue(np.all(spread > 0))

    def test_mismatched_seeds_are_rejected(self):
        with self.assertRaises(ValueError):
            RecoveryGrid(
                version="v",
                model_id="ddm_v",
                parameter_names=("a",),
                truths=((1.0,), (2.0,)),
                seeds=(1,),
            )

    def test_duplicate_seeds_are_rejected(self):
        with self.assertRaises(ValueError):
            RecoveryGrid(
                version="v",
                model_id="ddm_v",
                parameter_names=("a",),
                truths=((1.0,), (2.0,)),
                seeds=(1, 1),
            )


class RecoveryGridV2Test(unittest.TestCase):
    def setUp(self):
        self.grid = RECOVERY_GRID_V2
        self.truth = self.grid.truth_array

    def test_version_and_shape(self):
        self.assertEqual(self.grid.version, "recovery-grid-ddm_v-2.0.0")
        self.assertEqual(self.grid.model_id, "ddm_v")
        self.assertEqual(self.truth.shape, (24, 5))
        self.assertEqual(len(set(self.grid.seeds)), 24)

    def test_omega_spans_the_bayes_optimal_operating_range(self):
        omega = self.truth[:, 0]
        self.assertLessEqual(omega.min(), -5.4)
        self.assertGreaterEqual(omega.max(), -3.2)
        self.assertGreaterEqual(len(np.unique(omega)), 6)
        # Relocated below the old default of -3.
        self.assertLess(omega.max(), -3.0)

    def test_ter_logit_stays_in_the_finite_likelihood_region(self):
        ter = self.truth[:, 4]
        self.assertLessEqual(ter.max(), 2.0)

    def test_slope_is_never_zero(self):
        self.assertTrue(np.all(np.abs(self.truth[:, 3]) > 0))

    def test_every_column_varies(self):
        self.assertTrue(np.all(np.std(self.truth, axis=0) > 0))

    def test_truth_columns_are_nearly_orthogonal(self):
        correlation = np.corrcoef(self.truth.T)
        off_diagonal = np.abs(correlation - np.eye(5))
        self.assertLess(off_diagonal.max(), 0.15)

    def test_nuisance_levels_are_balanced(self):
        for column in (1, 2, 4):
            counts = np.unique(self.truth[:, column], return_counts=True)[1]
            self.assertEqual(len(set(counts)), 1)

    def test_v2_digest_differs_from_v1(self):
        self.assertNotEqual(freeze_digest(RECOVERY_GRID_V1), freeze_digest(self.grid))

    def test_manifest_uses_v2(self):
        manifest = gate_manifest(grids=(RECOVERY_GRID_V2,))
        self.assertEqual(
            manifest["recovery_grids"][0]["declaration"]["version"],
            "recovery-grid-ddm_v-2.0.0",
        )


class FinalRecoveryGridV3Test(unittest.TestCase):
    def test_final_gate_contains_reduced_and_full_models(self):
        self.assertEqual(
            tuple(grid.model_id for grid in FINAL_RECOVERY_GRIDS_V3),
            ("ddm_w", "ddm_full_c"),
        )

    def test_versions_shapes_and_unique_seeds(self):
        expected = (
            (RECOVERY_GRID_W_V3, "recovery-grid-ddm_w-tie_v0-3.1.0", (32, 5)),
            (
                RECOVERY_GRID_FULL_C_V3,
                "recovery-grid-ddm_full_c-tie_v0-3.1.0",
                (32, 8),
            ),
        )
        for grid, version, shape in expected:
            self.assertEqual(grid.version, version)
            self.assertEqual(grid.truth_array.shape, shape)
            self.assertEqual(len(set(grid.seeds)), 32)

    def test_truth_columns_are_exactly_orthogonal_and_balanced(self):
        for grid in FINAL_RECOVERY_GRIDS_V3:
            truth = grid.truth_array
            correlation = np.corrcoef(truth.T)
            np.testing.assert_allclose(correlation, np.eye(truth.shape[1]), atol=1e-14)
            for column in range(truth.shape[1]):
                counts = np.unique(truth[:, column], return_counts=True)[1]
                self.assertEqual(len(set(counts)), 1)

    def test_omega_and_ter_ranges_are_predeclared_and_safe(self):
        for grid in FINAL_RECOVERY_GRIDS_V3:
            omega = grid.truth_array[:, 0]
            ter = grid.truth_array[:, -1]
            self.assertEqual(set(np.round(omega, 10)), {-5.5, -4.9, -3.7, -3.1})
            self.assertGreaterEqual(ter.min(), 1.6)
            self.assertLessEqual(ter.max(), 2.0)

    def test_starting_point_is_nonzero_in_every_v3_case(self):
        for grid in FINAL_RECOVERY_GRIDS_V3:
            index = grid.parameter_names.index("ddm.b_w")
            self.assertTrue(np.all(np.abs(grid.truth_array[:, index]) == 1.2))

    def test_v3_digests_are_distinct_from_superseded_grid(self):
        digests = {freeze_digest(grid) for grid in FINAL_RECOVERY_GRIDS_V3}
        self.assertEqual(len(digests), 2)
        self.assertNotIn(freeze_digest(RECOVERY_GRID_V2), digests)


class RecoveryCriteriaTest(unittest.TestCase):
    def _summary(self, correlation, bias, rmse):
        return pd.DataFrame(
            [
                {
                    "parameter": "ddm.b_v",
                    "cases": 24,
                    "bias": bias,
                    "mean_absolute_error": abs(bias),
                    "rmse": rmse,
                    "correlation": correlation,
                }
            ]
        )

    def _evaluate(self, summary):
        result = RecoveryResult(
            parameter_names=("ddm.b_v",), cases=(), summary=summary
        )
        return evaluate_recovery(result, {"ddm.b_v": 2.0})

    def test_good_recovery_passes(self):
        evaluated = self._evaluate(self._summary(0.95, 0.05, 0.4))
        self.assertTrue(bool(evaluated["passes"].iloc[0]))
        self.assertTrue(recovery_verdict(evaluated)["gate_passed"])

    def test_rmse_at_prior_sd_fails(self):
        evaluated = self._evaluate(self._summary(0.95, 0.05, 2.0001))
        self.assertFalse(bool(evaluated["passes_rmse"].iloc[0]))
        self.assertFalse(recovery_verdict(evaluated)["gate_passed"])

    def test_low_correlation_fails(self):
        evaluated = self._evaluate(self._summary(0.5, 0.05, 0.4))
        self.assertFalse(bool(evaluated["passes_correlation"].iloc[0]))

    def test_large_bias_fails(self):
        evaluated = self._evaluate(self._summary(0.95, 0.9, 0.4))
        self.assertFalse(bool(evaluated["passes_bias"].iloc[0]))

    def test_missing_prior_sd_is_rejected(self):
        result = RecoveryResult(
            parameter_names=("ddm.b_v",),
            cases=(),
            summary=self._summary(0.9, 0.0, 0.3),
        )
        with self.assertRaises(ValueError):
            evaluate_recovery(result, {})

    def test_verdict_lists_failed_parameters(self):
        evaluated = self._evaluate(self._summary(0.1, 0.05, 0.4))
        self.assertEqual(
            recovery_verdict(evaluated)["failed_parameters"], ["ddm.b_v"]
        )

    def test_invalid_criteria_are_rejected(self):
        with self.assertRaises(ValueError):
            RecoveryCriteria(max_absolute_bias=0.0)
        with self.assertRaises(ValueError):
            RecoveryCriteria(min_correlation=1.5)


if __name__ == "__main__":
    unittest.main()
