import json
import unittest
from dataclasses import replace

import numpy as np

from pam_dot_task_python import (
    CalibrationDesign,
    EffectCalibration,
    EffectTargetSpec,
    JointModel,
    PriorPredictiveAudit,
    PriorPredictivePolicy,
    calibrate_primary_effects,
    calibration_design_digest,
    candidate_manifest,
    fixture_design,
    manifest_digest,
    run_prior_predictive_audit,
)


class PriorPredictiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = fixture_design()
        cue_white = fixture.u[:, 1].copy()
        cls.design = CalibrationDesign(
            subject_id="fixture-a",
            condition="normal",
            stimulus=fixture.u[:, 0].copy(),
            cue_white=cue_white,
            signed_coherence=fixture.u[:, 2].copy(),
            tie=fixture.u[:, 2] == 0.0,
            cue_evidence=(cue_white == 0.0).astype(float),
            likelihood_mask=np.all(np.isfinite(fixture.y), axis=1),
        )

    def test_design_digest_excludes_subject_identifier_and_outcomes(self):
        renamed = replace(self.design, subject_id="completely-different-id")
        self.assertEqual(
            calibration_design_digest((self.design,)),
            calibration_design_digest((renamed,)),
        )
        changed_mask = self.design.likelihood_mask.copy()
        changed_mask[100] = ~changed_mask[100]
        changed = replace(self.design, likelihood_mask=changed_mask)
        self.assertNotEqual(
            calibration_design_digest((self.design,)),
            calibration_design_digest((changed,)),
        )

    def test_default_effect_targets_are_reachable_for_both_architectures(self):
        results = calibrate_primary_effects((self.design,))
        self.assertEqual(len(results), 12)
        self.assertTrue(all(result.reachable for result in results))
        for result in results:
            self.assertAlmostEqual(
                result.achieved_choice_change,
                result.target_choice_change,
                places=10,
            )

    def test_prior_audit_is_identical_when_dummy_outcomes_change(self):
        first_y = np.full((380, 2), np.nan)
        second_y = np.column_stack(
            (np.full(380, 2.5), 1.0 - self.design.stimulus)
        )
        policy = replace(
            PriorPredictivePolicy(),
            draws=2,
            replicates_per_draw=1,
            decision_time_step=0.05,
        )
        design_hash = calibration_design_digest((self.design,))
        audits = []
        for responses in (first_y, second_y):
            model = JointModel(
                self.design.u,
                responses,
                model_id="cue_parallel_w_vbias",
                tie=self.design.tie,
                cue_evidence=self.design.cue_evidence,
            )
            audits.append(
                run_prior_predictive_audit(
                    model,
                    condition=self.design.condition,
                    design_digest=design_hash,
                    included_mask=self.design.likelihood_mask,
                    policy=policy,
                )
            )
        self.assertEqual(audits[0], audits[1])
        self.assertEqual(audits[0].valid_draws + audits[0].invalid_draws, 2)

    def test_candidate_manifest_is_strict_json_and_never_self_freezes(self):
        calibration = EffectCalibration(
            model_id="cue_parallel_w",
            parameter="gamma_w",
            locus="w",
            level="weak",
            target_choice_change=0.005,
            transformed_value=0.02,
            native_value=0.02,
            achieved_choice_change=0.005,
            reachable=True,
        )
        audit = PriorPredictiveAudit(
            model_id="cue_parallel_w_vbias",
            condition="normal",
            design_digest=calibration_design_digest((self.design,)),
            sampling_scope="cue_response_effects_only",
            sampled_parameter_names=(
                "ddm.b_H_w",
                "ddm.gamma_w",
                "ddm.gamma_v0",
            ),
            requested_draws=1,
            valid_draws=1,
            invalid_draws=0,
            invalid_draw_rate=0.0,
            extreme_w_fraction=0.0,
            median_captured_mass=1.0,
            minimum_captured_mass=0.99,
            maximum_captured_mass=1.01,
            median_choice_rate=0.5,
            median_rt_q10=0.3,
            median_rt_q50=0.8,
            median_rt_q90=1.5,
            invalid_reason_counts={},
            passed=True,
        )
        manifest = candidate_manifest(
            (self.design,),
            (calibration,),
            (audit,),
            EffectTargetSpec(),
            PriorPredictivePolicy(),
        )
        json.dumps(manifest, allow_nan=False)
        self.assertEqual(manifest["status"], "candidate_not_frozen")
        self.assertFalse(manifest["frozen_before_real_data_model_fit"])
        digest = manifest_digest(manifest)
        manifest["manifest_digest"] = digest
        self.assertEqual(manifest_digest(manifest), digest)

    def test_empty_results_cannot_vacuously_pass(self):
        manifest = candidate_manifest(
            (self.design,),
            (),
            (),
            EffectTargetSpec(),
            PriorPredictivePolicy(),
        )
        self.assertFalse(manifest["all_effect_targets_reachable"])
        self.assertFalse(manifest["all_prior_predictive_audits_passed"])


if __name__ == "__main__":
    unittest.main()
