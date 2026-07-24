import unittest

import numpy as np

from pam_dot_task_python.numerics import QuasiNewtonOptions
from pam_dot_task_python.objective import JointModel
from pam_dot_task_python.recovery import (
    run_recovery_case,
    simulate_recovery_dataset,
    summarize_recovery,
)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        trial = np.arange(30)
        stimulus = (trial % 2).astype(float)
        cue = ((trial // 2) % 2).astype(float)
        coherence = np.where(stimulus == 1, 0.2, -0.2)
        u = np.column_stack((stimulus, cue, coherence))
        y = np.column_stack((np.full(30, 0.8), stimulus))
        y[:10] = np.nan
        self.model = JointModel(u=u, y=y, model_id="ddm_null")

    def test_generated_dataset_preserves_mask_and_native_ter(self):
        truth = self.model.initial_free_parameters.copy()
        original_ter = self.model.evaluate(truth).ddm_parameters.Ter
        recovery = simulate_recovery_dataset(
            self.model,
            truth,
            seed=41,
            decision_time_step=0.01,
        )
        self.assertTrue(np.all(np.isnan(recovery.model.y[:10])))
        self.assertTrue(np.all(np.isfinite(recovery.model.y[10:])))
        recovered_scale_ter = recovery.model.evaluate(
            recovery.estimation_scale_truth
        ).ddm_parameters.Ter
        self.assertAlmostEqual(recovered_scale_ter, original_ter, places=14)
        np.testing.assert_array_equal(
            recovery.model.u,
            self.model.u,
        )

    def test_summary_reports_bias_rmse_and_correlation(self):
        truth = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        estimate = truth + np.array([0.5, -0.25])
        summary = summarize_recovery(truth, estimate, ("first", "second"))
        self.assertAlmostEqual(summary.loc[0, "bias"], 0.5)
        self.assertAlmostEqual(summary.loc[1, "rmse"], 0.25)
        np.testing.assert_allclose(summary["correlation"], 1.0)

    def test_recovery_runner_connects_generation_to_tapas_fit(self):
        result = run_recovery_case(
            self.model,
            self.model.initial_free_parameters,
            seed=43,
            optimizer_options=QuasiNewtonOptions(
                max_iterations=1,
                record_trace=False,
            ),
            decision_time_step=0.02,
        )
        self.assertEqual(result.fit.optimization.iterations, 1)
        self.assertEqual(result.error.shape, self.model.initial_free_parameters.shape)
        self.assertTrue(np.all(np.isfinite(result.estimated_free_parameters)))

    def test_cue_recovery_preserves_tie_and_signed_cue_contracts(self):
        cue_evidence = (self.model.u[:, 1] == 0.0).astype(float)
        cue_model = JointModel(
            u=self.model.u,
            y=self.model.y,
            model_id="cue_parallel_w_vbias",
            cue_evidence=cue_evidence,
        )
        recovery = simulate_recovery_dataset(
            cue_model,
            cue_model.initial_free_parameters,
            seed=47,
            decision_time_step=0.02,
        )
        np.testing.assert_array_equal(recovery.model.tie, cue_model.tie)
        np.testing.assert_array_equal(
            recovery.model.cue_evidence, cue_model.cue_evidence
        )
        self.assertEqual(recovery.model.model_id, "cue_parallel_w_vbias")


if __name__ == "__main__":
    unittest.main()
