import unittest

import numpy as np

from pam_dot_task_python.objective import JointModel
from pam_dot_task_python.numerics import QuasiNewtonOptions
from pam_dot_task_python.hgf import CueHGFResult, HGFResult


class JointObjectiveTests(unittest.TestCase):
    def test_initial_joint_objective_is_finite(self):
        trial = np.arange(40)
        stimulus = (trial % 2).astype(float)
        cue = ((trial // 2) % 2).astype(float)
        coherence = np.where(stimulus == 1, 0.3, -0.3)
        u = np.column_stack((stimulus, cue, coherence))
        y = np.column_stack((np.full(40, 0.8), stimulus))
        y[:10] = np.nan
        model = JointModel(u=u, y=y, model_id="ddm_v_c")
        evaluation = model.evaluate(model.initial_free_parameters)
        self.assertTrue(np.isfinite(evaluation.negative_log_joint))
        self.assertEqual(
            model.free_parameter_names,
            ("hgf.omega_2", "ddm.log_a_a", "ddm.log_a_v", "ddm.b_v", "ddm.b_c", "ddm.Ter_logit"),
        )
        self.assertTrue(np.all(np.isnan(evaluation.trial_log_likelihood[:10])))

    def test_tapas_optimizer_is_connected_as_primary_fit_path(self):
        trial = np.arange(20)
        stimulus = (trial % 2).astype(float)
        cue = ((trial // 2) % 2).astype(float)
        coherence = np.where(stimulus == 1, 0.2, -0.2)
        u = np.column_stack((stimulus, cue, coherence))
        y = np.column_stack((np.full(20, 0.8), stimulus))
        y[:5] = np.nan
        model = JointModel(u=u, y=y, model_id="ddm_null")
        fit = model.fit_map(
            options=QuasiNewtonOptions(max_iterations=1, record_trace=False),
            compute_lme=False,
        )
        self.assertEqual(fit.optimization.iterations, 1)
        self.assertIsNone(fit.laplace)
        self.assertTrue(np.isfinite(fit.evaluation.negative_log_joint))

    def test_ter_diagnostic_reports_the_minimum_rt_transform(self):
        stimulus = np.array([0.0, 1.0, 0.0, 1.0])
        u = np.column_stack((stimulus, stimulus, 0.2 * (2.0 * stimulus - 1.0)))
        y = np.column_stack((np.array([0.45, 0.80, 0.70, np.nan]), stimulus))
        model = JointModel(u=u, y=y, model_id="ddm_null")
        parameters = model.initial_free_parameters.copy()
        parameters[-1] = np.log(3.0)
        diagnostic = model.ter_diagnostic(parameters)
        self.assertAlmostEqual(diagnostic.minimum_fitted_rt, 0.45)
        self.assertAlmostEqual(diagnostic.ter_fraction_of_minimum_rt, 0.75)
        self.assertAlmostEqual(diagnostic.ter, 0.3375)
        self.assertAlmostEqual(diagnostic.decision_time_slack, 0.1125)
        self.assertAlmostEqual(diagnostic.decision_time_fraction, 0.25)
        self.assertAlmostEqual(diagnostic.transformed_ter, np.log(3.0))

    def test_cue_models_select_the_declared_perceptual_path(self):
        u, y, cue_evidence = _cue_inputs()
        history = JointModel(
            u=u,
            y=y,
            model_id="cue_parallel_w_vbias",
            cue_evidence=cue_evidence,
        )
        integrated = JointModel(
            u=u,
            y=y,
            model_id="cue_integrated_w_vbias",
            cue_evidence=cue_evidence,
        )
        history_evaluation = history.evaluate(history.initial_free_parameters)
        integrated_evaluation = integrated.evaluate(integrated.initial_free_parameters)
        self.assertIsInstance(history_evaluation.hgf, HGFResult)
        self.assertNotIsInstance(history_evaluation.hgf, CueHGFResult)
        self.assertIsInstance(integrated_evaluation.hgf, CueHGFResult)
        self.assertTrue(np.isfinite(history_evaluation.negative_log_joint))
        self.assertTrue(np.isfinite(integrated_evaluation.negative_log_joint))

    def test_every_cue_model_has_finite_initial_objective_and_exact_names(self):
        u, y, cue_evidence = _cue_inputs()
        expected_effects = {
            "cue_history_w": ("b_H_w",),
            "cue_parallel_w": ("b_H_w", "gamma_w"),
            "cue_parallel_vbias": ("b_H_w", "gamma_v0"),
            "cue_parallel_w_vbias": ("b_H_w", "gamma_w", "gamma_v0"),
            "cue_integrated_w": ("b_w",),
            "cue_integrated_vbias": ("b_v",),
            "cue_integrated_w_vbias": ("b_w", "b_v"),
        }
        for model_id, effects in expected_effects.items():
            model = JointModel(
                u=u, y=y, model_id=model_id, cue_evidence=cue_evidence
            )
            with self.subTest(model_id=model_id):
                expected_names = (
                    "hgf.omega_2",
                    "ddm.log_a_a",
                    "ddm.log_a_v",
                ) + tuple("ddm." + effect for effect in effects) + (
                    "ddm.b_c",
                    "ddm.Ter_logit",
                )
                self.assertEqual(model.free_parameter_names, expected_names)
                self.assertTrue(
                    np.isfinite(model.evaluate(model.initial_free_parameters).negative_log_joint)
                )

    def test_cue_model_requires_boundary_aligned_cue_evidence(self):
        u, y, cue_evidence = _cue_inputs()
        with self.assertRaisesRegex(ValueError, "require signed cue evidence"):
            JointModel(u=u, y=y, model_id="cue_parallel_w")
        invalid = cue_evidence.copy()
        invalid[u[:, 1] == 1.0] = 1.0
        with self.assertRaisesRegex(ValueError, "zero for white"):
            JointModel(
                u=u,
                y=y,
                model_id="cue_parallel_w",
                cue_evidence=invalid,
            )

    def test_hgf_cache_reuses_only_identical_hgf_parameters(self):
        u, y, cue_evidence = _cue_inputs()
        model = JointModel(
            u=u,
            y=y,
            model_id="cue_integrated_w_vbias",
            cue_evidence=cue_evidence,
            hgf_cache_size=8,
        )
        initial = model.initial_free_parameters.copy()
        first = model.evaluate(initial)
        ddm_change = initial.copy()
        ddm_change[1] += 0.1
        second = model.evaluate(ddm_change)
        hgf_change = initial.copy()
        hgf_change[0] += 0.1
        third = model.evaluate(hgf_change)
        diagnostics = model.hgf_cache_diagnostics
        self.assertEqual(diagnostics["hits"], 1)
        self.assertEqual(diagnostics["misses"], 2)
        self.assertEqual(diagnostics["entries"], 2)
        np.testing.assert_array_equal(
            first.hgf.active.muhat,
            second.hgf.active.muhat,
        )
        self.assertFalse(
            np.array_equal(first.hgf.active.muhat, third.hgf.active.muhat)
        )

    def test_hgf_cache_preserves_joint_evaluation_exactly(self):
        u, y, cue_evidence = _cue_inputs()
        uncached = JointModel(
            u=u,
            y=y,
            model_id="cue_parallel_w_vbias",
            cue_evidence=cue_evidence,
        )
        cached = JointModel(
            u=u,
            y=y,
            model_id="cue_parallel_w_vbias",
            cue_evidence=cue_evidence,
            hgf_cache_size=32,
        )
        parameters = uncached.initial_free_parameters.copy()
        for ddm_shift in (0.0, 0.05, -0.1):
            candidate = parameters.copy()
            candidate[-2] += ddm_shift
            expected = uncached.evaluate(candidate)
            observed = cached.evaluate(candidate)
            self.assertEqual(
                observed.negative_log_joint,
                expected.negative_log_joint,
            )
            self.assertEqual(
                observed.negative_log_likelihood,
                expected.negative_log_likelihood,
            )
            np.testing.assert_array_equal(
                observed.hgf.muhat,
                expected.hgf.muhat,
            )
            np.testing.assert_array_equal(
                observed.trial_log_likelihood,
                expected.trial_log_likelihood,
            )

    def test_hgf_cache_size_must_be_non_negative(self):
        u, y, cue_evidence = _cue_inputs()
        with self.assertRaisesRegex(ValueError, "non-negative"):
            JointModel(
                u=u,
                y=y,
                model_id="cue_parallel_w",
                cue_evidence=cue_evidence,
                hgf_cache_size=-1,
            )


def _cue_inputs():
    trial = np.arange(40)
    stimulus = (trial % 2).astype(float)
    cue_white = ((trial // 2) % 2).astype(float)
    coherence = np.where(stimulus == 1, 0.3, -0.3)
    u = np.column_stack((stimulus, cue_white, coherence))
    y = np.column_stack((np.full(40, 0.8), stimulus))
    y[:10] = np.nan
    cue_evidence = (cue_white == 0.0).astype(float)
    return u, y, cue_evidence


if __name__ == "__main__":
    unittest.main()
