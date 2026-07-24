import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from pam_dot_task_python.ppc import (
    PosteriorDrawPolicy,
    SimulationBatch,
    aggregate_ppc,
    make_sequential_spec,
    sequential_ppc,
    simulate_ddm,
    simulate_posterior_ddm,
)
from pam_dot_task_python.numerics import LaplaceResult, QuasiNewtonOptions
from pam_dot_task_python.objective import JointModel
from pam_dot_task_python.response import DDMParameters, trialwise_ddm


class SimulationTests(unittest.TestCase):
    def test_simulation_is_reproducible_and_has_trial_shape(self):
        stimulus = np.array([1.0, 0.0, 1.0, 0.0])
        muhat = np.array([0.55, 0.45, 0.6, 0.4])
        parameters = DDMParameters(1.2, 1.5, 0.1, 0.0, 0.2, 0.15)
        trialwise = trialwise_ddm(stimulus, muhat, parameters)
        first = simulate_ddm(
            trialwise,
            replicates=100,
            seed=19,
            decision_time_step=0.01,
        )
        second = simulate_ddm(
            trialwise,
            replicates=100,
            seed=19,
            decision_time_step=0.01,
        )
        np.testing.assert_array_equal(first.rt, second.rt)
        np.testing.assert_array_equal(first.choice, second.choice)
        self.assertEqual(first.rt.shape, (100, 4))
        self.assertTrue(np.all(first.captured_mass > 0))
        self.assertTrue(np.all(np.isin(first.choice, (0.0, 1.0))))

    def test_posterior_simulation_uses_valid_laplace_draws(self):
        model, fit = _small_fit()
        dimension = fit.optimization.argument_minimum.size
        covariance = np.eye(dimension) * 1e-8
        laplace = LaplaceResult(
            numerical_hessian=np.linalg.inv(covariance),
            numerical_error=np.zeros_like(covariance),
            hessian=np.linalg.inv(covariance),
            covariance=covariance,
            correlation=np.eye(dimension),
            lme=-10.0,
            decomposition={"logjoint": -10.0, "postpredcorr": 0.0, "freepars": 0.0},
            used_bfgs_fallback=False,
            fallback_reason=None,
            numerical_minimum_eigenvalue=1.0,
            numerical_condition_number=1.0,
        )
        fit = replace(fit, laplace=laplace)
        first = simulate_posterior_ddm(
            model,
            fit,
            replicates=4,
            seed=23,
            decision_time_step=0.01,
        )
        second = simulate_posterior_ddm(
            model,
            fit,
            replicates=4,
            seed=23,
            decision_time_step=0.01,
        )
        self.assertEqual(first.parameter_mode, "Laplace_draw")
        self.assertEqual(first.free_parameter_draws.shape, (4, dimension))
        self.assertIsNone(first.draw_diagnostics.fallback_reason)
        np.testing.assert_array_equal(first.rt, second.rt)
        np.testing.assert_array_equal(first.free_parameter_draws, second.free_parameter_draws)

    def test_missing_laplace_records_map_fallback(self):
        model, fit = _small_fit()
        posterior = simulate_posterior_ddm(
            model,
            fit,
            replicates=5,
            seed=29,
            decision_time_step=0.01,
        )
        direct = simulate_ddm(
            fit.evaluation.trialwise,
            replicates=5,
            seed=29,
            decision_time_step=0.01,
        )
        self.assertEqual(posterior.parameter_mode, "MAP_fixed")
        self.assertEqual(
            posterior.draw_diagnostics.fallback_reason,
            "laplace_not_computed",
        )
        np.testing.assert_array_equal(posterior.rt, direct.rt)
        np.testing.assert_array_equal(posterior.choice, direct.choice)

    def test_draw_policy_rejects_unusable_thresholds(self):
        with self.assertRaises(ValueError):
            PosteriorDrawPolicy(max_rejection_rate=1.0)

    def test_simulation_rejects_any_non_deadline_grid(self):
        stimulus = np.array([1.0, 0.0])
        parameters = DDMParameters(1.2, 1.5, 0.1, 0.0, 0.2, 0.15)
        trialwise = trialwise_ddm(stimulus, np.array([0.55, 0.45]), parameters)
        with self.assertRaisesRegex(ValueError, "fixed 3-second response deadline"):
            simulate_ddm(trialwise, replicates=1, max_decision_time=3.1)


class PPCTests(unittest.TestCase):
    def setUp(self):
        self.audit = _audit_fixture()
        rng = np.random.Generator(np.random.MT19937(101))
        choice = rng.binomial(1, 0.5, size=(250, 380)).astype(float)
        rt = np.clip(rng.normal(0.8, 0.12, size=(250, 380)), 0.2, 1.5)
        self.simulation = SimulationBatch(
            rt=rt,
            choice=choice,
            replicates=250,
            seed=101,
            algorithm="MT19937",
            decision_time_step=0.001,
            max_decision_time=3.0,
            captured_mass=np.ones(380),
        )

    def test_sequential_spec_preserves_trial_positions(self):
        spec = make_sequential_spec(self.audit)
        first = spec.windows[0]
        np.testing.assert_array_equal(first.indices, np.arange(100, 128))
        self.assertEqual(first.identifier, "test_global_01")
        self.assertEqual(len(spec.windows), 49)

    def test_sequential_and_aggregate_reuse_batch_without_mutation(self):
        rt_before = self.simulation.rt.copy()
        choice_before = self.simulation.choice.copy()
        sequential = sequential_ppc(self.audit, self.simulation)
        aggregate = aggregate_ppc(self.audit, self.simulation)
        np.testing.assert_array_equal(self.simulation.rt, rt_before)
        np.testing.assert_array_equal(self.simulation.choice, choice_before)
        self.assertEqual(sequential.replicated_statistics.shape, (250, 49, 7))
        self.assertEqual(aggregate.replicated_statistics.shape, (250, 7, 7))
        self.assertTrue(np.isfinite(sequential.global_tail_probability))
        self.assertTrue(np.isfinite(aggregate.simultaneous_threshold))

    def test_invalid_observed_trial_reduces_window_count_only(self):
        audit = self.audit.copy()
        audit.loc[100, "rt_seconds_raw"] = 0.1
        result = sequential_ppc(audit, self.simulation)
        rows = result.summary
        selected = rows[
            (rows.window_id == "test_global_01")
            & (rows.statistic == "choice_rate")
        ]
        self.assertEqual(int(selected.iloc[0].valid_trials), 27)


def _audit_fixture():
    trial = np.arange(1, 381)
    phase = np.where(trial <= 100, "learning", "test")
    test_position = np.maximum(trial - 101, 0)
    cue_white = (test_position % 2).astype(float)
    cue_white[:100] = (trial[:100] % 2).astype(float)
    level = np.array([0.0, 0.1, 0.2, 0.3])[(test_position // 4) % 4]
    stimulus = ((test_position // 2) % 2).astype(float)
    stimulus[level == 0.0] = 0.0
    signed = np.where(stimulus == 1.0, level, -level)
    choice = (trial % 2).astype(float)
    return pd.DataFrame(
        {
            "trial": trial,
            "phase": phase,
            "cue_white": cue_white,
            "signed_coherence": signed,
            "stimulus_category": stimulus,
            "rt_seconds_raw": np.full(380, 0.8),
            "choice_white": choice,
        }
    )


def _small_fit():
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
    return model, fit


if __name__ == "__main__":
    unittest.main()
