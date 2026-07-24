"""MAP-point and Laplace-formula checks against a MATLAB Online fixture."""

from pathlib import Path
import unittest

import numpy as np

from pam_dot_task_python.fixtures import fixture_design, load_fixture
from pam_dot_task_python.objective import JointModel
from pam_dot_task_python.numerics import QuasiNewtonOptions, RiddersOptions


FIXTURE_DIRECTORY = Path(__file__).parents[1] / "fixtures" / "matlab"


class MatlabJointFixtureTests(unittest.TestCase):
    """Evaluate the Python objective at TAPAS' independently fitted MAP point."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = load_fixture(str(FIXTURE_DIRECTORY / "joint.json"))
        design = fixture_design()
        # This fixture was exported from MATLAB Online before the tie rule of
        # plan section 5.2.1 existed, and the synthetic design carries 20 tie
        # trials per cue, so the MATLAB reference scores them under the
        # retracted "tie counts as black" convention.  Pin the legacy
        # convention here (tie mask all False) so the test keeps checking what
        # it was built to check: that the Python port reproduces TAPAS on the
        # standard path.  The tie rule itself is verified analytically in
        # test_hgf.py and test_response.py, not against MATLAB, because TAPAS
        # has no tie concept to be parity-checked against.  Drop this argument
        # once the fixtures are regenerated with the tie rule applied.
        cls.model = JointModel(
            design.u,
            design.y,
            "ddm_full",
            tie=np.zeros(design.u.shape[0], dtype=bool),
        )
        prc = np.asarray(cls.fixture["p_prc_ptrans"], dtype=float)
        obs = np.asarray(cls.fixture["p_obs_ptrans"], dtype=float)
        cls.matlab_free_parameters = np.concatenate(
            (
                prc[cls.model.hgf_prior.free_mask],
                obs[cls.model.response_prior.free_mask],
            )
        )

    def test_objective_at_matlab_map_matches(self):
        evaluation = self.model.evaluate(self.matlab_free_parameters)
        self.assertAlmostEqual(
            evaluation.negative_log_likelihood,
            float(self.fixture["negLl"]),
            delta=1e-12,
        )
        self.assertAlmostEqual(
            evaluation.negative_log_joint,
            float(self.fixture["negLj"]),
            delta=1e-12,
        )
        np.testing.assert_allclose(
            evaluation.hgf.active.muhat[:, 0],
            np.asarray(self.fixture["muhat_active"], dtype=float),
            rtol=0.0,
            atol=1e-12,
        )

    def test_laplace_formula_reproduces_matlab_lme_from_matlab_hessian(self):
        hessian = np.asarray(self.fixture["H"], dtype=float)
        sign, log_determinant = np.linalg.slogdet(hessian)
        self.assertGreater(sign, 0)
        dimension = hessian.shape[0]
        lme = (
            -float(self.fixture["negLj"])
            - 0.5 * log_determinant
            - dimension / 2.0 * np.log(2.0 * np.pi)
        )
        self.assertAlmostEqual(lme, float(self.fixture["LME"]), delta=1e-12)

    def test_matlab_hessian_is_positive_definite_and_matches_exported_covariance(self):
        hessian = np.asarray(self.fixture["H"], dtype=float)
        covariance = np.asarray(self.fixture["Sigma"], dtype=float)
        self.assertGreater(float(np.min(np.linalg.eigvalsh(hessian))), 0.0)
        np.testing.assert_allclose(
            hessian @ covariance,
            np.eye(hessian.shape[0]),
            rtol=0.0,
            atol=1e-10,
        )

    def test_independent_map_and_laplace_recover_matlab_fixture(self):
        """Recover the local solution without seeding from MATLAB parameters.

        SciPy is used only to locate the basin from the model's declared prior
        means. The ported TAPAS BFGS then makes the reported local step and
        computes the Ridders Hessian/Laplace LME. Different floating-point
        evaluation order affects two near-zero cross derivatives, so LME is
        compared at a declared numerical tolerance rather than bitwise.
        """
        scipy_fit = self.model.fit_map_scipy(max_iterations=100)
        self.assertTrue(scipy_fit.optimization.success)
        self.assertLessEqual(
            abs(scipy_fit.evaluation.negative_log_joint - float(self.fixture["negLj"])),
            1e-3,
        )
        np.testing.assert_allclose(
            scipy_fit.optimization.x,
            self.matlab_free_parameters,
            rtol=0.0,
            atol=0.02,
        )

        options = QuasiNewtonOptions(
            max_iterations=5,
            gradient_options=RiddersOptions(min_steps=10, max_steps=30),
        )
        tapas_fit = self.model.fit_map(
            initial=scipy_fit.optimization.x,
            options=options,
            compute_lme=True,
        )
        self.assertEqual(tapas_fit.optimization.convergence_reason, "argument_tolerance")
        self.assertLessEqual(
            tapas_fit.evaluation.negative_log_joint,
            scipy_fit.evaluation.negative_log_joint + 1e-8,
        )
        self.assertIsNotNone(tapas_fit.laplace)
        self.assertFalse(tapas_fit.laplace.used_bfgs_fallback)
        self.assertGreater(tapas_fit.laplace.numerical_minimum_eigenvalue, 0.0)
        self.assertAlmostEqual(
            tapas_fit.laplace.lme,
            float(self.fixture["LME"]),
            delta=0.03,
        )


if __name__ == "__main__":
    unittest.main()
