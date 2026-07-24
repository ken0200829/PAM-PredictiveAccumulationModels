import unittest

import numpy as np

from pam_dot_task_python.numerics import (
    OptimizationTrace,
    QuasiNewtonOptions,
    QuasiNewtonResult,
    RiddersOptions,
    laplace_evidence,
    ridders_gradient,
    ridders_hessian,
    tapas_quasi_newton,
)


class RiddersTests(unittest.TestCase):
    def setUp(self):
        self.hessian = np.array([[4.0, 1.25], [1.25, 3.0]])
        self.linear = np.array([-2.0, 0.75])
        self.point = np.array([0.3, -0.7])

    def function(self, point):
        return 0.5 * point @ self.hessian @ point + self.linear @ point + 1.2

    def test_gradient_matches_quadratic_derivative(self):
        gradient, error = ridders_gradient(self.function, self.point)
        np.testing.assert_allclose(
            gradient, self.hessian @ self.point + self.linear, rtol=0, atol=1e-10
        )
        self.assertTrue(np.all(error < 1e-9))

    def test_hessian_matches_quadratic_curvature(self):
        hessian, error = ridders_hessian(
            self.function, self.point, RiddersOptions(min_steps=10)
        )
        np.testing.assert_allclose(hessian, self.hessian, rtol=0, atol=1e-8)
        self.assertTrue(np.all(error < 1e-7))


class QuasiNewtonTests(unittest.TestCase):
    def test_optimizer_reaches_quadratic_minimum(self):
        hessian = np.array([[5.0, 0.5], [0.5, 2.0]])
        linear = np.array([-3.0, 1.0])

        def function(point):
            return 0.5 * point @ hessian @ point + linear @ point + 4.0

        expected = -np.linalg.solve(hessian, linear)
        result = tapas_quasi_newton(function, np.array([2.0, -2.0]))
        np.testing.assert_allclose(
            result.argument_minimum, expected, rtol=0, atol=2e-3
        )
        self.assertIn(
            result.convergence_reason, ("argument_tolerance", "gradient_tolerance")
        )

    def test_strict_armijo_constant_function_uses_reset_path(self):
        options = QuasiNewtonOptions(
            max_iterations=5,
            max_regularizations=2,
            max_resets=1,
            record_trace=False,
        )
        result = tapas_quasi_newton(lambda point: 1.0, np.array([0.0]), options)
        self.assertEqual(result.reset_count, 1)
        self.assertEqual(result.convergence_reason, "maximum_resets")

    def test_laplace_evidence_matches_quadratic_formula(self):
        hessian = np.array([[3.0, 0.4], [0.4, 2.0]])

        def function(point):
            return 7.0 + 0.5 * point @ hessian @ point

        optimization = tapas_quasi_newton(function, np.array([0.5, -0.5]))
        laplace = laplace_evidence(function, optimization)
        expected = (
            -optimization.value_minimum
            - 0.5 * np.linalg.slogdet(hessian)[1]
            - np.log(2.0 * np.pi)
        )
        self.assertFalse(laplace.used_bfgs_fallback)
        np.testing.assert_allclose(laplace.hessian, hessian, rtol=0, atol=1e-8)
        self.assertAlmostEqual(laplace.lme, expected, places=8)

    def test_nonpositive_numerical_hessian_uses_bfgs_fallback(self):
        trace = OptimizationTrace((), (), (), ())
        optimization = QuasiNewtonResult(
            value_minimum=0.0,
            argument_minimum=np.array([0.0, 0.0]),
            inverse_hessian=np.eye(2),
            iterations=0,
            reset_count=0,
            convergence_reason="fixture",
            gradient=np.zeros(2),
            trace=trace,
        )
        laplace = laplace_evidence(
            lambda point: point[0] ** 2 - point[1] ** 2,
            optimization,
        )
        self.assertTrue(laplace.used_bfgs_fallback)
        self.assertEqual(
            laplace.fallback_reason, "non_positive_definite_numerical_hessian"
        )
        np.testing.assert_allclose(laplace.hessian, np.eye(2), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
