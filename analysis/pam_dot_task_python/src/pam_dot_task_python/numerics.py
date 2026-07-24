"""TAPAS 6.1.0 numerical differentiation, BFGS, and Laplace evidence."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ScalarFunction = Callable[[FloatArray], float]


@dataclass(frozen=True)
class RiddersOptions:
    init_h: float = 1.0
    div: float = 1.2
    min_steps: int = 3
    max_steps: int = 100
    termination_factor: float = 2.0


@dataclass(frozen=True)
class QuasiNewtonOptions:
    """Defaults from ``tapas_quasinewton_optim_config``."""

    tolerance_gradient: float = 1e-3
    tolerance_argument: float = 1e-3
    max_step: float = 1.0
    max_iterations: int = 100
    max_regularizations: int = 16
    max_resets: int = 10
    record_trace: bool = True
    gradient_options: RiddersOptions = field(
        default_factory=lambda: RiddersOptions(min_steps=10)
    )


@dataclass(frozen=True)
class OptimizationTrace:
    arguments: Tuple[FloatArray, ...]
    values: Tuple[float, ...]
    inverse_hessians: Tuple[FloatArray, ...]
    reset_iterations: Tuple[int, ...]


@dataclass(frozen=True)
class QuasiNewtonResult:
    value_minimum: float
    argument_minimum: FloatArray
    inverse_hessian: FloatArray
    iterations: int
    reset_count: int
    convergence_reason: str
    gradient: FloatArray
    trace: OptimizationTrace


@dataclass(frozen=True)
class LaplaceResult:
    numerical_hessian: FloatArray
    numerical_error: FloatArray
    hessian: FloatArray
    covariance: FloatArray
    correlation: FloatArray
    lme: float
    decomposition: Dict[str, float]
    used_bfgs_fallback: bool
    fallback_reason: Optional[str]
    numerical_minimum_eigenvalue: float
    numerical_condition_number: float


def ridders_first(
    function: Callable[[float], float],
    point: float,
    options: RiddersOptions = RiddersOptions(),
) -> Tuple[float, float]:
    """Port of ``tapas_riddersdiff``."""

    return _ridders_extrapolation(
        lambda step: (function(point + step) - function(point - step)) / (2.0 * step),
        options,
    )


def ridders_second(
    function: Callable[[float], float],
    point: float,
    options: RiddersOptions = RiddersOptions(),
) -> Tuple[float, float]:
    """Port of ``tapas_riddersdiff2``."""

    return _ridders_extrapolation(
        lambda step: (
            function(point + step)
            - 2.0 * function(point)
            + function(point - step)
        )
        / step**2,
        options,
    )


def ridders_cross(
    function: Callable[[FloatArray], float],
    point: FloatArray,
    options: RiddersOptions = RiddersOptions(),
) -> Tuple[float, float]:
    """Port of ``tapas_riddersdiffcross``."""

    x = np.asarray(point, dtype=float)
    if x.shape != (2,):
        raise ValueError("Cross differentiation requires a two-element point.")

    def approximation(step: float) -> float:
        return (
            function(x + step)
            - function(x + np.array([step, -step]))
            - function(x + np.array([-step, step]))
            + function(x - step)
        ) / (4.0 * step**2)

    return _ridders_extrapolation(approximation, options)


def ridders_gradient(
    function: ScalarFunction,
    point: FloatArray,
    options: RiddersOptions = RiddersOptions(),
) -> Tuple[FloatArray, FloatArray]:
    """Port of ``tapas_riddersgradient``."""

    x = _validated_point(function, point)
    gradient = np.full(x.size, np.nan)
    error = np.full(x.size, np.nan)
    for index in range(x.size):
        def coordinate(value: float, coordinate_index: int = index) -> float:
            candidate = x.copy()
            candidate[coordinate_index] = value
            return float(function(candidate))

        gradient[index], error[index] = ridders_first(coordinate, x[index], options)
    return gradient, error


def ridders_hessian(
    function: ScalarFunction,
    point: FloatArray,
    options: RiddersOptions = RiddersOptions(),
) -> Tuple[FloatArray, FloatArray]:
    """Port of ``tapas_riddershessian``."""

    x = _validated_point(function, point)
    hessian = np.full((x.size, x.size), np.nan)
    error = np.full((x.size, x.size), np.nan)
    for row in range(x.size):
        def coordinate(value: float, coordinate_index: int = row) -> float:
            candidate = x.copy()
            candidate[coordinate_index] = value
            return float(function(candidate))

        hessian[row, row], error[row, row] = ridders_second(
            coordinate, x[row], options
        )

    for row in range(1, x.size):
        for column in range(row):
            def coordinates(
                values: FloatArray,
                row_index: int = row,
                column_index: int = column,
            ) -> float:
                candidate = x.copy()
                candidate[row_index] = values[0]
                candidate[column_index] = values[1]
                return float(function(candidate))

            derivative, estimate = ridders_cross(
                coordinates, np.array([x[row], x[column]]), options
            )
            hessian[row, column] = derivative
            hessian[column, row] = derivative
            error[row, column] = estimate
            error[column, row] = estimate
    return hessian, error


def tapas_quasi_newton(
    function: ScalarFunction,
    initial: FloatArray,
    options: QuasiNewtonOptions = QuasiNewtonOptions(),
) -> QuasiNewtonResult:
    """Port of TAPAS' reset-capable BFGS quasi-Newton optimizer."""

    init = np.asarray(initial, dtype=float)
    if init.ndim != 1 or init.size == 0 or np.any(~np.isfinite(init)):
        raise ValueError("Initial point must be a finite, non-empty vector.")
    x = init.copy()
    value = float(function(x))
    gradient, _ = ridders_gradient(function, x, options.gradient_options)
    inverse_hessian = np.eye(x.size)
    descent = -gradient
    slope = float(np.dot(gradient, descent))
    reset_count = 0
    reason = "maximum_iterations"
    iterations = 0

    trace_arguments: List[FloatArray] = [x.copy()] if options.record_trace else []
    trace_values: List[float] = [value] if options.record_trace else []
    trace_inverse: List[FloatArray] = (
        [inverse_hessian.copy()] if options.record_trace else []
    )
    reset_iterations: List[int] = []

    for iteration in range(1, options.max_iterations + 1):
        iterations = iteration
        step_size = float(np.sqrt(np.dot(descent, descent)))
        if step_size > options.max_step:
            descent = descent * options.max_step / step_size

        regularization_count = 0
        new_x = np.full_like(x, np.nan)
        new_value = np.nan
        difference = np.nan
        for regularization in range(options.max_regularizations + 1):
            regularization_count = regularization
            scale = 0.5**regularization
            new_x = x + scale * descent
            new_value = float(function(new_x))
            if np.isinf(new_value):
                continue
            difference = new_value - value
            if difference < 1e-4 * scale * slope:
                break

        if regularization_count < options.max_regularizations:
            delta_x = new_x - x
            x = new_x
            value = new_value
            _append_trace(
                options.record_trace,
                trace_arguments,
                trace_values,
                trace_inverse,
                x,
                value,
                inverse_hessian,
            )
        elif reset_count < options.max_resets:
            inverse_hessian = np.eye(x.size)
            x = x + 0.1 * (init - x)
            value = float(function(x))
            gradient, _ = ridders_gradient(function, x, options.gradient_options)
            descent = -gradient
            slope = float(np.dot(gradient, descent))
            reset_count += 1
            reset_iterations.append(iteration)
            _append_trace(
                options.record_trace,
                trace_arguments,
                trace_values,
                trace_inverse,
                x,
                value,
                inverse_hessian,
            )
            continue
        else:
            reason = "maximum_resets"
            break

        relative_step = np.max(np.abs(delta_x) / np.abs(np.maximum(x, 1.0)))
        if relative_step < options.tolerance_argument:
            reason = "argument_tolerance"
            break

        old_gradient = gradient
        gradient, _ = ridders_gradient(function, x, options.gradient_options)
        delta_gradient = gradient - old_gradient
        scaled_gradient = np.max(
            np.abs(gradient) * np.maximum(np.abs(x), 1.0) / max(abs(value), 1.0)
        )
        if scaled_gradient < options.tolerance_gradient:
            reason = "gradient_tolerance"
            break

        curvature = float(np.dot(delta_gradient, delta_x))
        curvature_floor = float(
            np.sqrt(
                np.finfo(float).eps
                * np.dot(delta_gradient, delta_gradient)
                * np.dot(delta_x, delta_x)
            )
        )
        if curvature > curvature_floor:
            gradient_times_hessian = delta_gradient @ inverse_hessian
            projected_curvature = float(gradient_times_hessian @ delta_gradient)
            correction = (
                delta_x / curvature
                - gradient_times_hessian / projected_curvature
            )
            inverse_hessian = (
                inverse_hessian
                + np.outer(delta_x, delta_x) / curvature
                - np.outer(gradient_times_hessian, gradient_times_hessian)
                / projected_curvature
                + projected_curvature * np.outer(correction, correction)
            )

        descent = -(inverse_hessian @ gradient)
        slope = float(np.dot(gradient, descent))
        if options.record_trace:
            trace_inverse[-1] = inverse_hessian.copy()

    trace = OptimizationTrace(
        arguments=tuple(trace_arguments),
        values=tuple(trace_values),
        inverse_hessians=tuple(trace_inverse),
        reset_iterations=tuple(reset_iterations),
    )
    return QuasiNewtonResult(
        value_minimum=value,
        argument_minimum=x,
        inverse_hessian=inverse_hessian,
        iterations=iterations,
        reset_count=reset_count,
        convergence_reason=reason,
        gradient=gradient,
        trace=trace,
    )


def laplace_evidence(
    function: ScalarFunction,
    optimization: QuasiNewtonResult,
    options: RiddersOptions = RiddersOptions(min_steps=10),
) -> LaplaceResult:
    """Port TAPAS' numerical-Hessian and BFGS-fallback LME calculation."""

    numerical_hessian, numerical_error = ridders_hessian(
        function, optimization.argument_minimum, options
    )
    finite = np.all(np.isfinite(numerical_hessian))
    if finite:
        numerical_eigenvalues = np.linalg.eigvalsh(
            (numerical_hessian + numerical_hessian.T) / 2.0
        )
        numerical_minimum = float(np.min(numerical_eigenvalues))
        numerical_condition = float(np.linalg.cond(numerical_hessian))
    else:
        numerical_minimum = np.nan
        numerical_condition = np.inf

    fallback = (not finite) or numerical_minimum <= 0
    fallback_reason = None
    if fallback:
        fallback_reason = (
            "non_finite_numerical_hessian"
            if not finite
            else "non_positive_definite_numerical_hessian"
        )
        hessian = np.linalg.inv(optimization.inverse_hessian)
        covariance = optimization.inverse_hessian.copy()
    else:
        hessian = numerical_hessian.copy()
        covariance = np.linalg.inv(hessian)

    hessian = nearest_positive_semidefinite(hessian)
    covariance = nearest_positive_semidefinite(covariance)
    correlation = covariance_to_correlation(covariance)
    sign, log_determinant = np.linalg.slogdet(hessian)
    if sign <= 0 or not np.isfinite(log_determinant):
        raise FloatingPointError("The selected Hessian has no finite positive determinant.")
    dimension = optimization.argument_minimum.size
    posterior_predictive_correction = -0.5 * float(log_determinant)
    free_parameter_penalty = dimension / 2.0 * np.log(2.0 * np.pi)
    lme = (
        -optimization.value_minimum
        + posterior_predictive_correction
        - free_parameter_penalty
    )
    return LaplaceResult(
        numerical_hessian=numerical_hessian,
        numerical_error=numerical_error,
        hessian=hessian,
        covariance=covariance,
        correlation=correlation,
        lme=float(lme),
        decomposition={
            "logjoint": -optimization.value_minimum,
            "postpredcorr": posterior_predictive_correction,
            "freepars": float(free_parameter_penalty),
        },
        used_bfgs_fallback=fallback,
        fallback_reason=fallback_reason,
        numerical_minimum_eigenvalue=numerical_minimum,
        numerical_condition_number=numerical_condition,
    )


def nearest_positive_semidefinite(matrix: FloatArray) -> FloatArray:
    """Port of ``tapas_nearest_psd``."""

    candidate = np.asarray(matrix, dtype=float)
    if candidate.ndim != 2 or candidate.shape[0] != candidate.shape[1]:
        raise ValueError("PSD projection requires a square matrix.")
    candidate = (candidate.T + candidate) / 2.0
    for _ in range(100):
        eigenvalues, eigenvectors = np.linalg.eigh(candidate)
        if np.all(eigenvalues >= 0):
            return candidate
        candidate = eigenvectors @ np.diag(np.maximum(0.0, eigenvalues)) @ eigenvectors.T
        candidate = (candidate.T + candidate) / 2.0
    raise FloatingPointError("PSD projection did not converge.")


def covariance_to_correlation(covariance: FloatArray) -> FloatArray:
    """Port of ``tapas_Cov2Corr``."""

    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    if not np.array_equal(matrix.T, matrix):
        raise ValueError("Covariance matrix must be exactly symmetric.")
    if np.any(~np.isfinite(matrix)) or np.any(np.linalg.eigvalsh(matrix) < 0):
        raise ValueError("Covariance matrix must be finite and positive semidefinite.")
    standard_deviation = np.sqrt(np.diag(matrix))
    return matrix / np.outer(standard_deviation, standard_deviation)


def _ridders_extrapolation(
    approximation: Callable[[float], float], options: RiddersOptions
) -> Tuple[float, float]:
    if options.init_h <= 0 or options.div <= 1:
        raise ValueError("Ridders init_h must be positive and div must exceed one.")
    if options.min_steps < 1 or options.max_steps < 2:
        raise ValueError("Ridders step counts are invalid.")
    polynomial = np.full((options.max_steps, options.max_steps), np.nan)
    derivative = np.nan
    error = float(np.finfo(float).max)
    step = options.init_h
    polynomial[0, 0] = approximation(step)
    division_squared = options.div**2

    for row in range(1, options.max_steps):
        step /= options.div
        polynomial[row, 0] = approximation(step)
        factor = division_squared
        for column in range(1, row + 1):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                polynomial[row, column] = (
                    factor * polynomial[row, column - 1]
                    - polynomial[row - 1, column - 1]
                ) / (factor - 1.0)
            factor *= division_squared
            with np.errstate(over="ignore", invalid="ignore"):
                current_error = max(
                    abs(polynomial[row, column] - polynomial[row, column - 1]),
                    abs(polynomial[row, column] - polynomial[row - 1, column - 1]),
                )
            if current_error < error:
                error = current_error
                derivative = polynomial[row, column]
        if (
            row + 1 > options.min_steps
            and abs(polynomial[row, row] - polynomial[row - 1, row - 1])
            > options.termination_factor * error
        ):
            return float(derivative), float(error)
    return float(derivative), float(error)


def _validated_point(function: ScalarFunction, point: FloatArray) -> FloatArray:
    x = np.asarray(point, dtype=float)
    if x.ndim != 1 or x.size == 0 or np.any(~np.isfinite(x)):
        raise ValueError("Differentiation point must be a finite non-empty vector.")
    value = float(function(x))
    if not np.isscalar(value):
        raise ValueError("Differentiated function must return a scalar.")
    return x


def _append_trace(
    enabled: bool,
    arguments: List[FloatArray],
    values: List[float],
    inverse_hessians: List[FloatArray],
    point: FloatArray,
    value: float,
    inverse_hessian: FloatArray,
) -> None:
    if enabled:
        arguments.append(point.copy())
        values.append(float(value))
        inverse_hessians.append(inverse_hessian.copy())
