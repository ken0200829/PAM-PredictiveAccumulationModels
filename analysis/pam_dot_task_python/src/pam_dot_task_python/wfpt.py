"""PAM Wiener first-passage-time density parity implementation."""

import math
from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatOrArray = Union[float, NDArray[np.float64]]


def wfpt_density(
    time: ArrayLike,
    drift: ArrayLike,
    boundary: ArrayLike,
    start: ArrayLike = 0.5,
    precision: float = 1e-4,
) -> FloatOrArray:
    """Density at the lower barrier using PAM's Gondan-series algorithm.

    This is a direct numerical port of ``utl_wfpt.m``, ``utl_fsw.m``, and
    ``utl_ks.m``. Inputs broadcast according to NumPy rules.
    """

    t, v, a, w = np.broadcast_arrays(
        np.asarray(time, dtype=float),
        np.asarray(drift, dtype=float),
        np.asarray(boundary, dtype=float),
        np.asarray(start, dtype=float),
    )
    if not np.isfinite(precision) or precision <= 0:
        raise ValueError("precision must be finite and positive.")
    if np.any(~np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("Decision time must be finite and positive.")
    if np.any(~np.isfinite(v)):
        raise ValueError("Drift must be finite.")
    if np.any(~np.isfinite(a)) or np.any(a <= 0):
        raise ValueError("Boundary separation must be finite and positive.")
    if np.any(~np.isfinite(w)) or np.any((w <= 0) | (w >= 1)):
        raise ValueError("Relative starting point must lie strictly inside (0,1).")

    result = np.empty(t.shape, dtype=float)
    for index in np.ndindex(t.shape):
        result[index] = _wfpt_scalar(
            float(t[index]),
            float(v[index]),
            float(a[index]),
            float(w[index]),
            precision,
        )
    if result.ndim == 0:
        return float(result)
    return result


def _wfpt_scalar(time: float, drift: float, boundary: float, start: float, precision: float) -> float:
    leading = (
        1.0
        / boundary**2
        * math.exp(-drift * boundary * start - drift**2 * time / 2.0)
    )
    if leading == 0.0:
        scaled_precision = math.inf
    else:
        scaled_precision = precision / leading
    normalized_time = time / boundary**2
    return leading * _small_time_density(normalized_time, start, scaled_precision)


def _small_time_density(time: float, start: float, precision: float) -> float:
    terms = _series_terms(time, start, precision)
    if terms <= 0 or not math.isfinite(terms):
        return 0.0
    series = 0.0
    for k in range(terms, 0, -1):
        positive = start + 2.0 * k
        negative = start - 2.0 * k
        series = (
            positive * math.exp(-(positive**2) / (2.0 * time))
            + negative * math.exp(-(negative**2) / (2.0 * time))
            + series
        )
    return (
        series + start * math.exp(-(start**2) / (2.0 * time))
    ) / math.sqrt(2.0 * math.pi * time**3)


def _series_terms(time: float, start: float, precision: float) -> int:
    first = (math.sqrt(2.0 * time) - start) / 2.0
    second = first
    if precision == math.inf:
        log_requirement = math.inf
    else:
        argument = 2.0 * math.pi * time**2 * precision**2
        log_requirement = -math.inf if argument == 0 else math.log(argument)
    u_epsilon = min(-1.0, log_requirement)
    argument = -time * (u_epsilon - math.sqrt(-2.0 * u_epsilon - 2.0))
    if argument > 0:
        second = 0.5 * math.sqrt(argument) - start / 2.0
    return int(math.ceil(max(first, second)))

