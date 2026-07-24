"""Numerical parity port of the TAPAS 6.1.0 enhanced binary HGF."""

from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HGFParameters:
    """Native-space parameters consumed by ``tapas_ehgf_binary``."""

    mu_0: FloatArray
    sa_0: FloatArray
    rho: FloatArray
    kappa: FloatArray
    omega: FloatArray
    theta: float


@dataclass(frozen=True)
class HGFResult:
    """Trial trajectories with the same field meanings as TAPAS."""

    mu: FloatArray
    sa: FloatArray
    muhat: FloatArray
    sahat: FloatArray
    v: FloatArray
    w: FloatArray
    da: FloatArray
    ud: FloatArray
    psi: FloatArray
    epsi: FloatArray
    wt: FloatArray

    @property
    def inf_states(self) -> FloatArray:
        return np.stack((self.muhat, self.sahat, self.mu, self.sa), axis=2)


@dataclass(frozen=True)
class CueHGFResult:
    """Two independent cue streams plus their active-cue stitched trajectory."""

    active: HGFResult
    white: HGFResult
    red: HGFResult
    white_indices: NDArray[np.int64]
    red_indices: NDArray[np.int64]


def transform_ehgf_binary(transformed: FloatArray) -> HGFParameters:
    """Match ``tapas_ehgf_binary_transp`` for a three-level eHGF."""

    p = np.asarray(transformed, dtype=float)
    if p.shape != (14,):
        raise ValueError("The three-level eHGF requires 14 transformed parameters.")
    return HGFParameters(
        mu_0=p[0:3].copy(),
        sa_0=np.exp(p[3:6]),
        rho=p[6:9].copy(),
        kappa=np.exp(p[9:11]),
        omega=p[11:13].copy(),
        theta=float(np.exp(p[13])),
    )


def binary_hgf(
    inputs: FloatArray,
    parameters: HGFParameters,
    ignored: Optional[NDArray[np.bool_]] = None,
) -> HGFResult:
    """Evaluate the three-level enhanced binary HGF trial by trial.

    The update order, clipping of level-one predictions, precision updates,
    and returned trajectories mirror ``tapas_ehgf_binary.m`` at TAPAS commit
    7155f99137c5e03f93a2a3afa6a8cb54c75dd4c2.
    """

    observed = np.asarray(inputs, dtype=float)
    if observed.ndim != 1 or observed.size == 0:
        raise ValueError("Binary HGF inputs must be a non-empty vector.")
    if np.any(~np.isfinite(observed)) or np.any(~np.isin(observed, (0.0, 1.0))):
        raise ValueError("Binary HGF inputs must be finite and coded 0/1.")
    _validate_parameters(parameters)

    n_trials = observed.size
    if ignored is None:
        ignored_mask = np.zeros(n_trials, dtype=bool)
    else:
        ignored_mask = np.asarray(ignored, dtype=bool)
        if ignored_mask.shape != (n_trials,):
            raise ValueError("Ignored-trial mask must match the input length.")

    levels = 3
    u = np.concatenate(([0.0], observed))
    n = u.size
    time = np.ones(n)

    mu = np.full((n, levels), np.nan)
    pi = np.full((n, levels), np.nan)
    muhat = np.full((n, levels), np.nan)
    pihat = np.full((n, levels), np.nan)
    variance = np.full((n, levels), np.nan)
    weight = np.full((n, levels - 1), np.nan)
    delta = np.full((n, levels), np.nan)

    mu[0, 0] = _sigmoid(parameters.mu_0[0])
    pi[0, 0] = np.inf
    mu[0, 1:] = parameters.mu_0[1:]
    pi[0, 1:] = 1.0 / parameters.sa_0[1:]

    for k in range(1, n):
        if ignored_mask[k - 1]:
            mu[k] = mu[k - 1]
            pi[k] = pi[k - 1]
            muhat[k] = muhat[k - 1]
            pihat[k] = pihat[k - 1]
            variance[k] = variance[k - 1]
            weight[k] = weight[k - 1]
            delta[k] = delta[k - 1]
            continue

        muhat[k, 1] = mu[k - 1, 1] + time[k] * parameters.rho[1]
        muhat[k, 0] = np.clip(
            _sigmoid(parameters.kappa[0] * muhat[k, 1]), 0.001, 0.999
        )
        pihat[k, 0] = 1.0 / (muhat[k, 0] * (1.0 - muhat[k, 0]))
        pi[k, 0] = np.inf
        mu[k, 0] = u[k]
        delta[k, 0] = mu[k, 0] - muhat[k, 0]

        pihat[k, 1] = 1.0 / (
            1.0 / pi[k - 1, 1]
            + np.exp(parameters.kappa[1] * mu[k - 1, 2] + parameters.omega[1])
        )
        pi[k, 1] = pihat[k, 1] + parameters.kappa[0] ** 2 / pihat[k, 0]
        mu[k, 1] = (
            muhat[k, 1]
            + parameters.kappa[0] / pi[k, 1] * delta[k, 0]
        )
        delta[k, 1] = (
            (1.0 / pi[k, 1] + (mu[k, 1] - muhat[k, 1]) ** 2) * pihat[k, 1]
            - 1.0
        )

        muhat[k, 2] = mu[k - 1, 2] + time[k] * parameters.rho[2]
        pihat[k, 2] = 1.0 / (1.0 / pi[k - 1, 2] + time[k] * parameters.theta)
        variance[k, 2] = time[k] * parameters.theta
        variance[k, 1] = time[k] * np.exp(
            parameters.kappa[1] * mu[k - 1, 2] + parameters.omega[1]
        )
        weight[k, 1] = variance[k, 1] * pihat[k, 1]
        mu[k, 2] = (
            muhat[k, 2]
            + 0.5
            / pihat[k, 2]
            * parameters.kappa[1]
            * weight[k, 1]
            * delta[k, 1]
        )

        updated_variance = time[k] * np.exp(
            parameters.kappa[1] * mu[k, 2] + parameters.omega[1]
        )
        lower_prediction_precision = 1.0 / (
            1.0 / pi[k - 1, 1] + updated_variance
        )
        updated_weight = updated_variance * lower_prediction_precision
        ratio = (
            updated_variance - 1.0 / pi[k - 1, 1]
        ) * lower_prediction_precision
        lower_delta = (
            (1.0 / pi[k, 1] + (mu[k, 1] - muhat[k, 1]) ** 2)
            * lower_prediction_precision
            - 1.0
        )
        pi[k, 2] = pihat[k, 2] + max(
            0.0,
            0.5
            * parameters.kappa[1] ** 2
            * updated_weight
            * (updated_weight + ratio * lower_delta),
        )
        delta[k, 2] = (
            (1.0 / pi[k, 2] + (mu[k, 2] - muhat[k, 2]) ** 2) * pihat[k, 2]
            - 1.0
        )

    sigmoid_mu2 = _sigmoid(parameters.kappa[0] * mu[:, 1])
    sigmoid_delta = u - sigmoid_mu2
    with np.errstate(divide="ignore", invalid="ignore"):
        learning_rate = np.diff(sigmoid_mu2) / sigmoid_delta[1:]
    learning_rate[delta[1:, 0] == 0] = 0.0

    mu = mu[1:]
    pi = pi[1:]
    muhat = muhat[1:]
    pihat = pihat[1:]
    variance = variance[1:]
    weight = weight[1:]
    delta = delta[1:]
    sa = 1.0 / pi
    sahat = 1.0 / pihat
    update = mu - muhat

    psi = np.full((n_trials, levels), np.nan)
    psi[:, 1] = 1.0 / pi[:, 1]
    psi[:, 2:] = pihat[:, 1:-1] / pi[:, 2:]
    epsi = np.full((n_trials, levels), np.nan)
    epsi[:, 1:] = psi[:, 1:] * delta[:, :-1]
    total_weight = np.full((n_trials, levels), np.nan)
    total_weight[:, 0] = learning_rate
    total_weight[:, 1] = psi[:, 1]
    total_weight[:, 2:] = (
        0.5 * variance[:, 1:-1] * parameters.kappa[1:] * psi[:, 2:]
    )

    return HGFResult(
        mu=mu,
        sa=sa,
        muhat=muhat,
        sahat=sahat,
        v=variance,
        w=weight,
        da=delta,
        ud=update,
        psi=psi,
        epsi=epsi,
        wt=total_weight,
    )


def _initial_trajectory_row(
    parameters: HGFParameters,
) -> tuple[FloatArray, FloatArray]:
    """Return the pre-observation state, matching ``tapas_ehgf_binary`` priors.

    TAPAS seeds ``mu(1,1) = sgm(mu_0(1))`` and ``pi(1,1) = Inf`` (the first
    level carries no variance without perceptual uncertainty), then copies
    ``mu_0``/``1/sa_0`` into the higher levels.
    """

    mu = np.asarray(parameters.mu_0, dtype=float).copy()
    mu[0] = float(_sigmoid(np.asarray(mu[0])))
    sa = np.asarray(parameters.sa_0, dtype=float).copy()
    sa[0] = 0.0
    return mu, sa


def _run_stream_with_ties(
    stimulus: FloatArray,
    tie: NDArray[np.bool_],
    parameters: HGFParameters,
) -> HGFResult:
    """Filter one cue stream through the HGF, holding belief across ties.

    A tie trial (ratio 0.5, i.e. 100 white and 100 black dots) carries no
    category evidence, so the plan specifies the identity update
    ``mu1 = muhat1`` with ``delta1 = 0`` (section 5.2.1).

    TAPAS's own ignored-trial branch is deliberately NOT used here.  It freezes
    ``muhat(k,:) = muhat(k-1,:)``, which is the *previous* prediction rather
    than the current one, so a tie trial would be scored against a one-step
    stale belief.  Tie trials are the most diagnostic probe of the
    starting-point bias, so a stale belief there defeats their purpose.

    Instead the stream is run on informative trials only.  Because a tie
    produces no update, the belief at a tie equals the prediction the stream
    makes at that point, which is exactly the prediction row of the next
    informative trial.  One padding trial is appended so trailing ties have a
    prediction row too; only its prediction is read and its own update is
    discarded.
    """

    stimulus = np.asarray(stimulus, dtype=float)
    tie = np.asarray(tie, dtype=bool)
    if stimulus.shape != tie.shape:
        raise ValueError("Tie mask must match the stream length.")

    informative = ~tie
    padded = np.concatenate((stimulus[informative], np.zeros(1)))
    rows = binary_hgf(padded, parameters)

    # updates_before[p] = number of informative trials strictly before p.  For
    # an informative trial this is its own row index; for a tie it is the row
    # whose prediction is still current.
    updates_before = np.concatenate(([0], np.cumsum(informative)))[:-1]

    n_trials = stimulus.size
    initial_mu, initial_sa = _initial_trajectory_row(parameters)
    prediction_fields = ("muhat", "sahat")
    posterior_fields = ("mu", "sa")

    values = {}
    for field in fields(HGFResult):
        source = getattr(rows, field.name)
        width = source.shape[1]
        out = source[updates_before].copy()
        if field.name in prediction_fields:
            # A prediction precedes the trial's own observation, so the same
            # row serves informative and tie trials alike.
            pass
        elif field.name in posterior_fields:
            initial = initial_mu if field.name == "mu" else initial_sa
            carried = np.where(
                updates_before[:, None] == 0,
                initial[:width][None, :],
                source[np.maximum(updates_before - 1, 0)],
            )
            out[tie] = carried[tie]
        else:
            # Every remaining field records what an update did.  No update
            # happens on a tie, so these are zero by construction -- including
            # da, which is the delta1 = 0 the plan requires.
            out[tie] = 0.0
        values[field.name] = out
    return HGFResult(**values)


def cue_blind_binary_hgf(
    stimulus: FloatArray,
    parameters: HGFParameters,
    tie: Optional[NDArray[np.bool_]] = None,
) -> HGFResult:
    """Run one global eHGF over stimulus history without conditioning on cue.

    The global trial axis is preserved.  Tie trials receive the current
    pre-observation prediction but do not update the HGF, matching the approved
    tie rule used by the two-stream implementation.
    """

    observed = np.asarray(stimulus, dtype=float)
    if observed.ndim != 1 or observed.size == 0:
        raise ValueError("Cue-blind HGF inputs must be a non-empty vector.")
    if np.any(~np.isfinite(observed)) or np.any(~np.isin(observed, (0.0, 1.0))):
        raise ValueError("Cue-blind HGF inputs must be finite and coded 0/1.")
    if tie is None:
        tie_mask = np.zeros(observed.size, dtype=bool)
    else:
        tie_mask = np.asarray(tie, dtype=bool)
        if tie_mask.shape != observed.shape:
            raise ValueError("Tie mask must match the trial count.")
    return _run_stream_with_ties(observed, tie_mask, parameters)


def cue_binary_hgf(
    u: FloatArray,
    parameters: HGFParameters,
    tie: Optional[NDArray[np.bool_]] = None,
) -> CueHGFResult:
    """Run two shared-parameter HGF streams on cue-presentation time.

    ``tie`` marks trials with no objective category (ratio 0.5).  It is passed
    explicitly rather than derived from a signed-coherence column so this
    function never depends on the ``u`` column ordering.
    """

    inputs = np.asarray(u, dtype=float)
    if inputs.ndim != 2 or inputs.shape[1] < 2:
        raise ValueError("Cue HGF requires stimulus and cue columns.")
    stimulus = inputs[:, 0]
    cue = inputs[:, 1]
    if np.any(~np.isfinite(stimulus)) or np.any(~np.isin(stimulus, (0.0, 1.0))):
        raise ValueError("Stimulus category must be finite and binary.")
    if np.any(~np.isfinite(cue)) or np.any(~np.isin(cue, (0.0, 1.0))):
        raise ValueError("Cue indicator must be finite and binary.")
    if tie is None:
        tie_mask = np.zeros(inputs.shape[0], dtype=bool)
    else:
        tie_mask = np.asarray(tie, dtype=bool)
        if tie_mask.shape != (inputs.shape[0],):
            raise ValueError("Tie mask must match the trial count.")

    white_indices = np.flatnonzero(cue == 1.0)
    red_indices = np.flatnonzero(cue == 0.0)
    if white_indices.size == 0 or red_indices.size == 0:
        raise ValueError("Both white and red cue streams must contain trials.")
    white = _run_stream_with_ties(
        stimulus[white_indices], tie_mask[white_indices], parameters
    )
    red = _run_stream_with_ties(
        stimulus[red_indices], tie_mask[red_indices], parameters
    )
    active = _stitch_active(inputs.shape[0], white_indices, red_indices, white, red)
    if np.any(~np.isfinite(active.muhat[:, 0])):
        raise RuntimeError("Every trial must receive a finite active-cue prediction.")
    return CueHGFResult(
        active=active,
        white=white,
        red=red,
        white_indices=white_indices,
        red_indices=red_indices,
    )


def _stitch_active(
    n_trials: int,
    white_indices: NDArray[np.int64],
    red_indices: NDArray[np.int64],
    white: HGFResult,
    red: HGFResult,
) -> HGFResult:
    values = {}
    for field in fields(HGFResult):
        white_values = getattr(white, field.name)
        red_values = getattr(red, field.name)
        active = np.full((n_trials, white_values.shape[1]), np.nan)
        active[white_indices] = white_values
        active[red_indices] = red_values
        values[field.name] = active
    return HGFResult(**values)


def _validate_parameters(parameters: HGFParameters) -> None:
    if parameters.mu_0.shape != (3,) or parameters.sa_0.shape != (3,):
        raise ValueError("mu_0 and sa_0 must contain three levels.")
    if parameters.rho.shape != (3,) or parameters.kappa.shape != (2,):
        raise ValueError("rho and kappa have invalid dimensions.")
    if parameters.omega.shape != (2,):
        raise ValueError("omega must contain the two defined volatility couplings.")
    if np.any(~np.isfinite(parameters.mu_0[1:])):
        raise ValueError("Defined initial means must be finite.")
    if np.any(~np.isfinite(parameters.sa_0[1:])) or np.any(parameters.sa_0[1:] <= 0):
        raise ValueError("Defined initial variances must be finite and positive.")
    if not np.isfinite(parameters.theta) or parameters.theta <= 0:
        raise ValueError("Theta must be finite and positive.")


def _sigmoid(value: FloatArray) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore"):
        return 1.0 / (1.0 + np.exp(-value))
