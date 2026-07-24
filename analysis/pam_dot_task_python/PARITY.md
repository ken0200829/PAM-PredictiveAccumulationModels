# Python parity gates

This directory is an independent GPL-3.0-or-later Python reimplementation,
not an unqualified execution of the official PAM or TAPAS MATLAB toolboxes.
The main 37-subject analysis must not use Python fit results until the
applicable gates below pass.

## Source mapping

| Python implementation | Frozen source behavior |
|---|---|
| `data.py` | `pam_dot_task_load_subject.m`, `pam_dot_task_condition.m` |
| `hgf.py::transform_ehgf_binary` | TAPAS `tapas_ehgf_binary_transp.m` |
| `hgf.py::binary_hgf` | TAPAS `tapas_ehgf_binary.m` |
| `hgf.py::cue_binary_hgf` | external MATLAB `cue_ehgf_binary.m` |
| `wfpt.py` | PAM `utl_wfpt.m`, `utl_fsw.m`, `utl_ks.m` |
| `response.py` | PAM `ddm_hgf.m` and approved `ddm_hgf_coherence.m` |
| `objective.py` | TAPAS `tapas_fitModel.m::negLogJoint` |
| `numerics.py::ridders_*` | TAPAS Ridders gradient/Hessian functions |
| `numerics.py::tapas_quasi_newton` | `tapas_quasinewton_optim.m` and its frozen config |
| `numerics.py::laplace_evidence` | `tapas_fitModel.m::optimrun` Hessian/LME branch |
| `bms.py::gibbs_bms` | SPM12 `spm_BMS_gibbs.m` |
| `bms.py::protected_bms` | SPM12 `spm_BMS.m`, `spm_BMS_bor.m`, `spm_dirichlet_exceedance.m` |
| `ppc.py::simulate_ddm` | external MATLAB `pam_dot_task_simulate_official_ddm.m` |
| `ppc.py::simulate_posterior_ddm` | integration-plan §12.3 Laplace-draw and MAP-fallback contract |
| `ppc.py::make_sequential_spec` | external MATLAB `pam_dot_task_ppc_spec.m` |
| `ppc.py::sequential_ppc` | external MATLAB `pam_dot_task_sequential_ppc.m` |
| `recovery.py` | integration-plan §12 parameter-recovery contract |
| `gates.py` | integration-plan Gate-PPC and §12.2 freeze contracts |
| `fixtures.py` | external MATLAB `pam_dot_task_fixture_design.m` |
| `bayes_optimal.py` | TAPAS `tapas_bayes_optimal_binary.m`, PAM demo lines 52-54 |

Exact source commits are recorded in `environment.lock.json`.

## Gates

- [x] Preserve 380 input trials and mask response likelihood rows only.
- [x] Keep cue streams independent with shared parameters and cue-count time.
- [x] Reproduce the official response formula exactly when `b_c=0`.
- [x] Check the WFPT probability-mass identity independently by integration.
- [x] Evaluate a finite joint negative log posterior on a synthetic problem.
- [x] Port Ridders gradients/Hessian and verify them on analytic quadratics.
- [x] Port BFGS regularization/reset behavior and verify its strict reset path.
- [x] Port numerical-Hessian and BFGS-fallback Laplace LME branches.
- [x] Compare complete single-stream and two-cue HGF trajectories against MATLAB Online fixtures.
- [x] Compare 2,240 WFPT values against PAM MATLAB output.
- [x] Compare trial DDM log likelihood against MATLAB Online fixture (5 official, 6 coherence cases).
- [x] Compare negative log joint at the MATLAB Online MAP point against MATLAB.
- [x] Recover the MATLAB local MAP independently, then compare the Ridders-Hessian Laplace LME within its declared numerical tolerance.
- [x] Port SPM Gibbs BMS and optional protected exceedance BMS with fixed-seed tests.
- [x] Port aggregate and time-resolved PPC without resimulating per window.
- [x] Build all 49 predeclared sequential windows for every one of the 37 subjects.
- [x] Compare Gibbs/protected BMS summaries against SPM MATLAB Online fixture.
- [x] Compare fixed-deadline DDM simulation captured mass and distributional summaries against MATLAB Online fixture.
- [x] Compare aggregate/sequential PPC summaries against MATLAB fixture.
- [x] Add Laplace posterior draws, draw rejection, diagnostics, and MAP fallback.
- [x] Add a trial-design-preserving parameter-recovery generation/refit runner.
- [x] Fix the PPC grid at the experimental three-second response deadline.
- [x] Freeze the Gate-PPC condition-number and rejection-rate thresholds.
- [x] Declare the recovery grid, its seeds, and the recovery pass criteria.
- [ ] Pass parameter recovery before any 37-subject fit.
- [x] Export MATLAB fixtures for joint MAP and fixed-deadline simulation.

## Frozen declarations

`gates.py` holds the pre-outcome decisions as versioned, hashable data, and
`manifests/gate_ppc_freeze_v1.json` is the artifact a run manifest must cite:

| Declaration | Digest (SHA-256, first 16) |
|---|---|
| `GATE_PPC_V1` | `a81d08fc024e7afd` |
| `RECOVERY_GRID_W_V3` (active 3.1) | `ad0555f74e460241` |
| `RECOVERY_GRID_FULL_C_V3` (active 3.1) | `abe447684a5c8b65` |
| `RECOVERY_GRID_V2` (failed/superseded) | `745fca209349836f` |
| `RECOVERY_GRID_V1` (superseded) | `560a4bd6fa2ff8bd` |
| `RECOVERY_CRITERIA_V1` | `f9336c2218a5c6a1` |

`RECOVERY_GRID_V2` replaces V1 after the V1 run showed V1's `omega_2` truths
sat above the real operating range. V2 spans `[-5.6, -3.1]` at six levels
(matching the observed Bayes-optimal range), caps `Ter_logit` at 2.0 to avoid
the six non-finite failures V1 produced at 2.5, and rebalances the nuisance
columns so every pairwise truth correlation is below 0.09. Recovery now refits
under the subject's Bayes-optimal `omega_2` prior mean, not the default -3.

After formulation A was finalized, V3 was frozen before running any V3 case.
V3.0's first `ddm_w` generation produced a response outside the finite
likelihood support at the declared prior-mean start, before any recovery
estimate existed. V3.1 therefore narrows only `log_a_a` from `{0.4,0.8}` to
`{0.2,0.6}`; all 64 seed/design combinations were then preflighted for a
finite initial objective before V3.1 was frozen. No recovery estimate or Gate
statistic was inspected in that feasibility step.
The Gate now requires both a 32-case `ddm_w` grid, which directly tests the
tie-driven starting-point hypothesis, and a 32-case `ddm_full_c` grid, which
tests the largest retained response model. Truth columns use fixed order-32
Walsh contrasts and are exactly uncorrelated. Omega has four balanced levels
from `-5.5` to `-3.1`; all other parameters have balanced two-level contrasts.
Seeds, criteria, versions, and full-grid digests are recorded in
`manifests/recovery_gate_v3_freeze.json`. Both grids must pass every unchanged
`RECOVERY_CRITERIA_V1` parameter criterion.

V3.1 results are saved in `recovery_runs/recovery_ddm_w_v3_1_tie_v0.json`
and `recovery_runs/recovery_ddm_full_c_v3_1_tie_v0.json`. All 64 fits
completed. The full coherence grid passed 8/8 parameters (`omega_2 r=0.815`,
`b_w r=0.902`, `b_a r=0.883`, `b_v r=0.781`, `b_c r=0.711`). The reduced
starting-point grid passed 4/5: `b_w` recovered at `r=0.950`, but `omega_2`
recovered at `r=0.547`, below `0.70`. Therefore the pre-frozen combined Gate
failed. The result is not reclassified based on the successful full model.
Across V3.1, fitted `Ter/min(RT_valid)` ranged from `0.820` to `0.985`, with
minimum decision-time slack `0.0058` seconds; this boundary proximity remains
a diagnostic for the later small-subject validation.

Both threshold rationales are in the `GatePPCFreeze` docstring. In short,
`max_condition_number = 1e8` bounds round-off amplification during covariance
inversion to about `1e-8` relative and nothing more; it does **not** bound the
Ridders finite-difference error, which is the larger term and is recorded per
fit as a separate diagnostic. `max_rejection_rate = 0.20` caps the truncated
Gaussian's density distortion at 25% and is a declared convention rather than
a derivation.

One structural fact constrains any recovery grid: at the prior mean every
belief-to-DDM slope is zero, so the response likelihood is exactly independent
of the HGF and the gradient in the `hgf.omega_2` direction vanishes. Measured
on the designated subject, `negLl` was identical to 10 decimal places across
`omega_2` in `[-6, 0]` while `b_v = 0`, and varied strongly once `b_v` moved
off zero. Every declared generating set therefore uses a non-zero slope.

The finalized tie formulation was evaluated on all 24 `RECOVERY_GRID_V2`
cases and saved in `recovery_runs/recovery_ddm_v_v2_tie_v0.json`. Every fit
converged on the gradient tolerance without a reset. The frozen Gate failed:
`hgf.omega_2` had correlation `0.092` and `ddm.b_v` had correlation `0.674`,
both below `0.70`; `log_a_a`, `log_a_v`, and `Ter_logit` passed all criteria.
The fitted `Ter/min(RT_valid)` range was `0.776–0.927`, leaving `0.033–0.112`
seconds of decision time at the fastest fitted response. Thresholds remain
unchanged and 37-subject fitting remains blocked by the failed recovery Gate.

## Per-subject Bayes-optimal prior means

`bayes_optimal.py` ports the step the official demo performs before the joint
fit: the perceptual model is fitted to the input sequence alone and the result
becomes the prior mean. Only `u` is read, so no response data enters the prior.
Results for all 37 subjects are in `priors/bayes_optimal_omega.json`.

Two findings from that table bear on frozen decisions and are not yet resolved:

- The hardcoded default of `omega_2 = -3` lies outside the observed range for
  every subject. Bayes-optimal values span `[-5.46, -3.15]` with mean `-4.44`
  and SD `0.57`, so the default was offset by about one prior SD in the same
  direction for all 37.
- Fitting each cue stream separately gives white minus red of `-2.41` on
  average, negative for 37 of 37 subjects, range `[-3.79, -1.26]`. The Gate-A
  shared-parameter assumption is therefore forcing a systematic compromise,
  not absorbing incidental noise.

Neither the joint-fit prior nor the Gate-A decision has been changed on the
strength of this; `cue_hgf_prior()` still returns the frozen `-3` mean and
variance 2.

## MATLAB fixture export

`analysis/pam_dot_task/matlab/pam_dot_task_export_fixtures.m` writes every
fixture the six comparison gates need. It needs no participant data: its
380-trial design comes from a Lehmer recurrence reproduced bit-for-bit by
`fixtures.py::fixture_design`, so it runs on a base MATLAB installation with
no toolboxes, including MATLAB Online Basic. Call
`fixtures.py::assert_design_matches` on the exported `design.json` before
trusting any other comparison.

The ported TAPAS path is `JointModel.fit_map()`. SciPy BFGS is retained only as
`fit_map_scipy()` for comparison. Neither path is yet certified for reported
37-subject results because MATLAB numerical parity and parameter recovery are
still open. Random streams use NumPy's MT19937; its gamma sampler is not assumed
to reproduce MATLAB draw-for-draw, so BMS parity is assessed on Monte Carlo
summaries within a prespecified tolerance rather than exact samples.
