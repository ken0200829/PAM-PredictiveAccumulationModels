# PAM dot-task Python port

This is the no-license analysis path for the 37-subject dot task. It ports the
approved two-cue HGF and coherence-aware PAM DDM into an auditable Python
package while leaving raw CSV files, `PAM_master`, TAPAS, SPM12, and the MATLAB
analysis layer unchanged.

## Implemented now

- Exact filename-based counterbalancing and 380-row response-mask contract.
- Boundary-aligned cue audit columns: `cue_red`, `red_prediction_sign`, and
  `cue_evidence` (`+1` red-predicts-white, `-1` red-predicts-black, `0` white).
- Three-level enhanced binary HGF equations from pinned TAPAS 6.1.0.
- Plan D: two independent cue streams, shared HGF parameters, frozen inactive
  state, and cue-presentation-count time.
- PAM's Gondan-series WFPT density.
- Official PAM DDM reductions and Gate-B coherence reductions.
- Versioned recovery-first cue-locus registry with a cue-blind history model,
  direct parallel `w`/`v0` models, and two-cue integrated `w`/`v0` models.
- Deterministic registry digest covering each model's architecture, free
  effects, parameter names, and candidate priors.
- Outcome-redacted cue-effect calibration on all 37 trial designs, plus a
  conditional prior-predictive audit that never reads observed choice/RT
  values and records design, registry, and manifest digests.
- Exact nesting: `b_c=0` returns the official trial-wise drift and likelihood.
- Gaussian transformed-space priors and the joint negative log posterior.
- TAPAS-style Ridders gradients, reset-capable BFGS, and iteration traces.
- Numerical Hessian diagnostics, BFGS fallback, covariance, correlation,
  Laplace LME, AIC, and BIC.
- Paper-faithful SPM Gibbs random-effects BMS, with protected exceedance BMS as
  a separately labelled sensitivity analysis.
- MAP-fixed and Laplace-draw PAM DDM simulation with the experimental
  three-second response deadline fixed; `captured_mass` records the predicted
  probability of an in-deadline response.
- Explicit diagnostic fallback to MAP when the uncorrected Hessian, covariance,
  condition number, or draw-rejection rate is unsuitable.
- Aggregate and 49-window time-resolved PPC; both views reuse one trial-indexed
  posterior-prediction batch.
- A parameter-recovery runner that preserves the trial/missingness design and
  corrects the true transformed `Ter` for each generated dataset's minimum RT.
- Per-fit `Ter` diagnostics reporting its fraction of the minimum included RT
  and the remaining decision-time slack.
- A separately named SciPy BFGS entry point for diagnostic comparison only.

The remaining parity gates are listed in `PARITY.md`. The TAPAS algorithms are
now ported and pass analytic tests, but must still be compared against MATLAB
reference fixtures before main-analysis estimates are reported.

## Read-only validation on the real dataset

The Python adapter has been run across the 37 top-level subject CSVs without
writing to the raw-data directory. It reproduced the frozen MATLAB audit:

- conditions: normal 10, normal_cb 10, reverse 7, reverse_cb 10;
- 14,060 total trials: 3,700 learning and 10,360 test;
- 10,335 response-likelihood trials and 25 invalid test responses;
- 7,770 white-cue and 6,290 red-cue trials (210/170 per subject).

For the designated smoke subject, the initial official `ddm_v` and extended
`ddm_v_c` models produced exactly equal trial likelihoods when `b_c=0`.
The initial `ddm_full_c` joint objective was finite for all 37 subjects.
A two-iteration, non-reportable `ddm_null` optimizer smoke on the designated
subject reduced the negative joint from 737.9211 to 260.5728 without a reset,
confirming that the ported numerical path is connected to the real objective.

All 37 subjects also produced the predeclared 49 time-resolved PPC windows.
For the designated subject, a non-reportable 100-replicate by 380-trial
simulation smoke was consumed unchanged by both the sequential 49-window and
aggregate 7-window summaries. Its WFPT grid captured 0.999992 probability mass
per trial. The resulting PPC tail values are deliberately not interpreted:
100 MAP-fixed replicates at initial parameters are an implementation smoke,
not a model check.

## Run without installing anything

The current machine already provides the locked NumPy, SciPy, and pandas
versions. Run the standard-library test suite from the repository root:

```bash
PYTHONPATH=analysis/pam_dot_task_python/src \
python3 -m unittest discover -s analysis/pam_dot_task_python/tests -v
```

The current suite contains 171 tests, including analytic quadratic checks for
Ridders gradients, Hessians, BFGS convergence, strict reset behavior, and both
Laplace Hessian-selection branches, plus fixed-seed BMS, Laplace/MAP-fallback
simulation, single-batch PPC, and recovery-path checks.

MATLAB Online fixture comparison verifies the deterministic input design, full
single-stream and two-cue HGF trajectories at three volatility settings, all
2,240 WFPT grid values, and trial-wise DDM likelihoods for five official and
six coherence-extended parameter sets. Every comparison currently agrees to
an absolute tolerance of `1e-12`. SPM protected-BMS variational quantities
also agree to `1e-12`; Gibbs and exceedance summaries agree within their
declared independent-chain Monte Carlo tolerances. The fixed three-second simulation's WFPT captured
mass agrees to `1e-12`; independent random-stream RT/choice summaries are
checked at a declared RMSE tolerance of `0.06`. Aggregate and sequential PPC
window metadata and observed statistics agree exactly; predictive summaries
agree within declared independent-stream RMSE tolerances. An independent
Python MAP recovery is within `0.02` transformed-parameter units and `1e-3`
negative-log-joint units of the MATLAB local solution; the ported Ridders
Hessian gives a positive-definite Laplace approximation within `0.03` LME
units. At the independently exported MATLAB MAP point, the Python joint
objective agrees to `1e-12`, and the MATLAB Hessian reproduces its exported
Laplace LME exactly.

The 24-case recovery grid has also been rerun under the finalized tie rule
(`HGF` holds its belief and the DDM uses `direction = v = 0`). All fits
converged without resets, but the frozen Gate did not pass: `hgf.omega_2`
recovered at `r=0.092` and `ddm.b_v` at `r=0.674`, below the declared `0.70`
criterion. The other three parameters passed. This result is retained as a
failed Gate and is not converted into a pass by relaxing the threshold.

The subsequent pre-frozen V3.1 Gate directly tested the finalized model. The
32-case `ddm_full_c` grid passed all eight parameters, including
`omega_2 (r=0.815)`, `b_w (r=0.902)`, and `b_c (r=0.711)`. The 32-case
`ddm_w` grid recovered `b_w` well (`r=0.950`) but failed `omega_2`
(`r=0.547`). Because the V3.1 rule required both grids to pass, the combined
recovery Gate remains failed and 37-subject fitting remains blocked.

## Minimal use

```python
from pam_dot_task_python import JointModel, load_subject

subject = load_subject("/path/to/normal_dot_task_subject.csv")
model = JointModel(subject.u, subject.y, model_id="ddm_v_c")
initial_value = model.evaluate(model.initial_free_parameters)
print(initial_value.negative_log_joint)

# Primary ported-TAPAS path; expensive because it uses Ridders derivatives.
fit = model.fit_map(compute_lme=True)
print(fit.laplace.lme)
```

The cue-locus models require the separately audited cue direction and retain
the existing three-column `u` contract:

```python
cue_model = JointModel(
    subject.u,
    subject.y,
    model_id="cue_parallel_w_vbias",
    tie=subject.is_tie,
    cue_evidence=subject.cue_evidence,
)
cue_initial = cue_model.evaluate(cue_model.initial_free_parameters)
```

The current `cue-prior-candidate-0.2.0` uses common outcome-redacted choice
probability targets (0.5, 1.5, and 2.5 percentage points) to put the strong
effect at the central-95% edge of each locus prior. Its fixed-seed conditional
prior-predictive audit passed for parallel and integrated models in all four
counterbalanced conditions. It remains explicitly labelled `candidate` and
must be frozen in a recovery manifest only after model recovery and prior-rank
sensitivity are complete; the audit is not itself a Gate pass.

Regenerate the candidate without exposing observed responses to calibration:

```bash
PYTHONPATH=analysis/pam_dot_task_python/src python3 -B \
analysis/pam_dot_task_python/scripts/calibrate_cue_priors.py \
/path/to/dot_task/analysis/real_data \
/tmp/cue_prior_candidate_0.2.0.json
```

One fitted subject can then be simulated once and summarized repeatedly:

```python
from pam_dot_task_python import (
    PosteriorDrawPolicy,
    aggregate_ppc,
    sequential_ppc,
    simulate_posterior_ddm,
)

# These thresholds must be frozen in the run manifest before inspecting PPC.
policy = PosteriorDrawPolicy(
    max_condition_number=1e8,
    max_rejection_rate=0.20,
)
batch = simulate_posterior_ddm(
    model,
    fit,
    replicates=2000,
    seed=20260720,
    policy=policy,
)
time_resolved = sequential_ppc(subject.audit, batch)
aggregate = aggregate_ppc(subject.audit, batch)
```

Group LME comparison keeps the paper-faithful and sensitivity analyses apart:

```python
from pam_dot_task_python import random_effects_bms

bms = random_effects_bms(
    lme_matrix,
    model_ids,
    subject_ids,
    samples=1_000_000,
    seed=20260720,
    run_sensitivity=True,
)
```

`fit_map()` uses the ported TAPAS numerical path. `fit_map_scipy()` remains a
non-parity diagnostic. Laplace posterior draws and auditable MAP fallback are
implemented, but the condition-number and rejection-rate thresholds shown
above are defaults, not an approved Gate-PPC freeze. The three-second grid is
the experimental response deadline; its `captured_mass` is a recorded
in-deadline response probability, while RT/choice PPC is conditional on valid
in-deadline responses. The recovery runner is also implemented, but an
adequate declared recovery grid has not yet passed.
Neither fitting nor PPC output may be used in the main analysis until the
MATLAB comparison, Gate-PPC freeze, and parameter-recovery gates pass.

## License and attribution

This port is licensed GPL-3.0-or-later. It reimplements behavior from the GPL
PAM and TAPAS sources identified in `environment.lock.json`; modified Python
files are clearly separated from the upstream MATLAB code.
