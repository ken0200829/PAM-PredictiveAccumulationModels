# Dot-task PAM integration

This directory contains the external analysis layer for applying the official
PAM HGF+DDM joint-MAP workflow to the 37-subject dot-task dataset. It does not
modify `PAM_master`, TAPAS, or any raw CSV.

## Implemented components

- `pam_dot_task_load_subject`: preserves all 380 stimulus trials and masks only
  response likelihood rows in `y`.
- `pam_dot_task_audit_all`: validates all subjects and optionally writes derived
  audit files outside the raw-data directory.
- `cue_ehgf_binary`: runs two independent standard binary HGF streams inside a
  single perceptual function, with one shared parameter vector and frozen
  non-active cue states.
- `pam_dot_task_ddm_config`: constructs exact `null`, `w`, `a`, `v`, and `full`
  reductions of the official `ddm_hgf` response model.
- `ddm_hgf_coherence` and `pam_dot_task_ddm_coherence_config`: add one nested
  drift-magnitude slope for absolute signed coherence. Setting `b_c=0`
  reproduces the official model.
- `pam_dot_task_fit_subject`: performs one individual joint-MAP fit through
  `tapas_fitModel` and checks trial masks and DDM parameter support.
- `pam_dot_task_simulate_official_ddm`: generates a replicate-by-trial response
  batch using the official WFPT parameterization.
- `pam_dot_task_ppc_spec` and `pam_dot_task_sequential_ppc`: retain trial
  position, reuse one generated batch across all windows, and calculate
  pointwise diagnostics plus a maximum-deviation simultaneous check.
- `pam_dot_task_bms`: uses `spm_BMS_gibbs` as the primary random-effects BMS and
  keeps `spm_BMS` output separate as an optional sensitivity analysis.

The scientific and operational rationale is recorded in
`docs/pam_37_integration_plan.md`.

## MATLAB path setup

Pinned dependency sources are recorded in `environment.lock.json`. Install or
verify them with:

```bash
./analysis/pam_dot_task/setup_environment.sh
```

MATLAB itself requires a user-managed MathWorks installation and license. Once
MATLAB is available, initialize all pinned paths with:

```matlab
project_root = "/path/to/PAM-PredictiveAccumulationModels";
addpath(fullfile(project_root, "analysis", "pam_dot_task", "matlab"));
environment = setup_pam_dot_task_environment(project_root);
```

Record resolved dependencies before fitting:

```matlab
dependency = pam_dot_task_dependency_audit;
disp(dependency.functions)
```

## Data audit

The audit writes nothing by default:

```matlab
data_dir = "/Users/utsumikensuke/Research/dot_task/analysis/real_data";
audit = pam_dot_task_audit_all(data_dir);
```

To write derived audit artifacts, provide a directory that is not the raw-data
directory:

```matlab
audit = pam_dot_task_audit_all(data_dir, ...
    OutputDir=fullfile(project_root, "runs", "input_audit"));
```

## One-subject fit

```matlab
csv_path = fullfile(data_dir, ...
    "normal_cb_dot_task_20260526_013329_6717f0ac88d9f27d9b79af31.csv");
result = pam_dot_task_fit_subject(csv_path, "ddm_w", ...
    PerceptualModel="cue_shared_effective2", ...
    Seed=20260720);
```

Set `PerceptualModel="official_single_hgf"` for the official single-stream
reference. The approved coherence models are `ddm_c`, `ddm_w_c`, `ddm_a_c`,
`ddm_v_c`, and `ddm_full_c`. They use the separately named
`ddm_hgf_coherence`; the official `ddm_hgf.m` remains unchanged.

## Sequential PPC

The current simulator holds MAP parameters fixed and represents response noise.
Laplace parameter draws will be added only after the Hessian diagnostics and
fallback thresholds are frozen at Gate PPC.

```matlab
simulation = pam_dot_task_simulate_official_ddm(result.trialwise, ...
    Replicates=2000, Seed=20260720);
spec = pam_dot_task_ppc_spec(result.subject.audit);
ppc = pam_dot_task_sequential_ppc( ...
    result.subject.audit, simulation, spec);
```

The complete replicate-by-trial arrays are generated once. Changing views or
plotting windows must reuse those arrays instead of simulating new responses.

## Tests

```matlab
results = run_pam_dot_task_tests(project_root);
```

Tests requiring TAPAS are marked incomplete when TAPAS is not on the path. A
working MATLAB installation is required to execute the suite.
