function manifest = pam_dot_task_export_fixtures(output_directory, selection)
%PAM_DOT_TASK_EXPORT_FIXTURES Write MATLAB reference fixtures for the Python port.
%
%   MANIFEST = PAM_DOT_TASK_EXPORT_FIXTURES(OUTPUT_DIRECTORY) writes every
%   fixture the Python parity gates in PARITY.md still need, then returns a
%   manifest describing what was written.
%
%   PAM_DOT_TASK_EXPORT_FIXTURES(OUTPUT_DIRECTORY, SELECTION) writes only the
%   named fixtures. SELECTION is a cell array drawn from:
%
%       'environment'  resolved paths and versions (always written)
%       'hgf'          eHGF and two-cue eHGF trajectories
%       'wfpt'         Gondan-series first-passage densities on a grid
%       'ddm'          official and coherence trial-wise DDM log likelihood
%       'joint'        tapas_fitModel MAP, Hessian, LME, AIC, BIC
%       'bms'          spm_BMS_gibbs and spm_BMS summaries
%       'simulation'   official DDM simulation kernel
%       'ppc'          aggregate and sequential PPC summaries from one batch
%
%   The script needs no subject data. Its 380-trial design is regenerated from
%   an integer Lehmer generator defined in PAM_DOT_TASK_FIXTURE_DESIGN, so the
%   Python side reconstructs bit-identical inputs without the private CSVs.
%   That keeps the fixtures runnable on MATLAB Online Basic, where uploading
%   participant data would be inappropriate.
%
%   Each fixture is written to its own JSON file as soon as it is produced, so
%   an interrupted session keeps whatever already completed. 'joint' is by far
%   the slowest; run it last or on its own.
%
%   Nothing in PAM_master, TAPAS, or SPM12 is modified or written to.

if nargin < 1 || isempty(output_directory)
    output_directory = fullfile(pwd, 'fixtures');
end
all_fixtures = {'hgf', 'wfpt', 'ddm', 'joint', 'bms', 'simulation', 'ppc'};
if nargin < 2 || isempty(selection)
    selection = all_fixtures;
end
if ischar(selection)
    selection = {selection};
end
unknown = setdiff(selection, all_fixtures);
if ~isempty(unknown)
    error('pam:fixture:selection', 'Unknown fixture: %s.', strjoin(unknown, ', '));
end
if ~exist(output_directory, 'dir')
    mkdir(output_directory);
end

manifest = struct;
manifest.generated_by = 'pam_dot_task_export_fixtures';
manifest.fixture_format_version = '1.0.0';
manifest.selection = selection;

environment = collect_environment();
write_fixture(output_directory, 'environment', environment);
manifest.environment = environment;
fprintf('[fixture] environment written\n');

design = pam_dot_task_fixture_design();
write_fixture(output_directory, 'design', design);
fprintf('[fixture] design written (%d trials)\n', size(design.u, 1));

for k = 1:numel(selection)
    name = selection{k};
    fprintf('[fixture] building %s ...\n', name);
    start_time = tic;
    switch name
        case 'hgf'
            payload = build_hgf_fixture(design);
        case 'wfpt'
            payload = build_wfpt_fixture();
        case 'ddm'
            payload = build_ddm_fixture(design);
        case 'joint'
            payload = build_joint_fixture(design);
        case 'bms'
            payload = build_bms_fixture();
        case 'simulation'
            payload = build_simulation_fixture(design);
        case 'ppc'
            payload = build_ppc_fixture(design);
    end
    payload.seconds = toc(start_time);
    write_fixture(output_directory, name, payload);
    fprintf('[fixture] %s written (%.1f s)\n', name, payload.seconds);
end
fprintf('[fixture] all requested fixtures complete in %s\n', output_directory);
end


function environment = collect_environment()
environment = struct;
environment.matlab_version = version;
environment.architecture = computer;
if exist('OCTAVE_VERSION', 'builtin')
    environment.interpreter = 'octave';
else
    environment.interpreter = 'matlab';
end
names = {'tapas_fitModel', 'tapas_ehgf_binary', 'tapas_ehgf_binary_transp', ...
    'tapas_quasinewton_optim', 'tapas_sgm', 'spm_BMS_gibbs', 'spm_BMS', ...
    'ddm_hgf', 'ddm_hgf_transp', 'utl_wfpt', 'cue_ehgf_binary', ...
    'ddm_hgf_coherence'};
resolved = cell(numel(names), 1);
for k = 1:numel(names)
    found = which(names{k});
    if isempty(found)
        resolved{k} = '';
    else
        resolved{k} = found;
    end
end
environment.function_names = names;
environment.resolved_paths = resolved;
end


function payload = build_hgf_fixture(design)
require({'tapas_ehgf_binary', 'cue_ehgf_binary', 'cue_ehgf_binary_config'});
config = cue_ehgf_binary_config;
payload = struct;
payload.prior_mus = config.priormus(:)';
payload.prior_sas = config.priorsas(:)';

omega_values = [-4, -3, -2];
payload.omega_values = omega_values;
omega_index = find_omega_index(config);
payload.omega_index = omega_index;

cases = cell(numel(omega_values), 1);
for k = 1:numel(omega_values)
    ptrans = config.priormus;
    ptrans(omega_index) = omega_values(k);
    r = struct;
    r.c_prc = config;
    r.u = design.u(:, 1:2);
    r.y = NaN(size(r.u, 1), 2);
    r.ign = [];
    r.irr = [];
    [traj, states] = cue_ehgf_binary(r, ptrans, 'trans');
    entry = struct;
    entry.omega_2 = omega_values(k);
    entry.ptrans = ptrans(:)';
    entry.muhat = traj.muhat;
    entry.sahat = traj.sahat;
    entry.mu = traj.mu;
    entry.sa = traj.sa;
    entry.states = reshape(states, size(states, 1), []);
    entry.states_size = size(states);
    entry.white_index = traj.cue.white.index(:)';
    entry.red_index = traj.cue.red.index(:)';
    cases{k} = entry;
end
payload.cue_cases = cases;

% Single-stream reference so the Python binary_hgf port is checked separately
% from the two-cue wrapper.
single = cell(numel(omega_values), 1);
for k = 1:numel(omega_values)
    ptrans = config.priormus;
    ptrans(omega_index) = omega_values(k);
    r = struct;
    r.c_prc = config;
    r.u = design.u(:, 1);
    r.y = NaN(size(r.u, 1), 2);
    r.ign = [];
    r.irr = [];
    traj = tapas_ehgf_binary(r, ptrans, 'trans');
    entry = struct;
    entry.omega_2 = omega_values(k);
    entry.muhat = traj.muhat;
    entry.sahat = traj.sahat;
    entry.mu = traj.mu;
    entry.sa = traj.sa;
    single{k} = entry;
end
payload.single_stream_cases = single;
end


function payload = build_wfpt_fixture()
require({'utl_wfpt'});
t_values = [0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00, 2.75];
v_values = [-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0];
a_values = [0.5, 1.0, 1.8, 3.0];
w_values = [0.2, 0.35, 0.5, 0.65, 0.8];
precisions = [1e-4, 1e-8];

rows = numel(t_values) * numel(v_values) * numel(a_values) * numel(w_values) ...
    * numel(precisions);
table = zeros(rows, 6);
index = 0;
for pi = 1:numel(precisions)
    for ti = 1:numel(t_values)
        for vi = 1:numel(v_values)
            for ai = 1:numel(a_values)
                for wi = 1:numel(w_values)
                    index = index + 1;
                    density = utl_wfpt(t_values(ti), v_values(vi), ...
                        a_values(ai), w_values(wi), precisions(pi));
                    table(index, :) = [t_values(ti), v_values(vi), ...
                        a_values(ai), w_values(wi), precisions(pi), density];
                end
            end
        end
    end
end
payload = struct;
payload.columns = {'t', 'v', 'a', 'w', 'prec', 'density'};
payload.grid = table;
end


function payload = build_ddm_fixture(design)
require({'ddm_hgf', 'ddm_hgf_transp', 'cue_ehgf_binary', 'cue_ehgf_binary_config'});
config = cue_ehgf_binary_config;
omega_index = find_omega_index(config);
ptrans_prc = config.priormus;
ptrans_prc(omega_index) = -3;

r = struct;
r.c_prc = config;
r.u = design.u;
r.y = design.y;
r.ign = [];
r.irr = find(~all(isfinite(design.y), 2))';
[traj, states] = cue_ehgf_binary(r, ptrans_prc, 'trans');

payload = struct;
payload.irregular_index = r.irr;
payload.muhat_active = traj.muhat(:, 1);

% Response parameter vectors span the official reduction and the Gate-B
% extension. b_c is last in the coherence transformation.
official_sets = [
    log(1.5), log(1.2),  0.0,  0.0,  0.0, 1.9;
    log(1.5), log(1.2),  0.8,  0.0,  0.0, 1.9;
    log(1.5), log(1.2),  0.0,  0.5,  0.0, 1.9;
    log(1.5), log(1.2),  0.0,  0.0,  1.2, 1.9;
    log(2.0), log(0.8), -0.6, -0.4,  1.5, 2.2];

official = cell(size(official_sets, 1), 1);
for k = 1:size(official_sets, 1)
    r_obs = r;
    r_obs.c_obs = ddm_hgf_config_for_fixture();
    logp = ddm_hgf(r_obs, states, official_sets(k, :)');
    entry = struct;
    entry.ptrans_obs = official_sets(k, :);
    entry.logp = logp(:)';
    entry.sum_logp = sum(logp(isfinite(logp)));
    official{k} = entry;
end
payload.official_cases = official;

if ~isempty(which('ddm_hgf_coherence'))
    coherence_sets = [official_sets(:, 1:5), zeros(size(official_sets, 1), 1), ...
        official_sets(:, 6)];
    coherence_sets(end + 1, :) = [log(1.5), log(1.2), 0.0, 0.0, 1.2, 0.9, 1.9];
    coherence = cell(size(coherence_sets, 1), 1);
    for k = 1:size(coherence_sets, 1)
        r_obs = r;
        r_obs.c_obs = ddm_hgf_coherence_config_for_fixture();
        logp = ddm_hgf_coherence(r_obs, states, coherence_sets(k, :)');
        entry = struct;
        entry.ptrans_obs = coherence_sets(k, :);
        entry.logp = logp(:)';
        entry.sum_logp = sum(logp(isfinite(logp)));
        entry.b_c = coherence_sets(k, 6);
        coherence{k} = entry;
    end
    payload.coherence_cases = coherence;
    payload.nesting_note = ['Rows with b_c = 0 must equal the matching ' ...
        'official row trial by trial.'];
else
    payload.coherence_cases = {};
    payload.nesting_note = 'ddm_hgf_coherence not on path; nesting not exported.';
end
end


function payload = build_joint_fixture(design)
require({'tapas_fitModel', 'cue_ehgf_binary_config', 'ddm_hgf'});
payload = struct;
payload.note = ['tapas_fitModel joint MAP on the deterministic fixture ' ...
    'design. This is the slowest fixture.'];
set_fixed_seed(20260721);
model = tapas_fitModel(design.y, design.u, ...
    'cue_ehgf_binary_config', ...
    'pam_dot_task_fixture_ddm_full_config', ...
    'tapas_quasinewton_optim_config');
payload.p_prc_ptrans = model.p_prc.ptrans(:)';
payload.p_obs_ptrans = model.p_obs.ptrans(:)';
payload.negLl = model.optim.negLl;
payload.negLj = model.optim.negLj;
payload.LME = model.optim.LME;
payload.AIC = model.optim.AIC;
payload.BIC = model.optim.BIC;
payload.H = model.optim.H;
payload.Sigma = model.optim.Sigma;
payload.Corr = model.optim.Corr;
if isfield(model.optim, 'trialLogLlsplit')
    payload.trial_log_likelihood = model.optim.trialLogLlsplit(:)';
end
payload.muhat_active = model.traj.muhat(:, 1)';
end


function payload = build_bms_fixture()
require({'spm_BMS_gibbs', 'spm_BMS'});
% A deterministic LME matrix with one clearly favoured model, one close
% competitor, and one poor model. Fixed here so the Python comparison never
% depends on a fitted result.
subjects = 20;
models = 3;
lme = zeros(subjects, models);
state = 20260721;
for i = 1:subjects
    for j = 1:models
        [state, value] = lehmer_next(state);
        lme(i, j) = -100 + 5 * (j == 1) + 4.5 * (j == 2) + 2 * value;
    end
end
payload = struct;
payload.lme = lme;
payload.alpha0 = ones(1, models);
payload.Nsamp = 100000;

set_fixed_seed(20260721);
[exp_r, xp, r_samp, g_post] = spm_BMS_gibbs(lme, payload.alpha0, payload.Nsamp);
payload.gibbs_exp_r = exp_r(:)';
payload.gibbs_xp = xp(:)';
payload.gibbs_r_samp_mean = mean(r_samp, 1);
payload.gibbs_r_samp_std = std(r_samp, 0, 1);
payload.gibbs_g_post_mean = mean(g_post, 1);
payload.gibbs_seed = 20260721;
payload.gibbs_note = ['MT19937 draw-for-draw agreement is not expected; ' ...
    'compare Monte Carlo summaries within a declared tolerance.'];

set_fixed_seed(20260721);
[alpha, exp_r2, xp2, pxp, bor] = spm_BMS(lme);
payload.bms_alpha = alpha(:)';
payload.bms_exp_r = exp_r2(:)';
payload.bms_xp = xp2(:)';
payload.bms_pxp = pxp(:)';
payload.bms_bor = bor;
end


function payload = build_simulation_fixture(design)
require({'pam_dot_task_simulate_official_ddm'});
payload = struct;
payload.note = ['Simulation and sequential PPC reference. RNG streams will ' ...
    'not match NumPy; compare distributional summaries, not draws.'];
set_fixed_seed(20260721);
payload.replicates = 200;
payload.decision_time_step = 0.001;
payload.response_deadline = 3.0;
w = 0.5 + 0.1 * (design.muhat_reference - 0.5);
a = 1.5 * ones(size(w));
v = (2 * design.u(:, 1) - 1) .* (1.0 + 0.5 * (design.muhat_reference - 0.5));
ter = 0.25 * ones(size(w));
payload.trialwise_w = w(:)';
payload.trialwise_a = a(:)';
payload.trialwise_v = v(:)';
payload.trialwise_Ter = ter(:)';
trialwise = table(w, a, v, ter, 'VariableNames', {'w', 'a', 'v', 'Ter'});
simulated = pam_dot_task_simulate_official_ddm(trialwise, ...
    Replicates=payload.replicates, ...
    Seed=20260721, ...
    DecisionTimeStep=payload.decision_time_step, ...
    MaxDecisionTime=payload.response_deadline);
payload.rt_mean = mean(simulated.rt, 1);
payload.rt_median = median(simulated.rt, 1);
payload.choice_rate = mean(simulated.choice, 1);
if isfield(simulated, 'captured_mass')
    payload.captured_mass = simulated.captured_mass(:)';
end
end


function payload = build_ppc_fixture(design)
% Export summary-level reference values only. MATLAB and NumPy do not share
% random-number streams, so raw simulated draws would be a misleading parity
% target. Both PPC views below read exactly the same `simulated` batch.
require({'pam_dot_task_simulate_official_ddm', 'pam_dot_task_sequential_ppc', ...
    'pam_dot_task_ppc_spec', 'pam_dot_task_aggregate_ppc_spec'});
replicates = 200;
decision_time_step = 0.001;
response_deadline = 3.0;
w = 0.5 + 0.1 * (design.muhat_reference - 0.5);
a = 1.5 * ones(size(w));
v = (2 * design.u(:, 1) - 1) .* (1.0 + 0.5 * (design.muhat_reference - 0.5));
ter = 0.25 * ones(size(w));
trialwise = table(w, a, v, ter, ...
    'VariableNames', {'w', 'a', 'v', 'Ter'});
simulated = pam_dot_task_simulate_official_ddm(trialwise, ...
    Replicates=replicates, ...
    Seed=20260721, ...
    DecisionTimeStep=decision_time_step, ...
    MaxDecisionTime=response_deadline);
audit = fixture_audit(design);

payload = struct;
payload.note = ['Aggregate and time-resolved PPC reuse one 200-by-380 ' ...
    'simulation batch. Compare observed values exactly and predictive ' ...
    'summaries distributionally because RNG streams are independent.'];
payload.replicates = replicates;
payload.response_deadline = response_deadline;
payload.decision_time_step = decision_time_step;
payload.sequential = compact_ppc(pam_dot_task_sequential_ppc(audit, simulated));
payload.aggregate = compact_ppc(pam_dot_task_sequential_ppc(audit, simulated, ...
    pam_dot_task_aggregate_ppc_spec(audit)));
end


function audit = fixture_audit(design)
audit = table(design.trial, string(design.phase), design.u(:, 1), ...
    design.u(:, 2), design.u(:, 3), design.ratio_corrected, ...
    design.y(:, 1), design.y(:, 2), 'VariableNames', ...
    {'trial', 'phase', 'stimulus_category', 'cue_white', ...
    'signed_coherence', 'ratio_corrected', 'rt_seconds_raw', 'choice_white'});
end


function output = compact_ppc(ppc)
% Convert the table-containing runtime result into JSON-writer-supported data.
summary = ppc.summary;
n_windows = numel(ppc.spec.windows);
n_statistics = numel(ppc.spec.statistics);
as_window_statistic = @(values) reshape(values, n_statistics, n_windows)';
output = struct;
output.version = ppc.spec.version;
output.window_ids = cellstr(string({ppc.spec.windows.id}));
output.families = cellstr(string({ppc.spec.windows.family}));
output.statistics = cellstr(ppc.spec.statistics);
output.valid_trials = reshape(summary.valid_trials, n_statistics, n_windows)';
output.observed_statistics = as_window_statistic(summary.observed_value);
output.predictive_median = as_window_statistic(summary.predictive_median);
output.predictive_lower = as_window_statistic(summary.predictive_lower);
output.predictive_upper = as_window_statistic(summary.predictive_upper);
output.predictive_percentile = as_window_statistic(summary.predictive_percentile);
output.tail_probability_two_sided = as_window_statistic( ...
    summary.tail_probability_two_sided);
output.observed_z = as_window_statistic(summary.observed_z);
output.outside_simultaneous = as_window_statistic( ...
    summary.outside_simultaneous);
output.global_observed = ppc.global_observed;
output.global_tail_probability = ppc.global_tail_probability;
output.simultaneous_threshold = ppc.simultaneous_threshold;
end


function config = ddm_hgf_config_for_fixture()
if ~isempty(which('pam_dot_task_ddm_config'))
    config = pam_dot_task_ddm_config('ddm_full');
else
    config = ddm_hgf_config();
end
end


function config = ddm_hgf_coherence_config_for_fixture()
if ~isempty(which('pam_dot_task_ddm_coherence_config'))
    config = pam_dot_task_ddm_coherence_config('ddm_full_c');
else
    config = ddm_hgf_coherence_config();
end
end


function index = find_omega_index(config)
% omega_2 is the second entry of the omega block in the eHGF prior vector.
% The block layout is mu_0, logsa_0, rho, logkappa, omega, logtheta.
levels = numel(config.mu_0mu);
index = 3 * levels + (levels - 1) + 2;
end


function require(names)
missing = {};
for k = 1:numel(names)
    if isempty(which(names{k}))
        missing{end + 1} = names{k}; %#ok<AGROW>
    end
end
if ~isempty(missing)
    error('pam:fixture:dependency', ...
        'Not on the MATLAB path: %s. Run setup_pam_dot_task_environment first.', ...
        strjoin(missing, ', '));
end
end


function set_fixed_seed(seed)
if exist('rng', 'builtin') || exist('rng', 'file')
    rng(seed, 'twister');
else
    rand('seed', seed); %#ok<RAND>
    randn('seed', seed); %#ok<RAND>
end
end


function [state, value] = lehmer_next(state)
state = mod(16807 * state, 2147483647);
value = state / 2147483647;
end


function write_fixture(output_directory, name, payload)
pam_dot_task_write_json(fullfile(output_directory, [name '.json']), payload);
end
