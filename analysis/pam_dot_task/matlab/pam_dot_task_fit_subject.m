function result = pam_dot_task_fit_subject(csv_path, model_id, options)
%PAM_DOT_TASK_FIT_SUBJECT Run one individual joint HGF+DDM MAP fit.
%
% This is a single-start execution unit. Call it separately with distinct
% Seed values for external multi-start checks; do not set nRandInit > 1.

arguments
    csv_path {mustBeTextScalar}
    model_id {mustBeTextScalar}
    options.PerceptualModel (1,1) string {mustBeMember(options.PerceptualModel, ...
        ["cue_shared_effective2", "official_single_hgf"])} = ...
        "cue_shared_effective2"
    options.Seed (1,1) double {mustBeInteger,mustBeNonnegative} = 20260720
    options.RandomInitialization (1,1) logical = false
    options.CalibrateOmega (1,1) logical = true
end

assert_dependencies;
rng(options.Seed, 'twister');
subject = pam_dot_task_load_subject(csv_path);

switch string(options.PerceptualModel)
    case "cue_shared_effective2"
        prc_model = cue_ehgf_binary_config;
        fit_inputs = subject.u;
    case "official_single_hgf"
        prc_model = tapas_ehgf_binary_config;
        prc_model.logkamu(end) = -Inf;
        prc_model.logkasa(end) = 0;
        prc_model.omsa(end) = 0;
        fit_inputs = subject.u;
end
prc_model = tapas_align_priors(prc_model);

opt_model = tapas_quasinewton_optim_config;
opt_model.nRandInit = double(options.RandomInitialization);
opt_model.seedRandInit = options.Seed;

prior_calibration = struct('performed', false, 'fit', []);
if options.CalibrateOmega
    bo_prc = prc_model;
    bo_prc.omsa(2) = 4;
    bo_prc = tapas_align_priors(bo_prc);
    bo_opt = opt_model;
    bo_opt.nRandInit = 0;
    bo_obs = tapas_bayes_optimal_binary_config;
    rng(options.Seed, 'twister');
    bo = tapas_fitModel([], fit_inputs, bo_prc, ...
        bo_obs, bo_opt);
    prc_model.ommu = bo.p_prc.om;
    prc_model.omsa(2) = 2;
    prc_model.omsa(end) = 0;
    prc_model = tapas_align_priors(prc_model);
    prior_calibration.performed = true;
    prior_calibration.fit = bo;
end

if endsWith(lower(string(model_id)), "_c") || ...
        lower(string(model_id)) == "ddm_c"
    obs_model = pam_dot_task_ddm_coherence_config(model_id);
else
    obs_model = pam_dot_task_ddm_config(model_id);
end
rng(options.Seed, 'twister');
fit = tapas_fitModel(subject.y, fit_inputs, prc_model, obs_model, opt_model);

expected_irregular = subject.irregular_trials;
if ~isequal(fit.irr, expected_irregular)
    error('pam:dot_task:IrregularTrialMismatch', ...
        'tapas_fitModel irregular trials do not match the adapter mask.');
end

quality_fields = {'negLj', 'negLl', 'LME', 'AIC', 'BIC'};
for k = 1:numel(quality_fields)
    value = fit.optim.(quality_fields{k});
    if ~isscalar(value) || ~isfinite(value)
        error('pam:dot_task:NonFiniteFit', ...
            'Fit returned non-finite optim.%s.', quality_fields{k});
    end
end

trialwise = pam_dot_task_trialwise_ddm(fit);
valid = subject.audit.likelihood_included;
if any(~isfinite(trialwise.muhat)) || ...
        any(~isfinite(trialwise.w(valid))) || ...
        any(~isfinite(trialwise.a(valid))) || ...
        any(~isfinite(trialwise.v(valid)))
    error('pam:dot_task:NonFiniteTrialwise', ...
        'Beliefs or trial-wise DDM parameters are non-finite.');
end
if any(trialwise.w(valid) <= 0 | trialwise.w(valid) >= 1)
    error('pam:dot_task:InvalidStartingPoint', ...
        'Trial-wise starting point must be strictly inside (0,1).');
end
if any(trialwise.a(valid) <= 0)
    error('pam:dot_task:InvalidBoundary', ...
        'Trial-wise boundary separation must be positive.');
end
if fit.p_obs.Ter <= 0 || fit.p_obs.Ter >= min(subject.y(valid, 1))
    error('pam:dot_task:InvalidNonDecisionTime', ...
        'Ter must be positive and below the minimum fitted RT.');
end

result = struct;
result.subject = subject;
result.model_id = string(model_id);
result.perceptual_model = string(options.PerceptualModel);
result.seed = options.Seed;
result.random_initialization = options.RandomInitialization;
result.prior_calibration = prior_calibration;
result.fit = fit;
result.trialwise = trialwise;
end

function assert_dependencies
required = [ ...
    "tapas_fitModel", ...
    "tapas_ehgf_binary_config", ...
    "tapas_bayes_optimal_binary_config", ...
    "tapas_quasinewton_optim_config", ...
    "tapas_align_priors", ...
    "ddm_hgf_config"];
missing = strings(0, 1);
for k = 1:numel(required)
    if isempty(which(required(k)))
        missing(end + 1, 1) = required(k); %#ok<AGROW>
    end
end
if ~isempty(missing)
    error('pam:dot_task:MissingDependency', ...
        'Missing MATLAB dependencies on path: %s', strjoin(missing, ', '));
end
end
