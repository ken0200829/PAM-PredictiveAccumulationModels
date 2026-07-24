function bms = pam_dot_task_bms(lme, model_ids, subject_ids, options)
%PAM_DOT_TASK_BMS Paper-faithful random-effects BMS with named inputs.
%
% Rows of lme are subjects and columns are models. The primary result uses
% spm_BMS_gibbs. spm_BMS is optional and stored separately as sensitivity.

arguments
    lme double
    model_ids string
    subject_ids string
    options.Alpha0 double = []
    options.Samples (1,1) double {mustBeInteger,mustBePositive} = 1000000
    options.Seed (1,1) double {mustBeInteger,mustBeNonnegative} = 20260720
    options.RunSPMSensitivity (1,1) logical = false
end

model_ids = model_ids(:)';
subject_ids = subject_ids(:);
if size(lme, 1) ~= numel(subject_ids) || size(lme, 2) ~= numel(model_ids)
    error('pam:bms:ShapeMismatch', ...
        'LME must be subjects-by-models and match both ID vectors.');
end
if numel(unique(model_ids)) ~= numel(model_ids) || ...
        numel(unique(subject_ids)) ~= numel(subject_ids)
    error('pam:bms:DuplicateIDs', 'Subject and model IDs must be unique.');
end
if any(~isfinite(lme), 'all')
    error('pam:bms:NonFiniteLME', ...
        ['BMS requires the common subject set with finite LME for every ' ...
         'model; missing LME is never imputed.']);
end
if isempty(which('spm_BMS_gibbs'))
    error('pam:bms:MissingSPMGibbs', ...
        'spm_BMS_gibbs is not on the MATLAB path.');
end

if isempty(options.Alpha0)
    alpha0 = ones(1, numel(model_ids));
else
    alpha0 = options.Alpha0(:)';
end
if numel(alpha0) ~= numel(model_ids) || any(~isfinite(alpha0) | alpha0 <= 0)
    error('pam:bms:InvalidAlpha0', ...
        'Alpha0 must contain one finite positive value per model.');
end

rng(options.Seed, 'twister');
[exp_r, xp, r_samp, g_post] = spm_BMS_gibbs( ...
    lme, alpha0, options.Samples);

primary = struct;
primary.method = 'spm_BMS_gibbs';
primary.exp_r = exp_r;
primary.xp = xp;
primary.r_samp = r_samp;
primary.g_post = g_post;

sensitivity = struct('performed', false);
if options.RunSPMSensitivity
    if isempty(which('spm_BMS'))
        error('pam:bms:MissingSPM', ...
            'RunSPMSensitivity=true but spm_BMS is not on the path.');
    end
    rng(options.Seed, 'twister');
    output_count = nargout('spm_BMS');
    if output_count >= 5 || output_count < 0
        [alpha, sens_exp_r, sens_xp, pxp, bor] = spm_BMS( ...
            lme, options.Samples, 0, 0, 1, alpha0);
    else
        [alpha, sens_exp_r, sens_xp] = spm_BMS( ...
            lme, options.Samples, 0, 0, 1, alpha0);
        pxp = NaN(size(sens_xp));
        bor = NaN;
    end
    sensitivity = struct( ...
        'performed', true, ...
        'method', 'spm_BMS', ...
        'alpha', alpha, ...
        'exp_r', sens_exp_r, ...
        'xp', sens_xp, ...
        'pxp', pxp, ...
        'bor', bor);
end

bms = struct;
bms.model_ids = model_ids;
bms.subject_ids = subject_ids;
bms.lme = lme;
bms.alpha0 = alpha0;
bms.samples = options.Samples;
bms.seed = options.Seed;
bms.algorithm = 'twister';
bms.primary = primary;
bms.sensitivity = sensitivity;
end
