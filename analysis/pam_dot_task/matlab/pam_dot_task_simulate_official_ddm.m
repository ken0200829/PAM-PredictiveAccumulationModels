function simulation = pam_dot_task_simulate_official_ddm(trialwise, options)
%PAM_DOT_TASK_SIMULATE_OFFICIAL_DDM Simulate the official PAM DDM mapping.
%
% Uses the same utl_wfpt parameterization as ddm_hgf.m and the official PAM
% example. A complete replicate-by-trial batch is generated once and can be
% reused by every sequential-PPC window.

arguments
    trialwise table
    options.Replicates (1,1) double {mustBeInteger,mustBePositive} = 2000
    options.Seed (1,1) double {mustBeInteger,mustBeNonnegative} = 20260720
    options.DecisionTimeStep (1,1) double {mustBePositive} = 0.001
    options.MaxDecisionTime (1,1) double {mustBePositive} = 3.0
end

if options.MaxDecisionTime ~= 3.0
    error('pam:ppc:ResponseDeadline', ...
        'The dot task response deadline is fixed at 3 seconds.');
end

required = ["w", "a", "v", "Ter"];
missing_columns = setdiff(required, string(trialwise.Properties.VariableNames));
if ~isempty(missing_columns)
    error('pam:ppc:MissingTrialwiseColumns', ...
        'trialwise is missing: %s', strjoin(missing_columns, ', '));
end
if isempty(which('utl_wfpt'))
    error('pam:ppc:MissingWFPT', 'utl_wfpt is not on the MATLAB path.');
end
if any(~isfinite(trialwise.w) | trialwise.w <= 0 | trialwise.w >= 1) || ...
        any(~isfinite(trialwise.a) | trialwise.a <= 0) || ...
        any(~isfinite(trialwise.v)) || ...
        any(~isfinite(trialwise.Ter) | trialwise.Ter <= 0)
    error('pam:ppc:InvalidDDMParameters', ...
        'All trial-wise DDM parameters must be finite and inside support.');
end

rng(options.Seed, 'twister');
n_trials = height(trialwise);
n_replicates = options.Replicates;
grid = (options.DecisionTimeStep:options.DecisionTimeStep: ...
    options.MaxDecisionTime)';
n_grid = numel(grid);

rt = NaN(n_replicates, n_trials);
choice = NaN(n_replicates, n_trials);
captured_mass = NaN(n_trials, 1);

for trial = 1:n_trials
    % ddm_hgf: choice=1 uses (-v, a, 1-w); choice=0 uses (v, a, w).
    % PAM's utl_wfpt/utl_fsw is scalar-time code: passing the complete grid
    % reaches a scalar short-circuit condition in utl_fsw. Keep PAM untouched
    % and evaluate the identical density one time point at a time here.
    choice_one_density = NaN(n_grid, 1);
    choice_zero_density = NaN(n_grid, 1);
    for grid_index = 1:n_grid
        decision_time = grid(grid_index);
        choice_one_density(grid_index) = utl_wfpt(decision_time, ...
            -trialwise.v(trial), trialwise.a(trial), 1 - trialwise.w(trial));
        choice_zero_density(grid_index) = utl_wfpt(decision_time, ...
            trialwise.v(trial), trialwise.a(trial), trialwise.w(trial));
    end
    weights = [flipud(choice_zero_density(:)); choice_one_density(:)];
    if any(~isfinite(weights)) || any(weights < 0) || sum(weights) <= 0
        error('pam:ppc:InvalidWFPTWeights', ...
            'WFPT simulation weights are invalid on trial %d.', trial);
    end
    captured_mass(trial) = sum(weights) * options.DecisionTimeStep;
    positive_index = find(weights > 0);
    positive_weights = weights(positive_index);
    cumulative = cumsum(positive_weights) ./ sum(positive_weights);
    cumulative(end) = 1;
    draw = rand(n_replicates, 1);
    positive_bin = discretize(draw, [0; cumulative]);
    sampled_index = positive_index(positive_bin);

    is_choice_one = sampled_index > n_grid;
    decision_time = NaN(n_replicates, 1);
    decision_time(is_choice_one) = grid(sampled_index(is_choice_one) - n_grid);
    zero_index = sampled_index(~is_choice_one);
    decision_time(~is_choice_one) = grid(n_grid - zero_index + 1);

    choice(:, trial) = double(is_choice_one);
    rt(:, trial) = decision_time + trialwise.Ter(trial);
end

simulation = struct;
simulation.rt = rt;
simulation.choice = choice;
simulation.replicates = n_replicates;
simulation.seed = options.Seed;
simulation.algorithm = 'twister';
simulation.decision_time_step = options.DecisionTimeStep;
simulation.max_decision_time = options.MaxDecisionTime;
simulation.captured_mass = captured_mass;
simulation.parameter_mode = 'MAP_fixed';
end
