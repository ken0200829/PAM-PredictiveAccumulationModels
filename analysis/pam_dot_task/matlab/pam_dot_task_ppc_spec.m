function spec = pam_dot_task_ppc_spec(audit)
%PAM_DOT_TASK_PPC_SPEC Pre-outcome window specification for sequential PPC.
%
% The windows depend only on trial design columns, never on observed RT or
% choice. Hash and freeze this structure at Gate PPC before inspecting fits.

arguments
    audit table
end

required = ["trial", "phase", "cue_white", "signed_coherence", ...
    "stimulus_category"];
missing_columns = setdiff(required, string(audit.Properties.VariableNames));
if ~isempty(missing_columns)
    error('pam:ppc:MissingAuditColumns', ...
        'audit is missing: %s', strjoin(missing_columns, ', '));
end
if height(audit) ~= 380 || ~isequal(audit.trial, (1:380)')
    error('pam:ppc:InvalidTrialIndex', ...
        'Sequential PPC requires the original 380-row trial index.');
end

windows = struct('id', {}, 'family', {}, 'indices', {});

% Test phase: ten non-overlapping global blocks of 28 trials.
test_index = find(audit.phase == "test");
for block = 1:10
    take = test_index((block - 1) * 28 + (1:28));
    windows(end + 1) = make_window( ...
        sprintf('test_global_%02d', block), 'test_global', take); %#ok<AGROW>
end

% Cue-presentation time: ten blocks of 14 trials within each cue stream.
for cue_value = [1, 0]
    cue_name = cue_label(cue_value);
    cue_index = find(audit.phase == "test" & audit.cue_white == cue_value);
    if numel(cue_index) ~= 140
        error('pam:ppc:UnexpectedCueCount', ...
            'Expected 140 test trials for cue %s, found %d.', ...
            cue_name, numel(cue_index));
    end
    for block = 1:10
        take = cue_index((block - 1) * 14 + (1:14));
        windows(end + 1) = make_window( ...
            sprintf('test_%s_%02d', cue_name, block), ...
            'test_cue_presentation', take); %#ok<AGROW>
    end
end

% Coherence magnitude is evaluated within stimulus category and cue.
coherence = round(abs(audit.signed_coherence), 10);
levels = [0.0, 0.1, 0.2, 0.3];
observed_levels = unique(coherence(audit.phase == "test"))';
if ~isequal(observed_levels, levels)
    error('pam:ppc:UnexpectedCoherenceLevels', ...
        'Expected test |signed coherence| levels [0, 0.1, 0.2, 0.3].');
end
for level_index = 1:numel(levels)
    level = levels(level_index);
    for stimulus = [0, 1]
        for cue_value = [0, 1]
            take = find(audit.phase == "test" & coherence == level & ...
                audit.stimulus_category == stimulus & ...
                audit.cue_white == cue_value);
            if isempty(take)
                continue
            end
            windows(end + 1) = make_window( ...
                sprintf('coh_%0.1f_stim%d_%s', level, stimulus, ...
                cue_label(cue_value)), 'test_coherence', take); %#ok<AGROW>
        end
    end
end

% Learning phase: five global blocks of 20 trials for conditional hold-out.
learning_index = find(audit.phase == "learning");
for block = 1:5
    take = learning_index((block - 1) * 20 + (1:20));
    windows(end + 1) = make_window( ...
        sprintf('learning_global_%02d', block), ...
        'learning_conditional_holdout', take); %#ok<AGROW>
end

spec = struct;
spec.version = '1.0.0';
spec.windows = windows;
spec.statistics = ["choice_rate", "accuracy", "rt_q10", "rt_q50", ...
    "rt_q90", "rt_choice0_q50", "rt_choice1_q50"];
spec.min_valid_trials = 5;
spec.point_interval = [0.025, 0.975];
spec.simultaneous_level = 0.95;
spec.global_discrepancy = 'max_absolute_standardized_deviation';
end

function window = make_window(id, family, indices)
window = struct('id', string(id), 'family', string(family), ...
    'indices', indices(:)');
end

function label = cue_label(value)
if value == 1
    label = 'white';
else
    label = 'red';
end
end
