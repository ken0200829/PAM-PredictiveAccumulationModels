function spec = pam_dot_task_aggregate_ppc_spec(audit)
%PAM_DOT_TASK_AGGREGATE_PPC_SPEC Aggregate views of the fixed PPC design.
%
% These seven windows deliberately use the same trial positions and response
% statistics as PAM_DOT_TASK_SEQUENTIAL_PPC.  Only the grouping differs, so
% one simulated response batch can be evaluated by both functions.

arguments
    audit table
end

required = ["trial", "phase", "cue_white", "signed_coherence"];
missing_columns = setdiff(required, string(audit.Properties.VariableNames));
if ~isempty(missing_columns)
    error('pam:ppc:MissingAuditColumns', ...
        'audit is missing: %s', strjoin(missing_columns, ', '));
end
if height(audit) ~= 380 || ~isequal(audit.trial, (1:380)')
    error('pam:ppc:InvalidTrialIndex', ...
        'Aggregate PPC requires the original 380-row trial index.');
end

windows = struct('id', {}, 'family', {}, 'indices', {});
test = audit.phase == "test";
windows(end + 1) = make_window('test_all', 'test_aggregate', find(test)); %#ok<AGROW>
for cue_value = [1, 0]
    if cue_value == 1
        label = 'white';
    else
        label = 'red';
    end
    windows(end + 1) = make_window(sprintf('test_cue_%s', label), ...
        'test_aggregate_cue', find(test & audit.cue_white == cue_value)); %#ok<AGROW>
end

coherence = round(abs(audit.signed_coherence), 10);
for level = [0.0, 0.1, 0.2, 0.3]
    windows(end + 1) = make_window(sprintf('test_coherence_%0.1f', level), ...
        'test_aggregate_coherence', find(test & coherence == level)); %#ok<AGROW>
end

spec = struct;
spec.version = 'aggregate-1.0.0';
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
