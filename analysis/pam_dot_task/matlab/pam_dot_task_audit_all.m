function audit = pam_dot_task_audit_all(data_dir, options)
%PAM_DOT_TASK_AUDIT_ALL Validate every subject and optionally write derivatives.
%
% No file is written unless OutputDir is explicitly provided. The raw data
% directory is never modified.

arguments
    data_dir {mustBeTextScalar}
    options.ExpectedSubjects (1,1) double {mustBeInteger,mustBePositive} = 37
    options.OutputDir {mustBeTextScalar} = ""
end

data_dir = string(data_dir);
if ~isfolder(data_dir)
    error('pam:dot_task:MissingDataDir', ...
        'Data directory does not exist: %s', data_dir);
end

files = dir(fullfile(data_dir, '*_dot_task_*.csv'));
[~, order] = sort({files.name});
files = files(order);
if numel(files) ~= options.ExpectedSubjects
    error('pam:dot_task:UnexpectedSubjectCount', ...
        'Expected %d subject CSVs, found %d in %s.', ...
        options.ExpectedSubjects, numel(files), data_dir);
end

subject_id = strings(numel(files), 1);
condition = strings(numel(files), 1);
n_trials = zeros(numel(files), 1);
n_learning = zeros(numel(files), 1);
n_test = zeros(numel(files), 1);
n_likelihood = zeros(numel(files), 1);
n_invalid_learning = zeros(numel(files), 1);
n_invalid_test = zeros(numel(files), 1);
n_white_cue = zeros(numel(files), 1);
n_red_cue = zeros(numel(files), 1);

write_outputs = strlength(string(options.OutputDir)) > 0;
if write_outputs
    output_dir = string(options.OutputDir);
    if paths_equal(output_dir, data_dir)
        error('pam:dot_task:UnsafeOutputDir', ...
            'OutputDir must not be the raw data directory.');
    end
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end
    subject_output_dir = fullfile(output_dir, 'subjects');
    if ~isfolder(subject_output_dir)
        mkdir(subject_output_dir);
    end
end

for k = 1:numel(files)
    csv_path = fullfile(files(k).folder, files(k).name);
    current = pam_dot_task_load_subject(csv_path);
    t = current.audit;

    subject_id(k) = current.subject_id;
    condition(k) = current.condition.name;
    n_trials(k) = height(t);
    n_learning(k) = sum(t.phase == "learning");
    n_test(k) = sum(t.phase == "test");
    n_likelihood(k) = sum(t.likelihood_included);
    n_invalid_learning(k) = sum(t.phase == "learning" & ...
        ~response_is_physically_valid(t));
    n_invalid_test(k) = sum(t.phase == "test" & ...
        ~response_is_physically_valid(t));
    n_white_cue(k) = sum(t.cue_white == 1);
    n_red_cue(k) = sum(t.cue_white == 0);

    if write_outputs
        writetable(t, fullfile(subject_output_dir, current.subject_id + ".csv"));
    end
end

subjects = table(subject_id, condition, n_trials, n_learning, n_test, ...
    n_likelihood, n_invalid_learning, n_invalid_test, n_white_cue, n_red_cue);

condition_order = ["normal", "normal_cb", "reverse", "reverse_cb"]';
condition_n = zeros(numel(condition_order), 1);
for k = 1:numel(condition_order)
    condition_n(k) = sum(condition == condition_order(k));
end
conditions = table(condition_order, condition_n, ...
    'VariableNames', {'condition', 'n_subjects'});

totals = struct;
totals.n_subjects = height(subjects);
totals.n_trials = sum(n_trials);
totals.n_learning = sum(n_learning);
totals.n_test = sum(n_test);
totals.n_likelihood = sum(n_likelihood);
totals.n_invalid_learning = sum(n_invalid_learning);
totals.n_invalid_test = sum(n_invalid_test);

assert(all(n_trials == 380), ...
    'pam:dot_task:AuditTrialCount', 'Every subject must have 380 trials.');
assert(all(n_learning == 100 & n_test == 280), ...
    'pam:dot_task:AuditPhaseCount', ...
    'Every subject must have 100 learning and 280 test trials.');
assert(all(n_white_cue == 210 & n_red_cue == 170), ...
    'pam:dot_task:AuditCueCount', ...
    'Every subject must have 210 white-cue and 170 red-cue trials.');

audit = struct('subjects', subjects, 'conditions', conditions, 'totals', totals);

if write_outputs
    writetable(subjects, fullfile(output_dir, 'subjects_summary.csv'));
    writetable(conditions, fullfile(output_dir, 'conditions_summary.csv'));
    json_path = fullfile(output_dir, 'totals.json');
    file_id = fopen(json_path, 'w');
    if file_id < 0
        error('pam:dot_task:CannotWriteAudit', ...
            'Could not open audit output: %s', json_path);
    end
    cleanup = onCleanup(@() fclose(file_id));
    fprintf(file_id, '%s\n', jsonencode(totals, 'PrettyPrint', true));
end
end

function valid = response_is_physically_valid(t)
valid_rt = isfinite(t.rt_seconds_raw) & ...
    t.rt_seconds_raw >= 0.15 & t.rt_seconds_raw <= 3.0;
valid_choice = isfinite(t.choice_white);
valid = valid_rt & valid_choice;
end

function same = paths_equal(left, right)
[left_ok, left_attributes] = fileattrib(char(left));
[right_ok, right_attributes] = fileattrib(char(right));
if left_ok
    left = string(left_attributes.Name);
end
if right_ok
    right = string(right_attributes.Name);
end
same = string(left) == string(right);
end
