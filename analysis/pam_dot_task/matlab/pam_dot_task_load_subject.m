function subject = pam_dot_task_load_subject(csv_path)
%PAM_DOT_TASK_LOAD_SUBJECT Build PAM inputs without deleting trial history.
%
% Returns a structure with:
%   audit - 380-row table retaining raw and derived values
%   u     - [stimulus_category, cue_white, signed_coherence]
%   y     - [RT_seconds, choice_white], masked to test likelihood trials
%
% Raw CSVs are read only. Learning responses and invalid test responses are
% represented by an all-NaN row in y while their finite stimulus remains in u.

arguments
    csv_path {mustBeTextScalar}
end

csv_path = string(csv_path);
if ~isfile(csv_path)
    error('pam:dot_task:MissingCSV', 'CSV does not exist: %s', csv_path);
end

condition = pam_dot_task_condition(csv_path);
raw = readtable(csv_path, 'TextType', 'string');
required = ["main_trial_number", "rt", "response", "ratio", "cross_color"];
missing_columns = setdiff(required, string(raw.Properties.VariableNames));
if ~isempty(missing_columns)
    error('pam:dot_task:MissingColumns', ...
        'CSV %s is missing required columns: %s', ...
        csv_path, strjoin(missing_columns, ', '));
end

main_trial = numeric_column(raw.main_trial_number);
main_mask = isfinite(main_trial);
main = raw(main_mask, :);
raw_row = find(main_mask);
trial = main_trial(main_mask);

if any(trial ~= fix(trial))
    error('pam:dot_task:NonIntegerTrial', ...
        'main_trial_number contains non-integer values: %s', csv_path);
end

[trial, order] = sort(trial);
main = main(order, :);
raw_row = raw_row(order);
expected_trials = (1:380)';
if ~isequal(trial, expected_trials)
    error('pam:dot_task:InvalidTrialSequence', ...
        ['Expected each main trial 1:380 exactly once in %s; ' ...
         'found %d rows with range [%g, %g].'], ...
        csv_path, numel(trial), min_or_nan(trial), max_or_nan(trial));
end

n = numel(trial);
phase = repmat("test", n, 1);
phase(trial <= 100) = "learning";

rt_seconds_raw = numeric_column(main.rt) ./ 1000;
rt_valid = isfinite(rt_seconds_raw) & ...
    rt_seconds_raw >= 0.15 & rt_seconds_raw <= 3.0;

response_key = lower(strtrim(string(main.response)));
response_missing = ismissing(response_key) | response_key == "";
response_valid = response_key == condition.white_key | ...
    response_key == condition.black_key;
choice_white = NaN(n, 1);
choice_white(response_key == condition.white_key) = 1;
choice_white(response_key == condition.black_key) = 0;

ratio_raw = numeric_column(main.ratio);
if any(~isfinite(ratio_raw) | ratio_raw < 0 | ratio_raw > 1)
    error('pam:dot_task:InvalidRatio', ...
        'ratio must be finite and within [0,1] for all 380 trials: %s', csv_path);
end
if condition.stimulus_reversed
    ratio_corrected = 1 - ratio_raw;
else
    ratio_corrected = ratio_raw;
end
signed_coherence = 2 .* ratio_corrected - 1;
stimulus_category = double(ratio_corrected > 0.5);

% Tie trials (ratio_corrected == 0.5) have no objective category: the display
% holds 100 white and 100 black dots.  The stimulus_category value stored for
% them is a placeholder carrying no semantics -- consumers must branch on
% is_tie instead (plan section 5.2.1).  Ratio 0.5 is exactly representable and
% 2*0.5 - 1 == 0 exactly, in both the normal and the reversed (1 - ratio)
% branch, so the equality test below is safe.
is_tie = double(signed_coherence == 0);

cue = lower(strtrim(string(main.cross_color)));
cue_valid = cue == "white" | cue == "red";
if any(~cue_valid)
    bad = unique(cue(~cue_valid));
    error('pam:dot_task:InvalidCue', ...
        'cross_color must be white or red in %s; found: %s', ...
        csv_path, strjoin(bad, ', '));
end
cue_white = double(cue == "white");

likelihood_included = phase == "test" & rt_valid & response_valid;
rt_for_pam = rt_seconds_raw;
choice_for_pam = choice_white;
rt_for_pam(~likelihood_included) = NaN;
choice_for_pam(~likelihood_included) = NaN;

exclude_reason = strings(n, 1);
exclude_reason(phase == "learning") = add_reason( ...
    exclude_reason(phase == "learning"), "learning_likelihood_mask");
exclude_reason(~isfinite(rt_seconds_raw)) = add_reason( ...
    exclude_reason(~isfinite(rt_seconds_raw)), "rt_missing");
exclude_reason(isfinite(rt_seconds_raw) & rt_seconds_raw < 0.15) = add_reason( ...
    exclude_reason(isfinite(rt_seconds_raw) & rt_seconds_raw < 0.15), "rt_low");
exclude_reason(isfinite(rt_seconds_raw) & rt_seconds_raw > 3.0) = add_reason( ...
    exclude_reason(isfinite(rt_seconds_raw) & rt_seconds_raw > 3.0), "rt_high");
exclude_reason(response_missing) = add_reason( ...
    exclude_reason(response_missing), "choice_missing");
invalid_key = ~response_missing & ~response_valid;
exclude_reason(invalid_key) = add_reason( ...
    exclude_reason(invalid_key), "invalid_key");
exclude_reason(likelihood_included) = "included";

[~, subject_stem, ~] = fileparts(char(csv_path));
subject_id = repmat(string(subject_stem), n, 1);
condition_name = repmat(condition.name, n, 1);
stimulus_reversed = repmat(condition.stimulus_reversed, n, 1);
white_key = repmat(condition.white_key, n, 1);

audit = table( ...
    subject_id, condition_name, stimulus_reversed, white_key, raw_row, ...
    trial, phase, rt_seconds_raw, rt_for_pam, response_key, choice_white, ...
    choice_for_pam, ratio_raw, ratio_corrected, signed_coherence, ...
    stimulus_category, is_tie, cue, cue_white, likelihood_included, ...
    exclude_reason, ...
    'VariableNames', { ...
    'subject_id', 'condition', 'stimulus_reversed', 'white_key', 'raw_row', ...
    'trial', 'phase', 'rt_seconds_raw', 'rt_for_pam', 'response_key', ...
    'choice_white', 'choice_for_pam', 'ratio_raw', 'ratio_corrected', ...
    'signed_coherence', 'stimulus_category', 'is_tie', 'cue', 'cue_white', ...
    'likelihood_included', 'exclude_reason'});

u = [stimulus_category, cue_white, signed_coherence];
y = [rt_for_pam, choice_for_pam];

assert(size(u, 1) == 380 && size(y, 1) == 380, ...
    'pam:dot_task:RowCountChanged', 'PAM inputs must retain all 380 trials.');
assert(all(isfinite(u), 'all'), ...
    'pam:dot_task:NonFiniteInput', 'All HGF inputs must remain finite.');
assert(all(all(isnan(y(~likelihood_included, :)))), ...
    'pam:dot_task:PartialMask', 'Every excluded response row must be all NaN.');
assert(all(all(isfinite(y(likelihood_included, :)))), ...
    'pam:dot_task:InvalidIncludedResponse', ...
    'Every included response row must be finite.');

subject = struct;
subject.subject_id = string(subject_stem);
subject.csv_path = csv_path;
subject.condition = condition;
subject.audit = audit;
subject.u = u;
subject.y = y;
subject.irregular_trials = find(~likelihood_included)';
subject.likelihood_trials = find(likelihood_included)';
end

function values = numeric_column(column)
if isnumeric(column) || islogical(column)
    values = double(column);
else
    values = str2double(strtrim(string(column)));
end
values = values(:);
end

function values = add_reason(values, reason)
empty = values == "";
values(empty) = reason;
values(~empty) = values(~empty) + ";" + reason;
end

function value = min_or_nan(values)
if isempty(values)
    value = NaN;
else
    value = min(values);
end
end

function value = max_or_nan(values)
if isempty(values)
    value = NaN;
else
    value = max(values);
end
end
