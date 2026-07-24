function tests = test_pam_dot_task_adapter
tests = functiontests(localfunctions);
end

function testConditionTable(testCase)
normal = pam_dot_task_condition('normal_dot_task_subject.csv');
normal_cb = pam_dot_task_condition('normal_cb_dot_task_subject.csv');
reverse = pam_dot_task_condition('reverse_dot_task_subject.csv');
reverse_cb = pam_dot_task_condition('reverse_cb_dot_task_subject.csv');

verifyEqual(testCase, normal.name, "normal");
verifyFalse(testCase, normal.stimulus_reversed);
verifyEqual(testCase, normal.white_key, "j");
verifyEqual(testCase, normal_cb.white_key, "f");
verifyTrue(testCase, reverse.stimulus_reversed);
verifyEqual(testCase, reverse.white_key, "f");
verifyTrue(testCase, reverse_cb.stimulus_reversed);
verifyEqual(testCase, reverse_cb.white_key, "j");
end

function testUnknownConditionFails(testCase)
verifyError(testCase, ...
    @() pam_dot_task_condition('unknown_dot_task_subject.csv'), ...
    'pam:dot_task:UnknownCondition');
end

function testAllTrialsRetainedAndResponsesMasked(testCase)
folder = tempname;
mkdir(folder);
cleanup = onCleanup(@() rmdir(folder, 's'));
csv_path = fullfile(folder, 'normal_dot_task_fixture.csv');

raw = fixture_table("j", 0.6);
raw.rt(101) = 50;
raw.response(102) = "x";
writetable(raw, csv_path);

subject = pam_dot_task_load_subject(csv_path);
verifySize(testCase, subject.u, [380, 3]);
verifySize(testCase, subject.y, [380, 2]);
verifyTrue(testCase, all(isfinite(subject.u), 'all'));
verifyTrue(testCase, all(isnan(subject.y(1:100, :)), 'all'));
verifyTrue(testCase, all(isnan(subject.y(101:102, :)), 'all'));
verifyTrue(testCase, all(isfinite(subject.y(103:380, :)), 'all'));
verifyEqual(testCase, subject.audit.rt_seconds_raw(101), 0.05, ...
    'AbsTol', 1e-12);
verifyEqual(testCase, subject.audit.choice_white(102), NaN);
verifySubstring(testCase, subject.audit.exclude_reason(101), "rt_low");
verifySubstring(testCase, subject.audit.exclude_reason(102), "invalid_key");
verifyEqual(testCase, subject.irregular_trials, 1:102);
end

function testReverseConditionCorrectsRatioAndChoice(testCase)
folder = tempname;
mkdir(folder);
cleanup = onCleanup(@() rmdir(folder, 's'));
csv_path = fullfile(folder, 'reverse_dot_task_fixture.csv');

raw = fixture_table("f", 0.8);
writetable(raw, csv_path);

subject = pam_dot_task_load_subject(csv_path);
verifyEqual(testCase, subject.audit.ratio_raw, repmat(0.8, 380, 1), ...
    'AbsTol', 1e-12);
verifyEqual(testCase, subject.audit.ratio_corrected, repmat(0.2, 380, 1), ...
    'AbsTol', 1e-12);
verifyEqual(testCase, subject.audit.stimulus_category, zeros(380, 1));
verifyEqual(testCase, subject.audit.choice_white, ones(380, 1));
verifyEqual(testCase, subject.y(101:380, 2), ones(280, 1));
% ratio 0.8 reverses to 0.2, so no trial is a tie.
verifyEqual(testCase, subject.audit.is_tie, zeros(380, 1));
end

function testTieTrialsAreFlaggedInBothDirections(testCase)
% ratio 0.5 has no objective category: 100 white and 100 black dots.  The
% flag must fire in the normal branch and in the reversed (1 - ratio) branch,
% and must agree exactly with signed_coherence == 0 (plan section 5.2.1).
folder = tempname;
mkdir(folder);
cleanup = onCleanup(@() rmdir(folder, 's')); %#ok<NASGU>

for spec = {"normal", "j"; "reverse", "f"}'
    condition_name = spec{1};
    white_key = spec{2};
    csv_path = fullfile(folder, ...
        sprintf('%s_dot_task_20260101_000000_tie.csv', condition_name));
    writetable(fixture_table(white_key, 0.5), csv_path);

    subject = pam_dot_task_load_subject(csv_path);
    verifyEqual(testCase, subject.audit.ratio_corrected, ...
        repmat(0.5, 380, 1), 'AbsTol', 1e-12);
    verifyEqual(testCase, subject.audit.signed_coherence, zeros(380, 1), ...
        'AbsTol', 0);
    verifyEqual(testCase, subject.audit.is_tie, ones(380, 1));
    verifyEqual(testCase, subject.audit.is_tie, ...
        double(subject.audit.signed_coherence == 0));
end
end

function testPartialMainTrialSequenceFails(testCase)
folder = tempname;
mkdir(folder);
cleanup = onCleanup(@() rmdir(folder, 's'));
csv_path = fullfile(folder, 'normal_dot_task_fixture.csv');

raw = fixture_table("j", 0.6);
raw(380, :) = [];
writetable(raw, csv_path);

verifyError(testCase, @() pam_dot_task_load_subject(csv_path), ...
    'pam:dot_task:InvalidTrialSequence');
end

function raw = fixture_table(response_value, ratio_value)
main_trial_number = (1:380)';
rt = repmat(500, 380, 1);
response = repmat(string(response_value), 380, 1);
ratio = repmat(ratio_value, 380, 1);
cross_color = repmat("white", 380, 1);
cross_color(2:2:end) = "red";
raw = table(main_trial_number, rt, response, ratio, cross_color);
end
