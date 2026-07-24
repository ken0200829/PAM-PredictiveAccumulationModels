function tests = test_pam_dot_task_sequential_ppc
tests = functiontests(localfunctions);
end

function testSpecUsesOriginalTimeAndCuePresentationAxes(testCase)
audit = fixture_audit;
spec = pam_dot_task_ppc_spec(audit);

families = string({spec.windows.family});
verifyEqual(testCase, sum(families == "test_global"), 10);
verifyEqual(testCase, sum(families == "test_cue_presentation"), 20);
verifyEqual(testCase, sum(families == "learning_conditional_holdout"), 5);

first_global = spec.windows(find(families == "test_global", 1));
verifyEqual(testCase, first_global.indices, 101:128);

cue_windows = spec.windows(families == "test_cue_presentation");
verifyEqual(testCase, numel(cue_windows(1).indices), 14);
end

function testOneBatchIsReusedAcrossAllWindows(testCase)
audit = fixture_audit;
spec = pam_dot_task_ppc_spec(audit);
n_replicates = 100;
rng(11, 'twister');

simulation = struct;
simulation.rt = repmat(audit.rt_seconds_raw', n_replicates, 1) + ...
    0.02 .* randn(n_replicates, height(audit));
probability_white = repmat(0.2 + 0.6 .* audit.stimulus_category', ...
    n_replicates, 1);
simulation.choice = double(rand(n_replicates, height(audit)) < probability_white);

ppc = pam_dot_task_sequential_ppc(audit, simulation, spec);
verifyEqual(testCase, ppc.replicates, n_replicates);
verifySize(testCase, ppc.replicated_statistics, ...
    [n_replicates, numel(spec.windows), numel(spec.statistics)]);
verifyTrue(testCase, isfinite(ppc.global_observed));
verifyTrue(testCase, isfinite(ppc.global_tail_probability));
verifyTrue(testCase, isfinite(ppc.simultaneous_threshold));
end

function audit = fixture_audit
trial = (1:380)';
phase = repmat("test", 380, 1);
phase(1:100) = "learning";

cue_white = zeros(380, 1);
cue_white(1:2:end) = 1;
% Match the real design counts used by the fixed cue windows.
cue_white(1:30) = 0;
cue_white(31:100) = 1;

levels = [0.0, 0.1, 0.2, 0.3];
signed_coherence = NaN(380, 1);
for k = 1:380
    magnitude = levels(mod(k - 1, numel(levels)) + 1);
    direction = 2 * mod(floor((k - 1) / numel(levels)), 2) - 1;
    signed_coherence(k) = direction * magnitude;
end
stimulus_category = double(signed_coherence > 0);
rt_seconds_raw = 0.7 + 0.2 .* abs(signed_coherence);
choice_white = stimulus_category;

audit = table(trial, phase, cue_white, signed_coherence, ...
    stimulus_category, rt_seconds_raw, choice_white);
end
