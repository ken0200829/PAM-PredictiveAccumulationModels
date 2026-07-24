function tests = test_ddm_hgf_coherence
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
required = {'ddm_hgf', 'ddm_hgf_config', 'utl_wfpt', ...
    'tapas_sgm', 'tapas_align_priors'};
for k = 1:numel(required)
    assumeFalse(testCase, isempty(which(required{k})), ...
        sprintf('Required dependency not on path: %s', required{k}));
end
end

function testZeroCoherenceSlopeReproducesOfficialLikelihood(testCase)
[r, infStates] = fixture_problem;
official = [log(1.2), log(2.0), 0.3, -0.2, 0.4, -0.5];
extended = [official(1:5), 0, official(6)];

official_logp = ddm_hgf(r, infStates, official);
extended_logp = ddm_hgf_coherence(r, infStates, extended);
verifyEqual(testCase, extended_logp, official_logp, 'AbsTol', 1e-12);
end

function testCoherenceSlopeChangesDriftMagnitude(testCase)
[r, infStates] = fixture_problem;
fit = struct;
fit.u = r.u;
fit.y = r.y;
fit.traj.muhat = infStates(:, 1, 1);
fit.p_obs = struct('a_a', 1.2, 'a_v', 2.0, 'b_w', 0.1, ...
    'b_a', 0.0, 'b_v', 0.0, 'b_c', 1.5, 'Ter', 0.2);

trialwise = pam_dot_task_trialwise_ddm(fit);
low = abs(trialwise.v(1));
high = abs(trialwise.v(4));
verifyGreaterThan(testCase, high, low);
verifyEqual(testCase, high - low, 1.5 * (0.3 - 0.1), ...
    'AbsTol', 1e-12);
end

function testCoherenceRegistryFreeSets(testCase)
cases = { ...
    'ddm_c',      [false, false, false]; ...
    'ddm_w_c',    [true,  false, false]; ...
    'ddm_a_c',    [false, true,  false]; ...
    'ddm_v_c',    [false, false, true]; ...
    'ddm_full_c', [true,  true,  true]};

for k = 1:size(cases, 1)
    c = pam_dot_task_ddm_coherence_config(cases{k, 1});
    verifyEqual(testCase, [c.b_wsa, c.b_asa, c.b_vsa] > 0, cases{k, 2});
    verifyGreaterThan(testCase, c.b_csa, 0);
    verifyEqual(testCase, numel(c.priormus), 7);
end
end

function testDefaultColumnContractIsUnchanged(testCase)
% No c_prc/c_obs present: the model must still resolve columns 1 and 3.
[r, infStates] = fixture_problem;
p = [log(1.2), log(2.0), 0.3, -0.2, 0.4, 0.8, -0.5];
verifyWarningFree(testCase, @() ddm_hgf_coherence(r, infStates, p));
end

function testResponseModelFollowsPerceptualStimulusColumn(testCase)
% The perceptual model addresses r.u through c_prc.stimulus_column.  The
% response model must resolve the same column, otherwise the HGF and the DDM
% silently disagree about which column carries the stimulus category.
[r, infStates] = fixture_problem;
p = [log(1.2), log(2.0), 0.3, -0.2, 0.4, 0.8, -0.5];
expected = ddm_hgf_coherence(r, infStates, p);

moved = r;
moved.u = [r.u(:, 2), r.u(:, 1), r.u(:, 3)];
moved.c_prc = struct('stimulus_column', 2, 'cue_column', 1);
actual = ddm_hgf_coherence(moved, infStates, p);

verifyEqual(testCase, actual, expected, 'AbsTol', 1e-12);
end

function testCollidingColumnsAreRejected(testCase)
[r, infStates] = fixture_problem;
p = [log(1.2), log(2.0), 0.3, -0.2, 0.4, 0.8, -0.5];
bad = r;
bad.c_prc = struct('stimulus_column', 3, 'cue_column', 2);
verifyError(testCase, @() ddm_hgf_coherence(bad, infStates, p), ...
    'pam:coherence_ddm:ColumnCollision');
end

function testNonBinaryStimulusColumnIsRejected(testCase)
% The [-1,1] coherence range check cannot separate the two columns because
% 0/1 also lies inside [-1,1].  A swapped contract must fail loudly.
[r, infStates] = fixture_problem;
p = [log(1.2), log(2.0), 0.3, -0.2, 0.4, 0.8, -0.5];
bad = r;
bad.u = [r.u, [0.25; -0.25; 0.5; -0.5]];
bad.c_prc = struct('stimulus_column', 4, 'cue_column', 2);
verifyError(testCase, @() ddm_hgf_coherence(bad, infStates, p), ...
    'pam:coherence_ddm:InvalidStimulus');
end

function [r, infStates] = fixture_problem
r = struct;
r.u = [ ...
    1, 1,  0.1; ...
    0, 0, -0.1; ...
    1, 0,  0.3; ...
    0, 1, -0.3];
r.y = [0.70, 1; 0.85, 0; 0.65, 1; 0.90, 0];
r.irr = [];
infStates = NaN(4, 1, 4);
infStates(:, 1, 1) = [0.55; 0.45; 0.60; 0.40];
end
