function tests = test_cue_ehgf_binary
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
required = {'tapas_ehgf_binary', 'tapas_ehgf_binary_transp', 'tapas_sgm'};
for k = 1:numel(required)
    assumeFalse(testCase, isempty(which(required{k})), ...
        sprintf('TAPAS dependency not on path: %s', required{k}));
end
end

function testConfigIsSharedEffectiveTwoLevel(testCase)
c = cue_ehgf_binary_config;
verifyTrue(testCase, c.shared_parameters);
verifyEqual(testCase, c.inactive_cue_policy, 'freeze');
verifyEqual(testCase, c.logkamu(end), -Inf);
verifyEqual(testCase, c.logkasa(end), 0);
verifyEqual(testCase, c.omsa(end), 0);
verifyGreaterThan(testCase, c.omsa(2), 0);
end

function testActiveStatesEqualTwoStandaloneHGFCalls(testCase)
[r, ptrans] = fixture_problem;
[traj, states] = cue_ehgf_binary(r, ptrans, 'trans');

white = find(r.u(:, 2) == 1);
red = find(r.u(:, 2) == 0);
r_white = standalone_problem(r, white);
r_red = standalone_problem(r, red);
[white_traj, white_states] = tapas_ehgf_binary(r_white, ptrans, 'trans');
[red_traj, red_states] = tapas_ehgf_binary(r_red, ptrans, 'trans');

verifyEqual(testCase, states(white, :, :), white_states, 'AbsTol', 1e-12);
verifyEqual(testCase, states(red, :, :), red_states, 'AbsTol', 1e-12);
verifyEqual(testCase, traj.muhat(white, :), white_traj.muhat, ...
    'AbsTol', 1e-12);
verifyEqual(testCase, traj.muhat(red, :), red_traj.muhat, ...
    'AbsTol', 1e-12);
verifyEqual(testCase, traj.cue.white.index, white);
verifyEqual(testCase, traj.cue.red.index, red);
end

function testChangingRedInputsCannotChangeWhiteBeliefs(testCase)
[r, ptrans] = fixture_problem;
[~, states_original] = cue_ehgf_binary(r, ptrans, 'trans');
white = r.u(:, 2) == 1;
red = ~white;

r_changed = r;
r_changed.u(red, 1) = 1 - r_changed.u(red, 1);
[~, states_changed] = cue_ehgf_binary(r_changed, ptrans, 'trans');

verifyEqual(testCase, states_original(white, :, :), ...
    states_changed(white, :, :), 'AbsTol', 1e-12);
end

function testPriorBeliefDoesNotLookAhead(testCase)
[r, ptrans] = fixture_problem;
[~, states_original] = cue_ehgf_binary(r, ptrans, 'trans');

target = 3; % second white-cue presentation
r_changed = r;
r_changed.u(target, 1) = 1 - r_changed.u(target, 1);
[~, states_changed] = cue_ehgf_binary(r_changed, ptrans, 'trans');

verifyEqual(testCase, states_original(target, 1, 1), ...
    states_changed(target, 1, 1), 'AbsTol', 1e-12);
next_same_cue = 5;
verifyGreaterThan(testCase, abs( ...
    states_original(next_same_cue, 1, 1) - ...
    states_changed(next_same_cue, 1, 1)), 1e-12);
end

function [r, ptrans] = fixture_problem
c = cue_ehgf_binary_config;
r = struct;
r.c_prc = c;
r.u = [ ...
    1 1; ...
    0 0; ...
    0 1; ...
    1 0; ...
    1 1; ...
    1 0; ...
    0 1; ...
    0 0];
r.y = NaN(size(r.u, 1), 2);
r.ign = [];
r.irr = 1:size(r.u, 1);
ptrans = c.priormus;
end

function r_sub = standalone_problem(r, index)
r_sub = r;
r_sub.u = r.u(index, 1);
r_sub.y = r.y(index, :);
r_sub.ign = [];
r_sub.irr = 1:numel(index);
end
