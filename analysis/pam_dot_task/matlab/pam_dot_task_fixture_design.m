function design = pam_dot_task_fixture_design()
%PAM_DOT_TASK_FIXTURE_DESIGN Deterministic 380-trial design for parity fixtures.
%
%   The design carries no participant data. Every value is regenerated from a
%   Lehmer generator, state_{k+1} = 16807 * state_k mod (2^31 - 1), seeded at
%   20260721. That recurrence is exactly reproducible in integer arithmetic on
%   the Python side, so both languages build bit-identical inputs and the
%   fixtures can be exported on a machine that never sees the real CSVs.
%
%   Structure mirrors the real task closely enough to exercise every code
%   path the parity gates cover:
%
%     trials 1-100    learning, response likelihood masked out
%     trials 101-380  test, 280 responses of which 5 are invalid
%     cue_white       70/30 white/red in learning, 140/140 in test
%     |coherence|     0, 0.1, 0.2, 0.3 in test, arranged so that the frozen
%                     49-window sequential PPC spec is realizable
%
%   Returned fields:
%     u                380 x 3  [stimulus_category, cue_white, signed_coherence]
%     y                380 x 2  [rt_seconds, choice_white], NaN where masked
%     trial, phase, ratio_corrected, muhat_reference

seed = 20260721;
state = seed;

n_learning = 100;
n_test = 280;
n_total = n_learning + n_test;

% --- cue assignment ---------------------------------------------------------
cue_learning = [ones(70, 1); zeros(30, 1)];
[cue_learning, state] = lehmer_shuffle(cue_learning, state);
cue_test = [ones(140, 1); zeros(140, 1)];
[cue_test, state] = lehmer_shuffle(cue_test, state);

% --- test ratios ------------------------------------------------------------
% Per cue: 20 trials at ratio 0.5 (|coherence| 0, stimulus 0) and 20 trials at
% each of the six signed levels. 20 + 120 = 140 trials per cue, and the
% resulting cell counts realize exactly 14 coherence windows.
per_cue_ratios = [repmat(0.5, 20, 1); ...
    repmat(0.45, 20, 1); repmat(0.55, 20, 1); ...
    repmat(0.40, 20, 1); repmat(0.60, 20, 1); ...
    repmat(0.35, 20, 1); repmat(0.65, 20, 1)];

ratio_test = zeros(n_test, 1);
white_positions = find(cue_test == 1);
red_positions = find(cue_test == 0);
[white_ratios, state] = lehmer_shuffle(per_cue_ratios, state);
[red_ratios, state] = lehmer_shuffle(per_cue_ratios, state);
ratio_test(white_positions) = white_ratios;
ratio_test(red_positions) = red_ratios;

% --- learning ratios --------------------------------------------------------
% Strong evidence so that the two cue streams actually learn something.
ratio_learning = zeros(n_learning, 1);
for k = 1:n_learning
    [state, value] = lehmer_next(state);
    if cue_learning(k) == 1
        % White cue is close to neutral in the real task.
        if value < 0.5
            ratio_learning(k) = 0.2;
        else
            ratio_learning(k) = 0.8;
        end
    else
        % Red cue is nearly deterministic during learning.
        if value < 0.85
            ratio_learning(k) = 0.8;
        else
            ratio_learning(k) = 0.2;
        end
    end
end

ratio = [ratio_learning; ratio_test];
cue = [cue_learning; cue_test];
stimulus = double(ratio > 0.5);
signed_coherence = 2 * ratio - 1;

% --- responses --------------------------------------------------------------
y = NaN(n_total, 2);
for k = (n_learning + 1):n_total
    [state, u_rt] = lehmer_next(state);
    [state, u_choice] = lehmer_next(state);
    rt = 0.35 + 1.10 * u_rt;
    probability = 0.5 + 0.35 * (2 * stimulus(k) - 1) * abs(signed_coherence(k)) / 0.3;
    probability = min(max(probability, 0.02), 0.98);
    y(k, 1) = rt;
    y(k, 2) = double(u_choice < probability);
end

% Five invalid test responses so the irregular-trial path is exercised. Both
% columns are cleared together, matching the plan's masking contract.
invalid = [131, 187, 244, 301, 365];
y(invalid, :) = NaN;

% --- reference belief series for the simulation fixture ---------------------
% Deterministic and independent of the HGF so that the simulation fixture can
% be built and compared without first passing the HGF gate.
muhat_reference = zeros(n_total, 1);
for k = 1:n_total
    [state, value] = lehmer_next(state);
    muhat_reference(k) = 0.15 + 0.70 * value;
end

design = struct;
design.seed = seed;
design.generator = 'lehmer_16807_mod_2147483647';
design.trial = (1:n_total)';
design.phase = [repmat({'learning'}, n_learning, 1); repmat({'test'}, n_test, 1)];
design.u = [stimulus, cue, signed_coherence];
design.y = y;
design.ratio_corrected = ratio;
design.muhat_reference = muhat_reference;
design.invalid_test_trials = invalid;
design.n_learning = n_learning;
design.n_test = n_test;
end


function [values, state] = lehmer_shuffle(values, state)
% Fisher-Yates driven by the Lehmer stream. Index convention is fixed here and
% must match the Python reconstruction exactly.
for i = numel(values):-1:2
    [state, u] = lehmer_next(state);
    j = floor(u * i) + 1;
    swap = values(i);
    values(i) = values(j);
    values(j) = swap;
end
end


function [state, value] = lehmer_next(state)
state = mod(16807 * state, 2147483647);
value = state / 2147483647;
end
