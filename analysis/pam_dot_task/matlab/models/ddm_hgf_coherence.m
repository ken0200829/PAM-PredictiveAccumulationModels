function [logp, yhat, res] = ddm_hgf_coherence(r, infStates, ptrans)
%DDM_HGF_COHERENCE Official PAM DDM plus a nested coherence drift slope.
%
% The extension changes drift only:
%
%   direction = 2*stimulus_category - 1
%   belief_presented = u*muhat + (1-u)*(1-muhat)
%   v = direction * (a_v + b_c*abs(signed_coherence) ...
%                    + b_v*(belief_presented - 0.5))
%
% b_c=0 reproduces the official ddm_hgf drift exactly. Starting point,
% boundary, non-decision time, response coding, and WFPT likelihood are
% unchanged. This file is external to PAM_master and does not overwrite the
% official response model.

% Column contract.  The perceptual model addresses r.u through configurable
% indices (cue_ehgf_binary_config sets stimulus_column/cue_column), so the
% response model must resolve the stimulus column the same way instead of
% hardcoding 1.  Otherwise a non-default perceptual configuration makes the
% HGF learn from one column while the DDM takes its drift sign from another,
% and every existing guard still passes (plan section 4.1).
stimulus_column = 1;
if isfield(r, 'c_prc') && isfield(r.c_prc, 'stimulus_column')
    stimulus_column = r.c_prc.stimulus_column;
end
coherence_column = 3;
if isfield(r, 'c_obs') && isfield(r.c_obs, 'coherence_column')
    coherence_column = r.c_obs.coherence_column;
end

if stimulus_column == coherence_column
    error('pam:coherence_ddm:ColumnCollision', ...
        'Stimulus and signed-coherence columns must differ (both %d).', ...
        stimulus_column);
end
if size(r.u, 2) < max(stimulus_column, coherence_column)
    error('pam:coherence_ddm:InputColumns', ...
        ['ddm_hgf_coherence needs stimulus in r.u(:,%d) and signed ' ...
         'coherence in r.u(:,%d), but r.u has %d columns.'], ...
        stimulus_column, coherence_column, size(r.u, 2));
end

a_a = exp(ptrans(1));
a_v = exp(ptrans(2));
b_w = 2 / (1 + exp(-ptrans(3))) - 1;
b_a = ptrans(4);
b_v = ptrans(5);
b_c = ptrans(6);

n = size(infStates, 1);
logp = NaN(n, 1);
yhat = NaN(n, 1);
res = NaN(n, 1);

mu1hat = infStates(:, 1, 1);
mu1hat(r.irr) = [];
rt = r.y(:, 1);
rt(r.irr) = [];
response = r.y(:, 2);
response(r.irr) = [];
stimulus = r.u(:, stimulus_column);
stimulus(r.irr) = [];
signed_coherence = r.u(:, coherence_column);
signed_coherence(r.irr) = [];

if any(~isfinite(signed_coherence)) || any(abs(signed_coherence) > 1)
    error('pam:coherence_ddm:InvalidCoherence', ...
        'signed coherence must be finite and within [-1,1].');
end
% The range check above does not separate the two columns: stimulus_category
% is 0/1, which also lies inside [-1,1].  Assert the stimulus column really is
% binary so a swapped column contract fails loudly instead of producing a
% finite but meaningless drift sign.
if any(~isfinite(stimulus)) || any(~ismember(stimulus, [0, 1]))
    error('pam:coherence_ddm:InvalidStimulus', ...
        'Stimulus column %d must be binary on every regular trial.', ...
        stimulus_column);
end

Ter = min(rt) / (1 + exp(-ptrans(7)));
rt = max(eps, rt - Ter);

w = 0.5 + b_w .* (mu1hat - 0.5);
precision = tapas_sgm(1 ./ (mu1hat .* (1 - mu1hat)) - 4, 1) - 0.5;
a = a_a + b_a .* precision;

direction = 2 .* stimulus - 1;
belief_presented = stimulus .* mu1hat + ...
    (1 - stimulus) .* (1 - mu1hat);
coherence_magnitude = abs(signed_coherence);
v = direction .* (a_v + b_c .* coherence_magnitude + ...
    b_v .* (belief_presented - 0.5));

logp_regular = NaN(numel(stimulus), 1);
for trial = 1:numel(stimulus)
    if rt(trial) > 0
        probability = ...
            utl_wfpt(rt(trial), -v(trial), a(trial), 1 - w(trial)) ...
            .* response(trial) + ...
            utl_wfpt(rt(trial), v(trial), a(trial), w(trial)) ...
            .* (1 - response(trial));
        if probability > 0
            logp_regular(trial) = log(probability + eps);
        end
    end
end

regular = ~ismember(1:n, r.irr);
logp(regular) = logp_regular;
end
