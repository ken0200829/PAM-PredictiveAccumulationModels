function trialwise = pam_dot_task_trialwise_ddm(fit)
%PAM_DOT_TASK_TRIALWISE_DDM Reconstruct official PAM DDM parameters.

required = {'u', 'y', 'traj', 'p_obs'};
for k = 1:numel(required)
    if ~isfield(fit, required{k})
        error('pam:dot_task:MissingFitField', ...
            'Fit structure is missing field: %s', required{k});
    end
end

muhat = fit.traj.muhat(:, 1);
stimulus = fit.u(:, 1);
p = fit.p_obs;
if isfield(p, 'b_c')
    b_c = p.b_c;
else
    b_c = 0;
end
if size(fit.u, 2) >= 3
    coherence_magnitude = abs(fit.u(:, 3));
else
    coherence_magnitude = zeros(size(stimulus));
end

precision_modulator = tapas_sgm( ...
    1 ./ (muhat .* (1 - muhat)) - 4, 1) - 0.5;
w = 0.5 + p.b_w .* (muhat - 0.5);
a = p.a_a + p.b_a .* precision_modulator;
direction = 2 .* stimulus - 1;
belief_presented = stimulus .* muhat + (1 - stimulus) .* (1 - muhat);
v = direction .* (p.a_v + b_c .* coherence_magnitude + ...
    p.b_v .* (belief_presented - 0.5));
Ter = repmat(p.Ter, size(muhat));

trial = (1:numel(muhat))';
trialwise = table(trial, stimulus, coherence_magnitude, muhat, ...
    precision_modulator, w, a, v, Ter);
end
