function [sim_rt, sim_choice] = pam_simulate_responses(u, muhat, p_obs, model_type, n_reps)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulates responses from fitted PAM models
%
% INPUTS:
%   u          - Task inputs (stimulus identity: 0 or 1)
%   muhat      - Predicted beliefs from HGF perceptual model [n_trials x 1]
%   p_obs      - Fitted observation parameters structure
%   model_type - String: 'DDM', 'RDM', or 'LNR'
%   n_reps     - Number of simulation repetitions
%
% OUTPUTS:
%   sim_rt     - Simulated response times [n_trials x n_reps]
%   sim_choice - Simulated choices [n_trials x n_reps]
%
% DESCRIPTION:
% This function simulates behavioral responses (RTs and choices) from fitted PAM models
% by sampling from the appropriate distributions for each model type:
% - DDM: Samples from Wiener First Passage Time (WFPT) distribution
% - RDM: Samples from competing Inverse Gaussian distributions (race model)
% - LNR: Samples from competing Lognormal distributions (race model)
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2025 Antonino Visalli
%
% This file is part of the PAM toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_trials = length(u);
sim_rt = nan(n_trials, n_reps);
sim_choice = nan(n_trials, n_reps);

% Extract non-decision time
if isfield(p_obs, 'Ter')
    Ter = p_obs.Ter;
else
    Ter = 0;
end

% Simulate responses for each repetition
for r = 1:n_reps
    if strcmp(model_type, 'DDM')
        % Set DDM-specific defaults
        if ~isfield(p_obs, 'b_v'), p_obs.b_v = 0; end
        if ~isfield(p_obs, 'b_a'), p_obs.b_a = 0; end
        if ~isfield(p_obs, 'b_w'), p_obs.b_w = 0; end
        
        % Calculate trial-wise drift rates
        v_vec = p_obs.a_v*((u==1)-(u==0)) + p_obs.b_v.*(muhat-0.5);
        
        % Calculate trial-wise boundary separation (modulated by precision)
        try
            precision = tapas_sgm(1./(muhat.*(1-muhat)) - 4, 1) - 0.5;
        catch
            precision = muhat.*(1-muhat);
        end
        a_vec = max(p_obs.a_a + p_obs.b_a .* precision, 0.1);
        
        % Calculate trial-wise starting point
        w_vec = min(max(0.5 + p_obs.b_w.*(muhat-0.5), 0.01), 0.99);
        
        % Sample from WFPT distribution
        t_grid = 0.002:0.002:5.0;
        domain = [-flip(t_grid), t_grid];
        
        for t = 1:n_trials
            % Calculate WFPT densities for both boundaries
            P1 = utl_wfpt(t_grid, -v_vec(t), a_vec(t), 1-w_vec(t));
            P2 = utl_wfpt(t_grid, v_vec(t), a_vec(t), w_vec(t));
            P = max([P2(end:-1:1), P1], 0);
            
            % Sample RT and choice
            if sum(P) == 0
                val = datasample(domain, 1);
            else
                val = randsample(domain, 1, true, P);
            end
            sim_rt(t,r) = abs(val) + Ter;
            sim_choice(t,r) = double(val > 0);
        end
        
    elseif strcmp(model_type, 'RDM')
        % Set RDM-specific defaults
        if ~isfield(p_obs, 'b_a'), p_obs.b_a = 0; end
        if ~isfield(p_obs, 'b_v'), p_obs.b_v = 0; end
        if ~isfield(p_obs, 'b_val'), p_obs.b_val = 0; end
        
        % Calculate trial-wise thresholds for both accumulators
        a_c1 = max(p_obs.a_a + p_obs.b_a.*(muhat-0.5), 0.01);
        a_c0 = max(p_obs.a_a + p_obs.b_a.*((1-muhat)-0.5), 0.01);
        
        % Calculate trial-wise drift rates for both accumulators
        v_c1 = max(p_obs.a_v + p_obs.b_val.*(u==1) + p_obs.b_v.*(muhat-0.5), 0.001);
        v_c0 = max(p_obs.a_v + p_obs.b_val.*(u==0) + p_obs.b_v.*((1-muhat)-0.5), 0.001);
        
        % Sample from inverse Gaussian distributions
        rt1 = random('InverseGaussian', a_c1./v_c1, a_c1.^2);
        rt0 = random('InverseGaussian', a_c0./v_c0, a_c0.^2);
        
        % Winner is accumulator that finishes first
        [mn, idx] = min([rt0, rt1], [], 2);
        sim_rt(:,r) = mn + Ter;
        sim_choice(:,r) = idx - 1;
        
    elseif strcmp(model_type, 'LNR')
        % Set LNR-specific defaults
        if ~isfield(p_obs, 'b'), p_obs.b = 0; end
        if ~isfield(p_obs, 'b_val'), p_obs.b_val = 0; end
        
        % Calculate trial-wise means for both accumulators
        m1 = p_obs.a + p_obs.b_val.*(u==1) + p_obs.b.*(muhat-0.5);
        m0 = p_obs.a + p_obs.b_val.*(u==0) + p_obs.b.*((1-muhat)-0.5);
        
        % Sample from lognormal distributions
        rt1 = lognrnd(m1, p_obs.sigma);
        rt0 = lognrnd(m0, p_obs.sigma);
        
        % Winner is accumulator that finishes first
        [mn, idx] = min([rt0, rt1], [], 2);
        sim_rt(:,r) = mn + Ter;
        sim_choice(:,r) = idx - 1;
        
    else
        error('PAM:InvalidModel', 'Unknown model type: %s. Use ''DDM'', ''RDM'', or ''LNR''.', model_type);
    end
end

end
