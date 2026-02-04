function [logp, yhat, res] = RDM_hgf(r, infStates, ptrans)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculates the log-probability of response model parameters given perceptual states
%
% INPUTS:
%   r         - Response structure containing behavior (RTs, choices) and task inputs
%   infStates - Inferred perceptual states from HGF perceptual model
%   ptrans    - Parameter vector in transformed space
%
% OUTPUTS:
%   logp - Log-probabilities of responses
%   yhat - Predictions (not used)
%   res  - Residuals (not used)
%
% PERCEPTUAL MODEL:
% Hierarchical Gaussian Filter for binary inputs introduced in
% Mathys C, Daunizeau J, Friston, KJ, and Stephan KE. (2011). A Bayesian foundation
% for individual learning under uncertainty. Frontiers in Human Neuroscience, 5:39.
% https://doi.org/10.3389/fnhum.2011.00039
%
% RESPONSE MODEL:
% Racing Diffusion Model (RDM) introduced in Tillman G, Van Zandt T, Logan GD (2020). 
% Sequential sampling models without random between-trial variability: The racing diffusion 
% model of speeded decision making. Psychonomic Bulletin & Review, 27(5), 911-936.
% https://doi.org/10.3758/s13423-020-01719-6
%
% The structure and methodologies of this file are adapted from the HGF Toolbox,
% part of the TAPAS software collection: Frässle et al. (2021). TAPAS: An Open-Source 
% Software Package for Translational Neuromodeling and Computational Psychiatry. 
% Frontiers in Psychiatry, 12:680811. https://doi.org/10.3389/fpsyt.2021.680811
% https://www.translationalneuromodeling.org/tapas
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2024 Antonino Visalli
%
% This file is part of the PAM toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Transform parameters to their native space
a_a   = exp(ptrans(1));
b_a   = ptrans(2);
a_v   = exp(ptrans(3));
b_val = exp(ptrans(4));
b_v   = ptrans(5);

% Initialize returned log-probabilities, predictions, and residuals
n = size(infStates,1);
logp = NaN(n,1);
yhat = NaN(n,1); 
res  = NaN(n,1); 

% Filter irregular trials
reg = ~ismember(1:n, r.irr);

% Extract regular data
mu1hat = infStates(reg, 1, 1);
rt     = r.y(reg, 1);
resp   = r.y(reg, 2);
u      = r.u(reg, 1);

% Calculate non-decision time based on the minimum RT
Ter_val = min(rt) / (1 + exp(-ptrans(6)));
rt = rt - Ter_val;
rt = max(eps, rt); % Ensure positive RTs

% Calculate trial-wise thresholds for both accumulators
a_c1 = a_a + b_a .* (mu1hat - 0.5);
a_c0 = a_a + b_a .* ((1 - mu1hat) - 0.5);

% Calculate trial-wise drift rates for both accumulators
drift_c1 = a_v + b_val .* (u == 1) + b_v .* (mu1hat - 0.5);
drift_c0 = a_v + b_val .* (u == 0) + b_v .* ((1 - mu1hat) - 0.5);

% Ensure positive values for Wald distribution
min_val = 1e-6;
a_c1 = max(a_c1, min_val); 
a_c0 = max(a_c0, min_val);
drift_c1 = max(drift_c1, min_val); 
drift_c0 = max(drift_c0, min_val);

% Create logical masks for response choices
is_resp1 = (resp == 1);
is_resp0 = (resp == 0);

% Initialize vectors for winning and losing accumulator parameters
drift_pdf = NaN(size(rt)); % Drift of the winner
drift_cdf = NaN(size(rt)); % Drift of the loser
a_pdf     = NaN(size(rt)); % Threshold of the winner
a_cdf     = NaN(size(rt)); % Threshold of the loser

% Map values based on actual choice (vectorized indexing)
% Case: Response = 1
drift_pdf(is_resp1) = drift_c1(is_resp1);
drift_cdf(is_resp1) = drift_c0(is_resp1);
a_pdf(is_resp1)     = a_c1(is_resp1);
a_cdf(is_resp1)     = a_c0(is_resp1);

% Case: Response = 0
drift_pdf(is_resp0) = drift_c0(is_resp0);
drift_cdf(is_resp0) = drift_c1(is_resp0);
a_pdf(is_resp0)     = a_c0(is_resp0);
a_cdf(is_resp0)     = a_c1(is_resp0);

% Calculate likelihood using inverse Gaussian (defective)
P = utl_inverse_gaussian_defective(rt, drift_pdf, drift_cdf, a_pdf, a_cdf);
P(P <= 0 | isnan(P)) = eps;
logp(reg) = log(P);

return;
