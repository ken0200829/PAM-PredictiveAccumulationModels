function [logp, yhat, res] = ddm_hgf(r, infStates, ptrans)
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
% This decision model uses the Wiener first-passage time (WFPT) to calculate response probabilities.
% Trial-wise DDM parameters (drift rate v, boundary separation a, starting point w) are modulated
% by perceptual beliefs from the HGF perceptual model.
%
% The WFPT densities are computed using the method from Gondan M, Blurton S, Kesselmeier M (2014). 
% Even faster and even more accurate first-passage time densities and distributions for the Wiener 
% diffusion model. J. Math. Psychol., 60, 20-22. https://doi.org/10.1016/j.jmp.2014.05.002
% via the utl_wfpt function.
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
a_a = exp(ptrans(1));
a_v = exp(ptrans(2));
b_w = 2/(1+exp(-ptrans(3)))-1;
b_a = ptrans(4);
b_v = ptrans(5);


% Initialize returned log-probabilities, predictions,
% and residuals as NaNs so that NaN is returned for all
% irregualar trials
n = size(infStates,1);
logp = NaN(n,1);
yhat = NaN(n,1); % not used
res  = NaN(n,1); % not used

% Weed irregular trials out from inferred states, responses, and inputs
mu1hat = infStates(:,1,1);
mu1hat(r.irr) = [];

rt = r.y(:,1);
rt(r.irr) = [];

resp = r.y(:,2);
resp(r.irr)  = [];

% Fitting the non-decision time with the minimum value of estimated
% non-decision time
Ter = min(rt)/(1+exp(-ptrans(6)));
rt = max(eps,rt-Ter);

% Extract the trial list and remove the irregular trials
u = r.u(:,1);
u(r.irr) = [];

% Calculate trial-wise starting point
w = .5 + b_w.*(mu1hat - .5);

% Calculate trial-wise boundary separation
precision = tapas_sgm(1./(mu1hat.*(1-mu1hat))-4,1)-.5;
a = a_a + b_a.*(precision);

% Calculate trial-wise drift rate
v = u.*(a_v + b_v.*(mu1hat - .5)) ...
    - (1-u).*(a_v + b_v.*((1-mu1hat) - .5));

% Calculate predicted log-likelihood
logp_reg = NaN(length(u),1);
for ntrial = 1:length(u)
    if rt(ntrial)>0
% Compute WFPT probability for both boundaries
        P = utl_wfpt(rt(ntrial), -v(ntrial), a(ntrial), (1-w(ntrial)))*resp(ntrial) + ...
            utl_wfpt(rt(ntrial), v(ntrial), a(ntrial), w(ntrial))*(1-resp(ntrial));

        if P>0
            logp_reg(ntrial) = log(P+eps);
        else
            logp_reg(ntrial) = NaN;
        end
    end
end

% Assign log-probabilities to regular trials
reg = ~ismember(1:n,r.irr);
logp(reg) = logp_reg;
return;
