function [logp, yhat, res] = lnr_hgf(r, infStates, ptrans)
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
% Lognormal Race Model (LNR) introduced in Rouder JN, Province JM, Morey RD, Gomez P,
% Heathcote A (2015). The lognormal race: A cognitive-process model of choice and latency
% with desirable psychometric properties. Psychometrika, 80(2), 491-513.
% https://doi.org/10.1007/s11336-013-9396-3
%
% See also: Heathcote A, Love J (2012). Linear deterministic accumulator models of simple choice.
% Frontiers in Psychology, 3:292. https://doi.org/10.3389/fpsyg.2012.00292
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
a = (ptrans(1));
b_val = (ptrans(2));
b = (ptrans(3));
sigma = exp(ptrans(4));

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
Ter = min(rt)/(1+exp(-ptrans(5)));
rt = max(eps,rt-Ter);

% Extract the trial list and remove the irregular trials
u = r.u(:,1);
u(r.irr) = [];

% Calculate trial-wise drift rates for the two accumulators
mu_c1 = a + b_val.* double(u == 1) + b .*(mu1hat-.5);
mu_c0 = a + b_val.* double(u == 0) + b .*((1 - mu1hat)-.5);

% Calculate predicted log-likelihood
logp_reg = NaN(length(u),1);
for ntrial = 1:length(u)
    if rt(ntrial)>0
        % Assign means to winning and losing accumulators
        mu_pdf = (resp(ntrial) == 1)*mu_c1(ntrial) + (resp(ntrial) == 0) * mu_c0(ntrial);
        mu_cdf = (resp(ntrial) == 1)*mu_c0(ntrial) + (resp(ntrial) == 0) * mu_c1(ntrial);

        % Calculate LNR probability
        P = utl_lnr_pdf(rt(ntrial),mu_pdf,mu_cdf,sigma);

        % Correct if P<0
        P(P<0)=0;

        % Calculate log-likelihood
        if P>0
            logp_reg(ntrial) = log(P+eps);
        else
            logp_reg(ntrial) = NaN;
        end
    else
        logp_reg(ntrial) = NaN;
    end
end

% Assign log-probabilities to regular trials
reg = ~ismember(1:n,r.irr);
logp(reg) = logp_reg;
return;
