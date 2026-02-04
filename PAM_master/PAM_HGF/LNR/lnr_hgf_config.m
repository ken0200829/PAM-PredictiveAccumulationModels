function c = lnr_hgf_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Configuration for LNR response model with HGF perceptual model
%
% OUTPUT:
%   c - Configuration structure with priors and model specifications
%
% PRIORS:
% All priors are Gaussian in the space where the quantity is estimated. They are specified by
% mean and variance (NOT standard deviation). Parameters can be fixed by setting prior variance to 0.
%
% PARAMETERS:
%   a      - Intercept of lognormal mean
%   b_val  - Slope of validity effect (stimulus-response match)
%   b      - Slope of prior belief effect on lognormal mean
%   sigma  - Standard deviation of lognormal distribution
%   Ter    - Non-decision time (default: not estimated, Termu = -Inf, Tersa = 0)
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

% Config structure
c = struct;

% Model name
c.model = 'lnr: hgf';

% Intercept of lognormal mu (resp == input)
c.amu = 0;
c.asa = 4;

% Beta for validity (resp = input)
c.b_valmu = 0;
c.b_valsa = 4;

% Beta for muhat
c.bmu = 0;
c.bsa = 4;

% Standard deviation
c.sigmamu = 0;
c.sigmasa = 4;

% Non decision time
c.Termu = -Inf;
c.Tersa = 0;


% Gather prior settings in vectors
c.priormus = [
    c.amu,...
    c.b_valmu,...
    c.bmu,...
    c.sigmamu,...
    c.Termu,...
    ];

c.priorsas = [
    c.asa,...
    c.b_valsa,...
    c.bsa,...
    c.sigmasa,...
    c.Tersa,...
    ];

% Model filehandle
c.obs_fun = @lnr_hgf;

% Handle to function that transforms observation parameters to their native space
% from the space they are estimated in
c.transp_obs_fun = @lnr_hgf_transp;

return;
