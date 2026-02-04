function c = RDM_hgf_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Configuration for RDM response model with HGF perceptual model
%
% OUTPUT:
%   c - Configuration structure with priors and model specifications
%
% PRIORS:
% All priors are Gaussian in the space where the quantity is estimated. They are specified by 
% mean and variance (NOT standard deviation). Parameters can be fixed by setting prior variance to 0.
%
% PARAMETERS:
%   a_a    - Intercept of decision threshold
%   b_a    - Slope of prior belief effect on decision threshold
%   a_v    - Intercept of drift rate
%   b_val  - Slope of validity effect (stimulus-response match) on drift rate
%   b_v    - Slope of prior belief effect on drift rate
%   Ter    - Non-decision time (default: not estimated, Termu = -Inf, Tersa = 0)
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

% Config structure
c = struct;

% Model name
c.model = 'racing diffusion model: hgf';

% Intercept of decision threshold "a"
c.a_amu = 0;
c.a_asa = 4;

% Muhat slope effect for "a"
c.b_amu = 0;
c.b_asa = 4;

% Intercept of drift rate "v"
c.a_vmu = 0;
c.a_vsa = 4;

% Effect of validity (resp = input) on the drift 
c.b_valmu = 0;
c.b_valsa = 4;

% Muhat slope effect for "v"
c.b_vmu = 0;
c.b_vsa = 4;

% Non decision time
c.Termu = -Inf;
c.Tersa = 0;


% Gather prior settings in vectors
c.priormus = [
    c.a_amu,...
    c.b_amu,...
    c.a_vmu,...
    c.b_valmu,...
    c.b_vmu,...
    c.Termu,...
         ];

c.priorsas = [
    c.a_asa,...
    c.b_asa,...
    c.a_vsa,...
    c.b_valsa,...
    c.b_vsa,...
    c.Tersa,...
     ];

% Model filehandle
c.obs_fun = @RDM_hgf;

% Handle to function that transforms observation parameters to their native space
% from the space they are estimated in
c.transp_obs_fun = @RDM_hgf_transp;

return;
