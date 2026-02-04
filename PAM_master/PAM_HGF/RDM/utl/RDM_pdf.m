function [pdf] = RDM_pdf(x,drift_pdf,threshold)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculates probability density function for a single accumulator in RDM
%
% INPUTS:
%   x         - Response time
%   drift_pdf - Drift rate of accumulator
%   threshold - Decision threshold of accumulator
%
% OUTPUTS:
%   pdf - Probability density at time x (inverse Gaussian/Wald distribution)
%
% COMPUTATION:
% This function calculates the first-passage time density for a single accumulator
% following an inverse Gaussian (Wald) distribution, which characterizes the racing
% diffusion process.
%
% REFERENCE:
% Tillman G, Van Zandt T, Logan GD (2020). Sequential sampling models without random 
% between-trial variability: The racing diffusion model of speeded decision making. 
% Psychonomic Bulletin & Review, 27(5), 911-936. https://doi.org/10.3758/s13423-020-01719-6
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2024 Antonino Visalli
%
% This file is part of the PAM toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate inverse Gaussian (Wald) PDF
pdf = (threshold ) ./ sqrt(2*pi*(x.^3)) .* exp(-0.5 * ((drift_pdf*(x) - threshold).^2) ./ (x));
