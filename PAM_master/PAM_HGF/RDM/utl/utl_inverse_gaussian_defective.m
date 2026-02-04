function [probs] = utl_inverse_gaussian_defective(x, drift_pdf, drift_cdf, threshold1, threshold2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculates joint probability density for Racing Diffusion Model
%
% INPUTS:
%   x          - Response time
%   drift_pdf  - Drift rate of winning accumulator
%   drift_cdf  - Drift rate of losing accumulator
%   threshold1 - Decision threshold of winning accumulator
%   threshold2 - Decision threshold of losing accumulator
%
% OUTPUTS:
%   probs - Joint probability that winning accumulator finishes at time x
%           while losing accumulator has not yet finished
%
% COMPUTATION:
% The joint probability is calculated as the product of:
% - PDF of winning accumulator at time x (inverse Gaussian/Wald distribution)
% - Survival function (1 - CDF) of losing accumulator at time x
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

% Calculate PDF of winning accumulator (inverse Gaussian/Wald distribution)
pdf = (threshold1) ./ sqrt(2*pi*(x.^3)) .* exp(-0.5 * ((drift_pdf.*(x) - threshold1).^2) ./ (x));

% Calculate CDF of losing accumulator
cdf = normcdf(((drift_cdf.*(x)) - threshold2) ./ sqrt(x)) + ...
    exp(2.*drift_cdf.*threshold2) .* normcdf((-(drift_cdf.*(x)) - threshold2) ./ sqrt(x));

% Joint probability: winner finishes AND loser has not finished
probs = pdf.*(1-cdf);

