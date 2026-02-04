function [probs] = utl_lnr_pdf(x,mu1,mu2,sigma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculates joint probability density for Lognormal Race Model
%
% INPUTS:
%   x     - Response time
%   mu1   - Mean parameter of winning accumulator (log-scale)
%   mu2   - Mean parameter of losing accumulator (log-scale)
%   sigma - Standard deviation of lognormal distribution
%
% OUTPUTS:
%   probs - Joint probability that winning accumulator finishes at time x
%           while losing accumulator has not yet finished
%
% COMPUTATION:
% The joint probability is calculated as the product of:
% - PDF of winning accumulator at time x
% - Survival function (1 - CDF) of losing accumulator at time x
%
% REFERENCE:
% Rouder JN, Province JM, Morey RD, Gomez P, Heathcote A (2015). The lognormal race: 
% A cognitive-process model of choice and latency with desirable psychometric properties. 
% Psychometrika, 80(2), 491-513. https://doi.org/10.1007/s11336-013-9396-3
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2024 Antonino Visalli
%
% This file is part of the PAM toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate probability density of winning accumulator
pdf = lognpdf(x,mu1,sigma);

% Calculate survival function of losing accumulator
survival = logncdf(x,mu2,sigma,'upper');

% Joint probability: winner finishes AND loser has not finished
probs = pdf.*survival;