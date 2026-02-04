function [pvec, pstruct] = ddm_hgf_transp(r, ptrans)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Transforms decision model parameters to their native space
%
% INPUTS:
%   r      - Response structure
%   ptrans - Parameter vector in transformed space
%
% OUTPUTS:
%   pvec    - Parameter vector in native space
%   pstruct - Parameter structure in native space
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


% Extract response times
try
    y = r.y(:,1);
    y(r.irr) = [];
catch
    y =nan;  % for "tapas_bayesian_parameter_average"
end

% Initialize an empty array to store the transformed values
pvec    = NaN(1,length(ptrans));
pstruct = struct;

% Apply exponential transformation to "a_a"
pvec(1)      = exp(ptrans(1));
pstruct.a_a   = pvec(1);

% Apply exponential transformation to "a_v"
pvec(2)      = exp(ptrans(2));
pstruct.a_v   = pvec(2);

% Transform "b_w" values in the range (-1,1)
pvec(3)      = 2/(1+exp(-ptrans(3)))-1;
pstruct.b_w   = pvec(3);

% Untrasformed "b_a"
pvec(4)      = ptrans(4);
pstruct.b_a   = pvec(4);

% Untrasformed "b_v"
pvec(5)      = ptrans(5);
pstruct.b_v   = pvec(5);

% Transform "Ter" using a sigmoid centered at the minimum value of y
if ~isnan(y)
    pvec(6)      = min(y)/(1+exp(-ptrans(6)));
    pstruct.Ter = pvec(6);
else % for "tapas_bayesian_parameter_average"
    pvec(6) = ptrans(6);
    pstruct.Ter = pvec(6);
end

return;