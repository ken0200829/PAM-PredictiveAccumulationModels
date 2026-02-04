function K = utl_ks(t, w, prec)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Determines number of terms needed for series expansion in density calculation
%
% INPUTS:
%   t    - Normalized time (t/a^2)
%   w    - Relative starting point
%   prec - Precision threshold
%
% OUTPUTS:
%   K - Number of terms required for the series
%
% REFERENCE:
% Implementation based on Gondan M, Blurton S, Kesselmeier M (2014). Even faster and even more
% accurate first-passage time densities and distributions for the Wiener diffusion model.
% J. Math. Psychol., 60, 20-22. https://doi.org/10.1016/j.jmp.2014.05.002
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2024 Antonino Visalli
%
% This file is part of the PAM toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% First estimate
K1 = (sqrt(2*t) - w)/2;
K2 = K1;

% Second estimate based on precision requirement
u_eps = min(-1, log(2.*pi.*t.*t.*prec.*prec));
arg = -t .* (u_eps - sqrt(-2.*u_eps - 2));
K2(arg > 0) = 1/2 .* sqrt(arg) - w/2;

% Return maximum of estimates
K = ceil(max([K1 K2]));
end