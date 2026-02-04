function p = utl_wfpt(t, v, a, w,prec)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculates first-passage time density for Wiener diffusion model
%
% INPUTS:
%   t    - Hitting time (e.g., response time in seconds)
%   v    - Drift rate
%   a    - Boundary separation (threshold)
%   w    - Relative starting point (bias) (default: 0.5)
%   prec - Error threshold for numerical precision (default: 1e-4)
%
% OUTPUTS:
%   p - Probability density at the lower barrier
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

% Set default values
if nargin<4; w = .5; end
if nargin<5; prec = 1e-4; end

% Calculate probability density
p = 1/a/a .* exp(-v*a*w - v*v.*t/2);
p = p .* utl_fsw(t/a/a, w, prec./p);

end