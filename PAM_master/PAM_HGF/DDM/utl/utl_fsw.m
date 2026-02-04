function f = utl_fsw(t, w, prec)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculates density at the lower barrier using one-parameter form
%
% INPUTS:
%   t    - Normalized time (t/a^2)
%   w    - Relative starting point
%   prec - Precision threshold for series truncation
%
% OUTPUTS:
%   f - Density at the lower barrier
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

% Determine number of terms needed for the series
K = utl_ks(t, w, prec);

% Initialize density
f = zeros(1,length(t));

% Calculate density using series expansion
if(K > 0 && isfinite(K))
    for k = K:-1:1
        f = (w+2*k) * exp(-(w+2*k) * (w+2*k)/2./t) + ...
            (w-2*k) * exp(-(w-2*k) * (w-2*k)/2./t) + f;
    end
    f= (1./sqrt(2*pi.*t.*t.*t) .* (f + w .* exp(-w*w/2./t)));
end


end