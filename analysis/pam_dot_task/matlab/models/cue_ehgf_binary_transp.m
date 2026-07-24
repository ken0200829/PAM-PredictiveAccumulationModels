function [pvec, pstruct] = cue_ehgf_binary_transp(r, ptrans)
%CUE_EHGF_BINARY_TRANSP Use the standard binary eHGF transformation.
%
% The two cue streams share one standard eHGF parameter vector, so no new
% parameter transformation is introduced.

[pvec, pstruct] = tapas_ehgf_binary_transp(r, ptrans);
end
