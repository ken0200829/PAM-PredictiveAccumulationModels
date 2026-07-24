function [pvec, pstruct] = ddm_hgf_coherence_transp(r, ptrans)
%DDM_HGF_COHERENCE_TRANSP Transform the nested coherence DDM parameters.

y = r.y(:, 1);
y(r.irr) = [];

pvec = NaN(1, numel(ptrans));
pstruct = struct;

pvec(1) = exp(ptrans(1));
pstruct.a_a = pvec(1);
pvec(2) = exp(ptrans(2));
pstruct.a_v = pvec(2);
pvec(3) = 2 / (1 + exp(-ptrans(3))) - 1;
pstruct.b_w = pvec(3);
pvec(4) = ptrans(4);
pstruct.b_a = pvec(4);
pvec(5) = ptrans(5);
pstruct.b_v = pvec(5);
pvec(6) = ptrans(6);
pstruct.b_c = pvec(6);

if isempty(y) || all(isnan(y))
    pvec(7) = ptrans(7);
else
    pvec(7) = min(y) / (1 + exp(-ptrans(7)));
end
pstruct.Ter = pvec(7);
end
