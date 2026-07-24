function c = ddm_hgf_coherence_config
%DDM_HGF_COHERENCE_CONFIG Full nested coherence-extended PAM DDM.

c = struct;
c.model = 'ddm:hgf:coherence_drift_nested';

% Column contract for r.u.  The stimulus column is resolved from the
% perceptual config (cue_ehgf_binary_config.stimulus_column) so both models
% address the same column; only the response-specific coherence column lives
% here.  These are not prior fields and are not swept into priormus/priorsas.
c.coherence_column = 3;

c.a_amu = 0;
c.a_asa = 4;
c.a_vmu = 0;
c.a_vsa = 4;
c.b_wmu = 0;
c.b_wsa = 4;
c.b_amu = 0;
c.b_asa = 4;
c.b_vmu = 0;
c.b_vsa = 4;
c.b_cmu = 0;
c.b_csa = 4;
c.Termu = 0;
c.Tersa = 4;

c.priormus = [c.a_amu, c.a_vmu, c.b_wmu, c.b_amu, ...
    c.b_vmu, c.b_cmu, c.Termu];
c.priorsas = [c.a_asa, c.a_vsa, c.b_wsa, c.b_asa, ...
    c.b_vsa, c.b_csa, c.Tersa];

c.obs_fun = @ddm_hgf_coherence;
c.transp_obs_fun = @ddm_hgf_coherence_transp;
end
