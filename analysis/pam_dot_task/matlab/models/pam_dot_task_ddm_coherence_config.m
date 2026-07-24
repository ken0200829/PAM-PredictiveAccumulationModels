function c = pam_dot_task_ddm_coherence_config(model_id)
%PAM_DOT_TASK_DDM_COHERENCE_CONFIG Registry for coherence-extended models.
%
% All models include a free coherence drift slope b_c. Belief-related slopes
% are reduced exactly as in the official PAM comparison.

arguments
    model_id {mustBeTextScalar}
end

model_id = lower(string(model_id));
c = ddm_hgf_coherence_config;

switch model_id
    case "ddm_c"
        c.b_wsa = 0;
        c.b_asa = 0;
        c.b_vsa = 0;
        expected_belief = strings(0, 1);
    case "ddm_w_c"
        c.b_wsa = 4;
        c.b_asa = 0;
        c.b_vsa = 0;
        expected_belief = "b_w";
    case "ddm_a_c"
        c.b_wsa = 0;
        c.b_asa = 4;
        c.b_vsa = 0;
        expected_belief = "b_a";
    case "ddm_v_c"
        c.b_wsa = 0;
        c.b_asa = 0;
        c.b_vsa = 4;
        expected_belief = "b_v";
    case "ddm_full_c"
        c.b_wsa = 4;
        c.b_asa = 4;
        c.b_vsa = 4;
        expected_belief = ["b_w"; "b_a"; "b_v"];
    otherwise
        error('pam:coherence_ddm:UnknownModel', ...
            'Unknown coherence DDM model ID: %s', model_id);
end

c.b_csa = 4;
c.model = char("dot_task:" + model_id + ":coherence_drift_nested");
c = tapas_align_priors(c);

belief_names = ["b_w"; "b_a"; "b_v"];
belief_variances = [c.b_wsa; c.b_asa; c.b_vsa];
actual_belief = belief_names(belief_variances > 0);
if ~isequal(actual_belief, expected_belief) || c.b_csa <= 0
    error('pam:coherence_ddm:RegistryMismatch', ...
        'Model %s has an unexpected free parameter set.', model_id);
end
end
