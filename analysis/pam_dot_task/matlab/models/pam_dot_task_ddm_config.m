function c = pam_dot_task_ddm_config(model_id)
%PAM_DOT_TASK_DDM_CONFIG Build tested reductions of the official PAM DDM.
%
% This function never edits ddm_hgf.m. It only fixes unused slope priors to
% zero variance and checks that the resulting free set matches the model ID.

arguments
    model_id {mustBeTextScalar}
end

model_id = lower(string(model_id));
c = ddm_hgf_config;

switch model_id
    case "ddm_null"
        c.b_wsa = 0;
        c.b_asa = 0;
        c.b_vsa = 0;
        expected_free = strings(0, 1);
    case "ddm_w"
        c.b_wsa = 4;
        c.b_asa = 0;
        c.b_vsa = 0;
        expected_free = "b_w";
    case "ddm_a"
        c.b_wsa = 0;
        c.b_asa = 4;
        c.b_vsa = 0;
        expected_free = "b_a";
    case "ddm_v"
        c.b_wsa = 0;
        c.b_asa = 0;
        c.b_vsa = 4;
        expected_free = "b_v";
    case "ddm_full"
        c.b_wsa = 4;
        c.b_asa = 4;
        c.b_vsa = 4;
        expected_free = ["b_w"; "b_a"; "b_v"];
    otherwise
        error('pam:dot_task:UnknownDDMModel', ...
            'Unknown DDM model ID: %s', model_id);
end

c.model = char("dot_task:" + model_id + ":official_binary_ddm");
c = tapas_align_priors(c);

slope_names = ["b_w"; "b_a"; "b_v"];
slope_variances = [c.b_wsa; c.b_asa; c.b_vsa];
actual_free = slope_names(slope_variances > 0);
if ~isequal(actual_free, expected_free)
    error('pam:dot_task:DDMRegistryMismatch', ...
        'Model %s has an unexpected free slope set: %s', ...
        model_id, strjoin(actual_free, ', '));
end
end
