function tests = test_pam_dot_task_ddm_config
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
assumeFalse(testCase, isempty(which('ddm_hgf_config')), ...
    'PAM DDM is not on the MATLAB path.');
assumeFalse(testCase, isempty(which('tapas_align_priors')), ...
    'TAPAS is not on the MATLAB path.');
end

function testReducedModelsHaveExactFreeSlopeSet(testCase)
cases = { ...
    'ddm_null', [false, false, false]; ...
    'ddm_w',    [true,  false, false]; ...
    'ddm_a',    [false, true,  false]; ...
    'ddm_v',    [false, false, true]; ...
    'ddm_full', [true,  true,  true]};

for k = 1:size(cases, 1)
    c = pam_dot_task_ddm_config(cases{k, 1});
    actual = [c.b_wsa, c.b_asa, c.b_vsa] > 0;
    verifyEqual(testCase, actual, cases{k, 2}, ...
        sprintf('Unexpected free slopes for %s', cases{k, 1}));
end
end

function testUnknownModelFails(testCase)
verifyError(testCase, @() pam_dot_task_ddm_config('ddm_typo'), ...
    'pam:dot_task:UnknownDDMModel');
end
