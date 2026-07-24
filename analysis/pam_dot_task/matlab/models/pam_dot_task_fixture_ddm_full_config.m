function c = pam_dot_task_fixture_ddm_full_config
%PAM_DOT_TASK_FIXTURE_DDM_FULL_CONFIG No-argument config for fixture MAP runs.
%
% TAPAS config function handles are called without arguments. The public
% registry deliberately requires an explicit model ID, so this narrow wrapper
% fixes the synthetic parity fixture to the official full DDM reduction.

c = pam_dot_task_ddm_config("ddm_full");
end
