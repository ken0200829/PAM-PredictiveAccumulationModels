function environment = setup_pam_dot_task_environment(project_root)
%SETUP_PAM_DOT_TASK_ENVIRONMENT Initialize pinned PAM/TAPAS/SPM12 paths.

arguments
    project_root {mustBeTextScalar} = fileparts(fileparts(fileparts( ...
        fileparts(mfilename('fullpath')))))
end

project_root = string(project_root);
analysis_root = fullfile(project_root, 'analysis', 'pam_dot_task');
tapas_root = fullfile(analysis_root, 'external', 'tapas');
spm_root = fullfile(analysis_root, 'external', 'spm12');
pam_root = fullfile(project_root, 'PAM_master');
matlab_root = fullfile(analysis_root, 'matlab');

required_directories = [tapas_root, spm_root, pam_root, matlab_root];
for k = 1:numel(required_directories)
    if ~isfolder(required_directories(k))
        error('pam:environment:MissingDirectory', ...
            'Required directory is missing: %s', required_directories(k));
    end
end

addpath(tapas_root);
tapas_init('HGF');
addpath(spm_root);
addpath(genpath(pam_root));
addpath(matlab_root);
addpath(fullfile(matlab_root, 'models'));
addpath(fullfile(matlab_root, 'tests'));

assert_resolves_within('tapas_fitModel', tapas_root);
assert_resolves_within('tapas_ehgf_binary', tapas_root);
assert_resolves_within('spm_BMS_gibbs', spm_root);
assert_resolves_within('spm_BMS', spm_root);
assert_resolves_within('ddm_hgf', pam_root);
assert_resolves_within('cue_ehgf_binary', matlab_root);
assert_resolves_within('ddm_hgf_coherence', matlab_root);

environment = pam_dot_task_dependency_audit;
environment.project_root = project_root;
environment.tapas_root = tapas_root;
environment.spm_root = spm_root;
environment.pam_root = pam_root;
end

function assert_resolves_within(function_name, expected_root)
matches = which(function_name, '-all');
if isempty(matches)
    error('pam:environment:MissingFunction', ...
        'Function is not available: %s', function_name);
end
if ischar(matches)
    matches = cellstr(matches);
end
resolved = string(matches{1});
if ~startsWith(resolved, string(expected_root))
    error('pam:environment:PathCollision', ...
        '%s resolves to %s instead of %s.', ...
        function_name, resolved, expected_root);
end
end
