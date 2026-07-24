function results = run_pam_dot_task_tests(project_root)
%RUN_PAM_DOT_TASK_TESTS Add project paths and run the MATLAB test suite.

arguments
    project_root {mustBeTextScalar} = fileparts(fileparts(fileparts( ...
        fileparts(fileparts(mfilename('fullpath'))))))
end

project_root = string(project_root);
analysis_root = fullfile(project_root, 'analysis', 'pam_dot_task', 'matlab');
addpath(analysis_root);
addpath(fullfile(analysis_root, 'models'));
addpath(fullfile(project_root, 'PAM_master', 'PAM_HGF', 'DDM'));
addpath(fullfile(project_root, 'PAM_master', 'PAM_HGF', 'DDM', 'utl'));

suite = testsuite(fileparts(mfilename('fullpath')), ...
    'IncludeSubfolders', true);
results = run(suite);
disp(table(results));
assert(all([results.Passed] | [results.Incomplete]), ...
    'pam:dot_task:TestFailure', 'One or more PAM dot-task tests failed.');
end
