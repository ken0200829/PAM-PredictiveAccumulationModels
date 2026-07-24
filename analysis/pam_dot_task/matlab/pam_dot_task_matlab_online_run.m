function manifest = pam_dot_task_matlab_online_run(selection, output_directory)
%PAM_DOT_TASK_MATLAB_ONLINE_RUN Export Python-parity fixtures in MATLAB Online.
%
%   MANIFEST = PAM_DOT_TASK_MATLAB_ONLINE_RUN(SELECTION) initializes the
%   bundled, pinned dependencies and exports the requested fixture JSON files
%   to matlab_fixture_output/. The 3-second response deadline and the
%   deterministic 380-trial fixture design are fixed in the exported sources.
%
%   Run lightweight fixtures first:
%       pam_dot_task_matlab_online_run({'hgf', 'wfpt', 'ddm'})
%
%   Then run each heavier fixture separately, so an interrupted MATLAB Online
%   session never discards completed JSON files:
%       pam_dot_task_matlab_online_run({'bms'})
%       pam_dot_task_matlab_online_run({'simulation'})
%       pam_dot_task_matlab_online_run({'ppc'})
%       pam_dot_task_matlab_online_run({'joint'})
%
%   This runner neither reads participant CSVs nor writes PAM_master, TAPAS,
%   or SPM12. Download the resulting JSON files and place them in
%   analysis/pam_dot_task_python/fixtures/matlab/ before running the Python
%   fixture-comparison tests.

if nargin < 1 || isempty(selection)
    selection = {'hgf', 'wfpt', 'ddm'};
end

project_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
if nargin < 2 || isempty(output_directory)
    output_directory = fullfile(project_root, 'matlab_fixture_output');
end

environment = setup_pam_dot_task_environment(project_root); %#ok<NASGU>
manifest = pam_dot_task_export_fixtures(output_directory, selection);
fprintf('\nMATLAB fixture files are ready in:\n%s\n', output_directory);
end
