function [traj, infStates] = cue_ehgf_binary(r, p, varargin)
%CUE_EHGF_BINARY Evaluate independent white/red HGF states in one objective.
%
% Input columns:
%   r.u(:,1) - binary stimulus category (black=0, white=1)
%   r.u(:,2) - active cue (red=0, white=1)
%
% A single shared parameter vector p is passed to two calls of the official
% tapas_ehgf_binary function. Each call sees only presentations of its cue.
% The active cue's pre-outcome state is then restored to the original global
% trial position. This retains joint MAP estimation because this function is
% evaluated inside the same tapas_fitModel objective as the DDM likelihood.

if ~isempty(varargin) && strcmp(varargin{1}, 'trans')
    p = cue_ehgf_binary_transp(r, p);
end

validate_inputs(r);

stimulus = r.u(:, r.c_prc.stimulus_column);
cue_white = r.u(:, r.c_prc.cue_column);
white_index = find(cue_white == 1);
red_index = find(cue_white == 0);
n_trials = size(r.u, 1);

if isempty(white_index) || isempty(red_index)
    error('pam:cue_hgf:MissingCueStream', ...
        'Both white and red cue streams must contain at least one trial.');
end

r_white = cue_subproblem(r, stimulus, white_index);
r_red = cue_subproblem(r, stimulus, red_index);

[white_traj, white_states] = tapas_ehgf_binary(r_white, p);
[red_traj, red_states] = tapas_ehgf_binary(r_red, p);

n_levels = size(white_states, 2);
n_state_types = size(white_states, 3);
if size(red_states, 2) ~= n_levels || ...
        size(red_states, 3) ~= n_state_types
    error('pam:cue_hgf:StateShapeMismatch', ...
        'White and red HGF streams returned incompatible state arrays.');
end

infStates = NaN(n_trials, n_levels, n_state_types);
infStates(white_index, :, :) = white_states;
infStates(red_index, :, :) = red_states;
if any(~isfinite(infStates(:, 1, 1)))
    error('pam:cue_hgf:MissingActiveBelief', ...
        'Every trial must receive a finite active-cue prior belief.');
end

% Preserve the standard active-cue trajectory fields expected downstream.
traj = stitch_active_trajectories( ...
    n_trials, white_index, red_index, white_traj, red_traj);

% Retain both uncompressed cue streams for audit and plotting.
traj.active_cue_white = cue_white;
traj.cue.white.index = white_index;
traj.cue.white.traj = white_traj;
traj.cue.red.index = red_index;
traj.cue.red.traj = red_traj;
traj.cue.time_axis = 'cue_presentation_count';
traj.cue.shared_parameters = true;
traj.cue.inactive_policy = 'freeze';
end

function validate_inputs(r)
if ~isfield(r, 'u') || size(r.u, 2) < 2
    error('pam:cue_hgf:InputColumns', ...
        'cue_ehgf_binary requires stimulus and cue columns in r.u.');
end
if ~isfield(r, 'c_prc') || ...
        ~isfield(r.c_prc, 'stimulus_column') || ...
        ~isfield(r.c_prc, 'cue_column')
    error('pam:cue_hgf:MissingConfig', ...
        'Use cue_ehgf_binary_config with cue_ehgf_binary.');
end

stimulus = r.u(:, r.c_prc.stimulus_column);
cue = r.u(:, r.c_prc.cue_column);
if any(~isfinite(stimulus)) || any(~ismember(stimulus, [0, 1]))
    error('pam:cue_hgf:InvalidStimulus', ...
        'Stimulus category must be finite and binary on every trial.');
end
if any(~isfinite(cue)) || any(~ismember(cue, [0, 1]))
    error('pam:cue_hgf:InvalidCue', ...
        'Cue indicator must be finite and binary on every trial.');
end
if isfield(r, 'ign') && ~isempty(r.ign)
    error('pam:cue_hgf:IgnoredTrialsUnsupported', ...
        ['The dot-task adapter must keep all inputs finite. Mask response ' ...
         'likelihood rows in y instead of marking trials ignored in u.']);
end
if isfield(r.c_prc, 'irregular_intervals') && r.c_prc.irregular_intervals
    error('pam:cue_hgf:IrregularIntervalsUnsupported', ...
        'The frozen non-active-cue model uses cue presentation count.');
end
end

function r_sub = cue_subproblem(r, stimulus, index)
r_sub = r;
r_sub.u = stimulus(index);
if isfield(r, 'y') && ~isempty(r.y)
    r_sub.y = r.y(index, :);
end
r_sub.ign = [];
if isfield(r, 'irr')
    r_sub.irr = find(ismember(index, r.irr));
end
end

function traj = stitch_active_trajectories( ...
        n_trials, white_index, red_index, white_traj, red_traj)
traj = struct;
fields = fieldnames(white_traj);
if ~isequal(fields, fieldnames(red_traj))
    error('pam:cue_hgf:TrajectoryFieldsMismatch', ...
        'White and red HGF streams returned different trajectory fields.');
end

for k = 1:numel(fields)
    field = fields{k};
    white_values = white_traj.(field);
    red_values = red_traj.(field);
    if size(white_values, 1) ~= numel(white_index) || ...
            size(red_values, 1) ~= numel(red_index)
        error('pam:cue_hgf:TrajectoryLengthMismatch', ...
            'Trajectory field %s does not match cue trial counts.', field);
    end
    active_values = NaN(n_trials, size(white_values, 2));
    active_values(white_index, :) = white_values;
    active_values(red_index, :) = red_values;
    traj.(field) = active_values;
end
end
