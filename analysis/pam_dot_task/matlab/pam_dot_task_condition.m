function condition = pam_dot_task_condition(filename)
%PAM_DOT_TASK_CONDITION Resolve the counterbalancing rules from a CSV name.
%
% The filename prefix is the only accepted source of condition metadata.
% Unknown prefixes fail loudly; no condition is guessed from the data.

arguments
    filename {mustBeTextScalar}
end

[~, stem, ~] = fileparts(char(filename));
stem = string(stem);

names = ["normal_cb", "reverse_cb", "normal", "reverse"];
stimulus_reversed = [false, true, false, true];
white_keys = ["f", "j", "j", "f"];

matched = false;
for k = 1:numel(names)
    if startsWith(stem, names(k) + "_dot_task_")
        condition = struct( ...
            'name', names(k), ...
            'stimulus_reversed', stimulus_reversed(k), ...
            'white_key', white_keys(k), ...
            'black_key', opposite_key(white_keys(k)));
        matched = true;
        break
    end
end

if ~matched
    error('pam:dot_task:UnknownCondition', ...
        'Unknown condition prefix in filename: %s', stem);
end
end

function key = opposite_key(white_key)
if white_key == "f"
    key = "j";
elseif white_key == "j"
    key = "f";
else
    error('pam:dot_task:InvalidWhiteKey', ...
        'White key must be f or j, received: %s', white_key);
end
end
