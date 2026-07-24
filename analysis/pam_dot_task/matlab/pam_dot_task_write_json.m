function pam_dot_task_write_json(file_path, value)
%PAM_DOT_TASK_WRITE_JSON Write a struct to JSON at full double precision.
%
%   Built-in JSONENCODE is not used because it rounds numeric output and is
%   unavailable in Octave. Parity fixtures must round-trip every bit of the
%   double, so all numbers are written with %.17g, which is the shortest
%   format guaranteed to reproduce an IEEE double exactly.
%
%   Supported values: struct (nested), numeric scalar/vector/matrix, logical,
%   char, and cell arrays of char. Matrices are written row-major as nested
%   arrays. NaN and +/-Inf are written as strings because JSON has no literal
%   for them; the Python reader converts them back.

file_id = fopen(file_path, 'w');
if file_id < 0
    error('pam:fixture:open', 'Cannot open %s for writing.', file_path);
end
cleaner = onCleanup(@() fclose(file_id));
fprintf(file_id, '%s\n', serialize_value(value, 0));
end

function text = serialize_value(value, depth)
if isstruct(value)
    text = serialize_struct(value, depth);
elseif ischar(value)
    text = serialize_string(value);
elseif iscell(value)
    text = serialize_cell(value, depth);
elseif islogical(value)
    text = serialize_numeric(double(value));
elseif isnumeric(value)
    text = serialize_numeric(value);
else
    error('pam:fixture:type', 'Unsupported fixture value class: %s.', class(value));
end
end

function text = serialize_struct(value, depth)
if numel(value) ~= 1
    error('pam:fixture:structarray', 'Struct arrays are not supported.');
end
names = fieldnames(value);
pad = repmat(' ', 1, 2 * (depth + 1));
parts = cell(numel(names), 1);
for k = 1:numel(names)
    parts{k} = sprintf('%s%s: %s', pad, serialize_string(names{k}), ...
        serialize_value(value.(names{k}), depth + 1));
end
if isempty(parts)
    text = '{}';
    return
end
closing = repmat(' ', 1, 2 * depth);
text = sprintf('{\n%s\n%s}', strjoin(parts', sprintf(',\n')), closing);
end

function text = serialize_cell(value, depth)
parts = cell(numel(value), 1);
for k = 1:numel(value)
    parts{k} = serialize_value(value{k}, depth + 1);
end
text = sprintf('[%s]', strjoin(parts', ', '));
end

function text = serialize_string(value)
escaped = strrep(value, '\', '\\');
escaped = strrep(escaped, '"', '\"');
escaped = strrep(escaped, sprintf('\n'), '\n');
text = sprintf('"%s"', escaped);
end

function text = serialize_numeric(value)
if isempty(value)
    text = '[]';
elseif isscalar(value)
    text = serialize_scalar(value);
elseif isvector(value)
    text = sprintf('[%s]', strjoin(arrayfun(@serialize_scalar, value(:)', ...
        'UniformOutput', false), ', '));
else
    rows = cell(size(value, 1), 1);
    for k = 1:size(value, 1)
        rows{k} = sprintf('[%s]', strjoin(arrayfun(@serialize_scalar, ...
            value(k, :), 'UniformOutput', false), ', '));
    end
    text = sprintf('[%s]', strjoin(rows', ', '));
end
end

function text = serialize_scalar(value)
if isnan(value)
    text = '"NaN"';
elseif isinf(value) && value > 0
    text = '"Infinity"';
elseif isinf(value)
    text = '"-Infinity"';
else
    text = sprintf('%.17g', value);
end
end
