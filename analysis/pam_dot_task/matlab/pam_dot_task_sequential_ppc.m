function ppc = pam_dot_task_sequential_ppc(audit, simulation, spec)
%PAM_DOT_TASK_SEQUENTIAL_PPC Reuse one batch across time-resolved windows.

arguments
    audit table
    simulation struct
    spec struct = struct()
end

if isempty(fieldnames(spec))
    spec = pam_dot_task_ppc_spec(audit);
end

required_audit = ["rt_seconds_raw", "choice_white", "stimulus_category"];
if ~all(ismember(required_audit, string(audit.Properties.VariableNames)))
    error('pam:ppc:MissingObservedColumns', ...
        'Audit table lacks observed response columns.');
end
if ~isfield(simulation, 'rt') || ~isfield(simulation, 'choice')
    error('pam:ppc:MissingSimulationArrays', ...
        'simulation must contain rt and choice arrays.');
end
if ~isequal(size(simulation.rt), size(simulation.choice)) || ...
        size(simulation.rt, 2) ~= height(audit)
    error('pam:ppc:SimulationShape', ...
        'Simulation arrays must be replicate-by-trial with 380 columns.');
end

n_replicates = size(simulation.rt, 1);
n_windows = numel(spec.windows);
n_statistics = numel(spec.statistics);
observed = NaN(n_windows, n_statistics);
replicated = NaN(n_replicates, n_windows, n_statistics);
n_valid = zeros(n_windows, 1);

physical_valid = isfinite(audit.rt_seconds_raw) & ...
    audit.rt_seconds_raw >= 0.15 & audit.rt_seconds_raw <= 3.0 & ...
    isfinite(audit.choice_white);

for window_index = 1:n_windows
    index = spec.windows(window_index).indices;
    index = index(physical_valid(index));
    n_valid(window_index) = numel(index);
    if n_valid(window_index) < spec.min_valid_trials
        continue
    end

    observed(window_index, :) = response_statistics( ...
        audit.rt_seconds_raw(index)', audit.choice_white(index)', ...
        audit.stimulus_category(index)');
    for replicate = 1:n_replicates
        values = response_statistics( ...
            simulation.rt(replicate, index), ...
            simulation.choice(replicate, index), ...
            audit.stimulus_category(index)');
        replicated(replicate, window_index, :) = reshape(values, 1, 1, []);
    end
end

rows = n_windows * n_statistics;
window_id = strings(rows, 1);
family = strings(rows, 1);
statistic = strings(rows, 1);
valid_trials = zeros(rows, 1);
observed_value = NaN(rows, 1);
predictive_median = NaN(rows, 1);
predictive_lower = NaN(rows, 1);
predictive_upper = NaN(rows, 1);
predictive_percentile = NaN(rows, 1);
tail_probability_two_sided = NaN(rows, 1);

replicate_matrix = reshape(permute(replicated, [1, 3, 2]), ...
    n_replicates, rows);
observed_vector = reshape(observed', rows, 1);
center = NaN(rows, 1);
scale = NaN(rows, 1);

row = 0;
for window_index = 1:n_windows
    for statistic_index = 1:n_statistics
        row = row + 1;
        window_id(row) = spec.windows(window_index).id;
        family(row) = spec.windows(window_index).family;
        statistic(row) = spec.statistics(statistic_index);
        valid_trials(row) = n_valid(window_index);
        observed_value(row) = observed(window_index, statistic_index);

        draws = replicate_matrix(:, row);
        draws = draws(isfinite(draws));
        if isempty(draws) || ~isfinite(observed_value(row))
            continue
        end
        predictive_median(row) = empirical_quantile(draws, 0.5);
        predictive_lower(row) = empirical_quantile(draws, spec.point_interval(1));
        predictive_upper(row) = empirical_quantile(draws, spec.point_interval(2));
        predictive_percentile(row) = mean(draws <= observed_value(row));
        tail_probability_two_sided(row) = min(1, 2 * min( ...
            predictive_percentile(row), 1 - predictive_percentile(row)));
        center(row) = predictive_median(row);
        scale(row) = std(draws);
    end
end

usable = isfinite(observed_vector) & isfinite(center) & isfinite(scale) & scale > 0;
if ~any(usable)
    error('pam:ppc:NoUsableStatistics', ...
        'No window statistic has finite non-zero predictive variation.');
end
observed_z = NaN(rows, 1);
observed_z(usable) = (observed_vector(usable) - center(usable)) ./ scale(usable);
replicate_z = NaN(size(replicate_matrix));
replicate_z(:, usable) = (replicate_matrix(:, usable) - center(usable)') ./ ...
    scale(usable)';

global_replicated = row_max_abs(replicate_z(:, usable));
global_observed = max(abs(observed_z(usable)));
global_tail_probability = mean(global_replicated >= global_observed);
simultaneous_threshold = empirical_quantile( ...
    global_replicated, spec.simultaneous_level);
outside_simultaneous = abs(observed_z) > simultaneous_threshold;

summary = table(window_id, family, statistic, valid_trials, ...
    observed_value, predictive_median, predictive_lower, predictive_upper, ...
    predictive_percentile, tail_probability_two_sided, observed_z, ...
    outside_simultaneous);

ppc = struct;
ppc.spec = spec;
ppc.summary = summary;
ppc.replicated_statistics = replicated;
ppc.global_observed = global_observed;
ppc.global_replicated = global_replicated;
ppc.global_tail_probability = global_tail_probability;
ppc.simultaneous_threshold = simultaneous_threshold;
ppc.replicates = n_replicates;
end

function values = response_statistics(rt, choice, stimulus)
valid = isfinite(rt) & isfinite(choice) & isfinite(stimulus);
rt = rt(valid);
choice = choice(valid);
stimulus = stimulus(valid);
if isempty(rt)
    values = NaN(1, 7);
    return
end
values = [ ...
    mean(choice), ...
    mean(choice == stimulus), ...
    empirical_quantile(rt, 0.10), ...
    empirical_quantile(rt, 0.50), ...
    empirical_quantile(rt, 0.90), ...
    conditional_median(rt, choice, 0), ...
    conditional_median(rt, choice, 1)];
end

function value = conditional_median(rt, choice, target)
selected = rt(choice == target);
if isempty(selected)
    value = NaN;
else
    value = empirical_quantile(selected, 0.5);
end
end

function value = empirical_quantile(values, probability)
values = sort(values(isfinite(values)));
if isempty(values)
    value = NaN;
    return
end
if numel(values) == 1
    value = values(1);
    return
end
position = 1 + (numel(values) - 1) * probability;
lower = floor(position);
upper = ceil(position);
weight = position - lower;
value = values(lower) * (1 - weight) + values(upper) * weight;
end

function maxima = row_max_abs(values)
maxima = NaN(size(values, 1), 1);
for row = 1:size(values, 1)
    current = abs(values(row, :));
    current = current(isfinite(current));
    if ~isempty(current)
        maxima(row) = max(current);
    end
end
maxima = maxima(isfinite(maxima));
end
