function pam_predictions_plot(fitted_models, model_family, method, filename)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plots group-level predicted RT distributions stratified by belief level
%
% INPUTS:
%   fitted_models - Cell array of fitted model structures from tapas_fitModel
%   model_family  - String: 'DDM', 'RDM', or 'LNR'
%   method        - String: 'bpa' (Bayesian) or 'math' (arithmetic mean) (default: 'math')
%   filename      - Optional string for saving figure (without extension)
%
% OUTPUT:
%   Generates figure with two panels showing predicted RT distributions for:
%   - Left panel: Correct responses (when choice matches stimulus)
%   - Right panel: Incorrect responses (when choice differs from stimulus)
%
%   Distributions are plotted for different belief levels (muhat = 0.1 to 0.9)
%   Color-coded: darker = higher belief
%
% FEATURES:
%   - Parameter averaging: BPA (precision-weighted) or arithmetic mean
%   - Computes RT distribution metrics: mean, median, peak, integral
%   - Visualizes how beliefs modulate decision dynamics
%
% USAGE:
%   pam_predictions_plot(fitted_models, 'DDM', 'bpa')
%   pam_predictions_plot(fitted_models, 'RDM', 'math', 'my_predictions')
%
% NOTE:
%   All predictions assume stimulus u=1 (rightward/choice 1)
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2024 Antonino Visalli
%
% This file is part of the PAM toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set defaults
if nargin < 3 || isempty(method), method = 'math'; end
if nargin < 4, filename = ''; end

n_subj = length(fitted_models);

% Find first valid model
idx = 1;
while idx <= n_subj && isempty(fitted_models{idx})
    idx = idx + 1;
end
if idx > n_subj
    error('No valid fitted models found.');
end

p = struct();

%% --- 1. PARAMETER AVERAGING ---
if strcmpi(method, 'bpa')
    % Bayesian Parameter Average (precision-weighted)
    fprintf('\n=======================================================\n');
    fprintf(' PAM BAYESIAN MEAN PREDICTIONS REPORT \n');
    fprintf('=======================================================\n');
    fprintf('Calculating Bayesian Parameter Average (Tapas)... ');

    try
        % BPA computes precision-weighted averages across subjects
        bavg = tapas_bayesian_parameter_average(fitted_models{:});
        p = bavg.p_obs;
        fprintf('Done.\n\n');
    catch ME
        warning('PAM:BPAFailed', 'BPA failed: %s', ME.message);
        return;
    end

else
    % Arithmetic mean across subjects
    fprintf('\n=======================================================\n');
    fprintf(' PAM ARITHMETIC MEAN PREDICTIONS REPORT \n');
    fprintf('=======================================================\n');

    fields = fieldnames(fitted_models{idx}.p_obs);
    fprintf('Calculating arithmetic means for %d subjects...\n\n', n_subj);

    for f = 1:length(fields)
        fn = fields{f};

        % Skip non-parameter fields
        if any(strcmp(fn, {'p', 'ptrans', 'model', 'type', 'Ter'}))
            continue;
        end

        % Extract parameter values across subjects
        vals = nan(n_subj, 1);
        for s = 1:n_subj
            if ~isempty(fitted_models{s}) && isfield(fitted_models{s}.p_obs, fn)
                vals(s) = fitted_models{s}.p_obs.(fn);
            end
        end

        % Compute arithmetic mean
        p.(fn) = mean(vals, 'omitnan');
    end
end

%% --- 2. PRINT PARAMETERS ---
fprintf('--- Group Average Parameters (%s) ---\n', upper(method));
f_names = fieldnames(p);
for i = 1:length(f_names)
    if ~isnumeric(p.(f_names{i})) || ~isscalar(p.(f_names{i}))
        continue;
    end
    fprintf('%-8s: %8.4f\n', f_names{i}, p.(f_names{i}));
end

%% --- 3. CONFIGURATION ---
% Define belief levels to plot
mu_vals = 0.1:0.2:0.9;
n_steps = length(mu_vals);

% Time grids for visualization and metrics
t_v = 0.01:0.01:3;   % Coarse grid for visualization
t_i = 0.001:0.001:5; % Fine grid for integral calculations

% Color maps (darker = higher belief)
red_map = [linspace(1.0, 0.5, n_steps)', zeros(n_steps, 2)];
blue_map = [zeros(n_steps, 2), linspace(1.0, 0.5, n_steps)'];

% Create figure
fig = figure('Color', 'w', 'Position', [100, 100, 1200, 600]);
tlo = tiledlayout(1, 2, 'Padding', 'loose', 'TileSpacing', 'compact');

% Panel labels
labels = {'Correct (Acc 1)', 'Incorrect (Acc 0)'};

% Y-axis label depends on model
y_label = if_val(strcmp(model_family, 'DDM'), 'WFPT Density', 'Winning PDF');

% Track maximum density for uniform y-axis
max_y = 0.01;

fprintf('\n--- Prediction Metrics (u=1) ---\n');
fprintf('mu-hat | Bound         | Mean    | Median  | Peak    | Prop (Integral)\n');
fprintf('----------------------------------------------------------------------\n');

%% --- 4. COMPUTATION, PLOTTING & METRICS ---
for col = 1:2
    nexttile; hold on;

    % Select color map based on response type
    colors = if_val(col == 1, red_map, blue_map);
    bound_name = labels{col};

    % Loop through belief levels
    for i = 1:n_steps
        mu1hat = mu_vals(i);

        % Compute RT distribution for current belief level
        switch model_family
            case 'DDM'
                % Extract parameters with defaults
                bv = if_field(p, 'b_v', 0);
                ba = if_field(p, 'b_a', 0);
                bw = if_field(p, 'b_w', 0);

                % Calculate effective drift rate
                v_eff = p.a_v + bv * (mu1hat - 0.5);

                % Calculate effective starting point
                w_eff = 0.5 + bw * (mu1hat - 0.5);

                % Calculate effective boundary (modulated by precision)
                prec = 1 / (1 + exp(-(1/(mu1hat*(1-mu1hat)) - 4))) - 0.5;
                a_eff = max(p.a_a + ba * prec, 0.1);

                % Compute WFPT for correct vs incorrect
                if col == 1
                    % Correct: lower boundary (response matches stimulus u=1)
                    pdf_v = utl_wfpt(t_v, -v_eff, a_eff, 1 - w_eff);
                    pdf_m = utl_wfpt(t_i, -v_eff, a_eff, 1 - w_eff);
                else
                    % Incorrect: upper boundary
                    pdf_v = utl_wfpt(t_v, v_eff, a_eff, w_eff);
                    pdf_m = utl_wfpt(t_i, v_eff, a_eff, w_eff);
                end

            case 'RDM'
                % Extract parameters with defaults
                ba = if_field(p, 'b_a', 0);
                bv = if_field(p, 'b_v', 0);
                bval = if_field(p, 'b_val', 0);

                % Calculate thresholds for both accumulators
                a_c1 = max(p.a_a + ba * (mu1hat - 0.5), 0.1);
                a_c0 = max(p.a_a + ba * ((1-mu1hat) - 0.5), 0.1);

                % Calculate drift rates for both accumulators
                v_c1 = max(p.a_v + bval + bv * (mu1hat - 0.5), 0.001);
                v_c0 = max(p.a_v + bv * ((1-mu1hat) - 0.5), 0.001);

                % Compute RDM PDF for winner
                if col == 1
                    % Correct: accumulator 1 wins
                    pdf_v = utl_inverse_gaussian_defective(t_v, v_c1, v_c0, a_c1, a_c0);
                    pdf_m = utl_inverse_gaussian_defective(t_i, v_c1, v_c0, a_c1, a_c0);
                else
                    % Incorrect: accumulator 0 wins
                    pdf_v = utl_inverse_gaussian_defective(t_v, v_c0, v_c1, a_c0, a_c1);
                    pdf_m = utl_inverse_gaussian_defective(t_i, v_c0, v_c1, a_c0, a_c1);
                end

            case 'LNR'
                % Extract parameters with defaults
                b = if_field(p, 'b', 0);
                bval = if_field(p, 'b_val', 0);
                sigma = p.sigma;

                % Calculate means for both accumulators
                theta_c1 = p.a + bval + b * (mu1hat - 0.5);
                theta_c0 = p.a + b * ((1-mu1hat) - 0.5);

                % Compute LNR PDF (winner PDF × loser survival)
                if col == 1
                    % Correct: accumulator 1 wins
                    pdf_v = lognpdf(t_v, theta_c1, sigma) .* (1 - logncdf(t_v, theta_c0, sigma));
                    pdf_m = lognpdf(t_i, theta_c1, sigma) .* (1 - logncdf(t_i, theta_c0, sigma));
                else
                    % Incorrect: accumulator 0 wins
                    pdf_v = lognpdf(t_v, theta_c0, sigma) .* (1 - logncdf(t_v, theta_c1, sigma));
                    pdf_m = lognpdf(t_i, theta_c0, sigma) .* (1 - logncdf(t_i, theta_c1, sigma));
                end
        end

        % Calculate RT distribution metrics
        prop = trapz(t_i, pdf_m);  % Integral (probability mass)
        m_rt = NaN;
        med_rt = NaN;
        peak_rt = NaN;

        if prop > 1e-5
            % Normalize PDF
            pdf_n = pdf_m / prop;

            % Mean RT
            m_rt = trapz(t_i, t_i .* pdf_n);

            % Median RT (50th percentile)
            cdf_n = cumtrapz(t_i, pdf_n);
            [~, med_idx] = min(abs(cdf_n - 0.5));
            med_rt = t_i(med_idx);

            % Peak RT (mode)
            [~, max_idx] = max(pdf_m);
            peak_rt = t_i(max_idx);
        end

        % Print metrics
        fprintf('%-6.1f | %-12s | %-7.3f | %-7.3f | %-7.3f | %-7.4f\n', ...
            mu1hat, bound_name, m_rt, med_rt, peak_rt, prop);

        % Plot RT distribution
        pdf_v(isnan(pdf_v)) = 0;
        plot(t_v, pdf_v, 'Color', colors(i,:), 'LineWidth', 2, ...
            'DisplayName', sprintf('$\\hat{\\mu} = %.1f$', mu1hat));

        % Track maximum for uniform y-axis
        max_y = max(max_y, max(pdf_v));
    end

    % Format panel
    grid on;
    title(labels{col}, 'FontSize', 12);
    xlabel('Time (s)');
    if col == 1
        ylabel(y_label);
    end
    legend('show', 'Location', 'northeast', 'FontSize', 9, 'Interpreter', 'latex');
end

% Apply uniform Y-limits to both panels
for ax = findobj(fig, 'Type', 'axes')'
    set(ax, 'YLim', [0, max_y * 1.15]);
end

%% --- 5. FORMATTING & SAVING ---
% Add main title
main_title = sprintf('%s Predictions: %s (u=1)', upper(method), model_family);
title(tlo, main_title, 'FontWeight', 'bold', 'FontSize', 14, 'Interpreter', 'none');

% Save figure if filename provided
if ~isempty(filename)
    exportgraphics(fig, [filename, '.png'], 'Resolution', 300);
    fprintf('\nFigure saved as: %s.png\n', filename);
end

fprintf('----------------------------------------------------------------------\n');
end

%% --- HELPER FUNCTIONS ---

function val = if_val(cond, v1, v2)
% Inline conditional: returns v1 if cond is true, otherwise v2
if cond
    val = v1;
else
    val = v2;
end
end

function val = if_field(s, f, def)
% Returns field value if it exists, otherwise returns default
if isfield(s, f)
    val = s.(f);
else
    val = def;
end
end
