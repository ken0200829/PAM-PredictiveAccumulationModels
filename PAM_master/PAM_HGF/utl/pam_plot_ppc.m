function pam_plot_ppc(fitted_models, model_type, rt_probs, min_trials_q, min_trials_m, only_corrects, filename)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Performs posterior predictive checks for PAM models
%
% INPUTS:
%   fitted_models - Cell array of fitted model structures from tapas_fitModel
%   model_type    - String: 'DDM', 'RDM', or 'LNR'
%   rt_probs      - Vector of quantiles to compute (default: [.1 .25 .5 .75 .9])
%   min_trials_q  - Minimum trials required for quantile estimation (default: 20)
%   min_trials_m  - Minimum trials required for mean estimation (default: 3)
%   only_corrects - Flag to plot error quantiles (1=off, 0=on) (default: 1)
%                   Note: Mean error RT is always plotted if data exists
%   filename      - Optional string for saving figure
%
% OUTPUT:
%   Generates figure with posterior predictive checks stratified by belief levels
%   comparing observed vs predicted:
%   - RT quantiles (correct and error responses)
%   - Mean RTs (correct and error responses)
%   - Choice accuracy
%
% USAGE:
%   pam_plot_ppc(fitted_models, 'DDM')
%   pam_plot_ppc(fitted_models, 'RDM', [], 20, 5, 1, 'MyFigure.png')
%
% STRATIFICATION:
%   Data are binned into 5 levels based on predicted belief (muhat), and results
%   are displayed for Low (bin 1), Neutral (bin 3), and High (bin 5) belief levels.
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2024 Antonino Visalli
%
% This file is part of the PAM toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set default values for optional inputs
if nargin < 3 || isempty(rt_probs), rt_probs = [0.10, 0.25, 0.50, 0.75, 0.90]; end
if nargin < 4 || isempty(min_trials_q), min_trials_q = 20; end
if nargin < 5 || isempty(min_trials_m), min_trials_m = 3; end
if nargin < 6 || isempty(only_corrects), only_corrects = 1; end
if nargin < 7, filename = []; end

% Initialize parameters
n_quantiles = length(rt_probs);
n_sims = 100; % Number of simulations per subject
n_subj = length(fitted_models);

% Stratification parameters
n_belief_bins = 5; % Total number of belief bins
target_bins = [1, 3, 5]; % Bins to plot (Low, Neutral, High)
row_labels = {'Low P(u)', 'Neutral P(u)', 'High P(u)'};
n_rows_plot = length(target_bins);

% Initialize storage arrays for observed and predicted statistics
res_qs_corr_obs  = nan(n_subj, n_rows_plot, n_quantiles); % Quantiles correct (observed)
res_qs_corr_pred = nan(n_subj, n_rows_plot, n_quantiles); % Quantiles correct (predicted)
res_qs_err_obs   = nan(n_subj, n_rows_plot, n_quantiles); % Quantiles error (observed)
res_qs_err_pred  = nan(n_subj, n_rows_plot, n_quantiles); % Quantiles error (predicted)
res_mu_corr_obs  = nan(n_subj, n_rows_plot); % Mean RT correct (observed)
res_mu_corr_pred = nan(n_subj, n_rows_plot); % Mean RT correct (predicted)
res_mu_err_obs   = nan(n_subj, n_rows_plot); % Mean RT error (observed)
res_mu_err_pred  = nan(n_subj, n_rows_plot); % Mean RT error (predicted)

res_acc_obs      = nan(n_subj, n_rows_plot); % Accuracy (observed)
res_acc_pred     = nan(n_subj, n_rows_plot); % Accuracy (predicted)

fprintf('\n------------------------------------------------------\n');
fprintf('Running PAM_PPC (%s)\n', model_type);

% Loop through subjects
for s = 1:n_subj
    if isempty(fitted_models{s}), continue; end

    model = fitted_models{s};
    u = model.u;
    y = model.y;

    % Remove invalid trials (NaN RTs)
    valid = ~isnan(y(:,1));
    u = u(valid);
    y = y(valid, :);
    muhat = model.traj.muhat(valid, 1);
    p_obs = model.p_obs;

    % 1. Simulate responses from fitted model
    [sim_rt_mat, sim_choice_mat] = pam_simulate_responses(u, muhat, p_obs, model_type, n_sims);

    % 2. Stratify data by belief level
    % Convert muhat to probability of observed stimulus
    prob_observed = muhat;
    idx_u0 = (u == 0);
    prob_observed(idx_u0) = 1 - muhat(idx_u0);

    % Code correct/incorrect responses
    obs_is_correct = double(y(:,2) == u);
    obs_rt = y(:,1);

    % Create belief bins
    p_levels = linspace(0, 1, n_belief_bins+1);
    edges = quantile(prob_observed, p_levels(2:end-1));

    % Handle case where edges are not unique
    if length(unique(edges)) < (n_belief_bins-1)
        edges = linspace(min(prob_observed), max(prob_observed), n_belief_bins+1);
        edges = edges(2:end-1);
    end

    bin_idx = discretize(prob_observed, [-inf, edges, inf]);

    % 3. Compute statistics for each belief bin
    for i = 1:n_rows_plot
        b = target_bins(i);
        mask_bin = (bin_idx == b);

        % Skip if insufficient trials
        if sum(mask_bin) < min_trials_m, continue; end

        % --- ACCURACY ---
        res_acc_obs(s, i) = mean(obs_is_correct(mask_bin));

        % Compute accuracy for each simulation
        sim_accs = [];
        for r=1:n_sims
            sim_correct = (sim_choice_mat(mask_bin, r) == u(mask_bin));
            sim_accs(end+1) = mean(sim_correct);
        end
        res_acc_pred(s, i) = mean(sim_accs);

        % --- CORRECT RESPONSES ---
        mask_c = mask_bin & (obs_is_correct == 1);

        % Quantiles (if enough trials)
        if sum(mask_c) >= min_trials_q
            res_qs_corr_obs(s, i, :) = quantile(obs_rt(mask_c), rt_probs);
        end

        % Mean (if enough trials)
        if sum(mask_c) >= min_trials_m
            res_mu_corr_obs(s, i) = mean(obs_rt(mask_c));
        end

        % --- ERROR RESPONSES ---
        mask_e = mask_bin & (obs_is_correct == 0);

        % Mean error (always calculated if enough trials)
        if sum(mask_e) >= min_trials_m
            res_mu_err_obs(s, i) = mean(obs_rt(mask_e));
        end

        % Quantiles error (only if requested and enough trials)
        if ~only_corrects && sum(mask_e) >= min_trials_q
            res_qs_err_obs(s, i, :) = quantile(obs_rt(mask_e), rt_probs);
        end

        % --- SIMULATION STATISTICS ---
        sq_c = []; sm_c = []; % Storage for correct responses
        sq_e = []; sm_e = []; % Storage for error responses

        for r=1:n_sims
            r_rt = sim_rt_mat(mask_bin, r);
            r_ch = sim_choice_mat(mask_bin, r);

            % Simulated correct responses
            sim_mask_c = (r_ch == u(mask_bin));
            val_c = r_rt(sim_mask_c);
            if length(val_c) >= min_trials_q
                sq_c(end+1, :) = quantile(val_c, rt_probs);
            end
            if length(val_c) >= min_trials_m
                sm_c(end+1) = mean(val_c);
            end

            % Simulated error responses
            sim_mask_e = (r_ch ~= u(mask_bin));
            val_e = r_rt(sim_mask_e);

            % Mean error (always)
            if length(val_e) >= min_trials_m
                sm_e(end+1) = mean(val_e);
            end

            % Quantile error (conditional)
            if ~only_corrects && length(val_e) >= min_trials_q
                sq_e(end+1, :) = quantile(val_e, rt_probs);
            end
        end

        % Aggregate predictions across simulations
        if ~isempty(sq_c)
            res_qs_corr_pred(s, i, :) = mean(sq_c, 1, 'omitnan');
        end
        res_mu_corr_pred(s, i) = mean(sm_c, 'omitnan');

        if ~isempty(sq_e)
            res_qs_err_pred(s, i, :) = mean(sq_e, 1, 'omitnan');
        end
        res_mu_err_pred(s, i) = mean(sm_e, 'omitnan');
    end
end

% Generate plot
plot_pam_final(res_qs_corr_obs, res_qs_corr_pred, ...
    res_qs_err_obs, res_qs_err_pred, ...
    res_mu_corr_obs, res_mu_corr_pred, ...
    res_mu_err_obs, res_mu_err_pred, ...
    res_acc_obs, res_acc_pred, ...
    rt_probs, row_labels, only_corrects, filename);
fprintf('Done.\n');
end

%% --- SUBFUNCTIONS ---

function plot_pam_final(qc_o, qc_p, qe_o, qe_p, mc_o, mc_p, me_o, me_p, acc_o, acc_p, probs, row_lbls, only_c, filename)
% Creates final PPC visualization with quantile-quantile and accuracy plot

% Create figure
f = figure('Name', 'PAM PPC Results', 'Color', 'w', ...
    'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.65]);

n_rows = size(qc_o, 2);
n_q = length(probs);
n_cols = n_q + 2; % Quantiles + MeanRT + Accuracy

t = tiledlayout(n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');

% Define colors
col_corr = [0.2 0.6 0.3]; % Green for correct
col_err  = [0.8 0.2 0.2]; % Red for error
pt_size = 18; alpha_val = 0.5;

% Calculate global RT limits for uniform axes
all_rt_vals = [qc_o(:); qc_p(:); mc_o(:); mc_p(:)];
if ~only_c
    all_rt_vals = [all_rt_vals; qe_o(:); qe_p(:)];
end
if any(~isnan(me_o(:)))
    all_rt_vals = [all_rt_vals; me_o(:); me_p(:)];
end

min_rt = min(all_rt_vals, [], 'omitnan');
max_rt = max(all_rt_vals, [], 'omitnan');
if isempty(min_rt), min_rt = 0; max_rt = 1; end

% Add padding
pad = 0.05 * (max_rt - min_rt);
rt_lims = [min_rt - pad, max_rt + pad];


% Initialize handles for legend
h_leg_c = [];
h_leg_e = [];

% Loop through rows and columns
for i = 1:n_rows
    for k = 1:n_cols
        nexttile; hold on;

        % === 1. RT PLOTS (Quantiles & Means) ===
        if k <= n_q + 1

            % Is this the Mean RT column?
            is_mean_col = (k == n_q + 1);

            if ~is_mean_col
                % Quantile plot
                xc = qc_o(:, i, k); yc = qc_p(:, i, k);
                xe = qe_o(:, i, k); ye = qe_p(:, i, k);
                title_str = sprintf('RT Q_{%.2f}', probs(k));
            else
                % Mean plot
                xc = mc_o(:, i); yc = mc_p(:, i);
                xe = me_o(:, i); ye = me_p(:, i);
                title_str = 'Mean RT';
            end

            % Plot correct responses
            s1 = scatter(xc, yc, pt_size, col_corr, 'filled', 'MarkerFaceAlpha', alpha_val);
            if is_mean_col, h_leg_c = s1; end % Capture handle from Mean plot

            % Plot error responses (conditional)
            plot_err = (~is_mean_col && ~only_c) || (is_mean_col);
            ccc_e_str = '';
            if plot_err && any(~isnan(xe))
                s2 = scatter(xe, ye, pt_size, col_err, 'filled', 'MarkerFaceAlpha', alpha_val);
                if is_mean_col, h_leg_e = s2; end % Capture handle from Mean plot

                ce = calc_ccc(xe, ye);
                if ~isnan(ce), ccc_e_str = sprintf('\nCCC(Err): %.2f', ce); end
            end

            % Setup Axes
            plot(rt_lims, rt_lims, 'k--');
            xlim(rt_lims); ylim(rt_lims);

            if i == 1, title(title_str, 'FontWeight', 'bold'); end

            % CCC statistics for correct
            cc = calc_ccc(xc, yc);
            txt = sprintf('CCC(Corr): %.2f%s', cc, ccc_e_str);
            text(0.05, 0.95, txt, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
                'FontSize', 7, 'FontWeight', 'bold');

            % Axis labels
            if i == n_rows, xlabel('Observed RT (s)', 'FontSize', 9); end
            if k == 1, ylabel('Predicted RT (s)', 'FontSize', 9); end

            % === 2. ACCURACY PLOT ===
        else
            x = acc_o(:, i); y = acc_p(:, i);
            scatter(x, y, pt_size, 'k', 'filled', 'MarkerFaceAlpha', 0.6);
            if i == 1, title('Accuracy', 'FontWeight', 'bold'); end
            plot([0 1], [0 1], 'k--'); xlim([0 1]); ylim([0 1]);

            % MAE statistic
            mae = mean(abs(x-y), 'omitnan');
            text(0.05, 0.95, sprintf('MAE: %.3f', mae), 'Units', 'normalized', ...
                'VerticalAlignment', 'top', 'FontSize', 8, 'FontWeight', 'bold');

            % Labels
            if i == n_rows, xlabel('Observed Accuracy', 'FontSize', 9); end
            ylabel('Predicted Accuracy', 'FontSize', 9);

            % Row labels on right axis
            yyaxis right; ylim([0 1]); set(gca,'YTick',[],'YColor','k');
            ylabel(row_lbls{i}, 'FontWeight','bold','FontSize',11,'Rotation',-90,'VerticalAlignment','bottom');
            yyaxis left;
        end
        axis square; box on; grid on;
    end
end

% Add global legend
if ~isempty(h_leg_c)
    handles_list = h_leg_c;
    labels_list = {'Correct'};
    if ~isempty(h_leg_e)
        handles_list(end+1) = h_leg_e;
        labels_list{end+1} = 'Error';
    end
    lg = legend(handles_list, labels_list, 'Orientation', 'horizontal', 'Box', 'off');
    lg.Layout.Tile = 'South';
end

title(t, 'Posterior Predictive Checks', 'FontSize', 14, 'FontWeight', 'bold');

% Save figure if filename provided
if ~isempty(filename)
    [filepath, name, ext] = fileparts(filename);
    if isempty(ext), ext = '.png'; end
    full_filename = fullfile(filepath, [name ext]);

    fprintf('Saving figure to: %s\n', full_filename);
    exportgraphics(f, full_filename, 'Resolution', 300);
end
end

function ccc = calc_ccc(x, y)
% Calculates Concordance Correlation Coefficient (CCC)
% CCC measures agreement between observed and predicted values
% CCC = 1 indicates perfect agreement

valid = ~isnan(x) & ~isnan(y);
x = x(valid);
y = y(valid);

% Need at least 3 points
if length(x) < 3
    ccc = NaN;
    return;
end

% Calculate CCC
rho = corr(x, y);
sd_x = std(x); sd_y = std(y);
ccc = (2 * rho * sd_x * sd_y) / (var(x) + var(y) + (mean(x) - mean(y))^2);
end

