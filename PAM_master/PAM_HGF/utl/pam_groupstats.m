function stats = pam_groupstats(fitted_models, model_type, min_trials)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generates group-level statistics and report for PAM fitted models
%
% INPUTS:
%   fitted_models - Cell array of fitted model structures from tapas_fitModel
%   model_type    - String: 'DDM', 'RDM', or 'LNR'
%   min_trials    - Minimum trials required per bin for inclusion (default: 5)
%
% OUTPUT:
%   stats - Structure containing:
%           .info         - Summary information (n_subj, model_type)
%           .params_desc  - Descriptive statistics for each parameter
%           .tests_rfx    - Random-effects tests (t-tests on slopes)
%           .tests_ffx    - Fixed-effects tests (z-tests on BPA posterior)
%           .ppc_data     - Posterior predictive check data
%           .ppc_summary  - Goodness-of-fit metrics (CCC, MAE)
%           .bpa_raw      - Raw Bayesian Parameter Average output
%
% FEATURES:
%   1. Master Parameter Table:
%      - T-tests on slope parameters (test vs 0, random-effects)
%      - Z-tests on BPA posterior means (test vs 0, fixed-effects using Bayesian posterior)
%   2. Descriptive Statistics:
%      - Adjusted means using Cousineau-Morey within-subject correction
%   3. Posterior Predictive Checks:
%      - CCC for correct/error RTs stratified by belief level
%      - MAE for accuracy
%
% USAGE:
%   stats = pam_groupstats(fitted_models, 'DDM');
%   save('group_results.mat', 'stats');
%
% STATISTICAL TESTS:
%   - RFX T-test: Tests if slope parameters differ from 0 across subjects (random-effects)
%   - FFX Z-test: Tests if BPA posterior mean differs from 0 using posterior SD (Wald test)
%     Formula: z = μ_posterior / σ_posterior, p = 2*(1 - Φ(|z|))
%     Reference: Liu et al. (2020) posterior-based Wald-type statistics
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2025 Antonino Visalli
%
% This file is part of the PAM toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set defaults
if nargin < 3, min_trials = 5; end

fprintf('\n=======================================================\n');
fprintf('          PAM GROUP STATISTICS & REPORT                \n');
fprintf('=======================================================\n');

n_subj = length(fitted_models);

% Find first valid model to extract parameter names
first_valid = 1;
while isempty(fitted_models{first_valid}), first_valid=first_valid+1; end

% Setup parameter list (exclude non-parameter fields)
ignore_list = {'p', 'ptrans', 'model', 'type', 'unsampled', 'irreg', 'Ter'};
obs_fields_raw = fieldnames(fitted_models{first_valid}.p_obs);
valid_obs_names = {};
for i = 1:length(obs_fields_raw)
    fn = obs_fields_raw{i};
    if ~any(strcmp(fn, ignore_list))
        valid_obs_names{end+1} = fn;
    end
end

% Initialize stats structure
stats = struct();
stats.info.n_subj = n_subj;
stats.info.model_type = model_type;
stats.params_desc = struct();
stats.tests_rfx = struct();   % Random-effects tests (t-tests)
stats.tests_ffx = struct();   % Fixed-effects tests (z-tests on BPA)
stats.ppc_data = struct();

%% --- PART 0: RUN BPA ---
fprintf('Calculating Bayesian Parameter Average (Tapas)... ');
try
    % Run BPA and suppress output
    run_bpa = @() tapas_bayesian_parameter_average(fitted_models{:});
        [~, bavg] = evalc('run_bpa()'); 
        has_bpa = true;
        stats.bpa_raw = bavg; 
        fprintf('Done.\n');
catch
    warning('PAM:BPA', 'BPA failed. Z-tests will be skipped.');
    has_bpa = false;
    bavg = [];
end

% Build mapping from parameter names to indices in Sigma matrix
% This is needed to extract posterior SDs for z-tests
sigma_map_obs = containers.Map();
if has_bpa && isfield(bavg, 'optim') && isfield(bavg.optim, 'Sigma')
    % Count number of fitted perceptual parameters
    n_prc_fitted = 0;
    if isfield(bavg, 'c_prc') && isfield(bavg.c_prc, 'priorsas')
        n_prc_fitted = sum(bavg.c_prc.priorsas(:) > 0 & ~isnan(bavg.c_prc.priorsas(:)));
    end

    % Map observation parameter names to Sigma indices
    if isfield(bavg, 'c_obs') && isfield(bavg.c_obs, 'priorsas')
        priorsas_obs = bavg.c_obs.priorsas(:);
        current_sigma_idx = n_prc_fitted;
        obs_field_counter = 0;

        for k = 1:length(obs_fields_raw)
            fn = obs_fields_raw{k};
            if any(strcmp(fn, {'p','ptrans','model','type','unsampled','irreg'}))
                continue;
            end

            obs_field_counter = obs_field_counter + 1;
            if obs_field_counter <= length(priorsas_obs)
                % Only count parameters that were actually estimated (prior variance > 0)
                if priorsas_obs(obs_field_counter) > 0 && ~isnan(priorsas_obs(obs_field_counter))
                    current_sigma_idx = current_sigma_idx + 1;
                    if ~strcmp(fn, 'Ter'), sigma_map_obs(fn) = current_sigma_idx; end
                end
            end
        end
    end
end

%% --- PART 1: MASTER PARAMETER STATISTICS ---
fprintf('\n--- 1. MASTER PARAMETER STATISTICS ---\n');
fprintf('RFX: T-test on Slopes ("b_") | FFX: Z-test on BPA Posterior\n');
fprintf('----------------------------------------------------------------------------------------------------------------------------\n');
fprintf('%-10s | %-16s | %-16s | %-12s || %-10s | %-10s | %-16s\n', ...
    'Param', 'Mean (SD)', 't(df) [Slopes]', 'p-val', 'BPA Mean', 'BPA SD', 'Z (p-val)');
fprintf('----------------------------------------------------------------------------------------------------------------------------\n');

% 1A. OBSERVATION PARAMETERS
for i = 1:length(valid_obs_names)
    fn = valid_obs_names{i};

    % Extract parameter values across subjects
    vals = nan(n_subj, 1);
    for s = 1:n_subj
        if isempty(fitted_models{s})
            continue;
        end
        val = fitted_models{s}.p_obs.(fn);
        if isnumeric(val), vals(s) = val(1); end
    end

    % Calculate arithmetic mean and SD
    mu_arith = mean(vals, 'omitnan');
    sd_arith = std(vals, 'omitnan');
    stats.params_desc.(fn).mean = mu_arith;
    stats.params_desc.(fn).sd = sd_arith;
    stats.params_desc.(fn).all_values = vals;

    % --- RANDOM-EFFECTS T-TEST (SLOPES ONLY) ---
    % Test H0: mean_slope = 0 across subjects
    is_slope = startsWith(fn, 'b_', 'IgnoreCase', true) || startsWith(fn, 'beta', 'IgnoreCase', true);
    t_str = '-';
    p_freq_str = '-';

    if is_slope && ~all(isnan(vals)) && std(vals, 'omitnan') > 0
        [~, p_val, ~, tstats] = ttest(vals);
        t_str = sprintf('%5.2f(%d)', tstats.tstat, tstats.df);

        % Format p-value with significance stars
        if p_val < 0.001
            p_freq_str = sprintf('%.4f***', p_val);
        elseif p_val < 0.05
            p_freq_str = sprintf('%.4f*', p_val);
        else
            p_freq_str = sprintf('%.4f', p_val);
        end

        stats.tests_rfx.(fn).t = tstats.tstat;
        stats.tests_rfx.(fn).p = p_val;
    end

    % --- FIXED-EFFECTS Z-TEST (ALL PARAMETERS) ---
    % Test H0: BPA_posterior_mean = 0 using posterior SD
    % This is a Wald test on the Bayesian posterior distribution
    bpa_mu_str = '-';
    bpa_sd_str = '-';
    z_res_str = '-';

    if has_bpa && isfield(bavg.p_obs, fn)
        mu_bpa = bavg.p_obs.(fn);
        if ~isscalar(mu_bpa), mu_bpa = mu_bpa(1); end

        % Extract posterior SD from Sigma matrix
        sd_bpa = NaN;
        if isKey(sigma_map_obs, fn)
            idx = sigma_map_obs(fn);
            if idx <= size(bavg.optim.Sigma, 1)
                sd_bpa = sqrt(bavg.optim.Sigma(idx, idx));
            end
        end

        if ~isnan(mu_bpa)
            bpa_mu_str = sprintf('%.3f', mu_bpa);

            if ~isnan(sd_bpa) && sd_bpa > 1e-9
                bpa_sd_str = sprintf('%.3f', sd_bpa);

                % Wald z-test: z = μ / σ (testing H0: μ = 0)
                z_val = mu_bpa / sd_bpa;

                % Two-tailed p-value
                p_z = 2 * (1 - normcdf(abs(z_val)));

                % Format with significance stars
                p_z_star = '';
                if p_z < 0.001
                    p_z_star = '***';
                elseif p_z < 0.05
                    p_z_star = '*';
                end

                z_res_str = sprintf('%5.2f (%.4f%s)', z_val, p_z, p_z_star);
                stats.tests_ffx.(fn).mean_bpa = mu_bpa;
                stats.tests_ffx.(fn).sd_bpa = sd_bpa;
                stats.tests_ffx.(fn).z = z_val;
                stats.tests_ffx.(fn).p = p_z;
            else
                bpa_sd_str = '(Fix)';
            end
        end
    end
    fprintf('%-10s | %5.3f (%5.3f)  | %-16s | %-12s || %-10s | %-10s | %-16s\n', ...
        fn, mu_arith, sd_arith, t_str, p_freq_str, bpa_mu_str, bpa_sd_str, z_res_str);
end
fprintf('----------------------------------------------------------------------------------------------------------------------------\n');

% 1B. PERCEPTUAL PARAMETERS
prc_fields = {'omega_2', 'omega_3'};
om_indices = [2, 3];

for k = 1:length(prc_fields)
    lbl = prc_fields{k};
    idx = om_indices(k);

    % Extract omega values across subjects
    vals_om = nan(n_subj, 1);
    for s = 1:n_subj
        try
            vals_om(s) = fitted_models{s}.p_prc.om(idx);
        catch
        end
    end

    mu_om = mean(vals_om, 'omitnan');
    sd_om = std(vals_om, 'omitnan');

    % Extract BPA value if available
    bpa_mu_str = '-';
    if has_bpa && isfield(bavg.p_prc, 'om') && length(bavg.p_prc.om) >= idx
        mu_bpa = bavg.p_prc.om(idx);
        bpa_mu_str = sprintf('%.3f', mu_bpa);
    end

    fprintf('%-10s | %5.3f (%5.3f)  | %-16s | %-12s || %-10s | %-10s | %-16s\n', ...
        lbl, mu_om, sd_om, '-', '-', bpa_mu_str, '-', '-');
end
fprintf('----------------------------------------------------------------------------------------------------------------------------\n');

%% --- 2. DATA COLLECTION & SIMULATION ---
n_sims = 100;
target_bins = [1, 3, 5]; % Low, Neutral, High belief
bin_labels = {'Low P(u)', 'Neutral P(u)', 'High P(u)'};
n_bins_plot = length(target_bins);

% Initialize storage matrices (Subj x Bins)
raw_rt_corr_mn = nan(n_subj, n_bins_plot);
raw_rt_err_mn  = nan(n_subj, n_bins_plot);
raw_acc        = nan(n_subj, n_bins_plot);
pred_rt_corr_mn = nan(n_subj, n_bins_plot);
pred_rt_err_mn  = nan(n_subj, n_bins_plot);
pred_acc        = nan(n_subj, n_bins_plot);

% Loop through subjects
for s = 1:n_subj
    try
        model = fitted_models{s};
        u = model.u;
        y = model.y;

        % Remove invalid trials
        valid = ~isnan(y(:,1));
        u = u(valid);
        y = y(valid, :);
        muhat = model.traj.muhat(valid, 1);
        p_obs = model.p_obs;

        % Simulate responses
        [sim_rt_mat, sim_choice_mat] = pam_simulate_responses(u, muhat, p_obs, model_type, n_sims);

        % Convert muhat to probability of observed stimulus
        prob_observed = muhat;
        idx_u0 = (u == 0);
        prob_observed(idx_u0) = 1 - muhat(idx_u0);

        % Code correct/incorrect
        obs_is_correct = double(y(:,2) == u);
        obs_rt = y(:,1);

        % Create belief bins (quantile-based)
        edges = quantile(prob_observed, [0.2 0.4 0.6 0.8]);
        if length(unique(edges)) < 4
            edges = [0.2 0.4 0.6 0.8];
        end
        bin_idx = discretize(prob_observed, [-inf, edges, inf]);

        % Compute statistics for each target bin
        for b_i = 1:n_bins_plot
            curr_bin = target_bins(b_i);
            mask = (bin_idx == curr_bin);

            if sum(mask) < min_trials, continue; end

            % A. ACCURACY
            raw_acc(s, b_i) = mean(obs_is_correct(mask));
            sim_accs = mean(sim_choice_mat(mask, :) == u(mask), 2);
            pred_acc(s, b_i) = mean(sim_accs);

            % B. CORRECT RT
            mask_c = mask & (obs_is_correct == 1);
            if sum(mask_c) >= min_trials
                raw_rt_corr_mn(s, b_i) = mean(obs_rt(mask_c));

                % Aggregate across simulations
                sm=[];
                for r=1:n_sims
                    val=sim_rt_mat(mask,r);
                    val=val(sim_choice_mat(mask,r)==u(mask));
                    if ~isempty(val), sm(end+1)=mean(val);
                    end
                end
                pred_rt_corr_mn(s, b_i) = mean(sm);
            end

            % C. ERROR RT
            mask_e = mask & (obs_is_correct == 0);
            if sum(mask_e) >= min_trials
                raw_rt_err_mn(s, b_i) = mean(obs_rt(mask_e));

                % Aggregate across simulations
                sm=[];
                for r=1:n_sims
                    val=sim_rt_mat(mask,r);
                    val=val(sim_choice_mat(mask,r)~=u(mask));
                    if ~isempty(val)
                        sm(end+1)=mean(val);
                    end
                end
                pred_rt_err_mn(s, b_i) = mean(sm);
            end
        end
    catch
        continue;
    end
end

% Store raw data
stats.ppc_data.raw_rt_corr_mn = raw_rt_corr_mn;
stats.ppc_data.raw_rt_err_mn  = raw_rt_err_mn;

%% --- 3. DESCRIPTIVE STATISTICS (Adjusted Means) ---
% Apply Cousineau-Morey within-subject correction
fprintf('\n--- 2. DESCRIPTIVE STATISTICS (Adjusted Means) ---\n');
fprintf('Method: Cousineau-Morey correction.\n');
fprintf('NOTE: If ANY subject has a missing cell (NaN) in a bin,\n');
fprintf('      the Adjusted Mean for that bin is reported as NaN.\n');
fprintf('------------------------------------------------------------------------------------------\n');
fprintf('%-12s | %-20s | %-20s | %-20s\n', 'Bin', 'Adj. Corr Mean (SE)', 'Adj. Err Mean (SE)', 'Adj. Acc Mean (SE)');
fprintf('------------------------------------------------------------------------------------------\n');

% Cousineau-Morey adjustment: remove subject mean, add grand mean
calc_adj = @(M) (M - mean(M, 2, 'omitnan') + mean(mean(M, 2, 'omitnan'), 'omitnan'));

adj_rt_corr_mn = calc_adj(raw_rt_corr_mn);
adj_rt_err_mn  = calc_adj(raw_rt_err_mn);
adj_acc        = calc_adj(raw_acc);

for b_i = 1:n_bins_plot
    % Correct RT (strict NaN check on raw data)
    if any(isnan(raw_rt_corr_mn(:, b_i)))
        str_c = 'NaN';
    else
        m_c = mean(adj_rt_corr_mn(:, b_i), 'omitnan');
        se_c = std(adj_rt_corr_mn(:, b_i), 'omitnan') / sqrt(n_subj);
        str_c = sprintf('%.3f (%.3f)', m_c, se_c);
    end

    % Error RT (strict NaN check on raw data)
    if any(isnan(raw_rt_err_mn(:, b_i)))
        str_e = 'NaN';
    else
        m_e = mean(adj_rt_err_mn(:, b_i), 'omitnan');
        se_e = std(adj_rt_err_mn(:, b_i), 'omitnan') / sqrt(n_subj);
        str_e = sprintf('%.3f (%.3f)', m_e, se_e);
    end

    % Accuracy (strict NaN check on raw data)
    if any(isnan(raw_acc(:, b_i)))
        str_a = 'NaN';
    else
        m_a = mean(adj_acc(:, b_i), 'omitnan');
        se_a = std(adj_acc(:, b_i), 'omitnan') / sqrt(n_subj);
        str_a = sprintf('%.3f (%.3f)', m_a, se_a);
    end

    fprintf('%-12s | %-20s | %-20s | %-20s\n', bin_labels{b_i}, str_c, str_e, str_a);
end
fprintf('------------------------------------------------------------------------------------------\n');

%% --- 4. FIT QUALITY TABLE (CCC & MAE) ---
fprintf('\n--- 3. GOODNESS OF FIT (PPC METRICS) ---\n');
fprintf('----------------------------------------------------------\n');
fprintf('%-12s | %-10s | %-10s | %-10s\n', 'Bin', 'CCC Corr', 'CCC Err', 'MAE Acc');
fprintf('----------------------------------------------------------\n');

for b_i = 1:n_bins_plot
    % CCC on correct RT means
    ccc_c = calc_ccc(raw_rt_corr_mn(:, b_i), pred_rt_corr_mn(:, b_i));

    % CCC on error RT means
    if all(isnan(raw_rt_err_mn(:, b_i))) || all(isnan(pred_rt_err_mn(:, b_i)))
        str_ccc_e = 'NaN';
    else
        ccc_e = calc_ccc(raw_rt_err_mn(:, b_i), pred_rt_err_mn(:, b_i));
        str_ccc_e = sprintf('%.3f', ccc_e);
    end

    % MAE on accuracy
    mae_a = mean(abs(raw_acc(:, b_i) - pred_acc(:, b_i)), 'omitnan');

    fprintf('%-12s | %-10.3f | %-10s | %-10.3f\n', bin_labels{b_i}, ccc_c, str_ccc_e, mae_a);

    % Store in summary
    stats.ppc_summary(b_i).bin = bin_labels{b_i};
    stats.ppc_summary(b_i).ccc_corr = ccc_c;
    stats.ppc_summary(b_i).mae_acc = mae_a;
end
fprintf('----------------------------------------------------------\n');
end

function ccc = calc_ccc(x, y)
% Calculates Concordance Correlation Coefficient (CCC)
% Measures agreement between observed and predicted values
valid = ~isnan(x) & ~isnan(y); x = x(valid); y = y(valid);
if length(x) < 3, ccc = NaN; return; end
rho = corr(x, y); sd_x = std(x); sd_y = std(y);
ccc = (2 * rho * sd_x * sd_y) / (var(x) + var(y) + (mean(x) - mean(y))^2);
end


