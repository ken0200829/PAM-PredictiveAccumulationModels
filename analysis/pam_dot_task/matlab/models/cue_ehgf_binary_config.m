function c = cue_ehgf_binary_config
%CUE_EHGF_BINARY_CONFIG Shared-parameter, cue-specific binary eHGF.
%
% Two independent HGF state sequences are evaluated inside one perceptual
% function. White- and red-cue streams share this single parameter vector.
% Non-active cues are frozen: elapsed time is cue-presentation count, not
% global trial count.
%
% The initial implementation is an effective two-level HGF, matching the
% current PAM tutorial strategy: coupling from level 3 is fixed to zero and
% omega_3 is fixed. omega_2 remains a free individual parameter.

c = struct;
c.model = 'cue_ehgf_binary_shared_effective2';
c.n_levels = 3;
c.irregular_intervals = false;
c.stimulus_column = 1;
c.cue_column = 2;
c.shared_parameters = true;
c.inactive_cue_policy = 'freeze';

% Initial states: both cue streams use the same neutral fixed values.
c.mu_0mu = [NaN, 0, 1];
c.mu_0sa = [NaN, 0, 0];
c.logsa_0mu = [NaN, log(0.1), log(1)];
c.logsa_0sa = [NaN, 0, 0];

% Drift is fixed to zero.
c.rhomu = [NaN, 0, 0];
c.rhosa = [NaN, 0, 0];

% kappa_1 is fixed to one. kappa_2 is fixed to zero, decoupling level 3.
c.logkamu = [log(1), -Inf];
c.logkasa = [0, 0];

% omega_2 is estimated; omega_3 is fixed and decoupled.
c.ommu = [NaN, -3, 2];
c.omsa = [NaN, 2, 0];

c.priormus = [ ...
    c.mu_0mu, ...
    c.logsa_0mu, ...
    c.rhomu, ...
    c.logkamu, ...
    c.ommu];
c.priorsas = [ ...
    c.mu_0sa, ...
    c.logsa_0sa, ...
    c.rhosa, ...
    c.logkasa, ...
    c.omsa];

expected_length = 3 * c.n_levels + 2 * (c.n_levels - 1) + 1;
if numel(c.priormus) ~= expected_length || ...
        numel(c.priorsas) ~= expected_length
    error('pam:cue_hgf:PriorLength', ...
        'Prior definition does not match the configured HGF level count.');
end

c.prc_fun = @cue_ehgf_binary;
c.transp_prc_fun = @cue_ehgf_binary_transp;
end
