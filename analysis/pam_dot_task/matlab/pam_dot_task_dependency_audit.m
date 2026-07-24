function dependency = pam_dot_task_dependency_audit
%PAM_DOT_TASK_DEPENDENCY_AUDIT Record resolved MATLAB model paths.

names = [ ...
    "tapas_fitModel", ...
    "tapas_ehgf_binary", ...
    "tapas_ehgf_binary_config", ...
    "tapas_quasinewton_optim", ...
    "spm_BMS_gibbs", ...
    "spm_BMS", ...
    "ddm_hgf", ...
    "cue_ehgf_binary"];

resolved = strings(numel(names), 1);
all_matches = cell(numel(names), 1);
available = false(numel(names), 1);
for k = 1:numel(names)
    matches = which(char(names(k)), '-all');
    if isempty(matches)
        matches = cell(0, 1);
    elseif ischar(matches)
        matches = cellstr(matches);
    end
    all_matches{k} = string(matches);
    available(k) = ~isempty(matches);
    if available(k)
        resolved(k) = string(matches{1});
    else
        resolved(k) = missing;
    end
end

dependency = struct;
dependency.matlab_version = version;
dependency.architecture = computer;
dependency.toolbox_versions = ver;
dependency.functions = table(names', available, resolved, all_matches, ...
    'VariableNames', {'name', 'available', 'resolved', 'all_matches'});
end
