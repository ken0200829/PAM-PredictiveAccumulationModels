#!/usr/bin/env bash
set -euo pipefail

# Build an uploadable MATLAB Online bundle without private CSVs or the large,
# unused portions of SPM12. It is intentionally a build helper: it does not
# modify PAM_master, TAPAS, or SPM12.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"
bundle_root="${1:-${project_root}/matlab_online_fixture_bundle}"
archive_path="${bundle_root}.zip"
tapas_root="${script_dir}/external/tapas"
spm_root="${script_dir}/external/spm12"

if [[ -e "${bundle_root}" || -e "${archive_path}" ]]; then
    echo "Refusing to overwrite existing bundle or archive: ${bundle_root}" >&2
    exit 1
fi
for required in "${tapas_root}" "${spm_root}" "${project_root}/PAM_master"; do
    if [[ ! -d "${required}" ]]; then
        echo "Missing required directory: ${required}" >&2
        exit 1
    fi
done

mkdir -p "${bundle_root}/analysis/pam_dot_task/external/spm12"
mkdir -p "${bundle_root}/analysis/pam_dot_task/external/tapas"
mkdir -p "${bundle_root}/analysis/pam_dot_task"
cp -R "${project_root}/PAM_master" "${bundle_root}/PAM_master"
cp -R "${script_dir}/matlab" "${bundle_root}/analysis/pam_dot_task/matlab"
cp "${script_dir}/environment.lock.json" "${bundle_root}/analysis/pam_dot_task/"

# Only the HGF toolbox (and the two directories tapas_init() unconditionally
# adds, "external" and "tools") are needed: HGF declares no dependency on any
# other TAPAS toolbox (misc/tapas_get_toolbox_infos.m: infos.hgf.dependencies
# = []), and every tapas_* symbol this project calls resolves inside HGF/.
# The full checkout also carries UniQC, sem, PhysIO, task, huge, rDCM, ceode,
# and genbed -- 1,197 files across unrelated toolboxes that MATLAB Online's
# browser-based zip extraction has to walk for no benefit. Keep the excluded
# TAPAS checkout on disk (external/tapas/) as the version-pinned source of
# truth for hashing and local tooling; only the bundle upload is trimmed.
tapas_dirs=(HGF external tools misc)
for tapas_dir in "${tapas_dirs[@]}"; do
    if [[ ! -d "${tapas_root}/${tapas_dir}" ]]; then
        echo "Missing required TAPAS directory: ${tapas_root}/${tapas_dir}" >&2
        exit 1
    fi
    cp -R "${tapas_root}/${tapas_dir}" "${bundle_root}/analysis/pam_dot_task/external/tapas/${tapas_dir}"
done
# tapas_init.m itself lives at the TAPAS repository root, not inside a
# toolbox subdirectory.
cp "${tapas_root}/tapas_init.m" "${bundle_root}/analysis/pam_dot_task/external/tapas/"

spm_files=(
    spm_BMS.m
    spm_BMS_bor.m
    spm_BMS_gibbs.m
    spm_Bcdf.m
    spm_Dpdf.m
    spm_dirichlet_exceedance.m
    spm_gamrnd.m
    spm_gamrnd.mexa64
    spm_multrnd.m
)
for spm_file in "${spm_files[@]}"; do
    if [[ ! -f "${spm_root}/${spm_file}" ]]; then
        echo "Missing required SPM fixture file: ${spm_root}/${spm_file}" >&2
        exit 1
    fi
    cp "${spm_root}/${spm_file}" "${bundle_root}/analysis/pam_dot_task/external/spm12/"
done

(
    cd "$(dirname "${bundle_root}")"
    zip -qr "$(basename "${archive_path}")" "$(basename "${bundle_root}")"
)
printf 'Created MATLAB Online upload bundle: %s\n' "${archive_path}"
