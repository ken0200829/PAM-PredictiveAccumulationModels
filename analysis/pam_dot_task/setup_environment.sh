#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
external_dir="${script_dir}/external"
tapas_dir="${external_dir}/tapas"
spm_dir="${external_dir}/spm12"
tapas_commit="7155f99137c5e03f93a2a3afa6a8cb54c75dd4c2"
spm_commit="3085dac00ac804adb190a7e82c6ef11866c8af02"

mkdir -p "${external_dir}"

if [[ ! -d "${tapas_dir}/.git" ]]; then
    git clone https://github.com/translationalneuromodeling/tapas.git "${tapas_dir}"
    git -C "${tapas_dir}" checkout --detach "${tapas_commit}"
fi

if [[ ! -d "${spm_dir}/.git" ]]; then
    git clone --branch r7771 --depth 1 \
        https://github.com/spm/spm12.git "${spm_dir}"
fi

actual_tapas="$(git -C "${tapas_dir}" rev-parse HEAD)"
actual_spm="$(git -C "${spm_dir}" rev-parse HEAD)"
if [[ "${actual_tapas}" != "${tapas_commit}" ]]; then
    echo "TAPAS commit mismatch: ${actual_tapas}" >&2
    exit 1
fi
if [[ "${actual_spm}" != "${spm_commit}" ]]; then
    echo "SPM12 commit mismatch: ${actual_spm}" >&2
    exit 1
fi
if [[ -n "$(git -C "${tapas_dir}" status --short)" ]]; then
    echo "TAPAS dependency has local modifications." >&2
    exit 1
fi
if [[ -n "$(git -C "${spm_dir}" status --short)" ]]; then
    echo "SPM12 dependency has local modifications." >&2
    exit 1
fi

echo "TAPAS ${actual_tapas}"
echo "SPM12 ${actual_spm}"

if command -v matlab >/dev/null 2>&1; then
    echo "MATLAB $(command -v matlab)"
else
    echo "MATLAB is not installed or is not on PATH." >&2
    echo "Install a licensed MATLAB release, then rerun this script." >&2
    exit 2
fi
