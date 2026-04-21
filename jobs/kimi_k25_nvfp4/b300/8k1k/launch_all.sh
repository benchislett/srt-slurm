#!/usr/bin/env bash
# Launch every variant under this islosl directory with srtctl apply -f.
# Outputs land in ./output/<job_id>/ (one sibling dir, each srtctl run adds a job_id subdir).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${here}/output"
mkdir -p "${output_dir}"

shopt -s nullglob
configs=("${here}"/*/vllm*.yaml)
if [[ ${#configs[@]} -eq 0 ]]; then
    echo "No vllm*.yaml configs found under ${here}" >&2
    exit 1
fi

echo "Launching ${#configs[@]} sweep(s); output base: ${output_dir}"
for cfg in "${configs[@]}"; do
    echo "=== srtctl apply -f ${cfg#${here}/} ==="
    srtctl apply -f "${cfg}" -o "${output_dir}"
done
