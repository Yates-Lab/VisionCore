#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
matrix_dir="$repo_root/outputs/dekel240_paper/m77_epoch279/figure4_real_trace_production_corrected/merged"
analysis_dir="$repo_root/outputs/dekel240_paper/m77_epoch279/figure4_real_trace_production_corrected/image_specific_joint_engagement_phase_spectrum"
direct_dir="$repo_root/outputs/dekel240_paper/m77_epoch279/figure4_real_trace_production_corrected/direct_rendered_joint_engagement"

if [[ ! -f "$matrix_dir/summary.json" || ! -f "$matrix_dir/stabilized_baseline_summary.json" ]]; then
    echo "Missing merged/stabilized M77 matrix in $matrix_dir" >&2
    exit 1
fi

cd "$repo_root"

conda run -n yatesfv python \
    paper/fig4/spatiotemporal_tuning/analyze_image_specific_joint_engagement.py \
    --matrix-dir "$matrix_dir" \
    --n-traces 100 \
    --n-bootstrap 4000 \
    --expected-checkpoint-sha256 36e51bf51593286a4dc3a330bdf6841f252fad3ed62b855ffb4932a7fe2cb853 \
    --expected-dataset-configs-sha256 05ccdcffb8d94d0a0c627d167394aeb9d65b9754da99fb3eddc7a9629fb6c8e1 \
    --require-native-240-contract \
    --out-dir "$analysis_dir"

conda run -n yatesfv python \
    paper/fig4/spatiotemporal_tuning/analyze_direct_rendered_joint_engagement.py \
    --matrix-dir "$matrix_dir" \
    --n-traces 16 \
    --n-bootstrap 4000 \
    --expected-checkpoint-sha256 36e51bf51593286a4dc3a330bdf6841f252fad3ed62b855ffb4932a7fe2cb853 \
    --expected-dataset-configs-sha256 05ccdcffb8d94d0a0c627d167394aeb9d65b9754da99fb3eddc7a9629fb6c8e1 \
    --require-native-240-contract \
    --out-dir "$direct_dir"
