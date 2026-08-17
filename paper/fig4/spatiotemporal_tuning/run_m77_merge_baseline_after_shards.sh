#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/jake/repos/VisionCore"
production="$repo_root/outputs/dekel240_paper/m77_epoch279/figure4_real_trace_production_corrected"
shard0="$production/shards/images_000_050"
shard1="$production/shards/images_050_100"
merged="$production/merged"

while [[ ! -f "$shard0/summary.json" || ! -f "$shard1/summary.json" ]]; do
    sleep 30
done

cd "$repo_root"

conda run -n yatesfv python \
    paper/fig4/upstream/merge_backimage_real_trace_ssi_matrix_shards.py \
    --out-dir "$merged" \
    "$shard0" \
    "$shard1"

conda run -n yatesfv python \
    paper/fig4/upstream/score_real_trace_stabilized_baseline.py \
    --matrix-dir "$merged" \
    --out-dir "$merged" \
    --rr100-version V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_medoidPosthocminRepcomplete0p45_movieMedoid \
    --n-timepoints 40 \
    --bin-seconds 0.008333333333333333 \
    --patch-size-px 540 \
    --device cuda:0 \
    --frame-batch-size 16 \
    --checkpoint-path /mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/dekel240/D240M77c_dekel_native240_freqmasked_floor0p5_gnlrnalpha0p1_presplit_s201/analysis_candidates/epoch=279-val_bps_overall=0.5912.ckpt \
    --dataset-configs paper/model_selection/configs/multi_240_long_split3_dekel35.yaml \
    --population-spec-dir "$repo_root/outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints"
