#!/usr/bin/env bash
# Step 2: rebuild the orientation-dependent Figure 4 artifacts with the unmodified production scripts, reading Jake's
# bundle read-only and writing only under outputs/stats/fig4_orientation_fix/bundle/figure4. Run from the repo root
# after build_shards.py. The stage trajectory needs a GPU and DataYatesV1.
set -euo pipefail

RUN=/home/jake/repos/VisionCore/outputs/no_phase_readout_comparison_20260910
B=$RUN/rank1/figure4
I=/home/jake/repos/VisionCore/outputs/figure4_gaussian_event_split_20260911/inputs/response_replay_bank_40img_x_200fix_250ms
C=outputs/stats/fig4_orientation_fix/bundle/figure4
SHARDS="$C/matrix_spectral_replay/shard_00/causal_chain_shard.npz $C/matrix_spectral_replay/shard_01/causal_chain_shard.npz"
PY=".venv/bin/python"
export PYTHONPATH=/home/declan/DataYatesV1:. MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/visioncore-mpl}
S=paper/fig4/spatiotemporal_tuning

# $I/image_feature_table.csv is not readable; the merged copy is hash-identical (404aa0...).
# The dataset config is this repo's hash-identical copy (98d6c2...).
$PY $S/analyze_top_passband_stage_trajectory.py \
  --image-table $B/response_matrix_40img_x_200fix/merged/image_feature_table.csv \
  --trace-array $I/trace_xy.npy --trace-table $I/trace_feature_table.csv --trace-provenance $I/trace_provenance.json \
  --spectral-shards $SHARDS \
  --checkpoint "$RUN/training/NO_PHASE_R1_s201/NO_PHASE_R1_s201_03_finetune/epoch=03-val_bps_overall=0.6210.ckpt" \
  --dataset-config $PWD/paper/model_selection/configs/multi_240_long_split3_dekel35_allgratings.yaml \
  --population-spec-dir $B/all_available_population_spec --population-version NO_PHASE_R1_s201_all_available_exactCID_v1 \
  --out-dir $C/top_passband_stage_trajectory_10img_x_10fix --model-label NO_PHASE_R1_s201 \
  --n-images 10 --n-traces 10 --device ${DEVICE:-cuda:0}

$PY $S/compare_passband_path_length.py --population-shards $SHARDS --trace-table $I/trace_feature_table.csv \
  --tuning-summary $B/all_available_yu_tuning/tuning_summary.csv \
  --tuning-contract-summary $B/all_available_yu_tuning/summary.json \
  --population-policy all_checkpoint_available --out-dir $C/passband_vs_path_length

# Release-layout render, audit and provenance: the release manifest's arguments with corrected shards/trajectory.
$PY $S/build_figure4.py --model-spec $RUN/rank1/model/no_phase_model.yaml \
  --panel-a-audit $B/panel_a_exemplar_audit --panel-b-reduction $B/panel_b_population_path_length \
  --tuning-table $B/all_available_yu_tuning/frequency_tuning_grouped.csv \
  --tuning-summary $B/all_available_yu_tuning/tuning_summary.csv \
  --example-fits $B/validated_tuning_contract/crossed_example_fits.csv --all-fits $B/all_available_yu_tuning/all_yu_fits.csv \
  --rucci-ensemble $B/kuang_rucci_ensemble --population-shards $SHARDS \
  --stage-trajectory $C/top_passband_stage_trajectory_10img_x_10fix --out-dir $C/production_figure4/figure \
  --population-policy all_checkpoint_available --n-bootstrap 5000 --seed 20260825

$PY $S/audit_revised_figure4_release.py --figure-dir $C/production_figure4/figure \
  --panel-a-audit $B/panel_a_exemplar_audit --panel-b-reduction $B/panel_b_population_path_length \
  --tuning-release-audit $B/tuning_release_audit --tuning-visual-audit $B/tuning_release_audit/manual_visual_audit.json \
  --tuning-contract $B/all_available_yu_tuning --example-contract $B/validated_tuning_contract \
  --population-spec $B/all_available_population_spec/population_spec_NO_PHASE_R1_s201_all_available_exactCID_v1.npz \
  --rucci-ensemble $B/kuang_rucci_ensemble --passband-comparison $C/passband_vs_path_length \
  --population-shards $SHARDS --spectral-replay-tuning-table $B/all_available_yu_tuning/frequency_tuning_grouped.csv \
  --stage-trajectory $C/top_passband_stage_trajectory_10img_x_10fix --out-dir $C/production_figure4/audit

$PY $S/build_figure4_results_provenance.py --production-audit $C/production_figure4/audit/production_audit.json \
  --panel-a-audit $B/panel_a_exemplar_audit --panel-b-reduction $B/panel_b_population_path_length \
  --tuning-release-audit $B/tuning_release_audit --tuning-contract $B/all_available_yu_tuning \
  --rucci-ensemble $B/kuang_rucci_ensemble --passband-comparison $C/passband_vs_path_length \
  --stage-trajectory $C/top_passband_stage_trajectory_10img_x_10fix --figure-dir $C/production_figure4/figure \
  --output $C/production_figure4/figure/results_provenance.json
