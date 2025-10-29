#!/bin/bash

# Smoke test: single-GPU, 4 datasets, 2 epochs
# Modeled after experiments/run_all_models_backimage.sh, but uses the
# single-GPU training script and very short run settings.

# Activate conda environment (adjust if needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yatesfv

# Basic CUDA / logging env (no DDP used here)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1   # keep both visible; script pins via --gpu
export NCCL_DEBUG=ERROR

# Project and data paths
PROJECT_NAME="smoke_sgpu_backimage"
DATASET_CONFIGS_PATH="/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_120_backimage_all.yaml"
CHECKPOINT_DIR="./checkpoints_smoke_sgpu"

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Minimal list of model configs to try (extend if desired)
MODEL_CONFIGS=(
    "experiments/model_configs/learned_dense_concat_convgru_gaussian.yaml"
)

# Training configuration (short and cheap)
MAX_DATASETS=30
BATCH_SIZE=8
LEARNING_RATE=1e-3
CORE_LR_SCALE=.5
LR_SCHEDULER="cosine_warmup"
WARMUP_EPOCHS=5
WEIGHT_DECAY=1e-4
MAX_EPOCHS=100
PRECISION="bf16-mixed"
DSET_DTYPE="bfloat16"
GPU_INDEX=0
NUM_WORKERS=16
STEPS_PER_EPOCH=1024
ENABLE_CURRICULUM=false
HOMOGENEOUS_BATCHES=true
ACCUMULATE_GRAD_BATCHES=32

run_training() {
    local MODEL_CONFIG=$1
    local MODEL_NAME=$(basename "$MODEL_CONFIG" .yaml)

    local EXPERIMENT_NAME="${MODEL_NAME}_sgpu_smoke_ds${MAX_DATASETS}_ep${MAX_EPOCHS}"

    local TRAINING_CMD=(
        python training/train_multidataset.py
            --model_config "$MODEL_CONFIG"
            --dataset_configs_path "$DATASET_CONFIGS_PATH"
            --max_datasets $MAX_DATASETS
            --batch_size $BATCH_SIZE
            --learning_rate $LEARNING_RATE
            --core_lr_scale $CORE_LR_SCALE
            --lr_scheduler $LR_SCHEDULER
            --warmup_epochs $WARMUP_EPOCHS
            --weight_decay $WEIGHT_DECAY
            --max_epochs $MAX_EPOCHS
            --precision $PRECISION
            --dset_dtype $DSET_DTYPE
            --gpu $GPU_INDEX
            --num_workers $NUM_WORKERS
            --steps_per_epoch $STEPS_PER_EPOCH
            --project_name "$PROJECT_NAME"
            --experiment_name "$EXPERIMENT_NAME"
            --checkpoint_dir "$CHECKPOINT_DIR"
            --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES
            --gradient_clip_val 1.0
    )

    if [ "$ENABLE_CURRICULUM" = true ]; then
        TRAINING_CMD+=( --enable_curriculum )
    fi
    if [ "$HOMOGENEOUS_BATCHES" = true ]; then
        TRAINING_CMD+=( --homogeneous_batches )
    fi

    echo "============================================================"
    echo "SMOKE TRAINING: $MODEL_NAME"
    echo "============================================================"
    echo "Command: ${TRAINING_CMD[@]}"
    echo "Start: $(date)"
    echo "------------------------------------------------------------"

    "${TRAINING_CMD[@]}"
    local STATUS=$?

    echo "------------------------------------------------------------"
    echo "End: $(date) | Status: $STATUS"
    echo "============================================================"

    return $STATUS
}

# Main
TOTAL=${#MODEL_CONFIGS[@]}
COMPLETED=0
FAILED=0

for (( i=0; i<${TOTAL}; i++ )); do
    MCfg="${MODEL_CONFIGS[$i]}"
    if run_training "$MCfg"; then
        COMPLETED=$((COMPLETED + 1))
        echo ">>> PROGRESS: $((i+1))/$TOTAL - Completed $(basename "$MCfg" .yaml) ✓ <<<"
    else
        FAILED=$((FAILED + 1))
        echo ">>> PROGRESS: $((i+1))/$TOTAL - Failed $(basename "$MCfg" .yaml) ❌ <<<"
        # Do not prompt; continue through remaining smoke tests
    fi
    echo ""
    sleep 2
done

echo "============================================================"
echo "SMOKE TEST SUMMARY"
echo "Completed: $COMPLETED | Failed: $FAILED | Total: $TOTAL"
if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi

