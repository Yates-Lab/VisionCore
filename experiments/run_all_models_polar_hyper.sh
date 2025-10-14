#!/bin/bash

# OPTIMIZED Polar Model Training Script
# This script uses hyperparameters optimized for the complex Polar-V1 model
#
# KEY OPTIMIZATIONS:
# 1. Higher core learning rate (1.5x) - Polar has many learnable components
# 2. Longer warmup (15 epochs) - Complex model needs more stabilization
# 3. More permissive gradient clipping (5.0) - Polar dynamics need flexibility
# 4. More steps per epoch (1024) - Better data coverage for gaze-dependent features
# 5. Larger batch size (384) - Polar is efficient, can handle it
# 6. Less weight decay (5e-5) - Specialized components need flexibility
# 7. Full training duration (100 epochs) - Complex model needs time

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yatesfv

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=ERROR
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# NCCL settings that fixed the DDP hang
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=1 
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# set up lcuda for compilation
mkdir -p $HOME/.local/lib
ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 $HOME/.local/lib/libcuda.so

export LIBRARY_PATH="$HOME/.local/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_SHOW_CPP_STACKTRACES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Training configuration - OPTIMIZED FOR POLAR MODEL
BATCH_SIZE=384          # Polar is efficient, can handle larger batches (was 256)
MAX_DATASETS=30
LEARNING_RATE=1e-3      # Readout LR
CORE_LR_SCALE=1.5       # Polar core needs higher LR (was 0.5, now 1.5e-3)
LR_SCHEDULER="cosine_warmup"
WARMUP_EPOCHS=15        # Longer warmup for complex model (was 5)
WEIGHT_DECAY=5e-5       # Less regularization for specialized components (was 1e-4)
MAX_EPOCHS=100          # Full training duration (was 50)
PRECISION="bf16-mixed"
DSET_DTYPE="bfloat16"
NUM_GPUS=2
NUM_WORKERS=16
STEPS_PER_EPOCH=1024    # More steps for better data coverage (was 512)

# Loss function configuration
USE_ZIP_LOSS=true

# Project and data paths
PROJECT_NAME="multidataset_polar_120_backimage_hyper"

DATASET_CONFIGS_PATH="/home/jake/repos/VisionCore/experiments/dataset_configs/multi_cones_120_backimage_all_eyepos.yaml"
CHECKPOINT_DIR="/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_polar_120_backimage_hyper/checkpoints"

# Create checkpoint directory
mkdir -p $CHECKPOINT_DIR

# Array of model configurations to run
MODEL_CONFIGS=(
    "experiments/model_configs/polar_v1_relaxed.yaml"          # NEW: Relaxed temporal constants
    "experiments/model_configs/polar_v1.yaml"                  # Original
    "experiments/model_configs/polar_v1_behavior_only.yaml"    # Behavior only
    "experiments/model_configs/polar_v1_minimal.yaml"          # Minimal (no dynamics)
)

# Function to run training for a single model config
run_training() {
    local MODEL_CONFIG=$1
    local MODEL_CONFIG_NAME=$(basename "$MODEL_CONFIG" .yaml)

    # Add ZIP suffix to experiment name if using ZIP loss
    if [ "$USE_ZIP_LOSS" = true ]; then
        local LOSS_SUFFIX="_zip"
    else
        local LOSS_SUFFIX=""
    fi

    local EXPERIMENT_NAME="${MODEL_CONFIG_NAME}_ddp_bs${BATCH_SIZE}_ds${MAX_DATASETS}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}_corelrscale${CORE_LR_SCALE}_warmup${WARMUP_EPOCHS}${LOSS_SUFFIX}"

    echo ""
    echo "============================================================"
    echo "STARTING TRAINING: $MODEL_CONFIG_NAME"
    echo "============================================================"
    echo "Model config: $MODEL_CONFIG"
    echo "Experiment: $EXPERIMENT_NAME"
    echo "Batch size per GPU: $BATCH_SIZE"
    echo "Total effective batch size: $((BATCH_SIZE * NUM_GPUS))"
    echo "Max datasets: $MAX_DATASETS"
    echo "Learning rate: $LEARNING_RATE"
    echo "Core LR scale: $CORE_LR_SCALE (Core LR: $(echo "$LEARNING_RATE * $CORE_LR_SCALE" | bc))"
    echo "LR scheduler: $LR_SCHEDULER"
    echo "Warmup epochs: $WARMUP_EPOCHS"
    echo "Weight decay: $WEIGHT_DECAY"
    echo "Max epochs: $MAX_EPOCHS"
    echo "Precision: $PRECISION"
    echo "Dataset dtype: $DSET_DTYPE"
    echo "Loss type: $([ "$USE_ZIP_LOSS" = true ] && echo "Zero-Inflated Poisson" || echo "Poisson")"
    echo "GPUs: $NUM_GPUS"
    echo "Workers: $NUM_WORKERS"
    echo "Steps per epoch: $STEPS_PER_EPOCH"
    echo "Dataset configs: $DATASET_CONFIGS_PATH"
    echo "Checkpoint dir: $CHECKPOINT_DIR"
    echo ""
    echo "OPTIMIZATIONS APPLIED:"
    echo "  ✓ Core LR 3x higher (0.5 → 1.5)"
    echo "  ✓ Warmup 3x longer (5 → 15 epochs)"
    echo "  ✓ Gradient clip 5x more permissive (1.0 → 5.0)"
    echo "  ✓ Steps per epoch 2x more (512 → 1024)"
    echo "  ✓ Batch size 1.5x larger (256 → 384)"
    echo "  ✓ Weight decay 2x less (1e-4 → 5e-5)"
    echo "  ✓ Max epochs 2x more (50 → 100)"
    echo "  ✓ AdamW betas optimized (0.9, 0.95)"
    echo "============================================================"

    # Build training command with optional ZIP loss flag
    local TRAINING_CMD="python training/train_ddp_multidataset.py \
        --model_config \"$MODEL_CONFIG\" \
        --dataset_configs_path \"$DATASET_CONFIGS_PATH\" \
        --max_datasets $MAX_DATASETS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --core_lr_scale $CORE_LR_SCALE \
        --lr_scheduler $LR_SCHEDULER \
        --warmup_epochs $WARMUP_EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --max_epochs $MAX_EPOCHS \
        --precision $PRECISION \
        --dset_dtype $DSET_DTYPE \
        --num_gpus $NUM_GPUS \
        --project_name \"$PROJECT_NAME\" \
        --experiment_name \"$EXPERIMENT_NAME\" \
        --checkpoint_dir \"$CHECKPOINT_DIR\" \
        --accumulate_grad_batches 1 \
        --gradient_clip_val 5.0 \
        --steps_per_epoch $STEPS_PER_EPOCH \
        --num_workers $NUM_WORKERS \
        --early_stopping_patience 50 \
        --early_stopping_min_delta 0.0"

    # Add loss type flag if using ZIP loss
    if [ "$USE_ZIP_LOSS" = true ]; then
        TRAINING_CMD="$TRAINING_CMD --loss_type zip"
    fi

    # Launch training
    eval $TRAINING_CMD
        # --compile \
        # --enable_curriculum
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Training completed successfully for $MODEL_CONFIG_NAME"
    else
        echo "❌ Training failed for $MODEL_CONFIG_NAME (exit code: $exit_code)"
        return $exit_code
    fi
}

# Main execution
echo "============================================================"
echo "OPTIMIZED POLAR MODEL TRAINING"
echo "============================================================"
echo "Found ${#MODEL_CONFIGS[@]} model configurations:"
for config in "${MODEL_CONFIGS[@]}"; do
    echo "  - $(basename "$config" .yaml)"
done
echo ""
echo "Starting batch training at $(date)"
echo "============================================================"

# Track overall progress
TOTAL_CONFIGS=${#MODEL_CONFIGS[@]}
COMPLETED=0
FAILED=0
FAILED_CONFIGS=()

# Run training for each model configuration
for i in "${!MODEL_CONFIGS[@]}"; do
    MODEL_CONFIG="${MODEL_CONFIGS[$i]}"
    MODEL_CONFIG_NAME=$(basename "$MODEL_CONFIG" .yaml)
    
    echo ""
    echo ">>> PROGRESS: $((i+1))/$TOTAL_CONFIGS - Starting $MODEL_CONFIG_NAME <<<"
    
    if run_training "$MODEL_CONFIG"; then
        COMPLETED=$((COMPLETED + 1))
        echo ">>> PROGRESS: $((i+1))/$TOTAL_CONFIGS - Completed $MODEL_CONFIG_NAME ✓ <<<"
    else
        FAILED=$((FAILED + 1))
        FAILED_CONFIGS+=("$MODEL_CONFIG_NAME")
        echo ">>> PROGRESS: $((i+1))/$TOTAL_CONFIGS - Failed $MODEL_CONFIG_NAME ❌ <<<"
        
        # Ask user if they want to continue after failure
        echo ""
        echo "Training failed for $MODEL_CONFIG_NAME. Continue with remaining configs? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Stopping batch training as requested."
            break
        fi
    fi
done

# Final summary
echo ""
echo "============================================================"
echo "BATCH TRAINING SUMMARY"
echo "============================================================"
echo "Completed at: $(date)"
echo "Total configs: $TOTAL_CONFIGS"
echo "Completed: $COMPLETED"
echo "Failed: $FAILED"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed configurations:"
    for config in "${FAILED_CONFIGS[@]}"; do
        echo "  - $config"
    done
fi

echo ""
if [ $FAILED -eq 0 ]; then
    echo "🎉 All model configurations completed successfully!"
else
    echo "⚠️  Some configurations failed. Check logs above for details."
fi
echo "============================================================"

