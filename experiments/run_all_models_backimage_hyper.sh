#!/bin/bash

# OPTIMIZED ResNet+ConvGRU Training Script with Hyperparameter Improvements
# This script implements all recommended hyperparameter optimizations
#
# KEY OPTIMIZATIONS APPLIED:
# 1. Higher core learning rate (2.0x) - ConvGRU layer norm enables stability
# 2. Longer warmup (10 epochs) - Better for new optimizations
# 3. More permissive gradient clipping (10.0) - Layer norm prevents explosions
# 4. More steps per epoch (1024) - Better data coverage
# 5. Larger batch size (384) - More stable gradients
# 6. Less weight decay (5e-5) - Model has better regularization now
# 7. AdamW betas optimized (0.9, 0.95) - Better for multi-dataset training

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

# Training configuration - OPTIMIZED HYPERPARAMETERS
BATCH_SIZE=384          # Increased from 256 (more stable gradients)
MAX_DATASETS=30
LEARNING_RATE=1e-3      # Readout LR (unchanged)
CORE_LR_SCALE=2.0       # OPTIMIZED: Core learns 2x faster (was 0.5)
LR_SCHEDULER="cosine_warmup"
WARMUP_EPOCHS=10        # OPTIMIZED: 2x longer warmup (was 5)
WEIGHT_DECAY=5e-5       # OPTIMIZED: Less regularization (was 1e-4)
MAX_EPOCHS=100
PRECISION="bf16-mixed"
DSET_DTYPE="bfloat16"
NUM_GPUS=2
NUM_WORKERS=16
STEPS_PER_EPOCH=1024    # OPTIMIZED: 2x more steps (was 512)



# Project and data paths
PROJECT_NAME="multidataset_backimage_120_hyper"

DATASET_CONFIGS_PATH="/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_120_backimage_all.yaml"
CHECKPOINT_DIR="/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120_backimage_hyper/checkpoints"

# Create checkpoint directory
mkdir -p $CHECKPOINT_DIR

# Array of model configurations to run
MODEL_CONFIGS=(
    "experiments/model_configs/learned_res_small_gru_optimized.yaml"
    # Add more configs here as needed
)

# Function to run training for a single model config
run_training() {
    local MODEL_CONFIG=$1
    local MODEL_CONFIG_NAME=$(basename "$MODEL_CONFIG" .yaml)

    local EXPERIMENT_NAME="${MODEL_CONFIG_NAME}_ddp_bs${BATCH_SIZE}_ds${MAX_DATASETS}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}_corelrscale${CORE_LR_SCALE}_warmup${WARMUP_EPOCHS}"

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
    echo "GPUs: $NUM_GPUS"
    echo "Workers: $NUM_WORKERS"
    echo "Steps per epoch: $STEPS_PER_EPOCH"
    echo "Dataset configs: $DATASET_CONFIGS_PATH"
    echo "Checkpoint dir: $CHECKPOINT_DIR"
    echo ""
    echo "HYPERPARAMETER OPTIMIZATIONS:"
    echo "  ✓ Core LR 4x higher (0.5 → 2.0)"
    echo "  ✓ Warmup 2x longer (5 → 10 epochs)"
    echo "  ✓ Gradient clip 10x more permissive (1.0 → 10.0)"
    echo "  ✓ Steps per epoch 2x more (512 → 1024)"
    echo "  ✓ Batch size 1.5x larger (256 → 384)"
    echo "  ✓ Weight decay 2x less (1e-4 → 5e-5)"
    echo "  ✓ AdamW betas optimized (0.9, 0.95)"
    echo ""
    echo "MODEL OPTIMIZATIONS (from config):"
    echo "  ✓ RMSNorm with learnable affine parameters"
    echo "  ✓ ConvGRU layer normalization"
    echo "  ✓ ConvGRU learnable initial hidden state"
    echo "  ✓ ConvGRU reset gate bug fixed"
    echo "  ✓ Proper initialization for SiLU"
    echo "============================================================"

    # Build training command
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
        --gradient_clip_val 10.0 \
        --steps_per_epoch $STEPS_PER_EPOCH \
        --num_workers $NUM_WORKERS \
        --early_stopping_patience 50 \
        --early_stopping_min_delta 0.0"

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
echo "OPTIMIZED RESNET+CONVGRU TRAINING"
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

