#!/bin/bash

# Batch training script to run all model configurations sequentially.
# Runs DDP training for each model config in the multi_dataset directory.


# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yatesfv

# ---------- DDP / NCCL (2x PCIe, no NVLink) ----------
export NCCL_DEBUG=ERROR
export NCCL_SOCKET_IFNAME=lo          # local loopback only
export NCCL_IB_DISABLE=1              # no InfiniBand
export NCCL_P2P_DISABLE=1             # no NVLink P2P
export NCCL_ALGO=Ring                 # simple & stable for 2 GPUs
export NCCL_PROTO=Simple              # avoids LL128 quirks on some PCIe setups
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Optional: if you still see intermittent stalls, try:
# export NCCL_SHM_DISABLE=1

# Cap autotune sweep sizes (big wins on compile time)
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_TRIALS=10
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE_TRIALS=5

# Make kernel scheduling a bit more predictable (often helps throughput)
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---------- cuDNN / TF32 ----------
export CUDNN_BENCHMARK=1              # let cuDNN pick fastest kernels
export NVIDIA_TF32_OVERRIDE=1         # allow TF32 on Ampere+ (friendly with bf16)

# ---------- Python / Torch QoL ----------
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# ---------- libcuda workaround (idempotent) ----------
mkdir -p "$HOME/.local/lib"
if [ ! -e "$HOME/.local/lib/libcuda.so" ]; then
  ln -s /lib/x86_64-linux-gnu/libcuda.so.1 "$HOME/.local/lib/libcuda.so"
fi
export LIBRARY_PATH="$HOME/.local/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"

# # Set environment variables for optimal performance
# export CUDA_VISIBLE_DEVICES=0,1
# export NCCL_DEBUG=ERROR
# export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# # NCCL settings that fixed the DDP hang
# export NCCL_SOCKET_IFNAME=lo
# export NCCL_P2P_DISABLE=1 
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# # set up lcuda for compilation
# mkdir -p $HOME/.local/lib          # or another dir you control
# ln -s /lib/x86_64-linux-gnu/libcuda.so.1 $HOME/.local/lib/libcuda.so

# export LIBRARY_PATH="$HOME/.local/lib:$LIBRARY_PATH"
# export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"

# # export TORCH_DISTRIBUTED_DEBUG=DETAIL
# # export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export CUDA_LAUNCH_BLOCKING=0
# export TORCH_SHOW_CPP_STACKTRACES=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Training configuration
BATCH_SIZE=256          # Optimal batch size per GPU
MAX_DATASETS=30        # Scale to all datasets (28 if removed the two bad sessions)
LEARNING_RATE=1e-2    # standard learning rate
CORE_LR_SCALE=.5
LR_SCHEDULER="cosine_warmup_restart"  # Use cosine annealing with warmup
WARMUP_EPOCHS=5        # Number of warmup epochs
WEIGHT_DECAY=1e-4
MAX_EPOCHS=150        # Long training run with early stopping protection
PRECISION="bf16-mixed"
DSET_DTYPE="bfloat16"
NUM_GPUS=2             # Use both RTX 6000 Ada GPUs
NUM_WORKERS=32         # Optimized for 64 CPU cores
STEPS_PER_EPOCH=1024    # Number of steps per epoch

# Model compilation configuration
COMPILE_MODEL=false # THIS ONLY SLOWS THINGS DOWN
ENABLE_LOGGING=true

# Project and data paths
PROJECT_NAME="multidataset_120"

DATASET_CONFIGS_PATH="/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_120_long.yaml"
CHECKPOINT_DIR="/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_120_long/checkpoints"

# Create checkpoint directory
mkdir -p $CHECKPOINT_DIR

# Array of model configurations to run
MODEL_CONFIGS=(
    # "experiments/model_configs/modulator_only_convgru.yaml"
    "experiments/model_configs/learned_resnet_none_none_gaussian.yaml"
    "experiments/model_configs/learned_resnet_none_convgru_gaussian.yaml"
    # "experiments/model_configs/learned_resnet_concat_convgru_gaussian.yaml"
        # "experiments/model_configs/learned_dense_none_convgru_gaussian.yaml"
    # "experiments/model_configs/learned_dense_concat_convgru_gaussian.yaml"
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
    echo "Core LR scale: $CORE_LR_SCALE"
    echo "LR scheduler: $LR_SCHEDULER"
    echo "Warmup epochs: $WARMUP_EPOCHS"
    echo "Weight decay: $WEIGHT_DECAY"
    echo "Max epochs: $MAX_EPOCHS"
    echo "Precision: $PRECISION"
    echo "Dataset dtype: $DSET_DTYPE"
    echo "GPUs: $NUM_GPUS"
    echo "Workers: $NUM_WORKERS"
    echo "Dataset configs: $DATASET_CONFIGS_PATH"
    echo "Checkpoint dir: $CHECKPOINT_DIR"
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

    if [ "$COMPILE_MODEL" = true ]; then
        TRAINING_CMD="$TRAINING_CMD --compile"
    fi

    if [ "$ENABLE_LOGGING" = true ]; then
        TRAINING_CMD="$TRAINING_CMD --enable_logging"
    fi

    # Launch training
    eval $TRAINING_CMD
        # --enable_curriculum
        # --limit_val_batches 20 \
    
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
echo "BATCH TRAINING: ALL MODEL CONFIGURATIONS"
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
