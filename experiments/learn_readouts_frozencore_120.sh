#!/bin/bash

# Training script for learning new readouts on frozen core components.
#
# This script:
# 1. Loads a pretrained model from a checkpoint directory
# 2. Freezes core components (frontend, convnet, modulator, recurrent)  
# 3. Builds new adapters and readouts for new dataset configs
# 4. Trains only the new adapters and readouts
#
# Usage:
#   ./experiments/learn_readouts_frozencore_120.sh

# Activate environment based on user
if [ "$USER" = "ryanress" ]; then
    source /home/ryanress/VisionCore/.venv/bin/activate
else
    source /home/jake/miniconda3/etc/profile.d/conda.sh
    conda activate yatesfv
fi


# ---------- DDP / NCCL (2x PCIe, no NVLink) ----------
export NCCL_DEBUG=ERROR
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Cap autotune sweep sizes
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_TRIALS=10
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE_TRIALS=5

# cuDNN / TF32
export CUDNN_BENCHMARK=1
export NVIDIA_TF32_OVERRIDE=1

# Python / Torch QoL
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# libcuda workaround
mkdir -p "$HOME/.local/lib"
if [ ! -e "$HOME/.local/lib/libcuda.so" ]; then
  ln -s /lib/x86_64-linux-gnu/libcuda.so.1 "$HOME/.local/lib/libcuda.so"
fi
export LIBRARY_PATH="$HOME/.local/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"

# ============================================================
# CONFIGURATION - MODIFY THESE FOR YOUR EXPERIMENT
# ============================================================

# Pretrained model configuration
PRETRAINED_CHECKPOINT_DIR="/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_120_long/checkpoints"
MODEL_TYPE="resnet_none_convgru"  # Type of model to load from checkpoint dir

# New dataset configuration  
DATASET_CONFIGS_PATH="/home/ryanress/VisionCore/experiments/dataset_configs/multi_basic_120_long_rowley.yaml"

# Output configuration
PROJECT_NAME="frozencore_readouts_120"
CHECKPOINT_DIR="/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/frozencore_readouts_120/checkpoints"

# Training hyperparameters
BATCH_SIZE=256
MAX_DATASETS=30
LEARNING_RATE=1e-3
WEIGHT_DECAY=1.0e-5
LR_SCHEDULER="cosine_warmup_restart"
WARMUP_EPOCHS=5
MAX_EPOCHS=100
PRECISION="bf16-mixed"
DSET_DTYPE="bfloat16"
NUM_GPUS=2
NUM_WORKERS=32
STEPS_PER_EPOCH=1024

# Logging
ENABLE_LOGGING=true

# ============================================================

# Create checkpoint directory
mkdir -p $CHECKPOINT_DIR

# Build experiment name
EXPERIMENT_NAME="frozencore_${MODEL_TYPE}_bs${BATCH_SIZE}_ds${MAX_DATASETS}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}_warmup${WARMUP_EPOCHS}"

echo ""
echo "============================================================"
echo "FROZEN CORE READOUT TRAINING"
echo "============================================================"
echo "Pretrained checkpoint dir: $PRETRAINED_CHECKPOINT_DIR"
echo "Model type: $MODEL_TYPE"
echo "New dataset configs: $DATASET_CONFIGS_PATH"
echo "Experiment: $EXPERIMENT_NAME"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Total effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "Max datasets: $MAX_DATASETS"
echo "Learning rate: $LEARNING_RATE"
echo "LR scheduler: $LR_SCHEDULER"
echo "Warmup epochs: $WARMUP_EPOCHS"
echo "Weight decay: $WEIGHT_DECAY"
echo "Max epochs: $MAX_EPOCHS"
echo "Precision: $PRECISION"
echo "Dataset dtype: $DSET_DTYPE"
echo "GPUs: $NUM_GPUS"
echo "Workers: $NUM_WORKERS"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "============================================================"
echo ""

# Build training command
TRAINING_CMD="python training/train_frozencore_newreadouts.py \
    --pretrained_checkpoint \"$PRETRAINED_CHECKPOINT_DIR\" \
    --model_type \"$MODEL_TYPE\" \
    --dataset_configs_path \"$DATASET_CONFIGS_PATH\" \
    --max_datasets $MAX_DATASETS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --lr_scheduler $LR_SCHEDULER \
    --warmup_epochs $WARMUP_EPOCHS \
    --max_epochs $MAX_EPOCHS \
    --precision $PRECISION \
    --dset_dtype $DSET_DTYPE \
    --num_gpus $NUM_GPUS \
    --project_name \"$PROJECT_NAME\" \
    --experiment_name \"$EXPERIMENT_NAME\" \
    --checkpoint_dir \"$CHECKPOINT_DIR\" \
    --accumulate_grad_batches 1 \
    --gradient_clip_val 1.0 \
    --steps_per_epoch $STEPS_PER_EPOCH \
    --num_workers $NUM_WORKERS \
    --early_stopping_patience 30 \
    --early_stopping_min_delta 0.0"

if [ "$ENABLE_LOGGING" = true ]; then
    TRAINING_CMD="$TRAINING_CMD --enable_logging"
fi

# Run training
echo "Starting training at $(date)"
eval $TRAINING_CMD
exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✓ Training completed successfully at $(date)"
else
    echo "❌ Training failed with exit code: $exit_code"
fi
echo "============================================================"

