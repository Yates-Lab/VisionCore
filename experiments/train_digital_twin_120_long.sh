#!/bin/bash
#
# Train a single digital twin model (120Hz, long training).
# All hyperparameters are configurable via CLI flags with sensible defaults.
# Checkpoints are saved to a timestamped directory for easy comparison.
#
# Usage:
#   bash experiments/train_digital_twin_120_long.sh
#   bash experiments/train_digital_twin_120_long.sh --lr 5e-3 --wd 1e-5
#   bash experiments/train_digital_twin_120_long.sh --model_config experiments/model_configs/learned_resnet_none_none_gaussian.yaml
#   bash experiments/train_digital_twin_120_long.sh --lr 1e-3 --tag "low_lr_test"
#

set -euo pipefail

# ===================== Parse CLI arguments =====================

# Defaults (matching Jake's best settings from run_all_models_120_base.sh)
#LR="1e-2" 
#LR="3.3e-4"
LR="1e-3" # Optimal
#MODEL_CONFIG="experiments/model_configs/learned_resnet_wide3_none_convgru_gaussian.yaml"
MODEL_CONFIG="experiments/model_configs/learned_resnet_concat_convgru_gaussian.yaml"
WD="1e-5"
CORE_LR_SCALE="1.0"
LR_SCHEDULER="cosine_warmup"
WARMUP_EPOCHS=2
BATCH_SIZE=256
ACCUMULATE_GRAD_BATCHES=4
STEPS_PER_EPOCH=512 # Really just determines how often to validate and log, not the actual epoch length
MAX_EPOCHS=50
#TIME_BUDGET="720" # In minutes. Set to empty string for no time limit.
TIME_BUDGET="2880" # 48 hours
TAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --lr)           LR="$2";             shift 2 ;;
        --model_config) MODEL_CONFIG="$2";   shift 2 ;;
        --wd)           WD="$2";             shift 2 ;;
        --core_lr_scale) CORE_LR_SCALE="$2"; shift 2 ;;
        --lr_scheduler) LR_SCHEDULER="$2";   shift 2 ;;
        --warmup_epochs) WARMUP_EPOCHS="$2"; shift 2 ;;
        --batch_size)   BATCH_SIZE="$2";     shift 2 ;;
        --max_epochs)   MAX_EPOCHS="$2";     shift 2 ;;
        --time_budget)  TIME_BUDGET="$2";    shift 2 ;;
        --tag)          TAG="$2";            shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Valid flags: --lr, --model_config, --wd, --core_lr_scale, --lr_scheduler, --warmup_epochs, --batch_size, --max_epochs, --time_budget, --tag"
            exit 1
            ;;
    esac
done

# ===================== Fixed parameters =====================

PRECISION="bf16-mixed"
DSET_DTYPE="bfloat16"
NUM_GPUS=1
NUM_WORKERS=32
MAX_DATASETS=30
GRADIENT_CLIP_VAL=10.0
LIMIT_VAL_BATCHES=0.2
COMPILE_MODEL=false
ENABLE_LOGGING=true

# ===================== Derived names =====================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_CONFIG_NAME=$(basename "$MODEL_CONFIG" .yaml)
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Build checkpoint directory name
CKPT_SUBDIR="${TIMESTAMP}_${MODEL_CONFIG_NAME}"
if [ -n "$TAG" ]; then
    CKPT_SUBDIR="${CKPT_SUBDIR}_${TAG}"
fi

CHECKPOINT_BASE="/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/digital_twin_120"
CHECKPOINT_DIR="${CHECKPOINT_BASE}/${CKPT_SUBDIR}"

PROJECT_NAME="digital_twin_120"
EXPERIMENT_NAME="${MODEL_CONFIG_NAME}_lr${LR}_wd${WD}_cls${CORE_LR_SCALE}_bs${BATCH_SIZE}_ga${ACCUMULATE_GRAD_BATCHES}"

DATASET_CONFIGS_PATH="${SCRIPT_DIR}/experiments/dataset_configs/multi_basic_120_long.yaml"

# ===================== Environment setup =====================

# Work from the VisionCore root so relative paths resolve correctly
cd "$SCRIPT_DIR"

# DDP / NCCL (2x PCIe, no NVLink)
export NCCL_DEBUG=ERROR
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Autotune limits
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_TRIALS=10
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE_TRIALS=5
export CUDA_DEVICE_MAX_CONNECTIONS=1

# cuDNN / TF32
export CUDNN_BENCHMARK=1
export NVIDIA_TF32_OVERRIDE=1

# Python / Torch
export PYTHONPATH="${PYTHONPATH:-}:${SCRIPT_DIR}"
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# libcuda workaround
mkdir -p "$HOME/.local/lib"
if [ ! -e "$HOME/.local/lib/libcuda.so" ]; then
    ln -s /lib/x86_64-linux-gnu/libcuda.so.1 "$HOME/.local/lib/libcuda.so"
fi
export LIBRARY_PATH="$HOME/.local/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"

# ===================== Create checkpoint dir & save config =====================

mkdir -p "$CHECKPOINT_DIR"

cat > "${CHECKPOINT_DIR}/run_config.txt" <<CONFIGEOF
# Digital Twin Training Run Configuration
# Generated: $(date)
# Host: $(hostname)
# User: $(whoami)
# Git commit: $(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Hyperparameters
learning_rate=$LR
weight_decay=$WD
core_lr_scale=$CORE_LR_SCALE
lr_scheduler=$LR_SCHEDULER
warmup_epochs=$WARMUP_EPOCHS
batch_size=$BATCH_SIZE
max_epochs=$MAX_EPOCHS

# Model
model_config=$MODEL_CONFIG
model_config_name=$MODEL_CONFIG_NAME

# Data
dataset_configs_path=$DATASET_CONFIGS_PATH
max_datasets=$MAX_DATASETS
dset_dtype=$DSET_DTYPE

# Training
precision=$PRECISION
num_gpus=$NUM_GPUS
num_workers=$NUM_WORKERS
steps_per_epoch=$STEPS_PER_EPOCH
gradient_clip_val=$GRADIENT_CLIP_VAL
accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES
limit_val_batches=$LIMIT_VAL_BATCHES

# Logging
project_name=$PROJECT_NAME
experiment_name=$EXPERIMENT_NAME
checkpoint_dir=$CHECKPOINT_DIR
tag=$TAG
CONFIGEOF

# ===================== Print run info =====================

echo ""
echo "============================================================"
echo "DIGITAL TWIN TRAINING: $MODEL_CONFIG_NAME"
echo "============================================================"
echo "Timestamp:          $TIMESTAMP"
echo "Model config:       $MODEL_CONFIG"
echo "Learning rate:      $LR"
echo "Core LR scale:      $CORE_LR_SCALE"
echo "Weight decay:       $WD"
echo "LR scheduler:       $LR_SCHEDULER"
echo "Warmup epochs:      $WARMUP_EPOCHS"
echo "Batch size/GPU:     $BATCH_SIZE"
echo "Effective batch:    $((BATCH_SIZE * NUM_GPUS))"
echo "Max epochs:         $MAX_EPOCHS"
echo "Time budget (min):  ${TIME_BUDGET:-<none>}"
echo "Checkpoint dir:     $CHECKPOINT_DIR"
echo "Dataset config:     $DATASET_CONFIGS_PATH"
echo "Steps/epoch:        $STEPS_PER_EPOCH"
echo "Val fraction:       $LIMIT_VAL_BATCHES"
echo "Tag:                ${TAG:-<none>}"
echo "============================================================"
echo ""

# ===================== Launch training =====================

TRAINING_CMD="uv run python training/train_ddp_multidataset.py \
    --model_config \"$MODEL_CONFIG\" \
    --dataset_configs_path \"$DATASET_CONFIGS_PATH\" \
    --max_datasets $MAX_DATASETS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --core_lr_scale $CORE_LR_SCALE \
    --lr_scheduler $LR_SCHEDULER \
    --warmup_epochs $WARMUP_EPOCHS \
    --weight_decay $WD \
    --max_epochs $MAX_EPOCHS \
    --precision $PRECISION \
    --dset_dtype $DSET_DTYPE \
    --num_gpus $NUM_GPUS \
    --project_name \"$PROJECT_NAME\" \
    --experiment_name \"$EXPERIMENT_NAME\" \
    --checkpoint_dir \"$CHECKPOINT_DIR\" \
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
    --gradient_clip_val $GRADIENT_CLIP_VAL \
    --steps_per_epoch $STEPS_PER_EPOCH \
    --num_workers $NUM_WORKERS \
    --limit_val_batches $LIMIT_VAL_BATCHES"

if [ "$COMPILE_MODEL" = true ]; then
    TRAINING_CMD="$TRAINING_CMD --compile"
fi

if [ "$ENABLE_LOGGING" = true ]; then
    TRAINING_CMD="$TRAINING_CMD --enable_logging"
fi

if [ -n "$TIME_BUDGET" ]; then
    TRAINING_CMD="$TRAINING_CMD --time_budget_minutes $TIME_BUDGET"
fi

eval $TRAINING_CMD
EXIT_CODE=$?

# ===================== Report result =====================

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully: $MODEL_CONFIG_NAME"
    echo "Checkpoints saved to: $CHECKPOINT_DIR"
else
    echo "Training FAILED for $MODEL_CONFIG_NAME (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE
