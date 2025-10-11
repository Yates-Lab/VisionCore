#!/bin/bash
# Training script for ViViT model
# Usage: bash experiments/train_vivit.sh [small|baseline]

set -e  # Exit on error

# Parse arguments
MODEL_SIZE=${1:-small}  # Default to small for faster experimentation

# Configuration
if [ "$MODEL_SIZE" = "small" ]; then
    MODEL_CONFIG="experiments/model_configs/vivit_small.yaml"
    BATCH_SIZE=32
    ACCUMULATE_GRAD_BATCHES=2
    EXPERIMENT_NAME="vivit_small"
elif [ "$MODEL_SIZE" = "baseline" ]; then
    MODEL_CONFIG="experiments/model_configs/vivit_baseline.yaml"
    BATCH_SIZE=16
    ACCUMULATE_GRAD_BATCHES=4
    EXPERIMENT_NAME="vivit_baseline"
else
    echo "Unknown model size: $MODEL_SIZE"
    echo "Usage: bash experiments/train_vivit.sh [small|baseline]"
    exit 1
fi

# Common settings
DATASET_CONFIGS_PATH="experiments/dataset_configs/multi_basic_120_backimage_all.yaml"
MAX_DATASETS=20
LEARNING_RATE=3.0e-4  # Standard transformer learning rate
CORE_LR_SCALE=1.0
LR_SCHEDULER="cosine"
WARMUP_EPOCHS=10
WEIGHT_DECAY=0.05  # Higher weight decay for transformers
MAX_EPOCHS=200
PRECISION="bf16-mixed"  # Use bfloat16 for better stability
NUM_GPUS=4
PROJECT_NAME="VisionCore-ViViT"
CHECKPOINT_DIR="checkpoints/vivit"
GRADIENT_CLIP_VAL=1.0
STEPS_PER_EPOCH=1000
NUM_WORKERS=8
EARLY_STOPPING_PATIENCE=50
EARLY_STOPPING_MIN_DELTA=0.0

# Print configuration
echo "=========================================="
echo "Training ViViT Model"
echo "=========================================="
echo "Model size: $MODEL_SIZE"
echo "Model config: $MODEL_CONFIG"
echo "Batch size: $BATCH_SIZE"
echo "Accumulate grad batches: $ACCUMULATE_GRAD_BATCHES"
echo "Effective batch size: $((BATCH_SIZE * ACCUMULATE_GRAD_BATCHES * NUM_GPUS))"
echo "Learning rate: $LEARNING_RATE"
echo "Weight decay: $WEIGHT_DECAY"
echo "Max epochs: $MAX_EPOCHS"
echo "Precision: $PRECISION"
echo "Number of GPUs: $NUM_GPUS"
echo "=========================================="

# Launch training
python training/train_ddp_multidataset.py \
    --model_config "$MODEL_CONFIG" \
    --dataset_configs_path "$DATASET_CONFIGS_PATH" \
    --max_datasets $MAX_DATASETS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --core_lr_scale $CORE_LR_SCALE \
    --lr_scheduler $LR_SCHEDULER \
    --warmup_epochs $WARMUP_EPOCHS \
    --weight_decay $WEIGHT_DECAY \
    --max_epochs $MAX_EPOCHS \
    --precision $PRECISION \
    --num_gpus $NUM_GPUS \
    --project_name "$PROJECT_NAME" \
    --experiment_name "$EXPERIMENT_NAME" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
    --gradient_clip_val $GRADIENT_CLIP_VAL \
    --steps_per_epoch $STEPS_PER_EPOCH \
    --num_workers $NUM_WORKERS \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --early_stopping_min_delta $EARLY_STOPPING_MIN_DELTA

echo "=========================================="
echo "Training completed!"
echo "=========================================="

