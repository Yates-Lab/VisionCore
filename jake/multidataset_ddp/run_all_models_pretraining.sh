#!/bin/bash

# Batch training script to run all model configurations with pretraining.
# Uses learned_res_small as the pretrained vision model for all modulator experiments.
# This script automatically finds the best learned_res_small checkpoint and uses it for pretraining.

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
# conda activate yatesfv
conda activate yatesfv-tejas

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=WARN
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# NCCL settings that fixed the DDP hang
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=1 
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# set up lcuda for compilation
mkdir -p $HOME/.local/lib          # or another dir you control
ln -s /lib/x86_64-linux-gnu/libcuda.so.1 $HOME/.local/lib/libcuda.so

export LIBRARY_PATH="$HOME/.local/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"

# Debug env (helps surface OOMs and CUDA errors)
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Training configuration
BATCH_SIZE=256          # Optimal batch size per GPU
MAX_DATASETS=30        # Scale to all datasets (28 if removed the two bad sessions)
LEARNING_RATE=1e-3     # Lower LR for pretraining fine-tuning
CORE_LR_SCALE=0.5      # Much lower LR for pretrained components
LR_SCHEDULER="cosine_warmup"  # Use cosine annealing with warmup
WARMUP_EPOCHS=5        # Number of warmup epochs
WEIGHT_DECAY=1e-4
MAX_EPOCHS=200        # Long training run with early stopping protection
PRECISION="bf16-mixed"  
NUM_GPUS=2             # Use both RTX 6000 Ada GPUs
NUM_WORKERS=16         # Optimized for 64 CPU cores
STEPS_PER_EPOCH=512   # Number of steps per epoch

# Pretraining configuration
# PRETRAINED_MODEL_TYPE="learned_res2d_small"
PRETRAINED_MODEL_TYPE="learned_res_small_30"
FREEZE_VISION=false    # Set to true for frozen vision experiments
ENABLE_COMPILE=false   # Set to true to enable torch.compile (disable for STN modulators)

# Project and data paths
# PROJECT_NAME="multidataset_pretraining_120"
# DATASET_CONFIGS_PATH="/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset/dataset_basic_multi_120"
# CHECKPOINT_DIR="/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120/checkpoints"
# PRETRAINED_CHECKPOINT_DIR="/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120/checkpoints"

PROJECT_NAME="multidataset_pretraining_30"
DATASET_CONFIGS_PATH="/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset/dataset_basic_multi_30"
CHECKPOINT_DIR="/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_30/checkpoints"
PRETRAINED_CHECKPOINT_DIR="/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_30/checkpoints"

# Create checkpoint directory
mkdir -p $CHECKPOINT_DIR

# Array of model configurations to run with pretraining
MODEL_CONFIGS=(
    # "configs_multi/learned_res_small.yaml"
    # "configs_multi/learned_res_small.yaml"     # Base model (will be trained from scratch if first)
    # "configs_multi/learned_res_small_gru.yaml" # ConvGRU modulator
    # "configs_multi/learned_res_small_none_gru_none_pool.yaml" # ConvGRU recurrent (no modulator)
    # "configs_multi/learned_res2d_small.yaml" # 2D resnet no pooling
    # "configs_multi/learned_res2d_small_none_gru_none_pool.yaml" # 2D resnet no pooling
    "configs_multi/learned_res_small_30.yaml"     # Base model (trained first from scratch)
    "configs_multi/learned_res_small_gru_30.yaml" # ConvGRU modulator
    # "configs_multi/learned_res_small_none_gru.yaml" # ConvGRU recurrent (no modulator)
    # "configs_multi/learned_res_tiny_pc.yaml" # Predictive Coding modulator
    # "configs_multi/learned_res_tiny_film.yaml"      # FiLM modulator
    # "configs_multi/learned_res_small_stn.yaml"  # Spatial Transformer Network modulator
    # Add more modulator configs here as you create them
)

# Function to find the best pretrained checkpoint
find_pretrained_checkpoint() {
    echo "Finding best $PRETRAINED_MODEL_TYPE checkpoint..."
    
    # Use the find_pretrained_checkpoint.py script to get the best checkpoint
    PRETRAINED_CHECKPOINT=$(python find_pretrained_checkpoint.py \
        --model_type "$PRETRAINED_MODEL_TYPE" \
        --checkpoint_dir "$PRETRAINED_CHECKPOINT_DIR" 2>/dev/null | \
        grep "Path:" | cut -d' ' -f4)
    
    if [ -z "$PRETRAINED_CHECKPOINT" ] || [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
        echo "❌ Could not find pretrained checkpoint for $PRETRAINED_MODEL_TYPE"
        echo "Available model types:"
        python find_pretrained_checkpoint.py --checkpoint_dir "$PRETRAINED_CHECKPOINT_DIR" --list_all
        exit 1
    fi
    
    echo "✓ Found pretrained checkpoint: $PRETRAINED_CHECKPOINT"
    return 0
}

# Function to run training for a single model config with pretraining
run_training() {
    local MODEL_CONFIG=$1
    local MODEL_CONFIG_NAME=$(basename "$MODEL_CONFIG" .yaml)
    
    # Create experiment name with pretraining suffix
    local FREEZE_SUFFIX=""
    if [ "$FREEZE_VISION" = true ]; then
        FREEZE_SUFFIX="_frozen"
    else
        FREEZE_SUFFIX="_finetune"
    fi
    
    local EXPERIMENT_NAME="${MODEL_CONFIG_NAME}_pretrained${FREEZE_SUFFIX}_ddp_bs${BATCH_SIZE}_ds${MAX_DATASETS}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}_corelrscale${CORE_LR_SCALE}_warmup${WARMUP_EPOCHS}"
    
    echo ""
    echo "============================================================"
    echo "STARTING PRETRAINING: $MODEL_CONFIG_NAME"
    echo "============================================================"
    echo "Model config: $MODEL_CONFIG"
    echo "Experiment: $EXPERIMENT_NAME"
    echo "Pretrained checkpoint: $PRETRAINED_CHECKPOINT"
    echo "Freeze vision: $FREEZE_VISION"
    echo "Compile enabled: $ENABLE_COMPILE"
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
    echo "GPUs: $NUM_GPUS"
    echo "Workers: $NUM_WORKERS"
    echo "Dataset configs: $DATASET_CONFIGS_PATH"
    echo "Checkpoint dir: $CHECKPOINT_DIR"
    echo "============================================================"
    
    # Build the training command
    local TRAINING_CMD="python train_ddp_multidataset.py \
        --model_config \"$MODEL_CONFIG\" \
        --dataset_configs_path \"$DATASET_CONFIGS_PATH\" \
        --pretrained_checkpoint \"$PRETRAINED_CHECKPOINT\" \
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
        --project_name \"$PROJECT_NAME\" \
        --experiment_name \"$EXPERIMENT_NAME\" \
        --checkpoint_dir \"$CHECKPOINT_DIR\" \
        --accumulate_grad_batches 1 \
        --gradient_clip_val 1.0 \
        --steps_per_epoch $STEPS_PER_EPOCH \
        --num_workers $NUM_WORKERS \
        --early_stopping_patience 50 \
        --early_stopping_min_delta 0.0"
    
    # Add freeze_vision flag if enabled
    if [ "$FREEZE_VISION" = true ]; then
        TRAINING_CMD="$TRAINING_CMD --freeze_vision"
    fi

    # Add compile flag if enabled
    if [ "$ENABLE_COMPILE" = true ]; then
        TRAINING_CMD="$TRAINING_CMD --compile"
    fi

    # Launch training
    eval $TRAINING_CMD
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Pretraining completed successfully for $MODEL_CONFIG_NAME"
    else
        echo "❌ Pretraining failed for $MODEL_CONFIG_NAME (exit code: $exit_code)"
        return $exit_code
    fi
}

# Function to run baseline (no pretraining) for comparison
run_baseline_training() {
    local MODEL_CONFIG=$1
    local MODEL_CONFIG_NAME=$(basename "$MODEL_CONFIG" .yaml)
    local EXPERIMENT_NAME="${MODEL_CONFIG_NAME}_baseline_ddp_bs${BATCH_SIZE}_ds${MAX_DATASETS}_lr1e-3_wd${WEIGHT_DECAY}_corelrscale0.5_warmup${WARMUP_EPOCHS}"
    
    echo ""
    echo "============================================================"
    echo "STARTING BASELINE (NO PRETRAINING): $MODEL_CONFIG_NAME"
    echo "============================================================"
    
    # Build the baseline training command
    local BASELINE_CMD="python train_ddp_multidataset.py \
        --model_config \"$MODEL_CONFIG\" \
        --dataset_configs_path \"$DATASET_CONFIGS_PATH\" \
        --max_datasets $MAX_DATASETS \
        --batch_size $BATCH_SIZE \
        --learning_rate 1e-3 \
        --core_lr_scale 0.5 \
        --lr_scheduler $LR_SCHEDULER \
        --warmup_epochs $WARMUP_EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --max_epochs $MAX_EPOCHS \
        --precision $PRECISION \
        --num_gpus $NUM_GPUS \
        --project_name \"$PROJECT_NAME\" \
        --experiment_name \"$EXPERIMENT_NAME\" \
        --checkpoint_dir \"$CHECKPOINT_DIR\" \
        --accumulate_grad_batches 1 \
        --gradient_clip_val 1.0 \
        --steps_per_epoch $STEPS_PER_EPOCH \
        --num_workers $NUM_WORKERS \
        --early_stopping_patience 40 \
        --early_stopping_min_delta 0.0"

    # Add compile flag if enabled
    if [ "$ENABLE_COMPILE" = true ]; then
        BASELINE_CMD="$BASELINE_CMD --compile"
    fi

    # Launch baseline training
    eval $BASELINE_CMD
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Baseline training completed successfully for $MODEL_CONFIG_NAME"
    else
        echo "❌ Baseline training failed for $MODEL_CONFIG_NAME (exit code: $exit_code)"
        return $exit_code
    fi
}

# Function to check if a config matches the pretrained model type
config_matches_pretrained_type() {
    local config_path=$1
    local config_name=$(basename "$config_path" .yaml)

    # Check if config name matches pretrained model type
    if [ "$config_name" = "$PRETRAINED_MODEL_TYPE" ]; then
        return 0  # Match found
    fi
    return 1  # No match
}

# Main execution
echo "============================================================"
echo "BATCH PRETRAINING: ALL MODEL CONFIGURATIONS"
echo "============================================================"
echo "Pretrained model type: $PRETRAINED_MODEL_TYPE"
echo "Freeze vision components: $FREEZE_VISION"
echo "Compile enabled: $ENABLE_COMPILE"
echo "Found ${#MODEL_CONFIGS[@]} model configurations:"
for config in "${MODEL_CONFIGS[@]}"; do
    echo "  - $(basename "$config" .yaml)"
done
echo ""
echo "Starting batch pretraining at $(date)"
echo "============================================================"

# Check if first model matches pretrained type
FIRST_MODEL_IS_PRETRAINED=false
if config_matches_pretrained_type "${MODEL_CONFIGS[0]}"; then
    FIRST_MODEL_IS_PRETRAINED=true
    echo "✓ First model ($(basename "${MODEL_CONFIGS[0]}" .yaml)) matches pretrained type - will train from scratch"
    PRETRAINED_CHECKPOINT=""  # Will be set after first model trains
else
    echo "First model doesn't match pretrained type - looking for existing checkpoint..."
    # Try to find the best pretrained checkpoint
    if find_pretrained_checkpoint; then
        echo "✓ Found existing pretrained checkpoint - will use for all models"
    else
        echo "⚠️  No existing pretrained checkpoint found - will train first model from scratch"
        FIRST_MODEL_IS_PRETRAINED=true
        PRETRAINED_CHECKPOINT=""  # Will be set after first model trains
    fi
fi

# Track overall progress
TOTAL_CONFIGS=${#MODEL_CONFIGS[@]}
COMPLETED=0
FAILED=0
FAILED_CONFIGS=()

# Run pretraining for each model configuration
for i in "${!MODEL_CONFIGS[@]}"; do
    MODEL_CONFIG="${MODEL_CONFIGS[$i]}"
    MODEL_CONFIG_NAME=$(basename "$MODEL_CONFIG" .yaml)

    echo ""
    echo ">>> PROGRESS: $((i+1))/$TOTAL_CONFIGS - Starting $MODEL_CONFIG_NAME <<<"

    # Handle first model if it's the pretrained type
    if [ $i -eq 0 ] && [ "$FIRST_MODEL_IS_PRETRAINED" = true ]; then
        echo "Training base model from scratch (no pretraining)..."

        # Run baseline training for the first model (no pretraining)
        if run_baseline_training "$MODEL_CONFIG"; then
            COMPLETED=$((COMPLETED + 1))
            echo ">>> PROGRESS: $((i+1))/$TOTAL_CONFIGS - Completed $MODEL_CONFIG_NAME ✓ <<<"

            # Find the checkpoint from this training to use for subsequent models
            echo "Finding checkpoint from newly trained $MODEL_CONFIG_NAME..."
            PRETRAINED_CHECKPOINT=$(python find_pretrained_checkpoint.py \
                --model_type "$PRETRAINED_MODEL_TYPE" \
                --checkpoint_dir "$CHECKPOINT_DIR" 2>/dev/null | \
                grep "Path:" | cut -d' ' -f4)

            if [ -z "$PRETRAINED_CHECKPOINT" ] || [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
                echo "❌ Could not find checkpoint from newly trained $MODEL_CONFIG_NAME"
                FAILED=$((FAILED + 1))
                FAILED_CONFIGS+=("$MODEL_CONFIG_NAME")
                break
            else
                echo "✓ Will use checkpoint for subsequent models: $PRETRAINED_CHECKPOINT"
            fi
        else
            FAILED=$((FAILED + 1))
            FAILED_CONFIGS+=("$MODEL_CONFIG_NAME")
            echo ">>> PROGRESS: $((i+1))/$TOTAL_CONFIGS - Failed $MODEL_CONFIG_NAME ❌ <<<"

            # Ask user if they want to continue after failure
            echo ""
            echo "Base model training failed for $MODEL_CONFIG_NAME. Continue with remaining configs? (y/n)"
            read -r response
            if [[ ! "$response" =~ ^[Yy]$ ]]; then
                echo "Stopping batch training as requested."
                break
            fi
        fi
    else
        # Regular pretraining for subsequent models
        echo "Training with pretraining from: $PRETRAINED_CHECKPOINT"

        if run_training "$MODEL_CONFIG"; then
            COMPLETED=$((COMPLETED + 1))
            echo ">>> PROGRESS: $((i+1))/$TOTAL_CONFIGS - Completed $MODEL_CONFIG_NAME ✓ <<<"
        else
            FAILED=$((FAILED + 1))
            FAILED_CONFIGS+=("$MODEL_CONFIG_NAME")
            echo ">>> PROGRESS: $((i+1))/$TOTAL_CONFIGS - Failed $MODEL_CONFIG_NAME ❌ <<<"

            # Ask user if they want to continue after failure
            echo ""
            echo "Pretraining failed for $MODEL_CONFIG_NAME. Continue with remaining configs? (y/n)"
            read -r response
            if [[ ! "$response" =~ ^[Yy]$ ]]; then
                echo "Stopping batch pretraining as requested."
                break
            fi
        fi
    fi
done

# Final summary
echo ""
echo "============================================================"
echo "BATCH PRETRAINING SUMMARY"
echo "============================================================"
echo "Completed at: $(date)"
if [ "$FIRST_MODEL_IS_PRETRAINED" = true ]; then
    echo "Base model trained from scratch: $(basename "${MODEL_CONFIGS[0]}" .yaml)"
fi
echo "Pretrained checkpoint used: $PRETRAINED_CHECKPOINT"
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
    echo "🎉 All model configurations completed successfully with pretraining!"
    echo ""
    echo "💡 NEXT STEPS:"
    echo "1. Run baseline comparisons: Set FREEZE_VISION=false and run again"
    echo "2. Evaluate results: Use eval_stack_multidataset.py to compare models"
    echo "3. Compare pretrained vs baseline performance"
else
    echo "⚠️  Some configurations failed. Check logs above for details."
fi
echo "============================================================"
