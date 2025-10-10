#!/bin/bash

# Example script showing how to train with Zero-Inflated Poisson loss
# This demonstrates the different ways to enable ZIP loss

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yatesfv

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# ============================================================================
# METHOD 1: Direct command-line flag (Recommended)
# ============================================================================
echo "============================================================"
echo "METHOD 1: Using --loss_type command-line flag"
echo "============================================================"

python training/train_ddp_multidataset.py \
    --model_config experiments/model_configs/res_small_gru.yaml \
    --dataset_configs_path experiments/dataset_configs/multi_basic_120_backimage_all.yaml \
    --loss_type zip \
    --batch_size 256 \
    --learning_rate 1e-3 \
    --core_lr_scale 0.5 \
    --weight_decay 1e-4 \
    --max_epochs 100 \
    --num_gpus 2 \
    --project_name "zip_loss_demo" \
    --experiment_name "res_small_gru_zip_demo"

# ============================================================================
# METHOD 2: Using model config file
# ============================================================================
echo ""
echo "============================================================"
echo "METHOD 2: Using model config with loss_type: zip"
echo "============================================================"

# Use the example config that has loss_type: zip in the YAML
python training/train_ddp_multidataset.py \
    --model_config experiments/model_configs/res_small_gru_zip.yaml \
    --dataset_configs_path experiments/dataset_configs/multi_basic_120_backimage_all.yaml \
    --batch_size 256 \
    --learning_rate 1e-3 \
    --core_lr_scale 0.5 \
    --weight_decay 1e-4 \
    --max_epochs 100 \
    --num_gpus 2 \
    --project_name "zip_loss_demo" \
    --experiment_name "res_small_gru_zip_from_config"

# ============================================================================
# METHOD 3: Override config with command-line
# ============================================================================
echo ""
echo "============================================================"
echo "METHOD 3: Override config file with command-line flag"
echo "============================================================"

# Even if config says loss_type: poisson, --loss_type zip will override it
python training/train_ddp_multidataset.py \
    --model_config experiments/model_configs/res_small_gru.yaml \
    --dataset_configs_path experiments/dataset_configs/multi_basic_120_backimage_all.yaml \
    --loss_type zip \
    --batch_size 256 \
    --learning_rate 1e-3 \
    --max_epochs 100 \
    --num_gpus 2 \
    --project_name "zip_loss_demo" \
    --experiment_name "res_small_gru_zip_override"

# ============================================================================
# COMPARISON: Train both Poisson and ZIP for comparison
# ============================================================================
echo ""
echo "============================================================"
echo "COMPARISON: Training with both Poisson and ZIP loss"
echo "============================================================"

# Train with standard Poisson
python training/train_ddp_multidataset.py \
    --model_config experiments/model_configs/res_small_gru.yaml \
    --dataset_configs_path experiments/dataset_configs/multi_basic_120_backimage_all.yaml \
    --loss_type poisson \
    --batch_size 256 \
    --learning_rate 1e-3 \
    --max_epochs 100 \
    --num_gpus 2 \
    --project_name "poisson_vs_zip" \
    --experiment_name "res_small_gru_poisson"

# Train with ZIP
python training/train_ddp_multidataset.py \
    --model_config experiments/model_configs/res_small_gru.yaml \
    --dataset_configs_path experiments/dataset_configs/multi_basic_120_backimage_all.yaml \
    --loss_type zip \
    --batch_size 256 \
    --learning_rate 1e-3 \
    --max_epochs 100 \
    --num_gpus 2 \
    --project_name "poisson_vs_zip" \
    --experiment_name "res_small_gru_zip"

echo ""
echo "============================================================"
echo "Training complete! Check WandB for comparison."
echo "============================================================"

