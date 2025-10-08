#!/usr/bin/env python3
"""
Helper script to find the best pretrained checkpoint for vision component loading.

Usage:
    python find_pretrained_checkpoint.py --model_type learned_res_small
    python find_pretrained_checkpoint.py --checkpoint_dir /path/to/checkpoints
"""

import argparse
from pathlib import Path
from eval_stack_utils import scan_checkpoints

def main():
    parser = argparse.ArgumentParser(description="Find best pretrained checkpoint")
    parser.add_argument("--model_type", type=str, default="learned_res_small",
                       help="Type of pretrained model to find")
    parser.add_argument("--checkpoint_dir", type=str, 
                       default="/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset/checkpoints",
                       help="Directory containing checkpoints")
    parser.add_argument("--list_all", action="store_true", 
                       help="List all available model types")
    
    args = parser.parse_args()
    
    print(f"Scanning checkpoint directory: {args.checkpoint_dir}")
    models_by_type = scan_checkpoints(args.checkpoint_dir)
    
    if args.list_all:
        print(f"\nFound {len(models_by_type)} model types:")
        for model_type, models in models_by_type.items():
            if models:
                best_model = models[0]
                if best_model['val_loss'] is not None:
                    best_metric = f"best loss: {best_model['val_loss']:.4f}"
                elif best_model['val_bps'] is not None:
                    best_metric = f"best BPS: {best_model['val_bps']:.4f}"
                else:
                    best_metric = "no metric"
            else:
                best_metric = "N/A"
            print(f"  {model_type}: {len(models)} models ({best_metric})")
        return
    
    if args.model_type not in models_by_type:
        print(f"❌ Model type '{args.model_type}' not found!")
        print(f"Available types: {list(models_by_type.keys())}")
        return
    
    # Find best checkpoint
    try:
        # Get the best model (already sorted by val_loss)
        best_model_info = models_by_type[args.model_type][0]
        best_checkpoint = str(best_model_info['path'])
        
        print(f"\n✓ Best {args.model_type} checkpoint:")
        print(f"  Path: {best_checkpoint}")

        # Display the best metric (BPS preferred over loss)
        if best_model_info['val_bps'] is not None:
            print(f"  Validation BPS: {best_model_info['val_bps']:.4f}")
        elif best_model_info['val_loss'] is not None:
            print(f"  Validation Loss: {best_model_info['val_loss']:.4f}")
        else:
            print(f"  Validation Metric: N/A")

        print(f"  Epoch: {best_model_info['epoch']}")
        print(f"  Experiment: {best_model_info['experiment']}")

        # Show command line usage
        print(f"\n📋 Usage in training command:")
        print(f"  --pretrained_checkpoint {best_checkpoint}")

        # Show top 3 checkpoints
        print(f"\n📊 Top 3 {args.model_type} checkpoints:")
        for i, model_info in enumerate(models_by_type[args.model_type][:3]):
            if model_info['val_bps'] is not None:
                metric_str = f"BPS: {model_info['val_bps']:.4f}"
            elif model_info['val_loss'] is not None:
                metric_str = f"loss: {model_info['val_loss']:.4f}"
            else:
                metric_str = "no metric"
            print(f"  {i+1}. {Path(model_info['path']).name} ({metric_str})")
            
    except Exception as e:
        print(f"❌ Error finding checkpoint: {e}")

if __name__ == "__main__":
    main()
