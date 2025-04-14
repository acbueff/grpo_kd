#!/usr/bin/env python3
"""
Top-level script to run the GRPO implementation comparison.
"""

import os
import sys
import argparse

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GRPO Implementation Comparison")
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Name of the model to fine-tune")
    
    parser.add_argument("--implementations", type=str, nargs="+",
                        default=["trl", "unsloth"],
                        help="List of implementations to evaluate")
    
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    
    parser.add_argument("--eval_only", action="store_true",
                        help="Only run evaluation on pretrained models")
    
    parser.add_argument("--use_wandb", action="store_true",
                        help="Log metrics to Weights & Biases")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    parser.add_argument("--eval_samples", type=int, default=100,
                        help="Number of samples to use for evaluation")
    
    return parser.parse_args()

def main():
    """Run the evaluation"""
    # Add the parent directory to the Python path
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Parse command line arguments
    args = parse_args()
    
    # Import the main module
    from grpo_evaluation.main import main as run_evaluation
    
    # Run the evaluation
    sys.exit(run_evaluation())

if __name__ == "__main__":
    main() 