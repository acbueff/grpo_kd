"""
Main script for evaluating GRPO implementations.
"""

import os
import argparse
import json
import wandb
from transformers import set_seed, AutoTokenizer

from .config import BASE_MODEL, IMPLEMENTATION_CONFIGS, WANDB_CONFIG, OUTPUT_DIR, MODELS_DIR, REPORTS_DIR
from .data_utils import prepare_gsm8k_for_grpo
from .training import train_with_grpo, setup_vllm_client
from .evaluation import evaluate_math_performance, log_hardware_metrics
from .visualization import generate_comparison_report, save_results_to_json

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate GRPO implementations")
    
    parser.add_argument("--model_name", type=str, default=BASE_MODEL,
                        help="Name of the model to fine-tune")
    
    parser.add_argument("--implementations", type=str, nargs="+",
                        default=["trl", "unsloth"],
                        help="List of implementations to evaluate")
    
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
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
    """Run the GRPO implementation comparison"""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=WANDB_CONFIG["project"],
            entity=WANDB_CONFIG["entity"],
            config={
                "model_name": args.model_name,
                "implementations": args.implementations,
                "seed": args.seed,
            }
        )
    
    # Prepare datasets
    print("Preparing datasets...")
    train_data, val_data = prepare_gsm8k_for_grpo(val_size=0.1, seed=args.seed)
    
    # Setup vLLM client if available
    vllm_client = setup_vllm_client({"model_name": args.model_name, "use_vllm": True})
    
    results = {}
    all_responses = {}
    
    # Train and evaluate each implementation
    for impl in args.implementations:
        print(f"\n{'='*50}")
        print(f"Evaluating implementation: {impl}")
        print(f"{'='*50}")
        
        # Set up output directories
        impl_output_dir = os.path.join(MODELS_DIR, f"{impl}")
        os.makedirs(impl_output_dir, exist_ok=True)
        
        if not args.eval_only:
            # Train model with current implementation
            print(f"Training with {impl} implementation...")
            
            try:
                # Get implementation-specific config
                impl_config = IMPLEMENTATION_CONFIGS.get(impl, {})
                impl_config["output_dir"] = impl_output_dir
                impl_config["use_wandb"] = args.use_wandb
                impl_config["seed"] = args.seed
                
                # Train model
                trained_model, training_metrics = train_with_grpo(
                    model_name=args.model_name,
                    implementation=impl,
                    train_data=train_data,
                    val_data=val_data,
                    output_dir=impl_output_dir,
                    config=impl_config,
                )
                
                # Save training metrics
                save_results_to_json(
                    training_metrics, 
                    os.path.join(impl_output_dir, "training_metrics.json")
                )
                
                print(f"Training with {impl} completed successfully!")
                
            except Exception as e:
                print(f"Error training with {impl}: {e}")
                continue
        else:
            # Try to load pretrained model
            print(f"Skipping training for {impl} (--eval_only flag is set)")
            trained_model = None  # We'll load it during evaluation
        
        # Evaluate model
        print(f"Evaluating {impl} implementation...")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            
            # Load model if in eval_only mode or if training failed
            if trained_model is None:
                # Try to load from the output directory
                from transformers import AutoModelForCausalLM
                model_path = os.path.join(impl_output_dir, "final_model")
                
                if os.path.exists(model_path):
                    print(f"Loading model from {model_path}")
                    trained_model = AutoModelForCausalLM.from_pretrained(model_path)
                else:
                    print(f"Model not found at {model_path}, using the base model")
                    trained_model = AutoModelForCausalLM.from_pretrained(args.model_name)
            
            # Record hardware metrics before evaluation
            if args.use_wandb:
                hardware_metrics = log_hardware_metrics()
                wandb.log({f"{impl}_hw": hardware_metrics})
            
            # Evaluate on validation set
            eval_results, responses = evaluate_math_performance(
                model=trained_model,
                tokenizer=tokenizer,
                eval_dataset=val_data,
                num_samples=args.eval_samples,
                use_vllm=vllm_client is not None,
                vllm_client=vllm_client,
            )
            
            # Store results and responses
            results[impl] = eval_results
            all_responses[impl] = responses
            
            # Log evaluation results
            save_results_to_json(
                eval_results, 
                os.path.join(impl_output_dir, "eval_results.json")
            )
            
            # Log to wandb if enabled
            if args.use_wandb:
                wandb.log({f"{impl}_eval": eval_results})
            
            print(f"Evaluation of {impl} completed successfully!")
            
        except Exception as e:
            print(f"Error evaluating {impl}: {e}")
            continue
    
    # Generate comparison report
    if results:
        print("\nGenerating comparison report...")
        report_path = os.path.join(REPORTS_DIR, "grpo_comparison_report.md")
        generate_comparison_report(results, report_path, all_responses)
        
        # Log final results to wandb
        if args.use_wandb:
            wandb.log({"final_results": results})
        
        # Save all results to a single JSON file
        all_results = {
            "model_name": args.model_name,
            "implementations": args.implementations,
            "results": results,
        }
        save_results_to_json(all_results, os.path.join(REPORTS_DIR, "all_results.json"))
        
        print(f"Comparison report saved to {report_path}")
    else:
        print("No results to report. All implementations failed.")
    
    print("\nExperiment completed!")

if __name__ == "__main__":
    main() 