"""
Visualization utilities for generating comparison reports.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union

def generate_comparison_report(
    results: Dict[str, Dict], 
    output_file: str,
    responses: Optional[Dict[str, List[str]]] = None,
    impl_notes: Optional[Dict[str, str]] = None,
) -> None:
    """
    Generate a comprehensive report comparing all implementations
    with visualizations and detailed analysis.
    
    Args:
        results: Dictionary mapping implementation names to their results
        output_file: Path to save the report
        responses: Dictionary mapping implementation names to lists of responses
        impl_notes: Dictionary mapping implementation names to their descriptions
    """
    # Create dataframe from results
    df = pd.DataFrame.from_dict(results, orient='index')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Plot comparison charts
    plot_comparison_charts(df, os.path.join(os.path.dirname(output_file), "grpo_comparison_charts.png"))
    
    # If no implementation notes provided, use default ones
    if impl_notes is None:
        impl_notes = {
            "trl": "Most complete implementation with multiple loss types support.",
            "verl": "Research framework with sequence-balanced training.",
            "unsloth": "Optimized for memory efficiency and speed.",
            "simplerl": "Specialized for mathematical reasoning tasks.",
            "search_r1": "Adapted for search/retrieval-augmented tasks.",
            "tinyzero": "Lightweight implementation for smaller workloads."
        }
    
    # Generate markdown report
    with open(output_file, 'w') as f:
        f.write("# GRPO Implementation Comparison Report\n\n")
        f.write("## Overview\n\n")
        f.write("This report compares different GRPO implementations for fine-tuning ")
        f.write("the Qwen 2.5 3B model on the GSM8K dataset.\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write(df.to_markdown())
        f.write("\n\n")
        
        f.write("## Key Observations\n\n")
        
        # Find best implementation for each key metric
        metrics_to_highlight = ["answer_accuracy", "reasoning_quality", "avg_inference_time_per_sample"]
        for metric in metrics_to_highlight:
            if metric in df.columns:
                if "time" in metric or "token" in metric:
                    # For time and token metrics, lower is better
                    best_impl = df[metric].idxmin()
                    f.write(f"- Best implementation for {metric.replace('_', ' ')}: **{best_impl}** ")
                    f.write(f"({df.loc[best_impl, metric]:.4f})\n")
                else:
                    # For accuracy metrics, higher is better
                    best_impl = df[metric].idxmax()
                    f.write(f"- Best implementation for {metric.replace('_', ' ')}: **{best_impl}** ")
                    f.write(f"({df.loc[best_impl, metric]:.2%})\n")
        
        f.write("\n## Implementation Characteristics\n\n")
        
        # Add implementation-specific notes
        for impl, note in impl_notes.items():
            if impl in results:
                f.write(f"### {impl}\n")
                f.write(f"{note}\n\n")
                
                # Add key metrics for this implementation
                for metric in metrics_to_highlight:
                    if metric in df.columns:
                        if "time" in metric or "token" in metric:
                            f.write(f"- {metric.replace('_', ' ')}: {df.loc[impl, metric]:.4f}\n")
                        else:
                            f.write(f"- {metric.replace('_', ' ')}: {df.loc[impl, metric]:.2%}\n")
                
                # Add sample responses if available
                if responses and impl in responses and len(responses[impl]) > 0:
                    f.write("\n#### Sample Response\n\n")
                    f.write("```\n")
                    f.write(responses[impl][0][:500] + ("..." if len(responses[impl][0]) > 500 else ""))
                    f.write("\n```\n\n")
        
        f.write("## Visualization\n\n")
        f.write("![GRPO Comparison Charts](./grpo_comparison_charts.png)\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("Based on the results of this comparison, we can observe that:\n\n")
        
        # Add some general conclusions based on the data
        accuracy_diff = df["answer_accuracy"].max() - df["answer_accuracy"].min()
        if accuracy_diff < 0.05:
            f.write("- There is minimal difference in answer accuracy across implementations, ")
            f.write("suggesting that the core GRPO algorithm works similarly across all libraries.\n")
        else:
            most_accurate = df["answer_accuracy"].idxmax()
            f.write(f"- The {most_accurate} implementation shows notably better answer accuracy, ")
            f.write(f"which may be due to its specific optimizations.\n")
        
        if "token_efficiency" in df.columns:
            most_efficient = df["token_efficiency"].idxmin()
            f.write(f"- The {most_efficient} implementation is most token-efficient, ")
            f.write("generating more concise answers.\n")
        
        if "avg_inference_time_per_sample" in df.columns:
            fastest = df["avg_inference_time_per_sample"].idxmin()
            f.write(f"- The {fastest} implementation offers the fastest inference time, ")
            f.write("which is important for production applications.\n")
        
        f.write("\nFor production use on mathematical reasoning tasks, we recommend ")
        
        # Make a final recommendation based on a weighted score of key metrics
        weighted_scores = {}
        for impl in results.keys():
            score = 0
            if "answer_accuracy" in df.columns:
                score += df.loc[impl, "answer_accuracy"] * 0.5  # 50% weight to accuracy
            if "reasoning_quality" in df.columns:
                score += df.loc[impl, "reasoning_quality"] * 0.3  # 30% weight to reasoning
            if "avg_inference_time_per_sample" in df.columns:
                # Lower is better for time, so invert the score
                time_score = 1 - (df.loc[impl, "avg_inference_time_per_sample"] / df["avg_inference_time_per_sample"].max())
                score += time_score * 0.2  # 20% weight to speed
            weighted_scores[impl] = score
        
        if weighted_scores:
            recommendation = max(weighted_scores.items(), key=lambda x: x[1])[0]
            f.write(f"the **{recommendation}** implementation, which offers the best balance of accuracy, ")
            f.write("reasoning quality, and performance.")

def plot_comparison_charts(df: pd.DataFrame, output_file: str) -> None:
    """
    Plot comparison charts for the given dataframe.
    
    Args:
        df: Dataframe with results
        output_file: Path to save the plot
    """
    plt.figure(figsize=(14, 10))
    
    # Group metrics by type
    accuracy_metrics = [col for col in df.columns if "accuracy" in col or "quality" in col]
    time_metrics = [col for col in df.columns if "time" in col]
    token_metrics = [col for col in df.columns if "token" in col]
    hardware_metrics = [col for col in df.columns if "memory" in col or "utilization" in col]
    
    # Plot accuracy metrics
    if accuracy_metrics:
        plt.subplot(2, 2, 1)
        df[accuracy_metrics].plot(kind='bar', ax=plt.gca())
        plt.title('Accuracy Metrics by Implementation')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
    
    # Plot time metrics
    if time_metrics:
        plt.subplot(2, 2, 2)
        df[time_metrics].plot(kind='bar', ax=plt.gca())
        plt.title('Time Metrics by Implementation')
        plt.ylabel('Seconds')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
    
    # Plot token metrics
    if token_metrics:
        plt.subplot(2, 2, 3)
        df[token_metrics].plot(kind='bar', ax=plt.gca())
        plt.title('Token Metrics by Implementation')
        plt.ylabel('Tokens')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
    
    # Plot hardware metrics if any
    if hardware_metrics:
        plt.subplot(2, 2, 4)
        df[hardware_metrics].plot(kind='bar', ax=plt.gca())
        plt.title('Hardware Metrics by Implementation')
        plt.ylabel('Value')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)

def save_results_to_json(results: Dict, output_file: str) -> None:
    """
    Save results to a JSON file.
    
    Args:
        results: Results dictionary
        output_file: Path to save the results
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert numpy values to Python native types for JSON serialization
    def convert_to_native_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # Check if it's a numpy type
            return obj.item()
        else:
            return obj
    
    # Convert and save
    with open(output_file, 'w') as f:
        json.dump(convert_to_native_types(results), f, indent=2) 