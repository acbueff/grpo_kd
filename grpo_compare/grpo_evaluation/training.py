"""
Generic training utilities for GRPO implementations.
"""

import os
import time
import importlib
import torch
import wandb
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from transformers import set_seed
from datasets import Dataset

from .config import COMMON_TRAINING_CONFIG, IMPLEMENTATION_CONFIGS

class TrainingMonitor:
    """
    Monitor training progress and collect metrics.
    """
    def __init__(self, use_wandb: bool = True):
        self.use_wandb = use_wandb
        self.start_time = time.time()
        self.metrics = {
            "train_time": 0,
            "peak_memory_usage": 0,
            "num_training_steps": 0,
            "kl_divergence_values": [],
        }
    
    def start_training(self):
        """Start training timer"""
        self.start_time = time.time()
    
    def end_training(self):
        """End training timer and record final metrics"""
        self.metrics["train_time"] = time.time() - self.start_time
        
        try:
            # Record peak memory usage
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # in GB
                self.metrics["peak_memory_usage"] = peak_memory
        except:
            pass
    
    def log_step(self, step: int, metrics: Dict[str, float]):
        """Log metrics for a single training step"""
        self.metrics["num_training_steps"] = max(self.metrics["num_training_steps"], step)
        
        if "kl_divergence" in metrics:
            self.metrics["kl_divergence_values"].append(metrics["kl_divergence"])
        
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        metrics = self.metrics.copy()
        
        # Calculate mean KL divergence if available
        if metrics["kl_divergence_values"]:
            metrics["mean_kl_divergence"] = sum(metrics["kl_divergence_values"]) / len(metrics["kl_divergence_values"])
        
        return metrics

def get_implementation_module(implementation: str):
    """
    Import the implementation module dynamically.
    
    Args:
        implementation: Name of the implementation
        
    Returns:
        The imported module
    """
    try:
        module_path = f".implementations.{implementation}_impl"
        module = importlib.import_module(module_path, package="grpo_evaluation")
        return module
    except ImportError as e:
        raise ImportError(f"Could not import {implementation} implementation: {e}")

def train_with_grpo(
    model_name: str,
    implementation: str,
    train_data: Dataset,
    val_data: Dataset,
    output_dir: str,
    config: Optional[Dict] = None,
) -> Tuple[Any, Dict]:
    """
    Generic training function that adapts to different GRPO implementations
    while maintaining consistent evaluation methodology.
    
    Args:
        model_name: The name of the model to fine-tune
        implementation: Name of the implementation to use
        train_data: Training dataset
        val_data: Validation dataset
        output_dir: Directory to save outputs
        config: Additional configuration parameters
        
    Returns:
        Tuple of (trained_model, training_metrics)
    """
    # Combine configs
    combined_config = COMMON_TRAINING_CONFIG.copy()
    
    # Add implementation-specific config
    if implementation in IMPLEMENTATION_CONFIGS:
        for k, v in IMPLEMENTATION_CONFIGS[implementation].items():
            combined_config[k] = v
    
    # Override with user-provided config
    if config:
        for k, v in config.items():
            combined_config[k] = v
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(combined_config["seed"])
    
    # Initialize monitor
    monitor = TrainingMonitor(use_wandb=True)
    
    # Get implementation module
    impl_module = get_implementation_module(implementation)
    
    # Setup GRPO trainer for the implementation
    trainer_setup_fn = getattr(impl_module, f"setup_{implementation}_grpo")
    trainer = trainer_setup_fn(model_name, combined_config)
    
    # Start monitoring
    monitor.start_training()
    
    # Train model
    print(f"Training with {implementation} implementation...")
    trained_model = trainer.train(train_data, val_data, monitor.log_step)
    
    # End monitoring
    monitor.end_training()
    
    # Get training metrics
    training_metrics = monitor.get_metrics()
    
    # Save the trained model
    model_save_path = os.path.join(output_dir, "final_model")
    try:
        if hasattr(trainer, "save_model"):
            trainer.save_model(model_save_path)
        elif hasattr(trained_model, "save_pretrained"):
            trained_model.save_pretrained(model_save_path)
        else:
            print(f"Warning: Could not save model for {implementation}. No save method found.")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    return trained_model, training_metrics

def setup_vllm_client(config: Dict) -> Any:
    """
    Set up a vLLM client for inference.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        vLLM client or None if vLLM is not available
    """
    if not config.get("use_vllm", False):
        return None
    
    try:
        from vllm import LLM, SamplingParams
        
        # Create a simple wrapper for the vLLM client to match the interface
        class VLLMClient:
            def __init__(self, model_name, **kwargs):
                self.llm = LLM(model=model_name, **kwargs)
                
            def generate(self, prompt, max_tokens=100, temperature=0.1, **kwargs):
                sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                outputs = self.llm.generate(prompt, sampling_params)
                return outputs
        
        # Initialize vLLM client
        vllm_client = VLLMClient(
            model_name=config.get("model_name", "Qwen/Qwen2.5-3B-Instruct"),
            tensor_parallel_size=config.get("tensor_parallel_size", 1),
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.8),
        )
        
        return vllm_client
    
    except ImportError:
        print("vLLM not available. Using standard generation.")
        return None 