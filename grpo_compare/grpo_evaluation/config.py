"""
Configuration for GRPO implementation comparison experiment.
"""

import os

# Base model to use for all implementations
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Dataset configuration
DATASET_CONFIG = {
    "name": "gsm8k",
    "split": "main",
    "val_size": 0.1,  # Portion of training data to use for validation
}

# Common training parameters across implementations
COMMON_TRAINING_CONFIG = {
    "num_epochs": 5,
    "max_steps": 20000,
    "learning_rate": 1e-5,
    "batch_size": 128,
    "mini_batch_size": 8,
    "beta": 0.1,  # KL penalty coefficient
    "preference_threshold": 0.1,
    "num_sample_pairs": 64,
    "paired_temperature_range": (0.7, 1.3),
    "max_grad_norm": 1.0,
    "eval_every": 200,
    "save_every": 500,
    "max_prompt_length": 1024,
    "max_completion_length": 512,
    "epsilon": 0.2,  # Clipping parameter
    "scale_rewards": True,  # Whether to normalize rewards
    "use_vllm": True,  # Use vLLM for generation
    "vllm_port": 8000,
    "seed": 42,
}

# Implementation-specific configurations
IMPLEMENTATION_CONFIGS = {
    "trl": {
        "loss_type": "bnpo",  # One of: "grpo", "bnpo", "dr_grpo"
        "use_reference_model": True,
        "mask_truncated_completions": False,
    },
    "verl": {
        "sequence_balance": True,
        "log_prob_micro_batch_size": 128,
        "kl_loss_type": "low_var_kl",
    },
    "unsloth": {
        "load_in_4bit": True,
        "use_flash_attention": True,
        "memory_efficient": True,
        "enable_fused_projections": True,
    },
    "simplerl": {
        "max_response_length": 3072,  # Support longer math reasoning
        "entropy_coefficient": 0.001,
        "kl_loss_type": "low_var_kl",
    },
    "search_r1": {
        "use_retrieval": False,  # No retrieval for math tasks
        "kl_loss_type": "low_var_kl",
    },
    "tinyzero": {
        "sequence_balance": True,
        "rollout_n": 8,
    },
}

# Output directories
OUTPUT_DIR = "./results"
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# Ensure all directories exist
for dir_path in [OUTPUT_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Wandb configuration
WANDB_CONFIG = {
    "project": "grpo-implementation-comparison",
    "entity": None,  # Change to your wandb entity
    "log_model": True,
}

# Hardware monitoring
MONITOR_HARDWARE = True 