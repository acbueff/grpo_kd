import os
import torch
import json
import logging
from typing import Dict, Any, Union, Optional, List
from datetime import datetime
import shutil

from .distributed import is_master, get_rank, barrier

logger = logging.getLogger(__name__)

def save_checkpoint(
    model: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
    optimizer: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
    path: str,
    epoch: int,
    step: int,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
    **kwargs
) -> None:
    """
    Save model checkpoint to disk.
    
    In distributed training, only the master process saves.
    
    Args:
        model: Model or dictionary of models to save
        optimizer: Optimizer or dictionary of optimizers
        path: Path to save the checkpoint
        epoch: Current epoch number
        step: Current step number
        metrics: Optional metrics dictionary
        config: Optional configuration dictionary
        is_best: Whether this is the best model so far
        **kwargs: Additional data to save
    """
    # Only save from master process
    if not is_master():
        barrier()  # Wait for master to save
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Prepare state dict
    if isinstance(model, dict):
        model_state = {name: m.state_dict() for name, m in model.items()}
    else:
        model_state = model.state_dict()
    
    # Prepare optimizer state
    if isinstance(optimizer, dict):
        optimizer_state = {name: opt.state_dict() for name, opt in optimizer.items()}
    else:
        optimizer_state = optimizer.state_dict()
    
    # Build checkpoint
    checkpoint = {
        "model": model_state,
        "optimizer": optimizer_state,
        "epoch": epoch,
        "step": step,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Add metrics if provided
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    # Add config if provided
    if config is not None:
        checkpoint["config"] = config
    
    # Add any additional kwargs
    checkpoint.update(kwargs)
    
    # Save checkpoint
    try:
        logger.info(f"Saving checkpoint to {path}")
        torch.save(checkpoint, path)
        
        # If best model, save a copy
        if is_best:
            best_path = os.path.join(os.path.dirname(path), "best_model.pt")
            logger.info(f"Saving best model to {best_path}")
            shutil.copyfile(path, best_path)
        
        # Save human-readable metrics json
        if metrics is not None:
            metrics_path = os.path.splitext(path)[0] + "_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
    
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
    
    # Wait for checkpoint to be saved before other processes continue
    barrier()

def load_checkpoint(
    path: str,
    model: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
    optimizer: Optional[Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]] = None,
    strict: bool = True,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load model checkpoint from disk.
    
    Args:
        path: Path to the checkpoint
        model: Model or dictionary of models to load weights into
        optimizer: Optional optimizer or dictionary of optimizers to load state into
        strict: Whether to strictly enforce that the keys in state_dict match
        map_location: Optional device to map tensors to
        
    Returns:
        Dictionary containing checkpoint data
    """
    logger.info(f"Loading checkpoint from {path}")
    
    # Determine device mapping
    if map_location is None:
        if torch.cuda.is_available():
            map_location = f"cuda:{get_rank() % torch.cuda.device_count()}"
        else:
            map_location = "cpu"
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location=map_location)
    
    # Load model state
    model_state = checkpoint["model"]
    if isinstance(model, dict):
        # Load each model if model is a dictionary
        for name, m in model.items():
            if name in model_state:
                try:
                    m.load_state_dict(model_state[name], strict=strict)
                    logger.info(f"Loaded state for model {name}")
                except Exception as e:
                    logger.error(f"Failed to load state for model {name}: {e}")
            else:
                logger.warning(f"Model {name} not found in checkpoint")
    else:
        # Load single model
        try:
            model.load_state_dict(model_state, strict=strict)
            logger.info("Loaded model state")
        except Exception as e:
            logger.error(f"Failed to load model state: {e}")
    
    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer_state = checkpoint["optimizer"]
        if isinstance(optimizer, dict):
            # Load each optimizer if optimizer is a dictionary
            for name, opt in optimizer.items():
                if name in optimizer_state:
                    try:
                        opt.load_state_dict(optimizer_state[name])
                        logger.info(f"Loaded state for optimizer {name}")
                    except Exception as e:
                        logger.error(f"Failed to load state for optimizer {name}: {e}")
                else:
                    logger.warning(f"Optimizer {name} not found in checkpoint")
        else:
            # Load single optimizer
            try:
                optimizer.load_state_dict(optimizer_state)
                logger.info("Loaded optimizer state")
            except Exception as e:
                logger.error(f"Failed to load optimizer state: {e}")
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}, "
                f"step {checkpoint.get('step', 'unknown')}")
    
    return checkpoint

def get_latest_checkpoint(directory: str, prefix: str = "checkpoint") -> Optional[str]:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        directory: Directory to search
        prefix: Prefix of checkpoint files
        
    Returns:
        Path to latest checkpoint or None if no checkpoints found
    """
    if not os.path.isdir(directory):
        return None
    
    # Find all checkpoint files
    checkpoint_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith(prefix) and f.endswith(".pt")
    ]
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return checkpoint_files[0]

def cleanup_checkpoints(
    directory: str,
    prefix: str = "checkpoint",
    keep: int = 5,
    keep_best: bool = True,
) -> None:
    """
    Clean up old checkpoints, keeping only the specified number.
    
    Args:
        directory: Directory containing checkpoints
        prefix: Prefix of checkpoint files
        keep: Number of recent checkpoints to keep
        keep_best: Whether to always keep the best model
    """
    if not os.path.isdir(directory):
        return
    
    # Find all checkpoint files
    checkpoint_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith(prefix) and f.endswith(".pt") and not f == "best_model.pt"
    ]
    
    if len(checkpoint_files) <= keep:
        return
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Keep the best model if requested
    best_model_path = os.path.join(directory, "best_model.pt")
    if keep_best and os.path.exists(best_model_path):
        if best_model_path in checkpoint_files:
            checkpoint_files.remove(best_model_path)
    
    # Delete old checkpoints
    for checkpoint_file in checkpoint_files[keep:]:
        logger.info(f"Removing old checkpoint: {checkpoint_file}")
        try:
            os.remove(checkpoint_file)
            
            # Also remove associated metrics file if it exists
            metrics_file = os.path.splitext(checkpoint_file)[0] + "_metrics.json"
            if os.path.exists(metrics_file):
                os.remove(metrics_file)
        
        except Exception as e:
            logger.error(f"Failed to remove checkpoint {checkpoint_file}: {e}")

def save_faroese_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: str,
    name: str = "faroese_model",
) -> None:
    """
    Save a Faroese-specific checkpoint with model and tokenizer.
    
    This is a specialized function for saving the trained Faroese model
    and its tokenizer in a format suitable for later use.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        output_dir: Output directory
        name: Name for the model
    """
    # Only save from master process
    if not is_master():
        barrier()
        return
    
    # Create output directory
    model_dir = os.path.join(output_dir, name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    logger.info(f"Saving Faroese model to {model_dir}")
    model.save_pretrained(model_dir)
    
    # Save tokenizer if it has the save_pretrained method
    if hasattr(tokenizer, "save_pretrained"):
        logger.info(f"Saving tokenizer to {model_dir}")
        tokenizer.save_pretrained(model_dir)
    elif hasattr(tokenizer, "base_tokenizer") and hasattr(tokenizer.base_tokenizer, "save_pretrained"):
        # If using our FaroeseTokenizerAdapter
        logger.info(f"Saving base tokenizer to {model_dir}")
        tokenizer.base_tokenizer.save_pretrained(model_dir)
        
        # Save additional Faroese-specific configuration
        config_path = os.path.join(model_dir, "faroese_tokenizer_config.json")
        logger.info(f"Saving Faroese tokenizer config to {config_path}")
        with open(config_path, "w") as f:
            json.dump({
                "faroese_chars": "ðøáíóúýæÐØÁÍÓÚÝÆ",
                "use_prefix": getattr(tokenizer, "use_prefix", False),
                "faroese_prefix": getattr(tokenizer, "faroese_prefix", "🇫🇴 "),
            }, f, indent=2)
    
    # Save model card with Faroese information
    model_card = f"""---
language:
- fo
license: apache-2.0
tags:
- faroese
- knowledge-distillation
- grpo
datasets:
- foqa
- faroese-corpus
---

# Faroese Language Model - {name}

This model was trained specifically for the Faroese language using Group Relative Policy Optimization (GRPO)
with Knowledge Distillation rewards from a larger teacher model.

## Model Details

* **Model Type:** Causal Language Model
* **Training Technique:** {name.split('_')[0].upper()}
* **Language:** Faroese (fo)
* **Base Model:** {getattr(model, "base_model_name", "Unknown")}
* **License:** Apache 2.0

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{name}")
tokenizer = AutoTokenizer.from_pretrained("{name}")

# For best results with Faroese, handle special characters
text = "Hvat eitur høvuðsstaðurin í Føroyum?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Details

This model was trained using {name.split('_')[0].upper()} with Knowledge Distillation from a teacher model.
It was specifically optimized for Faroese language understanding and generation.

## Intended Use

This model is intended for Faroese language tasks, including:
- Question answering
- Text completion
- Content generation
- Linguistic analysis

## Limitations

The model is specialized for Faroese and may have limited capabilities in other languages.
"""
    
    # Save model card
    with open(os.path.join(model_dir, "README.md"), "w") as f:
        f.write(model_card)
    
    logger.info(f"Faroese model successfully saved to {model_dir}")
    
    # Wait for save to complete
    barrier() 