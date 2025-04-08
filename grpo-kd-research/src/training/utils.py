import torch
import torch.nn as nn
import numpy as np
import random
import os
from typing import List, Tuple, Dict, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_reference_copy(model: nn.Module) -> nn.Module:
    """
    Create a deep copy of a model for reference policy.
    
    Args:
        model: Original model
        
    Returns:
        Copied model with identical parameters
    """
    reference_model = type(model)(**(model.config.to_dict() if hasattr(model, 'config') else {}))
    reference_model.load_state_dict(model.state_dict())
    
    # Ensure reference model doesn't receive gradients
    for param in reference_model.parameters():
        param.requires_grad = False
    
    return reference_model

def create_optimizer(
    model: nn.Module, 
    lr: float = 5e-5, 
    weight_decay: float = 0.01,
    optimizer_type: str = "adamw",
    betas: Tuple[float, float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    """
    Create optimizer for model parameters.
    
    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient
        optimizer_type: Type of optimizer (adamw, adam, sgd)
        betas: Beta parameters for Adam-based optimizers
        
    Returns:
        Optimizer instance
    """
    # Filter out parameters that don't require gradients
    parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    
    # Optional: Set different LR for different parameter groups
    # e.g., lower LR for embeddings, higher for task-specific layers
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in parameters if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in parameters if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    if optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=betas,
        )
    elif optimizer_type.lower() == "adam":
        return torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=lr,
            betas=betas,
        )
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=lr,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def compute_group_statistics(rewards: List[float]) -> Tuple[float, float]:
    """
    Compute mean and standard deviation of rewards within a group.
    
    Args:
        rewards: List of reward values
        
    Returns:
        Tuple of (mean, std) of the rewards
    """
    rewards_tensor = torch.tensor(rewards)
    mean = torch.mean(rewards_tensor).item()
    std = torch.std(rewards_tensor).item()
    return mean, std

def pad_to_length(
    token_ids: List[int], 
    max_length: int, 
    pad_token_id: int
) -> Tuple[List[int], List[int]]:
    """
    Pad token IDs to a fixed length, and create an attention mask.
    
    Args:
        token_ids: List of token IDs
        max_length: Maximum length to pad to
        pad_token_id: Token ID to use for padding
        
    Returns:
        Tuple of (padded_token_ids, attention_mask)
    """
    if len(token_ids) > max_length:
        # Truncate
        token_ids = token_ids[:max_length]
        attention_mask = [1] * max_length
    else:
        # Pad
        padding = [pad_token_id] * (max_length - len(token_ids))
        attention_mask = [1] * len(token_ids) + [0] * len(padding)
        token_ids = token_ids + padding
        
    return token_ids, attention_mask

def prepare_faroese_text(text: str) -> str:
    """
    Prepare Faroese text for processing.
    
    Args:
        text: Input text
        
    Returns:
        Processed text
    """
    # Ensure Faroese characters are properly handled
    # This is mostly a placeholder for more sophisticated preprocessing
    faroese_chars = "ðøáíóúýæ"
    
    # Check if any Faroese-specific characters are missing or corrupted
    for char in faroese_chars:
        if char in text and ord(char) > 127:
            # Characters seem to be properly encoded
            return text
            
    # If needed, convert from alternative representations
    replacements = {
        "dh": "ð",
        "oe": "ø",
        "aa": "á",
        "ii": "í",
        "oo": "ó",
        "uu": "ú",
        "yy": "ý",
        "ae": "æ",
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
        
    return text

def get_available_devices() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    device_info = {
        "cpu_available": True,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_devices": [],
    }
    
    if device_info["cuda_available"]:
        for i in range(device_info["cuda_device_count"]):
            device = torch.cuda.get_device_properties(i)
            device_info["cuda_devices"].append({
                "name": device.name,
                "total_memory": device.total_memory,
                "capability": f"{device.major}.{device.minor}",
            })
    
    return device_info 