import os
import logging
import json
import yaml
from typing import Dict, Any, Optional, Union, List
import time
from datetime import datetime
import wandb
import sys

from .distributed import is_master

def setup_logging(
    output_dir: str,
    experiment_name: str,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up logging for training.
    
    Args:
        output_dir: Directory to save log files
        experiment_name: Name of the experiment
        log_level: Logging level
        
    Returns:
        Logger instance
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Add console handler for all processes
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler only for master process
    if is_master():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"{experiment_name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_config(
    config: Dict[str, Any],
    output_dir: str,
    experiment_name: str,
    logger: logging.Logger,
) -> None:
    """
    Log configuration to file and console.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save config file
        experiment_name: Name of the experiment
        logger: Logger instance
    """
    if not is_master():
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config to YAML file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = os.path.join(output_dir, f"{experiment_name}_{timestamp}_config.yaml")
    
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Log config summary
    logger.info(f"Configuration saved to {config_file}")
    logger.info("Configuration summary:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {key}: {value}")

def init_tracking(
    project_name: str,
    experiment_name: str,
    config: Dict[str, Any],
    tracking_uri: Optional[str] = None,
) -> Union[str, Any]:
    """
    Initialize experiment tracking with W&B.
    
    Args:
        project_name: Name of the W&B project
        experiment_name: Name of this experiment
        config: Configuration to log
        tracking_uri: Optional URI for tracking server
        
    Returns:
        Run ID or tracking object
    """
    if not is_master():
        return None
    
    # Initialize W&B
    wandb.init(
        project=project_name,
        name=experiment_name,
        config=config,
        dir=os.path.join(os.getcwd(), "wandb"),
    )
    
    return wandb.run.id

def log_metrics(
    metrics: Dict[str, float],
    step: int,
    epoch: Optional[int] = None,
    prefix: str = "",
) -> None:
    """
    Log metrics to tracking system.
    
    Args:
        metrics: Dictionary of metric name to value
        step: Current step
        epoch: Optional epoch number
        prefix: Optional prefix for metric names
    """
    if not is_master():
        return
    
    # Prepare metrics with prefix
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    
    # Add epoch if provided
    if epoch is not None:
        metrics["epoch"] = epoch
    
    # Log to W&B
    if wandb.run is not None:
        wandb.log(metrics, step=step)

def log_model_summary(
    model,
    input_shapes: List[tuple],
    logger: logging.Logger,
) -> None:
    """
    Log model summary with parameter count and structure.
    
    Args:
        model: Model to summarize
        input_shapes: List of input shapes for model forward pass
        logger: Logger instance
    """
    if not is_master():
        return
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Model summary:")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable/Total: {trainable_params/max(1, total_params):.2%}")
    
    # Log to W&B
    if wandb.run is not None:
        wandb.run.summary["params/trainable"] = trainable_params
        wandb.run.summary["params/total"] = total_params
        wandb.run.summary["params/trainable_pct"] = trainable_params/max(1, total_params)

def log_validation_samples(
    prompts: List[str],
    responses: List[str],
    references: Optional[List[str]] = None,
    step: int = 0,
    max_samples: int = 5,
) -> None:
    """
    Log validation samples to tracking system.
    
    Args:
        prompts: List of input prompts
        responses: List of model responses
        references: Optional list of reference responses
        step: Current step
        max_samples: Maximum number of samples to log
    """
    if not is_master() or not wandb.run:
        return
    
    # Limit number of samples
    n_samples = min(len(prompts), max_samples)
    
    # Create table data
    table_data = []
    for i in range(n_samples):
        sample = {
            "prompt": prompts[i],
            "response": responses[i],
        }
        if references is not None and i < len(references):
            sample["reference"] = references[i]
        table_data.append(sample)
    
    # Create table
    columns = ["prompt", "response"]
    if references is not None:
        columns.append("reference")
    
    table = wandb.Table(columns=columns)
    for sample in table_data:
        row = [sample["prompt"], sample["response"]]
        if "reference" in sample:
            row.append(sample["reference"])
        table.add_data(*row)
    
    # Log table
    wandb.log({"validation_samples": table}, step=step)

def finish_run() -> None:
    """
    Finish the tracking run.
    """
    if is_master() and wandb.run:
        wandb.finish()

class Timer:
    """
    Simple timer for tracking execution time.
    """
    
    def __init__(self, name: str = "Timer"):
        """
        Initialize timer.
        
        Args:
            name: Name of the timer
        """
        self.name = name
        self.start_time = None
        self.splits = []
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()
        self.splits = []
    
    def split(self, split_name: str = "") -> float:
        """
        Record a split time.
        
        Args:
            split_name: Name of the split
            
        Returns:
            Time since last split or start
        """
        if self.start_time is None:
            self.start()
        
        current_time = time.time()
        if not self.splits:
            split_time = current_time - self.start_time
        else:
            split_time = current_time - self.splits[-1][1]
        
        self.splits.append((split_name, current_time, split_time))
        
        return split_time
    
    def stop(self) -> float:
        """
        Stop the timer.
        
        Returns:
            Total elapsed time
        """
        if self.start_time is None:
            return 0.0
        
        end_time = time.time()
        total_time = end_time - self.start_time
        
        self.splits.append(("end", end_time, 0.0))
        
        return total_time
    
    def log(self, logger: Optional[logging.Logger] = None) -> Dict[str, float]:
        """
        Log timer results.
        
        Args:
            logger: Logger to use (if None, use default)
            
        Returns:
            Dictionary of split name to time
        """
        if self.start_time is None:
            return {}
        
        if not self.splits or self.splits[-1][0] != "end":
            self.stop()
        
        total_time = self.splits[-1][1] - self.start_time
        
        log_func = logger.info if logger else print
        log_func(f"{self.name} total time: {total_time:.4f}s")
        
        # Log individual splits if there are any named splits
        named_splits = [(name, time) for name, _, time in self.splits if name and name != "end"]
        if named_splits:
            log_func(f"{self.name} splits:")
            split_dict = {}
            for name, time in named_splits:
                log_func(f"  {name}: {time:.4f}s ({time/total_time:.1%})")
                split_dict[name] = time
            
            return split_dict
        
        return {"total": total_time} 