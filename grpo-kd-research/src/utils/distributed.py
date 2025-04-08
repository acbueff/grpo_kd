import os
import torch
import torch.distributed as dist
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def init_distributed(
    local_rank: int,
    backend: str = "nccl",
    port: Optional[int] = None,
) -> Tuple[bool, int, int]:
    """
    Initialize distributed training environment.
    
    Args:
        local_rank: Local rank of this process (passed by torch.distributed.launch)
        backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
        port: Port to use for communication (if None, use env var or default)
        
    Returns:
        Tuple of (is_distributed, world_size, global_rank)
    """
    # Check if distributed training is enabled
    if not torch.cuda.is_available() or local_rank == -1:
        return False, 1, 0  # Not distributed
    
    # Check if already initialized
    if dist.is_initialized():
        return True, dist.get_world_size(), dist.get_rank()
    
    # Get environment variables
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    
    # Set master port
    if port is None:
        master_port = os.environ.get("MASTER_PORT", "29500")
    else:
        master_port = str(port)
    
    # Update environment variables if needed
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    
    logger.info(f"Initialized distributed training with backend={backend}, "
                f"world_size={world_size}, rank={rank}, local_rank={local_rank}, "
                f"master={master_addr}:{master_port}")
    
    return True, world_size, rank

def is_distributed() -> bool:
    """
    Check if process group is initialized for distributed training.
    
    Returns:
        True if distributed training is enabled
    """
    return dist.is_initialized()

def get_world_size() -> int:
    """
    Get world size (number of processes).
    
    Returns:
        Number of processes or 1 if not distributed
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1

def get_rank() -> int:
    """
    Get global rank of current process.
    
    Returns:
        Global rank of current process or 0 if not distributed
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0

def is_master() -> bool:
    """
    Check if current process is the master process.
    
    Returns:
        True if this process is master (rank 0) or not distributed
    """
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True

def get_local_rank() -> int:
    """
    Get local rank of current process (for setting device).
    
    Returns:
        Local rank of process within node
    """
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return 0

def setup_device_from_rank() -> torch.device:
    """
    Set up the appropriate device based on rank.
    
    Returns:
        Torch device for this process
    """
    if torch.cuda.is_available():
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    return device

def barrier() -> None:
    """
    Synchronize all processes.
    
    This function blocks until all processes reach this barrier.
    """
    if dist.is_initialized():
        dist.barrier()

def cleanup() -> None:
    """
    Clean up distributed training environment.
    """
    if dist.is_initialized():
        dist.destroy_process_group()

def all_gather_object(obj: object) -> list:
    """
    Gather objects from all processes.
    
    Args:
        obj: Object to gather
        
    Returns:
        List of objects gathered from all processes
    """
    # If not distributed, return singleton list
    if not dist.is_initialized():
        return [obj]
    
    # All gather objects
    world_size = dist.get_world_size()
    gathered_objects = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_objects, obj)
    
    return gathered_objects

def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """
    Reduce dictionary values across processes.
    
    Args:
        input_dict: Dictionary with tensors to reduce
        average: If True, average the values; otherwise, sum them
        
    Returns:
        Dictionary with reduced values
    """
    # If not distributed, return input dict
    if not dist.is_initialized():
        return input_dict
    
    world_size = dist.get_world_size()
    if world_size == 1:
        return input_dict
    
    # Prepare keys and values
    names = []
    values = []
    for k, v in sorted(input_dict.items()):
        names.append(k)
        values.append(v.clone().detach())
    
    # Reduce values
    dist.all_reduce_coalesced(values, dist.ReduceOp.SUM)
    
    # Average if requested
    if average:
        values = [v / world_size for v in values]
    
    # Create reduced dict
    reduced_dict = {k: v for k, v in zip(names, values)}
    
    return reduced_dict

def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute mean of tensor across all processes.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Mean of tensor across all processes
    """
    # If not distributed, return input tensor
    if not dist.is_initialized():
        return tensor
    
    # Clone and detach tensor to avoid modifying the original
    tensor = tensor.clone().detach()
    
    # All-reduce
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # Divide by world size to get mean
    tensor = tensor / dist.get_world_size()
    
    return tensor 