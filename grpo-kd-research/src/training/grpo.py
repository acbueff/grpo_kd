import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any
import logging

from ..utils.distributed import is_distributed, get_world_size, get_rank
from ..utils.checkpoint import save_checkpoint
from ..rewards.teacher_logprob import TeacherLogProbReward
from ..rewards.lpt_reward import LanguageModelingReward

logger = logging.getLogger(__name__)

class GRPO:
    """
    Group Relative Policy Optimization (GRPO) Implementation
    
    This class implements GRPO with teacher log-likelihood reward and
    optionally with language modeling loss (L_PT) in the MiniLLM-style.
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        reference_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        teacher_reward: TeacherLogProbReward,
        lpt_reward: Optional[LanguageModelingReward] = None,
        group_size: int = 4,
        kl_weight: float = 0.1,
        lpt_weight: float = 0.0,
        device: str = "cuda",
        fp16: bool = False,
        epsilon: float = 1e-8,
    ):
        """
        Initialize GRPO trainer.
        
        Args:
            student_model: The model being trained (policy)
            teacher_model: Larger model providing rewards
            reference_model: Fixed copy of student model for KL regularization
            optimizer: Optimizer for updating student model
            teacher_reward: Reward function based on teacher log-probability
            lpt_reward: Optional language modeling loss reward
            group_size: Number of responses to generate per prompt (G)
            kl_weight: Weight for KL divergence term (β)
            lpt_weight: Weight for language modeling loss (λ)
            device: Device to run on ("cuda" or "cpu")
            fp16: Whether to use mixed precision
            epsilon: Small constant for numerical stability
        """
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.reference_model = reference_model
        self.optimizer = optimizer
        self.teacher_reward = teacher_reward
        self.lpt_reward = lpt_reward
        self.group_size = group_size
        self.kl_weight = kl_weight
        self.lpt_weight = lpt_weight
        self.device = device
        self.fp16 = fp16
        self.epsilon = epsilon
        
        # Set up mixed precision if needed
        self.scaler = torch.cuda.amp.GradScaler() if fp16 else None
        
        # Track metrics
        self.metrics = {
            "policy_loss": [],
            "kl_loss": [],
            "lpt_loss": [],
            "total_loss": [],
            "mean_reward": [],
            "mean_advantage": [],
        }
    
    def train_step(
        self, 
        prompt_batch: List[str],
        pt_batch: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Perform a single GRPO training step.
        
        Args:
            prompt_batch: Batch of prompts for generating responses
            pt_batch: Optional batch of pretraining data for L_PT
            
        Returns:
            Dictionary with loss metrics
        """
        self.student_model.train()
        self.teacher_model.eval()
        self.reference_model.eval()
        
        trajectories = []
        batch_metrics = {
            "rewards": [],
            "advantages": [],
        }
        
        # Step 1: Generate responses and compute rewards
        for prompt in prompt_batch:
            # Generate G responses per prompt
            with torch.no_grad():
                outputs = [
                    self.student_model.generate(prompt) 
                    for _ in range(self.group_size)
                ]
            
            # Get rewards from teacher
            with torch.no_grad():
                rewards = [
                    self.teacher_reward(output, prompt) 
                    for output in outputs
                ]
            
            batch_metrics["rewards"].extend(rewards)
            
            # Compute group baseline and advantages
            mean_reward = sum(rewards) / len(rewards)
            std_reward = torch.std(torch.tensor(rewards, device=self.device))
            advantages = [
                (r - mean_reward) / (std_reward + self.epsilon) 
                for r in rewards
            ]
            
            batch_metrics["advantages"].extend(advantages)
            
            # Store trajectories
            for j in range(self.group_size):
                trajectories.append((
                    prompt, 
                    outputs[j], 
                    rewards[j], 
                    advantages[j]
                ))
        
        # Step 2: Compute losses
        policy_loss = 0
        kl_loss = 0
        lpt_loss = 0
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Context manager for mixed precision
        with torch.cuda.amp.autocast() if self.fp16 else nullcontext():
            for prompt, output, reward, advantage in trajectories:
                # Policy gradient loss
                log_prob = self.student_model.log_prob(output, prompt)
                policy_loss += -advantage * log_prob
                
                # KL divergence
                with torch.no_grad():
                    ref_log_prob = self.reference_model.log_prob(output, prompt)
                kl_div = log_prob - ref_log_prob
                kl_loss += kl_div
            
            # Average losses
            policy_loss = policy_loss / len(trajectories)
            kl_loss = kl_loss / len(trajectories)
            
            # L_PT loss if enabled
            if self.lpt_weight > 0 and self.lpt_reward is not None and pt_batch is not None:
                lpt_loss = self.lpt_reward(self.student_model, pt_batch)
                
            # Total loss
            total_loss = policy_loss + self.kl_weight * kl_loss
            if self.lpt_weight > 0:
                total_loss += self.lpt_weight * lpt_loss
        
        # Step 3: Update student model
        if self.fp16:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()
        
        # Track metrics
        self.metrics["policy_loss"].append(policy_loss.item())
        self.metrics["kl_loss"].append(kl_loss.item())
        if self.lpt_weight > 0:
            self.metrics["lpt_loss"].append(lpt_loss.item())
        self.metrics["total_loss"].append(total_loss.item())
        self.metrics["mean_reward"].append(torch.mean(torch.tensor(batch_metrics["rewards"])).item())
        self.metrics["mean_advantage"].append(torch.mean(torch.tensor(batch_metrics["advantages"])).item())
        
        return {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "lpt_loss": lpt_loss.item() if self.lpt_weight > 0 else 0,
            "total_loss": total_loss.item(),
            "mean_reward": torch.mean(torch.tensor(batch_metrics["rewards"])).item(),
            "mean_advantage": torch.mean(torch.tensor(batch_metrics["advantages"])).item(),
        }
    
    def save(self, path: str, epoch: int, step: int) -> None:
        """Save model checkpoint"""
        save_checkpoint(
            self.student_model,
            self.optimizer,
            path,
            epoch,
            step,
            self.metrics,
        )
        
class nullcontext:
    """A minimal context manager for compatibility with torch.autocast when fp16=False"""
    def __enter__(self): return self
    def __exit__(self, *args): pass 