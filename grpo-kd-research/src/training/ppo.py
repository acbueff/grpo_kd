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

class PPO:
    """
    Proximal Policy Optimization (PPO) Implementation with Teacher Rewards
    
    This class implements PPO with teacher log-likelihood reward and
    optionally with language modeling loss (L_PT) in the MiniLLM-style.
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        value_model: nn.Module,
        teacher_model: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        value_optimizer: torch.optim.Optimizer,
        teacher_reward: TeacherLogProbReward,
        lpt_reward: Optional[LanguageModelingReward] = None,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        lpt_weight: float = 0.0,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = 0.015,
        device: str = "cuda",
        fp16: bool = False,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Initialize PPO trainer.
        
        Args:
            student_model: The policy model being trained
            value_model: The critic model estimating values
            teacher_model: Larger model providing rewards
            policy_optimizer: Optimizer for policy model
            value_optimizer: Optimizer for value model
            teacher_reward: Reward function based on teacher log-probability
            lpt_reward: Optional language modeling loss reward
            clip_param: PPO clipping parameter epsilon
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            lpt_weight: Weight for language modeling loss (λ)
            max_grad_norm: Maximum norm for gradient clipping
            target_kl: Target KL divergence threshold for early stopping
            device: Device to run on ("cuda" or "cpu")
            fp16: Whether to use mixed precision
            gamma: Discount factor for rewards
            gae_lambda: GAE lambda parameter
        """
        self.student_model = student_model
        self.value_model = value_model
        self.teacher_model = teacher_model
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.teacher_reward = teacher_reward
        self.lpt_reward = lpt_reward
        
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.lpt_weight = lpt_weight
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.device = device
        self.fp16 = fp16
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Set up mixed precision if needed
        self.scaler = torch.cuda.amp.GradScaler() if fp16 else None
        
        # Track metrics
        self.metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "lpt_loss": [],
            "total_loss": [],
            "approx_kl": [],
            "clip_fraction": [],
            "mean_reward": [],
        }
    
    def collect_rollouts(self, prompt_batch: List[str]) -> Dict[str, torch.Tensor]:
        """
        Collect rollouts by generating responses and computing rewards.
        
        Args:
            prompt_batch: Batch of prompts for generating responses
            
        Returns:
            Dictionary with rollout data
        """
        self.student_model.eval()
        self.value_model.eval()
        self.teacher_model.eval()
        
        with torch.no_grad():
            rollouts = {
                "prompts": [],
                "responses": [],
                "log_probs": [],
                "values": [],
                "rewards": [],
                "advantages": [],
                "returns": [],
            }
            
            # Generate one response per prompt
            for prompt in prompt_batch:
                # Generate response
                response = self.student_model.generate(prompt)
                
                # Compute log probability under current policy
                log_prob = self.student_model.log_prob(response, prompt)
                
                # Estimate value
                value = self.value_model(prompt, response)
                
                # Compute reward from teacher
                reward = self.teacher_reward(response, prompt)
                
                # Store in rollouts
                rollouts["prompts"].append(prompt)
                rollouts["responses"].append(response)
                rollouts["log_probs"].append(log_prob)
                rollouts["values"].append(value)
                rollouts["rewards"].append(reward)
            
            # Convert lists to tensors
            rollouts["log_probs"] = torch.stack(rollouts["log_probs"]).to(self.device)
            rollouts["values"] = torch.stack(rollouts["values"]).to(self.device)
            rollouts["rewards"] = torch.tensor(rollouts["rewards"], device=self.device)
            
            # Compute advantages and returns using GAE
            advantages = self._compute_advantages(
                rollouts["rewards"], 
                rollouts["values"]
            )
            returns = advantages + rollouts["values"]
            
            rollouts["advantages"] = advantages
            rollouts["returns"] = returns
            
            # Normalize advantages
            rollouts["advantages"] = (rollouts["advantages"] - rollouts["advantages"].mean()) / (rollouts["advantages"].std() + 1e-8)
            
        return rollouts
    
    def _compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Simplified for language model setting where we have a single step.
        
        Args:
            rewards: Tensor of rewards [batch_size]
            values: Tensor of value estimates [batch_size]
            
        Returns:
            Tensor of advantages [batch_size]
        """
        # For LLM setting, this is a simplified version since we have single-step episodes
        advantages = rewards - values
        return advantages
    
    def update_policy(
        self, 
        rollouts: Dict[str, torch.Tensor], 
        epochs: int = 4,
        pt_batch: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Update policy using collected rollouts.
        
        Args:
            rollouts: Dictionary with rollout data
            epochs: Number of optimization epochs
            pt_batch: Optional batch of pretraining data for L_PT
            
        Returns:
            Dictionary with updated metrics
        """
        self.student_model.train()
        self.value_model.train()
        
        batch_size = len(rollouts["prompts"])
        epoch_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "lpt_loss": [],
            "total_loss": [],
            "approx_kl": [],
            "clip_fraction": [],
        }
        
        # Compute L_PT loss if enabled
        lpt_loss = 0
        if self.lpt_weight > 0 and self.lpt_reward is not None and pt_batch is not None:
            with torch.cuda.amp.autocast() if self.fp16 else nullcontext():
                lpt_loss = self.lpt_reward(self.student_model, pt_batch)
            epoch_metrics["lpt_loss"].append(lpt_loss.item())
        
        for epoch in range(epochs):
            # We could shuffle here, but for simplicity we'll process all at once
            
            # Compute current log probs, values, and entropy
            with torch.cuda.amp.autocast() if self.fp16 else nullcontext():
                current_log_probs = []
                current_values = []
                entropies = []
                
                for i in range(batch_size):
                    prompt = rollouts["prompts"][i]
                    response = rollouts["responses"][i]
                    
                    # Get current log prob
                    log_prob = self.student_model.log_prob(response, prompt)
                    current_log_probs.append(log_prob)
                    
                    # Get entropy estimate
                    entropy = self.student_model.entropy(prompt, response)
                    entropies.append(entropy)
                    
                    # Get value estimate
                    value = self.value_model(prompt, response)
                    current_values.append(value)
                
                current_log_probs = torch.stack(current_log_probs).to(self.device)
                current_values = torch.stack(current_values).to(self.device)
                entropy = torch.stack(entropies).mean()
                
                # Compute policy loss (clipped surrogate objective)
                log_ratio = current_log_probs - rollouts["log_probs"]
                ratio = torch.exp(log_ratio)
                
                surrogate1 = ratio * rollouts["advantages"]
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * rollouts["advantages"]
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(current_values, rollouts["returns"])
                
                # Compute KL divergence for logging and early stopping
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                
                # Compute clipping fraction for logging
                clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                
                # Compute total loss
                total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Add L_PT loss if enabled
                if self.lpt_weight > 0 and lpt_loss != 0:
                    total_loss += self.lpt_weight * lpt_loss
            
            # Update policy
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            
            if self.fp16:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.policy_optimizer)
                self.scaler.unscale_(self.value_optimizer)
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.max_grad_norm)
                self.scaler.step(self.policy_optimizer)
                self.scaler.step(self.value_optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()
            
            # Store metrics
            epoch_metrics["policy_loss"].append(policy_loss.item())
            epoch_metrics["value_loss"].append(value_loss.item())
            epoch_metrics["entropy"].append(entropy.item())
            epoch_metrics["total_loss"].append(total_loss.item())
            epoch_metrics["approx_kl"].append(approx_kl)
            epoch_metrics["clip_fraction"].append(clip_fraction)
            
            # Early stop if KL divergence is too high
            if self.target_kl is not None and approx_kl > self.target_kl:
                logger.info(f"Early stopping at epoch {epoch+1}/{epochs} due to reaching target KL: {approx_kl:.4f} > {self.target_kl:.4f}")
                break
        
        # Update overall metrics
        self.metrics["policy_loss"].append(np.mean(epoch_metrics["policy_loss"]))
        self.metrics["value_loss"].append(np.mean(epoch_metrics["value_loss"]))
        self.metrics["entropy"].append(np.mean(epoch_metrics["entropy"]))
        if self.lpt_weight > 0:
            self.metrics["lpt_loss"].append(np.mean(epoch_metrics["lpt_loss"]) if epoch_metrics["lpt_loss"] else 0)
        self.metrics["total_loss"].append(np.mean(epoch_metrics["total_loss"]))
        self.metrics["approx_kl"].append(np.mean(epoch_metrics["approx_kl"]))
        self.metrics["clip_fraction"].append(np.mean(epoch_metrics["clip_fraction"]))
        self.metrics["mean_reward"].append(rollouts["rewards"].mean().item())
        
        return {
            "policy_loss": np.mean(epoch_metrics["policy_loss"]),
            "value_loss": np.mean(epoch_metrics["value_loss"]),
            "entropy": np.mean(epoch_metrics["entropy"]),
            "lpt_loss": np.mean(epoch_metrics["lpt_loss"]) if epoch_metrics["lpt_loss"] else 0,
            "total_loss": np.mean(epoch_metrics["total_loss"]),
            "approx_kl": np.mean(epoch_metrics["approx_kl"]),
            "clip_fraction": np.mean(epoch_metrics["clip_fraction"]),
            "mean_reward": rollouts["rewards"].mean().item(),
        }
    
    def train_step(
        self, 
        prompt_batch: List[str],
        update_epochs: int = 4,
        pt_batch: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Perform a full PPO training step: collect rollouts and update policy.
        
        Args:
            prompt_batch: Batch of prompts for generating responses
            update_epochs: Number of policy update epochs per rollout collection
            pt_batch: Optional batch of pretraining data for L_PT
            
        Returns:
            Dictionary with metrics
        """
        # Collect rollouts
        rollouts = self.collect_rollouts(prompt_batch)
        
        # Update policy multiple times on the same rollout batch
        metrics = self.update_policy(rollouts, update_epochs, pt_batch)
        
        return metrics
    
    def save(self, path: str, epoch: int, step: int) -> None:
        """Save model checkpoint"""
        save_checkpoint(
            {
                "policy": self.student_model,
                "value": self.value_model,
            },
            {
                "policy": self.policy_optimizer,
                "value": self.value_optimizer,
            },
            path,
            epoch,
            step,
            self.metrics,
        )

class nullcontext:
    """A minimal context manager for compatibility with torch.autocast when fp16=False"""
    def __enter__(self): return self
    def __exit__(self, *args): pass 