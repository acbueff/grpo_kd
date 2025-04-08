import torch
import numpy as np
from typing import List, Dict, Callable, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

class RewardCombiner:
    """
    Base class for combining multiple reward components.
    
    This allows flexible combination of different reward signals,
    such as teacher log probability and language modeling loss.
    """
    
    def __init__(self):
        """Initialize reward combiner."""
        pass
    
    def __call__(
        self, 
        rewards: Dict[str, Union[float, torch.Tensor]]
    ) -> float:
        """
        Combine multiple rewards into a single scalar reward.
        
        Args:
            rewards: Dictionary mapping reward names to values
            
        Returns:
            Combined reward value
        """
        raise NotImplementedError("Subclasses must implement __call__")

class LinearRewardCombiner(RewardCombiner):
    """
    Linear combination of rewards with configurable weights.
    
    Computes: sum(weight_i * reward_i for i in rewards)
    """
    
    def __init__(
        self, 
        weights: Dict[str, float]
    ):
        """
        Initialize linear reward combiner.
        
        Args:
            weights: Dictionary mapping reward names to their weights
        """
        super().__init__()
        self.weights = weights
    
    def __call__(
        self, 
        rewards: Dict[str, Union[float, torch.Tensor]]
    ) -> float:
        """
        Compute weighted sum of rewards.
        
        Args:
            rewards: Dictionary mapping reward names to values
            
        Returns:
            Weighted sum of rewards
        """
        combined_reward = 0.0
        
        for name, value in rewards.items():
            if name in self.weights:
                # Convert tensor to float if necessary
                weight = self.weights[name]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                
                combined_reward += weight * value
        
        return combined_reward

class NonlinearRewardCombiner(RewardCombiner):
    """
    Nonlinear combination of rewards.
    
    Allows for more complex reward shaping through arbitrary
    combination functions.
    """
    
    def __init__(
        self, 
        combine_fn: Callable[[Dict[str, float]], float]
    ):
        """
        Initialize nonlinear reward combiner.
        
        Args:
            combine_fn: Function that takes a dictionary of rewards and returns a scalar
        """
        super().__init__()
        self.combine_fn = combine_fn
    
    def __call__(
        self, 
        rewards: Dict[str, Union[float, torch.Tensor]]
    ) -> float:
        """
        Apply combination function to rewards.
        
        Args:
            rewards: Dictionary mapping reward names to values
            
        Returns:
            Combined reward value
        """
        # Convert any tensors to floats
        float_rewards = {}
        for name, value in rewards.items():
            if isinstance(value, torch.Tensor):
                float_rewards[name] = value.item()
            else:
                float_rewards[name] = value
        
        return self.combine_fn(float_rewards)

class MiniLLMRewardCombiner(RewardCombiner):
    """
    Specific combiner for MiniLLM-style rewards.
    
    Combines teacher log probability with language modeling loss (L_PT)
    using the formula: teacher_logprob - λ * lpt_loss
    """
    
    def __init__(
        self, 
        teacher_logprob_weight: float = 1.0,
        lpt_weight: float = 0.1,
        normalize_teacher: bool = True,
    ):
        """
        Initialize MiniLLM reward combiner.
        
        Args:
            teacher_logprob_weight: Weight for teacher log probability reward
            lpt_weight: Weight for language modeling loss (λ)
            normalize_teacher: Whether to normalize teacher log probability by response length
        """
        super().__init__()
        self.teacher_logprob_weight = teacher_logprob_weight
        self.lpt_weight = lpt_weight
        self.normalize_teacher = normalize_teacher
    
    def __call__(
        self, 
        rewards: Dict[str, Union[float, torch.Tensor]],
        response_length: Optional[int] = None,
    ) -> float:
        """
        Combine teacher log probability and L_PT loss.
        
        Args:
            rewards: Dictionary with "teacher_logprob" and "lpt_loss" keys
            response_length: Length of response in tokens (for normalization)
            
        Returns:
            Combined reward
        """
        if "teacher_logprob" not in rewards:
            raise ValueError("Missing required reward component: teacher_logprob")
        
        teacher_reward = rewards["teacher_logprob"]
        if isinstance(teacher_reward, torch.Tensor):
            teacher_reward = teacher_reward.item()
        
        # Normalize by response length if requested and available
        if self.normalize_teacher and response_length is not None:
            teacher_reward = teacher_reward / response_length
        
        # Get L_PT loss if available
        lpt_loss = 0.0
        if "lpt_loss" in rewards:
            lpt_loss = rewards["lpt_loss"]
            if isinstance(lpt_loss, torch.Tensor):
                lpt_loss = lpt_loss.item()
        
        # Combine: teacher_logprob - λ * lpt_loss
        combined_reward = (
            self.teacher_logprob_weight * teacher_reward - 
            self.lpt_weight * lpt_loss
        )
        
        return combined_reward

class FaroeseRewardCombiner(RewardCombiner):
    """
    Faroese-specific reward combiner that prioritizes Faroese capabilities.
    
    Adds bonuses for proper handling of Faroese-specific constructs and
    characters in the generated outputs.
    """
    
    def __init__(
        self, 
        teacher_logprob_weight: float = 1.0,
        lpt_weight: float = 0.1,
        faroese_bonus_weight: float = 0.2,
    ):
        """
        Initialize Faroese reward combiner.
        
        Args:
            teacher_logprob_weight: Weight for teacher log probability reward
            lpt_weight: Weight for language modeling loss (λ)
            faroese_bonus_weight: Weight for Faroese-specific bonuses
        """
        super().__init__()
        self.teacher_logprob_weight = teacher_logprob_weight
        self.lpt_weight = lpt_weight
        self.faroese_bonus_weight = faroese_bonus_weight
        
        # Faroese-specific characters and patterns
        self.faroese_chars = set("ðøáíóúýæÐØÁÍÓÚÝÆ")
    
    def __call__(
        self, 
        rewards: Dict[str, Union[float, torch.Tensor]],
        response: Optional[str] = None,
    ) -> float:
        """
        Combine rewards with Faroese-specific bonuses.
        
        Args:
            rewards: Dictionary of reward components
            response: The generated response text (for calculating Faroese bonuses)
            
        Returns:
            Combined reward
        """
        # Get base reward components
        teacher_reward = rewards.get("teacher_logprob", 0.0)
        if isinstance(teacher_reward, torch.Tensor):
            teacher_reward = teacher_reward.item()
        
        lpt_loss = rewards.get("lpt_loss", 0.0)
        if isinstance(lpt_loss, torch.Tensor):
            lpt_loss = lpt_loss.item()
        
        # Add Faroese-specific bonus if response is provided
        faroese_bonus = 0.0
        if response is not None:
            faroese_bonus = self._calculate_faroese_bonus(response)
        
        # Combine all components
        combined_reward = (
            self.teacher_logprob_weight * teacher_reward - 
            self.lpt_weight * lpt_loss +
            self.faroese_bonus_weight * faroese_bonus
        )
        
        return combined_reward
    
    def _calculate_faroese_bonus(self, response: str) -> float:
        """
        Calculate bonus reward for proper use of Faroese constructs.
        
        Args:
            response: Generated text response
            
        Returns:
            Bonus reward value
        """
        # Count Faroese-specific characters in response
        faroese_char_count = sum(1 for char in response if char in self.faroese_chars)
        
        # Calculate density of Faroese characters
        faroese_char_density = faroese_char_count / max(1, len(response))
        
        # Simple bonus based on Faroese character density
        # This is a placeholder - a more sophisticated approach would
        # look at grammatical constructs, word usage, etc.
        bonus = faroese_char_density * 10.0  # Scale to reasonable range
        
        return bonus 