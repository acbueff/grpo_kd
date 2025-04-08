import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

class TeacherLogProbReward:
    """
    Reward function based on teacher model's log probability of student outputs.
    
    This implements the core reward mechanism for knowledge distillation,
    where we want to maximize the likelihood of the student's outputs
    under the teacher's distribution.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        normalize: bool = False,
        sequence_level: bool = True,
        temperature: float = 1.0,
        device: str = "cuda",
    ):
        """
        Initialize teacher log probability reward.
        
        Args:
            teacher_model: Teacher model (larger model)
            normalize: Whether to normalize log probabilities by sequence length
            sequence_level: Whether to compute log probs at sequence level vs token level
            temperature: Temperature for teacher model (higher = more uniform)
            device: Device to run inference on
        """
        self.teacher_model = teacher_model
        self.normalize = normalize
        self.sequence_level = sequence_level
        self.temperature = temperature
        self.device = device
        
        # Ensure teacher is in eval mode
        self.teacher_model.eval()
    
    def __call__(
        self, 
        response: str, 
        prompt: str
    ) -> float:
        """
        Compute reward as log probability of response given prompt under teacher model.
        
        Args:
            response: Generated response from student model
            prompt: Input prompt
            
        Returns:
            Log probability reward (higher = better)
        """
        with torch.no_grad():
            if self.sequence_level:
                reward = self._compute_sequence_level_logprob(response, prompt)
            else:
                reward = self._compute_token_level_logprob(response, prompt)
            
            return reward
    
    def _compute_sequence_level_logprob(self, response: str, prompt: str) -> float:
        """
        Compute sequence-level log probability of response given prompt.
        
        Args:
            response: Generated response from student model
            prompt: Input prompt
            
        Returns:
            Sequence-level log probability
        """
        # Get full text (prompt + response)
        full_text = prompt + response
        
        # Tokenize full text and prompt
        input_ids_full = self.teacher_model.tokenize(full_text)
        input_ids_prompt = self.teacher_model.tokenize(prompt)
        
        # Get length of prompt in tokens
        prompt_len = len(input_ids_prompt)
        
        # Forward pass through teacher model
        outputs = self.teacher_model(
            input_ids_full.to(self.device),
            return_dict=True,
            use_cache=False,
        )
        
        # Get logits for all tokens
        logits = outputs.logits
        
        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Compute log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Get target tokens (shifted by 1 from input)
        target_ids = input_ids_full[1:].to(self.device)
        
        # Get log prob for each target token
        token_log_probs = torch.gather(
            log_probs[:-1], 
            dim=-1, 
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Only consider log probs of response tokens (not prompt)
        response_log_probs = token_log_probs[prompt_len-1:]
        
        # Sum log probs to get sequence log prob
        sequence_log_prob = response_log_probs.sum().item()
        
        # Normalize by sequence length if requested
        if self.normalize:
            sequence_log_prob = sequence_log_prob / len(response_log_probs)
        
        return sequence_log_prob
    
    def _compute_token_level_logprob(self, response: str, prompt: str) -> float:
        """
        Compute token-level average log probability of response given prompt.
        
        This rewards each token individually, which can be better for
        long sequences or when we want to emphasize token distributions.
        
        Args:
            response: Generated response from student model
            prompt: Input prompt
            
        Returns:
            Average token-level log probability
        """
        # Compute same as sequence level but average across tokens
        sequence_log_prob = self._compute_sequence_level_logprob(response, prompt)
        
        # Tokenize response
        input_ids_response = self.teacher_model.tokenize(response)
        response_len = len(input_ids_response)
        
        # Average across tokens
        avg_token_log_prob = sequence_log_prob / response_len
        
        return avg_token_log_prob
    
    def batch_compute(
        self, 
        responses: List[str], 
        prompts: List[str]
    ) -> List[float]:
        """
        Compute rewards for a batch of responses and prompts.
        
        Args:
            responses: List of generated responses
            prompts: List of input prompts
            
        Returns:
            List of log probability rewards
        """
        rewards = []
        for response, prompt in zip(responses, prompts):
            reward = self(response, prompt)
            rewards.append(reward)
        return rewards 