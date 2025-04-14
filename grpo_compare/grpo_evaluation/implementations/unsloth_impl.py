"""
Unsloth implementation for GRPO.
"""

import os
import sys
import torch
from typing import Dict, Callable, Any, Optional

def setup_unsloth_grpo(model_name: str, config: Dict) -> Any:
    """
    Setup Unsloth's GRPO implementation.
    
    Args:
        model_name: Name of the model to use
        config: Configuration dictionary
        
    Returns:
        UnslothGRPOTrainer instance
    """
    try:
        # First, try to import unsloth
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError("Unsloth is required for this implementation. Install it with: pip install unsloth")
    
    # Add unsloth-main to path
    grpo_compare_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "unsloth-main")
    if os.path.exists(grpo_compare_dir):
        sys.path.append(grpo_compare_dir)
    
    # Load optimized model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.get("max_prompt_length", 1024) + config.get("max_completion_length", 512),
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=config.get("load_in_4bit", True),
        token=config.get("hf_token", None),
    )
    
    # Patch model for RL training with Flash Attention
    if config.get("use_flash_attention", True):
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.get("lora_r", 16),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=config.get("lora_alpha", 16),
            lora_dropout=config.get("lora_dropout", 0),
            bias="none",
            use_gradient_checkpointing=True,
            random_state=config.get("seed", 42),
            use_rslora=False,
            loftq_config=None,
        )
    
    # Get the reward model
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer as HFAutoTokenizer
        
        reward_model_name = config.get("reward_model_name", model_name)  # Default to same model
        reward_tokenizer = HFAutoTokenizer.from_pretrained(reward_model_name)
        reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name)
        
        # Move reward model to appropriate device
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        reward_model = reward_model.to(device)
    except Exception as e:
        print(f"Warning: Could not load reward model. Using random rewards. Error: {e}")
        reward_model = None
        reward_tokenizer = tokenizer
    
    # Patch TRL for Unsloth optimizations
    try:
        from unsloth.models.rl import patch_trl_for_rl
        patch_trl_for_rl(["grpo_trainer"])
    except Exception as e:
        print(f"Warning: Could not patch TRL for Unsloth. Error: {e}")
    
    # Create trainer
    try:
        from trl import GRPOConfig, GRPOTrainer
        
        # Configure GRPO
        grpo_config = GRPOConfig(
            beta=config.get("beta", 0.1),
            mini_batch_size=config.get("mini_batch_size", 8),
            batch_size=config.get("batch_size", 128),
            num_iterations=config.get("num_iterations", 3),
            learning_rate=config.get("learning_rate", 1e-5),
            epsilon=config.get("epsilon", 0.2),
            scale_rewards=config.get("scale_rewards", True),
            loss_type=config.get("loss_type", "bnpo"),
            
            # Generation parameters
            max_prompt_length=config.get("max_prompt_length", 1024),
            max_completion_length=config.get("max_completion_length", 512),
            temperature=config.get("temperature", 0.9),
            top_k=config.get("top_k", 50),
            top_p=config.get("top_p", 1.0),
            
            # Misc parameters
            output_dir=config.get("output_dir", "./results/unsloth"),
            logging_steps=10,
            save_steps=config.get("save_every", 500),
            eval_steps=config.get("eval_every", 200),
            num_train_epochs=config.get("num_epochs", 5),
            per_device_train_batch_size=config.get("mini_batch_size", 8),
            gradient_accumulation_steps=config.get("batch_size", 128) // config.get("mini_batch_size", 8),
            
            # Hardware optimization
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            
            # Report metrics to wandb
            report_to="wandb" if config.get("use_wandb", True) else "none",
        )
        
        # Create the GRPO trainer
        trainer = UnslothGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            args=grpo_config,
            train_dataset=None,  # Will be provided later
            eval_dataset=None,   # Will be provided later
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            enable_fused_projections=config.get("enable_fused_projections", True),
        )
        
        return trainer
    
    except ImportError:
        raise ImportError("TRL is required for this implementation. Install it with: pip install trl")

class UnslothGRPOTrainer:
    """
    Wrapper around Unsloth's optimized GRPO implementation.
    """
    def __init__(self, 
                 model, 
                 tokenizer, 
                 args, 
                 train_dataset=None, 
                 eval_dataset=None, 
                 reward_model=None, 
                 reward_tokenizer=None,
                 enable_fused_projections=True):
        try:
            from trl import GRPOTrainer
            
            # Store model and tokenizer
            self.model = model
            self.tokenizer = tokenizer
            
            # Create a reward function that uses the reward model
            def reward_function(samples):
                if reward_model is None:
                    # If no reward model, return random rewards
                    import random
                    import torch
                    return torch.tensor([random.random() for _ in range(len(samples))])
                
                # Get prompts and completions
                prompts = [s["prompt"] for s in samples]
                completions = [s.get("completion", "") for s in samples]
                
                # Tokenize
                inputs = reward_tokenizer(
                    prompts, 
                    completions,
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                ).to(reward_model.device)
                
                # Get rewards
                with torch.no_grad():
                    outputs = reward_model(**inputs)
                    rewards = outputs.logits.squeeze(-1)
                
                return rewards
            
            # Create the trainer
            self.trainer = GRPOTrainer(
                model=model,
                tokenizer=tokenizer,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                reward_function=reward_function,
            )
            
            # Enable fused projections if requested
            if enable_fused_projections and hasattr(model, "enable_fused_projections"):
                model.enable_fused_projections()
            
        except ImportError:
            raise ImportError("TRL is required for this implementation. Install it with: pip install trl")
    
    def train(self, train_dataset, eval_dataset, log_step_callback: Optional[Callable] = None):
        """Train the model"""
        # Set datasets
        self.trainer.train_dataset = train_dataset
        self.trainer.eval_dataset = eval_dataset
        
        # Register callback if provided
        if log_step_callback is not None:
            original_logging_step = self.trainer.log
            
            def logging_with_callback(*args, **kwargs):
                original_logging_step(*args, **kwargs)
                step = self.trainer.state.global_step
                metrics = {k: v for k, v in kwargs.items() if isinstance(v, (int, float))}
                log_step_callback(step, metrics)
            
            self.trainer.log = logging_with_callback
        
        # Train
        self.trainer.train()
        
        # Return the trained model
        return self.model
    
    def save_model(self, output_dir: str):
        """Save the model"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir) 