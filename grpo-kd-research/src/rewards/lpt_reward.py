import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

class LanguageModelingReward:
    """
    Language Modeling Loss (L_PT) for preventing catastrophic forgetting.
    
    This implements the pretraining loss component from MiniLLM, which
    helps the student model maintain its language modeling capabilities
    while adapting to imitate the teacher model.
    """
    
    def __init__(
        self,
        tokenizer = None,
        max_length: int = 512,
        device: str = "cuda",
    ):
        """
        Initialize language modeling reward.
        
        Args:
            tokenizer: Tokenizer to use (if None, will use student model's tokenizer)
            max_length: Maximum sequence length
            device: Device to run on
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
    
    def __call__(
        self, 
        student_model: nn.Module, 
        text_batch: List[str]
    ) -> torch.Tensor:
        """
        Compute language modeling loss on a batch of text.
        
        Args:
            student_model: The student model being trained
            text_batch: Batch of text sequences
            
        Returns:
            Language modeling loss tensor
        """
        # Use student model's tokenizer if none provided
        tokenizer = self.tokenizer if self.tokenizer else getattr(student_model, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("No tokenizer provided and student model doesn't have one")
        
        # Prepare inputs
        inputs = self._prepare_inputs(tokenizer, text_batch)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass through student model
        outputs = student_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"] if "labels" in inputs else inputs["input_ids"],
            return_dict=True,
        )
        
        # Extract loss
        lm_loss = outputs.loss
        
        return lm_loss
    
    def _prepare_inputs(
        self, 
        tokenizer, 
        text_batch: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for language modeling.
        
        Args:
            tokenizer: Tokenizer to use
            text_batch: Batch of text sequences
            
        Returns:
            Dictionary with input tensors
        """
        # Check tokenizer type to determine how to tokenize
        if hasattr(tokenizer, "batch_encode_plus"):
            # Hugging Face tokenizer
            encodings = tokenizer.batch_encode_plus(
                text_batch,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            # Create input_ids and attention_mask
            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]
            
            # Create labels (shift input_ids)
            labels = input_ids.clone()
            # Mask out padding tokens
            labels[labels == tokenizer.pad_token_id] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        else:
            # Custom tokenizer
            batch_input_ids = []
            batch_attention_mask = []
            
            for text in text_batch:
                # Tokenize
                input_ids = tokenizer.encode(text, max_length=self.max_length, truncation=True)
                
                # Pad or truncate to max_length
                if len(input_ids) > self.max_length:
                    input_ids = input_ids[:self.max_length]
                    attention_mask = [1] * self.max_length
                else:
                    padding = [tokenizer.pad_token_id] * (self.max_length - len(input_ids))
                    attention_mask = [1] * len(input_ids) + [0] * len(padding)
                    input_ids = input_ids + padding
                
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
            
            # Convert to tensors
            batch_input_ids = torch.tensor(batch_input_ids)
            batch_attention_mask = torch.tensor(batch_attention_mask)
            
            # Create labels (same as input_ids, but replace pad_token_id with -100)
            labels = batch_input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            
            return {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask,
                "labels": labels
            }

class FaroeseLanguageModelingReward(LanguageModelingReward):
    """
    Faroese-specific language modeling reward.
    
    Extends the base language modeling reward with Faroese-specific
    functionality, such as handling Faroese characters and potential
    language-specific preprocessing.
    """
    
    def __init__(
        self,
        tokenizer = None,
        max_length: int = 512,
        device: str = "cuda",
        weight_faroese_chars: bool = True,
        faroese_char_factor: float = 1.2,
    ):
        """
        Initialize Faroese language modeling reward.
        
        Args:
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            device: Device to run on
            weight_faroese_chars: Whether to weight tokens with Faroese chars more heavily
            faroese_char_factor: Weight factor for tokens with Faroese chars
        """
        super().__init__(tokenizer, max_length, device)
        self.weight_faroese_chars = weight_faroese_chars
        self.faroese_char_factor = faroese_char_factor
        self.faroese_chars = set("ðøáíóúýæÐØÁÍÓÚÝÆ")
    
    def __call__(
        self, 
        student_model: nn.Module, 
        text_batch: List[str]
    ) -> torch.Tensor:
        """
        Compute Faroese-specific language modeling loss.
        
        Applies additional weighting to tokens containing Faroese characters
        if specified.
        
        Args:
            student_model: The student model being trained
            text_batch: Batch of text sequences
            
        Returns:
            Language modeling loss tensor
        """
        # Preprocess Faroese text
        processed_batch = [self._preprocess_faroese(text) for text in text_batch]
        
        # Compute base loss
        lm_loss = super().__call__(student_model, processed_batch)
        
        # If not weighting Faroese chars, return base loss
        if not self.weight_faroese_chars:
            return lm_loss
        
        # If weighting Faroese chars, compute token-level loss and apply weights
        tokenizer = self.tokenizer if self.tokenizer else getattr(student_model, "tokenizer", None)
        
        # Separate function call to compute token-level loss with weights
        weighted_loss = self._compute_weighted_loss(student_model, processed_batch, tokenizer)
        
        return weighted_loss
    
    def _preprocess_faroese(self, text: str) -> str:
        """
        Preprocess Faroese text.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Specific handling for Faroese text if needed
        # Ensure proper encoding of Faroese characters
        replacements = {
            "dh": "ð",
            "oe": "ø",
            "aa": "á",
            "ii": "í",
            "oo": "ó",
            "uu": "ú",
            "yy": "ý",
            "ae": "æ",
            "DH": "Ð",
            "OE": "Ø",
            "AA": "Á",
            "II": "Í",
            "OO": "Ó",
            "UU": "Ú",
            "YY": "Ý",
            "AE": "Æ",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _compute_weighted_loss(
        self, 
        student_model: nn.Module, 
        text_batch: List[str],
        tokenizer
    ) -> torch.Tensor:
        """
        Compute weighted loss giving more importance to Faroese characters.
        
        This is a simplified implementation. A full implementation would
        track which tokens contain Faroese characters and weight them
        differently in the loss.
        
        Args:
            student_model: The student model
            text_batch: Batch of text sequences
            tokenizer: Tokenizer to use
            
        Returns:
            Weighted language modeling loss
        """
        # This is a placeholder for a more sophisticated implementation
        # In a full implementation, we would:
        # 1. Identify which tokens contain Faroese characters
        # 2. Create a weight tensor that upweights those tokens
        # 3. Apply the weights to the per-token loss
        
        # For now, we use the basic implementation
        return super().__call__(student_model, text_batch) 