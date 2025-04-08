from typing import List, Dict, Optional, Union, Any
import torch
import re
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

class FaroeseTokenizerAdapter:
    """
    Adapter for enhancing tokenizers with Faroese language support.
    
    This adapter wraps an existing tokenizer and adds special handling
    for Faroese characters and linguistic constructs to improve tokenization
    of Faroese text.
    """
    
    def __init__(
        self,
        base_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        add_special_tokens: bool = True,
        use_prefix: bool = False,
        faroese_prefix: str = "🇫🇴 ",
    ):
        """
        Initialize the Faroese tokenizer adapter.
        
        Args:
            base_tokenizer: The base tokenizer to enhance (e.g., from Transformers)
            add_special_tokens: Whether to add Faroese-specific special tokens
            use_prefix: Whether to add a Faroese language prefix to inputs
            faroese_prefix: The prefix to use if use_prefix is True
        """
        self.base_tokenizer = base_tokenizer
        self.add_special_tokens = add_special_tokens
        self.use_prefix = use_prefix
        self.faroese_prefix = faroese_prefix
        
        # Faroese special characters
        self.faroese_chars = "ðøáíóúýæÐØÁÍÓÚÝÆ"
        
        # Add Faroese-specific tokens if requested
        if add_special_tokens:
            self._add_faroese_special_tokens()
    
    def _add_faroese_special_tokens(self):
        """Add Faroese-specific tokens to the tokenizer vocabulary."""
        # Check if we need to add special tokens for Faroese characters
        # that might be underrepresented in the base tokenizer
        
        # Get existing vocabulary
        vocab = self.base_tokenizer.get_vocab() if hasattr(self.base_tokenizer, "get_vocab") else {}
        
        # Check if Faroese characters are already well-represented
        faroese_special_tokens = []
        
        # Add special tokens for Faroese characters combinations
        faroese_combinations = [
            "ð", "Ð", "ø", "Ø", "á", "Á", "í", "Í", 
            "ó", "Ó", "ú", "Ú", "ý", "Ý", "æ", "Æ"
        ]
        
        # Add common Faroese word parts
        faroese_common_words = [
            "føroysk", "Føroyar", "Tórshavn", "oyggj", "fjørð",
            "bygd", "býur", "fjall", "dalur", "vík",
            "maður", "kona", "barn", "skip", "bátur"
        ]
        
        # Combine all special tokens
        faroese_special_tokens = faroese_combinations + faroese_common_words
        
        # Filter out tokens that already exist in vocabulary
        new_tokens = [token for token in faroese_special_tokens if token not in vocab]
        
        # Add to tokenizer if there are new tokens
        if new_tokens:
            self.base_tokenizer.add_tokens(new_tokens)
            print(f"Added {len(new_tokens)} Faroese-specific tokens to the tokenizer")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text to improve Faroese handling.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert alternative representations to Faroese characters
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
        
        # Add Faroese prefix if requested
        if self.use_prefix:
            if not text.startswith(self.faroese_prefix):
                text = self.faroese_prefix + text
        
        return text
    
    def tokenize(
        self, 
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        **kwargs
    ) -> List[int]:
        """
        Tokenize text with Faroese-specific handling.
        
        Args:
            text: Input text or list of texts
            add_special_tokens: Whether to add special tokens
            **kwargs: Additional arguments for the base tokenizer
            
        Returns:
            List of token IDs
        """
        # Handle either single string or list of strings
        if isinstance(text, str):
            # Preprocess text for Faroese
            text = self.preprocess_text(text)
            
            # Tokenize with base tokenizer
            tokens = self.base_tokenizer.tokenize(
                text, 
                add_special_tokens=add_special_tokens,
                **kwargs
            )
            
            # Convert to IDs
            token_ids = self.base_tokenizer.convert_tokens_to_ids(tokens)
            
            return token_ids
        elif isinstance(text, list):
            # Process each string in the list
            all_token_ids = []
            for t in text:
                token_ids = self.tokenize(
                    t, 
                    add_special_tokens=add_special_tokens,
                    **kwargs
                )
                all_token_ids.append(token_ids)
            return all_token_ids
        else:
            raise ValueError(f"Unsupported input type: {type(text)}")
    
    def encode(
        self, 
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> Union[List[int], torch.Tensor]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text or list of texts
            add_special_tokens: Whether to add special tokens
            return_tensors: Output format (None, 'pt', 'tf', etc.)
            **kwargs: Additional arguments for the base tokenizer
            
        Returns:
            Token IDs as list or tensor
        """
        # Process single string or list of strings
        if isinstance(text, str):
            # Preprocess text
            text = self.preprocess_text(text)
            
            # Use base tokenizer's encode method
            token_ids = self.base_tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                return_tensors=return_tensors,
                **kwargs
            )
            
            return token_ids
        elif isinstance(text, list):
            # Process list of strings
            processed_texts = [self.preprocess_text(t) for t in text]
            
            # Encode with base tokenizer
            token_ids = self.base_tokenizer.batch_encode_plus(
                processed_texts,
                add_special_tokens=add_special_tokens,
                return_tensors=return_tensors,
                **kwargs
            )
            
            return token_ids
        else:
            raise ValueError(f"Unsupported input type: {type(text)}")
    
    def decode(
        self, 
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional arguments for the base tokenizer
            
        Returns:
            Decoded text
        """
        # Use base tokenizer's decode method
        text = self.base_tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            **kwargs
        )
        
        # Remove Faroese prefix if added during encoding
        if self.use_prefix and text.startswith(self.faroese_prefix):
            text = text[len(self.faroese_prefix):]
        
        return text
    
    def batch_decode(
        self, 
        sequences: List[Union[List[int], torch.Tensor]],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Decode multiple sequences of token IDs.
        
        Args:
            sequences: Batch of token ID sequences
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional arguments for the base tokenizer
            
        Returns:
            List of decoded texts
        """
        # Use base tokenizer's batch_decode method
        texts = self.base_tokenizer.batch_decode(
            sequences,
            skip_special_tokens=skip_special_tokens,
            **kwargs
        )
        
        # Remove Faroese prefix if added during encoding
        if self.use_prefix:
            texts = [
                text[len(self.faroese_prefix):] if text.startswith(self.faroese_prefix) else text
                for text in texts
            ]
        
        return texts
    
    def __call__(
        self, 
        text: Union[str, List[str]], 
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Call the tokenizer on text.
        
        Args:
            text: Input text or list of texts
            **kwargs: Additional arguments for the base tokenizer
            
        Returns:
            Dictionary with encoded inputs
        """
        # Process text to handle Faroese-specific characters
        if isinstance(text, str):
            text = self.preprocess_text(text)
        elif isinstance(text, list):
            text = [self.preprocess_text(t) for t in text]
        
        # Call base tokenizer
        return self.base_tokenizer(text, **kwargs)
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer."""
        return self.base_tokenizer.vocab_size
    
    @property
    def pad_token_id(self) -> int:
        """Get the padding token ID."""
        return self.base_tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        """Get the end-of-sequence token ID."""
        return self.base_tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> int:
        """Get the beginning-of-sequence token ID."""
        return self.base_tokenizer.bos_token_id
    
    @property
    def unk_token_id(self) -> int:
        """Get the unknown token ID."""
        return self.base_tokenizer.unk_token_id

def create_faroese_tokenizer(
    base_model_name: str,
    add_special_tokens: bool = True,
    use_prefix: bool = False,
    cache_dir: Optional[str] = None,
) -> FaroeseTokenizerAdapter:
    """
    Create a Faroese-enhanced tokenizer based on an existing model.
    
    Args:
        base_model_name: Name or path of the base model
        add_special_tokens: Whether to add Faroese-specific tokens
        use_prefix: Whether to use a Faroese language prefix
        cache_dir: Directory to cache downloaded tokenizer files
        
    Returns:
        Enhanced tokenizer for Faroese
    """
    # Load base tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
    )
    
    # Create adapter
    faroese_tokenizer = FaroeseTokenizerAdapter(
        base_tokenizer,
        add_special_tokens=add_special_tokens,
        use_prefix=use_prefix,
    )
    
    return faroese_tokenizer 