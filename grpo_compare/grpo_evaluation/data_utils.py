"""
Utilities for loading and preprocessing the GSM8K dataset for GRPO training.
"""

import re
from typing import Dict, List, Tuple, Union
from datasets import load_dataset, Dataset

def prepare_gsm8k_for_grpo(
    val_size: float = 0.1, 
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Load and prepare the GSM8K dataset specifically for GRPO training
    with proper formatting for mathematical reasoning tasks.
    
    Args:
        val_size: Portion of training data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Load GSM8K dataset
    dataset = load_dataset("gsm8k", "main")
    
    # Split into train and validation sets
    train_data = dataset["train"]
    # Use a portion of training data for validation
    train_split = train_data.train_test_split(test_size=val_size, seed=seed)
    train_data, val_data = train_split["train"], train_split["test"]
    
    # Format data for GRPO training
    train_data = train_data.map(format_sample_for_grpo)
    val_data = val_data.map(format_sample_for_grpo)
    
    return train_data, val_data

def format_sample_for_grpo(example: Dict) -> Dict:
    """
    Format a GSM8K example for GRPO training.
    
    Args:
        example: Raw GSM8K example with 'question' and 'answer' fields
        
    Returns:
        Formatted example with 'prompt' and 'reference_answer' fields
    """
    question = example["question"]
    answer = example["answer"]
    
    # Format the prompt for math problems
    formatted_prompt = f"Solve the following math problem step by step:\n\n{question}"
    
    return {
        "prompt": formatted_prompt,
        "reference_answer": answer,
    }

def extract_final_answer(text: str) -> str:
    """
    Extract the final numerical answer from a solution text.
    
    Args:
        text: Solution text containing steps and final answer
        
    Returns:
        The extracted final answer as a string
    """
    # GSM8K answers typically end with "#### <number>"
    match = re.search(r'####\s*(\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).strip()
    
    # If no #### format, try to find the last number in the text
    numbers = re.findall(r'(?:^|\s)(\d+(?:,\d+)*(?:\.\d+)?)(?:\s|$)', text)
    if numbers:
        return numbers[-1].strip()
    
    return ""

def evaluate_reasoning_steps(response: str, reference: str) -> float:
    """
    Evaluate the quality of reasoning steps in the response compared to reference.
    This is a simplified version - in production, you might use a more sophisticated
    approach like a trained evaluator model.
    
    Args:
        response: Model-generated response
        reference: Reference answer
        
    Returns:
        Score between 0 and 1 indicating reasoning quality
    """
    # Extract reasoning steps (everything before the final answer)
    response_steps = re.sub(r'####.*$', '', response).strip()
    reference_steps = re.sub(r'####.*$', '', reference).strip()
    
    # Count how many intermediate calculations appear in both texts
    response_calculations = set(re.findall(r'(\d+\s*[\+\-\*/]\s*\d+\s*=\s*\d+)', response_steps))
    reference_calculations = set(re.findall(r'(\d+\s*[\+\-\*/]\s*\d+\s*=\s*\d+)', reference_steps))
    
    # If there are no calculations in the reference, this method won't work well
    if not reference_calculations:
        return 0.5  # Default to middle score
    
    # Calculate overlap
    overlap = response_calculations.intersection(reference_calculations)
    score = len(overlap) / len(reference_calculations)
    
    return min(1.0, score)  # Cap at 1.0

def prepare_prompt_for_generation(prompt: str, tokenizer) -> Dict:
    """
    Prepare a prompt for model generation by tokenizing and formatting.
    
    Args:
        prompt: Text prompt
        tokenizer: The tokenizer for the model
        
    Returns:
        Dictionary with tokenized inputs
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    return inputs 