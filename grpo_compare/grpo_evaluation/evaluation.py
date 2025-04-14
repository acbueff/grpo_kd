"""
Evaluation utilities for measuring performance on mathematical reasoning tasks.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from .data_utils import extract_final_answer, evaluate_reasoning_steps

def evaluate_math_performance(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: Dataset,
    num_samples: Optional[int] = None,
    device: str = "cuda",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    use_vllm: bool = False,
    vllm_client=None,
) -> Dict[str, float]:
    """
    Evaluate model performance on math reasoning tasks.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer for the model
        eval_dataset: Dataset containing math problems
        num_samples: Number of samples to evaluate (if None, use all)
        device: Device to use for inference
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        use_vllm: Whether to use vLLM for generation
        vllm_client: vLLM client for generation
        
    Returns:
        Dictionary with evaluation metrics
    """
    if num_samples is not None:
        eval_dataset = eval_dataset.select(range(min(num_samples, len(eval_dataset))))
    
    correct_answers = 0
    correct_reasoning = 0
    total_tokens = 0
    inference_time = 0
    responses = []
    
    for sample in eval_dataset:
        prompt = sample["prompt"]
        reference = sample["reference_answer"]
        
        # Generate model response
        start_time = time.time()
        
        if use_vllm and vllm_client is not None:
            # Use vLLM for generation
            vllm_output = vllm_client.generate(
                prompt, 
                max_tokens=max_new_tokens,
                temperature=temperature
            )
            response = vllm_output.generations[0].text
            tokens_generated = len(tokenizer.encode(response)) - len(tokenizer.encode(prompt))
        else:
            # Use regular Hugging Face generation
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0),
                    num_return_sequences=1,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]
            
        inference_time += time.time() - start_time
        total_tokens += tokens_generated
        
        # Remove prompt from response if it's included
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        # Store the full response for later analysis
        responses.append(response)
        
        # Extract final answer
        model_answer = extract_final_answer(response)
        ref_answer = extract_final_answer(reference)
        
        # Check exact match
        if model_answer == ref_answer:
            correct_answers += 1
        
        # Check reasoning quality
        reasoning_score = evaluate_reasoning_steps(response, reference)
        if reasoning_score >= 0.7:  # Consider reasoning good if score >= 0.7
            correct_reasoning += 1
    
    num_samples = len(eval_dataset)
    results = {
        "answer_accuracy": correct_answers / num_samples,
        "reasoning_quality": correct_reasoning / num_samples,
        "token_efficiency": total_tokens / num_samples,
        "avg_inference_time_per_sample": inference_time / num_samples,
        "total_inference_time": inference_time,
    }
    
    return results, responses

def analyze_token_efficiency(responses: List[str], tokenizer: PreTrainedTokenizer) -> Dict[str, float]:
    """
    Analyze token efficiency of generated responses.
    
    Args:
        responses: List of generated responses
        tokenizer: Tokenizer to use for counting tokens
        
    Returns:
        Dictionary with token efficiency metrics
    """
    token_counts = [len(tokenizer.encode(resp)) for resp in responses]
    
    return {
        "mean_token_count": np.mean(token_counts),
        "median_token_count": np.median(token_counts),
        "min_token_count": min(token_counts),
        "max_token_count": max(token_counts),
    }

def evaluate_kl_divergence(
    model: PreTrainedModel, 
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: Dataset,
    num_samples: int = 10,
    device: str = "cuda"
) -> float:
    """
    Evaluate KL divergence between model and reference model on evaluation samples.
    
    Args:
        model: The current model
        ref_model: The reference model
        tokenizer: The tokenizer
        eval_dataset: Dataset containing evaluation samples
        num_samples: Number of samples to use
        device: Device to use for computation
        
    Returns:
        Average KL divergence
    """
    model.eval()
    ref_model.eval()
    
    # Use a small subset of data for efficiency
    eval_subset = eval_dataset.select(range(min(num_samples, len(eval_dataset))))
    
    kl_values = []
    
    for sample in eval_subset:
        prompt = sample["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Get logits from model and reference model
            outputs = model(**inputs)
            ref_outputs = ref_model(**inputs)
            
            # Calculate KL divergence
            log_probs = torch.log_softmax(outputs.logits, dim=-1)
            ref_log_probs = torch.log_softmax(ref_outputs.logits, dim=-1)
            
            kl = torch.sum(torch.exp(log_probs) * (log_probs - ref_log_probs), dim=-1)
            kl_values.append(kl.mean().item())
    
    return np.mean(kl_values)

def log_hardware_metrics() -> Dict[str, float]:
    """
    Track GPU utilization and memory during evaluation.
    
    Returns:
        Dictionary with hardware metrics
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_metrics = {}
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_gb = mem_info.used / 1024**3
            mem_total_gb = mem_info.total / 1024**3
            
            # Get utilization info
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            gpu_metrics[f"gpu_{i}_memory_used_gb"] = mem_used_gb
            gpu_metrics[f"gpu_{i}_memory_total_gb"] = mem_total_gb
            gpu_metrics[f"gpu_{i}_memory_percent"] = 100 * mem_used_gb / mem_total_gb
            gpu_metrics[f"gpu_{i}_utilization_percent"] = utilization.gpu
            
        return gpu_metrics
    
    except (ImportError, Exception) as e:
        print(f"Could not collect hardware metrics: {e}")
        return {"error": str(e)} 