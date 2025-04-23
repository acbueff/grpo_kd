import evaluate
import logging
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional # Added for type hinting
# from language_tool_python import LanguageTool # Optional

logger = logging.getLogger(__name__)

# Based on provided context for NeurIPS-ready evaluation
SUPPORTED_METRICS = [
    # Accuracy/Factual
    "accuracy", "exact_match", "f1",
    # Text Quality/Generation
    "rouge", "bleu", "bertscore", "perplexity",
    # Code
    "pass@k", # Needs specific calculation logic
    # Misc
    "matthews_correlation", # Example, if needed
    "spearmanr", "pearsonr",
    # Custom/Placeholder
    "gsm8k_accuracy", # Handled by parsing
    "truthfulqa_truthful", # Needs specific handling
    "mt_bench_score", # Requires LLM-as-judge
    "swe_bench_pass", # Requires execution environment
    "toxicity_rate", # Requires toxicity classifier (e.g. PerspectiveAPI or HF model)
    "grammatical_error_rate" # Requires language_tool_python
]

def load_metrics(metric_names: list):
    """Loads evaluation metrics from the evaluate library or registers placeholders."""
    metrics = {}
    for name in metric_names:
        logger.info(f"Attempting to load or register metric: {name}")
        if name not in SUPPORTED_METRICS:
            logger.warning(f"Metric '{name}' not in supported list. You might need to implement custom handling. Skipping automatic loading.")
            metrics[name] = None # Mark as needing custom handling
            continue

        # Handle metrics needing custom logic or external tools
        if name in ["gsm8k_accuracy", "truthfulqa_truthful", "mt_bench_score", "pass@k", "swe_bench_pass", "toxicity_rate", "grammatical_error_rate"]:
            logger.info(f"Metric '{name}' requires custom calculation logic. Registering as None.")
            metrics[name] = None # Mark as needing custom handling in the main eval script
            continue

        try:
            # Load standard metrics from `evaluate`
            metrics[name] = evaluate.load(name)
            logger.info(f"Successfully loaded metric '{name}' from evaluate library.")
        except Exception as e:
            logger.error(f"Failed to load standard metric '{name}' from evaluate library: {e}. Registering as None.")
            metrics[name] = None # Mark as unavailable or needing custom handling
    return metrics

def parse_gsm8k_answer(text: str) -> Optional[str]:
    """Extracts the final numerical answer from GSM8K reasoning strings (enhanced)."""
    if text is None: return None
    # Regex to find the final answer marked by ####
    match = re.search(r"####\s*([\d\.,]+)", text)
    if match:
        answer = match.group(1).replace(",", "").strip()
        try:
            float_ans = float(answer)
            if float_ans.is_integer():
                 return str(int(float_ans)) # Return as integer string if whole number
            return str(float_ans) # Return as float string
        except ValueError:
            logger.warning(f"Could not parse number after #### in GSM8K: {match.group(1)}")
            return None

    # Fallback: Check if the answer is just a number at the end of the string
    # More robust check for number possibly surrounded by minor text/spaces
    match_end = re.search(r"[Tt]he answer is[:\s]*([\d\.,]+)$", text.strip())
    if match_end:
        answer = match_end.group(1).replace(",", "").strip()
        try:
            float_ans = float(answer)
            if float_ans.is_integer():
                 return str(int(float_ans))
            return str(float_ans)
        except ValueError:
             pass # Ignore if trailing number isn't valid

    # Fallback: Last number in the string
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if numbers:
        last_num_str = numbers[-1]
        try:
             float_ans = float(last_num_str)
             if float_ans.is_integer():
                  return str(int(float_ans))
             return str(float_ans)
        except ValueError:
             pass

    logger.debug(f"Could not find GSM8K answer in: {text[:100]}...")
    return None # No valid number found

def calculate_perplexity(predictions: List[str], model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: torch.device, batch_size: int = 4) -> Optional[float]:
    """Calculates perplexity using the evaluate library."""
    # Ensure predictions are non-empty strings
    valid_predictions = [p for p in predictions if isinstance(p, str) and len(p.strip()) > 0]
    if not valid_predictions:
        logger.warning("No valid predictions provided for perplexity calculation.")
        return None

    try:
        # Ensure the metric exists before loading
        perplexity_metric = evaluate.load("perplexity", module_type="metric")
        results = perplexity_metric.compute(
            predictions=valid_predictions,
            model_id=None, # Using loaded model
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            add_start_token=False # For causal LMs generating sequences
        )
        return results['mean_perplexity']
    except Exception as e:
        # Log the full traceback for debugging
        logger.error(f"Failed to calculate perplexity: {e}", exc_info=True)
        return None # Return None on failure


# Placeholder for HumanEval / MBPP pass@k calculation (requires execution)
# def calculate_pass_at_k(predictions: List[str], references: List[dict], k: int = 1):
#     # Needs to execute code snippets and check correctness
#     pass

# Placeholder for SWE-bench calculation (requires execution environment)
# def calculate_swe_bench_pass(predictions: List[str], references: List[dict]):
#     # Needs complex setup for running tests within SWE-bench framework
#     pass

# Placeholder for TruthfulQA metric (requires fine-grained checks)
# def calculate_truthfulqa_score(predictions: List[str], references: List[dict]):
#     # Needs specific logic for judging truthfulness and informativeness
#     pass

# Placeholder for toxicity calculation
# def calculate_toxicity(predictions: List[str], toxicity_model, toxicity_tokenizer):
#     # Use a trained toxicity classifier model
#     pass
