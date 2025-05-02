import argparse
import yaml
import os
import json
import logging
import torch
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel, PeftConfig # Import PeftModel for LoRA
from evaluation_utils import load_metrics, parse_gsm8k_answer, calculate_perplexity
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path: str, config: Dict[str, Any]):
    """Loads the model and tokenizer, handling base models and LoRA adapters."""
    try:
        # Check if it's a PEFT model (LoRA adapter)
        PeftConfig.from_pretrained(model_path) # This will raise an error if no adapter_config.json
        is_peft = True
        logger.info(f"Detected PEFT adapter at {model_path}")
    except ValueError:
        is_peft = False
        logger.info(f"Detected base model at {model_path}")
    except Exception as e: # Catch other potential errors like directory not found
        logger.error(f"Error checking for PeftConfig at {model_path}: {e}")
        is_peft = False # Assume not PEFT if error occurs

    model_config = config.get('model_config', {})
    quantization_config = None
    load_in_8bit = model_config.get('load_in_8bit', False)
    load_in_4bit = model_config.get('load_in_4bit', False)
    torch_dtype = getattr(torch, model_config.get('torch_dtype', 'float32')) # Default to float32 if not specified

    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_config.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_compute_dtype=torch_dtype, # Match compute dtype
            bnb_4bit_use_double_quant=model_config.get('bnb_4bit_use_double_quant', False),
        )
        logger.info("Loading model with 4-bit quantization.")
    elif load_in_8bit:
        # quantization_config = BitsAndBytesConfig(load_in_8bit=True) # Simpler 8-bit config
        logger.info("Loading model with 8-bit quantization.")
        # 8-bit loading is often handled directly in from_pretrained
        pass
    else:
        logger.info(f"Loading model with dtype: {torch_dtype}")


    device_map = model_config.get("device_map", "auto") # Default to auto device map

    if is_peft:
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_name = peft_config.base_model_name_or_path
        logger.info(f"Loading base model ({base_model_name}) for PEFT adapter.")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_8bit=load_in_8bit and not load_in_4bit, # 8bit direct arg
            quantization_config=quantization_config if load_in_4bit else None,
            torch_dtype=torch_dtype if not (load_in_4bit or load_in_8bit) else None, # Dtype only if not quantizing
            device_map=device_map,
            trust_remote_code=model_config.get('trust_remote_code', False)
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=model_config.get('trust_remote_code', False))
        logger.info(f"Loading PEFT adapter from {model_path}...")
        model = PeftModel.from_pretrained(model, model_path)
        logger.info("Merging PEFT adapter into the base model for evaluation...")
        try:
            model = model.merge_and_unload() # Merge for faster inference
            logger.info("Successfully merged PEFT adapter.")
        except Exception as e:
            logger.warning(f"Could not merge PEFT adapter, continuing with unmerged model: {e}")
        model.eval()
    else:
        # Load a standard model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=load_in_8bit and not load_in_4bit,
            quantization_config=quantization_config if load_in_4bit else None,
            torch_dtype=torch_dtype if not (load_in_4bit or load_in_8bit) else None,
            device_map=device_map,
            trust_remote_code=model_config.get('trust_remote_code', False)
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=model_config.get('trust_remote_code', False))

    # Set pad token if missing (common issue with GPT2, Llama)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.warning(f"Tokenizer missing pad_token_id. Setting to eos_token_id: {tokenizer.eos_token_id}")
        else:
            # Add a pad token if EOS is also missing (very unlikely for standard models)
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            logger.warning("Tokenizer missing pad_token_id and eos_token_id. Added a new pad token '[PAD]'.")


    return model, tokenizer

def run_evaluation(config_path: str, model_dir: str, results_dir: str):
    """Loads models, runs evaluation on specified datasets, and saves results."""
    logger.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    eval_config = config.get('evaluation_config', {})
    datasets_to_eval = eval_config.get('evaluation_datasets', [])
    metrics_to_compute = eval_config.get('evaluation_metrics', [])
    eval_batch_size = eval_config.get('eval_batch_size', 4)
    generation_kwargs = config.get('generation_kwargs', {})
    max_prompt_length = config.get('dataset_config', {}).get('max_prompt_length', 512)

    if not datasets_to_eval:
        logger.warning("No evaluation datasets specified in the config. Skipping evaluation.")
        return
    if not metrics_to_compute:
        logger.warning("No evaluation metrics specified in the config. Skipping evaluation.")
        return

    logger.info(f"Metrics to compute: {metrics_to_compute}")
    metrics_loaded = load_metrics(metrics_to_compute)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    all_results = []

    model_subdirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    logger.info(f"Found model directories to evaluate: {model_subdirs}")

    for model_name in tqdm(model_subdirs, desc="Evaluating Models"):
        model_path = os.path.join(model_dir, model_name, "final_model") # Assuming model saved in 'final_model' subfolder
        if not os.path.exists(model_path):
             # Fallback: Check if model is saved directly in model_name dir (e.g. if not using trainer save)
             model_path_alt = os.path.join(model_dir, model_name)
             if os.path.exists(os.path.join(model_path_alt, "config.json")) or os.path.exists(os.path.join(model_path_alt, "adapter_config.json")):
                 model_path = model_path_alt
             else:
                logger.warning(f"No 'final_model' subdirectory or model config found in {os.path.join(model_dir, model_name)}. Skipping.")
                continue

        logger.info(f"--- Loading model: {model_name} from {model_path} ---")
        try:
            model, tokenizer = load_model_and_tokenizer(model_path, config)
            model.to(device) # Ensure model is on the evaluation device
            logger.info(f"Model {model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model {model_name} from {model_path}: {e}", exc_info=True)
            continue

        # Use GenerationConfig if defined, otherwise use generation_kwargs directly
        gen_config = GenerationConfig(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **generation_kwargs # Pass generation kwargs from config
        )

        model_results = {"model_name": model_name}

        for ds_config in tqdm(datasets_to_eval, desc=f"Datasets for {model_name}", leave=False):
            ds_name = ds_config['name']
            ds_path = ds_config['path']
            ds_split = ds_config.get('split', 'test')
            prompt_col = ds_config['prompt_column']
            answer_col = ds_config.get('answer_column', None) # Optional ground truth answer column
            ds_subset_size = ds_config.get('subset_size', None)

            logger.info(f"Loading dataset: {ds_name} ({ds_path}), split: {ds_split}")
            try:
                eval_dataset = load_dataset(ds_path, split=ds_split)
                if ds_subset_size:
                    eval_dataset = eval_dataset.select(range(ds_subset_size))
                logger.info(f"Loaded {len(eval_dataset)} examples from {ds_name}.")
            except Exception as e:
                logger.error(f"Failed to load dataset {ds_name}: {e}", exc_info=True)
                continue

            dataset_results = {}
            all_prompts = []
            all_predictions = []
            all_references = [] # For metrics requiring ground truth

            # --- Generation ---
            logger.info(f"Generating predictions for {ds_name} using {model_name}...")
            for i in tqdm(range(0, len(eval_dataset), eval_batch_size), desc="Generating", leave=False):
                batch = eval_dataset[i:i+eval_batch_size]
                prompt_texts = batch[prompt_col]
                all_prompts.extend(prompt_texts)

                # Prepare inputs
                inputs = tokenizer(
                    prompt_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_prompt_length
                ).to(device)

                # Generate
                with torch.no_grad():
                    outputs = model.generate(**inputs, generation_config=gen_config)

                # Decode - skip prompt part
                # Be careful with tokenizers (e.g., Llama) that might include the prompt
                decoded_outputs = []
                for j, output_ids in enumerate(outputs):
                     # Find the length of the input prompt tokens for this specific example
                    prompt_len = len(inputs["input_ids"][j])
                    # Slice the output tensor to get only the generated part
                    completion_ids = output_ids[prompt_len:]
                    # Decode the completion tokens
                    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                    decoded_outputs.append(completion_text.strip())


                all_predictions.extend(decoded_outputs)

                if answer_col and answer_col in batch:
                    all_references.extend(batch[answer_col])
                else:
                    # Handle cases where references are missing or not needed for all metrics
                     all_references.extend([None] * len(batch[prompt_col])) # Placeholder

            logger.info(f"Generated {len(all_predictions)} predictions for {ds_name}.")

            # --- Metric Calculation ---
            logger.info(f"Calculating metrics for {ds_name}...")
            for metric_name in metrics_to_compute:
                metric = metrics_loaded.get(metric_name)
                if metric is None and metric_name not in ["perplexity", "gsm8k_accuracy", "truthfulqa_truthful", "grammatical_error_rate"]: # Handle placeholders requiring custom logic
                     logger.warning(f"Metric '{metric_name}' was not loaded successfully or requires custom calculation. Skipping.")
                     dataset_results[metric_name] = None
                     continue

                try:
                    logger.debug(f"Calculating metric: {metric_name}")
                    if metric_name == "perplexity":
                        perplexity_score = calculate_perplexity(all_predictions, model, tokenizer, device, eval_batch_size)
                        dataset_results[metric_name] = perplexity_score
                        logger.debug(f"{metric_name}: {perplexity_score}")
                    elif metric_name == "gsm8k_accuracy":
                        # Requires parsing and exact match with reference answer
                        if not answer_col:
                            logger.warning(f"GSM8K accuracy requested but 'answer_column' not provided for {ds_name}. Skipping.")
                            dataset_results[metric_name] = None
                            continue
                        parsed_predictions = [parse_gsm8k_answer(p) for p in all_predictions]
                        parsed_references = [parse_gsm8k_answer(r) for r in all_references if r is not None]

                        if len(parsed_predictions) != len(parsed_references):
                            logger.warning(f"Mismatch between predictions ({len(parsed_predictions)}) and references ({len(parsed_references)}) for GSM8K. Skipping.")
                            dataset_results[metric_name] = None
                            continue

                        correct_count = sum(1 for pred, ref in zip(parsed_predictions, parsed_references) if pred is not None and ref is not None and pred == ref)
                        accuracy = correct_count / len(parsed_references) if parsed_references else 0
                        dataset_results[metric_name] = accuracy
                        logger.debug(f"{metric_name}: {accuracy}")
                    # Add elif blocks here for other custom metrics (truthfulqa, toxicity, grammar etc.)
                    # elif metric_name == "grammatical_error_rate":
                    #   # Requires language_tool_python or similar
                    #   pass # Placeholder
                    elif metric is not None: # Standard 'evaluate' library metrics
                        metric_input = {"predictions": all_predictions}
                        # Add references if the metric needs them (most do)
                        if answer_col and all_references and any(r is not None for r in all_references):
                             metric_input["references"] = [r if r is not None else "" for r in all_references] # Handle None refs if metric expects strings
                             # Some metrics might require specific reference formats (e.g., list of lists for BLEU/ROUGE)
                             # Adjust reference preparation here based on metric requirements if needed.

                        # Check if metric needs specific kwargs (like bertscore model type)
                        metric_kwargs = {}
                        if metric_name == "bertscore":
                             metric_kwargs["lang"] = config.get("evaluation_config",{}).get("bertscore_lang", "en") # Example: configurable lang

                        results = metric.compute(**metric_input, **metric_kwargs)
                        # Store potentially complex results (e.g., ROUGE has R1, R2, RL)
                        dataset_results[metric_name] = results
                        logger.debug(f"{metric_name}: {results}")
                    else:
                         logger.warning(f"No calculation logic defined for metric '{metric_name}'. Skipping.")
                         dataset_results[metric_name] = None

                except Exception as e:
                    logger.error(f"Failed to compute metric '{metric_name}' for dataset '{ds_name}': {e}", exc_info=True)
                    dataset_results[metric_name] = f"ERROR: {e}"

            model_results[ds_name] = dataset_results
            logger.info(f"Finished calculating metrics for {ds_name}.")

        all_results.append(model_results)
        # Clean up model and clear cache if possible to save memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"--- Finished evaluating model: {model_name} ---")


    # --- Save Results ---
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "evaluation_summary.json")
    logger.info(f"Saving all evaluation results to: {output_file}")
    try:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        logger.info("Evaluation results saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {e}")

    # Optional: Generate paired comparison CSV (more complex, requires matching seeds)
    # This would involve loading pairs of models (baseline_seedX, da-ppo_seedX)
    # and generating outputs side-by-side for the *same* prompts.
    # For simplicity, this part is omitted here but could be added.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLMs (Baseline PPO vs DA-PPO).")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the trained model runs (subdirectories like 'baseline_seed0', 'da-ppo_seed0').")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save the evaluation results JSON file.")
    args = parser.parse_args()

    run_evaluation(args.config, args.model_dir, args.results_dir)
