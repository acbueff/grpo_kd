import argparse
import os
import yaml
import torch
import wandb
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, set_seed

from reward_utils import calculate_da_reward # Import the DA reward function
from typing import Optional # Added for type hinting

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models_and_tokenizers(config, device):
    """Loads student, teacher, and tokenizers based on config."""
    logger.info(f"Loading student model: {config['student_model_name']}")

    bnb_config = None
    load_in_8bit = config.get("load_in_8bit", False) # Add quantization options to config if needed
    load_in_4bit = config.get("load_in_4bit", False)

    quantization_config = None # Default to None
    if load_in_8bit or load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit,
             bnb_4bit_quant_type="nf4",
             bnb_4bit_compute_dtype=torch.bfloat16, # or float16
             bnb_4bit_use_double_quant=True,
        )
        device_map = {"": device} # Adjust for multi-GPU if necessary
        torch_dtype = torch.bfloat16 # or float16
        logger.info(f"Using quantization: 8bit={load_in_8bit}, 4bit={load_in_4bit}")
    else:
        quantization_config = None
        device_map = None
        torch_dtype = None # Use default

    student_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config['student_model_name'],
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=config.get("trust_remote_code", True),
        use_flash_attention_2=config.get("use_flash_attention_2", False) # Optional Flash Attention
    )
    logger.info("Student model loaded.")

    # Load student tokenizer
    student_tokenizer = AutoTokenizer.from_pretrained(config['student_model_name'])
    if student_tokenizer.pad_token is None:
        logger.warning("Student tokenizer does not have a pad token. Setting pad_token=eos_token.")
        student_tokenizer.pad_token = student_tokenizer.eos_token
    logger.info("Student tokenizer loaded.")

    # LoRA configuration for student model
    if config.get('use_lora', False):
        logger.info("Applying LoRA to student model...")
        lora_config = LoraConfig(**config['lora_config'])

        # Ensure model is prepared for k-bit training if quantization is used
        if load_in_8bit or load_in_4bit:
             student_model = prepare_model_for_kbit_training(
                 student_model, use_gradient_checkpointing=config.get("gradient_checkpointing", False)
             )

        # student_model needs to be the raw model before applying value head for PEFT
        # Let's apply PEFT to the base model part
        # Note: TRL's AutoModelForCausalLMWithValueHead might handle this internally
        # Check TRL documentation/source if issues arise. Assuming it works directly for now.
        student_model = get_peft_model(student_model, lora_config)
        student_model.print_trainable_parameters()
        logger.info("LoRA applied.")


    # Load teacher model (only if in DA-PPO mode)
    teacher_model = None
    teacher_tokenizer = None
    if config['training_mode'] == 'da-ppo':
        logger.info(f"Loading teacher model: {config['teacher_model_name']}")
        # Teacher model doesn't need a value head and shouldn't require gradients
        # Load teacher potentially with quantization if memory is an issue
        teacher_bnb_config = None
        teacher_load_in_8bit = config.get("teacher_load_in_8bit", False) # Separate config for teacher?
        teacher_load_in_4bit = config.get("teacher_load_in_4bit", False)

        teacher_quantization_config = None # Default to None
        if teacher_load_in_8bit or teacher_load_in_4bit:
             teacher_quantization_config = BitsAndBytesConfig(
                 load_in_8bit=teacher_load_in_8bit, load_in_4bit=teacher_load_in_4bit,
                 bnb_4bit_quant_type="nf4",
                 bnb_4bit_compute_dtype=torch.bfloat16,
                 bnb_4bit_use_double_quant=True,
             )
             teacher_device_map = {"": device}
             teacher_torch_dtype = torch.bfloat16
             logger.info(f"Using quantization for teacher: 8bit={teacher_load_in_8bit}, 4bit={teacher_load_in_4bit}")
        else:
             teacher_quantization_config = None
             teacher_device_map = None
             teacher_torch_dtype = None

        teacher_model = AutoModelForCausalLM.from_pretrained(
             config['teacher_model_name'],
             quantization_config=teacher_quantization_config,
             device_map=teacher_device_map,
             torch_dtype=teacher_torch_dtype,
             trust_remote_code=config.get("trust_remote_code", True),
             use_flash_attention_2=config.get("use_flash_attention_2", False)
        )
        teacher_model.eval() # Set to evaluation mode
        # Ensure teacher model doesn't compute gradients
        for param in teacher_model.parameters():
            param.requires_grad = False
        logger.info("Teacher model loaded and set to eval mode.")

        # Load teacher tokenizer
        teacher_tokenizer = AutoTokenizer.from_pretrained(config['teacher_model_name'])
        if teacher_tokenizer.pad_token is None:
            logger.warning("Teacher tokenizer does not have a pad token. Setting pad_token=eos_token.")
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        logger.info("Teacher tokenizer loaded.")

    return student_model, student_tokenizer, teacher_model, teacher_tokenizer


def main(config_path: str, training_mode: str, run_seed: Optional[int] = None):
    """Main training function."""

    # --- Configuration ---
    logger.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['training_mode'] = training_mode # Add mode to config dict
    logger.info(f"Running in mode: {training_mode}")

    # Handle seeding
    seed = run_seed if run_seed is not None else config['ppo_config']['seed']
    set_seed(seed)
    config['ppo_config']['seed'] = seed # Ensure PPOConfig uses the correct seed
    logger.info(f"Using random seed: {seed}")

    # Adjust output dir based on mode and seed
    base_output_dir = config['output_dir']
    run_name = f"{training_mode}_seed{seed}"
    config['output_dir'] = os.path.join(base_output_dir, run_name)
    os.makedirs(config['output_dir'], exist_ok=True)
    logger.info(f"Output directory set to: {config['output_dir']}")


    # --- Device ---
    # Note: Device mapping is handled during model loading if quantization is used
    # PPOTrainer uses accelerator, let's rely on that primarily.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # logger.info(f"Using device: {device}")

    # --- Load Models and Tokenizers ---
    # Pass device='auto' or let device_map handle it in the function
    student_model, student_tokenizer, teacher_model, teacher_tokenizer = load_models_and_tokenizers(config, device='auto')

    # --- Load Dataset ---
    logger.info(f"Loading dataset: {config['dataset_name']}")
    dataset = load_dataset(config['dataset_name'], split="train") # Adjust split if necessary
    # Preprocess dataset (e.g., select prompts, tokenize)
    # We only need prompts for PPO
    # Filter or sample prompts if needed
    if config.get("dataset_subset_size", -1) > 0:
        dataset = dataset.select(range(config["dataset_subset_size"]))
        logger.info(f"Using subset of dataset: {config['dataset_subset_size']} samples.")


    def tokenize_prompt(element):
        # Tokenize prompts for the PPOTrainer (expects tensor inputs)
        # Ensure prompt text exists
        prompt_text = element.get(config['prompt_column'])
        if prompt_text is None:
            logger.warning(f"Prompt column '{config['prompt_column']}' not found or is None in element: {element}. Skipping.")
            # Return None or a structure that can be filtered out later
            return None # Or handle differently, e.g., return empty dict

        tokenized = student_tokenizer(
             prompt_text,
             truncation=True,
             max_length=config['max_prompt_length'],
             padding=False # TRL handles padding later potentially
        )
        return {"input_ids": tokenized["input_ids"], "query": prompt_text}

    # Filter out potential None entries if tokenizer returns None
    original_length = len(dataset)
    tokenized_dataset = dataset.map(tokenize_prompt, batched=False).filter(lambda x: x is not None and x.get('input_ids') is not None)
    filtered_length = len(tokenized_dataset)
    if filtered_length < original_length:
        logger.warning(f"Filtered out {original_length - filtered_length} samples during tokenization due to missing prompts.")

    tokenized_dataset.set_format(type="torch")

    # --- Initialize PPOTrainer ---
    ppo_config_dict = config['ppo_config']
    # Ensure model_name is set correctly if not using the alias (though alias should work)
    if 'model_name' not in ppo_config_dict or ppo_config_dict['model_name'] != config['student_model_name']:
         ppo_config_dict['model_name'] = config['student_model_name']

    # Set pad token id in generation kwargs if not already set
    gen_kwargs = config['generation_kwargs']
    if 'pad_token_id' not in gen_kwargs or gen_kwargs['pad_token_id'] is None:
        gen_kwargs['pad_token_id'] = student_tokenizer.pad_token_id
        logger.info(f"Generation pad_token_id set to: {gen_kwargs['pad_token_id']}")

    # Create PPOConfig object
    ppo_config_obj = PPOConfig(**ppo_config_dict)

    ppo_trainer = PPOTrainer(
        config=ppo_config_obj,
        model=student_model, # Pass the potentially PEFT-wrapped model
        ref_model=None, # PPOTrainer creates ref model automatically if None
        tokenizer=student_tokenizer,
        dataset=tokenized_dataset, # Pass the tokenized dataset
        data_collator=None # PPOTrainer handles collation internally
    )

    # --- WandB Initialization (Optional) ---
    if ppo_trainer.accelerator.is_main_process and ppo_config_dict.get("log_with") == "wandb":
        # Ensure wandb is installed: pip install wandb
        try:
            import wandb
            wandb.init(
                project=config.get("project_name", "da-ppo-project"),
                name=run_name,
                config=config, # Log the entire config
                reinit=True
            )
            logger.info("Weights & Biases initialized.")
        except ImportError:
            logger.warning("wandb not installed. Skipping wandb logging. Run `pip install wandb`")
            ppo_config_obj.log_with = None # Disable logging if import fails


    # --- Training Loop ---
    logger.info("Starting training...")
    # Determine number of steps based on epochs or max_steps
    if config['max_steps'] > 0:
        num_steps = config['max_steps']
    else:
        # Calculate steps based on epochs, dataset size, batch size
        # Use ppo_trainer.dataloader which respects batch_size and accelerator
        num_steps_per_epoch = len(ppo_trainer.dataloader)
        num_steps = int(num_steps_per_epoch * config['num_train_epochs'])
        logger.info(f"Calculated num_steps: {num_steps} ({config['num_train_epochs']} epochs over {len(tokenized_dataset)} samples with batch size {ppo_config_dict['batch_size']})")


    # Manually iterate using the trainer's dataloader
    progress_bar = tqdm(range(num_steps), disable=not ppo_trainer.accelerator.is_local_main_process)
    current_step = 0
    stats_to_log = {}

    for epoch in range(config['num_train_epochs']):
        logger.info(f"Starting Epoch {epoch+1}/{config['num_train_epochs']}")
        for step, batch in enumerate(ppo_trainer.dataloader):
            if current_step >= num_steps:
                logger.info(f"Reached max_steps ({num_steps}). Stopping training.")
                break

            query_tensors = batch["input_ids"] # List of prompt tensors
            prompt_strings = batch["query"]    # List of original prompt strings

            # Generate responses from student model
            # Use accelerator device for generation context if needed
            # with ppo_trainer.accelerator.autocast(): # Maybe needed for mixed precision
            response_tensors = ppo_trainer.generate(
                query_tensors,       # Pass the list of tensors
                return_prompt=False, # We only want the generated part
                length_sampler=None, # Use generation_kwargs instead
                **gen_kwargs,
            )
            # response_tensors is now a list of tensors, one per prompt

            # Decode response tensors to strings for reward calculation
            # Ensure correct decoding, handle potential padding/eos tokens
            batch['response'] = [student_tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

            # --- Calculate Rewards --- 
            # The reward calculation happens on the main process or needs careful handling with accelerator
            # For simplicity, let's calculate on main process and broadcast, or calculate per device if teacher is sharded.
            # Assuming teacher model is on a single device (potentially 'auto') accessible by main process for now.

            rewards_list = [] # List to hold scalar reward values for the batch
            if training_mode == 'da-ppo':
                if teacher_model is None or teacher_tokenizer is None:
                    raise ValueError("Teacher model/tokenizer not loaded for DA-PPO mode.")

                # Calculate DA reward
                try:
                    rewards_list = calculate_da_reward(
                        prompts=prompt_strings, # Pass original prompt strings
                        student_responses=batch['response'],
                        teacher_model=teacher_model,
                        teacher_tokenizer=teacher_tokenizer,
                        lambda_da=config['lambda_da'],
                        intrinsic_reward_weight=config.get('intrinsic_reward_weight', 0.0),
                        # Pass intrinsic rewards if they exist for the batch
                        intrinsic_rewards=batch.get('intrinsic_rewards'), # Assumes batch might have this key
                        da_reward_type=config['da_reward_type'],
                        device=ppo_trainer.accelerator.device, # Calculate on the correct device
                        ll_batch_size=config.get('reward_ll_batch_size', 4) # Configurable batch size
                    )
                except Exception as e:
                    logger.error(f"Error in calculate_da_reward at step {current_step}: {e}")
                    # Handle error, e.g., assign zero reward for the batch
                    rewards_list = [0.0] * len(prompt_strings)

            elif training_mode == 'baseline':
                # Baseline: Zero reward or a fixed small reward
                rewards_list = [0.0] * len(prompt_strings)
            else:
                raise ValueError(f"Unknown training_mode: {training_mode}")

            # Convert rewards to tensors on the correct device for PPOTrainer.step
            rewards_tensors = [torch.tensor(r, device=ppo_trainer.accelerator.device) for r in rewards_list]

            # --- PPO Step --- 
            # Ensure query_tensors and response_tensors are lists of tensors
            # query_tensors comes from dataloader (List[Tensor])
            # response_tensors comes from generate (List[Tensor])
            # rewards should be List[torch.Tensor] matching the batch
            try:
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards_tensors)
            except Exception as e:
                logger.error(f"Error during ppo_trainer.step at step {current_step}: {e}")
                logger.warning("Skipping PPO step due to error.")
                stats = {} # Assign empty stats or handle as needed

            # --- Logging --- 
            # Ensure logging happens only on the main process
            if ppo_trainer.accelerator.is_main_process:
                if stats: # Log only if step was successful
                    stats_to_log = {k: v if isinstance(v, (int, float)) else v.mean().item() for k, v in stats.items() if torch.is_tensor(v) or isinstance(v, (int, float))}\
                    stats_to_log["reward/mean"] = torch.stack(rewards_tensors).mean().item()
                    stats_to_log["reward/std"] = torch.stack(rewards_tensors).std().item()
                    # Use ppo_trainer.log_stats for proper handling with accelerator and logging integrations
                    ppo_trainer.log_stats(stats, batch, rewards_tensors)
                    if (current_step + 1) % config['logging_steps'] == 0:
                         logger.info(f"Step {current_step+1}/{num_steps} | Epoch {epoch+1} | Stats: {stats_to_log}")
                else:
                     if (current_step + 1) % config['logging_steps'] == 0:
                         logger.info(f"Step {current_step+1}/{num_steps} | Epoch {epoch+1} | PPO step skipped due to error.")

            # --- Saving --- 
            if (current_step + 1) % config['save_steps'] == 0:
                 if ppo_trainer.accelerator.is_main_process:
                     save_path = os.path.join(config['output_dir'], f"checkpoint-{current_step+1}")
                     logger.info(f"Saving checkpoint to {save_path}")
                     # Use ppo_trainer.save_pretrained for proper handling with PEFT/accelerator
                     ppo_trainer.save_pretrained(save_path)
                     logger.info(f"Checkpoint saved at step {current_step+1}")


            current_step += 1
            progress_bar.update(1)
            # Only update progress bar description if stats were logged
            if ppo_trainer.accelerator.is_main_process and stats and (current_step % config['logging_steps'] == 0):
                progress_bar.set_postfix(stats_to_log)

            # Check if max_steps reached within the inner loop too
            if current_step >= num_steps:
                break
        # End of epoch loop
        if current_step >= num_steps:
             break # Exit outer loop if max_steps reached

    progress_bar.close()

    # --- Final Save --- 
    if ppo_trainer.accelerator.is_main_process:
        final_save_path = os.path.join(config['output_dir'], "final_model")
        logger.info(f"Saving final model to {final_save_path}")
        ppo_trainer.save_pretrained(final_save_path)
        logger.info("Final model saved.")

    logger.info("Training complete.")
    # Finish WandB run if used
    if ppo_config_dict.get("log_with") == "wandb" and ppo_trainer.accelerator.is_main_process:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DA-PPO or Baseline PPO")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    parser.add_argument("--mode", type=str, required=True, choices=['baseline', 'da-ppo'], help="Training mode: 'baseline' or 'da-ppo'.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed to override config for multiple runs.")
    args = parser.parse_args()

    main(args.config, args.mode, args.seed)
