# Distillation-Augmented PPO (DA-PPO) for LLM Fine-tuning

This project implements and evaluates a Distillation-Augmented Proximal Policy Optimization (DA-PPO) pipeline for fine-tuning Large Language Models (LLMs). It compares DA-PPO against a standard baseline PPO implementation using the `trl` library from Hugging Face. The goal is to leverage knowledge from a larger 'teacher' model to guide the training of a smaller 'student' model during RL fine-tuning.

## Project Structure

```
da_ppo_project/
├── src/                     # Source code
│   ├── __init__.py
│   ├── train_da_ppo.py      # Main training script for Baseline PPO and DA-PPO
│   ├── evaluate_models.py   # Script to evaluate trained models
│   ├── analyze_results.py   # Script for statistical analysis of results
│   ├── reward_utils.py      # Utilities for DA reward calculation
│   └── evaluation_utils.py  # Utilities for evaluation metrics and parsing
├── configs/                 # Configuration files
│   └── da_ppo_train_config.yaml # Main configuration file
├── data/                     # Placeholder for local datasets (if any)
├── models/                   # Directory to save trained model checkpoints/adapters
├── results/                  # Directory to save evaluation results and analysis reports
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd da_ppo_project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Depending on your hardware (GPU), you might need specific versions of `torch`, `cuda`, etc. Quantization features require `bitsandbytes`.*
    *Optional: For the Grammatical Error Rate metric (`grammatical_error_rate`), you might need `language_tool_python` and a local Java installation.*

## Configuration (`configs/da_ppo_train_config.yaml`)

This YAML file controls all aspects of training and evaluation. Key sections include:

*   **Model Configuration:**
    *   `student_model_name`: Path/name of the student model (e.g., `gpt2`, `meta-llama/Llama-2-7b-hf`).
    *   `teacher_model_name`: Path/name of the teacher model (e.g., `gpt2-medium`, `meta-llama/Llama-2-13b-hf`). Ensure you have access and sufficient resources (RAM/VRAM).
    *   `use_lora`: Whether to use LoRA for parameter-efficient fine-tuning of the student.
    *   `lora_config`: Parameters for LoRA (rank, alpha, etc.).
    *   Quantization options (`load_in_8bit`, `load_in_4bit`) can be added here if needed for student/teacher.
*   **Dataset Configuration:**
    *   `dataset_name`: Hugging Face dataset for prompts (e.g., `lvwerra/stack-exchange-paired`).
    *   `prompt_column`: Name of the column containing prompts.
    *   `max_prompt_length`: Maximum length for tokenizing prompts.
    *   `dataset_subset_size`: (Optional) Use a smaller subset for faster runs/debugging.
*   **PPO Configuration (`ppo_config`):** Maps directly to `trl.PPOConfig` arguments (learning rate, batch sizes, KL settings, seed, etc.).
*   **DA-PPO Specific Configuration:**
    *   `da_reward_type`: How the distillation reward is calculated (currently supports `log_likelihood`).
    *   `lambda_da`: Weight applied to the distillation reward signal.
    *   `intrinsic_reward_weight`: Weight for any non-distillation reward (e.g., from task correctness). Set to `0.0` if using only DA reward.
*   **Generation Configuration (`generation_kwargs`):** Parameters passed to the `generate` function during PPO rollouts (e.g., `max_new_tokens`, `temperature`, `top_p`). Ensure `pad_token_id` is correctly set for your student model/tokenizer.
*   **Training Configuration:** Epochs, steps, saving/logging frequency, output directories.
*   **Evaluation Configuration:**
    *   `evaluation_datasets`: List of datasets to evaluate on (HF path, name, split, prompt column, optional answer column, subset size).
    *   `evaluation_metrics`: List of metrics to compute (e.g., `accuracy`, `rouge`, `perplexity`). Custom metrics require implementation in `evaluation_utils.py` or the evaluation script.
    *   `eval_batch_size`: Batch size for evaluation generation.
*   **Statistical Analysis Configuration:**
    *   `num_runs`: Expected number of independent runs per mode (used for analysis/pairing).
    *   `significance_level`: Alpha value (e.g., 0.05) for statistical tests.

**Adjust model names and paths based on availability and system resources.** You might need to log in to Hugging Face Hub (`huggingface-cli login`) to access gated models like Llama-2.

## Running the Pipeline

Ensure the configuration file (`configs/da_ppo_train_config.yaml`) is correctly set up before running.

1.  **Training:**
    Run the `train_da_ppo.py` script, specifying the config, mode (`baseline` or `da-ppo`), and an optional seed (crucial for multiple runs).

    ```bash
    # --- Run Baseline PPO ---
    # Run 1 (Seed 0)
    python src/train_da_ppo.py --config configs/da_ppo_train_config.yaml --mode baseline --seed 0
    # Run 2 (Seed 1)
    python src/train_da_ppo.py --config configs/da_ppo_train_config.yaml --mode baseline --seed 1
    # Run 3 (Seed 2)
    python src/train_da_ppo.py --config configs/da_ppo_train_config.yaml --mode baseline --seed 2

    # --- Run DA-PPO ---
    # Run 1 (Seed 0)
    python src/train_da_ppo.py --config configs/da_ppo_train_config.yaml --mode da-ppo --seed 0
    # Run 2 (Seed 1)
    python src/train_da_ppo.py --config configs/da_ppo_train_config.yaml --mode da-ppo --seed 1
    # Run 3 (Seed 2)
    python src/train_da_ppo.py --config configs/da_ppo_train_config.yaml --mode da-ppo --seed 2
    ```
    *Adjust the number of runs and seeds according to `num_runs` in the config.*
    *Models will be saved in subdirectories under `models/` (e.g., `models/baseline_seed0/final_model`, `models/da-ppo_seed0/final_model`).*

2.  **Evaluation:**
    Run the `evaluate_models.py` script after training is complete for all runs. Provide the config file path, the main directory containing the model runs (`models/`), and the desired directory for results (`results/`).

    ```bash
    python src/evaluate_models.py --config configs/da_ppo_train_config.yaml --model_dir models/ --results_dir results/
    ```
    *This will generate evaluation scores for all models found in `model_dir` on the datasets specified in the config.*

3.  **Analysis:**
    Run the `analyze_results.py` script after evaluation is complete. Provide the config file path (to get the significance level) and the results directory.

    ```bash
    python src/analyze_results.py --config configs/da_ppo_train_config.yaml --results_dir results/
    ```
    *This performs statistical tests comparing baseline vs. DA-PPO across the runs.*

## Output

*   **`models/`**: Contains subdirectories for each training run (e.g., `baseline_seed0`, `da-ppo_seed0`), storing the final trained model (or LoRA adapter) and tokenizer.
*   **`results/`**:
    *   `evaluation_summary.json`: A JSON file containing the raw evaluation metrics for every model run on every dataset.
    *   `paired_comparisons.csv`: (Optional, if generated successfully) A CSV file containing prompts and the corresponding outputs from paired baseline and DA-PPO runs, useful for human evaluation or side-by-side review.
    *   `statistical_analysis_report.md`: A Markdown report summarizing the descriptive statistics (mean, std dev, CI) and the results of paired statistical tests (t-test, Wilcoxon) comparing Baseline PPO and DA-PPO for each metric/dataset combination.
    *   `analysis_summary.json`: A JSON file containing the structured results of the statistical analysis (means, CIs, p-values, significance status, winner).

## Interpretation

The key output for comparing DA-PPO and Baseline PPO is `results/statistical_analysis_report.md`. For each metric and dataset:

*   Check the **Mean Difference** and its **Confidence Interval (CI)**. If the CI does not include zero, it suggests a potentially significant difference.
*   Look at the **p-values** from the Paired t-test and Wilcoxon test. If a p-value is less than the significance level (alpha, typically 0.05), the difference between the two methods is considered statistically significant for that metric.
*   The **RESULT** line indicates whether a significant difference was found and identifies the "Winner" based on the mean difference (assuming higher scores are better for most metrics, adjusting for metrics like perplexity where lower is better).

This allows you to draw conclusions about whether the Distillation-Augmented approach provides a statistically significant improvement over the standard PPO baseline under the tested configuration.
