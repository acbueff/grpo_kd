# GRPO Implementation Comparison Framework

This framework allows for a systematic comparison of different GRPO (Group Relative Policy Optimization) implementations by fine-tuning the Qwen 2.5 3B model on the GSM8K dataset and evaluating mathematical reasoning performance.

## 📋 Overview

This framework evaluates the following GRPO implementations:

1. **TRL**: The most complete implementation with multiple loss types support
2. **Unsloth**: Optimized for memory efficiency and speed
3. **Verl**: Research framework with sequence-balanced training
4. **SimpleRL**: Specialized for mathematical reasoning tasks
5. **Search-R1**: Adapted for search/retrieval-augmented tasks
6. **TinyZero**: Lightweight implementation for smaller workloads

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU with at least 16GB memory
- The required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/grpo-compare.git
   cd grpo-compare
   ```

2. Install the required dependencies:
   ```bash
   pip install -r grpo_evaluation/requirements.txt
   ```

3. Install the framework in development mode:
   ```bash
   pip install -e .
   ```

## 🔧 Usage

### Running the Comparison

To run the full comparison with default parameters:

```bash
python -m grpo_evaluation.main
```

This will:
1. Prepare the GSM8K dataset
2. Train models using each GRPO implementation
3. Evaluate the trained models
4. Generate a comparison report

### Command-line Options

- `--model_name`: Model to fine-tune (default: "Qwen/Qwen2.5-3B-Instruct")
- `--implementations`: List of implementations to evaluate (default: ["trl", "unsloth"])
- `--output_dir`: Directory to save results (default: "./results")
- `--eval_only`: Only run evaluation without training
- `--use_wandb`: Log metrics to Weights & Biases
- `--seed`: Random seed for reproducibility (default: 42)
- `--eval_samples`: Number of samples to use for evaluation (default: 100)

Example with custom options:

```bash
python -m grpo_evaluation.main \
  --model_name "Qwen/Qwen2.5-7B-Instruct" \
  --implementations trl unsloth verl \
  --output_dir "./results_7B" \
  --use_wandb \
  --eval_samples 200
```

### Evaluating Specific Implementations

To evaluate only specific implementations:

```bash
python -m grpo_evaluation.main --implementations trl unsloth
```

### Evaluation Only Mode

If you've already trained models and want to evaluate them:

```bash
python -m grpo_evaluation.main --eval_only
```

## 📊 Results

The framework generates a comprehensive report that includes:

- Performance metrics for each implementation
- Visualizations comparing accuracy, reasoning quality, and efficiency
- Sample responses from each implementation
- Hardware utilization metrics
- Training time and efficiency analysis

The report is saved to `./results/reports/grpo_comparison_report.md` by default.

## 🧩 Framework Structure

- `config.py`: Configuration parameters
- `data_utils.py`: Dataset preparation utilities
- `evaluation.py`: Model evaluation functions
- `training.py`: Generic GRPO training loop
- `visualization.py`: Report generation utilities
- `main.py`: Main script for running the comparison
- `implementations/`: Implementation-specific modules
  - `trl_impl.py`: TRL implementation
  - `unsloth_impl.py`: Unsloth implementation
  - ...and others

## 🔍 Key Metrics

The framework evaluates the following metrics:

1. **Answer Accuracy**: Correctness of final numerical answers
2. **Reasoning Quality**: Quality of step-by-step reasoning
3. **Token Efficiency**: Number of tokens generated
4. **Inference Time**: Time taken to generate responses
5. **Training Time**: Time taken to train the model
6. **Memory Usage**: Peak memory consumption during training

## 🛠️ Adding New Implementations

To add a new GRPO implementation:

1. Create a new file in the `implementations/` directory (e.g., `new_impl.py`)
2. Implement the `setup_new_grpo` function that returns a trainer object
3. Create a wrapper class with a standardized interface
4. Add the implementation to the `IMPLEMENTATION_CONFIGS` dictionary in `config.py`

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

This framework builds upon the following open-source projects:
- [TRL](https://github.com/huggingface/trl)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Verl](https://github.com/verl-main)
- And other GRPO implementations in the `grpo_compare` directory 