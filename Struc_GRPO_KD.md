# GRPO-KD: Research Project Structure

## Project Overview
```
grpo-kd-research/
├── README.md
├── CONTEXT.md
├── data/
├── src/
│   ├── models/
│   ├── training/
│   ├── rewards/
│   ├── evaluation/
│   └── utils/
├── configs/
├── scripts/
│   ├── slurm/
│   └── preprocessing/
├── experiments/
├── results/
├── notebooks/
└── docs/
```

## Folder Structure Explanation

### 1. Root Directory (`grpo-kd-research/`)
The root directory contains the main project files and subdirectories.

**Key Files:**
- `README.md`: Project overview, installation instructions, and basic usage guidelines
- `CONTEXT.md`: Comprehensive research context document for guiding the LLM assistant

### 2. Data Directory (`data/`)
Storage for all datasets used in the project, including Faroese corpora and evaluation benchmarks.

**Structure:**
```
data/
├── raw/
│   ├── faroese/
│   │   ├── wikipedia/
│   │   ├── news/
│   │   └── general_corpus/
│   └── parallel/
│       ├── faroese_english/
│       └── faroese_danish/
├── processed/
│   ├── training/
│   ├── validation/
│   └── test/
└── benchmarks/
    ├── foqa/
    ├── fone/
    └── euroeval/
```

**Purpose:** Contains all data required for training, validation, and testing. Includes raw Faroese text corpora, parallel corpora for translation tasks, and benchmark datasets like FoQA (Faroese Question-Answering) for evaluation.

**Key Files:**
- `data/raw/faroese/wikipedia/`: Faroese Wikipedia dumps
- `data/raw/faroese/news/`: Faroese news articles from sources like Sosialurin
- `data/benchmarks/foqa/`: FoQA dataset for evaluating question-answering capabilities
- `data/benchmarks/fone/`: Faroese Named Entity Recognition dataset
- `data/processed/training/pretrain_corpus.jsonl`: Preprocessed Faroese corpus for L_PT loss

### 3. Source Code Directory (`src/`)
Contains all implementation code for the project, organized into logical modules.

**Structure:**
```
src/
├── models/
│   ├── student.py
│   ├── teacher.py
│   ├── reference.py
│   └── tokenizers/
│       ├── faroese_tokenizer.py
│       └── tokenizer_utils.py
├── training/
│   ├── grpo.py
│   ├── ppo.py
│   ├── trainer.py
│   └── data_loader.py
├── rewards/
│   ├── teacher_logprob.py
│   ├── lpt_reward.py
│   ├── auxiliary_rewards.py
│   └── reward_combiner.py
├── evaluation/
│   ├── metrics.py
│   ├── foqa_evaluator.py
│   ├── fone_evaluator.py
│   └── perplexity_evaluator.py
└── utils/
    ├── logging_utils.py
    ├── checkpoint_utils.py
    ├── distributed_utils.py
    └── faroese_utils.py
```

**Purpose:** The core implementation of all algorithms, models, and support functionality.

**Key Files:**
- `src/models/student.py`: Implementation of the student model architecture
- `src/models/teacher.py`: Wrapper for accessing the teacher model
- `src/training/grpo.py`: Implementation of Group Relative Policy Optimization
- `src/training/ppo.py`: Implementation of Proximal Policy Optimization
- `src/rewards/teacher_logprob.py`: Reward function based on teacher model log probabilities
- `src/rewards/lpt_reward.py`: Implementation of the L_PT loss for preserving language capabilities
- `src/evaluation/foqa_evaluator.py`: Evaluation on the Faroese QA benchmark

### 4. Configuration Directory (`configs/`)
Contains configuration files for different experiments and model settings.

**Structure:**
```
configs/
├── models/
│   ├── student_configs/
│   │   ├── mistral_7b.yaml
│   │   ├── llama_7b.yaml
│   │   └── gpt_sw3_6_7b.yaml
│   └── teacher_configs/
│       ├── gemma_27b.yaml
│       ├── llama_70b.yaml
│       └── gpt4.yaml
├── training/
│   ├── grpo_base.yaml
│   ├── grpo_minillm.yaml
│   ├── ppo_base.yaml
│   └── ppo_minillm.yaml
└── reward/
    ├── teacher_logprob.yaml
    ├── lpt.yaml
    └── combined.yaml
```

**Purpose:** Centralized configuration management for all experiments, allowing easy parameter tuning and experiment tracking.

**Key Files:**
- `configs/models/student_configs/mistral_7b.yaml`: Configuration for Mistral 7B as the student model
- `configs/training/grpo_minillm.yaml`: Configuration for GRPO with MiniLLM-style rewards
- `configs/reward/combined.yaml`: Configuration for combining teacher log-prob and L_PT rewards

### 5. Scripts Directory (`scripts/`)
Contains utility scripts for data preprocessing, job submission, and result analysis.

**Structure:**
```
scripts/
├── slurm/
│   ├── train_grpo.slurm
│   ├── train_ppo.slurm
│   ├── eval_benchmarks.slurm
│   └── job_array.slurm
└── preprocessing/
    ├── prepare_faroese_corpus.py
    ├── tokenize_data.py
    ├── create_pretrain_dataset.py
    └── prepare_benchmarks.py
```

**Purpose:** Provides automation tools for various research tasks, including HPC job submission and data preparation.

**Key Files:**
- `scripts/slurm/train_grpo.slurm`: SLURM script for training with GRPO algorithm
- `scripts/slurm/train_ppo.slurm`: SLURM script for training with PPO algorithm
- `scripts/preprocessing/prepare_faroese_corpus.py`: Script to process raw Faroese data

### 6. Experiments Directory (`experiments/`)
Organizational structure for tracking different experimental runs.

**Structure:**
```
experiments/
├── grpo_teacher_logprob/
├── grpo_minillm_style/
├── ppo_teacher_logprob/
├── ppo_minillm_style/
└── ablation_studies/
    ├── no_lpt/
    ├── varying_group_size/
    └── teacher_comparison/
```

**Purpose:** Organizes experiments by algorithm and configuration, making it easy to locate results and artifacts from specific runs.

**Key Files:**
- `experiments/grpo_minillm_style/experiment_config.yaml`: Configuration for a specific experiment
- `experiments/ablation_studies/no_lpt/results.json`: Results from an ablation study without L_PT loss

### 7. Results Directory (`results/`)
Storage for all experiment results, model checkpoints, and evaluation metrics.

**Structure:**
```
results/
├── models/
│   ├── grpo_faroese_mistral/
│   ├── ppo_faroese_llama/
│   └── best_models/
├── metrics/
│   ├── foqa_results/
│   ├── fone_results/
│   └── perplexity/
└── analysis/
    ├── reward_curves/
    ├── convergence_analysis/
    └── sample_generations/
```

**Purpose:** Centralized storage for all experimental outputs, including trained models, metrics, and analysis.

**Key Files:**
- `results/models/grpo_faroese_mistral/checkpoint_10000.pt`: Checkpoint of the GRPO-trained Mistral model
- `results/metrics/foqa_results/comparison.csv`: Comparative results on the FoQA benchmark
- `results/analysis/sample_generations/qualitative_examples.json`: Samples of model generations

### 8. Notebooks Directory (`notebooks/`)
Jupyter notebooks for exploratory data analysis, visualization, and result interpretation.

**Structure:**
```
notebooks/
├── exploratory/
│   ├── faroese_corpus_analysis.ipynb
│   └── teacher_model_exploration.ipynb
├── training/
│   ├── reward_visualization.ipynb
│   └── training_curves.ipynb
└── evaluation/
    ├── benchmark_results.ipynb
    ├── qualitative_analysis.ipynb
    └── comparative_performance.ipynb
```

**Purpose:** Interactive environments for data exploration, visualization, and result analysis.

**Key Files:**
- `notebooks/exploratory/faroese_corpus_analysis.ipynb`: Analysis of the Faroese corpus statistics
- `notebooks/evaluation/benchmark_results.ipynb`: Visualizations of performance on benchmarks

### 9. Documentation Directory (`docs/`)
Comprehensive documentation for the project, including technical details and research findings.

**Structure:**
```
docs/
├── algorithms/
│   ├── grpo.md
│   ├── ppo.md
│   └── knowledge_distillation.md
├── models/
│   ├── student_models.md
│   └── teacher_models.md
├── faroese/
│   ├── language_overview.md
│   └── resources.md
├── results/
│   ├── main_findings.md
│   └── performance_comparison.md
└── presentations/
    ├── project_overview.pptx
    └── result_presentation.pptx
```

**Purpose:** Provides detailed documentation of algorithms, models, and research findings.

**Key Files:**
- `docs/algorithms/grpo.md`: Detailed explanation of the GRPO algorithm
- `docs/faroese/language_overview.md`: Overview of the Faroese language characteristics
- `docs/results/main_findings.md`: Summary of key research findings