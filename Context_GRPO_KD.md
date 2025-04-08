# GRPO-KD: Context Document for LLM Assistant

## Research Overview

This document provides context for the LLM assistant regarding the research project on integrating Group Relative Policy Optimization (GRPO) with Knowledge Distillation (KD) techniques for training a Faroese language model. The research aims to enhance a smaller language model's capabilities in the low-resource Faroese language by leveraging knowledge from a larger teacher model while preserving the student model's language-specific strengths.

## Project Goals

1. Implement and adapt GRPO algorithm with knowledge distillation rewards inspired by MiniLLM and GEMMA 3
2. Train a smaller language model (7B parameters) for Faroese using a larger teacher model (27B-70B parameters)
3. Evaluate the trained model on Faroese benchmarks, particularly FoQA
4. Compare performance against baseline approaches and larger models
5. Analyze the effectiveness of different reward components (teacher log-probability, L_PT)

## Key Algorithms

The research centers around four main algorithmic approaches:

1. **GRPO with Teacher Log-Likelihood Reward**: Using the teacher model's log-probability of the student's output as reward
2. **GRPO with MiniLLM-style Reward**: Combining teacher log-probability with a language modeling loss (L_PT)
3. **PPO with Teacher Log-Likelihood Reward**: Standard PPO using teacher model's log-probability as reward
4. **PPO with MiniLLM-style Reward**: PPO with combined teacher log-probability and L_PT

The GRPO approach eliminates the need for a separate value function by using a group baseline, making it more memory-efficient than standard PPO.

## Models and Data

### Student Models
- Mistral 7B (adapted for Faroese)
- LLaMA-2 7B (adapted for Faroese)
- GPT-SW3 6.7B (Nordic language specialist)

### Teacher Models
- GEMMA-3 27B (instruction-tuned)
- LLaMA-2 70B Chat
- GPT-4 (via API, for evaluation and possibly data generation)

### Datasets
- Faroese corpora from Wikipedia, news sources, and general text
- FoQA (Faroese Question-Answering dataset)
- FoNE (Faroese Named Entity Recognition)
- EuroEval benchmark (includes Faroese language tasks)
- Parallel corpora for Faroese-English, Faroese-Danish

## Technical Requirements

- The research will be conducted on an HPC cluster using SLURM for job scheduling
- Training will require multiple GPUs for the teacher model and efficient batching
- Distributed training should be supported for scaling experiments
- Checkpointing is essential for resuming training and evaluating at different stages

## Constraints and Guidelines

### Scope Constraints

The LLM assistant should:

1. **ONLY** create files and directories within the defined project structure
2. Focus on implementing the four algorithmic approaches outlined in the pseudocode
3. Prioritize Faroese language support in all components
4. Ensure implementations are compatible with HPC/SLURM environment
5. Consider memory efficiency, especially for GRPO which aims to reduce memory requirements

### Out of Scope

The LLM assistant should NOT:

1. Implement unrelated algorithms or language model architectures
2. Create training scripts for languages other than Faroese (except for multilingual comparisons)
3. Design web interfaces or deployment solutions (research-focused project)
4. Attempt to create entirely new language models from scratch (focus on adaptation)
5. Deviate from the defined folder structure

### File Creation Guidelines

When creating files, the assistant should:

1. Place implementation code in appropriate `src/` subdirectories
2. Put configuration files in the `configs/` directory
3. Create SLURM scripts in `scripts/slurm/`
4. Document algorithms and approaches in `docs/`
5. Ensure each file has a clear purpose aligned with the research goals

### Expected Deliverables

The key files the assistant should help develop include:

1. Core algorithm implementations in `src/training/` (grpo.py, ppo.py)
2. Reward functions in `src/rewards/` (teacher_logprob.py, lpt_reward.py)
3. Training configurations in `configs/training/`
4. SLURM scripts for running experiments in `scripts/slurm/`
5. Evaluation code for Faroese benchmarks in `src/evaluation/`
6. Documentation of the approach in `docs/`

## Technical References

The implementation should reference the following key concepts:

1. **GRPO**: Group-based advantages, no value network, relative normalization
2. **MiniLLM**: Reverse KL divergence, language modeling loss (L_PT)
3. **Faroese Language**: Character set including ð, ø, á, í, ó, ú, ý, æ
4. **Knowledge Distillation**: Teacher model scoring, token-level vs sequence-level
5. **HPC/SLURM**: Job scheduling, distributed training, checkpointing

## Conclusion

This context document provides the LLM assistant with the necessary information to help develop the GRPO-KD research project focusing on Faroese language model enhancement. The assistant should use this document as a reference to ensure all contributions remain within the scope of the research goals and follow the specified project structure.