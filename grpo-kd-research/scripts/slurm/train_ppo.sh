#!/bin/bash
#SBATCH --job-name=ppo_faroese
#SBATCH --output=logs/ppo_faroese_%j.out
#SBATCH --error=logs/ppo_faroese_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Print info about the job
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# Load necessary modules (modify as needed for your HPC environment)
module load cuda/11.7
module load python/3.10

# Activate virtual environment
source /path/to/venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Set cache directories for Hugging Face models and datasets
export HF_HOME=/scratch/user/huggingface
export HF_DATASETS_CACHE=/scratch/user/huggingface/datasets

# Set PYTHONPATH to include project directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Parse arguments
CONFIG_FILE=${1:-"configs/training/ppo_teacher_ll.yaml"}
OUTPUT_DIR=${2:-"results/ppo_teacher_ll"}
STUDENT_MODEL=${3:-"mistral-7b-v0.1"}
TEACHER_MODEL=${4:-"gemma-7b"}

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Print configuration
echo "Configuration:"
echo "  - Config file: $CONFIG_FILE"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - Student model: $STUDENT_MODEL"
echo "  - Teacher model: $TEACHER_MODEL"

# Run training script with distributed data parallel
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=$(($RANDOM + 10000)) \
    src/training/train_ppo.py \
    --config $CONFIG_FILE \
    --output_dir $OUTPUT_DIR \
    --student_model $STUDENT_MODEL \
    --teacher_model $TEACHER_MODEL \
    --fp16 \
    --gradient_checkpointing \
    --deepspeed configs/deepspeed/ds_config_zero2.json \
    "$@"  # Pass any remaining arguments

# Print completion message
echo "Job completed at $(date)" 