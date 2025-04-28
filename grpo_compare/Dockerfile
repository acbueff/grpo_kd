# Dockerfile for GRPO Evaluation Comparison

# 1. Base Image: Use an official NVIDIA CUDA image.
# Choose a version compatible with PyTorch/vLLM/Unsloth requirements and host driver.
# CUDA 12.1 is generally a good choice for recent library versions.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 2. Set up Environment & Install System Dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python/python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip

# 3. Set Working Directory
WORKDIR /app

# 4. Copy Requirements File FIRST (leverages Docker cache)
# Assumes requirements.txt is inside the grpo_evaluation folder relative to build context
COPY grpo_evaluation/requirements.txt ./grpo_evaluation/requirements.txt

# 5. Install Python Dependencies
# Consider pinning exact versions in requirements.txt for stricter reproducibility
RUN python -m pip install --no-cache-dir -r grpo_evaluation/requirements.txt
# Note: Some implementations (like unsloth) might have complex dependencies
# or require specific CUDA toolkit versions not covered by the base image's default tools.
# If build errors occur here, check specific library documentation.

# 6. Copy the entire project code into the container
# This includes grpo_evaluation and all implementation subfolders (verl-main, etc.)
# assuming the build context is the 'grpo_compare' directory.
COPY . .

# 7. Make Python scripts executable if needed (optional, good practice)
# RUN chmod +x grpo_evaluation/*.py run_evaluation.py

# 8. Ensure GPUs are visible and usable by NVIDIA Container Toolkit
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# 9. Define the default command to run when the container starts
# Runs the main evaluation script via the python module execution
ENTRYPOINT ["python", "-m", "grpo_evaluation.main"]

# Example CMD providing default arguments if none are specified on 'docker run'
# CMD ["--implementations", "trl", "unsloth", "--output_dir", "/app/results"] 