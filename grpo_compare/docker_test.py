#!/usr/bin/env python3
"""
Test script to verify key library installations within the Docker container.
"""

import sys
import importlib

# Libraries to test based on requirements.txt
libraries = [
    # Core ML
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    # GRPO Implementations/Support
    "trl",
    "unsloth",
    "vllm",
    # Monitoring & Logging
    "pynvml",
    "wandb",
    # Data Handling & Plotting
    "numpy",
    "pandas",
    "matplotlib",
    "tqdm",
    "tabulate",
    # Optional Jupyter
    # "IPython", # Usually not needed for script execution tests
    # "jupyter",
]

print("--- Docker Environment Library Check ---")
print(f"Python Version: {sys.version}")

all_successful = True

for lib_name in libraries:
    try:
        lib = importlib.import_module(lib_name)
        version = getattr(lib, '__version__', 'N/A')
        print(f"[ OK ] {lib_name} imported successfully (Version: {version})")

        # Add specific checks for key libraries
        if lib_name == "torch":
            print(f"    - PyTorch CUDA available: {lib.cuda.is_available()}")
            if lib.cuda.is_available():
                print(f"    - CUDA devices found: {lib.cuda.device_count()}")
                print(f"    - Current device: {lib.cuda.current_device()}")
                print(f"    - Device name: {lib.cuda.get_device_name(lib.cuda.current_device())}")
        elif lib_name == "pynvml":
            try:
                lib.nvmlInit()
                print("    - pynvml initialized successfully.")
                count = lib.nvmlDeviceGetCount()
                print(f"    - Found {count} GPU device(s) via pynvml.")
                # handle = lib.nvmlDeviceGetHandleByIndex(0)
                # mem_info = lib.nvmlDeviceGetMemoryInfo(handle)
                # print(f"    - GPU 0 Memory (Total): {mem_info.total / 1024**3:.2f} GB")
                lib.nvmlShutdown()
            except Exception as e:
                print(f"    - pynvml check failed: {e}")
                all_successful = False
        elif lib_name == "unsloth":
             # Unsloth might not have an easily accessible __version__ or simple check
             print(f"    - Unsloth imported. Specific checks might require model loading.")
        elif lib_name == "vllm":
             # vLLM might not have an easily accessible __version__
             print(f"    - vLLM imported. Specific checks might require LLM instantiation.")


    except ImportError:
        print(f"[FAIL] {lib_name} could not be imported.")
        # Distinguish between expected failures (e.g., vLLM if GPU unsuitable) vs. installation problems
        if lib_name in ["vllm", "unsloth"]:
             print(f"    - Note: Import failure for {lib_name} might be due to specific hardware/driver requirements not met in the test environment, not necessarily an installation issue.")
        all_successful = False
    except Exception as e:
        print(f"[ERROR] Error during check for {lib_name}: {e}")
        all_successful = False

print("----------------------------------------")

if all_successful:
    print("All essential libraries seem to be installed and importable.")
    sys.exit(0)
else:
    print("Some libraries failed to import or caused errors. Please check the Docker build logs and environment.")
    sys.exit(1) 