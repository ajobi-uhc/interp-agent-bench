"""Shared Modal images for GPU techniques."""

import modal

# Base image for HuggingFace models
hf_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch",
    "transformers",
    "accelerate",
    "safetensors",
    "peft",
    "cloudpickle",
)

# Image for general ML tasks
ml_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch",
    "transformers",
    "accelerate",
    "safetensors",
    "numpy",
    "scipy",
    "scikit-learn",
    "cloudpickle",
)
