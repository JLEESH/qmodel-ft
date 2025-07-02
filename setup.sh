# !/bin/bash

# This script sets up the environment for the project.
# Install required Python packages
pip install -r requirements.txt

# Initialize Weights & Biases
# wandb init --project "qmodel-ft" --entity "your_wandb_entity"

# Set up the PyTorch environment
export PYTORCH_ENABLE_MPS_FALLBACK=1