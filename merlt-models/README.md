# MERL-T Models

> Proprietary trained models for the MERL-T framework

## CONFIDENTIAL

This repository contains proprietary trained models and is **NOT for public distribution**.

## Contents

```
merlt-models/
├── weights/          # Trained model weights
├── checkpoints/      # Training checkpoints
├── config/           # Model configurations
│   └── experts/      # Expert-specific configs
└── merlt_models/     # Python loader utilities
```

## Requirements

- Git LFS (for large model files)
- Python 3.10+
- PyTorch 2.0+ (for model loading)

## Setup

```bash
# Install Git LFS
git lfs install

# Clone (requires access)
git clone https://github.com/visualex/merlt-models-private
cd merlt-models-private

# Install loader utilities
pip install -e .
```

### Local start script

```bash
./start_dev.sh
```

This script pulls Git LFS assets (if available), installs the package in
editable mode, and prints available models and configs.

## Usage

```python
from merlt_models import load_model, load_config, list_models

# List available models
print(list_models())

# Load a model
model = load_model("expert_literal_v1", device="cuda")

# Load configuration
config = load_config("rlcf_training")
```

## Git LFS

Large model files are tracked with Git LFS. After cloning:

```bash
git lfs pull
```

## Access Control

This repository is private. Contact the team for access.

## License

Proprietary - All rights reserved (c) 2026 VisuaLex

Unauthorized copying, distribution, or modification is strictly prohibited.
