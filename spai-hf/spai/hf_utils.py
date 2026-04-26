# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi, ModelCard
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def push_model_to_hub(
    model_path: str,
    repo_id: str,
    hf_token: Optional[str] = None,
    commit_message: str = "Upload trained SPAI model",
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Upload model checkpoint and configuration to Hugging Face Hub.
    
    Args:
        model_path: Path to the directory containing model checkpoint and config
        repo_id: Repository ID in format 'username/repo-name'
        hf_token: Hugging Face API token. If None, will try to use HF_TOKEN env var
        commit_message: Commit message for the upload
        logger: Logger instance for logging messages
        
    Returns:
        True if upload was successful, False otherwise
    """
    if not HF_AVAILABLE:
        msg = "huggingface_hub is not installed. Install it with: pip install huggingface_hub"
        if logger:
            logger.error(msg)
        else:
            print(f"ERROR: {msg}")
        return False
    
    # Get token from argument or environment variable
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        msg = "HF_TOKEN not provided and HF_TOKEN environment variable not set"
        if logger:
            logger.error(msg)
        else:
            print(f"ERROR: {msg}")
        return False
    
    try:
        api = HfApi(token=hf_token)
        
        if logger:
            logger.info(f"Uploading model to Hugging Face Hub: {repo_id}")
        else:
            print(f"Uploading model to Hugging Face Hub: {repo_id}")
        
        # Create repo if it doesn't exist
        try:
            api.create_repo(
                repo_id=repo_id,
                exist_ok=True,
                private=False
            )
            if logger:
                logger.info(f"Repository {repo_id} ready")
            else:
                print(f"Repository {repo_id} ready")
        except Exception as e:
            if logger:
                logger.warning(f"Could not create repo (may already exist): {e}")
            else:
                print(f"Warning: Could not create repo (may already exist): {e}")
        
        # Upload model folder
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            commit_message=commit_message,
            ignore_patterns=["*.log", "events.out.tfevents*"]
        )
        
        if logger:
            logger.info(f"Model successfully uploaded to {repo_id}")
        else:
            print(f"Model successfully uploaded to {repo_id}")
        
        return True
        
    except Exception as e:
        msg = f"Error uploading model to Hugging Face: {str(e)}"
        if logger:
            logger.error(msg)
        else:
            print(f"ERROR: {msg}")
        return False


def create_model_card(
    repo_id: str,
    model_description: str,
    task: str = "image-classification",
    tags: Optional[list] = None,
    hf_token: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Create and push a model card to Hugging Face Hub.
    
    Args:
        repo_id: Repository ID in format 'username/repo-name'
        model_description: Description of the model
        task: Model task type (default: image-classification)
        tags: List of tags for the model
        hf_token: Hugging Face API token
        logger: Logger instance for logging messages
        
    Returns:
        True if model card creation was successful, False otherwise
    """
    if not HF_AVAILABLE:
        msg = "huggingface_hub is not installed. Install it with: pip install huggingface_hub"
        if logger:
            logger.error(msg)
        else:
            print(f"ERROR: {msg}")
        return False
    
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        msg = "HF_TOKEN not provided and HF_TOKEN environment variable not set"
        if logger:
            logger.error(msg)
        else:
            print(f"ERROR: {msg}")
        return False
    
    try:
        if tags is None:
            tags = ["ai-generated-image-detection", "forensics", "mfm"]
        
        card_data = {
            "tags": tags,
            "license": "apache-2.0",
        }
        
        # Create markdown content for the card
        content = f"""
# SPAI Model

## Model Details

This is a model trained using the SPAI framework for AI-generated image detection.

### Model Description

{model_description}

### Intended Use

This model is designed to detect AI-generated images using the Masked Frequency Modeling (MFM) approach.

### Training Data

The model was trained on a combination of:
- Real images from standard datasets (COCO, ImageNet, OpenImages, FODB, RAISE)
- AI-generated images from various generators (DALL-E 2, DALL-E 3, Firefly, Flux, GigaGAN, GLIDE, Midjourney v5/v6.1, Stable Diffusion v1.3/1.4/2.0/3.0/XL)

## How to Use

```python
from PIL import Image
import torch
from spai.models import build_model

# Load the model
model = build_model(config)  # Load your config
model.load_state_dict(torch.load('model_checkpoint.pth'))
model.eval()

# Prepare image
image = Image.open('path/to/image.jpg')
# ... apply preprocessing ...

# Get prediction
with torch.no_grad():
    output = model(image_tensor)
    probability = torch.sigmoid(output).item()
```

## Training Procedure

The model was trained using:
- Masked Frequency Modeling (MFM) pre-training approach
- Distributed training with PyTorch
- Vision Transformer backbone with frequency-domain filters

## License

Apache License 2.0

## Citation

If you use this model, please cite the SPAI project.
"""
        
        card = ModelCard(content)
        card.push_to_hub(repo_id, token=hf_token)
        
        if logger:
            logger.info(f"Model card created for {repo_id}")
        else:
            print(f"Model card created for {repo_id}")
        
        return True
        
    except Exception as e:
        msg = f"Error creating model card: {str(e)}"
        if logger:
            logger.error(msg)
        else:
            print(f"ERROR: {msg}")
        return False
