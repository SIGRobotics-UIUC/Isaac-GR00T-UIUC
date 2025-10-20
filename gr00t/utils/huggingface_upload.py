# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Utilities for automatically uploading trained models to Hugging Face Hub.
"""

import json
import os
import shutil
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from huggingface_hub import HfApi, create_repo
# Handle different import locations for RepositoryNotFoundError across huggingface_hub versions
try:
    from huggingface_hub import RepositoryNotFoundError
except ImportError:
    # Fallback for older versions or different import locations
    RepositoryNotFoundError = Exception
from transformers import Trainer

from gr00t.model.gr00t_n1 import GR00T_N1_5


class HuggingFaceUploader:
    """
    Utility class for uploading GR00T models to Hugging Face Hub.
    
    Features:
    - Automatic model upload after training completion
    - Upload with model cards, configuration files, and metadata
    - Support for both public and private repositories
    - Automatic README generation with training information
    - Error handling and retry mechanisms
    """
    
    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        create_repo_if_not_exists: bool = True,
        auto_upload: bool = True,
        upload_on_save: bool = False,
    ):
        """
        Initialize the Hugging Face uploader.
        
        Args:
            repo_id: Repository ID in format "username/repository_name"
            token: Hugging Face API token. If None, uses HF_TOKEN environment variable
            private: Whether to create a private repository
            create_repo_if_not_exists: Whether to create repository if it doesn't exist
            auto_upload: Whether to automatically upload after training completion
            upload_on_save: Whether to upload on every model save (checkpoints)
        """
        self.repo_id = repo_id
        self.token = token or os.getenv("HF_TOKEN")
        self.private = private
        self.create_repo_if_not_exists = create_repo_if_not_exists
        self.auto_upload = auto_upload
        self.upload_on_save = upload_on_save
        
        if not self.token:
            raise ValueError(
                "Hugging Face token is required. Set HF_TOKEN environment variable or pass token parameter."
            )
        
        self.api = HfApi(token=self.token)
        
        # Create repository if needed
        if self.create_repo_if_not_exists:
            self._create_repo_if_needed()
    
    def _create_repo_if_needed(self):
        """Create repository if it doesn't exist."""
        try:
            self.api.repo_info(repo_id=self.repo_id, repo_type="model")
            print(f"Repository {self.repo_id} already exists.")
        except RepositoryNotFoundError:
            print(f"Creating repository {self.repo_id}...")
            create_repo(
                repo_id=self.repo_id,
                repo_type="model",
                private=self.private,
                token=self.token,
            )
            print(f"Repository {self.repo_id} created successfully.")
    
    def generate_model_card(
        self,
        model_path: str,
        training_info: Optional[Dict] = None,
        dataset_info: Optional[Dict] = None,
        performance_metrics: Optional[Dict] = None,
    ) -> str:
        """
        Generate a model card (README.md) for the uploaded model.
        
        Args:
            model_path: Path to the trained model
            training_info: Dictionary containing training information
            dataset_info: Dictionary containing dataset information
            performance_metrics: Dictionary containing performance metrics
        
        Returns:
            Model card content as string
        """
        model_card = f"""---
library_name: transformers
base_model: nvidia/GR00T-N1.5-3B
tags:
- robotics
- policy-learning
- gr00t
- embodied-ai
license: apache-2.0
---

# GR00T N1.5 Fine-tuned Model

This model is a fine-tuned version of [nvidia/GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B) for robotic policy learning.

## Model Description

GR00T (Generalist Robot 00 Technology) is a foundation model for general-purpose robotics. This fine-tuned version has been adapted for specific robotic tasks and embodiments.

## Model Details

- **Base Model**: nvidia/GR00T-N1.5-3B
- **Model Type**: Multimodal Policy Learning Model
- **Framework**: PyTorch + Transformers
"""

        if training_info:
            model_card += f"""
## Training Details

- **Training Script**: {training_info.get('script', 'N/A')}
- **Total Steps**: {training_info.get('total_steps', 'N/A')}
- **Learning Rate**: {training_info.get('learning_rate', 'N/A')}
- **Batch Size**: {training_info.get('batch_size', 'N/A')}
- **Training Duration**: {training_info.get('duration', 'N/A')}
- **GPU**: {training_info.get('gpu_info', 'N/A')}
"""

        if dataset_info:
            model_card += f"""
## Dataset

- **Dataset Path**: {dataset_info.get('path', 'N/A')}
- **Number of Episodes**: {dataset_info.get('num_episodes', 'N/A')}
- **Total Steps**: {dataset_info.get('total_steps', 'N/A')}
- **Embodiments**: {dataset_info.get('embodiments', 'N/A')}
"""

        if performance_metrics:
            model_card += f"""
## Performance Metrics

"""
            for metric, value in performance_metrics.items():
                model_card += f"- **{metric}**: {value}\n"

        model_card += f"""
## Usage

```python
from gr00t.model.gr00t_n1 import GR00T_N1_5

# Load the fine-tuned model
model = GR00T_N1_5.from_pretrained("{self.repo_id}")

# Use for inference
# ... (add your inference code here)
```

## Citation

```bibtex
@misc{{groot_finetune,
    title={{GR00T N1.5 Fine-tuned Model}},
    author={{Your Name}},
    year={{2025}},
    url={{https://huggingface.co/{self.repo_id}}}
}}
```

## License

This model is licensed under the Apache 2.0 License.
"""
        
        return model_card
    
    def upload_model(
        self,
        model_path: str,
        commit_message: Optional[str] = None,
        training_info: Optional[Dict] = None,
        dataset_info: Optional[Dict] = None,
        performance_metrics: Optional[Dict] = None,
        include_model_card: bool = True,
    ) -> str:
        """
        Upload a trained model to Hugging Face Hub.
        
        Args:
            model_path: Path to the saved model directory
            commit_message: Custom commit message for the upload
            training_info: Training information for model card
            dataset_info: Dataset information for model card
            performance_metrics: Performance metrics for model card
            include_model_card: Whether to generate and upload a model card
        
        Returns:
            URL to the uploaded model repository
        """
        try:
            print(f"Starting upload of model from {model_path} to {self.repo_id}...")
            
            # Validate model path
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                raise FileNotFoundError(f"Model path {model_path} does not exist")
            
            # Create temporary directory for upload preparation
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Copy model files to temp directory
                print("Preparing model files for upload...")
                
                # Copy all model files
                for item in model_path_obj.iterdir():
                    if item.is_file():
                        shutil.copy2(item, temp_path / item.name)
                    elif item.is_dir() and item.name not in ['.git', '__pycache__', '.DS_Store']:
                        shutil.copytree(item, temp_path / item.name, ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
                
                # Generate and save model card if requested
                if include_model_card:
                    model_card_content = self.generate_model_card(
                        str(model_path_obj),
                        training_info=training_info,
                        dataset_info=dataset_info,
                        performance_metrics=performance_metrics,
                    )
                    (temp_path / "README.md").write_text(model_card_content, encoding='utf-8')
                
                # Create metadata file
                metadata = {
                    "model_type": "gr00t_n1_5",
                    "framework": "pytorch",
                    "base_model": "nvidia/GR00T-N1.5-3B",
                    "upload_timestamp": datetime.now().isoformat(),
                }
                if training_info:
                    metadata.update(training_info)
                
                (temp_path / "upload_metadata.json").write_text(
                    json.dumps(metadata, indent=2), encoding='utf-8'
                )
                
                # Upload to Hugging Face Hub
                commit_msg = commit_message or f"Upload fine-tuned GR00T model from {model_path_obj.name}"
                
                print("Uploading files to Hugging Face Hub...")
                self.api.upload_folder(
                    folder_path=str(temp_path),
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message=commit_msg,
                    token=self.token,
                )
            
            repo_url = f"https://huggingface.co/{self.repo_id}"
            print(f"✅ Model uploaded successfully to {repo_url}")
            return repo_url
            
        except Exception as e:
            print(f"❌ Error uploading model: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise e
    
    def upload_checkpoint(
        self,
        checkpoint_path: str,
        step: int,
        commit_message: Optional[str] = None,
    ) -> str:
        """
        Upload a training checkpoint to a branch.
        
        Args:
            checkpoint_path: Path to the checkpoint directory
            step: Training step number
            commit_message: Custom commit message
        
        Returns:
            URL to the uploaded checkpoint
        """
        branch_name = f"checkpoint-{step}"
        
        try:
            # Create branch for this checkpoint
            self.api.create_branch(
                repo_id=self.repo_id,
                branch=branch_name,
                repo_type="model",
                token=self.token,
            )
        except Exception:
            pass  # Branch might already exist
        
        commit_msg = commit_message or f"Upload checkpoint at step {step}"
        
        self.api.upload_folder(
            folder_path=checkpoint_path,
            repo_id=self.repo_id,
            repo_type="model",
            revision=branch_name,
            commit_message=commit_msg,
            token=self.token,
        )
        
        return f"https://huggingface.co/{self.repo_id}/tree/{branch_name}"


def create_uploader_from_env() -> Optional[HuggingFaceUploader]:
    """
    Create a HuggingFaceUploader from environment variables.
    
    Environment variables:
    - HF_UPLOAD_ENABLED: Set to "true" to enable automatic uploading
    - HF_REPO_ID: Repository ID (e.g., "username/model-name")
    - HF_TOKEN: Hugging Face API token
    - HF_PRIVATE: Set to "true" for private repositories
    - HF_UPLOAD_ON_SAVE: Set to "true" to upload on every checkpoint save
    
    Returns:
        HuggingFaceUploader instance if enabled, None otherwise
    """
    if not os.getenv("HF_UPLOAD_ENABLED", "false").lower() == "true":
        return None
    
    repo_id = os.getenv("HF_REPO_ID")
    if not repo_id:
        print("⚠️  HF_REPO_ID not set. Skipping Hugging Face upload.")
        return None
    
    token = os.getenv("HF_TOKEN")
    if not token:
        print("⚠️  HF_TOKEN not set. Skipping Hugging Face upload.")
        return None
    
    private = os.getenv("HF_PRIVATE", "false").lower() == "true"
    upload_on_save = os.getenv("HF_UPLOAD_ON_SAVE", "false").lower() == "true"
    
    print(f"🚀 Initializing Hugging Face uploader for repository: {repo_id}")
    
    return HuggingFaceUploader(
        repo_id=repo_id,
        token=token,
        private=private,
        upload_on_save=upload_on_save,
    )


def extract_training_info(trainer: Trainer) -> Dict:
    """Extract training information from a Trainer object."""
    training_info = {}
    
    if hasattr(trainer, 'args'):
        args = trainer.args
        training_info.update({
            'learning_rate': getattr(args, 'learning_rate', 'N/A'),
            'batch_size': getattr(args, 'per_device_train_batch_size', 'N/A'),
            'total_steps': getattr(args, 'max_steps', 'N/A'),
            'num_epochs': getattr(args, 'num_train_epochs', 'N/A'),
            'output_dir': getattr(args, 'output_dir', 'N/A'),
        })
    
    if hasattr(trainer, 'state'):
        state = trainer.state
        training_info.update({
            'global_step': getattr(state, 'global_step', 'N/A'),
            'epoch': getattr(state, 'epoch', 'N/A'),
        })
    
    # Add GPU information
    if torch.cuda.is_available():
        training_info['gpu_info'] = f"{torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)"
    
    return training_info


def extract_dataset_info(dataset) -> Dict:
    """Extract dataset information."""
    dataset_info = {}
    
    if hasattr(dataset, 'dataset_path'):
        dataset_info['path'] = str(dataset.dataset_path)
    
    if hasattr(dataset, '__len__'):
        dataset_info['total_steps'] = len(dataset)
    
    # Try to get embodiment information
    if hasattr(dataset, 'embodiment_tag'):
        dataset_info['embodiments'] = str(dataset.embodiment_tag)
    elif hasattr(dataset, 'embodiment_tags'):
        dataset_info['embodiments'] = [str(tag) for tag in dataset.embodiment_tags]
    
    return dataset_info