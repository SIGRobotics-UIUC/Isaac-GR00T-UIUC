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
Hugging Face upload callback for training integration.
"""

from typing import Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from gr00t.utils.huggingface_upload import (
    HuggingFaceUploader,
    extract_dataset_info,
    extract_training_info,
)


class HuggingFaceUploadCallback(TrainerCallback):
    """
    Callback for automatically uploading models to Hugging Face Hub during training.
    
    Features:
    - Upload final model after training completion
    - Optionally upload checkpoints during training
    - Automatic model card generation with training metadata
    - Error handling without interrupting training
    """
    
    def __init__(
        self,
        uploader: HuggingFaceUploader,
        upload_final: bool = True,
        upload_checkpoints: bool = False,
        checkpoint_upload_steps: Optional[int] = None,
    ):
        """
        Initialize the upload callback.
        
        Args:
            uploader: HuggingFaceUploader instance
            upload_final: Whether to upload the final model after training
            upload_checkpoints: Whether to upload checkpoints during training
            checkpoint_upload_steps: Upload checkpoints every N steps (if None, uploads based on save_steps)
        """
        self.uploader = uploader
        self.upload_final = upload_final
        self.upload_checkpoints = upload_checkpoints
        self.checkpoint_upload_steps = checkpoint_upload_steps
        self._uploaded_checkpoints = set()
    
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        model=None,
        tokenizer=None,
        **kwargs,
    ):
        """Called when a checkpoint is saved."""
        if not self.upload_checkpoints:
            return
        
        # Check if we should upload this checkpoint
        should_upload = False
        if self.checkpoint_upload_steps is not None:
            should_upload = state.global_step % self.checkpoint_upload_steps == 0
        else:
            should_upload = True  # Upload every time a checkpoint is saved
        
        if should_upload and state.global_step not in self._uploaded_checkpoints:
            try:
                checkpoint_path = f"{args.output_dir}/checkpoint-{state.global_step}"
                print(f"📤 Uploading checkpoint at step {state.global_step}...")
                
                self.uploader.upload_checkpoint(
                    checkpoint_path=checkpoint_path,
                    step=state.global_step,
                    commit_message=f"Training checkpoint at step {state.global_step}",
                )
                
                self._uploaded_checkpoints.add(state.global_step)
                print(f"✅ Checkpoint {state.global_step} uploaded successfully")
                
            except Exception as e:
                print(f"⚠️  Failed to upload checkpoint {state.global_step}: {str(e)}")
                # Don't raise the exception to avoid interrupting training
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        model=None,
        tokenizer=None,
        **kwargs,
    ):
        """Called when training ends."""
        if not self.upload_final:
            return
        
        try:
            print("🚀 Training completed! Uploading final model to Hugging Face Hub...")
            
            # Extract training information
            trainer = kwargs.get('trainer')
            training_info = extract_training_info(trainer) if trainer else {}
            
            # Add final training stats
            training_info.update({
                'final_step': state.global_step,
                'final_epoch': state.epoch,
                'total_flos': state.total_flos,
            })
            
            # Add performance metrics from logs
            performance_metrics = {}
            if logs:
                for key, value in logs.items():
                    if 'loss' in key.lower() or 'accuracy' in key.lower() or 'metric' in key.lower():
                        performance_metrics[key] = value
            
            # Extract dataset information if available
            dataset_info = {}
            if trainer and hasattr(trainer, 'train_dataset'):
                dataset_info = extract_dataset_info(trainer.train_dataset)
            
            # Upload the final model
            self.uploader.upload_model(
                model_path=args.output_dir,
                commit_message=f"Final model after {state.global_step} training steps",
                training_info=training_info,
                dataset_info=dataset_info,
                performance_metrics=performance_metrics,
                include_model_card=True,
            )
            
            print("🎉 Final model uploaded successfully to Hugging Face Hub!")
            
        except Exception as e:
            print(f"⚠️  Failed to upload final model: {str(e)}")
            # Don't raise the exception as training is already complete
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        model=None,
        tokenizer=None,
        **kwargs,
    ):
        """Called when logging occurs."""
        # This can be used to log upload progress or metrics
        pass