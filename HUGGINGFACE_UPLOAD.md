# 🤗 Automatic Hugging Face Upload for GR00T Models

This feature enables automatic uploading of your fine-tuned GR00T models to Hugging Face Hub during or after training.

## 🚀 Quick Start

1. **Install dependencies:**

   ```bash
   pip install huggingface_hub
   ```

2. **Set environment variables:**

   ```bash
   export HF_UPLOAD_ENABLED=true
   export HF_REPO_ID="your-username/gr00t-model-name"
   export HF_TOKEN="hf_your_token_here"
   ```

3. **Train with auto-upload:**
   ```bash
   python scripts/gr00t_finetune.py \
       --dataset-path ./demo_data/robot_sim.PickNPlace \
       --num-gpus 1
   ```

Your model will be automatically uploaded when training completes! 🎉

## 📁 New Files Added

- `gr00t/utils/huggingface_upload.py` - Core upload functionality
- `gr00t/utils/huggingface_callback.py` - Training integration callback
- `docs/huggingface_upload.md` - Complete documentation
- `examples/huggingface_upload_example.py` - Usage examples

## ✨ Features

- ✅ **Automatic upload** after training completion
- ✅ **Checkpoint uploads** during training (optional)
- ✅ **Auto-generated model cards** with training metadata
- ✅ **Public/private repository** support
- ✅ **Environment variable** configuration
- ✅ **Manual upload** option for existing models
- ✅ **Error handling** without interrupting training
- ✅ **Metadata tracking** (training params, dataset info, metrics)

## 🔧 Integration

The upload functionality is automatically integrated into the existing `TrainRunner` class. When environment variables are set, it will:

1. Create an uploader from environment variables
2. Add a callback to the trainer
3. Upload the final model after training
4. Generate a model card with training information
5. Upload checkpoints during training (if enabled)
