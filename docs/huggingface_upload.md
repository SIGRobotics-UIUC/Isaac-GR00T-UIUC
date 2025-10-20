# Automatic Hugging Face Upload for GR00T Models

This guide explains how to automatically upload your fine-tuned GR00T models to Hugging Face Hub.

## Features

- ✅ Automatic model upload after training completion
- ✅ Optional checkpoint uploads during training
- ✅ Automatic model card generation with training metadata
- ✅ Support for both public and private repositories
- ✅ Environment variable configuration
- ✅ Manual upload option

## Quick Setup

### 1. Install Required Dependencies

```bash
pip install huggingface_hub
```

### 2. Set Environment Variables

```bash
# Enable automatic upload
export HF_UPLOAD_ENABLED=true

# Set your repository (will be created if it doesn't exist)
export HF_REPO_ID="your-username/gr00t-finetuned-model"

# Set your Hugging Face token
export HF_TOKEN="hf_your_token_here"

# Optional: Make repository private
export HF_PRIVATE=true

# Optional: Upload checkpoints during training (not just final model)
export HF_UPLOAD_ON_SAVE=true
```

### 3. Run Training

The upload functionality is automatically integrated into the existing training pipeline. Just run normal training script.

## Environment Variables Reference

| Variable            | Description                                 | Default | Required |
| ------------------- | ------------------------------------------- | ------- | -------- |
| `HF_UPLOAD_ENABLED` | Enable automatic uploading                  | `false` | Yes      |
| `HF_REPO_ID`        | Repository ID (e.g., `username/model-name`) | -       | Yes      |
| `HF_TOKEN`          | Your Hugging Face API token                 | -       | Yes      |
| `HF_PRIVATE`        | Create private repository                   | `false` | No       |
| `HF_UPLOAD_ON_SAVE` | Upload checkpoints during training          | `false` | No       |

## What Gets Uploaded

When you upload a model, the following files are included:

- **Model files**: `pytorch_model.bin`, `config.json`, etc.
- **Configuration**: All model configuration files
- **README.md**: Auto-generated model card with:
  - Model description and usage
  - Training parameters
  - Dataset information
  - Performance metrics
  - Citation information
- **Metadata**: `upload_metadata.json` with upload details

## Example Model Card

The generated model card includes:

````markdown
# GR00T N1.5 Fine-tuned Model

This model is a fine-tuned version of nvidia/GR00T-N1.5-3B for robotic policy learning.

## Training Details

- Training Script: gr00t_finetune.py
- Total Steps: 1500
- Learning Rate: 5e-5
- Batch Size: 4

## Dataset

- Dataset Path: /data/robot_sim.PickNPlace
- Number of Episodes: 1000
- Embodiments: fourier_gr1

## Usage

```python
from gr00t.model.gr00t_n1 import GR00T_N1_5
model = GR00T_N1_5.from_pretrained("username/gr00t-finetuned-model")
```
````

## Advanced Configuration

### Custom Upload Callback

If you need more control, you can create a custom callback:

```python
from gr00t.utils.huggingface_upload import HuggingFaceUploader
from gr00t.utils.huggingface_callback import HuggingFaceUploadCallback

# Create uploader with custom settings
uploader = HuggingFaceUploader(
    repo_id="username/model",
    token="hf_token",
    private=True,
    upload_on_save=False
)

# Create callback
callback = HuggingFaceUploadCallback(
    uploader=uploader,
    upload_final=True,
    upload_checkpoints=False
)

# Add to trainer
trainer.add_callback(callback)
```

### Upload Only Specific Checkpoints

```python
# Upload every 1000 steps
callback = HuggingFaceUploadCallback(
    uploader=uploader,
    upload_checkpoints=True,
    checkpoint_upload_steps=1000
)
```

## Troubleshooting

### Authentication Issues

```bash
# Login to Hugging Face CLI (alternative to token)
huggingface-cli login

# Or set token in environment
export HF_TOKEN="hf_your_token_here"
```

### Repository Issues

```bash
# Check if repository exists
import requests
response = requests.get("https://huggingface.co/api/models/username/model-name")
print(response.status_code)  # Should be 200 if exists
```

### Large Model Files

For models >5GB, consider using Git LFS:

```bash
# Install git-lfs if not already installed
git lfs install

# The uploader automatically handles LFS for large files
```

### Network Issues

```python
# Add retry logic for network issues
uploader = HuggingFaceUploader(
    repo_id="username/model",
    token="token",
    # Will automatically retry on network failures
)
```

## Integration Examples

### With Weights & Biases

```bash
export HF_UPLOAD_ENABLED=true
export HF_REPO_ID="username/model"
export WANDB_PROJECT="gr00t-training"

python scripts/gr00t_finetune.py \\
    --report_to wandb \\
    --output_dir ./results
```

### With Multiple Datasets

```bash
export HF_UPLOAD_ENABLED=true
export HF_REPO_ID="username/gr00t-multi-embodiment"

python scripts/gr00t_finetune.py \\
    --dataset_path /data/dataset1 /data/dataset2 \\
    --output_dir ./multi_results
```

### With LoRA Fine-tuning

```bash
export HF_UPLOAD_ENABLED=true
export HF_REPO_ID="username/gr00t-lora-model"

python scripts/gr00t_finetune.py \\
    --use_lora \\
    --lora_r 16 \\
    --lora_alpha 32 \\
    --output_dir ./lora_results
```

## Best Practices

1. **Repository Naming**: Use descriptive names like `gr00t-robot-task-embodiment`
2. **Documentation**: Add detailed information in training_info and dataset_info
3. **Versioning**: Use different repositories or branches for different versions
4. **Privacy**: Set `HF_PRIVATE=true` for proprietary datasets/models
5. **Checkpoints**: Only upload checkpoints for long training runs (>1000 steps)
6. **Testing**: Test upload with a small dummy model first

## Security Notes

- Store tokens securely (use environment variables, not hardcoded)
- Use private repositories for sensitive models
- Consider using organization repositories for team projects
- Regularly rotate API tokens

## Support

For issues:

1. Check this documentation
2. Verify environment variables are set correctly
3. Test with a simple manual upload first
4. Check Hugging Face Hub status
5. Create an issue in this repository
