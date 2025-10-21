NVIDIA GR00T N-1.5 adapted for bimanual SO101 arms + data conversion from HF 3.0 to 2.1 to support all types of data collection w/ auto model upload to HF

run ts to finetune jits
python scripts/gr00t_finetune.py \
  --dataset-path ../datasets/matcha-making \
  --num-gpus 2 \
  --output-dir ../models \
  --max-steps 2500 \
  --data-config examples.SO-101.custom_data_config:So101BimanualDataConfig \
  --video-backend torchvision_av \
  --embodiment-tag new_embodiment \
  --gradient-accumulation-steps 8 \
  --dataloader-num-workers 1 \
  --batch-size 1
