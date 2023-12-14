export MODEL_NAME="stable-diffusion-xl-base-1.0"
export DATASET_NAME="Your dataset"

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --enable_xformers_memory_efficient_attention \
  --resolution=512 --center_crop --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=500 \
  --use_8bit_adam \
  --learning_rate=1e-05 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --checkpointing_steps=50 \
  --output_dir="sdxl-pokemon-model" \
  --good_image_column="good_jpg" \
  --bad_image_column="bad_jpg" \
  --caption_column="caption" \
  --max_grad_norm=1 \
  --resume_from_checkpoint="latest"