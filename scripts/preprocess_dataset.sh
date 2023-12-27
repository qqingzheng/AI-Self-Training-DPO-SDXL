python preprocess_dataset.py \
    --pretrained_model_name_or_path "stable-diffusion-xl-base-1.0"\
    --dataset_name "dataset/ai_feedback"\
    --center_crop \
    --resolution 1024 \
    --output_dir "dataset/preprocessed_ai_feedback"