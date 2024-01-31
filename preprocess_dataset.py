import functools
import os
import random
import numpy as np
import torch
import torch.utils.checkpoint
from datasets import load_dataset, load_from_disk
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import argparse
from diffusers import (
    AutoencoderKL,
)

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def preprocess_train(examples, vae, args):
    def preprocess_images(image_column):
        images = [image.convert("RGB") for image in examples[image_column]]
        original_sizes = []
        crop_top_lefts = []
        processed_images = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(
                    image, (args.resolution, args.resolution)
                )
                image = crop(image, y1, x1, h, w)
            if args.random_flip and random.random() < 0.5:
                x1 = image.width - x1
                image = train_flip(image)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = train_transforms(image).squeeze(0)
            processed_images.append(image)
        with torch.no_grad():
            model_input = vae.encode(torch.stack(processed_images).to(vae.device)).latent_dist.sample().squeeze()
        examples[image_column + "_original_sizes"] = original_sizes
        examples[image_column + "_crop_top_lefts"] = crop_top_lefts
        examples[image_column + "_model_input"] = model_input
    for image_column in [args.good_image_column, args.bad_image_column]:
        preprocess_images(image_column)
    return examples

def encode_prompt(
    batch,
    text_encoders,
    tokenizers,
    args,
    is_train=True,
):
    prompt_embeds_list = []
    prompt_batch = batch[args.caption_column]
    captions = []
    for caption in prompt_batch:
        if random.random() < args.proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            if args.sd_version == "xl":
                prompt_embeds = text_encoder(
                    text_input_ids.to(text_encoder.device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                
                prompt_embeds = prompt_embeds.hidden_states[-2]
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)
            else:
                prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device),
                    return_dict=False)[0]
                prompt_embeds_list.append(prompt_embeds)
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    if args.sd_version == "xl":
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return {
            "prompt_embeds": prompt_embeds.cpu(),
            "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
        }
    else:
        return {
            "prompt_embeds": prompt_embeds.cpu(),
        }
    
def main(args):
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    if args.sd_version == "xl":
        tokenizer_two = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=args.revision,
            use_fast=False,
        )
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    if args.sd_version == "xl":
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
        )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    if args.sd_version == "xl":
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            revision=args.revision,
            variant=args.variant,
        )
    
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    if args.sd_version == "xl":
        text_encoder_two.requires_grad_(False)
    weight_dtype = torch.float16
    vae.to(args.device, dtype=torch.float32)
    text_encoder_one.to(args.device, dtype=weight_dtype)
    if args.sd_version == "xl":
        text_encoder_two.to(args.device, dtype=weight_dtype)

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_from_disk(
            args.dataset_name
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
        )
    column_names = dataset.column_names

    if args.good_image_column is None:
        raise ValueError(f"--good_image_column must not be None.")
    else:
        good_image_column = args.good_image_column
        if good_image_column not in column_names:
            raise ValueError(
                f"--good_image_column' value '{args.good_image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.bad_image_column is None:
        raise ValueError(f"--bad_image_column must not be None.")
    else:
        bad_image_column = args.bad_image_column
        if bad_image_column not in column_names:
            raise ValueError(
                f"--bad_image_column' value '{args.bad_image_column}' needs to be one of: {', '.join(column_names)}"
            )
    caption_column = args.caption_column
    if caption_column not in column_names:
        raise ValueError(
            f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
        )
    if args.sd_version == "xl":
        text_encoders = [text_encoder_one, text_encoder_two]
        tokenizers = [tokenizer_one, tokenizer_two]
    else:
        text_encoders = [text_encoder_one]
        tokenizers = [tokenizer_one]

    preprocess_train_fn = functools.partial(
        preprocess_train,
        vae=vae,
        args=args
    )
    if args.max_train_samples is not None:
        dataset = (
            dataset
            .shuffle(seed=args.seed)
            .select(range(args.max_train_samples))
        )
    train_dataset = dataset.map(preprocess_train_fn, batched=True, batch_size=args.batch_size, remove_columns=['good_jpg', 'bad_jpg'])

    compute_embeddings_fn = functools.partial(
        encode_prompt,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        args=args
    )
    train_dataset = train_dataset.map(
        compute_embeddings_fn, batched=True, batch_size=args.batch_size
    )

    train_dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    
    # parser.add_argument("--pretrained_model_name_or_path", default="models/stable-diffusion-xl-base-1.0", type=str)
    # parser.add_argument("--pretrained_vae_model_name_or_path", default=None, type=str)
    # parser.add_argument("--dataset_name", default="dataset/ccd_ai_feedback", type=str)
    # parser.add_argument("--resolution", default=512, type=int)
    # parser.add_argument("--center_crop", action='store_true')
    # parser.add_argument("--random_flip", action='store_true')
    # parser.add_argument("--caption_column", default="caption", type=str)
    # parser.add_argument("--good_image_column", default="good_jpg", type=str)
    # parser.add_argument("--bad_image_column", default="bad_jpg", type=str)
    # parser.add_argument("--output_dir", default="dataset/ccd_ai_feedbacks", type=str)
    # parser.add_argument("--proportion_empty_prompts", default=0, type=int)
    # parser.add_argument("--train_data_dir", default=None, type=str)
    # parser.add_argument("--dataset_config_name", default=None, type=str)
    # parser.add_argument("--variant", default=None, type=str)
    # parser.add_argument("--revision", default=None, type=str)
    # parser.add_argument("--device", default="cuda:5", type=str)
    # parser.add_argument("--max_train_samples", default=None, type=int)
    # parser.add_argument("--seed", default=1234, type=int)
    # parser.add_argument("--batch_size", default=32, type=int)
    # parser.add_argument("--sd_version", default="1.5", type=str)

    parser.add_argument("--pretrained_model_name_or_path", default="models/stable-diffusion-xl-base-1.0", type=str)
    parser.add_argument("--pretrained_vae_model_name_or_path", default=None, type=str)
    parser.add_argument("--dataset_name", default="dataset/ccd_ai_feedback", type=str)
    parser.add_argument("--resolution", default=1024, type=int)
    parser.add_argument("--center_crop", action='store_true')
    parser.add_argument("--random_flip", action='store_true')
    parser.add_argument("--caption_column", default="caption", type=str)
    parser.add_argument("--good_image_column", default="good_jpg", type=str)
    parser.add_argument("--bad_image_column", default="bad_jpg", type=str)
    parser.add_argument("--output_dir", default="dataset/ccd_ai_feedbacks", type=str)
    parser.add_argument("--proportion_empty_prompts", default=0, type=int)
    parser.add_argument("--train_data_dir", default=None, type=str)
    parser.add_argument("--dataset_config_name", default=None, type=str)
    parser.add_argument("--variant", default=None, type=str)
    parser.add_argument("--revision", default=None, type=str)
    parser.add_argument("--device", default="cuda:1", type=str)
    parser.add_argument("--max_train_samples", default=None, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--sd_version", default="xl", type=str)


    args = parser.parse_args()
    
    train_resize = transforms.Resize(
        args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
    )
    train_crop = (
        transforms.CenterCrop(args.resolution)
        if args.center_crop
        else transforms.RandomCrop(args.resolution)
    )
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    
    main(args)