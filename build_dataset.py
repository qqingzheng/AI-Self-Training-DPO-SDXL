import scorer
import os, json
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor
from scorer.aesthetic_scorer import MLP, AestheticScorer
from scorer.clip_scorer import CLIPScorer
# from scorer.vqa_scorer import VQAScorer
import torch
import numpy as np
from datasets import Dataset
import datasets
import multiprocessing
import time

def score_images(device, scorer_list, args, queue, result_queue):
    try:
        print("loading model.")
        model = CLIPModel.from_pretrained(args.clip_path).to(device)
        model.requires_grad_(False)
        model.eval()
        processor = CLIPProcessor.from_pretrained(args.clip_path)
        aes_model = MLP(768).to(device)
        aes_model.load_state_dict(torch.load(args.aes_path))
        aes_model.requires_grad_(False)
        aes_model.eval()
        image_processor = CLIPImageProcessor.from_pretrained(args.clip_path)

        while not queue.empty():
            idx, prompt = queue.get()   

            images = []
            report = {"scorer": {}, "prompt": prompt}
            scores = np.array([0] * args.num_images, dtype=np.float16)
            for img_id in range(args.num_images):
                images.append(Image.open(os.path.join(args.image_dir, f"{idx}_{img_id}.png")))
            for scorer_cls, weight in scorer_list:
                if scorer_cls is AestheticScorer:
                    scorer = scorer_cls(prompt, images, model, image_processor, aes_model)
                elif scorer_cls is CLIPScorer:
                    scorer = scorer_cls(prompt[0:77], images, model, processor)
                else:
                    raise NotImplementedError("scorer class error.")
                tmp_scores = scorer.get_score()
                tmp_scores = np.array(tmp_scores, dtype=np.float16)
                scores += weight * tmp_scores
            report["win"] = int(np.argmax(scores))
            report["lose"] = int(np.argmax(-scores))
            result_queue.put((idx, report))
    except Exception as e:
        print(f"process crack: {str(e)}")
        
def main(args):
    # Define Scorer list
    # scorer_list = [(CLIPScorer, 0.4), (AestheticScorer, 0.2), (VQAScorer, 0.4)]
    scorer_list = [(CLIPScorer, 0.6), (AestheticScorer, 0.4)]
    
    
    # Load prompts
    with open(args.prompt_path, "r") as file:
        prompts = json.load(file)
    
    # Create queues
    gen_queue = multiprocessing.Queue(maxsize=len(prompts))
    score_result_queue = multiprocessing.Queue(maxsize=len(prompts))

    # Populate generation queue
    for idx, prompt in enumerate(prompts):
        gen_queue.put((idx, prompt))

    # Start generation processes
    score_processes = []
    for device in args.device.split(","):
        p = multiprocessing.Process(target=score_images, args=(device, scorer_list, args, gen_queue, score_result_queue))
        score_processes.append(p)
        p.start()

    # Wait for scoring processes to finish
    results = {}
    
    while len(results) != len(prompts):
        print(f"\r {len(results)}/{len(prompts)}")
        # Collect results
        while not score_result_queue.empty():
            idx, result = score_result_queue.get()
            results[idx] = result
        time.sleep(1)
        
    for p in score_processes:
        p.terminate()
        
    # Gen huggingface dataset
    print(f"dataset size: {len(results)}")
    def gen():
        for key, value in results.items():
            yield {
                "caption": value["prompt"],
                "good_jpg": os.path.join(args.image_dir, f"{key}_{value['win']}.png"),
                "bad_jpg": os.path.join(args.image_dir, f"{key}_{value['lose']}.png"),
            }

    ds = Dataset.from_generator(gen)
    ds = ds.cast_column("good_jpg", datasets.Image())
    ds = ds.cast_column("bad_jpg", datasets.Image())
    ds.save_to_disk(args.output_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_path", type=str, default="models/clip-vit-large-patch14"
    )
    parser.add_argument(
        "--aes_path",
        type=str,
        default="models/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth",
    )
    parser.add_argument("--image_dir", type=str, default="dataset/images/")
    parser.add_argument("--output_dir", type=str, default="dataset/ai_feedback/")
    parser.add_argument(
        "--prompt_path", type=str, default="dataset/prompts_squeeze.json"
    )
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7,cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7,cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7,cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7,cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7")
    args = parser.parse_args()

    main(args)
