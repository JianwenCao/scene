import os
import sys
import pickle
import gzip
import torch
import numpy as np
import open_clip
from tqdm import tqdm

# Add HOV-SG to path for GoatDataset
sys.path.append('/home/scene/HOV-SG')
from goat_dataset import GoatDataset
from scene.evaluate import evaluate_submission

# Add OpenFunGraph to path for MapObjectList
sys.path.append('/home/scene/OpenFunGraph')
from openfungraph.slam.slam_classes import MapObjectList

# Paths
GOAT_ROOT = "/root/autodl-tmp/Goat-core"
OUTPUT_ROOT = "/root/autodl-tmp/Goat-core/dataset"
GSA_VARIANT = "ram_withbg_allclasses"
SAVE_SUFFIX = "eval"

def load_results(scene_id):
    path = f"{OUTPUT_ROOT}/{scene_id}/pcd_saves/full_pcd_{GSA_VARIANT}_{SAVE_SUFFIX}_post.pkl.gz"
    if not os.path.exists(path):
        return []
    
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    
    objects = MapObjectList()
    objects.load_serializable(data['objects'])
    return objects

def get_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    return model.to("cuda"), tokenizer

def run_eval(strategy='image_only'):
    print(f"\nEvaluating Strategy: {strategy}")
    clip_model, clip_tokenizer = get_clip_model()
    ds = GoatDataset(GOAT_ROOT)
    
    ALL_SCENES = ['4ok', '5cd', 'nfv', 'tee']
    scene_objects_map = {}
    for scene in ALL_SCENES:
        objs = load_results(scene)
        if objs:
            scene_objects_map[scene] = objs

    predictions = {}
    scene_samples = [s for s in ds.samples if s['scene'] in scene_objects_map and s['task_type'] == 'language']

    templates = [
        "{}",
        "a photo of {}",
        "a centered photo of {}",
        "a close-up photo of {}",
        "a view of {}",
        "this is {}"
    ]

    for sample in tqdm(scene_samples):
        query = sample['query']
        scene = sample['scene']
        
        with torch.no_grad():
            if strategy == 'multi_prompt':
                prompts = [t.format(query) for t in templates]
                tokens = clip_tokenizer(prompts).to("cuda")
                query_features = clip_model.encode_text(tokens)
                query_features /= query_features.norm(dim=-1, keepdim=True)
                query_feature = query_features.mean(dim=0)
                query_feature /= query_feature.norm(dim=-1, keepdim=True)
                query_feature = query_feature.cpu().numpy()
            else:
                text = clip_tokenizer([query]).to("cuda")
                query_feature = clip_model.encode_text(text)
                query_feature /= query_feature.norm(dim=-1, keepdim=True)
                query_feature = query_feature.cpu().numpy()[0]
        
        best_score = -float('inf')
        best_obj = None
        current_map = scene_objects_map[scene]
        
        for obj in current_map:
            if 'clip_ft' not in obj or obj['clip_ft'] is None: continue
            
            feat = obj['clip_ft']
            if isinstance(feat, list):
                feat = torch.stack(feat).cpu().numpy()
            else:
                feat = feat.cpu().numpy()
            if len(feat.shape) == 1: feat = feat.reshape(1, -1)
            
            scores = np.dot(feat, query_feature)
            score = np.max(scores)
            
            if score > best_score:
                best_score = score
                best_obj = obj
        
        if best_obj:
            center = best_obj['pcd'].get_center()
            key = f"{sample['scene']}/{sample['episode']}/{sample['target_name']}"
            predictions[key] = [center]

    evaluate_submission(ds, predictions, filters=['language'], verbose=True)

if __name__ == "__main__":
    run_eval(strategy='image_only')
    run_eval(strategy='multi_prompt')
