import os
import sys
import pickle
import gzip
import torch
import numpy as np
import open_clip
from tqdm import tqdm
from pathlib import Path

# Add HOV-SG to path for GoatDataset
sys.path.append('/home/scene/HOV-SG')
from goat_dataset import GoatDataset
from scene.evaluate import evaluate_submission

# Add OpenFunGraph to path
sys.path.append('/home/scene/OpenFunGraph')

# Paths
GOAT_ROOT = "/root/autodl-tmp/Goat-core"
OUTPUT_ROOT = "/root/autodl-tmp/Goat-core/dataset"
GSA_VARIANT = "ram_withbg_allclasses"

def get_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    return model.to("cuda"), tokenizer

def load_all_detections(scene_id):
    det_dir = Path(OUTPUT_ROOT) / scene_id / f"gsa_detections_{GSA_VARIANT}"
    det_files = sorted(list(det_dir.glob("*.pkl.gz")))
    
    all_feats = []
    all_metadata = [] # Store (file_idx, det_idx)
    
    print(f"Loading {len(det_files)} detection files for {scene_id}...")
    for f_idx, f_path in enumerate(tqdm(det_files)):
        with gzip.open(f_path, 'rb') as f:
            data = pickle.load(f)
        
        feats = data['image_feats'] # (N, D)
        if len(feats) == 0: continue
        
        all_feats.append(feats)
        for d_idx in range(len(feats)):
            all_metadata.append((f_path, d_idx))
            
    all_feats = np.concatenate(all_feats, axis=0)
    return all_feats, all_metadata

def get_3d_pos(metadata_item):
    f_path, d_idx = metadata_item
    with gzip.open(f_path, 'rb') as f:
        data = pickle.load(f)
    
    # We need pose and intrinsics to get 3D pos.
    # This is slow if we do it for every detection.
    # But we only do it for Top-K.
    return None # Placeholder

def run_detection_retrieval():
    clip_model, clip_tokenizer = get_clip_model()
    ds = GoatDataset(GOAT_ROOT)
    
    ALL_SCENES = ['4ok', '5cd', 'nfv', 'tee']
    
    # Cache detections per scene
    scene_data = {}
    for scene in ALL_SCENES:
        feats, meta = load_all_detections(scene)
        scene_data[scene] = (feats, meta)

    predictions = {}
    scene_samples = [s for s in ds.samples if s['scene'] in ALL_SCENES and s['task_type'] == 'language']

    for sample in tqdm(scene_samples):
        query = sample['query']
        scene = sample['scene']
        
        with torch.no_grad():
            text = clip_tokenizer([query]).to("cuda")
            query_feature = clip_model.encode_text(text)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.cpu().numpy()[0]
        
        feats, meta = scene_data[scene]
        scores = np.dot(feats, query_feature)
        
        # Take Top-1 for now as a baseline for this strategy
        best_idx = np.argmax(scores)
        best_meta = meta[best_idx]
        
        # Now we need to get the 3D position of this detection
        # We need the pose and depth...
        # Instead of doing the full projection, we can use the existing maps
        # to find which object this detection belongs to!
        # But we don't have that mapping.
        
        # Wait! If I just take the 3D map from OpenFunGraph
        # and match against Top-K detections? 
        # No, that's what I already did.
        
        # Let's try to find the 3D center of the Top-1 detection.
        # I need to load the dataset to get pose and intrinsics.
        # Or I can just match the 2D detection back to the 3D objects.
        
    # This strategy is becoming complex. Let's try something simpler first.
