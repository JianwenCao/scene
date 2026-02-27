import os
import subprocess
import sys
import pickle
import gzip
import torch
import numpy as np
import open_clip
from tqdm import tqdm

# Add HOV-SG to path for GoatDataset
sys.path.append('/home/scene/HOV-SG')
try:
    from goat_dataset import GoatDataset as GoatDatasetEval
    from scene.evaluate import evaluate_submission
except ImportError:
    # Adjust path if needed
    sys.path.append('/home/scene/HOV-SG/scene')
    from goat_dataset import GoatDataset as GoatDatasetEval
    from evaluate import evaluate_submission

# Paths
PYTHON_EXE = "/root/autodl-tmp/conda/envs/openfungraph/bin/python"
OPENFUNGRAPH_ROOT = "/home/scene/OpenFunGraph"
GOAT_ROOT = "/root/autodl-tmp/Goat-core"
DATASET_CONFIG = f"{OPENFUNGRAPH_ROOT}/openfungraph/dataset/dataconfigs/goat/goat.yaml"
GSA_PATH = "/home/scene/Grounded-Segment-Anything"
OUTPUT_ROOT = "/root/autodl-tmp/Goat-core/dataset"  # Use data disk for output

# Set Env
env = os.environ.copy()
env["GSA_PATH"] = GSA_PATH
python_path = f"{OPENFUNGRAPH_ROOT}:{GSA_PATH}:"
python_path += f"{GSA_PATH}/GroundingDINO:"
python_path += f"{GSA_PATH}/segment_anything:"
python_path += f"{GSA_PATH}/recognize-anything:"
python_path += env.get("PYTHONPATH", "")
env["PYTHONPATH"] = python_path

# Model params
CLASS_SET = "ram"
EXP_SUFFIX = "withbg_allclasses"
GSA_VARIANT = f"{CLASS_SET}_{EXP_SUFFIX}"
SAVE_SUFFIX = "eval"
SIM_THRESHOLD = "1.0" 

def run_detection(scene_id, overwrite=False):
    print(f"Running detection for {scene_id}...")
    
    cmd = [
        PYTHON_EXE,
        f"{OPENFUNGRAPH_ROOT}/openfungraph/scripts/generate_gsa_results.py",
        "--dataset_root", f"{GOAT_ROOT}/dataset",
        "--save_root", OUTPUT_ROOT,  # Pass OUTPUT_ROOT
        "--dataset_config", DATASET_CONFIG,
        "--scene_id", scene_id,
        "--class_set", CLASS_SET,
        "--add_bg_classes",
        "--accumu_classes",
        "--exp_suffix", EXP_SUFFIX,
        "--device", "cuda"
    ]
    if overwrite:
        cmd.append("--overwrite")
    subprocess.check_call(cmd, env=env)

def run_fusion(scene_id):
    print(f"Running fusion for {scene_id}...")
    cmd = [
        PYTHON_EXE,
        f"{OPENFUNGRAPH_ROOT}/openfungraph/slam/cfslam_pipeline_batch.py",
        f"dataset_root={GOAT_ROOT}/dataset",
        f"+save_root={OUTPUT_ROOT}",  # Pass the new save root
        f"dataset_config={DATASET_CONFIG}",
        f"scene_id={scene_id}",
        "stride=1",
        "spatial_sim_type=overlap",
        "mask_conf_threshold=0.3",
        "match_method=sim_sum",
        f"sim_threshold={SIM_THRESHOLD}",
        "dbscan_eps=0.1",
        f"gsa_variant={GSA_VARIANT}",
        "skip_bg=False",
        "max_bbox_area_ratio=0.9",
        "merge_overlap_thresh=0.9",
        f"save_suffix={SAVE_SUFFIX}",
        "merge_visual_sim_thresh=0.75",
        "merge_text_sim_thresh=0.7",
        "obj_min_detections=3",
        "save_pcd=True"
    ]
    subprocess.check_call(cmd, env=env)

def load_results(scene_id):
    path = f"{OUTPUT_ROOT}/{scene_id}/pcd_saves/full_pcd_{GSA_VARIANT}_{SAVE_SUFFIX}_post.pkl.gz"
    print(f"Loading results from {path}")
    if not os.path.exists(path):
        return []
    
    from openfungraph.slam.slam_classes import MapObjectList
    
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
        
    # Reconstruct MapObjectList from serializable dicts
    objects = MapObjectList()
    objects.load_serializable(data['objects'])
    
    return objects

def get_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    return model.to("cuda"), tokenizer, preprocess

def main():
    import argparse
    parser = argparse.ArgumentParser(description="OpenFunGraph Pipeline: Build and Query")
    parser.add_argument("--mode", choices=["build", "query", "all"], default="all", 
                        help="Mode: 'build' to run SLAM/Fusion, 'query' to evaluate queries, 'all' for both.")
    parser.add_argument("--scene", type=str, default="4ok", 
                        help="Scene ID to process, or 'all' for all scenes (default: 4ok)")
    parser.add_argument("--overwrite", action="store_true", 
                        help="Overwrite existing results during the build process.")
    args = parser.parse_args()
    
    ALL_SCENES = ['4ok', '5cd', 'nfv', 'tee']
    target_scenes = ALL_SCENES if args.scene == 'all' else [args.scene]
    
    # --- 1. BUILD STAGE ---
    if args.mode in ["build", "all"]:
        for scene in target_scenes:
            print(f"\n=== Starting BUILD stage for scene: {scene} ===")
            detection_dir = f"{OUTPUT_ROOT}/{scene}/gsa_detections_{GSA_VARIANT}"
            fusion_output = f"{OUTPUT_ROOT}/{scene}/pcd_saves/full_pcd_{GSA_VARIANT}_{SAVE_SUFFIX}_post.pkl.gz"
            
            # Check/Run Detection
            if not args.overwrite and os.path.exists(detection_dir) and len(os.listdir(detection_dir)) > 0:
                print(f"[INFO] Detection directory exists and is not empty. Skipping detection.\n   -> {detection_dir}")
            else:
                 try:
                    run_detection(scene, overwrite=args.overwrite)
                 except Exception as e:
                    print(f"[ERROR] Detection failed for {scene}: {e}")
                    # If detection fails, we skip fusion for this scene
                    continue

            # Check/Create class file (needed for fusion)
            class_file = f"{OUTPUT_ROOT}/{scene}/gsa_classes_{GSA_VARIANT}.json"
            if not os.path.exists(class_file) or args.overwrite:
                print(f"[WARN] Class file missing or overwrite set. Creating default classes.")
                default_classes = ["remote", "cabinet", "chest", "electric outlet", "drawer", "closet", 
                                   "dresser", "radiator", "other item", "door", "floor", "wall", 
                                   "trashcan", "stool", "window", "chair", "bed", "desk", "sofa", 
                                   "table", "lamp", "pillow", "sink", "monitor"]
                import json
                with open(class_file, 'w') as f:
                    json.dump(default_classes, f)

            # Check/Run Fusion
            if not args.overwrite and os.path.exists(fusion_output):
                 print(f"[INFO] Fusion output exists. Skipping fusion.\n   -> {fusion_output}")
            else:
                 try:
                    run_fusion(scene)
                 except Exception as e:
                    print(f"[ERROR] Fusion failed for {scene}: {e}")
                    continue
            print(f"=== BUILD stage completed for {scene} ===")

    # --- 2. QUERY STAGE ---
    if args.mode in ["query", "all"]:
        print(f"\n=== Starting QUERY stage for scenes: {target_scenes} ===")
        
        print("Initializing CLIP for query matching...")
        try:
            clip_model, clip_tokenizer, clip_preprocess = get_clip_model()
        except Exception as e:
            print(f"[ERROR] Failed to load CLIP: {e}")
            return

        print("Loading dataset and Ground Truth...")
        ds = GoatDatasetEval(GOAT_ROOT)
        
        # Load all available maps
        scene_objects_map = {}
        for scene in target_scenes:
            result_path = f"{OUTPUT_ROOT}/{scene}/pcd_saves/full_pcd_{GSA_VARIANT}_{SAVE_SUFFIX}_post.pkl.gz"
            if not os.path.exists(result_path):
                print(f"[WARN] Map file not found for {scene}: {result_path}. Skipping this scene.")
                continue
            
            print(f"Loading 3D Map for {scene}...")
            objs = load_results(scene)
            if objs:
                scene_objects_map[scene] = objs
                print(f"Loaded {len(objs)} objects for {scene}.")
            else:
                print(f"[ERROR] Failed to load objects for {scene}.")

        if not scene_objects_map:
            print("No maps loaded. Exiting.")
            return

        predictions = {}
        
        # Filter samples for target scenes
        scene_samples = [s for s in ds.samples if s['scene'] in scene_objects_map]
        print(f"Evaluating {len(scene_samples)} queries for loaded scenes...")

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
            task_type = sample['task_type']
            scene = sample['scene']
            
            # Prepare query feature
            query_feature = None
            if task_type == 'language' or task_type == 'object':
                 with torch.no_grad():
                     prompts = [t.format(query) for t in templates]
                     text = clip_tokenizer(prompts).to("cuda")
                     query_features = clip_model.encode_text(text)
                     query_features /= query_features.norm(dim=-1, keepdim=True)
                     query_feature = query_features.mean(dim=0)
                     query_feature /= query_feature.norm(dim=-1, keepdim=True)
                     query_feature = query_feature.cpu().numpy()
            elif task_type == 'image':
                 # query is a path to an image
                 if os.path.exists(query):
                     from PIL import Image
                     img_query = Image.open(query).convert("RGB")
                     img_input = clip_preprocess(img_query).unsqueeze(0).to("cuda")
                     with torch.no_grad():
                         query_feature = clip_model.encode_image(img_input)
                         query_feature /= query_feature.norm(dim=-1, keepdim=True)
                         query_feature = query_feature.cpu().numpy().squeeze()
                 else:
                     print(f"[WARN] Image query path does not exist: {query}")
            
            if query_feature is None:
                continue
            
            # Find best match in the specific scene map
            best_score = -float('inf')
            best_obj = None
            
            current_map = scene_objects_map[scene]
            
            for obj in current_map:
                if 'clip_ft' not in obj or obj['clip_ft'] is None: 
                    continue
                
                feats = obj['clip_ft']
                # Check for single tensor vs list
                if isinstance(feats, list):
                    if len(feats) == 0: continue
                    # list of tensors
                    try:
                        feats_stack = torch.stack(feats)
                        feats_np = feats_stack.cpu().numpy()
                    except:
                        continue
                elif isinstance(feats, torch.Tensor):
                    feats_np = feats.cpu().numpy()
                elif isinstance(feats, np.ndarray):
                    feats_np = feats
                else:
                    continue
                
                # Check dimensions
                if len(feats_np.shape) == 1:
                    feats_np = feats_np.reshape(1, -1)
                
                if feats_np.shape[1] != query_feature.shape[0]:
                    continue

                scores = np.dot(feats_np, query_feature) # (N,)
                max_obj_score = np.max(scores)
                
                if max_obj_score > best_score:
                    best_score = max_obj_score
                    best_obj = obj
            
            if best_obj:
                pcd = best_obj['pcd']
                center = pcd.get_center()
                
                key = f"{sample['scene']}/{sample['episode']}/{sample['target_name']}"
                predictions[key] = [center]
                
                # Debug print for first few
                if len(predictions) <= 3:
                    gt_goals = sample['goals']
                    print(f"\n[DEBUG] Scene: {scene} | Query: {query}")
                    print(f"        Best Obj ID: {best_score:.4f}")
                    print(f"        Pred Center: {center}")
                    print(f"        GT Goals:    {[g.tolist() for g in gt_goals]}")

        # Submit / Evaluate
        print(f"Running evaluation metrics for scenes: {list(scene_objects_map.keys())}...")
        
        # 1. Global stats with filter
        print("\n--- GLOBAL LANGUAGE PERFORMANCE ---")
        evaluate_submission(ds, predictions, filters=['language'], verbose=True)
        
        # 2. Scene-wise breakdown for language only
        print("\n--- SCENE-WISE LANGUAGE PERFORMANCE ---")
        print(f"{'Scene ID':<15} | {'Language Samples':<18} | {'Success Rate':<12}")
        print("-" * 50)
        
        for scene in target_scenes:
            scene_lang_samples = [s for s in ds.samples if s['scene'] == scene and s['task_type'] == 'language']
            if not scene_lang_samples:
                continue
                
            # Filter predictions for this scene and language task
            scene_lang_preds = {}
            for s in scene_lang_samples:
                key = f"{s['scene']}/{s['episode']}/{s['target_name']}"
                if key in predictions:
                    scene_lang_preds[key] = predictions[key]
            
            # Use evaluate_submission to get stats for this subset
            # Note: We pass verbose=False to get the raw numbers
            stats, _ = evaluate_submission(scene_lang_samples, scene_lang_preds, filters=['language'], verbose=False)
            
            lang_stats = stats['language']
            success_rate = (lang_stats['success'] / lang_stats['total'] * 100) if lang_stats['total'] > 0 else 0
            
            print(f"{scene:<15} | {lang_stats['total']:<18} | {success_rate:.2f}%")
        print("-" * 50)

if __name__ == "__main__":
    main()
