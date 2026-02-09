import torch
from goat_dataset import GoatDataset
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
import os
import sys
import time
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json
from PIL import Image
from scipy.spatial.transform import Rotation
from evaluate import evaluate_submission

# Add local sam3 to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sam3'))
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results
from qwen3_vl_embedding import Qwen3VLEmbedder
from qwen3_vl_reranker import Qwen3VLReranker

def get_scene_features(scene_dir, model, processor, device, model_name):
    feature_dir = os.path.join(scene_dir, "features", model_name)
    os.makedirs(feature_dir, exist_ok=True)
    
    features_path = os.path.join(feature_dir, "features.npy")
    names_path = os.path.join(feature_dir, "image_names.json")
    
    image_root = os.path.join(scene_dir, "images")
    images_list = sorted(os.listdir(image_root))
    
    if os.path.exists(features_path) and os.path.exists(names_path):
        try:
            with open(names_path, 'r') as f:
                cached_names = json.load(f)
            
            if len(cached_names) == len(images_list) and cached_names == images_list:
                print(f"\n{'='*20} CACHE HIT {'='*20}")
                print(f"  [CACHE] Loading features from: {feature_dir}")
                print(f"{'='*51}\n")
                features = np.load(features_path)
                return torch.from_numpy(features).to(device), cached_names
        except Exception as e:
            print(f"  [ERROR] Cache invalid: {e}. Proceeding to extraction...")

    print(f"\n{'*'*20} FEATURE EXTRACTION {'*'*20}")
    print(f"  [EXTRACT] Scene: {scene_dir}")
    print(f"  [EXTRACT] Model: {model_name}")
    print(f"  [EXTRACT] Images: {len(images_list)}")
    print(f"{'*'*60}")
    features_list = []
    batch_size = 16
    
    for i in range(0, len(images_list), batch_size):
        batch_names = images_list[i:i+batch_size]
        batch_images = []
        for name in batch_names:
            img_path = os.path.join(image_root, name)
            batch_images.append(Image.open(img_path).convert("RGB"))
            
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_embeddings = model.get_image_features(**inputs)
            if not isinstance(image_embeddings, torch.Tensor):
                image_embeddings = image_embeddings.pooler_output
            image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
            features_list.append(image_embeddings.cpu())
            
    all_features = torch.cat(features_list, dim=0)
    
    np.save(features_path, all_features.numpy())
    with open(names_path, 'w') as f:
        json.dump(images_list, f)
        
    return all_features.to(device), images_list

def load_intrinsics(scene_dir):
    # Try to find cameras.txt
    cam_file = os.path.join(scene_dir, "sparse", "0", "cameras.txt")
    if not os.path.exists(cam_file):
        return None
        
    with open(cam_file, 'r') as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.split()
            # 1 PINHOLE 640 480 fx fy cx cy
            model_type = parts[1]
            if model_type == "PINHOLE":
                return {
                    'fx': float(parts[4]), 'fy': float(parts[5]),
                    'cx': float(parts[6]), 'cy': float(parts[7])
                }
            elif model_type == "SIMPLE_PINHOLE":
                f_val = float(parts[4])
                return {
                    'fx': f_val, 'fy': f_val,
                    'cx': float(parts[5]), 'cy': float(parts[6])
                }
    return None

def load_poses(scene_dir):
    pos_file = os.path.join(scene_dir, "local_pos.txt")
    poses = {} 
    if not os.path.exists(pos_file):
        return poses
        
    with open(pos_file, 'r') as f:
        for line in f:
            parts = [float(x) for x in line.split()]
            # Format: frame_idx qw qx qy qz tx ty tz
            idx = int(parts[0])
            qw, qx, qy, qz = parts[1], parts[2], parts[3], parts[4]
            tx, ty, tz = parts[5], parts[6], parts[7]
            
            # Scipy uses (x, y, z, w)
            rot = Rotation.from_quat([qx, qy, qz, qw])
            R_wc = rot.as_matrix()
            t_wc = np.array([tx, ty, tz])
            
            # Assume local_pos.txt stores Camera-to-World (T_wc)
            # So no inversion needed.
            
            poses[idx] = (R_wc, t_wc)
    return poses

def project_mask_to_3d(mask, depth_map, intrinsics, pose, debug_vis_path=None, gt_goals=None):
    if depth_map is None or intrinsics is None or pose is None:
        return None

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
        
    # Sample points
    z_vals = depth_map[ys, xs]
    valid = (z_vals > 0.1) & (z_vals < 20.0) # Clamp depth
    
    if not np.any(valid):
        return None
        
    xs = xs[valid]
    ys = ys[valid]
    zs = z_vals[valid]
    
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    # Backproject
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    z_cam = zs
    
    # Coordinate Conversion: OpenCV (Y down, Z forward) -> ARKit/OpenGL (Y up, -Z forward)
    # This fixes the "fly to sky" issue where floor points (+Y) become ceiling points.
    P_cam = np.stack([x_cam, -y_cam, -z_cam], axis=1) # (N, 3)
    
    R, t = pose
    P_world = (R @ P_cam.T).T + t

    if debug_vis_path:
        try:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 1. Plot Point Cloud (Downsample)
            if len(P_world) > 2000:
                idx = np.random.choice(len(P_world), 2000, replace=False)
                P_vis = P_world[idx]
            else:
                P_vis = P_world
                
            ax.scatter(P_vis[:,0], P_vis[:,1], P_vis[:,2], s=1, c='blue', alpha=0.3, label='Predicted Points')
            
            mean_p = np.mean(P_world, axis=0)
            ax.scatter(mean_p[0], mean_p[1], mean_p[2], c='blue', marker='x', s=100, label='Pred Centroid', depthshade=False)

            # 2. Plot Camera Frustum
            H, W = depth_map.shape
            Z_frust = 1.0 # Scale of frustum visual
            
            # 4 Corners in Image Plane (OpenCV)
            corners_uv = np.array([[0,0], [W, 0], [W, H], [0, H]])
            corners_cam = np.zeros((4, 3))
            
            # OpenCV Unprojection
            corners_cam[:, 0] = (corners_uv[:, 0] - cx) * Z_frust / fx
            corners_cam[:, 1] = (corners_uv[:, 1] - cy) * Z_frust / fy
            corners_cam[:, 2] = Z_frust
            
            # Flip to ARKit/Pose Frame
            corners_cam[:, 1] *= -1
            corners_cam[:, 2] *= -1
            
            # Transform to World
            corners_world = (R @ corners_cam.T).T + t
            
            # Draw lines from Camera Center (t) to Corners
            for i in range(4):
                ax.plot([t[0], corners_world[i,0]], 
                        [t[1], corners_world[i,1]], 
                        [t[2], corners_world[i,2]], 'k-', linewidth=1)
            
            # Draw lines connecting corners (Frame)
            # 0-1, 1-2, 2-3, 3-0
            for i in range(4):
                j = (i + 1) % 4
                ax.plot([corners_world[i,0], corners_world[j,0]],
                        [corners_world[i,1], corners_world[j,1]],
                        [corners_world[i,2], corners_world[j,2]], 'k-', linewidth=1)
                        
            ax.scatter(t[0], t[1], t[2], c='red', marker='^', s=50, label='Camera Pos', depthshade=False)
            
            # 3. Plot GT Goals
            if gt_goals is not None:
                for idx, goal in enumerate(gt_goals):
                    # goal is tensor or list [x, y, z]
                    if hasattr(goal, 'tolist'): goal = goal.tolist()
                    label = 'GT Goal' if idx == 0 else None
                    ax.scatter(goal[0], goal[1], goal[2], c='lime', marker='*', s=150, edgecolors='black', label=label, depthshade=False)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Equal aspect ratio hack for 3D
            # Limits
            all_pts = np.vstack([P_vis, t.reshape(1,3)])
            if gt_goals:
                # Convert list of tensors/lists to numpy array
                gt_arr = np.array([g.tolist() if hasattr(g, 'tolist') else g for g in gt_goals])
                all_pts = np.vstack([all_pts, gt_arr])
            
            X, Y, Z = all_pts[:,0], all_pts[:,1], all_pts[:,2]
            max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
            mid_x = (X.max()+X.min()) * 0.5
            mid_y = (Y.max()+Y.min()) * 0.5
            mid_z = (Z.max()+Z.min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            ax.legend()
            plt.title(f"3D Vis: {os.path.basename(debug_vis_path)}")
            plt.tight_layout()
            plt.savefig(debug_vis_path)
            plt.close()
        except Exception as e:
            print(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()
    
    return np.mean(P_world, axis=0).tolist()

def visualize_projection(image_path, mask, center_3d, pose, intrinsics, save_path, title_prefix=""):
    img = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    
    # Draw Mask
    masked_image = np.ma.masked_where(mask == 0, mask)
    plt.imshow(masked_image, alpha=0.4, cmap='jet', interpolation='none')

    # Reproject 3D Center
    R, t = pose
    P_world = np.array(center_3d)
    
    # World -> Cam
    P_cam = R.T @ (P_world - t)
    x, y, z = P_cam
    
    if z > 0:
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']
        
        u = (x * fx / z) + cx
        v = (y * fy / z) + cy
        
        plt.plot(u, v, 'ro', markersize=10, markeredgecolor='white')
        plt.text(u + 10, v, f"Z={z:.2f}m", color='white', backgroundcolor='red', fontsize=12)

    plt.axis('off')
    plt.title(f"{title_prefix}\n{os.path.basename(image_path)}")
    plt.savefig(save_path)
    plt.close()

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", default="siglip-giant", help="Select model type: clip-base, siglip, clip-large, qwen3-vl-2b")
parser.add_argument("--reranker", default="qwen3-vl-4b-instruct", help="Optional reranker model: qwen3-vl-reranker-2b, qwen3-vl-reranker-8b, qwen3-vl-4b-instruct, qwen3-vl-thinking-8b or HF path")
parser.add_argument("--rerank_strategy", default="all_at_once", choices=["one_by_one", "all_at_once"], help="Reranking strategy for thinking model")
parser.add_argument("--use_diversity_filter", action="store_true", help="Filter out too-similar images during retrieval")
parser.add_argument("--visualize", action="store_true", help="Enable visualization")
args = parser.parse_args()

# Available models
model_dict = {
    "clip-base": "openai/clip-vit-base-patch16",
    "clip-large": "openai/clip-vit-large-patch14",
    "siglip-so400m": "google/siglip2-so400m-patch16-512",
    "siglip-giant": "google/siglip2-giant-opt-patch16-384",
    "qwen3-vl-2b": "Qwen/Qwen3-VL-Embedding-2B",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-Embedding-8B",
}

if args.visualize:
    os.makedirs('./vis', exist_ok=True)

if args.model_type not in model_dict:
    raise ValueError(f"Invalid model type: {args.model_type}. Choose from {list(model_dict.keys())}")

ckpt = model_dict[args.model_type]

if args.model_type in ["qwen3-vl-2b", "qwen3-vl-8b"]:
    print(f"Loading Qwen3-VL model: {ckpt}")
    qwen_model = Qwen3VLEmbedder(model_name_or_path=ckpt)
    
    # Wrapper for Processor
    class QwenProcessorWrapper:
        def __call__(self, images=None, text=None, return_tensors="pt", **kwargs):
            class Inputs(dict):
                def to(self, device): return self
            
            if images is not None:
                return Inputs({'images': images})
            if text is not None:
                return Inputs({'text': text})
            return Inputs({})

    # Wrapper for Model
    class QwenModelWrapper:
        def __init__(self, model):
            self.model = model
            self.device = model.model.device
            
        def get_image_features(self, images=None, **kwargs):
            # images is list of PIL images
            inputs = [{'image': img} for img in images]
            return self.model.process(inputs)

        def get_text_features(self, text=None, **kwargs):
            # text is list of strings
            inputs = [{'text': t} for t in text]
            return self.model.process(inputs)
            
    model = QwenModelWrapper(qwen_model)
    processor = QwenProcessorWrapper()
else:
    model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(ckpt, use_fast=True)

# Load Reranker if specified
reranker_model = None
if args.reranker:
    reranker_dict = {
        "qwen3-vl-reranker-2b": "Qwen/Qwen3-VL-Reranker-2B",
        "qwen3-vl-reranker-8b": "Qwen/Qwen3-VL-Reranker-8B",
        "qwen3-vl-4b-instruct": "Qwen/Qwen3-VL-4B-Instruct",
        "qwen3-vl-thinking-4b": "Qwen/Qwen3-VL-4B-Thinking",
        "qwen3-vl-thinking-8b": "Qwen/Qwen3-VL-8B-Thinking-FP8",
    }
    reranker_ckpt = reranker_dict.get(args.reranker, args.reranker)
    print(f"Loading Reranker model: {reranker_ckpt}")
    # Qwen3VLReranker will automatically handle dtype and attention implementation
    reranker_model = Qwen3VLReranker(model_name_or_path=reranker_ckpt, device_map="auto")

sam3_ckpt = "facebook/sam3"
print(f"Loading SAM3 model (local)...")
# Using local sam3 builder
sam3_model = build_sam3_image_model(
    device=str(model.device).split(':')[0] if 'cuda' in str(model.device) else 'cpu',
    load_from_HF=True
)
sam3_model.to(model.device)
sam3_processor = Sam3Processor(sam3_model, device=model.device)

dataset = GoatDataset('/home/jianwen/data/Goat-core')
data_root = "/home/jianwen/data/Goat-core/dataset"

all_predictions = {} 
perf_stats = {}
attempted_keys = []

# Cache for scene data to avoid reloading
scene_cache = {}

print(f"Processing {len(dataset)} samples...")

MAX_LANGUAGE_QUERIES = -1
language_count = 0

for i, sample in enumerate(dataset):
    if sample['task_type'] != 'language': continue
    
    if MAX_LANGUAGE_QUERIES > 0 and language_count >= MAX_LANGUAGE_QUERIES:
        break
        
    language_count += 1
    start_time = time.time()
    
    scene = sample['scene']
    episode = sample['episode']
    target_name = sample['target_name']
    query = sample['query']
    gt_goals = sample.get('goals', []) # Get GT goals
    
    key = f"{scene}/{episode}/{target_name}"
    attempted_keys.append(key)
    print(f"\n[{i+1}/{len(dataset)}] Sample: {key}")
    print(f"  Query: {query}")
    if gt_goals:
        print(f"  GT Goals: {len(gt_goals)} locations")

    # Load Scene Data if needed
    if scene not in scene_cache:
        scene_dir = os.path.join(data_root, scene)
        intrinsics = load_intrinsics(scene_dir)
        poses = load_poses(scene_dir)
        
        model_name_safe = ckpt.replace('/', '_')
        features, image_names = get_scene_features(scene_dir, model, processor, model.device, model_name_safe)
        
        scene_cache[scene] = (intrinsics, poses, features, image_names)
    
    intrinsics, poses, scene_features, scene_image_names = scene_cache[scene]
    scene_dir = os.path.join(data_root, scene)
    image_root = os.path.join(scene_dir, "images")
    depth_root = os.path.join(scene_dir, "depth")

    # 1. Text Embeddings
    with torch.no_grad():
        text_inputs = processor(text=[query.lower()], return_tensors="pt", padding="max_length", max_length=64, truncation=True).to(model.device)
        text_features = model.get_text_features(**text_inputs)
        if not isinstance(text_features, torch.Tensor):
            text_features = text_features.pooler_output
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    # 2. Retrieve Top Images
    with torch.no_grad():
        sims = (text_features @ scene_features.T).squeeze(0)
    
    # Base candidates retrieval
    if args.use_diversity_filter:
        # Filter out too-similar images (Diversity Filtering)
        sorted_values, sorted_indices = torch.sort(sims, descending=True)
        sorted_indices = sorted_indices.cpu().numpy()
        
        candidate_indices = []
        similarity_threshold = 0.9
        scene_features_cpu = scene_features.cpu().numpy()

        # If we have a reranker, we want more candidates to rerank
        max_candidates = 10 if reranker_model else 5

        for idx in sorted_indices:
            if len(candidate_indices) >= max_candidates:
                break
                
            is_unique = True
            current_feat = scene_features_cpu[idx]
            for prev_idx in candidate_indices:
                prev_feat = scene_features_cpu[prev_idx]
                if np.dot(current_feat, prev_feat) > similarity_threshold:
                    is_unique = False
                    break
            if is_unique:
                candidate_indices.append(idx)
        print(f"  Retrieved {len(candidate_indices)} unique candidates (Diversity Filtered)")
    else:
        k_base = 10 if reranker_model else 5
        k_base = min(k_base, len(scene_image_names))
        values, indices = torch.topk(sims, k_base)
        candidate_indices = indices.cpu().numpy().tolist()
        print(f"  Retrieved {len(candidate_indices)} candidates (Simple Top-K)")

    # Optional Reranking
    if reranker_model and len(candidate_indices) > 0:
        candidate_images_paths = [os.path.join(image_root, scene_image_names[idx]) for idx in candidate_indices]
        print(f"  Reranking {len(candidate_indices)} candidates using {args.rerank_strategy}...")
        
        selected_rel_indices = reranker_model.rerank_batch(
            query=query, 
            document_images=candidate_images_paths, 
            top_k=5, 
            strategy=args.rerank_strategy
        )
        
        selected_indices = [candidate_indices[i] for i in selected_rel_indices]
        print(f"  Top 5 (Reranked): {[scene_image_names[idx] for idx in selected_indices]}")
    else:
        selected_indices = candidate_indices[:5]
        label = "Diversity Filtered" if args.use_diversity_filter else "Simple Top-K"
        print(f"  Top {len(selected_indices)} ({label}): {[scene_image_names[idx] for idx in selected_indices]}")

    top_5_images = [scene_image_names[idx] for idx in selected_indices]

    candidates_3d = []
    sam3_processor.set_confidence_threshold(0.5, use_presence_score=False) 

    # 3. Segment and Project
    for img_name in top_5_images:
        img_path = os.path.join(image_root, img_name)
        raw_image = Image.open(img_path).convert("RGB")
        
        try:
            base_name = os.path.splitext(img_name)[0]
            frame_idx = int(''.join(filter(str.isdigit, base_name)))
        except:
            continue

        pose = poses.get(frame_idx)
        depth_path = os.path.join(depth_root, base_name + ".npy")
        
        if pose is None or not os.path.exists(depth_path):
             continue
        depth_map = np.load(depth_path)

        # SAM3
        state = sam3_processor.set_image(raw_image)
        state = sam3_processor.set_text_prompt(query, state)
        
        masks = state['masks']
        scores = state['scores']
        
        if len(scores) > 0:
            best_idx = torch.argmax(scores).item()
            best_mask = masks[best_idx].cpu().numpy().squeeze()
            best_score = scores[best_idx].item()
            
            vis_3d_path = os.path.join('./vis', f"{key.replace('/','_')}_{img_name}_3d.png") if args.visualize else None
            center_3d = project_mask_to_3d(best_mask, depth_map, intrinsics, pose, debug_vis_path=vis_3d_path, gt_goals=gt_goals)
            if center_3d:
                candidates_3d.append(center_3d)
                
                # Visualize (Only for the best match per image)
                if args.visualize:
                    vis_path = os.path.join('./vis', f"{key.replace('/','_')}_{img_name}.png")
                    visualize_projection(img_path, best_mask, center_3d, pose, intrinsics, vis_path, 
                                         title_prefix=f"{query}")

    if candidates_3d:
        all_predictions[key] = candidates_3d
        print(f"  Found {len(candidates_3d)} candidates.")

        # Print GT and Predictions with Distances
        formatted_goals = [g.tolist() if hasattr(g, 'tolist') else g for g in gt_goals]
        print(f"  GT Positions: {formatted_goals}")
        
        for idx, pred in enumerate(candidates_3d[:5]):
            dists = []
            for gt in gt_goals:
                gt_arr = np.array(gt.tolist() if hasattr(gt, 'tolist') else gt)
                dists.append(np.linalg.norm(np.array(pred) - gt_arr))
            
            min_dist = min(dists) if dists else float('inf')
            print(f"    Pred {idx+1}: {[round(x, 4) for x in pred]}, Min Dist to GT: {min_dist:.4f}")

    else:
        print(f"  No candidates found.")

    query_duration = time.time() - start_time
    print(f"  Time taken: {query_duration:.4f}s")
    
    if scene not in perf_stats:
        perf_stats[scene] = {}
    if 'Text' not in perf_stats[scene]:
        perf_stats[scene]['Text'] = []
    perf_stats[scene]['Text'].append(query_duration)


print("\n=== PERFORMANCE REPORT ===")
# Filter dataset to only include samples that were actually processed (either successful or failed)
attempted_keys_set = set(attempted_keys)
processed_samples = [s for s in dataset if f"{s['scene']}/{s['episode']}/{s['target_name']}" in attempted_keys_set]

stats, scene_stats = evaluate_submission(processed_samples, all_predictions, filters=['language'], verbose=False)

# Calculate totals
all_text_times_list = [t for s in perf_stats.values() for t in s.get('Text', [])]
total_queries = len(all_text_times_list)
total_time_sum = sum(all_text_times_list)
print(f"Total Queries: {total_queries}")
print(f"Total Time: {total_time_sum:.2f}s")

print(f"{'Scene':<20} | {'Text (Avg)':<20} | {'Image (Avg)':<12} | {'Object (Avg)':<12} | {'All (Avg)':<20}")
print("-" * 90)

all_text_times = []
sorted_scenes = sorted(perf_stats.keys())

total_success = 0
total_samples = 0

for scene in sorted_scenes:
    # Time Stats
    time_stats = perf_stats[scene]
    text_times = time_stats.get('Text', [])
    
    if text_times:
        text_time_avg = sum(text_times) / len(text_times)
        all_text_times.extend(text_times)
    else:
        text_time_avg = 0.0

    # Accuracy Stats (Language/Text)
    # scene_stats contains aggregated stats for the scene. Since we filtered by 'language' only,
    # these stats are for language tasks.
    s_stats = scene_stats[scene]
    if s_stats['total'] > 0:
        acc = (s_stats['success'] / s_stats['total']) * 100
    else:
        acc = 0.0
        
    text_str = f"{acc:.2f}% ({text_time_avg:.2f}s)"
    
    # Image/Object (Placeholder - N/A as we only process language)
    img_str = "N/A"
    obj_str = "N/A"
    
    # Row Average (Same as Text for now since others are N/A)
    row_str = text_str
        
    print(f"{scene:<20} | {text_str:<20} | {img_str:<12} | {obj_str:<12} | {row_str:<20}")

print("-" * 90)

# Global Averages
if all_text_times:
    global_text_time_avg = sum(all_text_times) / len(all_text_times)
else:
    global_text_time_avg = 0.0

# Global Accuracy (Language)
# We can use the 'stats' dict returned by evaluate_submission which aggregates by task type
lang_stats = stats['language']
if lang_stats['total'] > 0:
    global_acc = (lang_stats['success'] / lang_stats['total']) * 100
else:
    global_acc = 0.0

global_text_str = f"{global_acc:.2f}% ({global_text_time_avg:.2f}s)"

print(f"{'Average':<20} | {global_text_str:<20} | {'N/A':<12} | {'N/A':<12} | {global_text_str:<20}") 