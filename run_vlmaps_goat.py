import os
import sys
import time
import numpy as np
import torch
import shutil
import cv2
from pathlib import Path
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# 强制无缓冲
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

sys.path.append('/home/scene/vlmaps')
sys.path.append('/home/scene/HOV-SG')

from vlmaps.map.vlmap import VLMap
from goat_dataset import GoatDataset
from scene.evaluate import evaluate_submission
from vlmaps.utils.index_utils import get_segment_islands_pos, get_lseg_score

def prepare_vlmaps_config(scene_dir):
    return OmegaConf.create({
        "map_config": {
            "map_type": "vlmap",
            "pose_info": {
                "pose_type": "camera_base", "rot_type": "quat", "camera_height": 0.0,
                "base2cam_rot": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                "base_forward_axis": [1, 0, 0], "base_left_axis": [0, 1, 0], "base_up_axis": [0, 0, 1],
            },
            "cam_calib_mat": [388.19, 0, 320, 0, 388.19, 240, 0, 0, 1],
            "grid_size": 1000, "cell_size": 0.05, "depth_sample_rate": 20, 
            "dilate_iter": 3, "gaussian_sigma": 1.0, "skip_frame": 1,
            "potential_obstacle_names": ["chair", "wall", "table", "floor", "other"],
            "obstacle_names": ["wall", "chair"],
            "categories": "mp3d",
        }
    })

def convert_pose_to_vlmaps(scene_dir):
    local_pos_path = scene_dir / "local_pos.txt"
    vlmaps_pos_path = scene_dir / "poses.txt"
    if not local_pos_path.exists(): return
    poses = np.loadtxt(local_pos_path)
    R_cv_to_arkit = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    vl_poses = []
    for i in range(poses.shape[0]):
        qw, qx, qy, qz = poses[i, 1:5]
        tx, ty, tz = poses[i, 5:8]
        rot = R.from_quat([qx, qy, qz, qw])
        T_arkit_world = np.eye(4); T_arkit_world[:3, :3] = rot.as_matrix(); T_arkit_world[:3, 3] = [tx, ty, tz]
        # 严格执行: T_world_cv = T_world_arkit @ T_arkit_cv
        # T_arkit_cv = T_cv_arkit.inv()
        T_cv_to_arkit_mat = np.eye(4); T_cv_to_arkit_mat[:3, :3] = R_cv_to_arkit
        T_cv_world = T_arkit_world @ T_cv_to_arkit_mat
        vl_poses.append(np.concatenate([T_cv_world[:3, 3], R.from_matrix(T_cv_world[:3, :3]).as_quat()]))
    np.savetxt(vlmaps_pos_path, np.array(vl_poses))

def get_3d_pos_v2(vlmap, query):
    """
    改进的检索逻辑：
    1. 不再使用 argmax(other)，而是直接取 CLIP Score 最高的 Top-N 体素。
    2. 对这些体素进行空间聚类。
    """
    # 直接获取所有体素对 query 的原始得分
    scores = get_lseg_score(vlmap.clip_model, [query], vlmap.grid_feat, vlmap.clip_feat_dim, 
                            use_multiple_templates=True, add_other=False)
    scores = scores.flatten()
    
    # 取前 0.5% 的高分体素作为候选
    num_candidates = max(100, int(len(scores) * 0.005))
    indices = np.argsort(scores)[-num_candidates:]
    
    mask = np.zeros(len(scores), dtype=bool)
    mask[indices] = True
    
    if not np.any(mask): return []

    grid_size = vlmap.occupied_ids.shape
    mask_2d = np.zeros((grid_size[0], grid_size[2]), dtype=bool)
    mask_2d[vlmap.grid_pos[mask, 0], vlmap.grid_pos[mask, 2]] = True
    
    # 膨胀掩码以连接邻近点
    mask_2d = cv2.dilate(mask_2d.astype(np.uint8), np.ones((5,5), np.uint8), iterations=1)
    
    contours, centers, bbox_list, _ = get_segment_islands_pos(mask_2d, 1)
    if not centers: return []
        
    results = []
    # 对每个 Segment 计算平均分，按得分排序而不是面积
    segment_scores = []
    for idx in range(len(contours)):
        xmin, xmax, ymin, ymax = bbox_list[idx]
        rows, cols = vlmap.grid_pos[:, 0], vlmap.grid_pos[:, 2]
        matching_indices = np.where(mask & (rows >= xmin) & (rows <= xmax) & (cols >= ymin) & (cols <= ymax))[0]
        if len(matching_indices) == 0:
            segment_scores.append(-1.0)
            continue
        segment_scores.append(np.mean(scores[matching_indices]))
    
    sorted_seg_indices = np.argsort(segment_scores)[::-1]
    
    for idx in sorted_seg_indices[:5]:
        if segment_scores[idx] < 0: continue
        xmin, xmax, ymin, ymax = bbox_list[idx]
        rows, cols = vlmap.grid_pos[:, 0], vlmap.grid_pos[:, 2]
        matching_mask = mask & (rows >= xmin) & (rows <= xmax) & (cols >= ymin) & (cols <= ymax)
        
        pts = vlmap.grid_pos[matching_mask] * vlmap.map_config.cell_size + vlmap.pcd_min
        results.append(np.mean(pts, axis=0).tolist())
        
    return results

def main():
    dataset_root = Path("/root/autodl-tmp/Goat-core")
    data_root = dataset_root / "dataset"
    scenes = ['4ok', '5cd', 'nfv', 'tee']
    ds = GoatDataset(str(dataset_root))
    all_predictions = {}
    reports = {}

    for scene_id in scenes:
        print(f"\n--- [Processing {scene_id}] ---", flush=True)
        scene_dir = data_root / scene_id
        convert_pose_to_vlmaps(scene_dir)
        if not (scene_dir / "rgb").exists(): os.symlink(scene_dir / "images", scene_dir / "rgb")
        
        config = prepare_vlmaps_config(scene_dir)
        vlmap = VLMap(config.map_config, data_dir=str(scene_dir))
        
        # 如果已经建过图，直接加载以节省时间
        map_path = scene_dir / "vlmap_cam" / "vlmaps_cam.h5df"
        if not map_path.exists():
            print(f"  - Building new map...", flush=True)
            vlmap.create_map(str(scene_dir))
        
        vlmap.load_map(str(scene_dir))
        vlmap._init_clip()
        
        queries = [s for s in ds.samples if s['scene'] == scene_id and s['task_type'] == 'language']
        q_times = []
        for sample in tqdm(queries, desc=f"Querying {scene_id}"):
            t0 = time.time()
            all_predictions[f"{sample['scene']}/{sample['episode']}/{sample['target_name']}"] = get_3d_pos_v2(vlmap, sample['query'])
            q_times.append(time.time() - t0)
        
        reports[scene_id] = {"build_time": 0, "grid_size": vlmap.occupied_ids.shape, "avg_query": np.mean(q_times)}

    print("\n" + "="*25 + " FINAL PERFORMANCE REPORT " + "="*25, flush=True)
    samples = [s for s in ds.samples if s['scene'] in scenes and s['task_type'] == 'language']
    stats, scene_stats = evaluate_submission(samples, all_predictions, filters=['language'], verbose=False)
    
    for s_id in scenes:
        acc = (scene_stats[s_id]['success'] / scene_stats[s_id]['total'] * 100) if scene_stats[s_id]['total'] > 0 else 0
        print(f"Scene {s_id}: SR = {acc:.2f}% | Grid = {reports[s_id]['grid_size']}", flush=True)
    
    overall_acc = (stats['language']['success'] / stats['language']['total'] * 100)
    print(f"\nOverall Success Rate: {overall_acc:.2f}% (Target: 61.0%)", flush=True)
    print("="*76 + "\n")

if __name__ == "__main__":
    main()
