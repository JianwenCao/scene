import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
import cv2
from scipy.spatial.transform import Rotation as R
import shutil

# Add paths
sys.path.append('/home/scene/vlmaps')
sys.path.append('/home/scene/HOV-SG')

from vlmaps.map.vlmap import VLMap
from goat_dataset import GoatDataset
from scene.evaluate import evaluate_submission
from vlmaps.utils.index_utils import get_segment_islands_pos

def prepare_vlmaps_config(scene_dir, gs=1000, cs=0.05):
    config = {
        "map_config": {
            "map_type": "vlmap",
            "pose_info": {
                "pose_type": "camera_base",
                "rot_type": "quat",
                "camera_height": 0.0,
                "base2cam_rot": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                "base_forward_axis": [1, 0, 0],
                "base_left_axis": [0, 1, 0],
                "base_up_axis": [0, 0, 1],
            },
            "cam_calib_mat": [388.19, 0, 320, 0, 388.19, 240, 0, 0, 1],
            "grid_size": gs,
            "cell_size": cs,
            "depth_sample_rate": 100,
            "dilate_iter": 3,
            "gaussian_sigma": 1.0,
            "skip_frame": 1,
            "potential_obstacle_names": ["chair", "wall", "table", "window", "floor", "stairs", "other"],
            "obstacle_names": ["wall", "chair", "table", "window", "stairs", "other"],
            "categories": "mp3d",
        },
        "params": {
            "gs": gs,
            "cs": cs,
        },
        "data_paths": {
            "vlmaps_data_dir": str(scene_dir.parent)
        }
    }
    return OmegaConf.create(config)

def convert_pose_to_vlmaps(scene_dir):
    local_pos_path = scene_dir / "local_pos.txt"
    vlmaps_pos_path = scene_dir / "poses.txt"
    
    poses = np.loadtxt(local_pos_path)
    
    R_cv_arkit = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    
    vl_poses = []
    for i in range(poses.shape[0]):
        qw, qx, qy, qz = poses[i, 1:5]
        tx, ty, tz = poses[i, 5:8]
        
        rot = R.from_quat([qx, qy, qz, qw])
        T_arkit_world = np.eye(4)
        T_arkit_world[:3, :3] = rot.as_matrix()
        T_arkit_world[:3, 3] = [tx, ty, tz]
        
        T_cv_arkit = np.eye(4)
        T_cv_arkit[:3, :3] = R_cv_arkit
        
        T_cv_world = T_arkit_world @ T_cv_arkit
        
        final_pos = T_cv_world[:3, 3]
        final_quat = R.from_matrix(T_cv_world[:3, :3]).as_quat() # x y z w
        
        vl_poses.append(np.concatenate([final_pos, final_quat]))
        
    np.savetxt(vlmaps_pos_path, np.array(vl_poses))
    return vlmaps_pos_path

def pool_3d_label_to_2d_cam(mask_3d, grid_pos, grid_size):
    mask_2d = np.zeros((grid_size[0], grid_size[2]), dtype=bool)
    rows = grid_pos[:, 0]
    cols = grid_pos[:, 2]
    mask_2d[rows[mask_3d], cols[mask_3d]] = True
    return mask_2d

def get_3d_pos(vlmap, query):
    mask = vlmap.index_map(query, with_init_cat=False)
    if not np.any(mask):
        return []

    grid_size = vlmap.occupied_ids.shape
    mask_2d = pool_3d_label_to_2d_cam(mask, vlmap.grid_pos, grid_size)
    
    contours, centers, bbox_list, _ = get_segment_islands_pos(mask_2d, 1)
    
    if not centers:
        return []
        
    segment_areas = []
    for contour in contours:
        area = cv2.contourArea(contour.astype(np.float32))
        segment_areas.append(area)
    
    sorted_indices = np.argsort(segment_areas)[::-1]
    
    results = []
    for idx in sorted_indices[:5]:
        center_2d = centers[idx] # [row, col]
        xmin, xmax, ymin, ymax = bbox_list[idx]
        
        rows = vlmap.grid_pos[:, 0]
        cols = vlmap.grid_pos[:, 2]
        in_bbox = (rows >= xmin) & (rows <= xmax) & (cols >= ymin) & (cols <= ymax)
        matching_mask = mask & in_bbox
        
        if not np.any(matching_mask):
            continue
            
        matching_voxels_pos = vlmap.grid_pos[matching_mask]
        world_pts = matching_voxels_pos * vlmap.map_config.cell_size + vlmap.pcd_min
        center_3d = np.mean(world_pts, axis=0)
        results.append(center_3d.tolist())
        
    return results

def main():
    dataset_root = Path("/root/autodl-tmp/Goat-core")
    data_root = dataset_root / "dataset"
    scenes = ['4ok', '5cd', 'nfv', 'tee']
    
    ds = GoatDataset(str(dataset_root))
    all_predictions = {}
    scene_reports = {}
    
    for scene_id in scenes:
        print(f"\nProcessing scene: {scene_id}")
        scene_dir = data_root / scene_id
        
        convert_pose_to_vlmaps(scene_dir)
        
        config = prepare_vlmaps_config(scene_dir)
        config.map_config.rgb_paths = sorted(list((scene_dir / "images").glob("*.png")))
        config.map_config.depth_paths = sorted(list((scene_dir / "depth").glob("*.npy")))
        config.map_config.pose_path = scene_dir / "poses.txt"
        
        if not (scene_dir / "rgb").exists():
            os.symlink(scene_dir / "images", scene_dir / "rgb")
            
        vlmap = VLMap(config.map_config, data_dir=str(scene_dir))
        
        build_start = time.time()
        map_path = scene_dir / "vlmap_cam" / "vlmaps_cam.h5df"
        if map_path.exists():
            shutil.rmtree(scene_dir / "vlmap_cam")

        print(f"Building map for {scene_id}...")
        vlmap.create_map(str(scene_dir))
        vlmap.load_map(str(scene_dir))
        
        build_end = time.time()
        build_time = build_end - build_start
        print(f"Map built in {build_time:.2f}s")
        
        grid_size = vlmap.occupied_ids.shape
        vlmap._init_clip()
        
        scene_samples = [s for s in ds.samples if s['scene'] == scene_id and s['task_type'] == 'language']
        query_times = []
        
        print(f"Querying {len(scene_samples)} language tasks...")
        for i, sample in enumerate(scene_samples):
            query = sample['query']
            key = f"{sample['scene']}/{sample['episode']}/{sample['target_name']}"
            
            if i % 10 == 0:
                print(f"  Query {i}/{len(scene_samples)}: {query}")
            
            q_start = time.time()
            preds = get_3d_pos(vlmap, query)
            q_end = time.time()
            
            query_times.append(q_end - q_start)
            if preds:
                all_predictions[key] = preds
        
        avg_query_time = np.mean(query_times) if query_times else 0
        scene_reports[scene_id] = {
            "build_time": build_time,
            "grid_size": grid_size,
            "avg_query_time": avg_query_time,
            "num_queries": len(scene_samples)
        }
        
    print("\n=== EVALUATION REPORT ===")
    relevant_samples = [s for s in ds.samples if s['scene'] in scenes and s['task_type'] == 'language']
    stats, scene_stats = evaluate_submission(relevant_samples, all_predictions, filters=['language'], verbose=False)
    
    print(f"{'Scene':<10} | {'Build Time (s)':<15} | {'Grid Size':<20} | {'Avg Query (s)':<15} | {'Success Rate':<12}")
    print("-" * 80)
    for scene_id in scenes:
        report = scene_reports[scene_id]
        s_stats = scene_stats[scene_id]
        acc = (s_stats['success'] / s_stats['total'] * 100) if s_stats['total'] > 0 else 0.0
        print(f"{scene_id:<10} | {report['build_time']:<15.2f} | {str(report['grid_size']):<20} | {report['avg_query_time']:<15.4f} | {acc:.2f}%")
        
    overall_acc = (stats['language']['success'] / stats['language']['total'] * 100) if stats['language']['total'] > 0 else 0.0
    print("-" * 80)
    print(f"Overall Accuracy: {overall_acc:.2f}%")

if __name__ == "__main__":
    main()
