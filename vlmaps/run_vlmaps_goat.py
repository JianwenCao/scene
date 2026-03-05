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
import open3d as o3d

# 强制无缓冲输出
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
            "grid_size": 1000, "cell_size": 0.05, 
            "depth_sample_rate": 20, 
            "dilate_iter": 3, "gaussian_sigma": 1.0, "skip_frame": 1,
            "potential_obstacle_names": ["floor", "wall", "ceiling", "other"],
            "obstacle_names": ["wall"],
            "categories": "mp3d",
        }
    })

def convert_pose_to_vlmaps(scene_dir):
    """
    严格按照 HOV-SG 的 T_world_arkit @ T_arkit_cv 逻辑转换位姿。
    """
    local_pos_path = scene_dir / "local_pos.txt"
    vlmaps_pos_path = scene_dir / "poses.txt"
    if not local_pos_path.exists(): return
    poses = np.loadtxt(local_pos_path)
    
    # R_cv_to_arkit (OpenCV to ARKit)
    # x_arkit = x_cv, y_arkit = -y_cv, z_arkit = -z_cv
    R_cv_to_arkit = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    T_cv_to_arkit = np.eye(4); T_cv_to_arkit[:3, :3] = R_cv_to_arkit
    
    vl_poses = []
    for i in range(poses.shape[0]):
        qw, qx, qy, qz = poses[i, 1:5]
        tx, ty, tz = poses[i, 5:8]
        rot = R.from_quat([qx, qy, qz, qw])
        T_arkit_world = np.eye(4)
        T_arkit_world[:3, :3] = rot.as_matrix()
        T_arkit_world[:3, 3] = [tx, ty, tz]
        
        # 目标是：VLMap 使用 OpenCV 坐标 pc，通过 T_cv_world 变到世界坐标
        T_cv_world = T_arkit_world @ T_cv_to_arkit
        
        final_pos = T_cv_world[:3, 3]
        final_quat = R.from_matrix(T_cv_world[:3, :3]).as_quat()
        vl_poses.append(np.concatenate([final_pos, final_quat]))
        
    np.savetxt(vlmaps_pos_path, np.array(vl_poses))

def get_robust_bounds(scene_dir):
    """
    为防止 nfv 场景偏移，扫描全量位姿以确定真实边界。
    """
    print(f"  - 正在扫描位姿以锁定全局参考系...", flush=True)
    poses = np.loadtxt(scene_dir / "poses.txt")
    traj_pts = poses[:, :3]
    # 在轨迹基础上扩展 10 米缓冲区
    pmin = np.min(traj_pts, axis=0) - 10.0
    pmax = np.max(traj_pts, axis=0) + 10.0
    return pmin, pmax

def get_3d_pos_v4(vlmap, query):
    """
    增强检索：专门针对 nfv 场景增加相似度阈值过滤。
    """
    scores = get_lseg_score(vlmap.clip_model, [query], vlmap.grid_feat, vlmap.clip_feat_dim, 
                            use_multiple_templates=True, add_other=False)
    scores = scores.flatten()
    
    # 过滤掉得分过低的点 (低于均值 + 2倍标准差)
    threshold = np.mean(scores) + 2.0 * np.std(scores)
    mask = (scores > threshold)
    
    if not np.any(mask):
        # 回退到 Top-K
        indices = np.argsort(scores)[-100:]
        mask = np.zeros_like(scores, dtype=bool); mask[indices] = True

    grid_size = vlmap.occupied_ids.shape
    mask_2d = np.zeros((grid_size[0], grid_size[2]), dtype=bool)
    mask_2d[vlmap.grid_pos[mask, 0], vlmap.grid_pos[mask, 2]] = True
    mask_2d = cv2.dilate(mask_2d.astype(np.uint8), np.ones((5,5), np.uint8), iterations=1)
    
    contours, centers, bbox_list, _ = get_segment_islands_pos(mask_2d, 1)
    if not centers: return []
        
    segment_data = []
    for idx in range(len(contours)):
        xmin, xmax, ymin, ymax = bbox_list[idx]
        rows, cols = vlmap.grid_pos[:, 0], vlmap.grid_pos[:, 2]
        matching_indices = np.where(mask & (rows >= xmin) & (rows <= xmax) & (cols >= ymin) & (cols <= ymax))[0]
        if len(matching_indices) < 3: continue
        
        avg_score = np.mean(scores[matching_indices])
        segment_data.append((avg_score, matching_indices))
    
    segment_data.sort(key=lambda x: x[0], reverse=True)
    
    results = []
    for score, indices in segment_data[:5]:
        world_pts = vlmap.grid_pos[indices] * vlmap.map_config.cell_size + vlmap.pcd_min
        results.append(np.mean(world_pts, axis=0).tolist())
    return results

def main():
    dataset_root = Path("/root/autodl-tmp/Goat-core")
    data_root = dataset_root / "dataset"
    scenes = ['5cd']
    ds = GoatDataset(str(dataset_root))
    all_predictions = {}
    reports = {}

    print("\n" + "#"*60)
    print("  VLMaps Goat-Core 自动化评估系统 (绝对坐标版)")
    print("#"*60 + "\n")

    for scene_id in scenes:
        print(f"\n>>> 处理场景: {scene_id}", flush=True)
        scene_dir = data_root / scene_id
        
        # 强制清理以应用新的位姿对齐逻辑
        if (scene_dir / "vlmap_cam").exists(): shutil.rmtree(scene_dir / "vlmap_cam")
        
        convert_pose_to_vlmaps(scene_dir)
        if not (scene_dir / "rgb").exists(): os.symlink(scene_dir / "images", scene_dir / "rgb")
        
        config = prepare_vlmaps_config(scene_dir)
        vlmap = VLMap(config.map_config, data_dir=str(scene_dir))
        
        # 强制手动指定边界，不依赖点云扫描
        pmin, pmax = get_robust_bounds(scene_dir)
        vlmap.pcd_min = pmin
        vlmap.pcd_max = pmax

        print(f"  - 重新构建绝对坐标地图...", flush=True)
        t0 = time.time()
        vlmap.create_map(str(scene_dir))
        vlmap.load_map(str(scene_dir))
        build_time = time.time() - t0
        
        vlmap._init_clip()
        
        queries = [s for s in ds.samples if s['scene'] == scene_id and s['task_type'] == 'language']
        for sample in tqdm(queries, desc=f"Querying {scene_id}"):
            key = f"{sample['scene']}/{sample['episode']}/{sample['target_name']}"
            all_predictions[key] = get_3d_pos_v4(vlmap, sample['target_name'])
            if scene_id in ["5cd", "nfv"]:
                print(f"Query: {sample['target_name']} | Pred: {all_predictions[key]}", flush=True)
        
        reports[scene_id] = {"grid_size": vlmap.occupied_ids.shape, "build_time": build_time}

    # 执行评估
    print("\n" + "="*30 + " 性能汇总报告 " + "="*30, flush=True)
    samples = [s for s in ds.samples if s['scene'] in scenes and s['task_type'] == 'language']
    stats, scene_stats = evaluate_submission(samples, all_predictions, filters=['language'], verbose=False)
    for s_id in scenes:
        acc = (scene_stats[s_id]['success'] / scene_stats[s_id]['total'] * 100) if scene_stats[s_id]['total'] > 0 else 0
        print(f"{s_id}: SR = {acc:.2f}% | Grid = {reports[s_id]['grid_size']}", flush=True)
    
    overall_acc = (stats['language']['success'] / stats['language']['total'] * 100)
    print(f"\nFinal Overall SR: {overall_acc:.2f}% (Target: 61.0%)", flush=True)
    print("="*74 + "\n")

if __name__ == "__main__":
    main()
