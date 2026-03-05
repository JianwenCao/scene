import sys
import time
import numpy as np
import cv2
from pathlib import Path
from omegaconf import OmegaConf

sys.path.append('/home/scene/vlmaps')
sys.path.append('/home/scene/HOV-SG')

from vlmaps.map.vlmap import VLMap
from vlmaps.utils.index_utils import get_segment_islands_pos, get_lseg_score

def get_3d_pos_v4(vlmap, query):
    scores = get_lseg_score(vlmap.clip_model, [query], vlmap.grid_feat, vlmap.clip_feat_dim, 
                            use_multiple_templates=True, add_other=False)
    scores = scores.flatten()
    
    threshold = np.mean(scores) + 2.0 * np.std(scores)
    mask = (scores > threshold)
    
    if not np.any(mask):
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
    config = OmegaConf.create({
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
    
    scene_id = '5cd'
    data_dir = f'/root/autodl-tmp/Goat-core/dataset/{scene_id}'
    vlmap = VLMap(config.map_config, data_dir=data_dir)
    vlmap.load_map(data_dir)
    vlmap._init_clip()
    
    queries = ['chair', 'table', 'door', 'bed', 'sofa']
    times = []
    for q in queries:
        t0 = time.time()
        get_3d_pos_v4(vlmap, q)
        times.append(time.time() - t0)
        print(f"Query '{q}' took {times[-1]:.4f}s")
    
    print(f"Average query time: {np.mean(times):.4f}s")

if __name__ == "__main__":
    main()
