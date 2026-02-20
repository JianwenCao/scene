import os
import glob
import torch

class GoatDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.gt_root = os.path.join(root_dir, 'groundtruth')
        self.sensor_root = os.path.join(root_dir, 'dataset')
        self.samples = []
        self._scan_dataset()

    def _scan_dataset(self):
        # Assume standard structure: root/groundtruth/scene/episode/object
        scenes = sorted(os.listdir(self.gt_root))
        
        for scene in scenes:
            scene_gt_path = os.path.join(self.gt_root, scene)
            scene_sensor_path = os.path.join(self.sensor_root, scene)
            
            # Episodes 0 to 5
            for ep_id in range(6):
                ep_path = os.path.join(scene_gt_path, str(ep_id))
                if not os.path.exists(ep_path): continue
                
                for obj_name in sorted(os.listdir(ep_path)):
                    obj_path = os.path.join(ep_path, obj_name)
                    if not os.path.isdir(obj_path): continue
                    
                    # 1. Parse Task Type
                    with open(os.path.join(obj_path, 'task_type.txt'), 'r') as f:
                        task_type = f.read().strip()
                    
                    # 2. Parse Goal Positions (GT) - Supports multiple lines
                    goals = []
                    with open(os.path.join(obj_path, 'pos.txt'), 'r') as f:
                        for line in f:
                            if line.strip():
                                # format: [x, y, z] or x y z
                                clean_line = line.replace('[', '').replace(']', '').replace(',', ' ')
                                goals.append(torch.tensor([float(x) for x in clean_line.split()]))

                    # 3. Determine Query (Text or Image Path)
                    query = None
                    if task_type == 'image':
                        # Find the first png file
                        img_files = glob.glob(os.path.join(obj_path, '*.png'))
                        query = img_files[0] # Assume existence
                    else:
                        # 'language' or 'object' -> read text description
                        with open(os.path.join(obj_path, 'language.txt'), 'r') as f:
                            query = f.read().strip()

                    self.samples.append({
                        'scene': scene,
                        'episode': ep_id,
                        'target_name': obj_name,
                        'task_type': task_type,
                        'goals': goals,        # List of [x, y, z]
                        'query': query,        # Text string or absolute image path
                        'sensor_path': scene_sensor_path # Path to images/depth/local_pos.txt
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    # Quick Test
    ds = GoatDataset('/home/scene/Goat-core')
    print(f"Loaded {len(ds)} samples.")
    if len(ds) > 0:
        s = ds[0]
        print(f"Sample 0: {s['scene']} Ep{s['episode']} {s['task_type']} -> Goal: {s['goals'][0]}")
        
        # Check a multi-goal sample (from previous analysis, usually in Ep 3-5)
        for s in ds.samples:
            if len(s['goals']) > 1:
                print(f"Multi-goal Sample: {s['scene']} Ep{s['episode']} {s['target_name']} -> {len(s['goals'])} goals")
                break
