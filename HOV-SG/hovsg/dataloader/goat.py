
import os
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from hovsg.dataloader.generic import RGBDDataset

class GoatDataset(RGBDDataset):
    """
    Dataset class for the Goat-Core dataset.
    """
    
    def __init__(self, cfg):
        """
        Args:
            root_dir: Path to the scene directory (e.g. .../dataset/4ok).
            transforms: Optional transformations to apply to the data.
        """
        super(GoatDataset, self).__init__(cfg)
        self.root_dir = cfg["root_dir"]
        self.transforms = cfg["transforms"]
        
        # Load intrinsics from sparse/0/cameras.txt
        cameras_path = os.path.join(self.root_dir, "sparse", "0", "cameras.txt")
        self.rgb_intrinsics = self._load_intrinsics(cameras_path)
        self.depth_intrinsics = self.rgb_intrinsics # Assuming aligned
        
        self.scale = 1000.0 # Depth is likely in millimeters if saved as npy? Need to verify. 
                            # If it is float in meters, scale should be 1.0. 
                            # Usually numpy depth arrays are float meters or mm.
                            # Standard generic dataloader might expect integer depth and scale it?
                            # Scannet uses 1000.0.
                            # Let's assume standard behavior first. If npy is float meters, scale=1.0.
                            # If npy is uint16 mm, scale=1000.0.
                            # I'll verify depth content later. For now assume 1.0 if float, 1000.0 if int.
                            # Actually, typically .npy saves float meters. I'll stick with 1.0 for now but check later.
                            # Wait, Scannet uses 1000.0.
                            # Let's verify a depth file content.
        
        self.data_list = self._get_data_list()
        
        # Check depth type to set scale
        if len(self.data_list) > 0:
            sample_depth = np.load(self.data_list[0][1])
            if sample_depth.dtype == np.float32 or sample_depth.dtype == np.float64:
                # If values are small (like < 100), likely meters.
                if np.max(sample_depth) < 100:
                    self.scale = 1.0
                else:
                    self.scale = 1000.0 # mm
            else:
                self.scale = 1000.0 # integer mm

    def __getitem__(self, idx):
        rgb_path, depth_path, pose = self.data_list[idx]
        rgb_image = self._load_image(rgb_path)
        depth_image = self._load_depth(depth_path)
        
        if self.transforms is not None:
            rgb_image = self.transforms(rgb_image)
            depth_image = self.transforms(depth_image)
            
        return rgb_image, depth_image, pose, self.rgb_intrinsics, self.depth_intrinsics

    def _get_data_list(self):
        """
        Get a list of RGB-D data samples.
        Returns:
            List of (rgb_path, depth_path, pose_matrix) tuples.
        """
        image_dir = os.path.join(self.root_dir, "images")
        depth_dir = os.path.join(self.root_dir, "depth")
        pose_path = os.path.join(self.root_dir, "local_pos.txt")
        
        print(f"[GoatDataset] Loading data from root: {self.root_dir}")
        print(f"[GoatDataset] Image dir: {image_dir}, exists: {os.path.exists(image_dir)}")
        print(f"[GoatDataset] Depth dir: {depth_dir}, exists: {os.path.exists(depth_dir)}")
        print(f"[GoatDataset] Pose path: {pose_path}, exists: {os.path.exists(pose_path)}")

        # Get sorted file lists
        rgb_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")]) if os.path.exists(image_dir) else []
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".npy")]) if os.path.exists(depth_dir) else []
        
        print(f"[GoatDataset] Found {len(rgb_files)} RGB files.")
        print(f"[GoatDataset] Found {len(depth_files)} Depth files.")

        # Parse poses
        poses_dict = {}
        if os.path.exists(pose_path):
            with open(pose_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 8: continue
                    # Format: ID, qw, qx, qy, qz, tx, ty, tz
                    # ID usually matches image number, e.g. "1" for img0001
                    try:
                        idx = int(parts[0])
                        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                        
                        # Convert to matrix
                        rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix() # scipy uses [x, y, z, w]
                        pose_mat = np.eye(4)
                        pose_mat[:3, :3] = rot
                        pose_mat[:3, 3] = [tx, ty, tz]
                        
                        # Apply T_gl_cv transformation (CV to OpenGL)
                        # CV: Y-Down, Z-Forward
                        # OpenGL: Y-Up, Z-Back
                        # Transformation: diag(1, -1, -1, 1)
                        T_gl_cv = np.diag([1, -1, -1, 1])
                        pose_mat = pose_mat @ T_gl_cv
                        
                        poses_dict[idx] = pose_mat
                    except ValueError:
                        continue
        
        print(f"[GoatDataset] Loaded {len(poses_dict)} poses.")
        if len(poses_dict) > 0:
            print(f"[GoatDataset] Sample pose keys: {list(poses_dict.keys())[:5]}")

        data_list = []
        for rgb_f, depth_f in zip(rgb_files, depth_files):
            # Extract ID from filename (e.g. img0001.png -> 1)
            try:
                # Assuming format imgXXXX.png
                idx = int(rgb_f.replace("img", "").split(".")[0])
                if idx in poses_dict:
                    data_list.append((
                        os.path.join(image_dir, rgb_f),
                        os.path.join(depth_dir, depth_f),
                        poses_dict[idx]
                    ))
            except ValueError:
                continue
        
        print(f"[GoatDataset] Final data_list size: {len(data_list)}")
        return data_list

    def _load_image(self, path):
        return Image.open(path).convert("RGB")

    def _load_depth(self, path):
        return np.load(path)

    def _load_pose(self, path):
        # Poses are pre-loaded in _get_data_list, so this is unused or needs adaptation if used elsewhere.
        # But base class calls it? No, base class __getitem__ is abstract. 
        # Scannet implemented it. I implemented __getitem__ to use pre-loaded pose.
        pass

    def _load_intrinsics(self, path):
        # 1 PINHOLE 640 480 388.19 388.19 320 240
        with open(path, "r") as f:
            line = f.readline().split()
            # ID, TYPE, W, H, fx, fy, cx, cy
            fx = float(line[4])
            fy = float(line[5])
            cx = float(line[6])
            cy = float(line[7])
            
            intrinsics = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
        return intrinsics
