print("Testing Imports...", flush=True)
import os
print("  - os imported", flush=True)
import sys
print("  - sys imported", flush=True)
import numpy as np
print("  - numpy imported", flush=True)
import torch
print("  - torch imported", flush=True)
import cv2
print("  - cv2 imported", flush=True)
from scipy.spatial.transform import Rotation as R
print("  - scipy imported", flush=True)
# 怀疑对象 1: open3d
import open3d as o3d
print("  - open3d imported", flush=True)
# 怀疑对象 2: clip
import clip
print("  - clip imported", flush=True)
# 怀疑对象 3: 路径中的本地库
sys.path.append('/home/scene/vlmaps')
sys.path.append('/home/scene/HOV-SG')
from vlmaps.map.vlmap import VLMap
print("  - VLMap imported (ALL OK)", flush=True)
