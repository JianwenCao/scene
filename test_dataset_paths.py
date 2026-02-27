import os
import sys
from pathlib import Path

# Add OpenFunGraph to path
sys.path.append('/home/scene/OpenFunGraph')

from openfungraph.dataset.datasets_common import get_dataset

GOAT_ROOT = "/root/autodl-tmp/Goat-core"
DATASET_CONFIG = "/home/scene/OpenFunGraph/openfungraph/dataset/dataconfigs/goat/goat.yaml"
SCENE_ID = "5cd"

dataset = get_dataset(
    dataconfig=DATASET_CONFIG,
    basedir=os.path.join(GOAT_ROOT, "dataset"),
    sequence=SCENE_ID,
)

print(f"Dataset length: {len(dataset)}")
if len(dataset) > 0:
    print(f"First color path: {dataset.color_paths[0]}")
    color_path = Path(dataset.color_paths[0])
    print(f"Parent.Parent: {color_path.parent.parent}")
