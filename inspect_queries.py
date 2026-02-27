import sys
sys.path.append('/home/scene/HOV-SG')
from goat_dataset import GoatDataset

GOAT_ROOT = "/root/autodl-tmp/Goat-core"
ds = GoatDataset(GOAT_ROOT)

print(f"Total samples: {len(ds)}")
count = 0
for s in ds.samples:
    if s['scene'] == '5cd' and s['task_type'] == 'language':
        print(f"Target: {s['target_name']} | Query: {s['query']}")
        count += 1
        if count >= 20: break
