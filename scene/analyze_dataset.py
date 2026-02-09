import os
import glob
from collections import defaultdict

root_dir = '/home/jianwen/data/Goat-core/groundtruth'
scenes = ['4ok', '5cd', 'nfv', 'tee']

object_stats = defaultdict(lambda: {'episodes': [], 'tasks': []})
task_counts = defaultdict(int)
pos_counts = defaultdict(list) # To check line counts in pos.txt

total_objects_count = 0
unique_objects = set()

print(f"Scanning {root_dir}...")

for scene in scenes:
    scene_path = os.path.join(root_dir, scene)
    if not os.path.exists(scene_path):
        continue
    
    for ep in range(6):
        ep_str = str(ep)
        ep_path = os.path.join(scene_path, ep_str)
        if not os.path.exists(ep_path):
            continue
            
        # List object directories
        obj_dirs = [d for d in os.listdir(ep_path) if os.path.isdir(os.path.join(ep_path, d))]
        
        for obj_name in obj_dirs:
            unique_id = f"{scene}/{obj_name}"
            unique_objects.add(unique_id)
            
            obj_path = os.path.join(ep_path, obj_name)
            
            # Read task type
            task_type = "unknown"
            task_file = os.path.join(obj_path, 'task_type.txt')
            if os.path.exists(task_file):
                with open(task_file, 'r') as f:
                    task_type = f.read().strip()
            
            # Read pos lines
            pos_lines = 0
            pos_file = os.path.join(obj_path, 'pos.txt')
            if os.path.exists(pos_file):
                with open(pos_file, 'r') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                    pos_lines = len(lines)
            
            object_stats[unique_id]['episodes'].append(ep)
            object_stats[unique_id]['tasks'].append(task_type)
            task_counts[task_type] += 1
            
            if pos_lines > 1:
                pos_counts['multi_line'].append(f"{scene}/{ep}/{obj_name}")
            else:
                pos_counts['single_line'].append(f"{scene}/{ep}/{obj_name}")

print(f"Total unique objects (Scene/ObjName): {len(unique_objects)}")
print(f"Task Type Counts: {dict(task_counts)}")
print(f"Objects with multi-line pos.txt: {len(pos_counts['multi_line'])}")
print(f"Objects with single-line pos.txt: {len(pos_counts['single_line'])}")

# Analyze overlaps and rotations
ep012_objects = set()
ep345_objects = set()

task_distribution = defaultdict(lambda: defaultdict(int)) # object -> task -> count

for unique_id, data in object_stats.items():
    eps = data['episodes']
    tasks = data['tasks']
    
    # Track which group this object belongs to
    in_012 = any(e in [0, 1, 2] for e in eps)
    in_345 = any(e in [3, 4, 5] for e in eps)
    
    if in_012: ep012_objects.add(unique_id)
    if in_345: ep345_objects.add(unique_id)
    
    # Count tasks per object
    for t in tasks:
        task_distribution[unique_id][t] += 1

print(f"Objects in Ep 0-2: {len(ep012_objects)}")
print(f"Objects in Ep 3-5: {len(ep345_objects)}")
print(f"Intersection: {len(ep012_objects.intersection(ep345_objects))}")

# Check rotation logic in Ep 3-5
print("\nChecking Task Rotation in Ep 3-5 (Sample):")
sample_count = 0
for unique_id in list(ep345_objects)[:5]:
    tasks = []
    eps = []
    for i, ep in enumerate(object_stats[unique_id]['episodes']):
        if ep in [3, 4, 5]:
            eps.append(ep)
            tasks.append(object_stats[unique_id]['tasks'][i])
    print(f"{unique_id}: Episodes {eps} -> Tasks {tasks}")

print("\nDeep Dive Analysis:")

# Analyze Ep 0-2 consistency
ep012_consistent = 0
ep012_total = 0
for unique_id in ep012_objects:
    eps = []
    tasks = []
    for i, ep in enumerate(object_stats[unique_id]['episodes']):
        if ep in [0, 1, 2]:
            eps.append(ep)
            tasks.append(object_stats[unique_id]['tasks'][i])
    
    if len(eps) == 3 and len(set(tasks)) == 3:
        ep012_consistent += 1
    ep012_total += 1

print(f"Ep 0-2 Objects with full 3-task rotation: {ep012_consistent}/{ep012_total}")

# Analyze Ep 3-5 consistency
ep345_counts = defaultdict(int)
for unique_id in ep345_objects:
    count = 0
    for ep in object_stats[unique_id]['episodes']:
        if ep in [3, 4, 5]:
            count += 1
    ep345_counts[count] += 1

print(f"Ep 3-5 Objects appearance counts: {dict(ep345_counts)}")

# Check what task types exist for single-appearance objects in Ep 3-5
single_appearance_tasks = defaultdict(int)
for unique_id in ep345_objects:
    my_tasks = []
    for i, ep in enumerate(object_stats[unique_id]['episodes']):
        if ep in [3, 4, 5]:
            my_tasks.append(object_stats[unique_id]['tasks'][i])
    
    if len(my_tasks) == 1:
        single_appearance_tasks[my_tasks[0]] += 1

print(f"Task types for objects appearing only once in Ep 3-5: {dict(single_appearance_tasks)}")
print(f"Total Object instances found: {sum(task_counts.values())}")
