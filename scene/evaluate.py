import math
from collections import defaultdict
from goat_dataset import GoatDataset

def evaluate_submission(dataset, predictions, filters=None, threshold=1.5, verbose=True):
    """
    Evaluate predictions against the dataset ground truth.

    Args:
        dataset: GoatDataset instance.
        predictions: dict. Key is "{scene}/{episode}/{target_name}". 
                     Value is a list of [x, y, z] coordinates (Top-K candidates).
        filters: list of str (optional). Filter by task_type (e.g., ['language']). 
                 If None, evaluate all tasks found in predictions/dataset.
        threshold: float. Success distance radius in meters.
        verbose: bool. Whether to print results.

    Returns:
        tuple: (stats, scene_stats)
    """
    
    # Statistics storage: {'image': {'success': 0, 'total': 0}, ...}
    stats = defaultdict(lambda: {'success': 0, 'total': 0})
    scene_stats = defaultdict(lambda: {'success': 0, 'total': 0})
    
    if verbose:
        print(f"Starting Evaluation...")
        print(f"Filters: {filters if filters else 'None (All)'}")
        print(f"Threshold: {threshold}m")

    for sample in dataset:
        # 1. Identify Sample
        key = f"{sample['scene']}/{sample['episode']}/{sample['target_name']}"
        task_type = sample['task_type']
        scene_id = sample['scene']

        # 2. Apply Filters
        if filters and task_type not in filters:
            continue

        # 3. Retrieve Prediction
        if key not in predictions:
            # If a sample matches the filter but is missing from predictions, 
            # we count it as a Total but NOT a Success (i.e., Failure).
            stats[task_type]['total'] += 1
            stats['overall']['total'] += 1
            scene_stats[scene_id]['total'] += 1
            continue

        preds_top_k = predictions[key] # Expecting list of lists [[x,y,z], ...]
        gt_goals = sample['goals']     # List of lists [[x,y,z], ...]

        # 4. Check Success
        # Definition: Success if ANY prediction in Top-K is within threshold of ANY GT goal.
        is_success = False
        for p in preds_top_k:
            for g in gt_goals:
                dist = math.dist(p, g)
                if dist <= threshold:
                    is_success = True
                    break
            if is_success:
                break
        
        # 5. Update Stats
        stats[task_type]['total'] += 1
        stats['overall']['total'] += 1
        scene_stats[scene_id]['total'] += 1
        
        if is_success:
            stats[task_type]['success'] += 1
            stats['overall']['success'] += 1
            scene_stats[scene_id]['success'] += 1

    if verbose:
        # Print Results Table
        print("\n" + "="*45)
        print(f"{ 'Task Type':<15} | {'Samples':<8} | {'Success Rate':<12}")
        print("-" * 45)
        
        for task in sorted(stats.keys()):
            if task == 'overall': continue
            s = stats[task]
            rate = (s['success'] / s['total'] * 100) if s['total'] > 0 else 0.0
            print(f"{task:<15} | {s['total']:<8} | {rate:.2f}%")
            
        print("-" * 45)
        overall = stats['overall']
        rate = (overall['success'] / overall['total'] * 100) if overall['total'] > 0 else 0.0
        print(f"{ 'OVERALL':<15} | {overall['total']:<8} | {rate:.2f}%")
        print("="*45 + "\n")

        # Print Scene Breakdown
        print("\n" + "="*45)
        print(f"{ 'Scene ID':<15} | {'Samples':<8} | {'Success Rate':<12}")
        print("-" * 45)
        
        for scene in sorted(scene_stats.keys()):
            s = scene_stats[scene]
            rate = (s['success'] / s['total'] * 100) if s['total'] > 0 else 0.0
            print(f"{scene:<15} | {s['total']:<8} | {rate:.2f}%")
        print("="*45 + "\n")

    return stats, scene_stats

if __name__ == "__main__":
    # Example / Self-Test
    import random
    
    # 1. Load Dataset
    ds = GoatDataset('/home/jianwen/data/Goat-core')
    
    # 2. Generate Dummy Predictions (format demo)
    print("Generating dummy predictions (approx 50% success)...")
    dummy_preds = {}
    
    for s in ds.samples:
        key = f"{s['scene']}/{s['episode']}/{s['target_name']}"
        
        # Randomly decide to generate a "Good" prediction or "Bad" one
        if random.random() > 0.5:
            # Good: Take the first GT and add small noise
            gt = s['goals'][0]
            p = [gt[0]+0.1, gt[1]+0.1, gt[2]+0.1]
            dummy_preds[key] = [p] # Top-1
        else:
            # Bad: Far away
            dummy_preds[key] = [[1000.0, 1000.0, 1000.0]]
            
    # 3. Run Evaluation
    # Case A: Evaluate All
    evaluate_submission(ds, dummy_preds)
    
    # Case B: Evaluate only Language
    evaluate_submission(ds, dummy_preds, filters=['language'])
