import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from collections import defaultdict
import math
import time

# Add paths
sys.path.append(os.path.abspath("HOV-SG"))
sys.path.append(os.path.abspath("scene"))

from hovsg.graph.graph import Graph
from goat_dataset import GoatDataset
from evaluate import evaluate_submission

# Helper to load graph
def load_hovsg_graph(cfg, scene_id):
    # Construct path to the graph
    # cfg.main.save_path should be the root output dir, e.g. /workspace/Goat-core/output
    # The graph is saved in {save_path}/{dataset}/{scene_id}/graph
    graph_path = os.path.join(cfg.main.save_path, cfg.main.dataset, scene_id, "graph")
    
    if not os.path.exists(graph_path):
        print(f"Graph not found at {graph_path}")
        return None

    # We need to trick the Graph init to think we are loading this specific scene
    # But Graph init mostly uses cfg for model params.
    # The load_graph method takes the path.
    
    print(f"Loading graph for scene {scene_id} from {graph_path}...")
    hovsg = Graph(cfg)
    hovsg.load_graph(graph_path)
    
    # Generate room names (needed for some query types)
    hovsg.generate_room_names(
        generate_method="view_embedding",
        default_room_types=[
            "office", "kitchen", "bathroom", "seminar room", 
            "meeting room", "dinning room", "corridor",
            "bedroom", "living room", "hallway", "lobby" 
        ]
    )
    return hovsg

@hydra.main(version_base=None, config_path="HOV-SG/config", config_name="visualize_query_graph")
def main(cfg: DictConfig):
    # 1. Load Dataset
    dataset_root = "/workspace/Goat-core"
    print(f"Loading GoatDataset from {dataset_root}...")
    dataset = GoatDataset(dataset_root)
    
    # 2. Setup storage for predictions
    predictions = {}
    
    # Cache graphs: scene_id -> hovsg_instance
    graphs = {}
    
    # Filter for scenes we want to evaluate
    target_scenes = ['4ok', '5cd', 'nfv', 'tee'] 
    
    # Timing statistics
    query_times = {s: [] for s in target_scenes}
    
    # 3. Iterate samples
    print(f"Evaluating on scenes: {target_scenes}")
    
    processed_count = 0
    
    for i, sample in enumerate(dataset):
        scene_id = sample['scene']
        task_type = sample['task_type']
        
        # Filter: Only language tasks and only target scenes
        if task_type != 'language':
            continue
        if scene_id not in target_scenes:
            continue
            
        # Load graph if not cached
        if scene_id not in graphs:
            graphs[scene_id] = load_hovsg_graph(cfg, scene_id)
            
        hovsg = graphs[scene_id]
        if hovsg is None:
            continue # Graph not found
            
        # Run Query
        query = sample['query']
        key = f"{sample['scene']}/{sample['episode']}/{sample['target_name']}"
        
        # print(f"Processing {key}: '{query}'")
        
        candidates_3d = []
        
        t0 = time.time()
        try:
            # Try hierarchical query first
            # We assume OpenAI key might be missing, so we catch it
            try:
                floor, room, objs = hovsg.query_hierarchy(query, top_k=5)
                # objs is a list of Object instances
                for obj in objs:
                    candidates_3d.append(obj.pcd.get_center().tolist())
            except KeyError as e:
                if "OPENAI_KEY" in str(e) or "OPENAI_API_KEY" in str(e) or "GEMINI_API_KEY" in str(e):
                    # Fallback to simple object search
                    print(f"Key error ({e}), falling back to simple search.")
                    obj_indices, room_indices = hovsg.query_object(query, top_k=5)
                    for obj_idx in obj_indices:
                        obj = hovsg.objects[obj_idx]
                        candidates_3d.append(obj.pcd.get_center().tolist())
                else:
                    raise e
            except Exception as e:
                print(f"Error querying '{query}': {e}")
                
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        t1 = time.time()
        query_duration = t1 - t0
        query_times[scene_id].append(query_duration)

        if candidates_3d:
            predictions[key] = candidates_3d
            processed_count += 1
            
    print(f"Processed {processed_count} samples.")
    
    # Print Timing Stats
    print("\n=== TIMING STATISTICS ===")
    print(f"{'Scene':<10} | {'Avg Query Time (s)':<20} | {'Total Time (s)':<15} | {'Samples':<10}")
    print("-" * 65)
    
    all_times = []
    for s in target_scenes:
        times = query_times[s]
        if times:
            avg_t = sum(times) / len(times)
            tot_t = sum(times)
            count = len(times)
            print(f"{s:<10} | {avg_t:<20.4f} | {tot_t:<15.4f} | {count:<10}")
            all_times.extend(times)
        else:
            print(f"{s:<10} | {'N/A':<20} | {'0.0000':<15} | {'0':<10}")
            
    if all_times:
        global_avg = sum(all_times) / len(all_times)
        global_tot = sum(all_times)
        global_count = len(all_times)
        print("-" * 65)
        print(f"{'OVERALL':<10} | {global_avg:<20.4f} | {global_tot:<15.4f} | {global_count:<10}")
    print("=========================\n")
    
    # 4. Run Evaluation
    if processed_count > 0:
        relevant_samples = [
            s for s in dataset 
            if s['scene'] in target_scenes and s['task_type'] == 'language'
        ]
        
        evaluate_submission(relevant_samples, predictions, filters=['language'], threshold=1.5)
    else:
        print("No samples processed.")

if __name__ == "__main__":
    main()