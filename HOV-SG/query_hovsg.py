import sys
import os
from copy import deepcopy
from hovsg.graph.graph import Graph
import hydra
from omegaconf import DictConfig
import numpy as np

# Ensure HOV-SG is in path if needed, though installed via pip -e .
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@hydra.main(version_base=None, config_path="config", config_name="visualize_query_graph")
def main(params: DictConfig):
    # Load graph
    print(f"Loading graph from: {params.main.graph_path}")
    hovsg = Graph(params)
    hovsg.load_graph(params.main.graph_path)
    
    # generate room names
    print("Generating room names...")
    hovsg.generate_room_names(
            generate_method="view_embedding",
            default_room_types=[
                "office",
                "kitchen",
                "bathroom",
                "seminar room",
                "meeting room",
                "dinning room",
                "corridor",
            ])
    
    query = params.main.get("query", "chair")
    print(f"Processing query: {query}")
    
    try:
        floor, room, obj = hovsg.query_hierarchy(query, top_k=1)
    except KeyError as e:
        if "OPENAI_KEY" in str(e) or "OPENAI_API_KEY" in str(e):
            print("OPENAI_KEY not found, falling back to simple object search.")
            # Fallback: search for object in all rooms
            # query_object returns (indices in self.objects, indices in self.rooms)
            obj_indices, room_indices = hovsg.query_object(query, top_k=1)
            obj = [hovsg.objects[i] for i in obj_indices]
        else:
            raise e
    
    if obj:
        target_obj = obj[0]
        center = target_obj.pcd.get_center()
        print(f"Found object: {target_obj.name}")
        print(f"Object ID: {target_obj.object_id}")
        print(f"3D Center: {center}")
    else:
        print("No object found.")

if __name__ == "__main__":
    main()
