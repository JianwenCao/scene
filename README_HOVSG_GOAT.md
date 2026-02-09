# HOV-SG Adaptation for Goat-Core Dataset

This directory contains the adapted implementation of HOV-SG (Hierarchical Open-Vocabulary Scene Graph) for the Goat-Core dataset. The system generates hierarchical scene graphs from RGB-D sequences and supports open-vocabulary 3D object retrieval using language queries.

## 1. Modifications & Implementation Details

To adapt HOV-SG for the Goat-Core dataset and modernize its backend, the following specific changes were made to the original codebase:

### A. Data Loading (`hovsg/dataloader/`)
*   **Created `hovsg/dataloader/goat.py`**:
    *   Implemented `GoatDataset` class inheriting from `GenericDataset`.
    *   **Quaternion Parsing**: Correctly parses `local_pos.txt` which uses `[ID, qw, qx, qy, qz, tx, ty, tz]` format (Goat-Core specific). Original generic loaders expected forward vectors.
    *   **Coordinate Transformation**: Applied a `T_gl_cv` transformation matrix (diagonal `[1, -1, -1, 1]`) to convert the input Computer Vision frame (Y-Down, Z-Forward) to the OpenGL World frame (Y-Up, Z-Back) required by HOV-SG's geometry processing.
*   **Updated `hovsg/dataloader/__init__.py`**: Registered the `goat` dataset module to be accessible via Hydra configuration (`main.dataset=goat`).

### B. Graph Construction (`hovsg/graph/navigation_graph.py`)
*   **Fix `get_floor_poses` Logic**:
    *   **Issue**: The original implementation filtered camera poses based on a hardcoded `camera_height=1.5` meter offset, assuming a human-height sensor relative to the floor. This caused all poses in Goat-Core (which has variable sensor heights) to be rejected, leading to empty graphs and DBSCAN failures ("0 samples found").
    *   **Fix**: Changed the default parameter to `camera_height=0.0`. This treats the pose position as the absolute sensor location without assuming a specific offset from the floor, allowing the navigation graph to be built correctly for any sensor trajectory.

### C. LLM Backend (`hovsg/utils/llm_utils.py`)
*   **Removed OpenAI Dependency**: Completely rewrote the module to remove calls to the legacy `openai.Completion.create` API.
*   **Integrated Google Gemini**:
    *   Added dependency on `google-genai` SDK.
    *   Implemented `parse_hier_query` and `parse_floor_room_object_gpt35` (renamed but kept for interface compatibility) using **Gemini 2.0 Flash**.
    *   The system now reliably parses natural language queries (e.g., "Find the chair in the kitchen") into structured `[Floor, Room, Object]` tuples to filter the search space.

### D. Dependencies
*   Added `google-genai` to the environment to support the new LLM backend.

### E. New Scripts (Root Directory)
*   `evaluate_hovsg.py`: A new standardized evaluation script that:
    *   Loads generated scene graphs.
    *   Iterates through Goat-Core language queries.
    *   Computes Success Rate @ 1.5m (as defined in `scene/evaluate.py`).
    *   Records timing and success statistics.
*   `query_hovsg.py`: A standalone script for ad-hoc querying and visualization of specific targets without running the full benchmark.
*   `goat_dataset.py`: A standalone dataloader for the evaluation script to read ground truth JSONs and `local_pos.txt` independently of the HOV-SG pipeline logic.

## 2. Performance Statistics

### Accuracy (Success Rate @ 1.5m)
| Scene | Success Rate |
| :--- | :--- |
| **4ok** | 73.17% |
| **5cd** | 26.83% |
| **nfv** | 51.43% |
| **tee** | 51.22% |
| **OVERALL** | **51.27%** |

### Execution Time
*   **Graph Generation (Build Time)**:
    *   `4ok`: ~2.5 mins
    *   `5cd`: ~15 mins
    *   `nfv`: ~5 mins
    *   `tee`: ~6 mins
*   **Query Time**: **~1.0141 seconds** per query (Average), 1.0428, 1.0033, 0.9713, 1.0329.

### Disk Usage
*   **Generated Graphs** (`output/goat/`): **~15.8 GB** (Contains dense point clouds, feature maps, and graph nodes).
*   **Raw Dataset** (`dataset/`): **~2.3 GB** (Required for graph generation, but not for querying).

## 3. How to Run

### Prerequisites
Ensure the environment is set up and `HOV-SG` is installed.
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Step 1: Generate Scene Graphs
Run the graph generation script for each scene. This processes RGB-D frames, segments objects, and builds the hierarchical graph.

```bash
# Add HOV-SG to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/workspace/HOV-SG

# Example for scene '4ok'
python HOV-SG/application/create_graph.py \
    main.dataset=goat \
    main.dataset_path=/workspace/Goat-core/dataset \
    main.split=. \
    main.scene_id=4ok \
    main.save_path=/workspace/Goat-core/output \
    pipeline.create_graph=True \
    models.clip.checkpoint=/workspace/HOV-SG/checkpoints/laion2b_s32b_b79k.bin \
    models.sam.checkpoint=/workspace/HOV-SG/checkpoints/sam_vit_h_4b8939.pth
```
*Repeat for `5cd`, `nfv`, and `tee`.*

### Step 2: Run Evaluation
Execute the evaluation script to process all language queries in the Goat-Core dataset against the generated graphs.

```bash
export PYTHONPATH=$PYTHONPATH:/workspace/HOV-SG
python evaluate_hovsg.py \
    +main.dataset=goat \
    +main.save_path=/workspace/Goat-core/output \
    models.clip.checkpoint=/workspace/HOV-SG/checkpoints/laion2b_s32b_b79k.bin
```

### Step 3: Interactive Query (Optional)
To query a specific object in a specific scene:

```bash
export PYTHONPATH=$PYTHONPATH:/workspace/HOV-SG
python query_hovsg.py \
    main.graph_path=/workspace/Goat-core/output/goat/4ok/graph \
    +main.query="chair" \
    models.clip.checkpoint=/workspace/HOV-SG/checkpoints/laion2b_s32b_b79k.bin
```
