# HOV-SG Adaptation for Goat-Core Dataset

This directory contains the adapted implementation of HOV-SG (Hierarchical Open-Vocabulary Scene Graph) for the Goat-Core dataset. The system generates hierarchical scene graphs from RGB-D sequences and supports open-vocabulary 3D object retrieval using language queries.

## 1. Modifications & Implementation Details

To adapt HOV-SG for the Goat-Core dataset and modernize its backend, the following specific changes were made to the original codebase:

### A. Data Loading (`hovsg/dataloader/`)
*   **Created `hovsg/dataloader/goat.py`**:
    *   Implemented `GoatDataset` class inheriting from `RGBDDataset`.
    *   **Quaternion Parsing**: Correctly parses `local_pos.txt` which uses `[ID, qw, qx, qy, qz, tx, ty, tz]` format (Goat-Core specific).
    *   **Coordinate Transformation**: Applied a `T_gl_cv` transformation matrix (diagonal `[1, -1, -1, 1]`) to convert the input Computer Vision frame (Y-Down, Z-Forward) to the OpenGL World frame (Y-Up, Z-Back) required by HOV-SG's geometry processing.
*   **Updated `hovsg/dataloader/__init__.py`**: Registered the `goat` dataset module to be accessible via Hydra configuration (`main.dataset=goat`).

### B. Graph Construction (`hovsg/graph/navigation_graph.py`)
*   **Fix `get_floor_poses` Logic**:
    *   **Issue**: The original implementation filtered camera poses based on a hardcoded `camera_height=1.5` meter offset. This caused all poses in Goat-Core (which has variable sensor heights) to be rejected.
    *   **Fix**: Changed the default parameter to `camera_height=0.0`. This treats the pose position as the absolute sensor location.
*   **Headless Support**: Switched `matplotlib` backend to `Agg` to support running in headless environments.

### C. LLM Backend (`hovsg/utils/llm_utils.py`)
*   **Removed OpenAI Dependency**: Completely rewrote the module to remove calls to the legacy OpenAI API.
*   **Integrated Google Gemini**:
    *   Added dependency on `google-genai` SDK.
    *   Implemented `parse_hier_query` and `parse_floor_room_object_gpt35` using **Gemini 2.0 Flash**.
    *   The system now parses natural language queries into structured `[Floor, Room, Object]` tuples.

### D. Dependencies
*   Added `google-genai` to the environment.

### E. New Scripts (Root Directory)
*   `evaluate_hovsg.py`: Standardized evaluation script for Goat-Core.
*   `query_hovsg.py`: Standalone script for ad-hoc querying and visualization.
*   `goat_dataset.py`: Standalone dataloader for evaluation.

## 2. Performance Statistics

### Accuracy (Success Rate @ 1.5m)
| Scene | Success Rate |
| :--- | :--- |
| **4ok** | 73.17% |
| **5cd** | 26.83% |
| **nfv** | 51.43% |
| **tee** | 51.22% |
| **OVERALL** | **51.27%** |

## 3. How to Run

### Prerequisites
Ensure the environment is set up and `HOV-SG` is installed.
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Step 1: Generate Scene Graphs
Run the graph generation script for each scene.

```bash
# Add HOV-SG to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/HOV-SG

# Example for scene '4ok'
python HOV-SG/application/create_graph.py \
    main.dataset=goat \
    main.dataset_path=$(pwd)/Goat-core/dataset \
    main.split=. \
    main.scene_id=4ok \
    main.save_path=$(pwd)/Goat-core/output \
    pipeline.create_graph=True \
    models.clip.checkpoint=/home/scene/HOV-SG/checkpoints/laion2b_s32b_b79k.bin \
    models.sam.checkpoint=/home/scene/HOV-SG/checkpoints/sam_vit_h_4b8939.pth
```
*Repeat for `5cd`, `nfv`, and `tee`.*

### Step 2: Run Evaluation
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/HOV-SG
python evaluate_hovsg.py \
    +main.dataset=goat \
    +main.save_path=$(pwd)/Goat-core/output \
    models.clip.checkpoint=/home/scene/HOV-SG/checkpoints/laion2b_s32b_b79k.bin
```

### Step 3: Interactive Query (Optional)
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/HOV-SG
python query_hovsg.py \
    main.graph_path=$(pwd)/Goat-core/output/goat/4ok/graph \
    +main.query="chair" \
    models.clip.checkpoint=/home/scene/HOV-SG/checkpoints/laion2b_s32b_b79k.bin
```
