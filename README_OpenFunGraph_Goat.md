# OpenFunGraph on Goat-Core

This repository contains scripts to run OpenFunGraph pipeline on Goat-Core dataset for 3D visual grounding.

## Requirements

- OpenFunGraph environment (`conda activate openfungraph`)
- Goat-Core dataset in `/root/autodl-tmp/Goat-core`
- Grounded-Segment-Anything checkpoints (symlinked in `/home/scene/Grounded-Segment-Anything`)

## Usage

Run the pipeline script:

```bash
# Build map for a single scene
/root/autodl-tmp/conda/envs/openfungraph/bin/python OpenFunGraph/run_openfungraph_pipeline.py --mode build --scene 4ok --overwrite

# Query a single scene
/root/autodl-tmp/conda/envs/openfungraph/bin/python OpenFunGraph/run_openfungraph_pipeline.py --mode query --scene 4ok

# Build ALL scenes (4ok, 5cd, nfv, tee)
/root/autodl-tmp/conda/envs/openfungraph/bin/python OpenFunGraph/run_openfungraph_pipeline.py --mode build --scene all --overwrite

# Query ALL scenes and get aggregated metrics
/root/autodl-tmp/conda/envs/openfungraph/bin/python OpenFunGraph/run_openfungraph_pipeline.py --mode query --scene all
```

This script will:
1. Run Grounded-SAM detection on scene `4ok` (and potentially others if uncommented).
2. Fuse detections into 3D objects using OpenFunGraph SLAM pipeline.
3. Evaluate the results against Goat-Core ground truth using CLIP-based query matching.

## Files

- `run_openfungraph_pipeline.py`: Main script.
- `OpenFunGraph/openfungraph/dataset/dataconfigs/goat/goat.yaml`: Config for Goat-Core.
- `OpenFunGraph/openfungraph/dataset/datasets_common.py`: Modified to support GoatDataset.

## Output

Results are saved in `Goat-core/dataset/{scene}/pcd_saves/`.
Evaluation metrics are printed to stdout.
