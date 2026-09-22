# PCT-GCN

PyTorch implementation of "PCT-GCN: Prior-Constrained Adaptive Topology Learning with Kinematics-Aware Routing for Skeleton-Based Action Recognition"

## Setup

```bash
uv sync
```

## Data

Place preprocessed `.npz` datasets under `datasets/processed/`:

| Benchmark   | File                       |
|-------------|----------------------------|
| NTU60 xsub  | `NTU60_CS.npz`             |
| NTU60 xview | `NTU60_CV.npz`             |
| NTU120 xsub | `NTU120_CSub.npz`          |
| NTU120 xset | `NTU120_CSet.npz`          |
| NW-UCLA     | `NW_UCLA.npz`              |

## Train / test

```bash
python main.py -c config/PCT-GCN/ntu60_xview/j.yaml
```

Modalities: `j` (joint), `b` (bone), `jm` (joint motion), `bm` (bone motion),
`k` (K-bone), `km` (K-bone motion).
