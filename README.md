# PCT-GCN

PyTorch implementation of "Prior-Constrained Adaptive Topology Learning with Kinematics-Aware Routing for Skeleton-Based Action Recognition"

## Setup

```bash
uv sync
```

## Data

Place preprocessed datasets under `datasets/processed/`. Each benchmark is a directory holding four `.npy` files (the `data_path` in the configs points to this directory):

```
datasets/processed/<benchmark>/
├── x_train.npy
├── x_test.npy
├── y_train.npy
└── y_test.npy
```

| Benchmark   | Directory      | `x_*.npy` shape (float32)       | `y_*.npy` shape (one-hot) |
|-------------|----------------|---------------------------------|---------------------------|
| NTU60 xsub  | `NTU60_CS`     | `(N, 300, 150)`                 | `(N, 60)`                 |
| NTU60 xview | `NTU60_CV`     | `(N, 300, 150)`                 | `(N, 60)`                 |
| NTU120 xsub | `NTU120_CSub`  | `(N, 300, 150)`                 | `(N, 120)`                |
| NTU120 xset | `NTU120_CSet`  | `(N, 300, 150)`                 | `(N, 120)`                |
| NW-UCLA     | `NW_UCLA`      | `(N, T, 20, 3)`                 | `(N, 10)` |

- NTU: `150 = 2 persons × 25 joints × 3 (x, y, z)`, zero-padded to 300 frames; the feeder reshapes to `(N, 3, 300, 25, 2)`.
- NW-UCLA: 20 joints, single person; the feeder reshapes to `(N, 3, T, 20, 1)`.
- Labels are one-hot; the feeder converts them to class indices with `argmax`.

## Train / test

```bash
python main.py -c config/PCT-GCN/ntu60_xview/j.yaml
```

Modalities: `j` (joint), `b` (bone), `jm` (joint motion), `bm` (bone motion)
