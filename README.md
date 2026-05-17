# SHARP: Safe Heterogeneous Allocation with Risk Prediction

Pedestrian-aware multi-robot task allocation in human-shared warehouses, formulated as a chance-constrained CMDP.

## Project Layout

```
SHARP/
├── config/                  # YAML configs (scenarios, model, training)
├── sharp/
│   ├── env/                 # warehouse simulator + pedestrian model
│   ├── data/                # task instance generator
│   ├── model/               # SHARP neural network
│   ├── train/               # primal-dual PPO trainer
│   └── utils/               # visualization, logging
├── scripts/                 # entry points (demo, train, eval)
├── tests/                   # unit tests
├── paper/                   # LaTeX paper draft
└── results/                 # experiment outputs (logs, checkpoints, figs)
```

## Setup

```bash
python -m pip install -r requirements.txt
# When ready to train, install CUDA torch (current install is CPU-only):
# python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Phase Roadmap

| Phase | Status |
|---|---|
| 1. Simulator + theorems draft | in progress |
| 2. SHARP model + trainer | pending |
| 3. Smoke test | pending |
| 4. Baselines | pending |
| 5. Main experiments (24 runs) | pending — user runs |
| 6. Ablations (12 runs) | pending — user runs |
| 7. Paper finalization | pending |
| 8. Submit to Mathematics | pending |
