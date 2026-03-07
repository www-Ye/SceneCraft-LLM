# SceneCraft-LLM: LLM-Agent Driven Interactive Indoor Scene Generation

> NeurIPS 2026 Submission

## Overview

SceneCraft-LLM is a training-free framework for generating simulation-ready 3D indoor scenes using LLM-based planning agents. Given a natural language description (e.g., "a cozy living room with a sofa facing the TV"), our system automatically:

1. **Parses** the description into structured scene requirements
2. **Plans** a spatially coherent furniture layout via LLM reasoning
3. **Retrieves** matching 3D assets from a large-scale database
4. **Optimizes** the layout with physical constraints (collision-free, gravity-stable)
5. **Outputs** a simulation-ready scene (compatible with Habitat, AI2-THOR, etc.)

## Key Contributions

- **Training-free**: No GPU training required; leverages LLM reasoning capabilities
- **Physically grounded**: Constraint-based optimization ensures realistic placements
- **Simulation-ready**: Direct output to popular embodied AI simulators
- **Scalable**: Can generate diverse scenes at scale via LLM prompting

## Project Structure

```
scene-gen/
├── src/
│   ├── planner/        # LLM-based scene planning agent
│   ├── layout/         # Layout optimization & constraint solving
│   ├── assets/         # 3D asset retrieval & management
│   ├── evaluation/     # Metrics & evaluation pipeline
│   └── utils/          # Shared utilities
├── configs/            # Configuration files
├── scripts/            # Data processing & experiment scripts
├── data/               # Dataset storage (3D-FRONT, etc.)
├── outputs/            # Generated scenes & results
└── docs/               # Documentation & paper drafts
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python -m src.planner.generate --prompt "a modern living room" --output outputs/
```

## Dataset

We use [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) as our primary dataset for evaluation.

## License

MIT
