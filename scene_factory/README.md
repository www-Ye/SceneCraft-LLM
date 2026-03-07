# SceneFactory

**LLM-Driven Procedural Generation of Physically-Grounded MuJoCo Training Environments for Embodied AI**

Generate diverse, physics-validated MuJoCo environments for robot manipulation and navigation training.

## Project Structure

```
scene_factory/
├── README.md                       # This file
├── generators/
│   ├── tabletop_gen.py             # Procedural tabletop environment generator
│   └── room_gen.py                 # Procedural room navigation environment generator
├── llm_tabletop_pipeline.py        # MuJoCo XML builder with real assets + robot arm
├── self_iterate.py                 # LLM-in-the-loop: design → render → VLM review → refine
├── realistic_tabletop.py           # Realistic rendering with validated 3D meshes
├── fix_and_render.py               # Axis-corrected rendering with best mesh variants
├── piper_grasp.py                  # Piper robot arm grasp planning (pick & place)
├── mujoco_menagerie/               # Google DeepMind's robot models
│   └── agilex_piper/              # Piper 6-DOF robot arm + gripper
├── outputs/
│   ├── llm_tabletop/              # LLM-in-the-loop iteration outputs
│   │   ├── test/                  # Initial manual test scene
│   │   └── iterate_YYYYMMDD/      # Each LLM iteration run
│   │       ├── iter_0/            # First iteration
│   │       ├── iter_1/            # Second iteration (refined)
│   │       ├── iter_2/            # Third iteration
│   │       └── final/             # Best accepted layout
│   ├── realistic/
│   │   ├── demo_breakfast/        # First demo (before axis fix)
│   │   ├── fixed_breakfast/       # After axis mapping correction
│   │   └── llm_breakfast/         # LLM-designed compact layout
│   ├── piper_grasp/               # Piper arm grasp demonstrations
│   └── demo/                      # Procedural generator demos
└── envs/                          # Batch-generated environments
```

## Output Directories Explained

### `outputs/realistic/`

| Directory | Description |
|-----------|-------------|
| `demo_breakfast/` | First version with all 15 mesh types. Some had axis mapping issues causing distortion. |
| `fixed_breakfast/` | After fixing axis permutation — each STL tested all 6 axis orderings. Only uses meshes with distortion ratio < 2.0. Manually designed layout. |
| `llm_breakfast/` | **Best version.** LLM (Claude Sonnet) designs a compact, realistic layout. Axis-corrected meshes. Natural object grouping. |

### `outputs/llm_tabletop/`

| Directory | Description |
|-----------|-------------|
| `test/` | Initial pipeline test with hand-designed layout and simplified Panda arm. |
| `iterate_YYYYMMDD/` | Full LLM-in-the-loop runs. Each contains `iter_0/`, `iter_1/`, etc., showing how the VLM review drives layout improvements. |

## Rendered Image Types

Each scene directory contains multiple rendered views:

| Filename | Description |
|----------|-------------|
| `main.png` | **Primary view.** 3/4 perspective from front-right (elev=-28°, azim=145°). Best for visual inspection. |
| `side.png` | Side perspective view (elev=-22°, azim=215°). Shows depth and robot arm profile. |
| `top.png` / `topdown.png` | Top-down orthographic view (elev=-85°). Best for layout analysis and collision checking. |
| `close.png` / `closeup.png` | Close-up view (distance=0.7m). Shows object detail, mesh quality, and material properties. |
| `before_sim.png` | Scene before physics simulation. Objects at their initial placement. |
| `after_sim.png` / `after_physics.png` | Scene after 1000 physics steps. Verifies stability — if objects stayed in place, the layout is physically valid. |
| `floorplan.png` | 2D matplotlib floorplan with labeled objects and facing arrows (PhysScene-Bench only). |
| `mujoco_topdown.png` | MuJoCo rendered top-down with box proxies (PhysScene-Bench only). |
| `real_topdown.png` | MuJoCo rendered top-down with real Objaverse meshes. |
| `real_perspective.png` | MuJoCo rendered perspective with real meshes. |

## Pipelines

### 1. Procedural Generation (Zero-Cost)
```bash
# Generate tabletop environments (10K+/sec, no API needed)
python3 generators/tabletop_gen.py

# Generate room navigation environments (4K+/sec)
python3 generators/room_gen.py
```

### 2. LLM-in-the-Loop (API-Driven)
```bash
# LLM designs → MuJoCo renders → VLM reviews → LLM refines
OPENROUTER_API_KEY=... python3 self_iterate.py
```

Pipeline steps:
1. **LLM designs layout** (Claude Sonnet) → JSON with object positions
2. **Build MuJoCo XML** with real 3D assets + robot arm
3. **Physics simulation** (1000 steps gravity test)
4. **Render** 4 camera views (main, side, top, close)
5. **VLM reviews** (Qwen3-VL-30B, free) → scores 0-10 + feedback
6. **LLM revises** based on feedback → goto step 2
7. Repeat until score ≥ 8 or max iterations reached

### 3. Realistic Rendering (Axis-Corrected)
```bash
# With validated meshes and correct axis mapping
MUJOCO_GL=osmesa python3 fix_and_render.py
```

### 4. Piper Robot Grasp Planning
```bash
# Pick apple from table and place on plate (or vice versa)
MUJOCO_GL=osmesa python3 piper_grasp.py
```

## 3D Assets

### Tabletop Objects (15 categories, from Objaverse)

All objects are real 3D meshes (STL), NOT primitive shapes.

**Well-validated (distortion ratio < 2.0):**
plate, mug, bowl, apple, cup, can, pen, pencil, remote_control, box

**Usable but with some stretch:**
book, banana, bottle, dish, scissors

Each STL has:
- Unified coordinate system: bottom at Z=0, centered at XY origin
- Tested all 6 axis permutations for best match to target dimensions
- Per-object material: rgba, shininess, specular, friction

### Furniture (17 categories, from Objaverse)
sofa, armchair, bed, chair, desk, dining_table, coffee_table, cabinet, wardrobe, dresser, lamp, table_lamp, television_set, etc.

### Robot Arms
- **Simplified Panda** (built-in): 6-DOF + parallel gripper, defined in XML
- **Piper** (mujoco_menagerie): 6-DOF + finger gripper, position controlled

## Update Log

### 2026-03-08
- **v0.5: Piper Grasp Planning**
  - Integrated Piper arm from mujoco_menagerie
  - Pick-and-place task: apple ↔ plate
  - Motion planning via mocap body tracking

- **v0.4: Axis Fix + LLM Layout Design**
  - Fixed axis permutation for all tabletop STLs
  - Selected best variant per category (lowest distortion)
  - LLM (Claude Sonnet) designs compact, realistic layouts
  - 3-point lighting for better depth perception

- **v0.3: Realistic Tabletop with Real Meshes**
  - Downloaded 30+ Objaverse tabletop object STLs
  - 15 object categories with per-object materials
  - 6-DOF Panda-style arm with parallel gripper
  - Physics-validated: all objects stable after simulation

- **v0.2: LLM-in-the-Loop Self-Iteration**
  - LLM designs → render → VLM reviews → LLM refines
  - Qwen3-VL-30B (free) as visual judge
  - Iterative improvement: 6/10 → 6/10 → 7/10

- **v0.1: Procedural Generators**
  - TabletopGenerator: 10K+ envs/sec, 18 object types
  - RoomGenerator: 4K+ envs/sec, 14 furniture types
  - Built-in robot arm + mobile robot

### 2026-03-07
- **v0.0: PhysScene-Bench**
  - 10 layout prompts × 6 LLMs benchmark
  - 3-layer evaluation: semantic + physical + functional
  - VLM-as-Judge evaluation

## Requirements

```
mujoco >= 3.5.0
trimesh
numpy
Pillow
matplotlib
requests  # for LLM/VLM API calls
```

Environment: `MUJOCO_GL=osmesa` (headless rendering)
