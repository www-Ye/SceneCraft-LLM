# SceneCraft-LLM

AI-driven 3D scene generation with physics simulation for NeurIPS paper submission.

## 🚀 Quick Start

```bash
# Generate a predefined scene
python scripts/generate_scene.py --scene breakfast

# Run a quick demo
python scripts/render_demo.py

# Generate from custom config
python scripts/generate_scene.py --config configs/my_scene.json
```

## 📁 Project Structure

```
SceneCraft-LLM/
├── scenecraft/            # Main package
│   ├── assets/            # Asset management + download
│   ├── generation/        # Scene generation
│   ├── simulation/        # Physics + rendering  
│   ├── layout/            # Layout planning (future)
│   ├── evaluation/        # Metrics (future)
│   └── utils/             # Utilities
├── scripts/               # Runnable scripts
├── benchmark/             # PhysScene benchmark
├── configs/               # Configuration files
└── data/                  # Asset catalogs (binary assets gitignored)
```

## 🛠 Installation

1. **Prerequisites:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Assets:**
   ```python
   from scenecraft.assets.download import TabletopAssetDownloader
   downloader = TabletopAssetDownloader()
   downloader.download_assets()
   ```

## 🎨 Scene Generation

### Predefined Scenes

- `breakfast`: Dense breakfast table setup  
- `study_desk`: Academic workspace with books and stationery
- `tea_ceremony`: Traditional tea service arrangement

### Custom Scenes

Create a JSON configuration:

```json
{
  "name": "my_scene",
  "table": {"w": 1.0, "d": 0.7, "h": 0.75},
  "objects": [
    {"cat": "mug", "dims": [0.08, 0.08, 0.10], "pos": [0.0, 0.0]},
    {"cat": "plate", "dims": [0.25, 0.25, 0.02], "pos": [0.2, 0.1]}
  ]
}
```

## 🐛 Bug Fixes (v2)

### Fixed Bug 1: Convex Hull Destroying Mesh Quality
- **Problem:** Non-watertight meshes (bowls, mugs) converted to convex hulls, destroying concave details
- **Solution:** 
  - Preserve original geometry for visual rendering
  - Use separate collision meshes (simplified)
  - Gentle mesh repair (fill holes, fix normals) instead of convex hull

### Fixed Bug 2: Objects Disappearing After Physics  
- **Problem:** Objects placed with initial penetration get launched by physics
- **Solution:**
  - Compute actual mesh bottom Z coordinate after scaling
  - Velocity clamping during physics settling
  - Safety checks for fallen objects

## 🔧 Key Improvements

- **Separate Visual/Collision Meshes:** High-detail visuals + simplified physics
- **Better Physics Stability:** Velocity clamping + object tracking
- **Cleaner Architecture:** Modular design with clear separation of concerns
- **Improved Asset Management:** Smart mesh loading and scaling

## 📊 Evaluation

Run the PhysScene benchmark:

```bash
python scripts/run_benchmark.py
# Or directly:
python benchmark/scripts/[specific_benchmark].py
```

## 🏗 Development

### Architecture

- `AssetManager`: Handles 3D asset loading, scaling, materials
- `TabletopGenerator`: Main scene generation with collision detection  
- `PhysicsEngine`: Stable physics simulation with monitoring
- `Renderer`: Multi-view rendering with standard camera positions
- `XMLBuilder`: Builds MuJoCo XML with separate visual/collision geoms

### Adding New Objects

1. Add assets to the download script
2. Define material properties in `AssetManager`
3. Objects automatically available in scene generation

## 📄 License

MIT License - see LICENSE file.

## 📚 Citation

If you use SceneCraft-LLM in your research, please cite:

```bibtex
@inproceedings{scenecraft2024,
  title={SceneCraft-LLM: AI-Driven 3D Scene Generation with Physics Simulation},
  author={...},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

---

For questions or issues, please open a GitHub issue.