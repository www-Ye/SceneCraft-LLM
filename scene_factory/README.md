# SceneFactory

**Procedural Generation of Diverse MuJoCo Training Environments for Embodied AI**

Generate unlimited, physically valid MuJoCo environments for robot training — 
no GPU, no API, no cost. Pure CPU procedural generation.

## Features

- **TableTop Environments**: Randomized object arrangements for manipulation tasks
- **Room Environments**: Randomized furniture layouts for navigation tasks  
- **Real 3D Assets**: Objaverse meshes, not just boxes
- **Physics-Validated**: Every generated scene is MuJoCo-compatible
- **Task Integration**: Built-in robot + task definitions for immediate training
- **Infinite Diversity**: Procedural generation with configurable randomization
- **Curriculum Support**: Difficulty levels from simple to complex

## Quick Start

```bash
# Generate 100 tabletop environments
python3 generate.py --type tabletop --count 100 --output envs/tabletop/

# Generate 100 room navigation environments  
python3 generate.py --type room --count 100 --output envs/room/

# Train a robot arm on generated tabletop environments
python3 train_manipulation.py --env-dir envs/tabletop/

# Train navigation agent on generated rooms
python3 train_navigation.py --env-dir envs/room/
```
