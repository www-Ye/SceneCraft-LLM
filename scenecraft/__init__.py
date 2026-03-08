"""
SceneCraft-LLM: AI-driven 3D scene generation with physics simulation.

This package provides tools for:
- Asset management and download from Objaverse
- Layout planning with LLM-based constraint solving
- Scene generation with MuJoCo XML output
- Physics simulation and stability checks
- Evaluation metrics for scene quality

Author: SceneCraft Team
"""

__version__ = "0.1.0"

from .assets import AssetManager
from .generation import TabletopGenerator
from .simulation import PhysicsEngine, Renderer

__all__ = [
    "AssetManager",
    "TabletopGenerator", 
    "PhysicsEngine",
    "Renderer",
]