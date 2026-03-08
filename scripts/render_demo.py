#!/usr/bin/env python3
"""
Quick demo script for SceneCraft rendering.

Generates a simple scene and renders multiple views.
"""
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scenecraft.generation.tabletop import TabletopGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Generate and render a quick demo scene."""
    logger.info("Starting SceneCraft demo...")
    
    # Simple demo scene
    demo_scene = {
        "name": "demo",
        "table": {"w": 0.80, "d": 0.60, "h": 0.75},
        "objects": [
            {"cat": "plate", "dims": [0.20, 0.20, 0.02], "pos": [0.0, 0.0]},
            {"cat": "mug", "dims": [0.08, 0.08, 0.10], "pos": [0.15, 0.12]},
            {"cat": "apple", "dims": [0.07, 0.07, 0.07], "pos": [-0.10, 0.05]},
            {"cat": "book", "dims": [0.15, 0.22, 0.03], "pos": [0.0, -0.15]},
        ]
    }
    
    # Generate scene
    generator = TabletopGenerator()
    results = generator.generate_scene(demo_scene)
    
    if results:
        logger.info(f"✅ Demo complete! Check: {results['output_dir']}")
        logger.info(f"📊 {results['stable']} stable, {results['fallen']} fallen objects")
    else:
        logger.error("❌ Demo failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())