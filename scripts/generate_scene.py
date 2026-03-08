#!/usr/bin/env python3
"""
Main scene generation script for SceneCraft.

Usage:
    python scripts/generate_scene.py --scene breakfast
    python scripts/generate_scene.py --config configs/my_scene.json
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scenecraft.generation.tabletop import TabletopGenerator, breakfast_scene, study_desk_scene, tea_ceremony_scene
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PREDEFINED_SCENES = {
    "breakfast": breakfast_scene,
    "study_desk": study_desk_scene, 
    "tea_ceremony": tea_ceremony_scene
}

def main():
    parser = argparse.ArgumentParser(description="Generate tabletop scenes with SceneCraft")
    parser.add_argument("--scene", type=str, choices=list(PREDEFINED_SCENES.keys()),
                       help="Generate a predefined scene")
    parser.add_argument("--config", type=str, 
                       help="Path to scene configuration JSON file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (optional)")
    parser.add_argument("--physics-steps", type=int, default=2000,
                       help="Number of physics simulation steps")
    
    args = parser.parse_args()
    
    if not args.scene and not args.config:
        parser.error("Must specify either --scene or --config")
    
    # Load scene configuration
    if args.scene:
        logger.info(f"Generating predefined scene: {args.scene}")
        scene_config = PREDEFINED_SCENES[args.scene]()
    else:
        logger.info(f"Loading scene from config: {args.config}")
        with open(args.config) as f:
            scene_config = json.load(f)
    
    # Initialize generator
    generator = TabletopGenerator()
    
    # Override output directory if specified
    if args.output:
        generator.output_dir = Path(args.output)
    
    # Generate scene
    try:
        results = generator.generate_scene(scene_config)
        
        if results:
            logger.info("Scene generation completed successfully!")
            logger.info(f"Output directory: {results['output_dir']}")
            logger.info(f"Objects placed: {results['placed_objects']}")
            logger.info(f"Stable objects: {results['stable']}")
            logger.info(f"Fallen objects: {results['fallen']}")
            
            if results['fallen'] > 0:
                logger.warning("Some objects fell during physics simulation")
        else:
            logger.error("Scene generation failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error during scene generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()