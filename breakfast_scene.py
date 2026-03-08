#!/usr/bin/env python3
"""
Generate a breakfast scene with 8-10 objects to showcase improved mesh quality.
"""
import os
os.environ['PYTHONPATH'] = '.'

from scenecraft.generation.tabletop import TabletopGenerator
import numpy as np

def create_breakfast_scene():
    """Create a breakfast scene with multiple objects."""
    
    generator = TabletopGenerator()
    
    # Create breakfast scene configuration
    breakfast_config = {
        "objects": [
            {"cat": "plate", "dims": [0.25, 0.25, 0.02], "pos": [0.0, 0.1]},
            {"cat": "mug", "dims": [0.08, 0.08, 0.10], "pos": [0.25, 0.15]},
            {"cat": "cup", "dims": [0.08, 0.08, 0.10], "pos": [-0.25, 0.15]},
            {"cat": "bowl", "dims": [0.15, 0.15, 0.08], "pos": [0.1, -0.15]},
            {"cat": "apple", "dims": [0.08, 0.08, 0.08], "pos": [-0.05, 0.25]},
            {"cat": "banana", "dims": [0.04, 0.18, 0.04], "pos": [0.15, 0.25]},
            {"cat": "book", "dims": [0.15, 0.22, 0.03], "pos": [-0.3, -0.1]},
            {"cat": "pen", "dims": [0.01, 0.15, 0.01], "pos": [-0.25, -0.25]},
            {"cat": "can", "dims": [0.06, 0.06, 0.12], "pos": [0.3, -0.15]},
        ],
        "table": {"w": 1.2, "d": 0.8, "h": 0.75}
    }
    
    # Generate the scene
    xml_str, placed_count = generator.build_scene_xml(breakfast_config)
    print(f"Generated breakfast scene with {placed_count} objects")
    
    # Render with physics
    output_dir = generator.output_dir
    generator.render_scene_with_physics(
        xml_str, output_dir, "breakfast", 
        table_height=0.75, physics_steps=3000
    )
    
    print(f"✅ Breakfast scene complete! Check: {output_dir}/breakfast")

if __name__ == "__main__":
    create_breakfast_scene()