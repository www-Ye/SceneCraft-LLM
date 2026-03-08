#!/usr/bin/env python3
"""
Asset manager for SceneCraft - handles loading, scaling, and managing 3D assets.
"""

import json
import trimesh
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AssetManager:
    """Manages 3D assets for scene generation."""
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
        
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.assets_dir = self.data_dir / "assets"
        self.stl_dir = self.assets_dir / "_mujoco_stl"
        self.catalog_path = self.data_dir / "tabletop_asset_catalog.json"
        
        self.catalog = self._load_catalog()
    
    def _load_catalog(self):
        """Load asset catalog."""
        if not self.catalog_path.exists():
            logger.warning(f"Catalog not found: {self.catalog_path}")
            return {}
        
        with open(self.catalog_path) as f:
            catalog = json.load(f)
        
        logger.info(f"Loaded catalog with {len(catalog)} categories")
        return catalog
    
    def get_available_categories(self):
        """Get list of available asset categories."""
        return list(self.catalog.keys())
    
    def get_asset_info(self, category, index=0):
        """Get asset information for a category."""
        if category not in self.catalog:
            return None
        
        assets = self.catalog[category]
        if index >= len(assets):
            return None
        
        return assets[index]
    
    def load_visual_mesh(self, category, index=0):
        """Load visual mesh for an asset."""
        info = self.get_asset_info(category, index)
        if not info:
            return None
        
        # Try new format first (with separate visual/collision files)
        if "visual_file" in info:
            visual_path = self.stl_dir / info["visual_file"]
        else:
            # Fallback to old format
            visual_path = self.stl_dir / info["file"]
        
        if not visual_path.exists():
            logger.warning(f"Visual mesh not found: {visual_path}")
            return None
        
        return trimesh.load(str(visual_path))
    
    def load_collision_mesh(self, category, index=0):
        """Load collision mesh for an asset."""
        info = self.get_asset_info(category, index)
        if not info:
            return None
        
        # Try new format first (with separate visual/collision files)
        if "collision_file" in info:
            collision_path = self.stl_dir / info["collision_file"]
            if collision_path.exists():
                return trimesh.load(str(collision_path))
        
        # Fallback: create collision mesh from visual mesh
        visual_mesh = self.load_visual_mesh(category, index)
        if visual_mesh is None:
            return None
        
        return visual_mesh.convex_hull
    
    def get_mesh_scale(self, category, target_dims, index=0):
        """Calculate scale factors for a mesh to match target dimensions."""
        info = self.get_asset_info(category, index)
        if not info:
            return None, None
        
        # Load visual mesh to get current dimensions
        visual_mesh = self.load_visual_mesh(category, index)
        if visual_mesh is None:
            return None, None
        
        current_dims = visual_mesh.bounds[1] - visual_mesh.bounds[0]
        target_arr = np.array(target_dims)
        
        # Calculate scale factors
        scale_factors = target_arr / np.maximum(current_dims, 0.001)
        
        # Get visual and collision mesh paths
        visual_path = None
        collision_path = None
        
        if "visual_file" in info:
            visual_path = str(self.stl_dir / info["visual_file"])
            collision_path = str(self.stl_dir / info["collision_file"])
        else:
            visual_path = str(self.stl_dir / info["file"])
            collision_path = visual_path
        
        return {
            "visual_scale": scale_factors.tolist(),
            "collision_scale": scale_factors.tolist(),
            "visual_mesh": visual_path,
            "collision_mesh": collision_path,
            "mass": max(np.prod(target_arr) * 1200, 0.01)
        }, info
    
    def get_material_properties(self, category):
        """Get material properties for a category."""
        # Material properties per category
        material_props = {
            # Ceramics - white/cream, slight shininess
            "mug":    {"rgba": "0.92 0.90 0.88 1", "shininess": "0.4", "specular": "0.3", "friction": "0.6 0.005 0.001"},
            "cup":    {"rgba": "0.95 0.93 0.90 1", "shininess": "0.5", "specular": "0.3", "friction": "0.6 0.005 0.001"},
            "bowl":   {"rgba": "0.93 0.91 0.87 1", "shininess": "0.4", "specular": "0.25", "friction": "0.6 0.005 0.001"},
            "plate":  {"rgba": "0.97 0.96 0.94 1", "shininess": "0.5", "specular": "0.3", "friction": "0.6 0.005 0.001"},
            "dish":   {"rgba": "0.96 0.95 0.92 1", "shininess": "0.5", "specular": "0.3", "friction": "0.6 0.005 0.001"},
            # Glass - transparent look
            "glass":  {"rgba": "0.85 0.90 0.95 0.6", "shininess": "0.8", "specular": "0.6", "friction": "0.4 0.005 0.001"},
            "bottle": {"rgba": "0.25 0.55 0.25 0.75", "shininess": "0.7", "specular": "0.5", "friction": "0.4 0.005 0.001"},
            # Metal - shiny
            "can":    {"rgba": "0.75 0.18 0.18 1", "shininess": "0.6", "specular": "0.5", "friction": "0.5 0.005 0.001"},
            "scissors": {"rgba": "0.72 0.72 0.75 1", "shininess": "0.7", "specular": "0.6", "friction": "0.5 0.005 0.001"},
            # Food - organic colors
            "apple":  {"rgba": "0.82 0.12 0.10 1", "shininess": "0.5", "specular": "0.3", "friction": "0.7 0.005 0.001"},
            "banana": {"rgba": "0.95 0.88 0.22 1", "shininess": "0.3", "specular": "0.15", "friction": "0.7 0.005 0.001"},
            "orange": {"rgba": "0.95 0.60 0.10 1", "shininess": "0.3", "specular": "0.15", "friction": "0.7 0.005 0.001"},
            # Paper/wood
            "book":   {"rgba": "0.22 0.32 0.62 1", "shininess": "0.15", "specular": "0.05", "friction": "0.8 0.005 0.001"},
            "box":    {"rgba": "0.60 0.42 0.22 1", "shininess": "0.1", "specular": "0.05", "friction": "0.7 0.005 0.001"},
            "pen":    {"rgba": "0.12 0.12 0.14 1", "shininess": "0.5", "specular": "0.3", "friction": "0.5 0.005 0.001"},
            "pencil": {"rgba": "0.85 0.75 0.15 1", "shininess": "0.2", "specular": "0.1", "friction": "0.6 0.005 0.001"},
            "remote_control": {"rgba": "0.15 0.15 0.18 1", "shininess": "0.4", "specular": "0.2", "friction": "0.5 0.005 0.001"},
        }
        
        default_props = {"rgba": "0.6 0.6 0.6 1", "shininess": "0.3", "specular": "0.2", "friction": "0.6 0.005 0.001"}
        return material_props.get(category, default_props)