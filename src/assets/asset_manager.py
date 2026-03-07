"""
Asset Manager for SceneCraft-LLM
Manages 3D furniture assets from Objaverse dataset.
Handles loading, normalization, and selection of assets.
"""

import json
import os
import trimesh
import numpy as np
from typing import Dict, List, Optional, Tuple


# Standard furniture dimensions in meters (width, depth, height)
STANDARD_DIMENSIONS = {
    'sofa':           {'width': 2.0,  'depth': 0.9,  'height': 0.85},
    'armchair':       {'width': 0.9,  'depth': 0.85, 'height': 0.85},
    'coffee_table':   {'width': 1.2,  'depth': 0.6,  'height': 0.45},
    'dining_table':   {'width': 1.6,  'depth': 0.9,  'height': 0.75},
    'desk':           {'width': 1.4,  'depth': 0.7,  'height': 0.75},
    'chair':          {'width': 0.5,  'depth': 0.5,  'height': 0.9},
    'bed':            {'width': 1.6,  'depth': 2.0,  'height': 0.6},
    'dresser':        {'width': 1.2,  'depth': 0.5,  'height': 0.8},
    'cabinet':        {'width': 0.8,  'depth': 0.45, 'height': 1.8},
    'wardrobe':       {'width': 1.2,  'depth': 0.6,  'height': 2.0},
    'table_lamp':     {'width': 0.3,  'depth': 0.3,  'height': 0.5},
    'lamp':           {'width': 0.4,  'depth': 0.4,  'height': 1.6},
    'television_set': {'width': 1.2,  'depth': 0.1,  'height': 0.7},
    'stool':          {'width': 0.4,  'depth': 0.4,  'height': 0.6},
    'bench':          {'width': 1.2,  'depth': 0.4,  'height': 0.45},
    'runner_(carpet)':{'width': 2.0,  'depth': 3.0,  'height': 0.01},
    'nightstand':     {'width': 0.5,  'depth': 0.4,  'height': 0.6},
    'bookshelf':      {'width': 0.8,  'depth': 0.3,  'height': 1.8},
    'rug':            {'width': 2.5,  'depth': 3.5,  'height': 0.01},
}

# Mapping from common names to catalog categories
CATEGORY_ALIASES = {
    'couch': 'sofa',
    'tv': 'television_set',
    'television': 'television_set',
    'side_table': 'coffee_table',
    'end_table': 'coffee_table',
    'bookcase': 'cabinet',
    'nightstand': 'dresser',
    'floor_lamp': 'lamp',
    'carpet': 'runner_(carpet)',
    'rug': 'runner_(carpet)',
    'dining_chair': 'chair',
    'office_chair': 'chair',
    'accent_chair': 'armchair',
}


class AssetManager:
    """Manages 3D furniture assets for scene generation."""
    
    def __init__(self, asset_dir: str = None, catalog_path: str = None):
        """
        Initialize the asset manager.
        
        Args:
            asset_dir: Path to asset directory containing GLB files
            catalog_path: Path to asset_catalog.json
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        if asset_dir is None:
            asset_dir = os.path.join(project_root, 'data', 'assets')
        if catalog_path is None:
            catalog_path = os.path.join(project_root, 'data', 'asset_catalog.json')
        
        self.asset_dir = asset_dir
        self.catalog = self._load_catalog(catalog_path)
        self._mesh_cache = {}
    
    def _load_catalog(self, catalog_path: str) -> Dict:
        """Load asset catalog from JSON."""
        if not os.path.exists(catalog_path):
            print(f"Warning: catalog not found at {catalog_path}")
            return {}
        with open(catalog_path) as f:
            return json.load(f)
    
    def get_categories(self) -> List[str]:
        """Get available asset categories."""
        return list(self.catalog.keys())
    
    def resolve_category(self, name: str) -> str:
        """Resolve a furniture name to a catalog category."""
        name_lower = name.lower().strip()
        # Direct match
        if name_lower in self.catalog:
            return name_lower
        # Alias match
        if name_lower in CATEGORY_ALIASES:
            alias = CATEGORY_ALIASES[name_lower]
            if alias in self.catalog:
                return alias
        # Fuzzy match: check if any catalog category contains the name
        for cat in self.catalog:
            if name_lower in cat or cat in name_lower:
                return cat
        return name_lower
    
    def get_assets(self, category: str) -> List[Dict]:
        """Get all assets for a category."""
        resolved = self.resolve_category(category)
        return self.catalog.get(resolved, [])
    
    def select_asset(self, category: str, index: int = 0) -> Optional[Dict]:
        """Select an asset from a category by index."""
        assets = self.get_assets(category)
        if not assets:
            return None
        return assets[index % len(assets)]
    
    def get_standard_dimensions(self, category: str) -> Dict[str, float]:
        """Get standard real-world dimensions for a furniture category."""
        resolved = self.resolve_category(category)
        if resolved in STANDARD_DIMENSIONS:
            return STANDARD_DIMENSIONS[resolved]
        # Default dimensions
        return {'width': 0.8, 'depth': 0.8, 'height': 0.8}
    
    def load_mesh(self, category: str, index: int = 0, 
                  normalize: bool = True) -> Optional[trimesh.Scene]:
        """
        Load a 3D mesh for a furniture category.
        
        Args:
            category: Furniture category name
            index: Which asset to select from the category
            normalize: Whether to normalize to standard dimensions
            
        Returns:
            trimesh.Scene or trimesh.Trimesh object
        """
        asset = self.select_asset(category, index)
        if asset is None:
            return None
        
        file_path = os.path.join(self.asset_dir, asset['file'])
        
        # Check cache
        cache_key = (file_path, normalize)
        if cache_key in self._mesh_cache:
            return self._mesh_cache[cache_key]
        
        if not os.path.exists(file_path):
            print(f"Warning: asset file not found: {file_path}")
            return None
        
        try:
            mesh = trimesh.load(file_path, force='scene')
            
            if normalize:
                mesh = self._normalize_mesh(mesh, category)
            
            self._mesh_cache[cache_key] = mesh
            return mesh
        except Exception as e:
            print(f"Error loading mesh {file_path}: {e}")
            return None
    
    def _normalize_mesh(self, mesh: trimesh.Scene, category: str) -> trimesh.Scene:
        """
        Normalize a mesh to standard real-world dimensions.
        Centers at origin and scales to standard size.
        Handles arbitrary axis orientations by sorting axes to match W/D/H.
        """
        std_dims = self.get_standard_dimensions(category)
        
        # Get current bounds
        bounds = mesh.bounding_box.bounds
        current_size = bounds[1] - bounds[0]
        current_center = (bounds[0] + bounds[1]) / 2
        
        # Center first
        transform = np.eye(4)
        transform[:3, 3] = -current_center
        mesh.apply_transform(transform)
        
        # Determine axis mapping: sort current axes to match target W/D/H
        # Standard convention: X=width, Y=depth, Z=height
        target = np.array([std_dims['width'], std_dims['depth'], std_dims['height']])
        
        # Sort both target and current by size to find best mapping
        target_order = np.argsort(target)  # ascending
        current_order = np.argsort(current_size)  # ascending
        
        # Non-uniform scale: map each current axis to corresponding target axis
        current_size = np.maximum(current_size, 1e-6)
        
        scale = np.ones(3)
        for t_idx, c_idx in zip(target_order, current_order):
            scale[c_idx] = target[t_idx] / current_size[c_idx]
        
        scale_mat = np.eye(4)
        scale_mat[0, 0] = scale[0]
        scale_mat[1, 1] = scale[1]
        scale_mat[2, 2] = scale[2]
        
        mesh.apply_transform(scale_mat)
        
        # Re-center after scaling (ensure bottom at z=0)
        bounds = mesh.bounding_box.bounds
        adjust = np.eye(4)
        adjust[2, 3] = -bounds[0][2]  # lift so bottom is at z=0
        mesh.apply_transform(adjust)
        
        return mesh
    
    def export_for_mujoco(self, category: str, index: int = 0,
                          output_dir: str = None,
                          max_faces: int = 100000) -> Optional[Dict]:
        """
        Export a mesh as STL for MuJoCo consumption.
        Loads raw mesh, converts to single Trimesh, normalizes, simplifies, exports.
        
        Args:
            category: Furniture category
            index: Asset index
            output_dir: Where to save STL files
            max_faces: Maximum faces allowed (MuJoCo limit ~200000)
            
        Returns:
            Dict with 'stl_path', 'dimensions', 'category'
        """
        if output_dir is None:
            output_dir = os.path.join(self.asset_dir, '_mujoco_stl')
        os.makedirs(output_dir, exist_ok=True)
        
        asset = self.select_asset(category, index)
        if asset is None:
            return None
        
        file_path = os.path.join(self.asset_dir, asset['file'])
        if not os.path.exists(file_path):
            return None
        
        resolved = self.resolve_category(category)
        safe_name = resolved.replace('(', '').replace(')', '').replace(' ', '_')
        stl_name = f"{safe_name}_{index}.stl"
        stl_path = os.path.join(output_dir, stl_name)
        
        try:
            # Load raw mesh fresh (don't use cache to avoid in-place transform issues)
            raw = trimesh.load(file_path, force='scene')
            
            # Convert scene to single Trimesh
            if isinstance(raw, trimesh.Scene):
                meshes = [g for g in raw.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if not meshes:
                    return None
                combined = trimesh.util.concatenate(meshes)
            else:
                combined = raw.copy()
            
            # Normalize: center, scale to standard dimensions, bottom at z=0
            std_dims = self.get_standard_dimensions(category)
            bounds = combined.bounding_box.bounds
            current_size = bounds[1] - bounds[0]
            current_center = (bounds[0] + bounds[1]) / 2
            
            # Center at origin
            combined.apply_translation(-current_center)
            
            # Map axes: sort by size to match target width/depth/height
            target = np.array([std_dims['width'], std_dims['depth'], std_dims['height']])
            target_order = np.argsort(target)
            current_order = np.argsort(current_size)
            current_size = np.maximum(current_size, 1e-6)
            
            scale = np.ones(3)
            for t_idx, c_idx in zip(target_order, current_order):
                scale[c_idx] = target[t_idx] / current_size[c_idx]
            
            combined.apply_scale(scale)
            
            # Move bottom to z=0
            new_bounds = combined.bounds
            combined.apply_translation([0, 0, -new_bounds[0][2]])
            
            # Simplify if too many faces
            if len(combined.faces) > max_faces:
                combined = combined.simplify_quadric_decimation(face_count=max_faces)
            
            combined.export(stl_path)
            
        except Exception as e:
            print(f"  Warning: STL export failed for {category}: {e}")
            return None
        
        # Get final dimensions
        final_bounds = combined.bounds
        size = final_bounds[1] - final_bounds[0]
        
        return {
            'stl_path': stl_path,
            'category': resolved,
            'dimensions': {
                'width': round(float(size[0]), 3),
                'depth': round(float(size[1]), 3),
                'height': round(float(size[2]), 3),
            }
        }
    
    def generate_mujoco_xml_body(self, category: str, position: Tuple[float, float, float],
                                  rotation: float = 0.0, index: int = 0,
                                  name: str = None) -> Optional[str]:
        """
        Generate MuJoCo XML body element for a furniture piece using real mesh.
        
        Args:
            category: Furniture category
            position: (x, y, z) position
            rotation: Rotation angle in degrees around Z axis
            index: Asset index
            name: Optional body name
            
        Returns:
            XML string for the MuJoCo body element
        """
        export = self.export_for_mujoco(category, index)
        if export is None:
            # Fallback to box primitive
            dims = self.get_standard_dimensions(category)
            if name is None:
                name = category
            return f'''    <body name="{name}" pos="{position[0]:.3f} {position[1]:.3f} {position[2]:.3f}">
      <geom type="box" size="{dims['width']/2:.3f} {dims['depth']/2:.3f} {dims['height']/2:.3f}" 
            rgba="0.6 0.4 0.2 1" mass="10"/>
    </body>'''
        
        dims = export['dimensions']
        stl_path = os.path.abspath(export['stl_path'])
        
        if name is None:
            name = category
        
        rot_rad = np.radians(rotation)
        quat = [np.cos(rot_rad/2), 0, 0, np.sin(rot_rad/2)]  # Z-axis rotation
        quat_str = ' '.join(f'{q:.4f}' for q in quat)
        
        # Position the object so its bottom sits at the given height
        z_pos = position[2] + dims['height'] / 2
        
        return f'''    <body name="{name}" pos="{position[0]:.3f} {position[1]:.3f} {z_pos:.3f}" quat="{quat_str}">
      <geom type="mesh" mesh="{name}_mesh" rgba="0.6 0.5 0.3 1" mass="10"/>
    </body>'''

    def get_mujoco_asset_xml(self, category: str, index: int = 0,
                              name: str = None) -> Optional[str]:
        """Generate MuJoCo asset (mesh) XML element."""
        export = self.export_for_mujoco(category, index)
        if export is None:
            return None
        
        if name is None:
            name = category
        
        stl_path = os.path.abspath(export['stl_path'])
        return f'    <mesh name="{name}_mesh" file="{stl_path}"/>'


def print_catalog_summary(asset_dir=None, catalog_path=None):
    """Print a summary of available assets."""
    mgr = AssetManager(asset_dir, catalog_path)
    print("=== SceneCraft-LLM Asset Catalog ===\n")
    
    total_count = 0
    for cat in sorted(mgr.get_categories()):
        assets = mgr.get_assets(cat)
        std = mgr.get_standard_dimensions(cat)
        print(f"  {cat}: {len(assets)} models | standard: "
              f"{std['width']:.1f}x{std['depth']:.1f}x{std['height']:.1f}m")
        total_count += len(assets)
    
    print(f"\n  Total: {total_count} models across {len(mgr.get_categories())} categories")
    print(f"  Source: Objaverse (LVIS subset)")


if __name__ == '__main__':
    print_catalog_summary()
