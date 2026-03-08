#!/usr/bin/env python3
"""
High-Quality Tabletop Scene Generator (FIXED)

Fixes:
- Bug 2: Objects disappearing after physics - improved placement and physics stability
- Separated visual and collision meshes
- Better collision detection
- Velocity clamping during physics settling
"""
import os, json, math, tempfile
import numpy as np
from pathlib import Path
from PIL import Image
import logging

os.environ['MUJOCO_GL'] = 'osmesa'
import mujoco
import trimesh

from ..assets import AssetManager
from ..simulation import PhysicsEngine
from .xml_builder import MuJoCoXMLBuilder

logger = logging.getLogger(__name__)

class TabletopGenerator:
    """Generate tabletop scenes with physics simulation."""
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
        
        self.base_dir = Path(base_dir)
        self.asset_manager = AssetManager(base_dir)
        self.xml_builder = MuJoCoXMLBuilder(self.asset_manager)
        self.physics_engine = PhysicsEngine()
        
        self.output_dir = self.base_dir / "outputs" / "tabletop_scenes"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def check_collision_2d(self, new_pos, new_radius, placed, safety_margin=0.01):
        """Check if new object collides with any placed object (2D XY AABB)."""
        for pos, radius in placed:
            dx = abs(new_pos[0] - pos[0])
            dy = abs(new_pos[1] - pos[1])
            min_dist = new_radius + radius + safety_margin
            if dx < min_dist and dy < min_dist:
                return True
        return False
    
    def compute_mesh_bottom_z(self, mesh, scale, axis_permutation=None):
        """
        Compute the actual bottom Z coordinate of a mesh after scaling and axis permutation.
        
        BUGFIX: This addresses Bug 2 by computing the actual mesh bottom position
        to avoid initial penetration with the table.
        """
        vertices = mesh.vertices.copy()
        
        # Apply axis permutation if provided
        if axis_permutation and axis_permutation != [0, 1, 2]:
            vertices = vertices[:, axis_permutation]
        
        # Apply scaling
        vertices = vertices * np.array(scale)
        
        # Return the bottom Z coordinate
        return vertices[:, 2].min()
    
    def place_object_safely(self, obj_info, table_height, placed_objects, max_attempts=7):
        """
        Place an object safely on the table surface, avoiding collisions.
        
        BUGFIX: Computes actual mesh bottom position to prevent initial penetration.
        """
        cat = obj_info["cat"]
        dims = obj_info["dims"]
        pos = obj_info["pos"].copy()
        
        # Get mesh information
        scale_info, asset_info = self.asset_manager.get_mesh_scale(cat, dims)
        if scale_info is None:
            logger.warning(f"Cannot place {cat} - no mesh available")
            return None
        
        # Load visual mesh to compute bottom position
        visual_mesh = self.asset_manager.load_visual_mesh(cat)
        if visual_mesh is None:
            logger.warning(f"Cannot load mesh for {cat}")
            return None
        
        # Compute actual bottom Z after scaling
        mesh_bottom_z = self.compute_mesh_bottom_z(visual_mesh, scale_info["visual_scale"])
        
        # Calculate proper Z position (table surface - mesh bottom + small clearance)
        clearance = 0.005  # 5mm clearance
        z_position = table_height + 0.02 - mesh_bottom_z + clearance  # 2cm table thickness
        
        # Collision detection in 2D
        radius = max(dims[0], dims[1]) / 2
        original_pos = pos.copy()
        
        # Try original position first
        if not self.check_collision_2d(pos, radius, placed_objects):
            placed_objects.append((pos.copy(), radius))
            obj_info["final_pos"] = [pos[0], pos[1], z_position]
            return obj_info
        
        # Try small offsets if collision detected
        offset_attempts = [
            (0.03, 0), (-0.03, 0), (0, 0.03), (0, -0.03),
            (0.05, 0.02), (-0.05, 0.02), (0.02, -0.05), (-0.02, 0.05),
            (0.06, 0), (-0.06, 0), (0, 0.06), (0, -0.06)
        ]
        
        for dx, dy in offset_attempts:
            test_pos = [original_pos[0] + dx, original_pos[1] + dy]
            if not self.check_collision_2d(test_pos, radius, placed_objects):
                placed_objects.append((test_pos.copy(), radius))
                obj_info["final_pos"] = [test_pos[0], test_pos[1], z_position]
                logger.info(f"Placed {cat} with offset ({dx:.2f}, {dy:.2f})")
                return obj_info
        
        logger.warning(f"Could not place {cat} - no valid position found")
        return None
    
    def build_scene_xml(self, scene_config):
        """Build MuJoCo XML for a scene configuration."""
        objects = scene_config.get("objects", [])
        table_config = scene_config.get("table", {"w": 1.00, "d": 0.70, "h": 0.75})
        
        placed_objects = []  # (pos, radius) for collision tracking
        valid_objects = []
        
        # Process each object with safe placement
        for obj in objects:
            placed_obj = self.place_object_safely(obj, table_config["h"], placed_objects)
            if placed_obj:
                valid_objects.append(placed_obj)
        
        logger.info(f"Successfully placed {len(valid_objects)}/{len(objects)} objects")
        
        # Generate XML using the builder
        xml_str = self.xml_builder.build_xml(valid_objects, table_config)
        
        return xml_str, len(valid_objects)
    
    def render_scene_with_physics(self, xml_str, output_dir, scene_name, table_height=0.75, physics_steps=2000):
        """
        Render scene with physics simulation and stability checking.
        
        BUGFIX: Adds velocity clamping and object tracking to prevent physics explosions.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save XML file
        xml_path = output_path / f"{scene_name}_scene.xml"
        with open(xml_path, 'w') as f:
            f.write(xml_str)
        
        # Load and simulate
        try:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            
            renderer = mujoco.Renderer(model, width=1920, height=1440)
            
            # Initial render before physics
            self._render_views(renderer, data, output_path, f"{scene_name}_initial", table_height)
            
            # Physics simulation with velocity clamping and monitoring
            logger.info(f"Running physics simulation ({physics_steps} steps)...")
            fallen_objects = set()
            
            for step in range(physics_steps):
                # BUGFIX: Velocity clamping every 10 steps to prevent explosions
                if step % 10 == 0:
                    np.clip(data.qvel, -2.0, 2.0, out=data.qvel)
                
                mujoco.mj_step(model, data)
                
                # Check for objects that have fallen below table (every 50 steps)
                if step % 50 == 0 and step > 100:
                    for i in range(model.nbody):
                        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                        if body_name and body_name.startswith("obj_"):
                            z = data.xpos[i][2]
                            if z < table_height - 0.1 and body_name not in fallen_objects:
                                fallen_objects.add(body_name)
                                logger.warning(f"Object {body_name} fell below table at step {step} (z={z:.3f})")
            
            # Final render after physics
            self._render_views(renderer, data, output_path, f"{scene_name}_final", table_height)
            
            # Count final object stability
            stable_count = 0
            fallen_count = len(fallen_objects)
            
            for i in range(model.nbody):
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name and body_name.startswith("obj_"):
                    z = data.xpos[i][2]
                    if z >= table_height - 0.1:
                        stable_count += 1
            
            logger.info(f"Physics complete: {stable_count} stable, {fallen_count} fallen objects")
            
            renderer.close()
            
            return {"stable": stable_count, "fallen": fallen_count, "xml_path": str(xml_path)}
            
        except Exception as e:
            logger.error(f"Error during simulation: {e}")
            return {"stable": 0, "fallen": 0, "error": str(e)}
    
    def _render_views(self, renderer, data, output_path, prefix, table_height):
        """Render multiple camera views."""
        views = {
            "perspective": {"l": [0.02, 0.0, table_height+0.08], "d": 1.40, "e": -30, "a": 145},
            "front":       {"l": [0.0, 0.0, table_height+0.06], "d": 1.30, "e": -25, "a": 180},
            "closeup":     {"l": [0.05, 0.05, table_height+0.10], "d": 0.65, "e": -28, "a": 135},
            "topdown":     {"l": [0.0, 0.0, table_height+0.04], "d": 1.10, "e": -88, "a": 90},
            "side":        {"l": [0.0, 0.0, table_height+0.06], "d": 1.30, "e": -22, "a": 230},
        }
        
        for view_name, params in views.items():
            cam = mujoco.MjvCamera()
            cam.lookat[:] = params["l"]
            cam.distance = params["d"]
            cam.elevation = params["e"]
            cam.azimuth = params["a"]
            
            renderer.update_scene(data, cam)
            img = renderer.render()
            
            filename = f"{prefix}_{view_name}.png"
            Image.fromarray(img).save(str(output_path / filename))
    
    def generate_scene(self, scene_config, scene_name=None):
        """Generate a complete scene with physics simulation."""
        if scene_name is None:
            scene_name = scene_config.get("name", "scene")
        
        logger.info(f"Generating scene: {scene_name}")
        
        # Build XML
        xml_str, placed_count = self.build_scene_xml(scene_config)
        
        if placed_count == 0:
            logger.error("No objects could be placed in the scene")
            return None
        
        # Create output directory
        scene_output_dir = self.output_dir / scene_name
        
        # Render with physics
        results = self.render_scene_with_physics(
            xml_str, 
            scene_output_dir, 
            scene_name,
            scene_config.get("table", {}).get("h", 0.75)
        )
        
        # Save scene configuration
        config_path = scene_output_dir / f"{scene_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(scene_config, f, indent=2)
        
        results["output_dir"] = str(scene_output_dir)
        results["placed_objects"] = placed_count
        
        return results

# Predefined scene configurations
def breakfast_scene():
    """Dense breakfast scene - all objects on table surface, no stacking."""
    return {
        "name": "breakfast",
        "table": {"w": 1.00, "d": 0.70, "h": 0.75},
        "objects": [
            {"cat": "plate", "dims": [0.26, 0.26, 0.02], "pos": [0.05, 0.06],
             "color": "0.97 0.96 0.93 1", "shin": 0.50, "spec": 0.35},
            {"cat": "bowl", "dims": [0.15, 0.15, 0.08], "pos": [-0.18, 0.10],
             "color": "0.95 0.93 0.90 1", "shin": 0.45, "spec": 0.30},
            {"cat": "mug", "dims": [0.08, 0.08, 0.10], "pos": [0.26, 0.18], "rot": -15,
             "color": "0.82 0.80 0.76 1", "shin": 0.40, "spec": 0.25},
            {"cat": "cup", "dims": [0.07, 0.07, 0.08], "pos": [0.36, 0.06], "rot": 5,
             "color": "0.93 0.91 0.88 1", "shin": 0.50, "spec": 0.35},
            {"cat": "apple", "dims": [0.07, 0.07, 0.07], "pos": [0.20, 0.02],
             "color": "0.80 0.15 0.10 1", "shin": 0.55, "spec": 0.30},
            {"cat": "apple", "dims": [0.065, 0.065, 0.065], "pos": [-0.04, -0.06],
             "color": "0.25 0.65 0.15 1", "shin": 0.50, "spec": 0.28},
            {"cat": "can", "dims": [0.06, 0.06, 0.12], "pos": [0.38, -0.10],
             "color": "0.90 0.55 0.10 1", "shin": 0.65, "spec": 0.50},
            {"cat": "box", "dims": [0.12, 0.08, 0.18], "pos": [-0.35, 0.22], "rot": 8,
             "color": "0.85 0.70 0.20 1", "shin": 0.10, "spec": 0.05},
            {"cat": "pen", "dims": [0.01, 0.14, 0.01], "pos": [0.15, -0.16], "rot": 12,
             "color": "0.10 0.10 0.12 1", "shin": 0.50, "spec": 0.30},
        ]
    }

def study_desk_scene():
    """Study desk scene with books, stationery, and beverages."""
    return {
        "name": "study_desk",
        "table": {"w": 1.20, "d": 0.70, "h": 0.75},
        "objects": [
            {"cat": "book", "dims": [0.17, 0.24, 0.025], "pos": [0.0, 0.05],
             "color": "0.95 0.93 0.88 1", "shin": 0.10, "spec": 0.05},
            {"cat": "book", "dims": [0.15, 0.22, 0.03], "pos": [-0.28, 0.15], "rot": 3,
             "color": "0.20 0.30 0.60 1", "shin": 0.12, "spec": 0.06},
            {"cat": "mug", "dims": [0.08, 0.08, 0.10], "pos": [0.32, 0.18], "rot": -20,
             "color": "0.25 0.25 0.28 1", "shin": 0.40, "spec": 0.25},
            {"cat": "pen", "dims": [0.01, 0.14, 0.01], "pos": [0.22, -0.05], "rot": 2,
             "color": "0.08 0.08 0.10 1", "shin": 0.50, "spec": 0.30},
            {"cat": "pencil", "dims": [0.008, 0.17, 0.008], "pos": [0.24, -0.02], "rot": 5,
             "color": "0.82 0.72 0.12 1", "shin": 0.15, "spec": 0.08},
            {"cat": "apple", "dims": [0.07, 0.07, 0.07], "pos": [0.38, 0.04],
             "color": "0.78 0.12 0.10 1", "shin": 0.55, "spec": 0.30},
        ]
    }

def tea_ceremony_scene():
    """Traditional tea ceremony setup."""
    return {
        "name": "tea_ceremony",
        "table": {"w": 0.90, "d": 0.60, "h": 0.75},
        "objects": [
            {"cat": "mug", "dims": [0.10, 0.10, 0.10], "pos": [0.0, 0.08],
             "color": "0.40 0.25 0.15 1", "shin": 0.35, "spec": 0.20},
            {"cat": "cup", "dims": [0.06, 0.06, 0.06], "pos": [-0.16, -0.06],
             "color": "0.92 0.90 0.85 1", "shin": 0.50, "spec": 0.35},
            {"cat": "cup", "dims": [0.06, 0.06, 0.06], "pos": [0.16, -0.06],
             "color": "0.92 0.90 0.85 1", "shin": 0.50, "spec": 0.35},
            {"cat": "plate", "dims": [0.20, 0.20, 0.018], "pos": [0.0, 0.24],
             "color": "0.94 0.92 0.88 1", "shin": 0.45, "spec": 0.30},
            {"cat": "bowl", "dims": [0.12, 0.12, 0.07], "pos": [-0.25, 0.12],
             "color": "0.35 0.55 0.30 1", "shin": 0.30, "spec": 0.15},
        ]
    }