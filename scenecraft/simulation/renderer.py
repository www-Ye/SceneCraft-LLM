#!/usr/bin/env python3
"""
Rendering utilities for SceneCraft.

Handles high-quality rendering with multiple camera views.
"""
import os
import mujoco
import numpy as np
from PIL import Image
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Renderer:
    """High-quality scene renderer with multiple views."""
    
    def __init__(self, width=1920, height=1440):
        self.width = width
        self.height = height
    
    def render_scene(self, model, data, output_dir, scene_name, table_height=0.75):
        """
        Render scene with multiple camera views.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data 
            output_dir: Output directory for images
            scene_name: Name for output files
            table_height: Height of table surface
        
        Returns:
            dict: Rendered view information
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        renderer = mujoco.Renderer(model, width=self.width, height=self.height)
        
        # Define camera views
        views = self._get_camera_views(table_height)
        rendered_views = {}
        
        try:
            for view_name, params in views.items():
                # Setup camera
                cam = mujoco.MjvCamera()
                cam.lookat[:] = params["lookat"]
                cam.distance = params["distance"]
                cam.elevation = params["elevation"]
                cam.azimuth = params["azimuth"]
                
                # Render
                renderer.update_scene(data, cam)
                img = renderer.render()
                
                # Save image
                filename = f"{scene_name}_{view_name}.png"
                filepath = output_path / filename
                Image.fromarray(img).save(str(filepath))
                
                rendered_views[view_name] = str(filepath)
                logger.info(f"Rendered {view_name} -> {filename}")
            
        finally:
            renderer.close()
        
        return rendered_views
    
    def _get_camera_views(self, table_height):
        """Define standard camera views for tabletop scenes."""
        return {
            "perspective": {
                "lookat": [0.02, 0.0, table_height + 0.08],
                "distance": 1.40,
                "elevation": -30,
                "azimuth": 145
            },
            "front": {
                "lookat": [0.0, 0.0, table_height + 0.06],
                "distance": 1.30,
                "elevation": -25,
                "azimuth": 180
            },
            "closeup": {
                "lookat": [0.05, 0.05, table_height + 0.10],
                "distance": 0.65,
                "elevation": -28,
                "azimuth": 135
            },
            "topdown": {
                "lookat": [0.0, 0.0, table_height + 0.04],
                "distance": 1.10,
                "elevation": -88,
                "azimuth": 90
            },
            "side": {
                "lookat": [0.0, 0.0, table_height + 0.06],
                "distance": 1.30,
                "elevation": -22,
                "azimuth": 230
            }
        }
    
    def render_single_view(self, model, data, output_path, view_params):
        """
        Render a single camera view.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            output_path: Output file path
            view_params: Camera parameters dict
        """
        renderer = mujoco.Renderer(model, width=self.width, height=self.height)
        
        try:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = view_params["lookat"]
            cam.distance = view_params["distance"] 
            cam.elevation = view_params["elevation"]
            cam.azimuth = view_params["azimuth"]
            
            renderer.update_scene(data, cam)
            img = renderer.render()
            
            Image.fromarray(img).save(str(output_path))
            logger.info(f"Rendered single view -> {output_path}")
            
        finally:
            renderer.close()
    
    def create_video_frames(self, model, data, output_dir, scene_name, 
                           num_frames=60, table_height=0.75):
        """
        Create frames for video animation by rotating camera around scene.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            output_dir: Output directory
            scene_name: Base name for frames
            num_frames: Number of frames to generate
            table_height: Table height
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        renderer = mujoco.Renderer(model, width=self.width, height=self.height)
        frame_paths = []
        
        try:
            for frame in range(num_frames):
                # Calculate rotation angle
                azimuth = (frame / num_frames) * 360
                
                # Setup camera
                cam = mujoco.MjvCamera()
                cam.lookat[:] = [0.0, 0.0, table_height + 0.08]
                cam.distance = 1.5
                cam.elevation = -25
                cam.azimuth = azimuth
                
                # Render frame
                renderer.update_scene(data, cam)
                img = renderer.render()
                
                # Save frame
                filename = f"{scene_name}_frame_{frame:03d}.png"
                filepath = output_path / filename
                Image.fromarray(img).save(str(filepath))
                frame_paths.append(str(filepath))
            
            logger.info(f"Created {num_frames} animation frames")
            
        finally:
            renderer.close()
        
        return frame_paths