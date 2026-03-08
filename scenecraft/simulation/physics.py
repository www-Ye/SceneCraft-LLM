#!/usr/bin/env python3
"""
Physics simulation engine for SceneCraft.

Provides physics settling with velocity clamping and stability monitoring.
"""
import numpy as np
import mujoco
import logging

logger = logging.getLogger(__name__)

class PhysicsEngine:
    """Handle physics simulation with stability monitoring."""
    
    def __init__(self):
        self.max_velocity = 2.0  # Maximum allowed velocity (m/s)
        self.clamp_interval = 10  # Clamp velocities every N steps
    
    def settle_scene(self, model, data, steps=2000, monitor_interval=50, table_height=0.75):
        """
        Settle scene with physics simulation.
        
        Returns:
            dict: Results including stable/fallen object counts
        """
        logger.info(f"Starting physics settling ({steps} steps)")
        
        fallen_objects = set()
        results = {
            "stable": 0,
            "fallen": 0,
            "steps_completed": 0,
            "fallen_objects": []
        }
        
        try:
            for step in range(steps):
                # Velocity clamping to prevent explosions
                if step % self.clamp_interval == 0:
                    np.clip(data.qvel, -self.max_velocity, self.max_velocity, out=data.qvel)
                
                # Step physics
                mujoco.mj_step(model, data)
                
                # Monitor object stability
                if step % monitor_interval == 0 and step > 100:
                    self._check_fallen_objects(model, data, fallen_objects, table_height, step)
                
                results["steps_completed"] = step + 1
            
            # Final stability check
            stable_count, fallen_count = self._count_stable_objects(model, data, table_height)
            
            results.update({
                "stable": stable_count,
                "fallen": fallen_count,
                "fallen_objects": list(fallen_objects)
            })
            
            logger.info(f"Physics complete: {stable_count} stable, {fallen_count} fallen")
            
        except Exception as e:
            logger.error(f"Physics simulation error: {e}")
            results["error"] = str(e)
        
        return results
    
    def _check_fallen_objects(self, model, data, fallen_objects, table_height, step):
        """Check for objects that have fallen below the table."""
        for i in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and body_name.startswith("obj_"):
                z = data.xpos[i][2]
                if z < table_height - 0.1 and body_name not in fallen_objects:
                    fallen_objects.add(body_name)
                    logger.warning(f"Object {body_name} fell at step {step} (z={z:.3f})")
    
    def _count_stable_objects(self, model, data, table_height):
        """Count stable vs fallen objects."""
        stable_count = 0
        fallen_count = 0
        
        for i in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and body_name.startswith("obj_"):
                z = data.xpos[i][2]
                if z >= table_height - 0.1:
                    stable_count += 1
                else:
                    fallen_count += 1
        
        return stable_count, fallen_count
    
    def check_stability(self, model, data, table_height=0.75, velocity_threshold=0.01):
        """
        Check if the scene has reached a stable state.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            table_height: Height of table surface
            velocity_threshold: Max velocity to consider stable
        
        Returns:
            bool: True if scene is stable
        """
        # Check object velocities
        max_velocity = np.max(np.abs(data.qvel))
        if max_velocity > velocity_threshold:
            return False
        
        # Check if any objects are falling
        for i in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and body_name.startswith("obj_"):
                z = data.xpos[i][2]
                if z < table_height - 0.05:  # Object is below table
                    return False
        
        return True