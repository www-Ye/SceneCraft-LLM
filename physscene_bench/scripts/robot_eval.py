#!/usr/bin/env python3
"""
Robot Navigation Evaluation for MuJoCo scenes.
Tests navigation capability of a simple point-mass robot in generated room layouts.
"""

import json
import math
import numpy as np
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
import mujoco
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt

# Set MuJoCo to headless mode
os.environ['MUJOCO_GL'] = 'osmesa'

class RobotNavigator:
    """Simple point-mass robot for navigation evaluation in MuJoCo scenes."""
    
    def __init__(self, layout: List[Dict], room_size: List[float], 
                 robot_radius: float = 0.15):
        """
        Initialize robot navigator with scene layout.
        
        Args:
            layout: List of furniture objects with positions and dimensions
            room_size: [width, height] of the room in meters
            robot_radius: Radius of spherical robot in meters
        """
        self.layout = layout
        self.room_size = room_size
        self.robot_radius = robot_radius
        
        # Build obstacle list for navigation planning
        self.obstacles = self._build_obstacle_list()
        
        # MuJoCo model and data (will be created when needed)
        self.model = None
        self.data = None
        self._xml_path = None
    
    def _build_obstacle_list(self) -> List[Dict]:
        """Build list of obstacles from furniture layout."""
        obstacles = []
        
        # Add room walls as obstacles
        w, h = self.room_size
        wall_thickness = 0.1
        
        # Wall obstacles (x, y, width, height)
        walls = [
            {"pos": [0, -h/2 - wall_thickness/2], "size": [w + wall_thickness, wall_thickness]},  # Bottom
            {"pos": [0, h/2 + wall_thickness/2], "size": [w + wall_thickness, wall_thickness]},   # Top
            {"pos": [-w/2 - wall_thickness/2, 0], "size": [wall_thickness, h + wall_thickness]}, # Left
            {"pos": [w/2 + wall_thickness/2, 0], "size": [wall_thickness, h + wall_thickness]},  # Right
        ]
        
        obstacles.extend(walls)
        
        # Add furniture as obstacles
        for obj in self.layout:
            pos = obj["position"][:2]  # Only x, y
            dims = obj["dimensions"][:2]  # Only width, depth
            
            obstacles.append({
                "pos": pos,
                "size": dims,
                "name": obj.get("name", "furniture")
            })
        
        return obstacles
    
    def _build_mujoco_xml(self, start_pos: Tuple[float, float]) -> str:
        """Build MuJoCo XML with room, furniture, and robot."""
        xml_parts = [
            '<?xml version="1.0"?>',
            '<mujoco>',
            '  <option timestep="0.02" gravity="0 0 -9.81"/>',
            '  <asset>',
            '    <material name="floor_mat" rgba="0.8 0.8 0.8 1"/>',
            '    <material name="wall_mat" rgba="0.6 0.6 0.6 1"/>',
            '    <material name="furniture_mat" rgba="0.4 0.6 0.8 1"/>',
            '    <material name="robot_mat" rgba="1 0 0 0.8"/>',
            '  </asset>',
            '  <worldbody>',
            '    <!-- Floor -->',
            f'    <geom name="floor" type="box" size="{self.room_size[0]/2} {self.room_size[1]/2} 0.05"',
            '          pos="0 0 -0.05" material="floor_mat"/>',
        ]
        
        # Add walls
        w, h = self.room_size
        wall_height = 2.0
        wall_thickness = 0.1
        
        walls = [
            {"name": "wall_bottom", "pos": [0, -h/2 - wall_thickness/2, wall_height/2], 
             "size": [w/2 + wall_thickness/2, wall_thickness/2, wall_height/2]},
            {"name": "wall_top", "pos": [0, h/2 + wall_thickness/2, wall_height/2],
             "size": [w/2 + wall_thickness/2, wall_thickness/2, wall_height/2]},
            {"name": "wall_left", "pos": [-w/2 - wall_thickness/2, 0, wall_height/2],
             "size": [wall_thickness/2, h/2 + wall_thickness/2, wall_height/2]},
            {"name": "wall_right", "pos": [w/2 + wall_thickness/2, 0, wall_height/2],
             "size": [wall_thickness/2, h/2 + wall_thickness/2, wall_height/2]},
        ]
        
        for wall in walls:
            xml_parts.append(
                f'    <geom name="{wall["name"]}" type="box" '
                f'size="{wall["size"][0]} {wall["size"][1]} {wall["size"][2]}" '
                f'pos="{wall["pos"][0]} {wall["pos"][1]} {wall["pos"][2]}" '
                f'material="wall_mat"/>'
            )
        
        # Add furniture
        for i, obj in enumerate(self.layout):
            pos = obj["position"]
            dims = obj["dimensions"]
            name = obj.get("name", "furniture")
            
            # Place furniture on floor (adjust z position)
            furniture_z = dims[2] / 2
            
            xml_parts.append(
                f'    <geom name="furniture_{name}_{i}" type="box" '
                f'size="{dims[0]/2} {dims[1]/2} {dims[2]/2}" '
                f'pos="{pos[0]} {pos[1]} {furniture_z}" '
                f'material="furniture_mat"/>'
            )
        
        # Add robot
        robot_z = self.robot_radius
        xml_parts.extend([
            f'    <body name="robot" pos="{start_pos[0]} {start_pos[1]} {robot_z}">',
            '      <joint name="robot_x" type="slide" axis="1 0 0"/>',
            '      <joint name="robot_y" type="slide" axis="0 1 0"/>',
            f'      <geom name="robot_geom" type="sphere" size="{self.robot_radius}"',
            '            rgba="1 0 0 0.8" mass="1"/>',
            '    </body>',
        ])
        
        # Add actuators
        xml_parts.extend([
            '  </worldbody>',
            '  <actuator>',
            '    <motor joint="robot_x" ctrlrange="-5 5"/>',
            '    <motor joint="robot_y" ctrlrange="-5 5"/>',
            '  </actuator>',
            '</mujoco>'
        ])
        
        return '\n'.join(xml_parts)
    
    def _load_mujoco_scene(self, start_pos: Tuple[float, float]) -> bool:
        """Load MuJoCo scene for simulation."""
        try:
            # Build XML
            xml_content = self._build_mujoco_xml(start_pos)
            
            # Save to temporary file
            if self._xml_path and os.path.exists(self._xml_path):
                os.unlink(self._xml_path)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
                f.write(xml_content)
                self._xml_path = f.name
            
            # Load model
            self.model = mujoco.MjModel.from_xml_path(self._xml_path)
            self.data = mujoco.MjData(self.model)
            
            return True
            
        except Exception as e:
            print(f"Error loading MuJoCo scene: {e}")
            return False
    
    def compute_action(self, robot_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
        """
        Compute robot action using simple potential field navigation.
        
        Args:
            robot_pos: Current robot position [x, y]
            goal_pos: Goal position [x, y]
        
        Returns:
            Action forces [fx, fy]
        """
        # Attractive force toward goal
        goal_diff = goal_pos - robot_pos
        goal_distance = np.linalg.norm(goal_diff)
        
        if goal_distance < 1e-6:
            attract_force = np.zeros(2)
        else:
            attract_force = goal_diff / goal_distance * 3.0
        
        # Repulsive force from obstacles
        repulse_force = np.zeros(2)
        
        for obstacle in self.obstacles:
            obs_pos = np.array(obstacle["pos"])
            obs_size = np.array(obstacle["size"])
            
            # Calculate distance to obstacle (treating as rectangle)
            diff = robot_pos - obs_pos
            
            # Find closest point on obstacle rectangle
            closest_x = np.clip(diff[0], -obs_size[0]/2, obs_size[0]/2)
            closest_y = np.clip(diff[1], -obs_size[1]/2, obs_size[1]/2)
            closest_point = obs_pos + np.array([closest_x, closest_y])
            
            # Vector from closest point to robot
            to_robot = robot_pos - closest_point
            distance = np.linalg.norm(to_robot)
            
            # Effective distance considering robot radius
            effective_distance = distance - self.robot_radius
            
            # Apply repulsive force if within influence range
            influence_range = 0.5  # meters
            if effective_distance < influence_range and distance > 1e-6:
                repulse_magnitude = (influence_range - effective_distance) / (effective_distance**2 + 0.01)
                repulse_force += (to_robot / distance) * repulse_magnitude * 2.0
        
        # Combine forces
        total_force = attract_force + repulse_force
        
        # Limit force magnitude
        max_force = 5.0
        force_magnitude = np.linalg.norm(total_force)
        if force_magnitude > max_force:
            total_force = total_force / force_magnitude * max_force
        
        return total_force
    
    def navigate_to(self, start: Tuple[float, float], goal: Tuple[float, float], 
                   max_steps: int = 500) -> Dict[str, Any]:
        """
        Navigate from start to goal using potential field controller.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            max_steps: Maximum simulation steps
        
        Returns:
            Dictionary with navigation results
        """
        # Load MuJoCo scene
        if not self._load_mujoco_scene(start):
            return {
                "success": False,
                "error": "Failed to load MuJoCo scene",
                "path_length": 0,
                "collisions": 0,
                "steps": 0
            }
        
        try:
            # Get joint IDs
            robot_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'robot_x')
            robot_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'robot_y')
            robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'robot')
            
            # Initialize tracking variables
            path = [np.array(start)]
            collisions = 0
            goal_pos = np.array(goal)
            goal_threshold = 0.1  # 10cm
            
            # Simulation loop
            for step in range(max_steps):
                # Get current robot position
                robot_pos = np.array([
                    self.data.qpos[robot_x_id],
                    self.data.qpos[robot_y_id]
                ])
                
                path.append(robot_pos.copy())
                
                # Check if goal reached
                goal_distance = np.linalg.norm(robot_pos - goal_pos)
                if goal_distance < goal_threshold:
                    return {
                        "success": True,
                        "path_length": self._calculate_path_length(path),
                        "straight_line_distance": np.linalg.norm(np.array(goal) - np.array(start)),
                        "collisions": collisions,
                        "steps": step + 1,
                        "path": path
                    }
                
                # Compute control action
                action = self.compute_action(robot_pos, goal_pos)
                
                # Apply control
                self.data.ctrl[0] = action[0]  # x force
                self.data.ctrl[1] = action[1]  # y force
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                
                # Check for collisions (simplified: check if robot penetrates obstacles)
                if self._check_collision(robot_pos):
                    collisions += 1
            
            # Navigation failed (timeout)
            return {
                "success": False,
                "path_length": self._calculate_path_length(path),
                "straight_line_distance": np.linalg.norm(np.array(goal) - np.array(start)),
                "collisions": collisions,
                "steps": max_steps,
                "path": path,
                "reason": "timeout"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "path_length": 0,
                "collisions": 0,
                "steps": 0
            }
    
    def _calculate_path_length(self, path: List[np.ndarray]) -> float:
        """Calculate total path length."""
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(path)):
            total_length += np.linalg.norm(path[i] - path[i-1])
        
        return total_length
    
    def _check_collision(self, robot_pos: np.ndarray) -> bool:
        """Check if robot collides with any obstacle."""
        for obstacle in self.obstacles:
            obs_pos = np.array(obstacle["pos"])
            obs_size = np.array(obstacle["size"])
            
            # Check distance to obstacle rectangle
            diff = robot_pos - obs_pos
            closest_x = np.clip(diff[0], -obs_size[0]/2, obs_size[0]/2)
            closest_y = np.clip(diff[1], -obs_size[1]/2, obs_size[1]/2)
            closest_point = obs_pos + np.array([closest_x, closest_y])
            
            distance = np.linalg.norm(robot_pos - closest_point)
            
            # Collision if robot sphere intersects obstacle
            if distance < self.robot_radius:
                return True
        
        return False
    
    def _find_valid_positions(self, n_positions: int = 50) -> List[Tuple[float, float]]:
        """Find valid (collision-free) positions in the room."""
        valid_positions = []
        max_attempts = n_positions * 5
        
        # Room bounds (with some margin)
        margin = self.robot_radius + 0.1
        x_min, x_max = -self.room_size[0]/2 + margin, self.room_size[0]/2 - margin
        y_min, y_max = -self.room_size[1]/2 + margin, self.room_size[1]/2 - margin
        
        attempts = 0
        while len(valid_positions) < n_positions and attempts < max_attempts:
            # Random position
            pos = np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max)
            ])
            
            # Check if position is valid (no collision)
            if not self._check_collision(pos):
                valid_positions.append((pos[0], pos[1]))
            
            attempts += 1
        
        return valid_positions
    
    def evaluate_navigability(self, n_trials: int = 20) -> Dict[str, Any]:
        """
        Run multiple random start-goal navigation trials.
        
        Args:
            n_trials: Number of navigation trials
        
        Returns:
            Dictionary with aggregated navigation metrics
        """
        print(f"Evaluating navigability with {n_trials} trials...")
        
        # Find valid positions for start/goal points
        valid_positions = self._find_valid_positions(n_trials * 2 + 10)
        
        if len(valid_positions) < 4:
            return {
                "success_rate": 0.0,
                "avg_path_efficiency": 0.0,
                "avg_collisions": float('inf'),
                "avg_clearance": 0.0,
                "error": "Insufficient valid positions found",
                "trials": 0
            }
        
        results = []
        
        for trial in range(min(n_trials, len(valid_positions) // 2)):
            # Select start and goal positions
            start_idx = trial * 2
            goal_idx = trial * 2 + 1
            
            start = valid_positions[start_idx]
            goal = valid_positions[goal_idx]
            
            # Skip if start and goal are too close
            if np.linalg.norm(np.array(goal) - np.array(start)) < 0.5:
                continue
            
            print(f"  Trial {trial + 1}/{n_trials}: {start} -> {goal}")
            
            # Run navigation
            nav_result = self.navigate_to(start, goal)
            
            if "error" not in nav_result:
                # Calculate path efficiency
                path_efficiency = (nav_result["straight_line_distance"] / 
                                 (nav_result["path_length"] + 1e-6))
                
                # Calculate clearance score
                clearance_score = self._calculate_clearance_score(nav_result.get("path", []))
                
                results.append({
                    "success": nav_result["success"],
                    "path_efficiency": path_efficiency,
                    "collisions": nav_result["collisions"],
                    "clearance": clearance_score,
                    "steps": nav_result["steps"]
                })
        
        if not results:
            return {
                "success_rate": 0.0,
                "avg_path_efficiency": 0.0,
                "avg_collisions": 0.0,
                "avg_clearance": 0.0,
                "error": "No valid trials completed",
                "trials": 0
            }
        
        # Aggregate results
        successes = [r for r in results if r["success"]]
        
        return {
            "success_rate": len(successes) / len(results),
            "avg_path_efficiency": np.mean([r["path_efficiency"] for r in successes]) if successes else 0.0,
            "avg_collisions": np.mean([r["collisions"] for r in results]),
            "avg_clearance": np.mean([r["clearance"] for r in results]),
            "trials": len(results),
            "avg_steps": np.mean([r["steps"] for r in results])
        }
    
    def _calculate_clearance_score(self, path: List[np.ndarray]) -> float:
        """Calculate average clearance to nearest obstacle along path."""
        if not path:
            return 0.0
        
        clearances = []
        
        for pos in path[::5]:  # Sample every 5th point to reduce computation
            min_distance = float('inf')
            
            for obstacle in self.obstacles:
                obs_pos = np.array(obstacle["pos"])
                obs_size = np.array(obstacle["size"])
                
                # Distance to rectangle obstacle
                diff = pos - obs_pos
                closest_x = np.clip(diff[0], -obs_size[0]/2, obs_size[0]/2)
                closest_y = np.clip(diff[1], -obs_size[1]/2, obs_size[1]/2)
                closest_point = obs_pos + np.array([closest_x, closest_y])
                
                distance = np.linalg.norm(pos - closest_point)
                min_distance = min(min_distance, distance)
            
            clearances.append(min_distance)
        
        return np.mean(clearances) if clearances else 0.0
    
    def cleanup(self):
        """Clean up temporary files."""
        if self._xml_path and os.path.exists(self._xml_path):
            os.unlink(self._xml_path)

def test_with_mock_scene():
    """Test robot navigation with a mock living room scene."""
    print("Testing Robot Navigation with Mock Living Room Scene")
    print("=" * 55)
    
    # Mock living room layout (similar to P01 from layout_prompts.json)
    mock_layout = [
        {
            "name": "sofa",
            "position": [-1.5, -1.0, 0.425],
            "dimensions": [2.0, 0.9, 0.85],
            "rotation": 0
        },
        {
            "name": "television",
            "position": [1.5, -1.0, 0.35],
            "dimensions": [1.2, 0.2, 0.7],
            "rotation": 0
        },
        {
            "name": "coffee_table",
            "position": [0.0, -1.0, 0.2],
            "dimensions": [1.2, 0.6, 0.4],
            "rotation": 0
        },
        {
            "name": "armchair",
            "position": [-1.5, 1.0, 0.425],
            "dimensions": [0.8, 0.8, 0.85],
            "rotation": 0
        },
        {
            "name": "cabinet",
            "position": [1.5, 1.5, 0.9],
            "dimensions": [0.8, 0.4, 1.8],
            "rotation": 0
        }
    ]
    
    room_size = [5.0, 4.0]  # 5m x 4m living room
    
    # Initialize robot navigator
    navigator = RobotNavigator(mock_layout, room_size)
    
    try:
        # Test single navigation
        print("\n1. Testing single navigation...")
        start_pos = (-2.0, 0.0)
        goal_pos = (2.0, 0.0)
        
        nav_result = navigator.navigate_to(start_pos, goal_pos)
        
        print(f"Navigation result:")
        print(f"  Success: {nav_result.get('success', False)}")
        print(f"  Path length: {nav_result.get('path_length', 0):.3f}m")
        print(f"  Straight line: {nav_result.get('straight_line_distance', 0):.3f}m")
        print(f"  Collisions: {nav_result.get('collisions', 0)}")
        print(f"  Steps: {nav_result.get('steps', 0)}")
        
        if nav_result.get("success"):
            efficiency = (nav_result["straight_line_distance"] / 
                         nav_result["path_length"])
            print(f"  Path efficiency: {efficiency:.3f}")
        
        # Test full navigability evaluation
        print("\n2. Testing navigability evaluation...")
        eval_result = navigator.evaluate_navigability(n_trials=10)
        
        print(f"Navigability evaluation:")
        print(f"  Success rate: {eval_result['success_rate']:.3f}")
        print(f"  Avg path efficiency: {eval_result['avg_path_efficiency']:.3f}")
        print(f"  Avg collisions: {eval_result['avg_collisions']:.1f}")
        print(f"  Avg clearance: {eval_result['avg_clearance']:.3f}m")
        print(f"  Trials completed: {eval_result['trials']}")
        
        return eval_result
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return None
    
    finally:
        navigator.cleanup()

def main():
    """Main function for standalone testing."""
    # Check if we're in the right environment
    try:
        import mujoco
        print("MuJoCo available ✓")
    except ImportError:
        print("Error: MuJoCo not available")
        sys.exit(1)
    
    # Run test
    result = test_with_mock_scene()
    
    if result:
        print("\n" + "="*55)
        print("Robot Navigation Test Completed Successfully!")
    else:
        print("\n" + "="*55)
        print("Robot Navigation Test Failed!")

if __name__ == "__main__":
    main()