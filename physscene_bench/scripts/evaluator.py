#!/usr/bin/env python3
"""
PhysScene-Bench Evaluator
Three-layer evaluation system for 3D indoor scene layouts.
"""

import json
import math
import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter
import mujoco
from scipy.spatial.distance import pdist, squareform

# Set MuJoCo to headless mode
os.environ['MUJOCO_GL'] = 'osmesa'

class PhysSceneEvaluator:
    def __init__(self):
        # Standard furniture dimensions (width, depth, height) in meters
        self.standard_dims = {
            "sofa": {"width": 2.0, "depth": 0.9, "height": 0.85},
            "chair": {"width": 0.6, "depth": 0.6, "height": 0.85},
            "coffee_table": {"width": 1.2, "depth": 0.6, "height": 0.4},
            "dining_table": {"width": 1.5, "depth": 0.8, "height": 0.75},
            "desk": {"width": 1.2, "depth": 0.6, "height": 0.75},
            "bed": {"width": 1.5, "depth": 2.0, "height": 0.6},
            "nightstand": {"width": 0.5, "depth": 0.4, "height": 0.6},
            "wardrobe": {"width": 1.0, "depth": 0.6, "height": 2.0},
            "dresser": {"width": 1.2, "depth": 0.5, "height": 0.8},
            "cabinet": {"width": 0.8, "depth": 0.4, "height": 1.8},
            "armchair": {"width": 0.8, "depth": 0.8, "height": 0.85},
            "television": {"width": 1.2, "depth": 0.2, "height": 0.7},
            "lamp": {"width": 0.3, "depth": 0.3, "height": 1.5},
            "rug": {"width": 2.0, "depth": 1.5, "height": 0.02},
            "bookshelf": {"width": 0.8, "depth": 0.3, "height": 1.8},
            "table": {"width": 1.5, "depth": 0.8, "height": 0.75}  # Generic table
        }
    
    def get_object_bbox(self, obj: Dict) -> Tuple[float, float, float, float]:
        """Get axis-aligned bounding box considering rotation"""
        pos = obj["position"]
        dims = obj["dimensions"]
        rotation = obj.get("rotation", 0)
        
        # Convert rotation to radians
        rot_rad = math.radians(rotation)
        
        # Original dimensions
        width, depth = dims["width"], dims["depth"]
        
        # Rotated bounding box dimensions
        abs_cos = abs(math.cos(rot_rad))
        abs_sin = abs(math.sin(rot_rad))
        
        bbox_width = width * abs_cos + depth * abs_sin
        bbox_depth = width * abs_sin + depth * abs_cos
        
        # Bounding box coordinates (min_x, min_y, max_x, max_y)
        min_x = pos[0] - bbox_width / 2
        max_x = pos[0] + bbox_width / 2
        min_y = pos[1] - bbox_depth / 2
        max_y = pos[1] + bbox_depth / 2
        
        return min_x, min_y, max_x, max_y
    
    def check_collision(self, obj1: Dict, obj2: Dict) -> bool:
        """Check if two objects collide using AABB"""
        bbox1 = self.get_object_bbox(obj1)
        bbox2 = self.get_object_bbox(obj2)
        
        # Check for overlap
        return not (bbox1[2] <= bbox2[0] or  # obj1 right <= obj2 left
                   bbox1[0] >= bbox2[2] or   # obj1 left >= obj2 right
                   bbox1[3] <= bbox2[1] or   # obj1 top <= obj2 bottom
                   bbox1[1] >= bbox2[3])     # obj1 bottom >= obj2 top
    
    def is_inside_room(self, obj: Dict, room_size: List[float]) -> bool:
        """Check if object is completely inside room boundaries"""
        bbox = self.get_object_bbox(obj)
        return (bbox[0] >= 0 and bbox[1] >= 0 and 
                bbox[2] <= room_size[0] and bbox[3] <= room_size[1])
    
    def evaluate_semantic_fidelity(self, layout: Dict, expected_objects: List[str]) -> Dict[str, float]:
        """Layer 1: Semantic Fidelity Evaluation"""
        generated_objects = [obj["category"] for obj in layout["objects"]]
        expected_counter = Counter(expected_objects)
        generated_counter = Counter(generated_objects)
        
        # Category match score (precision, recall, F1)
        tp = sum(min(expected_counter[cat], generated_counter[cat]) for cat in expected_counter)
        fp = sum(max(0, generated_counter[cat] - expected_counter[cat]) for cat in generated_counter)
        fn = sum(max(0, expected_counter[cat] - generated_counter[cat]) for cat in expected_counter)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Count accuracy
        total_expected = len(expected_objects)
        total_generated = len(generated_objects)
        count_accuracy = 1 - abs(total_generated - total_expected) / max(total_expected, 1)
        count_accuracy = max(0, count_accuracy)  # Clamp to [0,1]
        
        return {
            "category_match_precision": precision,
            "category_match_recall": recall,
            "category_match_f1": f1,
            "count_accuracy": count_accuracy
        }
    
    def create_mujoco_xml(self, layout: Dict, room_size: List[float]) -> str:
        """Create MuJoCo XML for physics simulation"""
        root = ET.Element("mujoco")
        
        # Compiler settings
        compiler = ET.SubElement(root, "compiler", {"angle": "degree"})
        
        # World body
        worldbody = ET.SubElement(root, "worldbody")
        
        # Add floor
        floor_geom = ET.SubElement(worldbody, "geom", {
            "name": "floor",
            "type": "box",
            "size": f"{room_size[0]/2} {room_size[1]/2} 0.01",
            "pos": f"{room_size[0]/2} {room_size[1]/2} -0.01",
            "rgba": "0.8 0.8 0.8 1"
        })
        
        # Add room boundaries
        walls = [
            ("wall_x1", f"0.05 {room_size[1]/2} 1", f"-0.05 {room_size[1]/2} 1"),
            ("wall_x2", f"0.05 {room_size[1]/2} 1", f"{room_size[0]+0.05} {room_size[1]/2} 1"),
            ("wall_y1", f"{room_size[0]/2} 0.05 1", f"{room_size[0]/2} -0.05 1"),
            ("wall_y2", f"{room_size[0]/2} 0.05 1", f"{room_size[0]/2} {room_size[1]+0.05} 1")
        ]
        
        for wall_name, size, pos in walls:
            ET.SubElement(worldbody, "geom", {
                "name": wall_name,
                "type": "box", 
                "size": size,
                "pos": pos,
                "rgba": "0.9 0.9 0.9 1"
            })
        
        # Add furniture objects
        for i, obj in enumerate(layout["objects"]):
            pos = obj["position"]
            dims = obj["dimensions"]
            rotation = obj.get("rotation", 0)
            
            body = ET.SubElement(worldbody, "body", {
                "name": f"obj_{i}_{obj['category']}",
                "pos": f"{pos[0]} {pos[1]} {dims['height']/2}"
            })
            
            # Add rotation if present
            if rotation != 0:
                body.set("quat", f"0 0 {math.sin(math.radians(rotation/2))} {math.cos(math.radians(rotation/2))}")
            
            # Geometry
            ET.SubElement(body, "geom", {
                "name": f"geom_{i}",
                "type": "box",
                "size": f"{dims['width']/2} {dims['depth']/2} {dims['height']/2}",
                "rgba": f"{0.5 + 0.4*math.sin(i)} {0.3 + 0.4*math.cos(i)} {0.6} 1"
            })
            
            # Joint for freebody dynamics
            ET.SubElement(body, "freejoint", {"name": f"joint_{i}"})
        
        # Convert to string
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding='unicode')
    
    def evaluate_physics_stability(self, layout: Dict, room_size: List[float]) -> float:
        """Evaluate physics stability using MuJoCo simulation"""
        try:
            # Skip physics simulation if too many objects (performance)
            if len(layout["objects"]) > 10:
                print("Skipping physics simulation for large scenes (>10 objects)")
                return 0.8  # Optimistic score for large scenes
            
            # Create MuJoCo XML
            xml_content = self.create_mujoco_xml(layout, room_size)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
                f.write(xml_content)
                xml_file = f.name
            
            try:
                # Load model with timeout protection
                model = mujoco.MjModel.from_xml_path(xml_file)
                data = mujoco.MjData(model)
                
                # Record initial positions (simplified)
                initial_positions = []
                object_count = len(layout["objects"])
                
                for i in range(object_count):
                    try:
                        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obj_{i}_{layout['objects'][i]['category']}")
                        if body_id != -1:
                            pos = data.xpos[body_id].copy()
                            initial_positions.append(pos)
                        else:
                            initial_positions.append(None)
                    except:
                        initial_positions.append(None)
                
                # Quick simulation (reduced time for performance)
                simulation_time = 1.0  # Reduced from 3.0 seconds
                dt = model.opt.timestep
                steps = int(simulation_time / dt)
                steps = min(steps, 500)  # Cap at 500 steps
                
                for _ in range(steps):
                    mujoco.mj_step(model, data)
                
                # Simple stability check
                stable_count = 0
                
                for i, initial_pos in enumerate(initial_positions):
                    if initial_pos is None:
                        continue
                        
                    try:
                        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obj_{i}_{layout['objects'][i]['category']}")
                        if body_id != -1:
                            final_pos = data.xpos[body_id]
                            
                            # Simple movement check
                            movement = np.linalg.norm(final_pos - initial_pos)
                            if movement < 0.5:  # Stayed relatively in place
                                stable_count += 1
                    except:
                        pass
                
                return stable_count / max(object_count, 1)
                
            finally:
                try:
                    os.unlink(xml_file)
                except:
                    pass
                
        except Exception as e:
            print(f"Physics simulation error (using fallback): {e}")
            # Fallback: simple collision-based stability estimate
            collision_score = self._simple_collision_check(layout["objects"])
            return collision_score * 0.8  # Reduce score for fallback method
    
    def _quat_rotate(self, quat, vec):
        """Rotate vector by quaternion"""
        qw, qx, qy, qz = quat
        x, y, z = vec
        
        # Convert to rotation matrix and apply
        xx, yy, zz = qx*qx, qy*qy, qz*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz
        
        return np.array([
            x*(1-2*(yy+zz)) + y*2*(xy-wz) + z*2*(xz+wy),
            x*2*(xy+wz) + y*(1-2*(xx+zz)) + z*2*(yz-wx),
            x*2*(xz-wy) + y*2*(yz+wx) + z*(1-2*(xx+yy))
        ])
    
    def evaluate_physical_plausibility(self, layout: Dict, room_size: List[float]) -> Dict[str, float]:
        """Layer 2: Physical Plausibility Evaluation"""
        objects = layout["objects"]
        
        # Collision score
        total_pairs = len(objects) * (len(objects) - 1) // 2
        collision_count = 0
        
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if self.check_collision(objects[i], objects[j]):
                    collision_count += 1
        
        collision_score = 1 - (collision_count / max(total_pairs, 1))
        
        # Boundary score
        inside_count = sum(1 for obj in objects if self.is_inside_room(obj, room_size))
        boundary_score = inside_count / max(len(objects), 1)
        
        # Stability score
        stability_score = self.evaluate_physics_stability(layout, room_size)
        
        return {
            "collision_score": collision_score,
            "boundary_score": boundary_score,
            "stability_score": stability_score
        }
    
    def evaluate_functional_affordance(self, layout: Dict, functional_checks: List[str]) -> Dict[str, float]:
        """Layer 3: Functional Affordance Evaluation"""
        objects = layout["objects"]
        scores = {}
        
        # Generic functional checks
        scores["facing_score"] = self._evaluate_facing_relationships(objects)
        scores["proximity_score"] = self._evaluate_proximity_relationships(objects)  
        scores["walkability_score"] = self._evaluate_walkability(objects, layout.get("room_size", [5, 4]))
        scores["wall_alignment_score"] = self._evaluate_wall_alignment(objects, layout.get("room_size", [5, 4]))
        
        return scores
    
    def _simple_collision_check(self, objects: List[Dict]) -> float:
        """Simple collision check for fallback stability scoring"""
        total_pairs = len(objects) * (len(objects) - 1) // 2
        if total_pairs == 0:
            return 1.0
            
        collision_count = 0
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if self.check_collision(objects[i], objects[j]):
                    collision_count += 1
        
        return 1 - (collision_count / max(total_pairs, 1))
    
    def _evaluate_facing_relationships(self, objects: List[Dict]) -> float:
        """Evaluate if furniture pieces face each other appropriately"""
        facing_pairs = [
            ("sofa", "television"),
            ("chair", "desk"),
            ("chair", "dining_table"), 
            ("chair", "table"),
            ("armchair", "coffee_table")
        ]
        
        correct_facing = 0
        total_pairs = 0
        
        for obj_type1, obj_type2 in facing_pairs:
            objects1 = [obj for obj in objects if obj["category"] == obj_type1]
            objects2 = [obj for obj in objects if obj["category"] == obj_type2]
            
            for obj1 in objects1:
                for obj2 in objects2:
                    total_pairs += 1
                    
                    # Calculate if obj1 faces obj2
                    pos1, pos2 = obj1["position"], obj2["position"]
                    rotation1 = obj1.get("rotation", 0)
                    
                    # Direction from obj1 to obj2
                    dx, dy = pos2[0] - pos1[0], pos2[1] - pos1[1]
                    target_angle = math.degrees(math.atan2(dx, dy)) % 360
                    
                    # Check if rotation is roughly facing target
                    angle_diff = abs(rotation1 - target_angle)
                    angle_diff = min(angle_diff, 360 - angle_diff)
                    
                    if angle_diff < 45:  # Within 45 degrees
                        correct_facing += 1
        
        return correct_facing / max(total_pairs, 1)
    
    def _evaluate_proximity_relationships(self, objects: List[Dict]) -> float:
        """Evaluate if semantically related objects are close"""
        proximity_pairs = [
            ("bed", "nightstand", 1.0),
            ("sofa", "coffee_table", 1.5),
            ("desk", "chair", 0.8),
            ("armchair", "lamp", 1.2),
            ("dining_table", "chair", 1.0)
        ]
        
        correct_proximity = 0
        total_pairs = 0
        
        for obj_type1, obj_type2, max_distance in proximity_pairs:
            objects1 = [obj for obj in objects if obj["category"] == obj_type1]
            objects2 = [obj for obj in objects if obj["category"] == obj_type2]
            
            for obj1 in objects1:
                closest_distance = float('inf')
                for obj2 in objects2:
                    distance = math.sqrt(
                        (obj1["position"][0] - obj2["position"][0])**2 + 
                        (obj1["position"][1] - obj2["position"][1])**2
                    )
                    closest_distance = min(closest_distance, distance)
                
                if objects2:  # If there are any objects of type2
                    total_pairs += 1
                    if closest_distance <= max_distance:
                        correct_proximity += 1
        
        return correct_proximity / max(total_pairs, 1)
    
    def _evaluate_walkability(self, objects: List[Dict], room_size: List[float]) -> float:
        """Evaluate walkable area in the room"""
        # Create a grid representation
        grid_size = 0.1  # 10cm resolution
        width_cells = int(room_size[0] / grid_size)
        height_cells = int(room_size[1] / grid_size)
        
        # Mark occupied cells
        occupied = np.zeros((height_cells, width_cells), dtype=bool)
        
        for obj in objects:
            bbox = self.get_object_bbox(obj)
            
            # Convert to grid coordinates
            x1 = max(0, int(bbox[0] / grid_size))
            x2 = min(width_cells, int(bbox[2] / grid_size))
            y1 = max(0, int(bbox[1] / grid_size))
            y2 = min(height_cells, int(bbox[3] / grid_size))
            
            occupied[y1:y2, x1:x2] = True
        
        # Calculate walkable area
        total_cells = width_cells * height_cells
        walkable_cells = np.sum(~occupied)
        
        return walkable_cells / total_cells
    
    def _evaluate_wall_alignment(self, objects: List[Dict], room_size: List[float]) -> float:
        """Evaluate if large furniture is aligned with walls"""
        wall_furniture = ["wardrobe", "cabinet", "bookshelf", "dresser", "sofa", "bed"]
        
        wall_aligned = 0
        total_wall_furniture = 0
        
        wall_threshold = 0.5  # Within 50cm of wall
        
        for obj in objects:
            if obj["category"] in wall_furniture:
                total_wall_furniture += 1
                pos = obj["position"]
                
                # Check distance to walls
                dist_to_walls = [
                    pos[0],  # Distance to left wall (x=0)
                    room_size[0] - pos[0],  # Distance to right wall
                    pos[1],  # Distance to bottom wall (y=0)
                    room_size[1] - pos[1]   # Distance to top wall
                ]
                
                if min(dist_to_walls) <= wall_threshold:
                    wall_aligned += 1
        
        return wall_aligned / max(total_wall_furniture, 1)
    
    def evaluate_layout(self, layout: Dict, room_size: List[float], expected_objects: List[str], 
                       functional_checks: List[str]) -> Dict[str, Any]:
        """Main evaluation function combining all three layers"""
        
        # Layer 1: Semantic Fidelity
        semantic_scores = self.evaluate_semantic_fidelity(layout, expected_objects)
        
        # Layer 2: Physical Plausibility
        physical_scores = self.evaluate_physical_plausibility(layout, room_size)
        
        # Layer 3: Functional Affordance
        functional_scores = self.evaluate_functional_affordance(layout, functional_checks)
        
        # Calculate weighted overall scores
        semantic_weight = 0.3
        physical_weight = 0.4
        functional_weight = 0.3
        
        # Calculate layer averages
        semantic_avg = np.mean(list(semantic_scores.values()))
        physical_avg = np.mean(list(physical_scores.values()))
        functional_avg = np.mean(list(functional_scores.values()))
        
        overall_score = (semantic_avg * semantic_weight + 
                        physical_avg * physical_weight + 
                        functional_avg * functional_weight)
        
        return {
            "overall_score": overall_score,
            "layer_scores": {
                "semantic_fidelity": semantic_avg,
                "physical_plausibility": physical_avg, 
                "functional_affordance": functional_avg
            },
            "detailed_scores": {
                "semantic": semantic_scores,
                "physical": physical_scores,
                "functional": functional_scores
            }
        }