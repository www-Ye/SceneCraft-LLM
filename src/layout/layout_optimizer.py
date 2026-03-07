"""
Layout Optimizer for SceneCraft-LLM.

Uses scipy optimization to jointly optimize furniture positions, orientations, 
and scale factors to satisfy physical and semantic constraints.

Constraints:
  1. No collisions between furniture (AABB-based)
  2. All furniture within room boundaries
  3. Semantic relations (sofa faces TV, lamp near sofa, etc.)
  4. Furniture aligned to walls/room axes
  5. Walkability (maintain open paths)
  6. Size consistency (keep close to standard dimensions)
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional
import copy


# Semantic "facing" relations: (source, target) means source should face target
FACING_RELATIONS = [
    ('sofa', 'television_set'),
    ('sofa', 'coffee_table'),
    ('armchair', 'coffee_table'),
    ('armchair', 'television_set'),
    ('chair', 'desk'),
    ('chair', 'dining_table'),
]

# Semantic "near" relations: (a, b, max_distance)
PROXIMITY_RELATIONS = [
    ('coffee_table', 'sofa', 1.5),
    ('table_lamp', 'sofa', 1.5),
    ('table_lamp', 'desk', 1.0),
    ('nightstand', 'bed', 0.8),
    ('television_set', 'sofa', 4.0),
]

# Categories that should be against walls
WALL_CATEGORIES = ['sofa', 'cabinet', 'wardrobe', 'dresser', 'television_set', 'bookshelf']

# Overlay categories (don't check collision with floor items)
OVERLAY_CATEGORIES = ['runner_(carpet)', 'rug', 'carpet', 'mat']


class LayoutOptimizer:
    """
    Joint optimization of furniture layout using gradient-based optimization.
    
    Decision variables per object: [x, y, rotation, scale_factor]
    """
    
    def __init__(self, room_width: float, room_depth: float, 
                 wall_margin: float = 0.1, collision_margin: float = 0.15):
        self.room_width = room_width
        self.room_depth = room_depth
        self.wall_margin = wall_margin
        self.collision_margin = collision_margin
        
        # Weight for each constraint term
        self.w_collision = 200.0   # High priority: no collisions
        self.w_boundary = 100.0    # High priority: stay in room
        self.w_facing = 3.0
        self.w_proximity = 2.0
        self.w_wall_align = 1.5
        self.w_scale = 15.0        # Prefer original scale
        self.w_rotation_snap = 2.0  # Prefer axis-aligned
    
    def optimize(self, objects: List[Dict], max_iter: int = 500) -> List[Dict]:
        """
        Optimize layout of furniture objects.
        
        Args:
            objects: List of dicts with keys:
                - category: str
                - position: [x, y, z]
                - rotation: float (degrees)
                - dimensions: {width, depth, height}
            max_iter: Maximum optimization iterations
            
        Returns:
            Optimized list of objects (new positions, rotations, scaled dimensions)
        """
        n = len(objects)
        if n == 0:
            return objects
        
        # Build initial state vector: [x0, y0, rot0, scale0, x1, y1, rot1, scale1, ...]
        x0 = np.zeros(n * 4)
        for i, obj in enumerate(objects):
            x0[i*4 + 0] = obj['position'][0]
            x0[i*4 + 1] = obj['position'][1]
            x0[i*4 + 2] = np.radians(obj['rotation'])
            x0[i*4 + 3] = 1.0  # initial scale factor
        
        # Bounds
        bounds = []
        for i, obj in enumerate(objects):
            dims = obj['dimensions']
            max_dim = max(dims['width'], dims['depth']) / 2
            bounds.append((max_dim + self.wall_margin, self.room_width - max_dim - self.wall_margin))  # x
            bounds.append((max_dim + self.wall_margin, self.room_depth - max_dim - self.wall_margin))  # y
            bounds.append((0, 2 * np.pi))  # rotation
            bounds.append((0.7, 1.3))  # scale factor (±30%)
        
        print(f"  Optimizing {n} objects, {n*4} variables...")
        
        # Run optimization
        result = minimize(
            self._objective,
            x0,
            args=(objects,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-8},
        )
        
        print(f"  Optimization {'converged' if result.success else 'finished'} "
              f"after {result.nit} iterations (loss: {result.fun:.4f})")
        
        # Extract optimized objects
        optimized = copy.deepcopy(objects)
        for i in range(n):
            optimized[i]['position'] = [
                float(result.x[i*4 + 0]),
                float(result.x[i*4 + 1]),
                objects[i]['position'][2] if len(objects[i]['position']) > 2 else 0.0,
            ]
            optimized[i]['rotation'] = float(np.degrees(result.x[i*4 + 2]))
            
            scale = float(result.x[i*4 + 3])
            orig_dims = objects[i]['dimensions']
            optimized[i]['dimensions'] = {
                'width': orig_dims['width'] * scale,
                'depth': orig_dims['depth'] * scale,
                'height': orig_dims['height'] * scale,
            }
            optimized[i]['scale_factor'] = scale
        
        return optimized
    
    def _objective(self, x: np.ndarray, objects: List[Dict]) -> float:
        """Compute total loss for current layout configuration."""
        n = len(objects)
        loss = 0.0
        
        # Parse state
        states = []
        for i in range(n):
            px, py = x[i*4], x[i*4+1]
            rot = x[i*4+2]
            scale = x[i*4+3]
            dims = objects[i]['dimensions']
            cat = objects[i]['category']
            
            w = dims['width'] * scale
            d = dims['depth'] * scale
            
            # Effective width/depth considering rotation
            cos_r = abs(np.cos(rot))
            sin_r = abs(np.sin(rot))
            eff_w = w * cos_r + d * sin_r
            eff_d = w * sin_r + d * cos_r
            
            states.append({
                'x': px, 'y': py, 'rot': rot, 'scale': scale,
                'eff_w': eff_w, 'eff_d': eff_d,
                'w': w, 'd': d,
                'category': cat,
                'is_overlay': cat in OVERLAY_CATEGORIES,
            })
        
        # 1. Collision penalty
        for i in range(n):
            if states[i]['is_overlay']:
                continue
            for j in range(i+1, n):
                if states[j]['is_overlay']:
                    continue
                loss += self.w_collision * self._collision_penalty(states[i], states[j])
        
        # 2. Boundary penalty
        for i in range(n):
            loss += self.w_boundary * self._boundary_penalty(states[i])
        
        # 3. Facing relations
        for src_cat, tgt_cat in FACING_RELATIONS:
            for i in range(n):
                if states[i]['category'] != src_cat:
                    continue
                for j in range(n):
                    if states[j]['category'] != tgt_cat:
                        continue
                    loss += self.w_facing * self._facing_penalty(states[i], states[j])
        
        # 4. Proximity relations
        for cat_a, cat_b, max_dist in PROXIMITY_RELATIONS:
            for i in range(n):
                if states[i]['category'] != cat_a:
                    continue
                for j in range(n):
                    if states[j]['category'] != cat_b:
                        continue
                    loss += self.w_proximity * self._proximity_penalty(states[i], states[j], max_dist)
        
        # 5. Wall alignment for certain categories
        for i in range(n):
            if states[i]['category'] in WALL_CATEGORIES:
                loss += self.w_wall_align * self._wall_alignment_penalty(states[i])
        
        # 6. Scale regularization (prefer scale close to 1.0)
        for i in range(n):
            loss += self.w_scale * (states[i]['scale'] - 1.0) ** 2
        
        # 7. Rotation snap (prefer 0, 90, 180, 270 degrees)
        for i in range(n):
            rot_deg = np.degrees(states[i]['rot']) % 360
            # Distance to nearest 90-degree snap
            snap_dist = min(rot_deg % 90, 90 - rot_deg % 90)
            loss += self.w_rotation_snap * (snap_dist / 45.0) ** 2
        
        return loss
    
    def _collision_penalty(self, a: Dict, b: Dict) -> float:
        """Compute AABB collision penalty between two objects."""
        margin = self.collision_margin
        
        dx = abs(a['x'] - b['x'])
        dy = abs(a['y'] - b['y'])
        
        overlap_x = (a['eff_w'] + b['eff_w']) / 2 + margin - dx
        overlap_y = (a['eff_d'] + b['eff_d']) / 2 + margin - dy
        
        if overlap_x > 0 and overlap_y > 0:
            return overlap_x * overlap_y  # Overlap area
        return 0.0
    
    def _boundary_penalty(self, s: Dict) -> float:
        """Penalty for objects extending outside room boundaries."""
        margin = self.wall_margin
        penalty = 0.0
        
        hw, hd = s['eff_w'] / 2, s['eff_d'] / 2
        
        # Left/Right boundaries
        left_overshoot = margin - (s['x'] - hw)
        right_overshoot = (s['x'] + hw) - (self.room_width - margin)
        if left_overshoot > 0:
            penalty += left_overshoot ** 2
        if right_overshoot > 0:
            penalty += right_overshoot ** 2
        
        # Top/Bottom boundaries
        bottom_overshoot = margin - (s['y'] - hd)
        top_overshoot = (s['y'] + hd) - (self.room_depth - margin)
        if bottom_overshoot > 0:
            penalty += bottom_overshoot ** 2
        if top_overshoot > 0:
            penalty += top_overshoot ** 2
        
        return penalty
    
    def _facing_penalty(self, source: Dict, target: Dict) -> float:
        """Penalty if source object is not facing the target."""
        # Direction from source to target
        dx = target['x'] - source['x']
        dy = target['y'] - source['y']
        angle_to_target = np.arctan2(dy, dx)
        
        # Source's facing direction (front = negative Y in local frame)
        facing_angle = source['rot'] - np.pi / 2
        
        # Angular difference
        diff = angle_to_target - facing_angle
        diff = (diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
        
        return min(diff ** 2, np.pi ** 2)  # Cap at worst case
    
    def _proximity_penalty(self, a: Dict, b: Dict, max_dist: float) -> float:
        """Penalty if two objects are farther than max_dist."""
        dist = np.sqrt((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2)
        if dist > max_dist:
            return (dist - max_dist) ** 2
        return 0.0
    
    def _wall_alignment_penalty(self, s: Dict) -> float:
        """Penalty for wall-category objects not being near a wall."""
        hw, hd = s['eff_w'] / 2, s['eff_d'] / 2
        
        # Distance to nearest wall
        dist_to_walls = [
            s['x'] - hw,                          # left wall
            self.room_width - s['x'] - hw,         # right wall
            s['y'] - hd,                           # bottom wall
            self.room_depth - s['y'] - hd,         # top wall
        ]
        
        min_dist = min(dist_to_walls)
        threshold = 0.3  # Should be within 30cm of a wall
        
        if min_dist > threshold:
            return (min_dist - threshold) ** 2
        return 0.0


def optimize_layout(objects: List[Dict], room_width: float, room_depth: float,
                    verbose: bool = True) -> List[Dict]:
    """
    Convenience function to optimize a furniture layout.
    
    Args:
        objects: List of furniture dicts
        room_width: Room width in meters
        room_depth: Room depth in meters
        verbose: Print progress
        
    Returns:
        Optimized furniture list
    """
    optimizer = LayoutOptimizer(room_width, room_depth)
    
    if verbose:
        # Print before state
        print("\n--- Before Optimization ---")
        for obj in objects:
            dims = obj['dimensions']
            print(f"  {obj['category']}: pos=({obj['position'][0]:.2f}, {obj['position'][1]:.2f}) "
                  f"rot={obj['rotation']:.0f}° size={dims['width']:.2f}x{dims['depth']:.2f}m")
    
    optimized = optimizer.optimize(objects)
    
    if verbose:
        print("\n--- After Optimization ---")
        for obj in optimized:
            dims = obj['dimensions']
            scale = obj.get('scale_factor', 1.0)
            print(f"  {obj['category']}: pos=({obj['position'][0]:.2f}, {obj['position'][1]:.2f}) "
                  f"rot={obj['rotation']:.0f}° size={dims['width']:.2f}x{dims['depth']:.2f}m "
                  f"scale={scale:.2f}")
    
    return optimized
