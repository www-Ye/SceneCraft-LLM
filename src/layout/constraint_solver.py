"""
ConstraintSolver: Physics-based optimization for scene layouts.

Takes an LLM-generated layout and resolves physical constraint violations:
  - Object-object collisions → repulsion-based separation
  - Object-wall collisions → push inward
  - Accessibility → ensure walkable paths

Uses iterative projection (no GPU needed).
"""

import logging
import math
from typing import Optional

import numpy as np

from .scene_graph import SceneGraph

logger = logging.getLogger(__name__)


class ConstraintSolver:
    """Resolve layout constraint violations via iterative optimization."""

    def __init__(self, config: dict):
        self.collision_margin = config.get("collision_margin", 0.05)
        self.wall_margin = config.get("wall_margin", 0.02)
        self.max_iterations = config.get("max_iterations", 1000)
        self.step_size = 0.02  # meters per iteration step
        self.min_walkable_ratio = 0.3

    def optimize(self, scene: SceneGraph) -> SceneGraph:
        """
        Optimize scene layout by iteratively resolving constraint violations.

        Strategy:
        1. Fix boundary violations first (push objects inward)
        2. Resolve collisions (push apart along overlap axis)
        3. Verify accessibility

        Returns:
            Optimized SceneGraph (modified in-place)
        """
        logger.info(
            f"Starting constraint optimization "
            f"(max {self.max_iterations} iterations)..."
        )

        for iteration in range(self.max_iterations):
            changed = False

            # Step 1: Fix boundary violations
            changed |= self._fix_boundaries(scene)

            # Step 2: Fix collisions
            changed |= self._fix_collisions(scene)

            if not changed:
                logger.info(
                    f"Converged after {iteration + 1} iterations."
                )
                break

        # Final check
        violations = scene.check_violations(
            self.collision_margin, self.wall_margin
        )
        if violations:
            logger.warning(
                f"Optimization finished with {len(violations)} "
                f"remaining violations."
            )
        else:
            logger.info("All constraints satisfied.")

        # Accessibility check
        acc = scene.check_accessibility()
        if acc["walkable_ratio"] < self.min_walkable_ratio:
            logger.warning(
                f"Low walkable ratio: {acc['walkable_ratio']:.1%}. "
                f"Room may be overcrowded."
            )

        return scene

    def _fix_boundaries(self, scene: SceneGraph) -> bool:
        """Push out-of-bounds objects back inside room boundaries."""
        changed = False
        margin = self.wall_margin

        for obj_id in scene.find_out_of_bounds(margin):
            obj = scene.objects[obj_id]
            x_min, y_min, x_max, y_max = obj.bbox_2d
            cx, cy = obj.position

            # Calculate how much to push
            dx, dy = 0.0, 0.0
            half_w = (x_max - x_min) / 2
            half_h = (y_max - y_min) / 2

            if x_min < margin:
                dx = margin + half_w - cx
            elif x_max > scene.width - margin:
                dx = (scene.width - margin - half_w) - cx

            if y_min < margin:
                dy = margin + half_h - cy
            elif y_max > scene.length - margin:
                dy = (scene.length - margin - half_h) - cy

            new_pos = (cx + dx, cy + dy)
            scene.update_object(obj_id, position=new_pos)
            changed = True

        return changed

    def _fix_collisions(self, scene: SceneGraph) -> bool:
        """Separate colliding objects by pushing them apart."""
        changed = False
        collisions = scene.find_collisions(self.collision_margin)

        for a_id, b_id in collisions:
            a = scene.objects[a_id]
            b = scene.objects[b_id]

            a_bbox = a.bbox_2d
            b_bbox = b.bbox_2d

            # Calculate overlap on each axis
            overlap_x = min(a_bbox[2], b_bbox[2]) - max(a_bbox[0], b_bbox[0])
            overlap_y = min(a_bbox[3], b_bbox[3]) - max(a_bbox[1], b_bbox[1])

            if overlap_x <= 0 or overlap_y <= 0:
                continue  # No actual overlap (within margin only)

            # Push along the axis of minimum overlap (+ margin)
            push = self.collision_margin
            if overlap_x < overlap_y:
                # Push horizontally
                direction = 1 if a.position[0] < b.position[0] else -1
                shift = (overlap_x / 2 + push) * direction
                scene.update_object(
                    a_id,
                    position=(a.position[0] - shift, a.position[1]),
                )
                scene.update_object(
                    b_id,
                    position=(b.position[0] + shift, b.position[1]),
                )
            else:
                # Push vertically
                direction = 1 if a.position[1] < b.position[1] else -1
                shift = (overlap_y / 2 + push) * direction
                scene.update_object(
                    a_id,
                    position=(a.position[0], a.position[1] - shift),
                )
                scene.update_object(
                    b_id,
                    position=(b.position[0], b.position[1] + shift),
                )

            changed = True

        return changed
