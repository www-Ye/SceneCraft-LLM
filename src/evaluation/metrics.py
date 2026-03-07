"""
Evaluation metrics for generated 3D indoor scenes.

Metrics:
  1. Collision Rate: % of object pairs that collide
  2. Out-of-Bounds Rate: % of objects outside room boundaries
  3. Floor Coverage: ratio of floor area occupied by furniture
  4. Walkable Ratio: ratio of floor area that is walkable
  5. Object Count Accuracy: comparison with ground truth
  6. Spatial Relationship Score: correctness of object relationships
  7. FID (Layout): distributional similarity to real layouts
"""

import logging
from collections import Counter
from typing import Optional

import numpy as np

from ..layout.scene_graph import SceneGraph

logger = logging.getLogger(__name__)


class SceneMetrics:
    """Compute evaluation metrics for generated scenes."""

    def __init__(self, collision_margin: float = 0.05, wall_margin: float = 0.02):
        self.collision_margin = collision_margin
        self.wall_margin = wall_margin

    def evaluate(
        self,
        scene: SceneGraph,
        ground_truth: Optional[SceneGraph] = None,
    ) -> dict:
        """
        Compute all metrics for a generated scene.

        Args:
            scene: Generated scene to evaluate
            ground_truth: Optional reference scene for comparison

        Returns:
            Dictionary of metric name → value
        """
        metrics = {}

        # Physical plausibility metrics
        metrics.update(self._physical_metrics(scene))

        # Coverage & density metrics
        metrics.update(self._coverage_metrics(scene))

        # Comparison metrics (if ground truth available)
        if ground_truth is not None:
            metrics.update(self._comparison_metrics(scene, ground_truth))

        return metrics

    def _physical_metrics(self, scene: SceneGraph) -> dict:
        """Metrics related to physical plausibility."""
        n_objects = len(scene.objects)
        if n_objects == 0:
            return {
                "collision_rate": 0.0,
                "oob_rate": 0.0,
                "n_collisions": 0,
                "n_oob": 0,
            }

        collisions = scene.find_collisions(self.collision_margin)
        oob = scene.find_out_of_bounds(self.wall_margin)
        n_pairs = n_objects * (n_objects - 1) / 2

        return {
            "collision_rate": len(collisions) / max(n_pairs, 1),
            "oob_rate": len(oob) / n_objects,
            "n_collisions": len(collisions),
            "n_oob": len(oob),
            "n_objects": n_objects,
        }

    def _coverage_metrics(self, scene: SceneGraph) -> dict:
        """Metrics related to space utilization."""
        room_area = scene.width * scene.length
        furniture_area = sum(obj.area for obj in scene.objects.values())

        accessibility = scene.check_accessibility()

        return {
            "floor_coverage": furniture_area / room_area,
            "walkable_ratio": accessibility["walkable_ratio"],
            "occupied_ratio": accessibility["occupied_ratio"],
            "room_area": room_area,
            "furniture_area": furniture_area,
        }

    def _comparison_metrics(
        self, generated: SceneGraph, ground_truth: SceneGraph
    ) -> dict:
        """Metrics comparing generated scene to ground truth."""
        # Category distribution comparison
        gen_cats = Counter(
            obj.category for obj in generated.objects.values()
        )
        gt_cats = Counter(
            obj.category for obj in ground_truth.objects.values()
        )

        all_cats = set(gen_cats.keys()) | set(gt_cats.keys())
        category_precision = sum(
            min(gen_cats.get(c, 0), gt_cats.get(c, 0)) for c in all_cats
        ) / max(sum(gen_cats.values()), 1)
        category_recall = sum(
            min(gen_cats.get(c, 0), gt_cats.get(c, 0)) for c in all_cats
        ) / max(sum(gt_cats.values()), 1)

        cat_f1 = (
            2 * category_precision * category_recall
            / max(category_precision + category_recall, 1e-8)
        )

        # Object count accuracy
        count_diff = abs(len(generated.objects) - len(ground_truth.objects))
        count_accuracy = 1.0 - count_diff / max(
            len(ground_truth.objects), 1
        )

        return {
            "category_precision": category_precision,
            "category_recall": category_recall,
            "category_f1": cat_f1,
            "count_accuracy": max(0.0, count_accuracy),
            "count_diff": count_diff,
        }

    def evaluate_batch(
        self,
        scenes: list[SceneGraph],
        ground_truths: Optional[list[SceneGraph]] = None,
    ) -> dict:
        """Evaluate a batch of scenes and return aggregated metrics."""
        all_metrics = []
        for i, scene in enumerate(scenes):
            gt = ground_truths[i] if ground_truths else None
            all_metrics.append(self.evaluate(scene, gt))

        # Aggregate: mean and std
        aggregated = {}
        if not all_metrics:
            return aggregated

        keys = all_metrics[0].keys()
        for key in keys:
            values = [m[key] for m in all_metrics if key in m]
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))

        aggregated["n_scenes"] = len(scenes)
        return aggregated
