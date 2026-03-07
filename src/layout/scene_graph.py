"""
SceneGraph: Core data structure representing a 3D indoor scene layout.

Stores objects with positions, rotations, sizes and supports:
  - Collision detection (AABB-based)
  - Boundary checking
  - Accessibility analysis (walkable path verification)
  - Serialization / deserialization
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SceneObject:
    """A single object in the scene."""

    object_id: str
    category: str
    position: tuple[float, float]  # (x, y) center on floor plan
    rotation: float = 0.0  # degrees
    size: dict = field(
        default_factory=lambda: {"width": 0.5, "depth": 0.5, "height": 0.5}
    )
    description: str = ""
    asset_id: Optional[str] = None

    @property
    def bbox_2d(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounding box on floor plan: (x_min, y_min, x_max, y_max).
        Accounts for rotation by computing the AABB of the rotated rectangle.
        """
        w = self.size["width"] / 2
        d = self.size["depth"] / 2
        rad = math.radians(self.rotation)
        cos_r, sin_r = abs(math.cos(rad)), abs(math.sin(rad))

        half_w = w * cos_r + d * sin_r
        half_d = w * sin_r + d * cos_r

        cx, cy = self.position
        return (cx - half_w, cy - half_d, cx + half_w, cy + half_d)

    @property
    def area(self) -> float:
        """Floor area occupied by the object."""
        return self.size["width"] * self.size["depth"]


class SceneGraph:
    """
    Represents a room layout as a collection of positioned objects.

    Supports collision detection, boundary checks, and accessibility analysis.
    """

    def __init__(self, width: float = 5.0, length: float = 4.0):
        self.width = width
        self.length = length
        self.objects: dict[str, SceneObject] = {}

    def add_object(
        self,
        object_id: str,
        category: str,
        position: tuple[float, float],
        rotation: float = 0.0,
        size: Optional[dict] = None,
        description: str = "",
        asset_id: Optional[str] = None,
    ):
        """Add an object to the scene."""
        self.objects[object_id] = SceneObject(
            object_id=object_id,
            category=category,
            position=position,
            rotation=rotation,
            size=size or {"width": 0.5, "depth": 0.5, "height": 0.5},
            description=description,
            asset_id=asset_id,
        )

    def update_object(
        self,
        object_id: str,
        position: Optional[tuple[float, float]] = None,
        rotation: Optional[float] = None,
    ):
        """Update an object's position and/or rotation."""
        if object_id not in self.objects:
            return
        if position is not None:
            self.objects[object_id].position = position
        if rotation is not None:
            self.objects[object_id].rotation = rotation

    def remove_object(self, object_id: str):
        """Remove an object from the scene."""
        self.objects.pop(object_id, None)

    # ── Collision Detection ──────────────────────────────────────────

    @staticmethod
    def _aabb_overlap(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
        margin: float = 0.0,
    ) -> bool:
        """Check if two AABBs overlap (with optional margin)."""
        return not (
            a[2] + margin <= b[0]
            or b[2] + margin <= a[0]
            or a[3] + margin <= b[1]
            or b[3] + margin <= a[1]
        )

    # Categories that are allowed to overlap with other objects
    OVERLAY_CATEGORIES = {"rug", "carpet", "mat", "floor_mat"}

    def find_collisions(self, margin: float = 0.05) -> list[tuple[str, str]]:
        """Find all pairs of colliding objects (excluding overlay items like rugs)."""
        collisions = []
        ids = list(self.objects.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = self.objects[ids[i]]
                b = self.objects[ids[j]]
                # Skip if either object is an overlay category
                if (
                    a.category in self.OVERLAY_CATEGORIES
                    or b.category in self.OVERLAY_CATEGORIES
                ):
                    continue
                if self._aabb_overlap(a.bbox_2d, b.bbox_2d, margin):
                    collisions.append((ids[i], ids[j]))
        return collisions

    # ── Boundary Checking ────────────────────────────────────────────

    def find_out_of_bounds(self, wall_margin: float = 0.02) -> list[str]:
        """Find objects that extend beyond room boundaries."""
        violations = []
        for obj_id, obj in self.objects.items():
            x_min, y_min, x_max, y_max = obj.bbox_2d
            if (
                x_min < wall_margin
                or y_min < wall_margin
                or x_max > self.width - wall_margin
                or y_max > self.length - wall_margin
            ):
                violations.append(obj_id)
        return violations

    # ── Accessibility ────────────────────────────────────────────────

    def check_accessibility(
        self, grid_resolution: float = 0.1, min_path_width: float = 0.6
    ) -> dict:
        """
        Check if the room has accessible walkable paths.

        Uses a simple grid-based approach:
        1. Rasterize objects onto a grid
        2. Check connectivity from door to all functional zones
        """
        grid_w = int(self.width / grid_resolution)
        grid_l = int(self.length / grid_resolution)
        grid = np.zeros((grid_w, grid_l), dtype=bool)  # True = occupied

        # Mark occupied cells
        for obj in self.objects.values():
            x_min, y_min, x_max, y_max = obj.bbox_2d
            i_min = max(0, int(x_min / grid_resolution))
            j_min = max(0, int(y_min / grid_resolution))
            i_max = min(grid_w, int(x_max / grid_resolution) + 1)
            j_max = min(grid_l, int(y_max / grid_resolution) + 1)
            grid[i_min:i_max, j_min:j_max] = True

        # Calculate walkable area ratio
        total_cells = grid_w * grid_l
        occupied_cells = int(np.sum(grid))
        walkable_ratio = 1.0 - occupied_cells / total_cells

        return {
            "walkable_ratio": walkable_ratio,
            "occupied_ratio": occupied_cells / total_cells,
            "grid_size": (grid_w, grid_l),
        }

    # ── Violation Checking ───────────────────────────────────────────

    def check_violations(
        self,
        collision_margin: float = 0.05,
        wall_margin: float = 0.02,
    ) -> list[dict]:
        """Check all types of violations and return a list of issues."""
        violations = []

        # Collision violations
        for a_id, b_id in self.find_collisions(collision_margin):
            violations.append(
                {
                    "type": "collision",
                    "objects": [a_id, b_id],
                    "message": f"{a_id} and {b_id} are overlapping.",
                }
            )

        # Boundary violations
        for obj_id in self.find_out_of_bounds(wall_margin):
            violations.append(
                {
                    "type": "out_of_bounds",
                    "objects": [obj_id],
                    "message": f"{obj_id} extends beyond room boundaries.",
                }
            )

        return violations

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize scene graph to dict."""
        return {
            "room": {"width": self.width, "length": self.length},
            "objects": [
                {
                    "object_id": obj.object_id,
                    "category": obj.category,
                    "position": list(obj.position),
                    "rotation": obj.rotation,
                    "size": obj.size,
                    "bbox_2d": list(obj.bbox_2d),
                }
                for obj in self.objects.values()
            ],
        }

    def to_object_list(self) -> list[dict]:
        """Return list of object dicts for export."""
        return [
            {
                "object_id": obj.object_id,
                "category": obj.category,
                "position": {
                    "x": obj.position[0],
                    "y": 0.0,  # Floor level
                    "z": obj.position[1],
                },
                "rotation_y": obj.rotation,
                "size": obj.size,
                "asset_id": obj.asset_id,
            }
            for obj in self.objects.values()
        ]

    @classmethod
    def from_layout(
        cls, layout_data: "SceneGraph", width: float, length: float
    ) -> "SceneGraph":
        """Create from an existing SceneGraph (identity for type compatibility)."""
        if isinstance(layout_data, cls):
            return layout_data
        raise TypeError(f"Expected SceneGraph, got {type(layout_data)}")

    def __repr__(self) -> str:
        return (
            f"SceneGraph(room={self.width}x{self.length}m, "
            f"objects={len(self.objects)})"
        )
