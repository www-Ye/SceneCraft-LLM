#!/usr/bin/env python3
"""
3D-FRONT Dataset Processor

Parses 3D-FRONT JSON scene files and extracts:
1. Room layouts (furniture positions, sizes, categories)
2. Room metadata (dimensions, types)
3. Converts to our SceneGraph format for evaluation

3D-FRONT dataset: https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset
Expected structure:
  data/3d-front/
    ├── 3D-FRONT/          # Scene JSON files
    │   ├── xxxxx.json
    │   └── ...
    └── 3D-FUTURE-model/   # 3D model files (optional for layout-only)
        ├── model_id/
        │   └── raw_model.obj
        └── ...
"""

import json
import logging
import os
import sys
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.layout.scene_graph import SceneGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── 3D-FRONT Category Mapping ────────────────────────────────────────

# Map 3D-FRONT fine-grained categories to coarse categories
CATEGORY_MAP = {
    # Seating
    "Lounge Chair": "chair", "Dining Chair": "chair", "Desk Chair": "chair",
    "Chinese Chair": "chair", "Dressing Chair": "chair", "Office Chair": "chair",
    "Arm Chair": "armchair", "Lazy Sofa": "armchair", "L-shaped Sofa": "sofa",
    "Loveseat Sofa": "sofa", "Multi-seat Sofa": "sofa", "Three-seat Sofa": "sofa",
    "Two-seat Sofa": "sofa", "U-shaped Sofa": "sofa", "Corner/Side Table": "side_table",
    "Round End Table": "side_table",

    # Tables
    "Coffee Table": "coffee_table", "Dining Table": "dining_table",
    "Desk": "desk", "Dressing Table": "dressing_table",

    # Storage
    "Bookcase / jewelry Armoire": "bookshelf", "Wardrobe": "wardrobe",
    "TV Stand": "tv_stand", "Sideboard / Side Cabinet": "cabinet",
    "Wine Cabinet": "cabinet", "Drawer Chest / Corner cabinet": "cabinet",
    "Shelf": "shelf", "Shoe Cabinet": "cabinet",

    # Beds
    "King-size Bed": "bed", "Single bed": "bed", "Kids Bed": "bed",
    "Bunk Bed": "bed",

    # Lighting
    "Ceiling Lamp": "ceiling_lamp", "Pendant Lamp": "pendant_lamp",
    "Floor Lamp": "floor_lamp", "Table Lamp": "table_lamp",
    "Wall Lamp": "wall_lamp",

    # Bathroom
    "Bathtub": "bathtub", "Toilet": "toilet",

    # Others
    "Nightstand": "nightstand", "Stool": "stool",
    "Barstool": "stool", "Children Cabinet": "cabinet",
}

# Room type keywords
ROOM_TYPES = {
    "LivingRoom": "living_room",
    "Bedroom": "bedroom",
    "DiningRoom": "dining_room",
    "Library": "office",
    "KidsRoom": "kids_room",
    "BathRoom": "bathroom",
    "Balcony": "balcony",
    "Kitchen": "kitchen",
    "MasterBathroom": "bathroom",
    "SecondBedroom": "bedroom",
    "GuestBedroom": "bedroom",
}


def parse_3dfront_scene(json_path: str) -> list[dict]:
    """
    Parse a single 3D-FRONT JSON file and extract room layouts.

    Returns:
        List of room dicts, each containing:
        - room_type, dimensions, furniture list
    """
    with open(json_path, "r") as f:
        scene = json.load(f)

    scene_id = scene.get("uid", os.path.basename(json_path))

    # Build furniture lookup from scene
    furniture_map = {}
    for item in scene.get("furniture", []):
        fid = item.get("uid")
        if fid:
            furniture_map[fid] = {
                "jid": item.get("jid", ""),
                "category": item.get("category", "Unknown"),
                "title": item.get("title", ""),
                "size": item.get("size", {}),  # Not always available
            }

    # Build mesh lookup for bounding boxes
    mesh_map = {}
    for mesh in scene.get("mesh", []):
        mesh_map[mesh.get("uid", "")] = mesh

    rooms = []

    for room_data in scene.get("scene", {}).get("room", []):
        room_type_raw = room_data.get("type", "Unknown")
        room_type = ROOM_TYPES.get(room_type_raw, room_type_raw.lower())

        # Get room bounding box from floor mesh
        room_bbox = _get_room_bbox(room_data)
        if room_bbox is None:
            continue

        x_min, y_min, z_min, x_max, y_max, z_max = room_bbox
        width = x_max - x_min
        length = z_max - z_min  # In 3D-FRONT, z is depth
        height = y_max - y_min

        # Skip tiny rooms
        if width < 1.5 or length < 1.5:
            continue

        # Extract furniture placements
        furniture_list = []
        for child in room_data.get("children", []):
            ref = child.get("ref")
            if ref and ref in furniture_map:
                finfo = furniture_map[ref]
                category_raw = finfo["category"]
                category = CATEGORY_MAP.get(category_raw, category_raw.lower().replace(" ", "_"))

                # Get position and rotation from child transform
                pos = child.get("pos", [0, 0, 0])
                rot = child.get("rot", [0, 0, 0, 1])  # quaternion
                scale = child.get("scale", [1, 1, 1])

                # Get size from bbox if available
                bbox = child.get("bbox", None)
                if bbox and len(bbox) == 6:
                    obj_w = abs(bbox[3] - bbox[0])
                    obj_d = abs(bbox[5] - bbox[2])
                    obj_h = abs(bbox[4] - bbox[1])
                else:
                    obj_w, obj_d, obj_h = 0.5, 0.5, 0.5

                # Convert to room-relative coordinates
                rel_x = pos[0] - x_min
                rel_z = pos[2] - z_min

                # Extract Y rotation from quaternion
                y_rot = _quat_to_y_rotation(rot)

                furniture_list.append({
                    "category": category,
                    "category_raw": category_raw,
                    "position": (rel_x, rel_z),
                    "rotation": y_rot,
                    "size": {"width": obj_w, "depth": obj_d, "height": obj_h},
                    "title": finfo["title"],
                })

        if len(furniture_list) < 2:
            continue

        rooms.append({
            "scene_id": scene_id,
            "room_type": room_type,
            "width": width,
            "length": length,
            "height": height,
            "n_furniture": len(furniture_list),
            "furniture": furniture_list,
        })

    return rooms


def _get_room_bbox(room_data: dict) -> Optional[tuple]:
    """Extract room bounding box from floor vertices."""
    for child in room_data.get("children", []):
        if child.get("type") == "Floor" or "floor" in child.get("type", "").lower():
            xyz = child.get("xyz", [])
            if len(xyz) >= 6:
                xs = xyz[0::3]
                ys = xyz[1::3]
                zs = xyz[2::3]
                return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))
    return None


def _quat_to_y_rotation(q: list) -> float:
    """Convert quaternion [x, y, z, w] to Y-axis rotation in degrees."""
    if len(q) != 4:
        return 0.0
    x, y, z, w = q
    # Y-axis rotation from quaternion
    siny_cosp = 2 * (w * y + x * z)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    angle = np.arctan2(siny_cosp, cosy_cosp)
    return float(np.degrees(angle))


def room_to_scene_graph(room: dict) -> SceneGraph:
    """Convert a parsed room dict to our SceneGraph format."""
    sg = SceneGraph(width=room["width"], length=room["length"])
    for i, f in enumerate(room["furniture"]):
        sg.add_object(
            object_id=f"{f['category']}_{i}",
            category=f["category"],
            position=f["position"],
            rotation=f["rotation"],
            size=f["size"],
            description=f.get("title", ""),
        )
    return sg


def process_dataset(data_dir: str, output_dir: str, max_scenes: int = None):
    """Process all 3D-FRONT scenes and save extracted layouts."""
    scene_dir = os.path.join(data_dir, "3D-FRONT")
    if not os.path.exists(scene_dir):
        logger.error(f"3D-FRONT directory not found: {scene_dir}")
        logger.info("Please download from: https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset")
        return

    json_files = [f for f in os.listdir(scene_dir) if f.endswith(".json")]
    if max_scenes:
        json_files = json_files[:max_scenes]

    logger.info(f"Processing {len(json_files)} scene files...")

    all_rooms = []
    room_type_counts = Counter()
    category_counts = Counter()

    for i, fname in enumerate(json_files):
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(json_files)} files...")

        try:
            rooms = parse_3dfront_scene(os.path.join(scene_dir, fname))
            for room in rooms:
                all_rooms.append(room)
                room_type_counts[room["room_type"]] += 1
                for f in room["furniture"]:
                    category_counts[f["category"]] += 1
        except Exception as e:
            logger.warning(f"Error processing {fname}: {e}")

    logger.info(f"\nExtracted {len(all_rooms)} rooms from {len(json_files)} scenes")
    logger.info(f"\nRoom type distribution:")
    for rtype, count in room_type_counts.most_common():
        logger.info(f"  {rtype:20s} {count}")
    logger.info(f"\nTop 20 furniture categories:")
    for cat, count in category_counts.most_common(20):
        logger.info(f"  {cat:20s} {count}")

    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rooms.json")
    with open(output_path, "w") as f:
        json.dump(all_rooms, f, indent=2, default=str)
    logger.info(f"\nSaved to: {output_path}")

    # Also save statistics
    stats = {
        "n_scenes": len(json_files),
        "n_rooms": len(all_rooms),
        "room_type_counts": dict(room_type_counts),
        "category_counts": dict(category_counts.most_common(50)),
        "avg_furniture_per_room": np.mean([r["n_furniture"] for r in all_rooms]) if all_rooms else 0,
        "avg_room_width": np.mean([r["width"] for r in all_rooms]) if all_rooms else 0,
        "avg_room_length": np.mean([r["length"] for r in all_rooms]) if all_rooms else 0,
    }
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=float)
    logger.info(f"Stats saved to: {stats_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process 3D-FRONT dataset")
    parser.add_argument("--data_dir", default="data/3d-front", help="Path to 3D-FRONT data")
    parser.add_argument("--output_dir", default="data/processed", help="Output directory")
    parser.add_argument("--max_scenes", type=int, default=None, help="Max scenes to process")
    args = parser.parse_args()

    process_dataset(args.data_dir, args.output_dir, args.max_scenes)
