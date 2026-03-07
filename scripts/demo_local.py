#!/usr/bin/env python3
"""
Local demo: Generate and evaluate a scene WITHOUT calling LLM APIs.
Uses hardcoded object placements to verify the pipeline works end-to-end.
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.layout.scene_graph import SceneGraph
from src.layout.constraint_solver import ConstraintSolver
from src.evaluation.metrics import SceneMetrics


def create_living_room_scene() -> SceneGraph:
    """Create a sample living room scene (simulating LLM output)."""
    scene = SceneGraph(width=5.0, length=4.0)

    # Sofa against the back wall
    scene.add_object(
        object_id="sofa_0",
        category="sofa",
        position=(2.5, 3.2),
        rotation=0,
        size={"width": 2.0, "depth": 0.9, "height": 0.85},
        description="3-seater modern sofa",
    )

    # TV stand against the opposite wall
    scene.add_object(
        object_id="tv_stand_0",
        category="tv_stand",
        position=(2.5, 0.4),
        rotation=0,
        size={"width": 1.5, "depth": 0.4, "height": 0.5},
        description="Modern TV stand",
    )

    # Coffee table in front of sofa
    scene.add_object(
        object_id="coffee_table_0",
        category="coffee_table",
        position=(2.5, 2.2),
        rotation=0,
        size={"width": 1.0, "depth": 0.6, "height": 0.45},
        description="Rectangular coffee table",
    )

    # Armchair to the left of sofa
    scene.add_object(
        object_id="armchair_0",
        category="armchair",
        position=(0.6, 2.8),
        rotation=90,
        size={"width": 0.8, "depth": 0.8, "height": 0.9},
        description="Single armchair",
    )

    # Bookshelf against left wall
    scene.add_object(
        object_id="bookshelf_0",
        category="bookshelf",
        position=(0.3, 1.0),
        rotation=0,
        size={"width": 0.4, "depth": 0.3, "height": 1.8},
        description="Narrow bookshelf",
    )

    # Floor lamp next to sofa
    scene.add_object(
        object_id="floor_lamp_0",
        category="floor_lamp",
        position=(4.2, 3.2),
        rotation=0,
        size={"width": 0.3, "depth": 0.3, "height": 1.6},
        description="Standing floor lamp",
    )

    # Side table next to armchair
    scene.add_object(
        object_id="side_table_0",
        category="side_table",
        position=(0.6, 1.8),
        rotation=0,
        size={"width": 0.45, "depth": 0.45, "height": 0.55},
        description="Round side table",
    )

    # Rug under coffee table (intentional overlap for testing)
    scene.add_object(
        object_id="rug_0",
        category="rug",
        position=(2.5, 2.2),
        rotation=0,
        size={"width": 2.5, "depth": 1.8, "height": 0.02},
        description="Area rug",
    )

    return scene


def create_ground_truth_living_room() -> SceneGraph:
    """Create a ground truth living room for comparison."""
    scene = SceneGraph(width=5.0, length=4.0)

    scene.add_object("sofa_0", "sofa", (2.5, 3.0), 0,
                      {"width": 2.2, "depth": 0.9, "height": 0.85})
    scene.add_object("tv_stand_0", "tv_stand", (2.5, 0.5), 0,
                      {"width": 1.6, "depth": 0.4, "height": 0.5})
    scene.add_object("coffee_table_0", "coffee_table", (2.5, 2.0), 0,
                      {"width": 1.2, "depth": 0.6, "height": 0.45})
    scene.add_object("armchair_0", "armchair", (0.7, 2.5), 90,
                      {"width": 0.8, "depth": 0.8, "height": 0.9})
    scene.add_object("bookshelf_0", "bookshelf", (0.3, 0.8), 0,
                      {"width": 0.4, "depth": 0.3, "height": 1.8})
    scene.add_object("floor_lamp_0", "floor_lamp", (4.3, 3.0), 0,
                      {"width": 0.3, "depth": 0.3, "height": 1.6})
    scene.add_object("plant_0", "plant", (4.5, 0.4), 0,
                      {"width": 0.3, "depth": 0.3, "height": 0.8})

    return scene


def main():
    print("=" * 60)
    print("SceneCraft-LLM Local Demo")
    print("=" * 60)

    # Step 1: Create scene
    print("\n📦 Creating sample living room scene...")
    scene = create_living_room_scene()
    print(f"   Created: {scene}")

    # Step 2: Check violations before optimization
    print("\n🔍 Checking violations (before optimization)...")
    violations = scene.check_violations()
    for v in violations:
        print(f"   ⚠️  {v['type']}: {v['message']}")
    if not violations:
        print("   ✅ No violations found!")

    # Step 3: Optimize layout
    print("\n⚙️  Running constraint solver...")
    solver_config = {
        "collision_margin": 0.05,
        "wall_margin": 0.02,
        "max_iterations": 1000,
    }
    solver = ConstraintSolver(solver_config)
    optimized = solver.optimize(scene)

    # Step 4: Check violations after optimization
    print("\n🔍 Checking violations (after optimization)...")
    violations = optimized.check_violations()
    for v in violations:
        print(f"   ⚠️  {v['type']}: {v['message']}")
    if not violations:
        print("   ✅ All constraints satisfied!")

    # Step 5: Evaluate
    print("\n📊 Evaluating scene quality...")
    gt = create_ground_truth_living_room()
    evaluator = SceneMetrics()
    metrics = evaluator.evaluate(optimized, gt)

    print("\n   Metric                    Value")
    print("   " + "-" * 40)
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"   {key:28s} {value:.4f}")
        else:
            print(f"   {key:28s} {value}")

    # Step 6: Export scene
    print("\n💾 Exporting scene...")
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "outputs", "demo_scene.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    scene_data = {
        "description": "A modern living room with sofa facing TV",
        "room": {"width": 5.0, "length": 4.0, "height": 2.8},
        "objects": optimized.to_object_list(),
        "metrics": metrics,
    }
    with open(output_path, "w") as f:
        json.dump(scene_data, f, indent=2)
    print(f"   Saved to: {output_path}")

    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
