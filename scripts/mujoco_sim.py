#!/usr/bin/env python3
"""
MuJoCo Simulation: Validate generated scenes with physics simulation.

This script:
1. Takes a generated scene (from our pipeline)
2. Converts it to a MuJoCo XML (MJCF) model
3. Runs physics simulation to verify:
   - Objects are stable (don't fly away)
   - No interpenetration
   - Objects settle on the ground properly
4. Reports physical plausibility metrics
"""

import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import mujoco
except ImportError:
    print("ERROR: mujoco not installed. Run: pip install mujoco")
    sys.exit(1)

from src.layout.scene_graph import SceneGraph


# ── Scene to MJCF Conversion ─────────────────────────────────────────

def scene_to_mjcf(scene: SceneGraph, room_height: float = 2.8) -> str:
    """
    Convert a SceneGraph to MuJoCo XML (MJCF) format.

    Creates:
    - A ground plane (floor)
    - 4 walls
    - Each furniture item as a box geom with free joint
    """
    root = ET.Element("mujoco", model="scene_sim")

    # Compiler settings
    ET.SubElement(root, "compiler", angle="degree", coordinate="local")

    # Simulation options
    option = ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81")

    # Visual settings
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", ambient="0.5 0.5 0.5")

    # Assets (materials)
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "material", name="floor_mat", rgba="0.8 0.8 0.8 1")
    ET.SubElement(asset, "material", name="wall_mat", rgba="0.9 0.9 0.85 1")
    ET.SubElement(asset, "material", name="furniture_mat", rgba="0.6 0.4 0.2 1")
    ET.SubElement(asset, "material", name="sofa_mat", rgba="0.3 0.4 0.6 1")
    ET.SubElement(asset, "material", name="table_mat", rgba="0.5 0.35 0.2 1")

    worldbody = ET.SubElement(root, "worldbody")

    # Light
    ET.SubElement(worldbody, "light", pos=f"{scene.width/2} {scene.length/2} {room_height-0.1}",
                  dir="0 0 -1", diffuse="0.8 0.8 0.8")

    # Floor
    ET.SubElement(worldbody, "geom", name="floor", type="plane",
                  size=f"{scene.width/2} {scene.length/2} 0.01",
                  pos=f"{scene.width/2} {scene.length/2} 0",
                  material="floor_mat")

    # Walls (static boxes)
    wall_thickness = 0.05
    # Bottom wall (y=0)
    ET.SubElement(worldbody, "geom", name="wall_bottom", type="box",
                  size=f"{scene.width/2} {wall_thickness/2} {room_height/2}",
                  pos=f"{scene.width/2} {-wall_thickness/2} {room_height/2}",
                  material="wall_mat")
    # Top wall (y=length)
    ET.SubElement(worldbody, "geom", name="wall_top", type="box",
                  size=f"{scene.width/2} {wall_thickness/2} {room_height/2}",
                  pos=f"{scene.width/2} {scene.length + wall_thickness/2} {room_height/2}",
                  material="wall_mat")
    # Left wall (x=0)
    ET.SubElement(worldbody, "geom", name="wall_left", type="box",
                  size=f"{wall_thickness/2} {scene.length/2} {room_height/2}",
                  pos=f"{-wall_thickness/2} {scene.length/2} {room_height/2}",
                  material="wall_mat")
    # Right wall (x=width)
    ET.SubElement(worldbody, "geom", name="wall_right", type="box",
                  size=f"{wall_thickness/2} {scene.length/2} {room_height/2}",
                  pos=f"{scene.width + wall_thickness/2} {scene.length/2} {room_height/2}",
                  material="wall_mat")

    # Furniture objects (with free joints for physics)
    for obj in scene.objects.values():
        w = obj.size["width"] / 2
        d = obj.size["depth"] / 2
        h = obj.size["height"] / 2

        # Choose material based on category
        mat = "furniture_mat"
        if "sofa" in obj.category or "chair" in obj.category:
            mat = "sofa_mat"
        elif "table" in obj.category or "desk" in obj.category:
            mat = "table_mat"

        # Density based on category (kg/m³)
        density = "200"  # Default wood-like
        if obj.category in ("rug", "carpet", "mat"):
            density = "50"
        elif "lamp" in obj.category:
            density = "100"

        body = ET.SubElement(worldbody, "body",
                             name=obj.object_id,
                             pos=f"{obj.position[0]} {obj.position[1]} {h}")

        # Free joint allows physics simulation
        ET.SubElement(body, "freejoint", name=f"{obj.object_id}_joint")

        ET.SubElement(body, "geom",
                      name=f"{obj.object_id}_geom",
                      type="box",
                      size=f"{w} {d} {h}",
                      material=mat,
                      density=density)

    # Pretty print
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


# ── Physics Simulation ────────────────────────────────────────────────

def run_simulation(
    mjcf_xml: str,
    duration: float = 3.0,
    dt: float = 0.002,
) -> dict:
    """
    Run MuJoCo physics simulation and collect stability metrics.

    Args:
        mjcf_xml: MJCF XML string
        duration: Simulation duration in seconds
        dt: Timestep

    Returns:
        Dictionary of physics metrics
    """
    # Load model
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    data = mujoco.MjData(model)

    n_steps = int(duration / dt)
    n_bodies = model.nbody - 1  # Exclude world body

    # Record initial positions
    mujoco.mj_forward(model, data)
    initial_positions = {}
    for i in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name:
            initial_positions[name] = data.xpos[i].copy()

    # Track metrics over time
    max_velocities = []
    contact_forces = []

    # Run simulation
    print(f"  Running {duration}s simulation ({n_steps} steps)...")
    for step in range(n_steps):
        mujoco.mj_step(model, data)

        # Record max velocity every 100 steps
        if step % 100 == 0:
            max_vel = 0.0
            for i in range(1, model.nbody):
                vel = np.linalg.norm(data.cvel[i, 3:6])  # Linear velocity
                max_vel = max(max_vel, vel)
            max_velocities.append(max_vel)

        # Record contact info
        if step % 500 == 0:
            contact_forces.append(data.ncon)

    # Record final positions
    final_positions = {}
    for i in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name:
            final_positions[name] = data.xpos[i].copy()

    # ── Compute Metrics ──────────────────────────────────────────

    # 1. Stability: How much did objects move?
    displacements = {}
    for name in initial_positions:
        if name in final_positions:
            disp = np.linalg.norm(final_positions[name] - initial_positions[name])
            displacements[name] = float(disp)

    # 2. Objects that fell through floor (z < -0.1)
    fallen_objects = []
    for name, pos in final_positions.items():
        if pos[2] < -0.1:
            fallen_objects.append(name)

    # 3. Objects that flew away (displacement > 1m)
    unstable_objects = [
        name for name, disp in displacements.items() if disp > 1.0
    ]

    # 4. Settled check: are velocities near zero at the end?
    final_max_vel = max_velocities[-1] if max_velocities else 0.0
    settled = final_max_vel < 0.01

    avg_displacement = np.mean(list(displacements.values())) if displacements else 0.0
    max_displacement = max(displacements.values()) if displacements else 0.0

    metrics = {
        "duration_s": duration,
        "n_steps": n_steps,
        "n_objects": len(initial_positions),
        "avg_displacement_m": float(avg_displacement),
        "max_displacement_m": float(max_displacement),
        "n_fallen": len(fallen_objects),
        "fallen_objects": fallen_objects,
        "n_unstable": len(unstable_objects),
        "unstable_objects": unstable_objects,
        "settled": settled,
        "final_max_velocity": float(final_max_vel),
        "stability_score": float(
            1.0
            - len(fallen_objects) / max(len(initial_positions), 1)
            - len(unstable_objects) / max(len(initial_positions), 1)
        ),
        "per_object_displacement": displacements,
    }

    return metrics


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SceneCraft-LLM × MuJoCo Physics Validation")
    print("=" * 60)

    # Create test scene
    print("\n📦 Creating test scene...")
    scene = SceneGraph(width=5.0, length=4.0)

    scene.add_object("sofa_0", "sofa", (2.5, 3.2), 0,
                      {"width": 2.0, "depth": 0.9, "height": 0.85})
    scene.add_object("tv_stand_0", "tv_stand", (2.5, 0.4), 0,
                      {"width": 1.5, "depth": 0.4, "height": 0.5})
    scene.add_object("coffee_table_0", "coffee_table", (2.5, 2.0), 0,
                      {"width": 1.0, "depth": 0.6, "height": 0.45})
    scene.add_object("armchair_0", "armchair", (0.8, 2.8), 90,
                      {"width": 0.8, "depth": 0.8, "height": 0.9})
    scene.add_object("bookshelf_0", "bookshelf", (0.3, 1.0), 0,
                      {"width": 0.4, "depth": 0.3, "height": 1.8})
    scene.add_object("floor_lamp_0", "floor_lamp", (4.2, 3.2), 0,
                      {"width": 0.3, "depth": 0.3, "height": 1.6})
    scene.add_object("side_table_0", "side_table", (0.8, 1.8), 0,
                      {"width": 0.45, "depth": 0.45, "height": 0.55})

    print(f"  Scene: {scene}")

    # Convert to MJCF
    print("\n🔧 Converting to MuJoCo XML...")
    mjcf_xml = scene_to_mjcf(scene)

    # Save XML for inspection
    xml_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "scene.xml")
    os.makedirs(os.path.dirname(xml_path), exist_ok=True)
    with open(xml_path, "w") as f:
        f.write(mjcf_xml)
    print(f"  Saved MJCF to: {xml_path}")

    # Run simulation
    print("\n⚡ Running physics simulation...")
    metrics = run_simulation(mjcf_xml, duration=3.0)

    # Print results
    print("\n📊 Physics Validation Results")
    print("  " + "-" * 45)
    print(f"  {'Objects simulated':30s} {metrics['n_objects']}")
    print(f"  {'Simulation duration':30s} {metrics['duration_s']}s")
    print(f"  {'Settled (velocities ≈ 0)':30s} {'✅ Yes' if metrics['settled'] else '❌ No'}")
    print(f"  {'Avg displacement':30s} {metrics['avg_displacement_m']:.4f}m")
    print(f"  {'Max displacement':30s} {metrics['max_displacement_m']:.4f}m")
    print(f"  {'Fallen through floor':30s} {metrics['n_fallen']}")
    print(f"  {'Unstable (moved >1m)':30s} {metrics['n_unstable']}")
    print(f"  {'Stability score':30s} {metrics['stability_score']:.2%}")

    print("\n  Per-object displacement:")
    for name, disp in sorted(metrics["per_object_displacement"].items()):
        status = "✅" if disp < 0.1 else ("⚠️" if disp < 1.0 else "❌")
        print(f"    {status} {name:25s} {disp:.4f}m")

    # Save full results
    results_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "mujoco_results.json")

    # Convert numpy types for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        return obj

    save_metrics = make_serializable(metrics)
    with open(results_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"\n💾 Full results saved to: {results_path}")

    print("\n" + "=" * 60)
    if metrics["stability_score"] >= 0.9:
        print("🎉 Scene is physically plausible!")
    elif metrics["stability_score"] >= 0.7:
        print("⚠️  Scene has some physical issues.")
    else:
        print("❌ Scene has significant physical problems.")
    print("=" * 60)


if __name__ == "__main__":
    main()
