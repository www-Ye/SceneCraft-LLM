#!/usr/bin/env python3
"""
Render MuJoCo simulation as a GIF animation.
Uses OSMesa for headless (no-display) rendering.
"""

import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import mujoco
    from PIL import Image
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install mujoco Pillow")
    sys.exit(1)

from src.layout.scene_graph import SceneGraph
from scripts.mujoco_sim import scene_to_mjcf


def render_simulation_gif(
    mjcf_xml: str,
    output_path: str,
    duration: float = 3.0,
    fps: int = 20,
    width: int = 800,
    height: int = 600,
    camera_distance: float = 8.0,
    camera_elevation: float = -35.0,
    camera_lookat: tuple = None,
    rotate_camera: bool = True,
):
    """
    Run simulation and render each frame, then save as GIF.

    Args:
        mjcf_xml: MJCF XML string
        output_path: Path to save the GIF
        duration: Simulation duration in seconds
        fps: Frames per second in the GIF
        width, height: Image resolution
        camera_distance: Camera distance from lookat point
        camera_elevation: Camera elevation angle (degrees)
        camera_lookat: (x, y, z) point to look at
        rotate_camera: Whether to slowly rotate the camera
    """
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    dt = model.opt.timestep
    steps_per_frame = max(1, int(1.0 / (fps * dt)))
    n_frames = int(duration * fps)

    # Default lookat: center of the room
    if camera_lookat is None:
        camera_lookat = (2.5, 2.0, 0.5)

    frames = []
    print(f"  Rendering {n_frames} frames at {width}x{height}...")

    for frame_idx in range(n_frames):
        # Step simulation
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)

        # Set camera
        scene_option = mujoco.MjvOption()

        # Camera rotation
        if rotate_camera:
            azimuth = 135 + frame_idx * (90.0 / n_frames)  # Slow pan
        else:
            azimuth = 135

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = camera_distance
        cam.azimuth = azimuth
        cam.elevation = camera_elevation
        cam.lookat[:] = camera_lookat

        renderer.update_scene(
            data,
            camera=cam,
            scene_option=scene_option,
        )

        img = renderer.render()
        frames.append(Image.fromarray(img))

        if (frame_idx + 1) % 10 == 0:
            print(f"    Frame {frame_idx + 1}/{n_frames}")

    renderer.close()

    # Save as GIF
    print(f"  Saving GIF to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),  # ms per frame
        loop=0,  # Infinite loop
        optimize=True,
    )

    file_size = os.path.getsize(output_path) / 1024
    print(f"  ✅ GIF saved: {file_size:.0f} KB, {n_frames} frames")


def main():
    print("=" * 60)
    print("SceneCraft-LLM × MuJoCo GIF Renderer")
    print("=" * 60)

    # Create scene
    print("\n📦 Creating scene...")
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
    print(f"  {scene}")

    # Convert to MJCF
    print("\n🔧 Converting to MJCF...")
    mjcf_xml = scene_to_mjcf(scene)

    # Render GIF
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, "simulation.gif")

    print("\n🎬 Rendering simulation...")
    render_simulation_gif(
        mjcf_xml,
        gif_path,
        duration=3.0,
        fps=15,
        width=640,
        height=480,
        camera_distance=7.0,
        camera_elevation=-30.0,
        camera_lookat=(2.5, 2.0, 0.6),
        rotate_camera=True,
    )

    print(f"\n🎉 Done! GIF at: {gif_path}")


if __name__ == "__main__":
    main()
