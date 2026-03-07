"""
Render MuJoCo physics simulation GIF with real 3D furniture assets.
Top-down view + rotating perspective view.
"""

import os
import sys
import numpy as np

os.environ['MUJOCO_GL'] = 'osmesa'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
from PIL import Image, ImageDraw, ImageFont


def render_gif():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    xml_path = os.path.join(output_dir, 'scene_with_assets.xml')
    
    if not os.path.exists(xml_path):
        print("Error: Run demo_with_assets.py first to generate scene_with_assets.xml")
        return
    
    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    W, H = 640, 480
    renderer = mujoco.Renderer(model, height=H, width=W)
    
    # Top-down camera
    cam_top = mujoco.MjvCamera()
    cam_top.lookat[0] = 2.5
    cam_top.lookat[1] = 2.0
    cam_top.lookat[2] = 0.3
    cam_top.distance = 7.0
    cam_top.elevation = -90
    cam_top.azimuth = 0
    
    # Rotating perspective camera
    cam_persp = mujoco.MjvCamera()
    cam_persp.lookat[0] = 2.5
    cam_persp.lookat[1] = 2.0
    cam_persp.lookat[2] = 0.4
    cam_persp.distance = 7.5
    cam_persp.elevation = -50
    
    try:
        font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf', 16)
    except:
        font = ImageFont.load_default()
    
    # Simulation parameters
    sim_duration = 4.0  # seconds
    fps = 15
    dt = model.opt.timestep
    steps_per_frame = max(1, int(1.0 / (fps * dt)))
    total_frames = int(sim_duration * fps)
    
    print(f"Rendering {total_frames} frames at {fps} fps ({sim_duration}s simulation)...")
    print(f"  Steps per frame: {steps_per_frame}, dt={dt}")
    
    frames = []
    
    for frame_idx in range(total_frames):
        # Step simulation
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        
        t = frame_idx / fps
        
        # Update perspective camera azimuth (rotate around scene)
        cam_persp.azimuth = 120 + frame_idx * (180.0 / total_frames)
        
        # Render top-down
        renderer.update_scene(data, cam_top)
        top_pixels = renderer.render()
        top_img = Image.fromarray(top_pixels)
        
        # Render perspective
        renderer.update_scene(data, cam_persp)
        persp_pixels = renderer.render()
        persp_img = Image.fromarray(persp_pixels)
        
        # Combine side by side
        combined_w = W * 2 + 10
        combined_h = H + 30  # extra for label
        combined = Image.new('RGB', (combined_w, combined_h), (255, 255, 255))
        draw = ImageDraw.Draw(combined)
        
        # Labels
        draw.text((10, 5), f'Top-Down View  t={t:.1f}s', fill=(0, 0, 0), font=font)
        draw.text((W + 20, 5), f'Perspective View (rotating)', fill=(0, 0, 0), font=font)
        
        combined.paste(top_img, (0, 28))
        combined.paste(persp_img, (W + 10, 28))
        
        frames.append(combined)
        
        if frame_idx % 10 == 0:
            print(f"  Frame {frame_idx}/{total_frames} (t={t:.1f}s)")
    
    # Save GIF
    gif_path = os.path.join(output_dir, 'simulation_assets.gif')
    print(f"\nSaving GIF ({len(frames)} frames)...")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    
    file_size = os.path.getsize(gif_path) / 1024
    print(f"✅ GIF saved: {gif_path} ({file_size:.0f} KB)")
    
    # Also save a compact single-view top-down GIF
    top_frames = []
    
    # Re-run for top-down only (smaller file)
    mujoco.mj_resetData(model, data)
    for frame_idx in range(total_frames):
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        
        t = frame_idx / fps
        renderer.update_scene(data, cam_top)
        pixels = renderer.render()
        img = Image.fromarray(pixels)
        
        # Add timestamp
        draw = ImageDraw.Draw(img)
        draw.rectangle([(5, 5), (120, 25)], fill=(0, 0, 0, 128))
        draw.text((10, 6), f't = {t:.1f}s', fill=(255, 255, 255), font=font)
        
        top_frames.append(img)
    
    top_gif_path = os.path.join(output_dir, 'simulation_topdown.gif')
    top_frames[0].save(
        top_gif_path,
        save_all=True,
        append_images=top_frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    top_size = os.path.getsize(top_gif_path) / 1024
    print(f"✅ Top-down GIF saved: {top_gif_path} ({top_size:.0f} KB)")
    
    # Check final stability
    stable = 0
    total = 0
    for i in range(model.nbody):
        if model.body_parentid[i] == 0 and i > 0:
            total += 1
            if data.xpos[i][2] > -0.5:
                stable += 1
    print(f"\nPhysics stability: {stable}/{total} objects stable ({100*stable/max(total,1):.0f}%)")


if __name__ == '__main__':
    render_gif()
