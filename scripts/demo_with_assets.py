"""
Demo: Generate a living room scene using real 3D furniture assets from Objaverse.
Renders 2D floorplan + MuJoCo 3D physics simulation.
"""

import os
import sys
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.assets.asset_manager import AssetManager
from src.layout.scene_graph import SceneGraph
from src.evaluation.metrics import SceneMetrics


def generate_living_room_with_assets():
    """Generate a living room scene using real 3D assets."""
    
    print("=" * 60)
    print("SceneCraft-LLM: Living Room with Real 3D Assets")
    print("=" * 60)
    
    # Initialize asset manager
    mgr = AssetManager()
    print(f"\nLoaded asset catalog: {len(mgr.get_categories())} categories")
    
    # Define room
    room_width, room_depth = 5.0, 4.0
    
    # Define furniture layout (from LLM planner output)
    furniture_layout = [
        {'category': 'sofa',           'position': [2.5, 3.5, 0.0], 'rotation': 0,   'index': 0},
        {'category': 'coffee_table',   'position': [2.5, 2.3, 0.0], 'rotation': 0,   'index': 0},
        {'category': 'armchair',       'position': [0.8, 2.5, 0.0], 'rotation': 90,  'index': 0},
        {'category': 'television_set', 'position': [2.5, 0.3, 0.0], 'rotation': 0,   'index': 0},
        {'category': 'table_lamp',     'position': [4.3, 3.5, 0.0], 'rotation': 0,   'index': 0},
        {'category': 'cabinet',        'position': [4.5, 0.5, 0.0], 'rotation': 0,   'index': 0},
        {'category': 'runner_(carpet)','position': [2.5, 2.0, 0.0], 'rotation': 0,   'index': 0},
    ]
    
    # Build scene graph
    sg = SceneGraph(width=room_width, length=room_depth)
    
    scene_objects = []
    for i, item in enumerate(furniture_layout):
        cat = item['category']
        dims = mgr.get_standard_dimensions(cat)
        
        obj_id = f"{cat}_{i}"
        sg.add_object(
            object_id=obj_id,
            category=cat,
            position=(item['position'][0], item['position'][1]),
            rotation=item['rotation'],
            size=dims,
        )
        
        obj = {
            'category': cat,
            'position': item['position'],
            'rotation': item['rotation'],
            'dimensions': dims,
        }
        scene_objects.append(obj)
        print(f"  Placed {cat}: {dims['width']:.1f}x{dims['depth']:.1f}x{dims['height']:.1f}m "
              f"at ({item['position'][0]:.1f}, {item['position'][1]:.1f})")
    
    # Evaluate layout
    print("\n--- Layout Evaluation ---")
    gt_categories = ['sofa', 'coffee_table', 'armchair', 'television_set', 
                     'table_lamp', 'cabinet', 'runner_(carpet)']
    pred_categories = [obj['category'] for obj in scene_objects]
    evaluator = SceneMetrics()
    metrics = evaluator.evaluate(sg)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    
    # Export MuJoCo STL meshes
    print("\n--- Exporting 3D Meshes for MuJoCo ---")
    stl_exports = {}
    mesh_xml_parts = []
    body_xml_parts = []
    
    for i, item in enumerate(furniture_layout):
        cat = item['category']
        safe_cat = cat.replace('(', '').replace(')', '').replace(' ', '_')
        name = f"{safe_cat}_{i}"
        
        # Export STL
        export = mgr.export_for_mujoco(cat, item['index'])
        if export:
            stl_exports[name] = export
            real_dims = export['dimensions']
            print(f"  {name}: STL exported ({real_dims['width']:.3f}x"
                  f"{real_dims['depth']:.3f}x{real_dims['height']:.3f}m)")
            
            # MuJoCo XML parts
            stl_path = os.path.abspath(export['stl_path'])
            mesh_xml_parts.append(f'    <mesh name="{name}_mesh" file="{stl_path}" scale="1 1 1"/>')
            
            # Position: center of object, z adjusted for ground contact
            z_pos = item['position'][2] + real_dims['height'] / 2
            rot_rad = np.radians(item['rotation'])
            qw, qx, qy, qz = np.cos(rot_rad/2), 0, 0, np.sin(rot_rad/2)
            
            body_xml_parts.append(f'''    <body name="{name}" pos="{item['position'][0]:.3f} {item['position'][1]:.3f} {z_pos:.3f}" quat="{qw:.4f} {qx:.4f} {qy:.4f} {qz:.4f}">
      <freejoint/>
      <geom type="mesh" mesh="{name}_mesh" rgba="{np.random.uniform(0.3,0.8):.2f} {np.random.uniform(0.3,0.7):.2f} {np.random.uniform(0.2,0.6):.2f} 1" mass="15"/>
    </body>''')
        else:
            # Fallback to box
            dims = mgr.get_standard_dimensions(cat)
            z_pos = item['position'][2] + dims['height'] / 2
            body_xml_parts.append(f'''    <body name="{name}" pos="{item['position'][0]:.3f} {item['position'][1]:.3f} {z_pos:.3f}">
      <freejoint/>
      <geom type="box" size="{dims['width']/2:.3f} {dims['depth']/2:.3f} {dims['height']/2:.3f}" rgba="0.6 0.4 0.2 1" mass="15"/>
    </body>''')
            print(f"  {name}: using box fallback")
    
    # Generate MuJoCo XML
    meshes_xml = '\n'.join(mesh_xml_parts)
    bodies_xml = '\n'.join(body_xml_parts)
    
    mujoco_xml = f'''<?xml version="1.0" ?>
<mujoco model="scenecraft_living_room">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  
  <asset>
    <texture type="2d" name="floor_tex" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="0.7 0.7 0.7" width="100" height="100"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="4 4"/>
    <texture type="2d" name="wall_tex" builtin="flat" rgb1="0.95 0.93 0.88" width="100" height="100"/>
    <material name="wall_mat" texture="wall_tex"/>
{meshes_xml}
  </asset>
  
  <worldbody>
    <!-- Floor -->
    <geom type="plane" size="{room_width/2:.1f} {room_depth/2:.1f} 0.01" material="floor_mat" pos="{room_width/2:.1f} {room_depth/2:.1f} 0"/>
    
    <!-- Walls (low height so top-down view is clear) -->
    <geom type="box" size="{room_width/2:.1f} 0.05 0.15" pos="{room_width/2:.1f} -0.05 0.15" rgba="0.85 0.82 0.78 1"/>
    <geom type="box" size="{room_width/2:.1f} 0.05 0.15" pos="{room_width/2:.1f} {room_depth+0.05:.1f} 0.15" rgba="0.85 0.82 0.78 1"/>
    <geom type="box" size="0.05 {room_depth/2:.1f} 0.15" pos="-0.05 {room_depth/2:.1f} 0.15" rgba="0.85 0.82 0.78 1"/>
    <geom type="box" size="0.05 {room_depth/2:.1f} 0.15" pos="{room_width+0.05:.1f} {room_depth/2:.1f} 0.15" rgba="0.85 0.82 0.78 1"/>
    
    <!-- Light -->
    <light pos="{room_width/2:.1f} {room_depth/2:.1f} 2.8" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
    
    <!-- Furniture with real meshes -->
{bodies_xml}
  </worldbody>
</mujoco>'''
    
    # Save MuJoCo XML
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    xml_path = os.path.join(output_dir, 'scene_with_assets.xml')
    with open(xml_path, 'w') as f:
        f.write(mujoco_xml)
    print(f"\nMuJoCo XML saved to: {xml_path}")
    
    # Save scene data
    scene_data = {
        'room': {'width': room_width, 'depth': room_depth},
        'objects': scene_objects,
        'metrics': metrics,
        'asset_source': 'objaverse_lvis',
    }
    json_path = os.path.join(output_dir, 'scene_with_assets.json')
    
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(json_path, 'w') as f:
        json.dump(scene_data, f, indent=2, default=make_serializable)
    print(f"Scene data saved to: {json_path}")
    
    # Render 2D floorplan
    print("\n--- Rendering 2D Floorplan ---")
    render_floorplan(scene_objects, room_width, room_depth, output_dir)
    
    # Run MuJoCo simulation
    print("\n--- Running MuJoCo Simulation ---")
    try:
        run_mujoco_sim(xml_path, output_dir)
    except Exception as e:
        print(f"MuJoCo simulation error: {e}")
        print("(This may happen if mesh files have issues. Falling back to screenshot only.)")
    
    print("\n" + "=" * 60)
    print("Done! Check outputs/ for results.")
    print("=" * 60)


def render_floorplan(objects, room_w, room_d, output_dir):
    """Render a 2D floorplan with labeled furniture."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Room boundary
    ax.add_patch(patches.Rectangle((0, 0), room_w, room_d, 
                                    linewidth=3, edgecolor='black', facecolor='#F5F5DC'))
    
    # Color map for categories
    colors = {
        'sofa': '#4169E1', 'coffee_table': '#8B4513', 'armchair': '#6A5ACD',
        'television_set': '#2F4F4F', 'table_lamp': '#FFD700', 'cabinet': '#A0522D',
        'runner_(carpet)': '#DEB887', 'chair': '#7B68EE', 'desk': '#CD853F',
        'bed': '#DC143C', 'dresser': '#8B7355', 'lamp': '#FFA500',
        'wardrobe': '#696969', 'stool': '#9370DB', 'bench': '#D2691E',
    }
    
    for obj in objects:
        cat = obj['category']
        pos = obj['position']
        dims = obj['dimensions']
        rot = obj.get('rotation', 0)
        
        color = colors.get(cat, '#888888')
        alpha = 0.3 if 'carpet' in cat or 'runner' in cat else 0.7
        
        # Calculate corner position
        w, d = dims['width'], dims['depth']
        if rot in [90, 270]:
            w, d = d, w
        
        x = pos[0] - w / 2
        y = pos[1] - d / 2
        
        rect = patches.Rectangle((x, y), w, d, linewidth=1.5, 
                                  edgecolor='black', facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        
        # Label
        label = cat.replace('_', '\n').replace('(carpet)', '')
        ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=7,
                fontweight='bold', color='white' if alpha > 0.5 else 'black')
    
    ax.set_xlim(-0.5, room_w + 0.5)
    ax.set_ylim(-0.5, room_d + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Width (m)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(f'SceneCraft-LLM: Living Room Layout ({room_w}m × {room_d}m)\n'
                 f'Using Objaverse 3D Assets', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'floorplan_assets.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved floorplan: {path}")


def run_mujoco_sim(xml_path, output_dir):
    """Run MuJoCo simulation and capture frames."""
    os.environ['MUJOCO_GL'] = 'osmesa'
    import mujoco
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Setup renderer
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    # Camera setup: top-down view to avoid wall occlusion
    cam = mujoco.MjvCamera()
    cam.lookat[0] = 2.5   # room center X
    cam.lookat[1] = 2.0   # room center Y
    cam.lookat[2] = 0.3   # slightly above floor
    cam.distance = 7.5
    cam.elevation = -90    # straight down (top-down)
    cam.azimuth = 0
    
    # Also set up a 45-degree perspective camera
    cam_persp = mujoco.MjvCamera()
    cam_persp.lookat[0] = 2.5
    cam_persp.lookat[1] = 2.0
    cam_persp.lookat[2] = 0.4
    cam_persp.distance = 8.0
    cam_persp.elevation = -55   # high angle, looking down
    cam_persp.azimuth = 135
    
    # Simulate and capture
    frames = []
    frame_times = [0.0, 1.5, 3.0]
    frame_labels = ['start', 'mid', 'end']
    
    from PIL import Image
    
    dt = model.opt.timestep
    total_steps = int(3.0 / dt)
    
    for step in range(total_steps + 1):
        mujoco.mj_step(model, data)
        
        t = step * dt
        for ft, label in zip(frame_times, frame_labels):
            if abs(t - ft) < dt / 2:
                # Top-down view
                renderer.update_scene(data, cam)
                pixels = renderer.render()
                img = Image.fromarray(pixels)
                img_path = os.path.join(output_dir, f'sim_assets_top_{label}.png')
                img.save(img_path)
                frames.append((f'top_{label}', img_path))
                print(f"  Top-down t={ft}s saved: {img_path}")
                
                # Perspective view
                renderer.update_scene(data, cam_persp)
                pixels2 = renderer.render()
                img2 = Image.fromarray(pixels2)
                img2_path = os.path.join(output_dir, f'sim_assets_persp_{label}.png')
                img2.save(img2_path)
                print(f"  Perspective t={ft}s saved: {img2_path}")
    
    # Check physics stability
    stable_count = 0
    total_objects = 0
    for i in range(model.nbody):
        if model.body_parentid[i] == 0 and i > 0:  # Direct children of world
            total_objects += 1
            z_pos = data.xpos[i][2]
            if z_pos > -0.5:  # Not fallen through floor
                stable_count += 1
    
    stability = stable_count / max(total_objects, 1)
    print(f"  Physics stability: {stability:.1%} ({stable_count}/{total_objects} objects stable)")
    
    # Create combined image: top row = top-down, bottom row = perspective
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf', 14)
    except:
        font = ImageFont.load_default()
    
    # Collect top-down frames
    top_frames = [(l, p) for l, p in frames if l.startswith('top_')]
    
    if len(top_frames) >= 3:
        top_imgs = [Image.open(fp) for _, fp in top_frames[:3]]
        persp_imgs = []
        for label in ['start', 'mid', 'end']:
            pp = os.path.join(output_dir, f'sim_assets_persp_{label}.png')
            if os.path.exists(pp):
                persp_imgs.append(Image.open(pp))
        
        col_w = top_imgs[0].width
        col_h = top_imgs[0].height
        label_h = 25
        gap = 5
        
        n_cols = 3
        total_w = col_w * n_cols + gap * (n_cols - 1)
        n_rows = 2 if persp_imgs else 1
        total_h = (col_h + label_h) * n_rows + gap * (n_rows - 1)
        
        combined = Image.new('RGB', (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(combined)
        
        labels_top = ['Top-down t=0s', 'Top-down t=1.5s', 'Top-down t=3.0s']
        labels_persp = ['Perspective t=0s', 'Perspective t=1.5s', 'Perspective t=3.0s']
        
        for i, (img, label) in enumerate(zip(top_imgs, labels_top)):
            x = i * (col_w + gap)
            draw.text((x + 5, 3), label, fill=(0, 0, 0), font=font)
            combined.paste(img, (x, label_h))
        
        if persp_imgs:
            y_offset = col_h + label_h + gap
            for i, (img, label) in enumerate(zip(persp_imgs, labels_persp)):
                x = i * (col_w + gap)
                draw.text((x + 5, y_offset + 3), label, fill=(0, 0, 0), font=font)
                combined.paste(img, (x, y_offset + label_h))
        
        combined_path = os.path.join(output_dir, 'sim_assets_combined.png')
        combined.save(combined_path)
        print(f"  Combined simulation image: {combined_path}")


if __name__ == '__main__':
    generate_living_room_with_assets()
