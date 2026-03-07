"""
Demo: Hand-designed living room layout with real 3D assets.
Layout is manually crafted for realistic furniture arrangement,
not mathematically optimized.

Room: 5m x 4m living room
Design concept: L-shaped seating area facing TV wall
"""

import os
import sys
import json
import numpy as np

os.environ['MUJOCO_GL'] = 'osmesa'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.assets.asset_manager import AssetManager
from src.layout.scene_graph_builder import build_scene_graph


def design_living_room():
    """
    Hand-designed living room layout.
    
    Room layout concept (5m x 4m, Y=0 is south wall, Y=4 is north wall):
    
        North Wall (Y=4)
    ┌──────────────────────────┐
    │  [Lamp]   [Sofa]        │
    │                         │
    │ [Armchair]  [Coffee     │
    │              Table]     │
    │         [Carpet]        │
    │                         │
    │  [Cabinet] [TV]         │
    └──────────────────────────┘
        South Wall (Y=0)
    
    Design decisions:
    - Sofa against north wall, centered, facing south toward TV
    - TV against south wall, centered, facing north toward sofa
    - Coffee table between sofa and TV, in the center
    - Armchair on the west side, angled 45° toward coffee table
    - Floor lamp behind/beside sofa on the west end
    - Cabinet in southeast corner against south wall
    - Carpet centered under the seating area
    """
    
    print("=" * 60)
    print("SceneCraft-LLM: Hand-Designed Living Room")
    print("=" * 60)
    
    mgr = AssetManager()
    room_w, room_d = 5.0, 4.0
    
    # === HAND-DESIGNED LAYOUT ===
    # Each item: category, [x, y, z], rotation_degrees, asset_index
    # Rotation: 0° = facing +Y (north), 90° = facing +X (east), 
    #          180° = facing -Y (south), 270° = facing -X (west)
    
    layout = [
        {
            'category': 'sofa',
            'position': [2.5, 3.50, 0.0],   # Centered, flush against north wall
            'rotation': 180,                  # Facing south (toward TV)
            'index': 1,                       # Try asset variant 1
            'note': 'Main sofa against north wall, facing TV',
        },
        {
            'category': 'coffee_table',
            'position': [2.5, 2.4, 0.0],    # 1.1m in front of sofa
            'rotation': 0,
            'index': 1,
            'note': 'Coffee table centered between sofa and TV',
        },
        {
            'category': 'armchair',
            'position': [0.8, 2.8, 0.0],    # West side, closer to sofa for conversation
            'rotation': 315,                  # Facing SE toward coffee table area  
            'index': 1,
            'note': 'Armchair forming L-shape conversation area',
        },
        {
            'category': 'television_set',
            'position': [2.5, 0.10, 0.0],   # Centered, flush against south wall
            'rotation': 0,                    # Facing north (toward sofa)
            'index': 1,
            'note': 'TV centered on south wall, facing sofa',
        },
        {
            'category': 'table_lamp',
            'position': [4.2, 3.6, 0.0],    # East end of sofa (right side)
            'rotation': 0,
            'index': 1,
            'note': 'Table lamp at east end of sofa',
        },
        {
            'category': 'cabinet',
            'position': [0.5, 0.3, 0.0],    # Southwest corner (away from TV)
            'rotation': 0,
            'index': 1,
            'note': 'Storage cabinet in southwest corner',
        },
        {
            'category': 'runner_(carpet)',
            'position': [2.5, 2.3, 0.0],    # Centered under seating area
            'rotation': 0,
            'index': 0,
            'note': 'Area rug centered under conversation zone',
        },
    ]
    
    print(f"\nRoom: {room_w}m × {room_d}m")
    print(f"Design: L-shaped seating, TV wall facing\n")
    
    # Build scene objects with real dimensions
    scene_objects = []
    for item in layout:
        cat = item['category']
        dims = mgr.get_standard_dimensions(cat)
        
        obj = {
            'category': cat,
            'position': item['position'],
            'rotation': item['rotation'],
            'dimensions': dims,
            'index': item['index'],
        }
        scene_objects.append(obj)
        print(f"  {cat}: ({item['position'][0]:.1f}, {item['position'][1]:.1f}) "
              f"rot={item['rotation']}° | {item['note']}")
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Build scene graph
    print("\n--- Scene Graph ---")
    sg_path = os.path.join(output_dir, 'scene_graph.json')
    sg_builder = build_scene_graph(scene_objects, room_w, room_d, output_path=sg_path)
    
    # Export MuJoCo scene
    print("\n--- Building MuJoCo Scene ---")
    mesh_xml_parts = []
    body_xml_parts = []
    
    # Color scheme for a cozy living room
    colors = {
        'sofa': '0.25 0.35 0.55 1',       # Navy blue
        'coffee_table': '0.55 0.35 0.20 1', # Walnut brown
        'armchair': '0.60 0.55 0.45 1',     # Warm beige
        'television_set': '0.15 0.15 0.18 1', # Dark gray/black
        'table_lamp': '0.85 0.75 0.55 1',   # Warm gold
        'cabinet': '0.45 0.30 0.18 1',      # Dark wood
        'runner_(carpet)': '0.75 0.68 0.58 1', # Cream/tan
    }
    
    for i, item in enumerate(scene_objects):
        cat = item['category']
        safe_cat = cat.replace('(', '').replace(')', '').replace(' ', '_')
        name = f"{safe_cat}_{i}"
        
        # Export real mesh as STL
        export = mgr.export_for_mujoco(cat, item['index'])
        rgba = colors.get(cat, '0.6 0.5 0.4 1')
        
        if export:
            stl_path = os.path.abspath(export['stl_path'])
            real_dims = export['dimensions']
            
            mesh_xml_parts.append(f'    <mesh name="{name}_mesh" file="{stl_path}"/>')
            
            # The mesh has bottom at Z=0 and is centered at XY origin.
            # MuJoCo body pos is the frame origin, mesh vertices are in body frame.
            # So body Z=0 means mesh bottom at Z=0 (on the floor). Perfect.
            z_pos = 0.0
            
            # Rotation quaternion (around Z axis)
            rot_rad = np.radians(item['rotation'])
            qw = np.cos(rot_rad / 2)
            qz = np.sin(rot_rad / 2)
            
            # Mass based on category
            mass = {'sofa': 40, 'coffee_table': 15, 'armchair': 25, 
                    'television_set': 8, 'table_lamp': 3, 'cabinet': 30,
                    'runner_(carpet)': 5}.get(cat, 10)
            
            body_xml_parts.append(f'''    <body name="{name}" pos="{item['position'][0]:.3f} {item['position'][1]:.3f} {z_pos:.3f}" quat="{qw:.4f} 0 0 {qz:.4f}">
      <freejoint/>
      <geom type="mesh" mesh="{name}_mesh" rgba="{rgba}" mass="{mass}"/>
    </body>''')
            
            print(f"  {name}: mesh ({real_dims['width']:.2f}×{real_dims['depth']:.2f}×{real_dims['height']:.2f}m)")
        else:
            # Fallback to colored box
            dims = item['dimensions']
            z_pos = item['position'][2] + dims['height'] / 2
            rot_rad = np.radians(item['rotation'])
            qw = np.cos(rot_rad / 2)
            qz = np.sin(rot_rad / 2)
            
            mass = 15
            body_xml_parts.append(f'''    <body name="{name}" pos="{item['position'][0]:.3f} {item['position'][1]:.3f} {z_pos:.3f}" quat="{qw:.4f} 0 0 {qz:.4f}">
      <freejoint/>
      <geom type="box" size="{dims['width']/2:.3f} {dims['depth']/2:.3f} {dims['height']/2:.3f}" rgba="{rgba}" mass="{mass}"/>
    </body>''')
            print(f"  {name}: box fallback ({dims['width']:.2f}×{dims['depth']:.2f}×{dims['height']:.2f}m)")
    
    meshes_xml = '\n'.join(mesh_xml_parts)
    bodies_xml = '\n'.join(body_xml_parts)
    
    mujoco_xml = f'''<?xml version="1.0" ?>
<mujoco model="scenecraft_designed_living_room">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  
  <asset>
    <texture type="2d" name="floor_tex" builtin="checker" rgb1="0.92 0.87 0.78" rgb2="0.82 0.77 0.68" width="200" height="200"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="8 6"/>
    <texture type="2d" name="wall_tex" builtin="flat" rgb1="0.95 0.94 0.91" width="100" height="100"/>
    <material name="wall_mat" texture="wall_tex"/>
{meshes_xml}
  </asset>
  
  <worldbody>
    <!-- Floor: warm wood texture -->
    <geom type="plane" size="{room_w/2:.1f} {room_d/2:.1f} 0.01" material="floor_mat" pos="{room_w/2:.1f} {room_d/2:.1f} 0"/>
    
    <!-- Walls: low border (won't block top-down view) -->
    <geom type="box" size="{room_w/2:.1f} 0.04 0.12" pos="{room_w/2:.1f} -0.04 0.12" rgba="0.90 0.88 0.85 1"/>
    <geom type="box" size="{room_w/2:.1f} 0.04 0.12" pos="{room_w/2:.1f} {room_d+0.04:.1f} 0.12" rgba="0.90 0.88 0.85 1"/>
    <geom type="box" size="0.04 {room_d/2:.1f} 0.12" pos="-0.04 {room_d/2:.1f} 0.12" rgba="0.90 0.88 0.85 1"/>
    <geom type="box" size="0.04 {room_d/2:.1f} 0.12" pos="{room_w+0.04:.1f} {room_d/2:.1f} 0.12" rgba="0.90 0.88 0.85 1"/>
    
    <!-- Ambient light from ceiling -->
    <light pos="{room_w/2:.1f} {room_d/2:.1f} 2.8" dir="0 0 -1" diffuse="0.9 0.85 0.8" specular="0.2 0.2 0.2" castshadow="true"/>
    <light pos="1.0 1.0 2.5" dir="0.3 0.3 -1" diffuse="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    
    <!-- Furniture -->
{bodies_xml}
  </worldbody>
</mujoco>'''
    
    xml_path = os.path.join(output_dir, 'scene_with_assets.xml')
    with open(xml_path, 'w') as f:
        f.write(mujoco_xml)
    print(f"\nMuJoCo XML: {xml_path}")
    
    # Save scene data
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    
    scene_data = {
        'room': {'width': room_w, 'depth': room_d},
        'design': 'L-shaped seating area facing TV wall',
        'objects': scene_objects,
        'asset_source': 'objaverse_lvis',
    }
    json_path = os.path.join(output_dir, 'scene_with_assets.json')
    with open(json_path, 'w') as f:
        json.dump(scene_data, f, indent=2, default=make_serializable)
    
    # Render floorplan
    print("\n--- Rendering 2D Floorplan ---")
    render_floorplan(scene_objects, room_w, room_d, output_dir)
    
    # Run MuJoCo simulation
    print("\n--- MuJoCo Simulation ---")
    run_simulation(xml_path, output_dir, room_w, room_d)
    
    # Render GIF
    print("\n--- Rendering GIF ---")
    render_gif(xml_path, output_dir, room_w, room_d)
    
    print("\n" + "=" * 60)
    print("✅ All outputs saved to outputs/")
    print("=" * 60)


def render_floorplan(objects, room_w, room_d, output_dir):
    """Render a clean 2D floorplan."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Room
    ax.add_patch(patches.Rectangle((0, 0), room_w, room_d, 
                 linewidth=3, edgecolor='#333', facecolor='#FFF8F0'))
    
    # Wall labels
    ax.text(room_w/2, -0.3, 'South Wall (TV)', ha='center', fontsize=10, color='gray')
    ax.text(room_w/2, room_d+0.3, 'North Wall (Sofa)', ha='center', fontsize=10, color='gray')
    
    colors = {
        'sofa': '#3A5BA0', 'coffee_table': '#8B5E3C', 'armchair': '#9B8B7A',
        'television_set': '#2C2C2E', 'table_lamp': '#D4A843', 'cabinet': '#6B4226',
        'runner_(carpet)': '#DDD0C0',
    }
    
    for obj in objects:
        cat = obj['category']
        pos = obj['position']
        dims = obj['dimensions']
        rot = obj.get('rotation', 0)
        
        color = colors.get(cat, '#888')
        alpha = 0.3 if 'carpet' in cat or 'runner' in cat else 0.8
        
        w, d = dims['width'], dims['depth']
        if rot in [90, 270]:
            w, d = d, w
        
        x = pos[0] - w / 2
        y = pos[1] - d / 2
        
        rect = patches.Rectangle((x, y), w, d, linewidth=1.5,
                                  edgecolor='#333', facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        
        # Direction indicator (small triangle showing facing direction)
        if cat not in ['runner_(carpet)', 'table_lamp']:
            rad = np.radians(rot)
            arrow_len = min(w, d) * 0.3
            ax.annotate('', xy=(pos[0] + arrow_len*np.sin(rad), pos[1] + arrow_len*np.cos(rad)),
                       xytext=(pos[0], pos[1]),
                       arrowprops=dict(arrowstyle='->', color='white', lw=2))
        
        label = cat.replace('_', '\n').replace('(carpet)', '').replace('television\nset', 'TV')
        fontcolor = 'white' if alpha > 0.5 and cat != 'runner_(carpet)' else '#333'
        ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=7,
                fontweight='bold', color=fontcolor)
    
    ax.set_xlim(-0.5, room_w + 0.5)
    ax.set_ylim(-0.5, room_d + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Width (m)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(f'SceneCraft-LLM: Living Room ({room_w}m × {room_d}m)\n'
                 f'Hand-Designed Layout with Objaverse 3D Assets', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'floorplan_assets.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def run_simulation(xml_path, output_dir, room_w, room_d):
    """Run MuJoCo and capture frames."""
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    # Top-down camera
    cam = mujoco.MjvCamera()
    cam.lookat[0] = room_w / 2
    cam.lookat[1] = room_d / 2
    cam.lookat[2] = 0.3
    cam.distance = 7.0
    cam.elevation = -90
    cam.azimuth = 0
    
    # Perspective camera (high angle, no wall occlusion)
    cam_p = mujoco.MjvCamera()
    cam_p.lookat[0] = room_w / 2
    cam_p.lookat[1] = room_d / 2
    cam_p.lookat[2] = 0.3
    cam_p.distance = 7.5
    cam_p.elevation = -55
    cam_p.azimuth = 135
    
    dt = model.opt.timestep
    
    try:
        font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf', 14)
    except:
        font = ImageFont.load_default()
    
    for t_target, label in [(0.0, 'start'), (1.5, 'mid'), (3.0, 'end')]:
        steps = int(t_target / dt) if t_target > 0 else 0
        while data.time < t_target - dt/2:
            mujoco.mj_step(model, data)
        
        # Top-down
        renderer.update_scene(data, cam)
        img = Image.fromarray(renderer.render())
        p = os.path.join(output_dir, f'sim_assets_top_{label}.png')
        img.save(p)
        print(f"  Top-down t={t_target}s: {p}")
        
        # Perspective
        renderer.update_scene(data, cam_p)
        img2 = Image.fromarray(renderer.render())
        p2 = os.path.join(output_dir, f'sim_assets_persp_{label}.png')
        img2.save(p2)
        print(f"  Perspective t={t_target}s: {p2}")
    
    # Stability check
    stable = sum(1 for i in range(model.nbody) 
                 if model.body_parentid[i] == 0 and i > 0 and data.xpos[i][2] > -0.5)
    total = sum(1 for i in range(model.nbody) if model.body_parentid[i] == 0 and i > 0)
    print(f"  Physics stability: {stable}/{total} ({100*stable/max(total,1):.0f}%)")


def render_gif(xml_path, output_dir, room_w, room_d):
    """Render simulation GIF with dual view."""
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    W, H = 640, 480
    renderer = mujoco.Renderer(model, height=H, width=W)
    
    cam_top = mujoco.MjvCamera()
    cam_top.lookat[0] = room_w / 2
    cam_top.lookat[1] = room_d / 2
    cam_top.lookat[2] = 0.3
    cam_top.distance = 7.0
    cam_top.elevation = -90
    cam_top.azimuth = 0
    
    cam_persp = mujoco.MjvCamera()
    cam_persp.lookat[0] = room_w / 2
    cam_persp.lookat[1] = room_d / 2
    cam_persp.lookat[2] = 0.4
    cam_persp.distance = 7.5
    cam_persp.elevation = -50
    
    try:
        font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf', 16)
    except:
        font = ImageFont.load_default()
    
    fps = 15
    duration = 4.0
    dt = model.opt.timestep
    steps_per_frame = max(1, int(1.0 / (fps * dt)))
    total_frames = int(duration * fps)
    
    frames_top = []
    frames_dual = []
    
    for fi in range(total_frames):
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        
        t = fi / fps
        cam_persp.azimuth = 120 + fi * (180.0 / total_frames)
        
        renderer.update_scene(data, cam_top)
        top = Image.fromarray(renderer.render())
        
        renderer.update_scene(data, cam_persp)
        persp = Image.fromarray(renderer.render())
        
        # Top-down only frame
        top_labeled = top.copy()
        draw = ImageDraw.Draw(top_labeled)
        draw.rectangle([(5, 5), (130, 28)], fill=(0, 0, 0))
        draw.text((10, 7), f't = {t:.1f}s', fill=(255, 255, 255), font=font)
        frames_top.append(top_labeled)
        
        # Dual view frame
        combined = Image.new('RGB', (W * 2 + 10, H + 30), (255, 255, 255))
        draw2 = ImageDraw.Draw(combined)
        draw2.text((10, 5), f'Top-Down  t={t:.1f}s', fill=(0, 0, 0), font=font)
        draw2.text((W + 20, 5), f'Perspective (rotating)', fill=(0, 0, 0), font=font)
        combined.paste(top, (0, 28))
        combined.paste(persp, (W + 10, 28))
        frames_dual.append(combined)
        
        if fi % 15 == 0:
            print(f"  Frame {fi}/{total_frames}")
    
    frame_duration = int(1000 / fps)
    
    top_path = os.path.join(output_dir, 'simulation_topdown.gif')
    frames_top[0].save(top_path, save_all=True, append_images=frames_top[1:],
                        duration=frame_duration, loop=0, optimize=True)
    print(f"  Top-down GIF: {top_path} ({os.path.getsize(top_path)//1024}KB)")
    
    dual_path = os.path.join(output_dir, 'simulation_assets.gif')
    frames_dual[0].save(dual_path, save_all=True, append_images=frames_dual[1:],
                         duration=frame_duration, loop=0, optimize=True)
    print(f"  Dual-view GIF: {dual_path} ({os.path.getsize(dual_path)//1024}KB)")


if __name__ == '__main__':
    design_living_room()
