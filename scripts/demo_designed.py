"""
Demo: Hand-designed living room layout with real 3D assets.
Iteration 3: Fixed orientations, realistic materials, proper placement.
"""

import os
import sys
import json
import itertools
import numpy as np

os.environ['MUJOCO_GL'] = 'osmesa'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.assets.asset_manager import AssetManager
from src.layout.scene_graph_builder import build_scene_graph


# Realistic MuJoCo material definitions per category
MATERIAL_DEFS = {
    'sofa': {
        'texture': 'builtin="gradient" rgb1="0.22 0.30 0.50" rgb2="0.18 0.25 0.42" width="128" height="128"',
        'material': 'texrepeat="3 3" specular="0.1" shininess="0.02" reflectance="0.0"',
        'desc': 'Navy fabric'
    },
    'coffee_table': {
        'texture': 'builtin="checker" rgb1="0.50 0.32 0.18" rgb2="0.45 0.28 0.14" width="256" height="256"',
        'material': 'texrepeat="6 6" specular="0.4" shininess="0.3" reflectance="0.05"',
        'desc': 'Walnut wood'
    },
    'armchair': {
        'texture': 'builtin="gradient" rgb1="0.58 0.50 0.38" rgb2="0.52 0.44 0.32" width="128" height="128"',
        'material': 'texrepeat="3 3" specular="0.1" shininess="0.02" reflectance="0.0"',
        'desc': 'Warm beige fabric'
    },
    'television_set': {
        'texture': 'builtin="flat" rgb1="0.08 0.08 0.10" width="64" height="64"',
        'material': 'specular="0.8" shininess="0.9" reflectance="0.2"',
        'desc': 'Black glossy'
    },
    'table_lamp': {
        'texture': 'builtin="gradient" rgb1="0.90 0.82 0.60" rgb2="0.80 0.72 0.50" width="64" height="64"',
        'material': 'specular="0.3" shininess="0.2" reflectance="0.0"',
        'desc': 'Warm brass'
    },
    'cabinet': {
        'texture': 'builtin="checker" rgb1="0.40 0.25 0.12" rgb2="0.35 0.20 0.08" width="256" height="256"',
        'material': 'texrepeat="8 8" specular="0.3" shininess="0.2" reflectance="0.03"',
        'desc': 'Dark oak wood'
    },
    'runner_(carpet)': {
        'texture': 'builtin="checker" rgb1="0.78 0.70 0.58" rgb2="0.72 0.64 0.52" width="256" height="256"',
        'material': 'texrepeat="4 4" specular="0.0" shininess="0.0" reflectance="0.0"',
        'desc': 'Woven cream'
    },
}


def design_living_room():
    """
    Hand-designed living room layout, iteration 3.
    
    Room: 5m × 4m
    
        North Wall (Y=4)
    ┌─────────────────────────────┐
    │  [Lamp]   [====Sofa====]   │
    │                             │
    │  [Arm-   [Coffee Table]    │
    │  chair]                     │
    │          [===Carpet===]     │
    │                             │
    │  [Cabinet]    [TV]          │
    └─────────────────────────────┘
        South Wall (Y=0)
    """
    
    print("=" * 60)
    print("SceneCraft-LLM: Hand-Designed Living Room (Iteration 3)")
    print("=" * 60)
    
    mgr = AssetManager()
    room_w, room_d = 5.0, 4.0
    
    layout = [
        {
            'category': 'sofa',
            'position': [2.5, 3.50, 0.0],
            'rotation': 180,
            'index': 1,
            'note': 'Sofa centered against north wall, facing south',
        },
        {
            'category': 'coffee_table',
            'position': [2.5, 2.3, 0.0],
            'rotation': 0,
            'index': 1,
            'note': 'Coffee table centered, 1.2m in front of sofa',
        },
        {
            'category': 'armchair',
            'position': [0.8, 2.7, 0.0],
            'rotation': 270,                  # Facing east toward center (orthogonal, stable)
            'index': 1,
            'note': 'Armchair on west side, facing east toward center',
        },
        {
            'category': 'television_set',
            'position': [2.5, 0.15, 0.0],
            'rotation': 0,
            'index': 1,
            'note': 'TV centered on south wall',
        },
        {
            'category': 'table_lamp',
            'position': [4.3, 3.5, 0.0],
            'rotation': 0,
            'index': 1,
            'note': 'Lamp at east end of sofa',
        },
        {
            'category': 'cabinet',
            'position': [0.6, 0.35, 0.0],    # SW corner, with enough clearance
            'rotation': 0,
            'index': 1,
            'note': 'Cabinet in SW corner, clear of walls',
        },
        {
            'category': 'runner_(carpet)',
            'position': [2.5, 2.2, 0.0],
            'rotation': 0,
            'index': 0,
            'note': 'Rug centered under seating area',
        },
    ]
    
    print(f"\nRoom: {room_w}m × {room_d}m\n")
    
    scene_objects = []
    for item in layout:
        cat = item['category']
        dims = mgr.get_standard_dimensions(cat)
        obj = {**item, 'dimensions': dims}
        scene_objects.append(obj)
        print(f"  {cat}: ({item['position'][0]:.1f}, {item['position'][1]:.1f}) "
              f"rot={item['rotation']}° — {item['note']}")
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Scene graph
    print("\n--- Scene Graph ---")
    sg_path = os.path.join(output_dir, 'scene_graph.json')
    build_scene_graph(scene_objects, room_w, room_d, output_path=sg_path)
    
    # Build MuJoCo XML with realistic materials
    print("\n--- Building MuJoCo Scene (with materials) ---")
    
    texture_xml = []
    material_xml = []
    mesh_xml = []
    body_xml = []
    
    for i, item in enumerate(scene_objects):
        cat = item['category']
        safe = cat.replace('(', '').replace(')', '').replace(' ', '_')
        name = f"{safe}_{i}"
        
        # Export mesh
        export = mgr.export_for_mujoco(cat, item['index'])
        
        # Material
        mat_def = MATERIAL_DEFS.get(cat, MATERIAL_DEFS['coffee_table'])
        tex_name = f"tex_{name}"
        mat_name = f"mat_{name}"
        
        texture_xml.append(f'    <texture type="2d" name="{tex_name}" {mat_def["texture"]}/>')
        material_xml.append(f'    <material name="{mat_name}" texture="{tex_name}" {mat_def["material"]}/>')
        
        if export:
            stl_path = os.path.abspath(export['stl_path'])
            dims = export['dimensions']
            mesh_xml.append(f'    <mesh name="{name}_mesh" file="{stl_path}"/>')
            
            rot_rad = np.radians(item['rotation'])
            qw, qz = np.cos(rot_rad/2), np.sin(rot_rad/2)
            
            mass = {'sofa': 45, 'coffee_table': 20, 'armchair': 30,
                    'television_set': 10, 'table_lamp': 4, 'cabinet': 35,
                    'runner_(carpet)': 3}.get(cat, 15)
            
            body_xml.append(f'''    <body name="{name}" pos="{item['position'][0]:.3f} {item['position'][1]:.3f} 0.000" quat="{qw:.4f} 0 0 {qz:.4f}">
      <freejoint/>
      <geom type="mesh" mesh="{name}_mesh" material="{mat_name}" mass="{mass}"/>
    </body>''')
            
            print(f"  ✓ {name}: {dims['width']:.2f}×{dims['depth']:.2f}×{dims['height']:.2f}m | {mat_def['desc']}")
        else:
            dims = item['dimensions']
            rot_rad = np.radians(item['rotation'])
            qw, qz = np.cos(rot_rad/2), np.sin(rot_rad/2)
            
            body_xml.append(f'''    <body name="{name}" pos="{item['position'][0]:.3f} {item['position'][1]:.3f} {dims['height']/2:.3f}" quat="{qw:.4f} 0 0 {qz:.4f}">
      <freejoint/>
      <geom type="box" size="{dims['width']/2:.3f} {dims['depth']/2:.3f} {dims['height']/2:.3f}" material="{mat_name}" mass="15"/>
    </body>''')
            print(f"  ✗ {name}: box fallback | {mat_def['desc']}")
    
    xml = f'''<?xml version="1.0" ?>
<mujoco model="scenecraft_living_room_v3">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  
  <visual>
    <quality shadowsize="4096"/>
    <map znear="0.01"/>
    <global offwidth="1600" offheight="1200"/>
  </visual>
  
  <asset>
    <!-- Floor: warm hardwood -->
    <texture type="2d" name="floor_tex" builtin="checker" 
             rgb1="0.75 0.60 0.42" rgb2="0.70 0.55 0.38" width="256" height="256"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="10 8" 
              specular="0.3" shininess="0.2" reflectance="0.05"/>
    
    <!-- Walls: light cream -->
    <texture type="2d" name="wall_tex" builtin="flat" 
             rgb1="0.94 0.92 0.88" width="64" height="64"/>
    <material name="wall_mat" texture="wall_tex" specular="0.1" shininess="0.05"/>
    
    <!-- Furniture textures -->
{chr(10).join(texture_xml)}
    
    <!-- Furniture materials -->
{chr(10).join(material_xml)}
    
    <!-- Meshes -->
{chr(10).join(mesh_xml)}
  </asset>
  
  <worldbody>
    <!-- Floor -->
    <geom type="plane" size="{room_w/2} {room_d/2} 0.01" material="floor_mat" 
          pos="{room_w/2} {room_d/2} 0"/>
    
    <!-- Wall borders (low, won't occlude top-down view) -->
    <geom type="box" size="{room_w/2} 0.03 0.10" pos="{room_w/2} -0.03 0.10" material="wall_mat"/>
    <geom type="box" size="{room_w/2} 0.03 0.10" pos="{room_w/2} {room_d+0.03} 0.10" material="wall_mat"/>
    <geom type="box" size="0.03 {room_d/2} 0.10" pos="-0.03 {room_d/2} 0.10" material="wall_mat"/>
    <geom type="box" size="0.03 {room_d/2} 0.10" pos="{room_w+0.03} {room_d/2} 0.10" material="wall_mat"/>
    
    <!-- Lighting: warm overhead + fill -->
    <light pos="{room_w/2} {room_d/2} 2.8" dir="0 0 -1" 
           diffuse="0.85 0.80 0.72" specular="0.3 0.3 0.3" castshadow="true"/>
    <light pos="1 1 2.5" dir="0.3 0.3 -1" diffuse="0.25 0.22 0.20" specular="0.1 0.1 0.1"/>
    <light pos="4 3 2.5" dir="-0.3 -0.3 -1" diffuse="0.20 0.20 0.22" specular="0.1 0.1 0.1"/>
    
    <!-- Furniture -->
{chr(10).join(body_xml)}
  </worldbody>
</mujoco>'''
    
    xml_path = os.path.join(output_dir, 'scene_with_assets.xml')
    with open(xml_path, 'w') as f:
        f.write(xml)
    print(f"\n  XML saved: {xml_path}")
    
    # Save JSON
    def ser(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    
    with open(os.path.join(output_dir, 'scene_with_assets.json'), 'w') as f:
        json.dump({'room': {'width': room_w, 'depth': room_d}, 'objects': scene_objects}, 
                  f, indent=2, default=ser)
    
    # Render floorplan
    print("\n--- 2D Floorplan ---")
    render_floorplan(scene_objects, room_w, room_d, output_dir)
    
    # Simulate + render
    print("\n--- MuJoCo Simulation ---")
    simulate_and_render(xml_path, output_dir, room_w, room_d)
    
    # GIF
    print("\n--- GIF ---")
    render_gif(xml_path, output_dir, room_w, room_d)
    
    print("\n" + "=" * 60)
    print("✅ Iteration 3 complete!")
    print("=" * 60)


def render_floorplan(objects, rw, rd, out):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.add_patch(patches.Rectangle((0,0), rw, rd, lw=3, ec='#333', fc='#FFF8F0'))
    
    ax.text(rw/2, -0.3, 'South Wall (TV)', ha='center', fontsize=10, color='gray')
    ax.text(rw/2, rd+0.3, 'North Wall (Sofa)', ha='center', fontsize=10, color='gray')
    
    colors = {
        'sofa': '#3A5BA0', 'coffee_table': '#8B5E3C', 'armchair': '#9B8B7A',
        'television_set': '#2C2C2E', 'table_lamp': '#D4A843', 'cabinet': '#6B4226',
        'runner_(carpet)': '#DDD0C0',
    }
    
    for obj in objects:
        cat, pos, dims = obj['category'], obj['position'], obj['dimensions']
        rot = obj.get('rotation', 0)
        color = colors.get(cat, '#888')
        alpha = 0.3 if 'carpet' in cat else 0.8
        
        w, d = dims['width'], dims['depth']
        if rot in [90, 270]:
            w, d = d, w
        
        rect = patches.Rectangle((pos[0]-w/2, pos[1]-d/2), w, d, lw=1.5,
                                  ec='#333', fc=color, alpha=alpha)
        ax.add_patch(rect)
        
        if cat not in ['runner_(carpet)', 'table_lamp']:
            rad = np.radians(rot)
            al = min(w,d)*0.3
            ax.annotate('', xy=(pos[0]+al*np.sin(rad), pos[1]+al*np.cos(rad)),
                       xytext=(pos[0], pos[1]),
                       arrowprops=dict(arrowstyle='->', color='white', lw=2))
        
        label = cat.replace('_','\n').replace('(carpet)','').replace('television\nset','TV')
        fc = 'white' if alpha > 0.5 and cat != 'runner_(carpet)' else '#333'
        ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=7, fontweight='bold', color=fc)
    
    ax.set_xlim(-0.5, rw+0.5); ax.set_ylim(-0.5, rd+0.5); ax.set_aspect('equal')
    ax.set_xlabel('Width (m)'); ax.set_ylabel('Depth (m)')
    ax.set_title(f'SceneCraft-LLM: Living Room ({rw}×{rd}m)\nWith Objaverse Assets + MuJoCo Materials', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, ls='--')
    plt.tight_layout()
    p = os.path.join(out, 'floorplan_assets.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")


def simulate_and_render(xml_path, out, rw, rd):
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=600, width=800)
    
    cam_top = mujoco.MjvCamera()
    cam_top.lookat[:] = [rw/2, rd/2, 0.3]
    cam_top.distance = 7.0; cam_top.elevation = -90; cam_top.azimuth = 0
    
    cam_p = mujoco.MjvCamera()
    cam_p.lookat[:] = [rw/2, rd/2, 0.4]
    cam_p.distance = 7.0; cam_p.elevation = -40; cam_p.azimuth = 150
    
    try: font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf', 14)
    except: font = ImageFont.load_default()
    
    for t_target, label in [(0.0, 'start'), (2.0, 'mid'), (4.0, 'end')]:
        while data.time < t_target - 0.001:
            mujoco.mj_step(model, data)
        
        renderer.update_scene(data, cam_top)
        Image.fromarray(renderer.render()).save(os.path.join(out, f'sim_assets_top_{label}.png'))
        
        renderer.update_scene(data, cam_p)
        Image.fromarray(renderer.render()).save(os.path.join(out, f'sim_assets_persp_{label}.png'))
        
        print(f"  t={t_target}s: frames saved")
    
    # Stability
    stable = sum(1 for i in range(model.nbody) 
                 if model.body_parentid[i] == 0 and i > 0 and data.xpos[i][2] > -0.5)
    total = sum(1 for i in range(model.nbody) if model.body_parentid[i] == 0 and i > 0)
    print(f"  Stability: {stable}/{total} ({100*stable/max(total,1):.0f}%)")
    
    # Check each object
    for i in range(1, model.nbody):
        if model.body_parentid[i] == 0:
            name = model.body(i).name
            pos = data.xpos[i]
            quat = data.xquat[i]
            # Check uprightness
            w, x, y, z = quat
            up = np.array([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)])
            upright = up[2] > 0.8
            print(f"    {name}: z={pos[2]:.3f} upright={'✓' if upright else '✗'} up={up[2]:.2f}")


def render_gif(xml_path, out, rw, rd):
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    W, H = 800, 600
    renderer = mujoco.Renderer(model, height=H, width=W)
    
    cam_top = mujoco.MjvCamera()
    cam_top.lookat[:] = [rw/2, rd/2, 0.3]
    cam_top.distance = 7.0; cam_top.elevation = -90; cam_top.azimuth = 0
    
    cam_p = mujoco.MjvCamera()
    cam_p.lookat[:] = [rw/2, rd/2, 0.4]
    cam_p.distance = 7.0; cam_p.elevation = -40
    
    try: font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf', 16)
    except: font = ImageFont.load_default()
    
    fps, duration = 15, 4.0
    dt = model.opt.timestep
    spf = max(1, int(1.0/(fps*dt)))
    nf = int(duration*fps)
    
    top_frames, dual_frames = [], []
    
    for fi in range(nf):
        for _ in range(spf):
            mujoco.mj_step(model, data)
        
        t = fi / fps
        cam_p.azimuth = 120 + fi * (200.0/nf)
        
        renderer.update_scene(data, cam_top)
        top = Image.fromarray(renderer.render())
        
        renderer.update_scene(data, cam_p)
        persp = Image.fromarray(renderer.render())
        
        # Top-down labeled
        tl = top.copy()
        d = ImageDraw.Draw(tl)
        d.rectangle([(5,5),(140,28)], fill=(0,0,0))
        d.text((10,7), f't = {t:.1f}s', fill=(255,255,255), font=font)
        top_frames.append(tl)
        
        # Dual
        comb = Image.new('RGB', (W*2+10, H+30), (255,255,255))
        d2 = ImageDraw.Draw(comb)
        d2.text((10,5), f'Top-Down  t={t:.1f}s', fill=(0,0,0), font=font)
        d2.text((W+20,5), 'Perspective (rotating)', fill=(0,0,0), font=font)
        comb.paste(top, (0,28))
        comb.paste(persp, (W+10,28))
        dual_frames.append(comb)
        
        if fi % 15 == 0: print(f"  Frame {fi}/{nf}")
    
    ms = int(1000/fps)
    tp = os.path.join(out, 'simulation_topdown.gif')
    top_frames[0].save(tp, save_all=True, append_images=top_frames[1:], duration=ms, loop=0, optimize=True)
    print(f"  Top-down GIF: {os.path.getsize(tp)//1024}KB")
    
    dp = os.path.join(out, 'simulation_assets.gif')
    dual_frames[0].save(dp, save_all=True, append_images=dual_frames[1:], duration=ms, loop=0, optimize=True)
    print(f"  Dual-view GIF: {os.path.getsize(dp)//1024}KB")


if __name__ == '__main__':
    design_living_room()
