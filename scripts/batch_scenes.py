"""
Batch Scene Generator: Generate 10 diverse indoor scenes with domain randomization.
Each scene has different room type, dimensions, furniture, and layout.
Outputs: MuJoCo XML, scene graph JSON, floorplan PNG, simulation GIF.
"""

import os, sys, json, random, numpy as np

os.environ['MUJOCO_GL'] = 'osmesa'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.assets.asset_manager import AssetManager, STANDARD_DIMENSIONS
from src.layout.scene_graph_builder import build_scene_graph


# ============================================================
# 10 SCENE DEFINITIONS - Hand-designed for realism
# ============================================================

SCENES = [
    # ---- Scene 0: Modern Living Room ----
    {
        'name': 'modern_living_room',
        'room': (5.0, 4.0),
        'desc': 'Modern living room with L-shaped seating',
        'objects': [
            ('sofa', [2.5, 3.50], 180, 2),
            ('coffee_table', [2.5, 2.30], 0, 2),
            ('armchair', [0.8, 2.70], 270, 2),
            ('television_set', [2.5, 0.15], 0, 0),
            ('table_lamp', [4.3, 3.50], 0, 0),
            ('cabinet', [0.6, 0.35], 0, 0),
            ('runner_(carpet)', [2.5, 2.20], 0, 0),
        ],
    },
    # ---- Scene 1: Kitchen / Dining ----
    {
        'name': 'kitchen_dining',
        'room': (4.0, 3.5),
        'desc': 'Kitchen dining area with table and chairs',
        'objects': [
            ('dining_table', [2.0, 2.0], 0, 0),
            ('chair', [1.3, 2.0], 90, 0),
            ('chair', [2.7, 2.0], 270, 0),
            ('chair', [2.0, 1.3], 0, 0),
            ('chair', [2.0, 2.7], 180, 0),
            ('cabinet', [3.6, 0.3], 0, 0),
            ('table_lamp', [0.3, 3.2], 0, 0),
        ],
    },
    # ---- Scene 2: Classroom ----
    {
        'name': 'classroom',
        'room': (7.0, 5.0),
        'desc': 'Classroom with rows of desks facing front',
        'objects': [
            ('desk', [1.5, 3.5], 180, 0),
            ('desk', [3.5, 3.5], 180, 0),
            ('desk', [5.5, 3.5], 180, 0),
            ('desk', [1.5, 2.0], 180, 0),
            ('desk', [3.5, 2.0], 180, 0),
            ('desk', [5.5, 2.0], 180, 0),
            ('chair', [1.5, 3.0], 180, 0),
            ('chair', [3.5, 3.0], 180, 0),
            ('chair', [5.5, 3.0], 180, 0),
            ('chair', [1.5, 1.5], 180, 0),
            ('chair', [3.5, 1.5], 180, 0),
            ('chair', [5.5, 1.5], 180, 0),
            ('desk', [3.5, 0.5], 0, 0),   # Teacher's desk
            ('cabinet', [6.5, 0.3], 0, 0),
        ],
    },
    # ---- Scene 3: Bedroom ----
    {
        'name': 'bedroom',
        'room': (4.0, 4.0),
        'desc': 'Cozy bedroom with bed, dresser, and lamp',
        'objects': [
            ('bed', [2.0, 3.0], 180, 0),
            ('dresser', [0.4, 1.0], 90, 0),
            ('table_lamp', [3.5, 3.5], 0, 0),
            ('wardrobe', [3.5, 0.4], 0, 0),
            ('runner_(carpet)', [2.0, 2.2], 0, 0),
            ('cabinet', [0.4, 3.5], 90, 0),
        ],
    },
    # ---- Scene 4: Home Office ----
    {
        'name': 'home_office',
        'room': (3.5, 3.0),
        'desc': 'Home office with desk, chair, and bookshelf',
        'objects': [
            ('desk', [1.75, 2.5], 180, 0),
            ('chair', [1.75, 1.8], 0, 0),
            ('cabinet', [3.2, 0.3], 0, 0),   # bookshelf substitute
            ('table_lamp', [0.5, 2.5], 0, 0),
            ('runner_(carpet)', [1.75, 1.5], 0, 0),
        ],
    },
    # ---- Scene 5: Conference Room ----
    {
        'name': 'conference_room',
        'room': (6.0, 4.0),
        'desc': 'Conference room with long table and chairs',
        'objects': [
            ('dining_table', [3.0, 2.0], 0, 0),  # long conference table
            ('chair', [1.5, 2.0], 90, 0),
            ('chair', [2.3, 2.0], 90, 0),
            ('chair', [3.7, 2.0], 270, 0),
            ('chair', [4.5, 2.0], 270, 0),
            ('chair', [3.0, 1.2], 0, 0),
            ('chair', [3.0, 2.8], 180, 0),
            ('television_set', [5.5, 2.0], 270, 0),
            ('cabinet', [0.5, 0.3], 0, 0),
        ],
    },
    # ---- Scene 6: Studio Apartment ----
    {
        'name': 'studio_apartment',
        'room': (5.0, 5.0),
        'desc': 'Open studio with sleeping, living, and work zones',
        'objects': [
            ('bed', [1.0, 4.0], 180, 0),
            ('sofa', [3.5, 4.3], 180, 0),
            ('coffee_table', [3.5, 3.2], 0, 0),
            ('desk', [4.3, 1.0], 0, 0),
            ('chair', [4.3, 1.7], 180, 0),
            ('television_set', [3.5, 1.8], 0, 0),
            ('dresser', [0.4, 2.0], 90, 0),
            ('table_lamp', [0.3, 4.5], 0, 0),
            ('runner_(carpet)', [2.5, 3.5], 0, 0),
        ],
    },
    # ---- Scene 7: Reading Room / Library ----
    {
        'name': 'reading_room',
        'room': (4.5, 4.0),
        'desc': 'Quiet reading room with armchairs and bookshelves',
        'objects': [
            ('armchair', [1.2, 3.0], 180, 2),
            ('armchair', [3.3, 3.0], 180, 2),
            ('coffee_table', [2.25, 2.2], 0, 0),
            ('cabinet', [0.4, 0.3], 0, 0),  # bookshelf
            ('cabinet', [4.1, 0.3], 0, 0),  # bookshelf
            ('table_lamp', [1.2, 3.7], 0, 0),
            ('table_lamp', [3.3, 3.7], 0, 0),
            ('runner_(carpet)', [2.25, 2.5], 0, 0),
        ],
    },
    # ---- Scene 8: Lounge Area ----
    {
        'name': 'lounge',
        'room': (6.0, 5.0),
        'desc': 'Spacious lounge with multiple seating groups',
        'objects': [
            ('sofa', [2.0, 4.5], 180, 0),
            ('sofa', [5.0, 4.5], 180, 0),
            ('coffee_table', [2.0, 3.3], 0, 0),
            ('coffee_table', [5.0, 3.3], 0, 0),
            ('armchair', [0.7, 3.3], 270, 2),
            ('armchair', [3.5, 3.3], 90, 0),
            ('television_set', [3.5, 0.15], 0, 0),
            ('lamp', [0.4, 4.7], 0, 0),
            ('runner_(carpet)', [3.0, 3.0], 0, 0),
            ('cabinet', [5.5, 0.3], 0, 0),
        ],
    },
    # ---- Scene 9: Kids Room ----
    {
        'name': 'kids_room',
        'room': (4.0, 3.5),
        'desc': 'Colorful kids room with bed, desk, and storage',
        'objects': [
            ('bed', [1.0, 2.8], 180, 0),
            ('desk', [3.2, 2.8], 180, 0),
            ('chair', [3.2, 2.1], 0, 0),
            ('cabinet', [3.5, 0.3], 0, 0),
            ('table_lamp', [3.2, 3.3], 0, 0),
            ('runner_(carpet)', [2.0, 1.5], 0, 0),
        ],
    },
]

# Material definitions
MATERIALS = {
    'sofa': ('gradient', '0.22 0.30 0.50', '0.18 0.25 0.42', '3 3', '0.1', '0.02'),
    'coffee_table': ('checker', '0.50 0.32 0.18', '0.45 0.28 0.14', '6 6', '0.4', '0.3'),
    'armchair': ('gradient', '0.58 0.50 0.38', '0.52 0.44 0.32', '3 3', '0.1', '0.02'),
    'television_set': ('flat', '0.08 0.08 0.10', '0.08 0.08 0.10', '1 1', '0.8', '0.9'),
    'table_lamp': ('gradient', '0.90 0.82 0.60', '0.80 0.72 0.50', '2 2', '0.3', '0.2'),
    'cabinet': ('checker', '0.40 0.25 0.12', '0.35 0.20 0.08', '8 8', '0.3', '0.2'),
    'runner_(carpet)': ('checker', '0.78 0.70 0.58', '0.72 0.64 0.52', '4 4', '0.0', '0.0'),
    'dining_table': ('checker', '0.55 0.38 0.22', '0.50 0.33 0.18', '6 6', '0.4', '0.3'),
    'desk': ('checker', '0.48 0.35 0.22', '0.43 0.30 0.18', '6 6', '0.3', '0.2'),
    'chair': ('gradient', '0.50 0.42 0.30', '0.45 0.38 0.26', '3 3', '0.2', '0.1'),
    'bed': ('gradient', '0.85 0.82 0.78', '0.80 0.77 0.73', '2 2', '0.05', '0.02'),
    'dresser': ('checker', '0.42 0.28 0.15', '0.38 0.24 0.12', '6 6', '0.3', '0.2'),
    'wardrobe': ('checker', '0.38 0.24 0.12', '0.34 0.20 0.08', '8 8', '0.2', '0.15'),
    'lamp': ('gradient', '0.88 0.80 0.58', '0.78 0.70 0.48', '2 2', '0.3', '0.2'),
    'stool': ('gradient', '0.45 0.38 0.28', '0.40 0.33 0.24', '3 3', '0.2', '0.1'),
    'bench': ('checker', '0.50 0.35 0.20', '0.45 0.30 0.16', '4 4', '0.3', '0.2'),
}

MASS = {
    'sofa': 45, 'coffee_table': 20, 'armchair': 30, 'television_set': 10,
    'table_lamp': 4, 'cabinet': 35, 'runner_(carpet)': 3, 'dining_table': 35,
    'desk': 25, 'chair': 8, 'bed': 50, 'dresser': 40, 'wardrobe': 50,
    'lamp': 5, 'stool': 6, 'bench': 15,
}


def generate_scene(scene_def, mgr, base_output):
    """Generate a single scene: XML, JSON, floorplan, GIF."""
    name = scene_def['name']
    rw, rd = scene_def['room']
    desc = scene_def['desc']
    
    out_dir = os.path.join(base_output, name)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"  {name}: {desc} ({rw}×{rd}m)")
    print(f"{'='*50}")
    
    # Build objects
    scene_objects = []
    for cat, pos, rot, idx in scene_def['objects']:
        resolved = mgr.resolve_category(cat)
        dims = mgr.get_standard_dimensions(cat)
        scene_objects.append({
            'category': cat, 'position': [pos[0], pos[1], 0.0],
            'rotation': rot, 'index': idx, 'dimensions': dims,
        })
    
    # Scene graph
    sg_path = os.path.join(out_dir, 'scene_graph.json')
    build_scene_graph(scene_objects, rw, rd, output_path=sg_path, verbose=False)
    print(f"  Scene graph: {sg_path}")
    
    # Build MuJoCo XML
    tex_parts, mat_parts, mesh_parts, body_parts = [], [], [], []
    
    for i, obj in enumerate(scene_objects):
        cat = obj['category']
        safe = cat.replace('(','').replace(')','').replace(' ','_')
        bname = f"{safe}_{i}"
        
        export = mgr.export_for_mujoco(cat, obj['index'])
        
        # Material
        m = MATERIALS.get(cat, MATERIALS.get(mgr.resolve_category(cat), MATERIALS['chair']))
        builtin, rgb1, rgb2, rep, spec, shin = m
        tex_parts.append(f'    <texture type="2d" name="tex_{bname}" builtin="{builtin}" rgb1="{rgb1}" rgb2="{rgb2}" width="128" height="128"/>')
        mat_parts.append(f'    <material name="mat_{bname}" texture="tex_{bname}" texrepeat="{rep}" specular="{spec}" shininess="{shin}"/>')
        
        mass = MASS.get(cat, 15)
        rot_rad = np.radians(obj['rotation'])
        qw, qz = np.cos(rot_rad/2), np.sin(rot_rad/2)
        
        if export:
            stl = os.path.abspath(export['stl_path'])
            mesh_parts.append(f'    <mesh name="{bname}_mesh" file="{stl}"/>')
            body_parts.append(f'''    <body name="{bname}" pos="{obj['position'][0]:.3f} {obj['position'][1]:.3f} 0.000" quat="{qw:.4f} 0 0 {qz:.4f}">
      <freejoint/>
      <geom type="mesh" mesh="{bname}_mesh" material="mat_{bname}" mass="{mass}"/>
    </body>''')
        else:
            dims = obj['dimensions']
            body_parts.append(f'''    <body name="{bname}" pos="{obj['position'][0]:.3f} {obj['position'][1]:.3f} {dims['height']/2:.3f}" quat="{qw:.4f} 0 0 {qz:.4f}">
      <freejoint/>
      <geom type="box" size="{dims['width']/2:.3f} {dims['depth']/2:.3f} {dims['height']/2:.3f}" material="mat_{bname}" mass="{mass}"/>
    </body>''')
    
    xml = f'''<?xml version="1.0" ?>
<mujoco model="scenecraft_{name}">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <visual>
    <quality shadowsize="4096"/>
    <global offwidth="1600" offheight="1200"/>
  </visual>
  <asset>
    <texture type="2d" name="floor_tex" builtin="checker" rgb1="0.75 0.60 0.42" rgb2="0.70 0.55 0.38" width="256" height="256"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="10 8" specular="0.3" shininess="0.2"/>
{chr(10).join(tex_parts)}
{chr(10).join(mat_parts)}
{chr(10).join(mesh_parts)}
  </asset>
  <worldbody>
    <geom type="plane" size="{rw/2} {rd/2} 0.01" material="floor_mat" pos="{rw/2} {rd/2} 0"/>
    <geom type="box" size="{rw/2} 0.03 0.10" pos="{rw/2} -0.03 0.10" rgba="0.90 0.88 0.85 1"/>
    <geom type="box" size="{rw/2} 0.03 0.10" pos="{rw/2} {rd+0.03} 0.10" rgba="0.90 0.88 0.85 1"/>
    <geom type="box" size="0.03 {rd/2} 0.10" pos="-0.03 {rd/2} 0.10" rgba="0.90 0.88 0.85 1"/>
    <geom type="box" size="0.03 {rd/2} 0.10" pos="{rw+0.03} {rd/2} 0.10" rgba="0.90 0.88 0.85 1"/>
    <light pos="{rw/2} {rd/2} 2.8" dir="0 0 -1" diffuse="0.85 0.80 0.72" castshadow="true"/>
    <light pos="1 1 2.5" dir="0.3 0.3 -1" diffuse="0.2 0.2 0.2"/>
{chr(10).join(body_parts)}
  </worldbody>
</mujoco>'''
    
    xml_path = os.path.join(out_dir, 'scene.xml')
    with open(xml_path, 'w') as f: f.write(xml)
    
    # Save JSON
    def ser(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    with open(os.path.join(out_dir, 'scene_data.json'), 'w') as f:
        json.dump({'name': name, 'desc': desc, 'room': {'width': rw, 'depth': rd},
                   'objects': scene_objects}, f, indent=2, default=ser)
    
    # Floorplan
    render_floorplan(scene_objects, rw, rd, out_dir, name, desc)
    
    # Simulate + GIF
    result = simulate_and_gif(xml_path, out_dir, rw, rd, name)
    
    return result


def render_floorplan(objects, rw, rd, out, name, desc):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    colors = {
        'sofa': '#3A5BA0', 'coffee_table': '#8B5E3C', 'armchair': '#9B8B7A',
        'television_set': '#2C2C2E', 'table_lamp': '#D4A843', 'cabinet': '#6B4226',
        'runner_(carpet)': '#DDD0C0', 'dining_table': '#A0522D', 'desk': '#CD853F',
        'chair': '#7B68EE', 'bed': '#DC143C', 'dresser': '#8B7355',
        'wardrobe': '#696969', 'lamp': '#FFA500', 'stool': '#9370DB', 'bench': '#D2691E',
    }
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.add_patch(patches.Rectangle((0,0), rw, rd, lw=3, ec='#333', fc='#FFF8F0'))
    
    for obj in objects:
        cat, pos, dims = obj['category'], obj['position'], obj['dimensions']
        rot = obj.get('rotation', 0)
        w, d = dims['width'], dims['depth']
        if rot in [90, 270]: w, d = d, w
        
        alpha = 0.3 if 'carpet' in cat else 0.8
        rect = patches.Rectangle((pos[0]-w/2, pos[1]-d/2), w, d, lw=1,
                                  ec='#333', fc=colors.get(cat, '#888'), alpha=alpha)
        ax.add_patch(rect)
        
        label = cat.replace('_',' ').replace('(carpet)','').replace('television set','TV')[:8]
        fc = 'white' if alpha > 0.5 else '#333'
        ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=5, fontweight='bold', color=fc)
    
    ax.set_xlim(-0.3, rw+0.3); ax.set_ylim(-0.3, rd+0.3); ax.set_aspect('equal')
    ax.set_title(f'{name.replace("_"," ").title()} ({rw}×{rd}m)\n{desc}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2, ls='--')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'floorplan.png'), dpi=120, bbox_inches='tight')
    plt.close()


def simulate_and_gif(xml_path, out, rw, rd, name):
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    W, H = 640, 480
    renderer = mujoco.Renderer(model, height=H, width=W)
    
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [rw/2, rd/2, 0.3]
    cam.distance = max(rw, rd) * 1.5
    cam.elevation = -90; cam.azimuth = 0
    
    cam_p = mujoco.MjvCamera()
    cam_p.lookat[:] = [rw/2, rd/2, 0.4]
    cam_p.distance = max(rw, rd) * 1.5
    cam_p.elevation = -45
    
    try: font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf', 14)
    except: font = ImageFont.load_default()
    
    fps, duration = 12, 3.0
    dt = model.opt.timestep
    spf = max(1, int(1.0/(fps*dt)))
    nf = int(duration*fps)
    
    frames = []
    for fi in range(nf):
        for _ in range(spf): mujoco.mj_step(model, data)
        t = fi / fps
        cam_p.azimuth = 120 + fi * (200.0/nf)
        
        renderer.update_scene(data, cam)
        top = Image.fromarray(renderer.render())
        renderer.update_scene(data, cam_p)
        persp = Image.fromarray(renderer.render())
        
        comb = Image.new('RGB', (W*2+6, H+25), (255,255,255))
        d = ImageDraw.Draw(comb)
        d.text((5,3), f'{name} t={t:.1f}s | Top-Down', fill=(0,0,0), font=font)
        d.text((W+10,3), 'Perspective', fill=(0,0,0), font=font)
        comb.paste(top, (0,23)); comb.paste(persp, (W+6,23))
        frames.append(comb)
    
    gif_path = os.path.join(out, 'simulation.gif')
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                    duration=int(1000/fps), loop=0, optimize=True)
    
    # Also save top-down start/end
    mujoco.mj_resetData(model, data)
    renderer.update_scene(data, cam)
    Image.fromarray(renderer.render()).save(os.path.join(out, 'top_start.png'))
    
    for _ in range(int(3.0/dt)): mujoco.mj_step(model, data)
    renderer.update_scene(data, cam)
    Image.fromarray(renderer.render()).save(os.path.join(out, 'top_end.png'))
    
    # Stability
    stable = 0; total = 0
    for i in range(model.nbody):
        if model.body_parentid[i] == 0 and i > 0:
            total += 1
            w, x, y, z = data.xquat[i]
            up = 1-2*(x*x+y*y)
            if up > 0.7 and data.xpos[i][2] > -0.5: stable += 1
    
    pct = 100*stable/max(total,1)
    gif_kb = os.path.getsize(gif_path)//1024
    print(f"  ✓ {name}: {stable}/{total} stable ({pct:.0f}%) | GIF {gif_kb}KB")
    
    return {'name': name, 'stable': stable, 'total': total, 'pct': pct}


def main():
    print("=" * 60)
    print("SceneCraft-LLM: Batch Scene Generation (10 scenes)")
    print("=" * 60)
    
    mgr = AssetManager()
    base_output = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'batch_scenes')
    os.makedirs(base_output, exist_ok=True)
    
    results = []
    for scene_def in SCENES:
        try:
            r = generate_scene(scene_def, mgr, base_output)
            results.append(r)
        except Exception as e:
            print(f"  ✗ {scene_def['name']}: FAILED - {e}")
            results.append({'name': scene_def['name'], 'stable': 0, 'total': 0, 'pct': 0, 'error': str(e)})
    
    # Summary
    print("\n" + "=" * 60)
    print("BATCH GENERATION SUMMARY")
    print("=" * 60)
    
    total_objects = sum(r['total'] for r in results)
    total_stable = sum(r['stable'] for r in results)
    
    for r in results:
        status = '✓' if r.get('pct', 0) > 80 else '✗'
        err = f" ERROR: {r['error']}" if 'error' in r else ""
        print(f"  {status} {r['name']}: {r.get('stable',0)}/{r.get('total',0)} stable ({r.get('pct',0):.0f}%){err}")
    
    print(f"\n  Total: {total_stable}/{total_objects} objects stable "
          f"({100*total_stable/max(total_objects,1):.0f}%)")
    print(f"  Scenes: {len([r for r in results if r.get('pct',0) > 80])}/{len(results)} successful")
    
    # Save summary
    with open(os.path.join(base_output, 'summary.json'), 'w') as f:
        json.dump({'scenes': results, 'total_objects': total_objects,
                   'total_stable': total_stable}, f, indent=2)
    
    print(f"\n  Output: {base_output}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
