#!/usr/bin/env python3
"""
Render layouts with REAL 3D assets from Objaverse (STL meshes in MuJoCo).
Replaces box proxies with actual furniture models.
"""
import json, os, sys, math, tempfile, shutil
import numpy as np
from pathlib import Path

os.environ['MUJOCO_GL'] = 'osmesa'
import mujoco

# Paths
SCENE_GEN_ROOT = Path(__file__).parent.parent.parent  # scene-gen/
STL_DIR = SCENE_GEN_ROOT / "data" / "assets" / "_mujoco_stl"
CATALOG_PATH = SCENE_GEN_ROOT / "data" / "asset_catalog.json"

# Category mapping: prompt category -> STL filename prefix
CATEGORY_TO_STL = {
    'sofa': 'sofa_0', 'couch': 'sofa_0',
    'armchair': 'armchair_0',
    'bed': 'bed_0',
    'coffee_table': 'coffee_table_0',
    'dining_table': 'dining_table_0', 'table': 'dining_table_0',
    'desk': 'desk_0',
    'chair': 'chair_0',
    'cabinet': 'cabinet_0', 'bookshelf': 'cabinet_0',
    'wardrobe': 'wardrobe_0',
    'television': 'television_set_0', 'tv': 'television_set_0', 'tv_stand': 'television_set_0',
    'lamp': 'lamp_0', 'floor_lamp': 'lamp_0',
    'table_lamp': 'table_lamp_0',
    'nightstand': 'dresser_0',
    'dresser': 'dresser_0',
    'rug': 'runner_carpet_0', 'carpet': 'runner_carpet_0',
    'stool': 'armchair_0',  # fallback
}

# MuJoCo material definitions for realism
MATERIALS_XML = """
<asset>
    {mesh_assets}
    <texture name="wood_tex" type="2d" builtin="checker" rgb1="0.55 0.35 0.18" rgb2="0.65 0.42 0.22" width="64" height="64"/>
    <texture name="fabric_tex" type="2d" builtin="gradient" rgb1="0.35 0.45 0.65" rgb2="0.25 0.35 0.55" width="64" height="64"/>
    <texture name="metal_tex" type="2d" builtin="flat" rgb1="0.7 0.7 0.72" rgb2="0.7 0.7 0.72" width="32" height="32"/>
    <texture name="carpet_tex" type="2d" builtin="checker" rgb1="0.6 0.55 0.45" rgb2="0.55 0.50 0.40" width="128" height="128"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.92 0.90 0.85" rgb2="0.88 0.86 0.82" width="128" height="128"/>
    
    <material name="wood" texture="wood_tex" shininess="0.3" specular="0.2"/>
    <material name="fabric" texture="fabric_tex" shininess="0.1" specular="0.05"/>
    <material name="metal" texture="metal_tex" shininess="0.8" specular="0.6"/>
    <material name="carpet" texture="carpet_tex" shininess="0.05" specular="0.02"/>
    <material name="floor" texture="floor_tex" shininess="0.2" specular="0.1"/>
</asset>
"""

# Category -> material mapping
CAT_MATERIAL = {
    'sofa': 'fabric', 'armchair': 'fabric', 'bed': 'fabric',
    'coffee_table': 'wood', 'dining_table': 'wood', 'table': 'wood', 'desk': 'wood',
    'chair': 'wood', 'cabinet': 'wood', 'wardrobe': 'wood', 'bookshelf': 'wood',
    'nightstand': 'wood', 'dresser': 'wood', 'stool': 'wood',
    'television': 'metal', 'tv': 'metal', 'tv_stand': 'metal',
    'lamp': 'metal', 'floor_lamp': 'metal', 'table_lamp': 'metal',
    'rug': 'carpet', 'carpet': 'carpet',
}


def find_stl(category):
    """Find the best STL file for a given category."""
    cat_lower = category.lower().replace(' ', '_')
    
    # Direct match
    stl_prefix = CATEGORY_TO_STL.get(cat_lower)
    if stl_prefix:
        stl_path = STL_DIR / f"{stl_prefix}.stl"
        if stl_path.exists():
            return stl_path
    
    # Fuzzy match: look for any STL containing the category name
    for stl_file in STL_DIR.glob("*.stl"):
        if cat_lower in stl_file.stem.lower():
            return stl_file
    
    # No real mesh found
    return None


def build_mujoco_xml(layout, room_size, tmp_dir):
    """Build MuJoCo XML with real mesh assets."""
    room_w, room_h = room_size
    objects = layout.get('objects', [])
    
    mesh_assets = []
    body_xml = []
    used_stls = set()
    
    for i, obj in enumerate(objects):
        cat = obj['category'].lower().replace(' ', '_')
        pos = obj['position']
        dims = obj.get('dimensions', {'width': 0.5, 'depth': 0.5, 'height': 0.5})
        rotation = obj.get('rotation', 0)
        
        stl_path = find_stl(cat)
        mat = CAT_MATERIAL.get(cat, 'wood')
        
        rot_rad = math.radians(rotation)
        qw = math.cos(rot_rad / 2)
        qz = math.sin(rot_rad / 2)
        
        if stl_path and stl_path.exists():
            # Use real mesh
            mesh_name = f"mesh_{i}_{stl_path.stem}"
            
            # Copy STL to tmp dir if not already there
            dst = Path(tmp_dir) / stl_path.name
            if not dst.exists():
                shutil.copy2(stl_path, dst)
            
            # STL files are already normalized to standard furniture dimensions
            # with bottom at Z=0. We just need to scale to the LLM-requested size.
            # First get the STL's own dimensions by reading the file
            import trimesh
            stl_mesh = trimesh.load(str(stl_path))
            stl_ext = stl_mesh.bounds[1] - stl_mesh.bounds[0]
            
            # Compute per-axis scale: target / actual STL size
            sx = dims['width'] / max(stl_ext[0], 0.01)
            sy = dims['depth'] / max(stl_ext[1], 0.01)
            sz = dims['height'] / max(stl_ext[2], 0.01)
            
            mesh_assets.append(
                f'<mesh name="{mesh_name}" file="{stl_path.name}" '
                f'scale="{sx:.4f} {sy:.4f} {sz:.4f}"/>'
            )
            
            # Body at ground level (Z=0), mesh bottom is already at Z=0
            body_xml.append(f'''
    <body name="obj_{i}_{obj['category']}" pos="{pos[0]:.3f} {pos[1]:.3f} 0.0" quat="{qw:.4f} 0 0 {qz:.4f}">
        <geom name="geom_{i}" type="mesh" mesh="{mesh_name}" material="{mat}"/>
    </body>''')
        else:
            # Fallback: colored box with material
            h = dims['height']
            # Use category-specific RGBA for better visual distinction
            rgba_map = {
                'sofa': '0.35 0.45 0.65 1', 'bed': '0.55 0.36 0.96 1',
                'television': '0.15 0.15 0.15 1', 'tv_stand': '0.2 0.2 0.2 1',
                'coffee_table': '0.55 0.35 0.18 1', 'dining_table': '0.55 0.35 0.18 1',
                'chair': '0.55 0.35 0.18 1', 'desk': '0.4 0.28 0.15 1',
                'cabinet': '0.45 0.30 0.15 1', 'wardrobe': '0.5 0.32 0.18 1',
                'lamp': '0.9 0.85 0.3 1', 'rug': '0.6 0.55 0.45 0.5',
                'armchair': '0.5 0.3 0.3 1', 'nightstand': '0.5 0.35 0.2 1',
                'dresser': '0.55 0.38 0.22 1',
            }
            rgba = rgba_map.get(cat, '0.6 0.6 0.6 1')
            
            body_xml.append(f'''
    <body name="obj_{i}_{obj['category']}" pos="{pos[0]:.3f} {pos[1]:.3f} {h/2:.3f}" quat="{qw:.4f} 0 0 {qz:.4f}">
        <geom name="geom_{i}" type="box" size="{dims['width']/2:.3f} {dims['depth']/2:.3f} {h/2:.3f}" material="{mat}"/>
    </body>''')
    
    mesh_assets_str = "\n    ".join(mesh_assets)
    bodies_str = "\n".join(body_xml)
    
    xml = f"""<mujoco model="scene">
    <compiler angle="degree" meshdir="{tmp_dir}"/>
    
    <visual>
        <global offwidth="1600" offheight="1200"/>
        <headlight diffuse="0.8 0.8 0.8" ambient="0.4 0.4 0.4" specular="0.3 0.3 0.3"/>
    </visual>
    
    {MATERIALS_XML.replace('{mesh_assets}', mesh_assets_str)}
    
    <worldbody>
        <light pos="{room_w/2} {room_h/2} 4" dir="0 0 -1" diffuse="0.9 0.9 0.85" castshadow="true"/>
        <light pos="{room_w*0.8} {room_h*0.2} 3" dir="-0.3 0.3 -1" diffuse="0.4 0.4 0.35"/>
        
        <!-- Floor -->
        <geom name="floor" type="plane" size="{room_w/2+1} {room_h/2+1} 0.01" 
              pos="{room_w/2} {room_h/2} 0" material="floor"/>
        
        <!-- Walls (low, 0.15m) -->
        <geom name="wall_s" type="box" size="{room_w/2} 0.03 0.075" pos="{room_w/2} 0.03 0.075" rgba="0.9 0.88 0.85 1"/>
        <geom name="wall_n" type="box" size="{room_w/2} 0.03 0.075" pos="{room_w/2} {room_h-0.03} 0.075" rgba="0.9 0.88 0.85 1"/>
        <geom name="wall_w" type="box" size="0.03 {room_h/2} 0.075" pos="0.03 {room_h/2} 0.075" rgba="0.9 0.88 0.85 1"/>
        <geom name="wall_e" type="box" size="0.03 {room_h/2} 0.075" pos="{room_w-0.03} {room_h/2} 0.075" rgba="0.9 0.88 0.85 1"/>
        
        <!-- Furniture -->
        {bodies_str}
    </worldbody>
</mujoco>"""
    
    return xml


def render_scene(layout, room_size, output_dir, prefix="real"):
    """Render top-down and perspective views with real assets."""
    room_w, room_h = room_size
    output_dir = Path(output_dir)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        xml = build_mujoco_xml(layout, room_size, tmp_dir)
        
        xml_path = os.path.join(tmp_dir, "scene.xml")
        with open(xml_path, 'w') as f:
            f.write(xml)
        
        try:
            model = mujoco.MjModel.from_xml_path(xml_path)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            
            renderer = mujoco.Renderer(model, width=1600, height=1200)
            
            # === Top-down view ===
            camera = mujoco.MjvCamera()
            camera.lookat[0] = room_w / 2
            camera.lookat[1] = room_h / 2
            camera.lookat[2] = 0.3
            camera.distance = max(room_w, room_h) * 1.0
            camera.elevation = -85  # Nearly top-down
            camera.azimuth = 90
            
            renderer.update_scene(data, camera)
            img = renderer.render()
            
            from PIL import Image
            pil_img = Image.fromarray(img)
            top_path = output_dir / f"{prefix}_topdown.png"
            pil_img.save(str(top_path))
            print(f"  ✅ {top_path.name}", flush=True)
            
            # === Perspective view (3/4 angle) ===
            camera2 = mujoco.MjvCamera()
            camera2.lookat[0] = room_w / 2
            camera2.lookat[1] = room_h / 2
            camera2.lookat[2] = 0.5
            camera2.distance = max(room_w, room_h) * 1.3
            camera2.elevation = -35
            camera2.azimuth = 135
            
            renderer.update_scene(data, camera2)
            img2 = renderer.render()
            
            pil_img2 = Image.fromarray(img2)
            persp_path = output_dir / f"{prefix}_perspective.png"
            pil_img2.save(str(persp_path))
            print(f"  ✅ {persp_path.name}", flush=True)
            
            renderer.close()
            return True
            
        except Exception as e:
            print(f"  ❌ Render error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False


def render_all_results(results_dir):
    """Render all layouts with real assets."""
    results_dir = Path(results_dir)
    prompts_file = results_dir.parent / "prompts" / "layout_prompts.json"
    
    with open(prompts_file) as f:
        prompts = {p['id']: p for p in json.load(f)}
    
    rendered = 0
    
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == '__pycache__':
            continue
        model_name = model_dir.name
        
        for prompt_dir in sorted(model_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue
            pid = prompt_dir.name
            
            layout_file = prompt_dir / "layout.json"
            if not layout_file.exists():
                continue
            
            # Skip if already rendered with real assets
            if (prompt_dir / "real_topdown.png").exists():
                continue
            
            with open(layout_file) as f:
                data = json.load(f)
            
            layout = data.get('layout', data)
            prompt_data = prompts.get(pid, {})
            room_size = prompt_data.get('room_size', [5, 4])
            
            print(f"Rendering {model_name}/{pid} with real assets...", flush=True)
            
            if render_scene(layout, room_size, prompt_dir, prefix="real"):
                rendered += 1
    
    print(f"\nRendered {rendered} scenes with real assets.", flush=True)


if __name__ == "__main__":
    # Check STL assets
    if not STL_DIR.exists():
        print(f"ERROR: STL directory not found: {STL_DIR}")
        sys.exit(1)
    
    stl_files = list(STL_DIR.glob("*.stl"))
    print(f"Found {len(stl_files)} STL mesh assets:", flush=True)
    for f in sorted(stl_files):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.0f}KB)", flush=True)
    
    print(f"\nRendering all benchmark layouts...", flush=True)
    results_dir = Path(__file__).parent / "../results"
    render_all_results(results_dir)
