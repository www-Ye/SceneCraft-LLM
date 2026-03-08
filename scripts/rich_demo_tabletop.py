#!/usr/bin/env python3
"""
Rich Tabletop Demo Scene Generator

Creates a diverse scene with 12 objects using corrected per-axis mesh scaling.
Tests the fixed normalization and generates a realistic tabletop environment.
"""
import json, os, sys, math, tempfile, shutil, random
import numpy as np
from pathlib import Path

os.environ['MUJOCO_GL'] = 'osmesa'
import mujoco
import trimesh
from PIL import Image

# Project paths
SCRIPT_DIR = Path(__file__).parent
SCENE_GEN_ROOT = SCRIPT_DIR.parent
STL_DIR = SCENE_GEN_ROOT / "data" / "assets" / "_mujoco_stl"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "rich_demo"

def scan_available_objects():
    """Scan available STL files and their dimensions."""
    objects = {}
    
    # Try all categories with their expected files
    categories = [
        "mug", "cup", "bowl", "plate", "dish", "bottle", "can", "apple", "banana",
        "book", "pen", "pencil", "box", "remote_control", "scissors"
    ]
    
    for cat in categories:
        for idx in [0, 1]:
            stl_path = STL_DIR / f"{cat}_{idx}_visual.stl"
            if stl_path.exists():
                try:
                    mesh = trimesh.load(str(stl_path))
                    if len(mesh.vertices) > 10:  # Valid mesh
                        ext = mesh.bounds[1] - mesh.bounds[0]
                        objects[f"{cat}_{idx}"] = {
                            "stl_path": str(stl_path),
                            "category": cat,
                            "index": idx,
                            "dimensions": ext.tolist(),
                            "vertices": len(mesh.vertices),
                            "z_range": [mesh.bounds[0][2], mesh.bounds[1][2]]
                        }
                except Exception as e:
                    print(f"Error loading {stl_path}: {e}")
    
    return objects

def get_material_for_category(cat):
    """Get realistic material properties for each category."""
    materials = {
        # Ceramics - white/cream, glossy
        "mug": {"rgba": "0.92 0.90 0.88 1", "shininess": "0.6", "specular": "0.4"},
        "cup": {"rgba": "0.95 0.93 0.90 1", "shininess": "0.6", "specular": "0.4"},
        "bowl": {"rgba": "0.93 0.91 0.87 1", "shininess": "0.5", "specular": "0.3"},
        "plate": {"rgba": "0.97 0.96 0.94 1", "shininess": "0.6", "specular": "0.4"},
        "dish": {"rgba": "0.96 0.95 0.92 1", "shininess": "0.6", "specular": "0.4"},
        
        # Glass/bottles
        "bottle": {"rgba": "0.25 0.55 0.25 0.75", "shininess": "0.8", "specular": "0.6"},
        
        # Metal
        "can": {"rgba": "0.75 0.18 0.18 1", "shininess": "0.7", "specular": "0.5"},
        "scissors": {"rgba": "0.72 0.72 0.75 1", "shininess": "0.8", "specular": "0.6"},
        
        # Food
        "apple": {"rgba": "0.82 0.12 0.10 1", "shininess": "0.4", "specular": "0.2"},
        "banana": {"rgba": "0.95 0.88 0.22 1", "shininess": "0.3", "specular": "0.1"},
        
        # Paper/plastic
        "book": {"rgba": "0.22 0.32 0.62 1", "shininess": "0.1", "specular": "0.05"},
        "box": {"rgba": "0.60 0.42 0.22 1", "shininess": "0.1", "specular": "0.05"},
        "pen": {"rgba": "0.12 0.12 0.14 1", "shininess": "0.5", "specular": "0.3"},
        "pencil": {"rgba": "0.85 0.75 0.15 1", "shininess": "0.2", "specular": "0.1"},
        "remote_control": {"rgba": "0.15 0.15 0.18 1", "shininess": "0.4", "specular": "0.2"},
    }
    return materials.get(cat, {"rgba": "0.6 0.6 0.6 1", "shininess": "0.3", "specular": "0.2"})

def create_rich_layout():
    """Create a rich tabletop layout with 12 diverse objects."""
    # Breakfast + office + kitchen theme
    layout = [
        # Main dishes
        {"category": "plate", "position": [0.0, 0.05], "rotation": 0},
        {"category": "bowl", "position": [-0.25, 0.10], "rotation": 0},
        
        # Drinks
        {"category": "mug", "position": [0.22, 0.18], "rotation": 15},
        {"category": "cup", "position": [0.25, -0.05], "rotation": 0},
        {"category": "bottle", "position": [0.32, 0.08], "rotation": 0},
        {"category": "can", "position": [-0.30, 0.20], "rotation": 0},
        
        # Food
        {"category": "apple", "position": [0.12, -0.15], "rotation": 0},
        {"category": "banana", "position": [-0.15, -0.20], "rotation": 25},
        
        # Office items
        {"category": "book", "position": [-0.05, -0.18], "rotation": 8},
        {"category": "pen", "position": [0.08, -0.25], "rotation": -15},
        {"category": "pencil", "position": [0.18, -0.22], "rotation": 5},
        
        # Tools/containers
        {"category": "box", "position": [-0.28, -0.08], "rotation": -10},
        {"category": "scissors", "position": [0.15, 0.25], "rotation": 30},
        {"category": "remote_control", "position": [-0.20, 0.25], "rotation": -5},
    ]
    
    return layout

def build_scene_xml(layout, available_objects):
    """Build MuJoCo XML for rich tabletop scene."""
    table_width, table_depth = 0.80, 0.60
    table_height = 0.75
    
    mesh_assets = []
    material_defs = []
    obj_bodies = []
    
    objects_used = []
    
    for i, item in enumerate(layout):
        cat = item["category"]
        pos = item["position"]
        rot = item["rotation"]
        
        # Find available object of this category
        candidates = [k for k, v in available_objects.items() if v["category"] == cat]
        if not candidates:
            print(f"⚠️ No {cat} available, skipping")
            continue
        
        obj_key = candidates[0]  # Use first available
        obj_info = available_objects[obj_key]
        
        mesh_name = f"mesh_{i}_{cat}"
        mat_name = f"mat_{i}_{cat}"
        
        # STL file reference
        stl_name = Path(obj_info["stl_path"]).name
        mesh_assets.append(f'<mesh name="{mesh_name}" file="{stl_name}"/>')
        
        # Material
        mat = get_material_for_category(cat)
        material_defs.append(
            f'<material name="{mat_name}" '
            f'rgba="{mat["rgba"]}" '
            f'shininess="{mat["shininess"]}" '
            f'specular="{mat["specular"]}"/>'
        )
        
        # Body position and rotation
        rot_rad = math.radians(rot)
        qw = math.cos(rot_rad / 2)
        qz = math.sin(rot_rad / 2)
        
        # Place on table surface
        z = table_height + 0.02 + 0.001  # table top + small offset to avoid penetration
        
        # Physics properties based on size
        dims = obj_info["dimensions"]
        volume = dims[0] * dims[1] * dims[2]
        density = {"ceramic": 2300, "glass": 2500, "metal": 7800, "organic": 800, 
                   "paper": 700, "plastic": 950}.get("default", 1200)
        mass = max(volume * density * 0.001, 0.01)  # Convert to reasonable mass
        
        obj_bodies.append(f"""
        <body name="obj_{i}_{cat}" pos="{pos[0]:.4f} {pos[1]:.4f} {z:.4f}" quat="{qw:.4f} 0 0 {qz:.4f}">
            <freejoint name="fj_{i}"/>
            <geom name="geom_{i}" type="mesh" mesh="{mesh_name}" 
                  material="{mat_name}" mass="{mass:.3f}"
                  friction="0.7 0.005 0.001" 
                  solimp="0.95 0.95 0.01" solref="0.02 1"/>
        </body>""")
        
        objects_used.append({
            "id": i,
            "category": cat,
            "position": pos + [z],
            "rotation": rot,
            "dimensions": dims,
            "stl_file": stl_name,
            "mass": mass
        })
    
    mesh_xml = "\\n        ".join(mesh_assets)
    mat_xml = "\\n        ".join(material_defs)
    body_xml = "\\n".join(obj_bodies)
    
    xml = f"""<mujoco model="rich_tabletop_demo">
    <compiler angle="degree" meshdir="{STL_DIR}"/>
    
    <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>
    
    <visual>
        <global offwidth="1920" offheight="1440"/>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4" specular="0.3 0.3 0.3"/>
        <quality shadowsize="4096"/>
    </visual>
    
    <asset>
        <!-- Real 3D meshes -->
        {mesh_xml}
        
        <!-- Textures -->
        <texture name="wood_tex" type="2d" builtin="checker" rgb1="0.62 0.42 0.24" rgb2="0.58 0.38 0.20" width="256" height="256"/>
        <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.88 0.86 0.84" rgb2="0.82 0.80 0.78" width="512" height="512"/>
        
        <!-- Materials -->
        <material name="table_wood" texture="wood_tex" shininess="0.4" specular="0.25" reflectance="0.05"/>
        <material name="floor_mat" texture="floor_tex" shininess="0.15" specular="0.1"/>
        
        <!-- Object materials -->
        {mat_xml}
    </asset>
    
    <worldbody>
        <!-- Lighting: 3-point setup -->
        <light name="key_light" pos="0.3 -0.5 2.0" dir="-0.15 0.25 -1" diffuse="0.85 0.82 0.78" castshadow="true"/>
        <light name="fill_light" pos="-0.5 0.3 1.5" dir="0.25 -0.15 -1" diffuse="0.40 0.40 0.42" castshadow="false"/>
        <light name="rim_light" pos="0 0.5 1.8" dir="0 -0.3 -1" diffuse="0.25 0.25 0.28" castshadow="false"/>
        
        <!-- Floor -->
        <geom name="floor" type="plane" size="3 3 0.01" material="floor_mat"/>
        
        <!-- Table -->
        <body name="table_body" pos="0 0 {table_height:.3f}">
            <geom name="tabletop" type="box" size="{table_width/2:.3f} {table_depth/2:.3f} 0.02" material="table_wood" mass="15"/>
        </body>
        <!-- Table legs -->
        <geom name="tleg1" type="cylinder" size="0.022 {table_height/2:.3f}" pos="{table_width/2-0.06:.3f} {table_depth/2-0.06:.3f} {table_height/2:.3f}" rgba="0.52 0.35 0.20 1"/>
        <geom name="tleg2" type="cylinder" size="0.022 {table_height/2:.3f}" pos="{-table_width/2+0.06:.3f} {table_depth/2-0.06:.3f} {table_height/2:.3f}" rgba="0.52 0.35 0.20 1"/>
        <geom name="tleg3" type="cylinder" size="0.022 {table_height/2:.3f}" pos="{table_width/2-0.06:.3f} {-table_depth/2+0.06:.3f} {table_height/2:.3f}" rgba="0.52 0.35 0.20 1"/>
        <geom name="tleg4" type="cylinder" size="0.022 {table_height/2:.3f}" pos="{-table_width/2+0.06:.3f} {-table_depth/2+0.06:.3f} {table_height/2:.3f}" rgba="0.52 0.35 0.20 1"/>
        
        <!-- Objects -->
        {body_xml}
    </worldbody>
</mujoco>"""
    
    return xml, objects_used

def simulate_and_render(xml_str, output_dir):
    """Simulate physics and render multiple views."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_str)
        xml_path = f.name
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        
        # Initialize
        mujoco.mj_forward(model, data)
        
        renderer = mujoco.Renderer(model, width=1920, height=1440)
        
        # Camera views
        views = {
            "perspective": {"lookat": [0, 0, 0.8], "dist": 1.8, "elev": -35, "azim": 135},
            "front": {"lookat": [0, 0, 0.8], "dist": 1.5, "elev": -15, "azim": 180},
            "topdown": {"lookat": [0, 0, 0.8], "dist": 1.2, "elev": -85, "azim": 90},
        }
        
        # Render initial state
        for name, params in views.items():
            cam = mujoco.MjvCamera()
            cam.lookat[:3] = params["lookat"]
            cam.distance = params["dist"]
            cam.elevation = params["elev"]
            cam.azimuth = params["azim"]
            
            renderer.update_scene(data, cam)
            img = renderer.render()
            Image.fromarray(img).save(str(output_dir / f"{name}_initial.png"))
            print(f"📸 {name}_initial.png")
        
        # Physics simulation (1000 steps ≈ 2 seconds)
        print("Running physics simulation (1000 steps)...")
        for step in range(1000):
            mujoco.mj_step(model, data)
            if step % 200 == 0:
                print(f"  Step {step}/1000")
        
        # Check object positions after physics
        stable_count = 0
        fallen_count = 0
        table_height = 0.75
        
        for i in range(model.nbody):
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if bname and bname.startswith("obj_"):
                z = data.xpos[i][2]
                if z < table_height - 0.1:
                    fallen_count += 1
                    print(f"⚠️ {bname} fell (z={z:.3f})")
                else:
                    stable_count += 1
        
        print(f"Physics result: {stable_count} stable, {fallen_count} fallen")
        
        # Render final state (only best views)
        final_views = ["perspective", "topdown"]  # Only keep 2 best views
        for name in final_views:
            params = views[name]
            cam = mujoco.MjvCamera()
            cam.lookat[:3] = params["lookat"]
            cam.distance = params["dist"]
            cam.elevation = params["elev"]
            cam.azimuth = params["azim"]
            
            renderer.update_scene(data, cam)
            img = renderer.render()
            Image.fromarray(img).save(str(output_dir / f"{name}_final.png"))
            print(f"📸 {name}_final.png")
        
        renderer.close()
        
        return {"stable": stable_count, "fallen": fallen_count, "total": stable_count + fallen_count}
        
    finally:
        os.unlink(xml_path)

def main():
    print("=== Rich Tabletop Demo Scene Generator ===\\n")
    
    # Scan available objects
    print("Scanning available STL files...")
    available = scan_available_objects()
    
    print(f"Found {len(available)} available objects:")
    by_category = {}
    for k, v in available.items():
        cat = v["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(v)
    
    for cat in sorted(by_category.keys()):
        objects = by_category[cat]
        dims = objects[0]["dimensions"]
        print(f"  ✅ {cat:15s} x{len(objects)} | {dims[0]:.3f}×{dims[1]:.3f}×{dims[2]:.3f}m")
    
    # Create rich layout
    print(f"\\nCreating rich layout with 12+ objects...")
    layout = create_rich_layout()
    print(f"Planned objects: {len(layout)}")
    
    # Build XML
    print("Building MuJoCo XML...")
    xml, objects_used = build_scene_xml(layout, available)
    print(f"Objects in scene: {len(objects_used)}")
    
    # Ensure examples/demo/ directory exists 
    examples_dir = SCENE_GEN_ROOT / "examples" / "demo"
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulate and render
    print(f"\\nSimulating and rendering to {OUTPUT_DIR}...")
    results = simulate_and_render(xml, OUTPUT_DIR)
    
    print(f"\\n✅ Physics simulation complete:")
    print(f"   Stable objects: {results['stable']}")
    print(f"   Fallen objects: {results['fallen']}")
    print(f"   Total objects: {results['total']}")
    print(f"   Success rate: {results['stable']/max(results['total'],1)*100:.1f}%")
    
    # Save metadata
    scene_data = {
        "description": "Rich tabletop demo scene with corrected per-axis mesh scaling",
        "objects": objects_used,
        "layout": layout,
        "simulation_results": results,
        "available_objects": len(available),
        "categories": list(by_category.keys()),
    }
    
    with open(OUTPUT_DIR / "scene_metadata.json", 'w') as f:
        json.dump(scene_data, f, indent=2)
    
    with open(OUTPUT_DIR / "scene.xml", 'w') as f:
        f.write(xml)
    
    # Copy best images to examples/demo/ (overwrite old ones)
    print(f"\\nCopying best renders to examples/demo/...")
    best_images = [
        ("perspective_final.png", "rich_tabletop_perspective.png"),
        ("topdown_final.png", "rich_tabletop_topdown.png"),
    ]
    
    for src_name, dst_name in best_images:
        src = OUTPUT_DIR / src_name
        dst = examples_dir / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  📋 {dst_name}")
    
    print(f"\\n🎉 Rich demo scene completed!")
    print(f"   Images: {OUTPUT_DIR}")
    print(f"   Examples: {examples_dir}")

if __name__ == "__main__":
    main()