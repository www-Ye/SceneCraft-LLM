#!/usr/bin/env python3
"""
Realistic Tabletop Environment Generator.

ALL objects use real 3D mesh assets (NO boxes/cylinders).
Unified coordinate system, realistic materials, proper physics.
"""
import json, os, sys, math, tempfile, shutil
import numpy as np
from pathlib import Path

os.environ['MUJOCO_GL'] = 'osmesa'
import mujoco
import trimesh
from PIL import Image

SCENE_GEN_ROOT = Path(__file__).parent.parent
STL_DIR = SCENE_GEN_ROOT / "data" / "assets" / "_mujoco_stl"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "realistic"

# === Asset Registry ===
# Only categories that have REAL STL meshes.
# Each entry: best STL file, target dims [w,d,h], material properties
ASSET_REGISTRY = {}

def build_asset_registry():
    """Scan STL directory and build registry of available real meshes."""
    global ASSET_REGISTRY
    
    # Read tabletop catalog if available
    catalog_path = SCENE_GEN_ROOT / "data" / "tabletop_asset_catalog.json"
    catalog = {}
    if catalog_path.exists():
        with open(catalog_path) as f:
            catalog = json.load(f)
    
    # Define material properties per category
    material_props = {
        # Ceramics - white/cream, slight shininess
        "mug":    {"rgba": "0.92 0.90 0.88 1", "shininess": "0.4", "specular": "0.3", "type": "ceramic"},
        "cup":    {"rgba": "0.95 0.93 0.90 1", "shininess": "0.5", "specular": "0.3", "type": "ceramic"},
        "bowl":   {"rgba": "0.93 0.91 0.87 1", "shininess": "0.4", "specular": "0.25", "type": "ceramic"},
        "plate":  {"rgba": "0.97 0.96 0.94 1", "shininess": "0.5", "specular": "0.3", "type": "ceramic"},
        "dish":   {"rgba": "0.96 0.95 0.92 1", "shininess": "0.5", "specular": "0.3", "type": "ceramic"},
        # Glass - transparent look
        "glass":  {"rgba": "0.85 0.90 0.95 0.6", "shininess": "0.8", "specular": "0.6", "type": "glass"},
        "bottle": {"rgba": "0.25 0.55 0.25 0.75", "shininess": "0.7", "specular": "0.5", "type": "glass"},
        # Metal - shiny
        "can":    {"rgba": "0.75 0.18 0.18 1", "shininess": "0.6", "specular": "0.5", "type": "metal"},
        "scissors": {"rgba": "0.72 0.72 0.75 1", "shininess": "0.7", "specular": "0.6", "type": "metal"},
        # Food - organic colors
        "apple":  {"rgba": "0.82 0.12 0.10 1", "shininess": "0.5", "specular": "0.3", "type": "organic"},
        "banana": {"rgba": "0.95 0.88 0.22 1", "shininess": "0.3", "specular": "0.15", "type": "organic"},
        "orange": {"rgba": "0.95 0.60 0.10 1", "shininess": "0.3", "specular": "0.15", "type": "organic"},
        # Paper/wood
        "book":   {"rgba": "0.22 0.32 0.62 1", "shininess": "0.15", "specular": "0.05", "type": "paper"},
        "box":    {"rgba": "0.60 0.42 0.22 1", "shininess": "0.1", "specular": "0.05", "type": "cardboard"},
        "pen":    {"rgba": "0.12 0.12 0.14 1", "shininess": "0.5", "specular": "0.3", "type": "plastic"},
        "pencil": {"rgba": "0.85 0.75 0.15 1", "shininess": "0.2", "specular": "0.1", "type": "wood"},
        "remote_control": {"rgba": "0.15 0.15 0.18 1", "shininess": "0.4", "specular": "0.2", "type": "plastic"},
        # Kitchenware
        "fork":   {"rgba": "0.78 0.78 0.82 1", "shininess": "0.8", "specular": "0.6", "type": "metal"},
        "knife":  {"rgba": "0.78 0.78 0.82 1", "shininess": "0.8", "specular": "0.6", "type": "metal"},
        "spoon":  {"rgba": "0.78 0.78 0.82 1", "shininess": "0.8", "specular": "0.6", "type": "metal"},
        "teapot": {"rgba": "0.55 0.35 0.18 1", "shininess": "0.4", "specular": "0.25", "type": "ceramic"},
    }
    
    # Standard target dimensions
    target_dims = {
        "mug": [0.08, 0.08, 0.10], "cup": [0.08, 0.08, 0.10],
        "bowl": [0.15, 0.15, 0.08], "plate": [0.25, 0.25, 0.02],
        "dish": [0.25, 0.25, 0.02], "glass": [0.07, 0.07, 0.12],
        "bottle": [0.07, 0.07, 0.25], "can": [0.06, 0.06, 0.12],
        "apple": [0.08, 0.08, 0.08], "banana": [0.04, 0.18, 0.04],
        "orange": [0.08, 0.08, 0.08],
        "book": [0.15, 0.22, 0.03], "box": [0.15, 0.10, 0.10],
        "pen": [0.01, 0.15, 0.01], "pencil": [0.01, 0.18, 0.01],
        "remote_control": [0.05, 0.18, 0.02],
        "scissors": [0.03, 0.18, 0.01],
        "fork": [0.02, 0.18, 0.01], "knife": [0.02, 0.20, 0.02],
        "spoon": [0.03, 0.17, 0.02], "teapot": [0.15, 0.12, 0.15],
    }
    
    # Find best STL for each category
    for cat in target_dims:
        # Try _0 first, then _1
        for idx in [0, 1]:
            stl_path = STL_DIR / f"{cat}_{idx}.stl"
            if stl_path.exists():
                # Validate mesh quality
                mesh = trimesh.load(str(stl_path))
                n_verts = len(mesh.vertices)
                ext = mesh.bounds[1] - mesh.bounds[0]
                
                # Skip degenerate meshes
                if n_verts < 10 or min(ext) < 0.001:
                    continue
                
                mat = material_props.get(cat, {"rgba": "0.6 0.6 0.6 1", "shininess": "0.3", "specular": "0.2", "type": "default"})
                
                ASSET_REGISTRY[cat] = {
                    "stl": str(stl_path),
                    "stl_name": stl_path.name,
                    "target_dims": target_dims[cat],
                    "actual_dims": ext.tolist(),
                    "n_vertices": n_verts,
                    "material": mat,
                    "mass": max(target_dims[cat][0] * target_dims[cat][1] * target_dims[cat][2] * 1200, 0.01),
                }
                break
    
    return ASSET_REGISTRY

# Build on import
build_asset_registry()


def get_available_objects():
    """Return list of categories with real meshes."""
    return list(ASSET_REGISTRY.keys())


def build_realistic_xml(layout, table_size=[0.80, 0.60], table_height=0.75, include_robot=True):
    """Build MuJoCo XML using ONLY real mesh assets. No fallback to primitives."""
    tw, td = table_size
    objects = layout.get("objects", [])
    
    mesh_assets = []
    material_defs = []
    obj_bodies = []
    
    # Collect unique STL files needed
    stl_files_needed = set()
    
    for i, obj in enumerate(objects):
        cat = obj["category"].lower().replace(' ', '_')
        
        if cat not in ASSET_REGISTRY:
            print(f"  ⚠️ Skipping '{cat}' — no real mesh available")
            continue
        
        asset = ASSET_REGISTRY[cat]
        pos = obj.get("position", [0, 0])
        rot = obj.get("rotation", 0)
        
        # Scale: target / actual STL size
        actual = asset["actual_dims"]
        target = asset["target_dims"]
        sx = target[0] / max(actual[0], 0.001)
        sy = target[1] / max(actual[1], 0.001)
        sz = target[2] / max(actual[2], 0.001)
        
        # Rotation quaternion
        rot_rad = math.radians(rot)
        qw = math.cos(rot_rad / 2)
        qz = math.sin(rot_rad / 2)
        
        mat = asset["material"]
        mesh_name = f"mesh_{i}_{cat}"
        mat_name = f"mat_{i}_{cat}"
        
        stl_files_needed.add(asset["stl_name"])
        
        mesh_assets.append(
            f'<mesh name="{mesh_name}" file="{asset["stl_name"]}" '
            f'scale="{sx:.4f} {sy:.4f} {sz:.4f}"/>'
        )
        
        material_defs.append(
            f'<material name="{mat_name}" '
            f'rgba="{mat["rgba"]}" '
            f'shininess="{mat["shininess"]}" '
            f'specular="{mat["specular"]}"/>'
        )
        
        # Body at table surface level (mesh bottom = 0)
        z = table_height + 0.02  # table top thickness
        
        # Friction: depends on material type
        friction = {"ceramic": "0.6 0.005 0.001", "glass": "0.4 0.005 0.001",
                    "metal": "0.5 0.005 0.001", "organic": "0.7 0.005 0.001",
                    "paper": "0.8 0.005 0.001", "cardboard": "0.7 0.005 0.001",
                    "plastic": "0.5 0.005 0.001", "wood": "0.6 0.005 0.001",
                    "default": "0.6 0.005 0.001"}.get(mat["type"], "0.6 0.005 0.001")
        
        obj_bodies.append(f"""
        <body name="obj_{i}_{cat}" pos="{pos[0]:.4f} {pos[1]:.4f} {z:.3f}" quat="{qw:.4f} 0 0 {qz:.4f}">
            <freejoint name="fj_{i}"/>
            <geom name="geom_{i}" type="mesh" mesh="{mesh_name}" 
                  material="{mat_name}" mass="{asset['mass']:.3f}"
                  friction="{friction}" 
                  solimp="0.95 0.95 0.01" solref="0.02 1"/>
        </body>""")
    
    mesh_xml = "\n        ".join(mesh_assets)
    mat_xml = "\n        ".join(material_defs)
    body_xml = "\n".join(obj_bodies)
    
    # Robot arm (Panda-like, white/silver finish)
    robot_xml = ""
    actuator_xml = ""
    if include_robot:
        robot_xml = f"""
        <!-- Franka Panda-style Robot Arm (simplified 6-DOF + parallel gripper) -->
        <body name="robot_base" pos="0 {-td/2 - 0.18:.3f} 0">
            <geom name="base_mount" type="cylinder" size="0.07 0.015" rgba="0.15 0.15 0.17 1" mass="5"/>
            <geom name="base_col" type="cylinder" size="0.06 0.08" pos="0 0 0.095" rgba="0.95 0.95 0.95 1" mass="3"/>
            
            <body name="shoulder" pos="0 0 0.175">
                <joint name="J1" type="hinge" axis="0 0 1" range="-166 166" damping="8" armature="0.1"/>
                <geom name="shoulder_geom" type="capsule" fromto="0 0 0 0 0 0.19" size="0.045" rgba="0.95 0.95 0.95 1" mass="2.5"/>
                
                <body name="upper_arm" pos="0 0 0.19">
                    <joint name="J2" type="hinge" axis="0 1 0" range="-101 101" damping="6" armature="0.1"/>
                    <geom name="upper_geom" type="capsule" fromto="0 0 0 0 0 0.19" size="0.04" rgba="0.95 0.95 0.95 1" mass="2.0"/>
                    
                    <body name="elbow" pos="0 0 0.19">
                        <joint name="J3" type="hinge" axis="0 0 1" range="-166 166" damping="5" armature="0.08"/>
                        <geom name="elbow_geom" type="capsule" fromto="0 0 0 0 0 0.16" size="0.035" rgba="0.95 0.95 0.95 1" mass="1.5"/>
                        
                        <body name="forearm" pos="0 0 0.16">
                            <joint name="J4" type="hinge" axis="0 -1 0" range="-176 4" damping="4" armature="0.06"/>
                            <geom name="forearm_geom" type="capsule" fromto="0 0 0 0 0 0.14" size="0.03" rgba="0.95 0.95 0.95 1" mass="1.0"/>
                            
                            <body name="wrist1" pos="0 0 0.14">
                                <joint name="J5" type="hinge" axis="0 0 1" range="-166 166" damping="2" armature="0.04"/>
                                <geom name="wrist1_geom" type="capsule" fromto="0 0 0 0 0 0.08" size="0.025" rgba="0.95 0.95 0.95 1" mass="0.5"/>
                                
                                <body name="wrist2" pos="0 0 0.08">
                                    <joint name="J6" type="hinge" axis="0 1 0" range="-166 166" damping="1.5" armature="0.03"/>
                                    <geom name="wrist2_geom" type="cylinder" size="0.03 0.01" rgba="0.25 0.25 0.28 1" mass="0.3"/>
                                    
                                    <!-- Parallel Jaw Gripper -->
                                    <body name="gripper_mount" pos="0 0 0.02">
                                        <geom name="grip_mount" type="box" size="0.035 0.025 0.008" rgba="0.25 0.25 0.28 1" mass="0.15"/>
                                        
                                        <body name="left_finger" pos="-0.035 0 0.02">
                                            <joint name="grip_l" type="slide" axis="1 0 0" range="0 0.04" damping="2" armature="0.02"/>
                                            <geom name="lfinger" type="box" size="0.006 0.012 0.03" rgba="0.50 0.50 0.52 1" mass="0.04" friction="1.0 0.005 0.001"/>
                                        </body>
                                        <body name="right_finger" pos="0.035 0 0.02">
                                            <joint name="grip_r" type="slide" axis="-1 0 0" range="0 0.04" damping="2" armature="0.02"/>
                                            <geom name="rfinger" type="box" size="0.006 0.012 0.03" rgba="0.50 0.50 0.52 1" mass="0.04" friction="1.0 0.005 0.001"/>
                                        </body>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>"""
        
        actuator_xml = """
    <actuator>
        <motor joint="J1" ctrlrange="-87 87" gear="80"/>
        <motor joint="J2" ctrlrange="-87 87" gear="80"/>
        <motor joint="J3" ctrlrange="-87 87" gear="60"/>
        <motor joint="J4" ctrlrange="-87 87" gear="50"/>
        <motor joint="J5" ctrlrange="-12 12" gear="25"/>
        <motor joint="J6" ctrlrange="-12 12" gear="20"/>
        <motor joint="grip_l" ctrlrange="-1 1" gear="15"/>
        <motor joint="grip_r" ctrlrange="-1 1" gear="15"/>
    </actuator>"""
    
    xml = f"""<mujoco model="realistic_tabletop">
    <compiler angle="degree" meshdir="{STL_DIR}"/>
    
    <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>
    
    <visual>
        <global offwidth="1920" offheight="1440"/>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0.4 0.4 0.4"/>
        <quality shadowsize="4096"/>
    </visual>
    
    <asset>
        <!-- Real 3D meshes -->
        {mesh_xml}
        
        <!-- Textures -->
        <texture name="wood_tex" type="2d" builtin="checker" rgb1="0.62 0.42 0.24" rgb2="0.58 0.38 0.20" width="256" height="256"/>
        <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.88 0.86 0.84" rgb2="0.82 0.80 0.78" width="512" height="512"/>
        <texture name="wall_tex" type="2d" builtin="gradient" rgb1="0.95 0.93 0.90" rgb2="0.88 0.86 0.83" width="64" height="64"/>
        
        <!-- Materials -->
        <material name="table_wood" texture="wood_tex" shininess="0.4" specular="0.25" reflectance="0.05"/>
        <material name="floor_mat" texture="floor_tex" shininess="0.15" specular="0.1"/>
        <material name="wall_mat" texture="wall_tex" shininess="0.1"/>
        
        <!-- Per-object materials -->
        {mat_xml}
    </asset>
    
    <worldbody>
        <!-- Lighting: 3-point setup for realism -->
        <light name="key_light" pos="0.3 -0.5 2.0" dir="-0.15 0.25 -1" diffuse="0.85 0.82 0.78" castshadow="true"/>
        <light name="fill_light" pos="-0.5 0.3 1.5" dir="0.25 -0.15 -1" diffuse="0.40 0.40 0.42" castshadow="false"/>
        <light name="rim_light" pos="0 0.5 1.8" dir="0 -0.3 -1" diffuse="0.25 0.25 0.28" castshadow="false"/>
        
        <!-- Floor -->
        <geom name="floor" type="plane" size="3 3 0.01" material="floor_mat"/>
        
        <!-- Background wall (optional, for depth) -->
        <geom name="back_wall" type="box" size="2 0.01 1.5" pos="0 1.2 0.75" material="wall_mat"/>
        
        <!-- Table (detailed) -->
        <body name="table_body" pos="0 0 {table_height:.3f}">
            <geom name="tabletop" type="box" size="{tw/2:.3f} {td/2:.3f} 0.02" material="table_wood" mass="15"/>
        </body>
        <!-- Table legs (tapered cylinders) -->
        <geom name="tleg1" type="cylinder" size="0.022 {table_height/2:.3f}" pos="{tw/2-0.06:.3f} {td/2-0.06:.3f} {table_height/2:.3f}" rgba="0.52 0.35 0.20 1"/>
        <geom name="tleg2" type="cylinder" size="0.022 {table_height/2:.3f}" pos="{-tw/2+0.06:.3f} {td/2-0.06:.3f} {table_height/2:.3f}" rgba="0.52 0.35 0.20 1"/>
        <geom name="tleg3" type="cylinder" size="0.022 {table_height/2:.3f}" pos="{tw/2-0.06:.3f} {-td/2+0.06:.3f} {table_height/2:.3f}" rgba="0.52 0.35 0.20 1"/>
        <geom name="tleg4" type="cylinder" size="0.022 {table_height/2:.3f}" pos="{-tw/2+0.06:.3f} {-td/2+0.06:.3f} {table_height/2:.3f}" rgba="0.52 0.35 0.20 1"/>
        
        {robot_xml}
        
        <!-- Objects (all real meshes) -->
        {body_xml}
    </worldbody>
    
    {actuator_xml}
</mujoco>"""
    
    return xml, stl_files_needed


def render_views(xml_str, output_dir, table_height=0.75):
    """Render multiple high-quality views."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_str)
        xml_path = f.name
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        
        renderer = mujoco.Renderer(model, width=1920, height=1440)
        
        views = {
            "perspective_front": {"lookat": [0, 0, table_height+0.05], "dist": 1.6, "elev": -25, "azim": 150},
            "perspective_side": {"lookat": [0, 0, table_height+0.05], "dist": 1.5, "elev": -20, "azim": 210},
            "topdown": {"lookat": [0, 0, table_height+0.05], "dist": 1.1, "elev": -85, "azim": 90},
            "closeup": {"lookat": [0, 0.05, table_height+0.08], "dist": 0.8, "elev": -30, "azim": 135},
        }
        
        for name, params in views.items():
            cam = mujoco.MjvCamera()
            cam.lookat[0] = params["lookat"][0]
            cam.lookat[1] = params["lookat"][1]
            cam.lookat[2] = params["lookat"][2]
            cam.distance = params["dist"]
            cam.elevation = params["elev"]
            cam.azimuth = params["azim"]
            
            renderer.update_scene(data, cam)
            img = renderer.render()
            
            Image.fromarray(img).save(str(output_dir / f"{name}.png"))
            print(f"  📸 {name}.png", flush=True)
        
        # Simulate physics and render after
        for _ in range(1000):
            mujoco.mj_step(model, data)
        
        cam_after = mujoco.MjvCamera()
        cam_after.lookat[:] = [0, 0, table_height+0.05]
        cam_after.distance = 1.6
        cam_after.elevation = -25
        cam_after.azimuth = 150
        renderer.update_scene(data, cam_after)
        img_after = renderer.render()
        Image.fromarray(img_after).save(str(output_dir / "after_physics.png"))
        print(f"  📸 after_physics.png", flush=True)
        
        # Check stability
        stable_count = 0
        fallen_count = 0
        for i in range(model.nbody):
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if bname and bname.startswith("obj_"):
                z = data.xpos[i][2]
                if z < table_height - 0.1:
                    fallen_count += 1
                    print(f"  ⚠️ {bname} fell (z={z:.2f})")
                else:
                    stable_count += 1
        
        renderer.close()
        
        return {"stable": stable_count, "fallen": fallen_count}
        
    finally:
        os.unlink(xml_path)


if __name__ == "__main__":
    print("=== Realistic Tabletop Environment Generator ===\n", flush=True)
    
    print(f"Available real mesh objects ({len(ASSET_REGISTRY)}):", flush=True)
    for cat, info in sorted(ASSET_REGISTRY.items()):
        dims = info["target_dims"]
        print(f"  ✅ {cat:20s} {dims[0]:.2f}x{dims[1]:.2f}x{dims[2]:.2f}m  "
              f"({info['n_vertices']} verts, {info['material']['type']})", flush=True)
    
    # Demo: breakfast scene with ONLY real meshes
    layout = {
        "objects": [
            {"category": "plate", "position": [0.0, 0.02], "rotation": 0},
            {"category": "mug", "position": [0.20, 0.15], "rotation": 15},
            {"category": "bowl", "position": [-0.22, 0.08], "rotation": 0},
            {"category": "apple", "position": [0.25, -0.08], "rotation": 0},
            {"category": "banana", "position": [-0.28, -0.12], "rotation": 30},
            {"category": "book", "position": [-0.05, -0.18], "rotation": 5},
            {"category": "pen", "position": [0.12, -0.18], "rotation": -10},
            {"category": "bottle", "position": [0.30, 0.05], "rotation": 0},
            {"category": "can", "position": [-0.30, 0.18], "rotation": 0},
        ]
    }
    
    print(f"\nBuilding scene with {len(layout['objects'])} objects...", flush=True)
    xml, stl_needed = build_realistic_xml(layout, include_robot=True)
    print(f"STL files used: {len(stl_needed)}", flush=True)
    
    out_dir = OUTPUT_DIR / "demo_breakfast"
    print(f"\nRendering to {out_dir}...", flush=True)
    results = render_views(xml, out_dir)
    
    print(f"\n✅ Stable: {results['stable']}, Fallen: {results['fallen']}")
    
    # Save scene XML
    with open(out_dir / "scene.xml", 'w') as f:
        f.write(xml)
    with open(out_dir / "layout.json", 'w') as f:
        json.dump(layout, f, indent=2)
    
    print(f"\n🎉 Done! Check {out_dir}/")
