#!/usr/bin/env python3
"""
LLM-in-the-Loop Tabletop Environment Generator

Pipeline:
  1. LLM designs layout (JSON) based on scene description
  2. Build MuJoCo XML with real 3D assets + robot arm
  3. Simulate physics (gravity test)
  4. Render top-down + perspective views
  5. VLM reviews rendered images, identifies issues
  6. LLM refines layout based on VLM feedback
  7. Repeat steps 2-6 until satisfactory (max N iterations)

Uses OpenClaw's sessions_send to call itself (the same LLM agent).
"""
import json, os, sys, math, tempfile, shutil, time, base64
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ['MUJOCO_GL'] = 'osmesa'
import mujoco
import trimesh
from PIL import Image

# === Paths ===
SCENE_GEN_ROOT = Path(__file__).parent.parent
STL_DIR = SCENE_GEN_ROOT / "data" / "assets" / "_mujoco_stl"
TABLETOP_CATALOG_PATH = SCENE_GEN_ROOT / "data" / "tabletop_asset_catalog.json"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "llm_tabletop"

# === Tabletop Object Definitions ===
# Standard dimensions for tabletop objects (meters)
TABLETOP_DIMS = {
    "mug":       [0.08, 0.08, 0.10],
    "cup":       [0.08, 0.08, 0.10],
    "bowl":      [0.15, 0.15, 0.08],
    "plate":     [0.25, 0.25, 0.02],
    "bottle":    [0.07, 0.07, 0.25],
    "can":       [0.06, 0.06, 0.12],
    "glass":     [0.07, 0.07, 0.12],
    "fork":      [0.02, 0.18, 0.01],
    "knife":     [0.02, 0.20, 0.02],
    "spoon":     [0.03, 0.17, 0.02],
    "book":      [0.15, 0.22, 0.03],
    "pen":       [0.01, 0.15, 0.01],
    "phone":     [0.07, 0.15, 0.008],
    "laptop":    [0.35, 0.25, 0.02],
    "keyboard":  [0.44, 0.14, 0.03],
    "mouse":     [0.06, 0.10, 0.04],
    "apple":     [0.08, 0.08, 0.08],
    "orange":    [0.08, 0.08, 0.08],
    "banana":    [0.04, 0.18, 0.04],
    "box":       [0.15, 0.10, 0.10],
    "remote":    [0.05, 0.18, 0.02],
    "stapler":   [0.06, 0.15, 0.05],
    "scissors":  [0.03, 0.18, 0.01],
    "tape":      [0.08, 0.08, 0.05],
    "teapot":    [0.15, 0.12, 0.15],
}

# Object RGBA colors (for box fallback)
OBJ_RGBA = {
    "mug": "0.85 0.85 0.85 1", "cup": "0.85 0.85 0.85 1",
    "bowl": "0.92 0.90 0.85 1", "plate": "0.96 0.96 0.93 1",
    "bottle": "0.20 0.55 0.20 0.8", "can": "0.80 0.15 0.15 1",
    "glass": "0.80 0.88 0.95 0.6", "fork": "0.78 0.78 0.80 1",
    "knife": "0.78 0.78 0.80 1", "spoon": "0.78 0.78 0.80 1",
    "book": "0.20 0.30 0.65 1", "pen": "0.10 0.10 0.12 1",
    "phone": "0.12 0.12 0.14 1", "laptop": "0.25 0.25 0.28 1",
    "keyboard": "0.20 0.20 0.22 1", "mouse": "0.15 0.15 0.17 1",
    "apple": "0.85 0.15 0.10 1", "orange": "0.95 0.60 0.10 1",
    "banana": "0.95 0.90 0.20 1", "box": "0.60 0.40 0.20 1",
    "remote": "0.15 0.15 0.17 1", "stapler": "0.12 0.12 0.14 1",
    "scissors": "0.70 0.70 0.72 1", "tape": "0.80 0.75 0.50 1",
    "teapot": "0.55 0.35 0.18 1",
}

# Shape overrides for round objects
OBJ_SHAPES = {
    "apple": "sphere", "orange": "sphere", "ball": "sphere",
    "mug": "cylinder", "cup": "cylinder", "can": "cylinder",
    "glass": "cylinder", "bottle": "cylinder",
}

# === STL Lookup ===
def find_stl(category):
    """Find STL mesh for a tabletop object category."""
    cat = category.lower().replace(' ', '_')
    # Direct filename match
    for suffix in ['_0.stl', '_1.stl', '.stl']:
        path = STL_DIR / f"{cat}{suffix}"
        if path.exists():
            return path
    # Partial match
    for f in STL_DIR.glob("*.stl"):
        if cat in f.stem.lower():
            return f
    return None


# === MuJoCo XML Builder ===
def build_tabletop_xml(layout, table_size=[0.80, 0.60], table_height=0.75, 
                       include_robot=True, tmp_dir=None):
    """Build MuJoCo XML for tabletop scene with real assets + Panda-like robot arm."""
    
    tw, td = table_size
    objects = layout.get("objects", [])
    
    mesh_assets = []
    obj_bodies = []
    
    for i, obj in enumerate(objects):
        cat = obj["category"].lower().replace(' ', '_')
        pos = obj["position"]  # [x, y] relative to table center
        rot = obj.get("rotation", 0)
        
        # Get dimensions
        dims = TABLETOP_DIMS.get(cat, [0.05, 0.05, 0.05])
        if "dimensions" in obj:
            dims = [obj["dimensions"].get("width", dims[0]),
                    obj["dimensions"].get("depth", dims[1]),
                    obj["dimensions"].get("height", dims[2])]
        
        w, d, h = dims
        
        # Z position: on table surface
        z = table_height + 0.02 + h / 2  # +0.02 for table top thickness
        
        rot_rad = math.radians(rot)
        qw = math.cos(rot_rad / 2)
        qz = math.sin(rot_rad / 2)
        
        rgba = OBJ_RGBA.get(cat, "0.6 0.6 0.6 1")
        shape = OBJ_SHAPES.get(cat, "box")
        
        # Try to find real STL mesh
        stl_path = find_stl(cat)
        
        if stl_path and tmp_dir:
            # Use real mesh
            mesh_name = f"mesh_{i}_{cat}"
            dst = Path(tmp_dir) / stl_path.name
            if not dst.exists():
                shutil.copy2(stl_path, dst)
            
            # Get STL actual dimensions for scaling
            stl_mesh = trimesh.load(str(stl_path))
            stl_ext = stl_mesh.bounds[1] - stl_mesh.bounds[0]
            sx = w / max(stl_ext[0], 0.001)
            sy = d / max(stl_ext[1], 0.001)
            sz = h / max(stl_ext[2], 0.001)
            
            mesh_assets.append(
                f'<mesh name="{mesh_name}" file="{stl_path.name}" scale="{sx:.4f} {sy:.4f} {sz:.4f}"/>'
            )
            # Mesh bottom at Z=0, so body pos at table surface
            obj_bodies.append(f"""
        <body name="obj_{i}_{cat}" pos="{pos[0]:.4f} {pos[1]:.4f} {table_height + 0.02:.3f}" quat="{qw:.4f} 0 0 {qz:.4f}">
            <freejoint name="fj_{i}"/>
            <geom name="geom_{i}" type="mesh" mesh="{mesh_name}" mass="{max(w*d*h*800, 0.01):.3f}" rgba="{rgba}"/>
        </body>""")
        else:
            # Fallback: primitive shape
            mass = max(w * d * h * 800, 0.01)
            
            if shape == "sphere":
                geom = f'<geom name="geom_{i}" type="sphere" size="{w/2:.4f}" rgba="{rgba}" mass="{mass:.3f}"/>'
            elif shape == "cylinder":
                geom = f'<geom name="geom_{i}" type="cylinder" size="{w/2:.4f} {h/2:.4f}" rgba="{rgba}" mass="{mass:.3f}"/>'
            else:
                geom = f'<geom name="geom_{i}" type="box" size="{w/2:.4f} {d/2:.4f} {h/2:.4f}" rgba="{rgba}" mass="{mass:.3f}"/>'
            
            obj_bodies.append(f"""
        <body name="obj_{i}_{cat}" pos="{pos[0]:.4f} {pos[1]:.4f} {z:.4f}" quat="{qw:.4f} 0 0 {qz:.4f}">
            <freejoint name="fj_{i}"/>
            {geom}
        </body>""")
    
    mesh_xml = "\n        ".join(mesh_assets)
    obj_xml = "\n".join(obj_bodies)
    
    # Panda-like 7-DOF robot arm
    robot_xml = ""
    actuator_xml = ""
    if include_robot:
        robot_xml = f"""
        <!-- Panda-like Robot Arm (simplified 6-DOF + gripper) -->
        <body name="robot_base" pos="0 {-td/2 - 0.20:.3f} 0">
            <geom name="base" type="cylinder" size="0.08 0.03" rgba="0.25 0.25 0.28 1" mass="8"/>
            
            <body name="link0" pos="0 0 0.06">
                <geom name="link0_vis" type="cylinder" size="0.06 0.08" rgba="0.95 0.95 0.95 1" mass="3"/>
                
                <body name="link1" pos="0 0 0.16">
                    <joint name="J1" type="hinge" axis="0 0 1" range="-170 170" damping="5"/>
                    <geom name="link1_geom" type="capsule" fromto="0 0 0 0 0 0.20" size="0.04" rgba="0.95 0.95 0.95 1" mass="2"/>
                    
                    <body name="link2" pos="0 0 0.20">
                        <joint name="J2" type="hinge" axis="0 1 0" range="-100 100" damping="4"/>
                        <geom name="link2_geom" type="capsule" fromto="0 0 0 0 0 0.20" size="0.035" rgba="0.95 0.95 0.95 1" mass="1.5"/>
                        
                        <body name="link3" pos="0 0 0.20">
                            <joint name="J3" type="hinge" axis="0 0 1" range="-170 170" damping="3"/>
                            <geom name="link3_geom" type="capsule" fromto="0 0 0 0 0 0.18" size="0.03" rgba="0.95 0.95 0.95 1" mass="1.2"/>
                            
                            <body name="link4" pos="0 0 0.18">
                                <joint name="J4" type="hinge" axis="0 1 0" range="-170 170" damping="2"/>
                                <geom name="link4_geom" type="capsule" fromto="0 0 0 0 0 0.15" size="0.025" rgba="0.95 0.95 0.95 1" mass="0.8"/>
                                
                                <body name="link5" pos="0 0 0.15">
                                    <joint name="J5" type="hinge" axis="0 0 1" range="-170 170" damping="1.5"/>
                                    <geom name="link5_geom" type="capsule" fromto="0 0 0 0 0 0.10" size="0.02" rgba="0.95 0.95 0.95 1" mass="0.5"/>
                                    
                                    <body name="wrist" pos="0 0 0.10">
                                        <joint name="J6" type="hinge" axis="0 1 0" range="-170 170" damping="1"/>
                                        <geom name="wrist_geom" type="cylinder" size="0.025 0.015" rgba="0.3 0.3 0.32 1" mass="0.3"/>
                                        
                                        <!-- Parallel Gripper -->
                                        <body name="gripper_base" pos="0 0 0.02">
                                            <geom name="grip_base" type="box" size="0.03 0.02 0.005" rgba="0.3 0.3 0.32 1" mass="0.1"/>
                                            
                                            <body name="finger_left" pos="-0.03 0 0.01">
                                                <joint name="grip_l" type="slide" axis="1 0 0" range="0 0.04" damping="1"/>
                                                <geom name="finger_l" type="box" size="0.005 0.01 0.025" rgba="0.5 0.5 0.52 1" mass="0.03"/>
                                            </body>
                                            
                                            <body name="finger_right" pos="0.03 0 0.01">
                                                <joint name="grip_r" type="slide" axis="-1 0 0" range="0 0.04" damping="1"/>
                                                <geom name="finger_r" type="box" size="0.005 0.01 0.025" rgba="0.5 0.5 0.52 1" mass="0.03"/>
                                            </body>
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
        <motor joint="J4" ctrlrange="-87 87" gear="40"/>
        <motor joint="J5" ctrlrange="-12 12" gear="20"/>
        <motor joint="J6" ctrlrange="-12 12" gear="15"/>
        <motor joint="grip_l" ctrlrange="-1 1" gear="10"/>
        <motor joint="grip_r" ctrlrange="-1 1" gear="10"/>
    </actuator>"""
    
    meshdir = tmp_dir or str(STL_DIR)
    
    xml = f"""<mujoco model="tabletop_scene">
    <compiler angle="degree" meshdir="{meshdir}"/>
    
    <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
    
    <visual>
        <global offwidth="1600" offheight="1200"/>
        <headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4" specular="0.3 0.3 0.3"/>
    </visual>
    
    <asset>
        {mesh_xml}
        <texture name="wood" type="2d" builtin="checker" rgb1="0.60 0.40 0.22" rgb2="0.65 0.43 0.25" width="128" height="128"/>
        <texture name="floor" type="2d" builtin="checker" rgb1="0.85 0.85 0.85" rgb2="0.78 0.78 0.78" width="256" height="256"/>
        <material name="wood_mat" texture="wood" shininess="0.4" specular="0.2"/>
        <material name="floor_mat" texture="floor"/>
    </asset>
    
    <worldbody>
        <light pos="0 0 2.5" dir="0 0 -1" diffuse="0.95 0.92 0.88" castshadow="true"/>
        <light pos="0.4 -0.3 1.5" dir="-0.2 0.2 -1" diffuse="0.35 0.35 0.30"/>
        
        <!-- Floor -->
        <geom name="floor" type="plane" size="3 3 0.01" material="floor_mat"/>
        
        <!-- Table -->
        <body name="table" pos="0 0 {table_height:.3f}">
            <geom name="table_top" type="box" size="{tw/2:.3f} {td/2:.3f} 0.02" material="wood_mat"/>
        </body>
        <geom name="leg1" type="cylinder" size="0.025 {table_height/2:.3f}" pos="{tw/2-0.05:.3f} {td/2-0.05:.3f} {table_height/2:.3f}" rgba="0.50 0.33 0.18 1"/>
        <geom name="leg2" type="cylinder" size="0.025 {table_height/2:.3f}" pos="{-tw/2+0.05:.3f} {td/2-0.05:.3f} {table_height/2:.3f}" rgba="0.50 0.33 0.18 1"/>
        <geom name="leg3" type="cylinder" size="0.025 {table_height/2:.3f}" pos="{tw/2-0.05:.3f} {-td/2+0.05:.3f} {table_height/2:.3f}" rgba="0.50 0.33 0.18 1"/>
        <geom name="leg4" type="cylinder" size="0.025 {table_height/2:.3f}" pos="{-tw/2+0.05:.3f} {-td/2+0.05:.3f} {table_height/2:.3f}" rgba="0.50 0.33 0.18 1"/>
        
        {robot_xml}
        
        <!-- Objects on table -->
        {obj_xml}
    </worldbody>
    
    {actuator_xml}
</mujoco>"""
    
    return xml


def simulate_and_render(xml_str, output_dir, sim_steps=500):
    """Load scene in MuJoCo, simulate gravity, render views."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as tmp:
        # Copy referenced STL files
        for f in STL_DIR.glob("*.stl"):
            shutil.copy2(f, tmp)
        
        xml_path = os.path.join(tmp, "scene.xml")
        with open(xml_path, 'w') as f:
            f.write(xml_str)
        
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        
        # Render BEFORE simulation
        renderer = mujoco.Renderer(model, width=1600, height=1200)
        
        # Perspective view
        cam = mujoco.MjvCamera()
        cam.lookat[0] = 0
        cam.lookat[1] = 0
        cam.lookat[2] = 0.80
        cam.distance = 1.8
        cam.elevation = -25
        cam.azimuth = 135
        
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, cam)
        img = renderer.render()
        Image.fromarray(img).save(str(output_dir / "before_sim.png"))
        
        # Top-down view
        cam2 = mujoco.MjvCamera()
        cam2.lookat[0] = 0
        cam2.lookat[1] = 0
        cam2.lookat[2] = 0.80
        cam2.distance = 1.2
        cam2.elevation = -85
        cam2.azimuth = 90
        
        renderer.update_scene(data, cam2)
        img2 = renderer.render()
        Image.fromarray(img2).save(str(output_dir / "topdown.png"))
        
        # Simulate
        for _ in range(sim_steps):
            mujoco.mj_step(model, data)
        
        # Render AFTER simulation
        renderer.update_scene(data, cam)
        img_after = renderer.render()
        Image.fromarray(img_after).save(str(output_dir / "after_sim.png"))
        
        # Check stability
        results = {"stable": [], "fallen": []}
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith("obj_"):
                z = data.xpos[i][2]
                if z < 0.5:  # fell off table
                    results["fallen"].append(name)
                else:
                    results["stable"].append(name)
        
        renderer.close()
        
    return results


def run_pipeline(scene_description, max_iterations=3):
    """Full LLM-in-the-loop pipeline.
    
    This function is meant to be called by the LLM agent itself.
    The LLM generates the layout, this code renders it,
    then the LLM reviews and iterates.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scene_dir = OUTPUT_DIR / timestamp
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scene: {scene_description}")
    print(f"Output: {scene_dir}")
    print(f"Max iterations: {max_iterations}")
    
    return {
        "scene_dir": str(scene_dir),
        "scene_description": scene_description,
        "status": "ready_for_layout",
        "instructions": """
The pipeline is ready. To use it:

1. Design a layout as JSON:
   {"objects": [
     {"category": "mug", "position": [x, y], "rotation": deg},
     ...
   ]}
   Positions are relative to table center (0,0). 
   Table size: 0.80m x 0.60m, so x in [-0.35, 0.35], y in [-0.25, 0.25]

2. Call build_tabletop_xml(layout) to get MuJoCo XML

3. Call simulate_and_render(xml, output_dir) to get images + stability results

4. Review the images, identify issues, refine layout

5. Repeat until satisfactory
"""
    }


# === Standalone test ===
if __name__ == "__main__":
    # Quick test: manually designed breakfast scene
    layout = {
        "objects": [
            {"category": "plate", "position": [0.0, 0.0], "rotation": 0},
            {"category": "mug", "position": [0.18, 0.12], "rotation": 0},
            {"category": "bowl", "position": [-0.20, 0.05], "rotation": 0},
            {"category": "fork", "position": [-0.10, -0.15], "rotation": 5},
            {"category": "knife", "position": [0.10, -0.15], "rotation": -5},
            {"category": "spoon", "position": [0.16, -0.10], "rotation": 10},
            {"category": "glass", "position": [-0.25, 0.15], "rotation": 0},
            {"category": "apple", "position": [0.25, -0.05], "rotation": 0},
        ]
    }
    
    print("Building MuJoCo XML...", flush=True)
    
    with tempfile.TemporaryDirectory() as tmp:
        for f in STL_DIR.glob("*.stl"):
            shutil.copy2(f, tmp)
        
        xml = build_tabletop_xml(layout, include_robot=True, tmp_dir=tmp)
        
        out_dir = OUTPUT_DIR / "test"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save XML for inspection
        with open(out_dir / "scene.xml", 'w') as f:
            f.write(xml)
        
        print("Simulating and rendering...", flush=True)
        results = simulate_and_render(xml, out_dir)
    
    print(f"\nStable: {len(results['stable'])} objects")
    print(f"Fallen: {len(results['fallen'])} objects")
    if results['fallen']:
        for name in results['fallen']:
            print(f"  ⚠️ {name} fell off table!")
    else:
        print("✅ All objects stable!")
    
    print(f"\nImages saved to: {out_dir}/")
    for f in sorted(out_dir.glob("*.png")):
        print(f"  📸 {f.name}")
