#!/usr/bin/env python3
"""
Fix axis mapping issues and render a carefully designed, realistic tabletop scene.
Only uses well-validated 3D meshes with minimal distortion.
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

# Load best asset choices
with open(SCENE_GEN_ROOT / "data" / "best_tabletop_assets.json") as f:
    BEST_ASSETS = json.load(f)

# Only use objects with ratio < 2.0 (minimal distortion)
GOOD_OBJECTS = {k: v for k, v in BEST_ASSETS.items() if v["max_ratio"] < 2.0}

# Material definitions
MATERIALS = {
    "plate":  {"rgba": "0.97 0.96 0.94 1", "shin": 0.5, "spec": 0.35},
    "mug":    {"rgba": "0.90 0.88 0.85 1", "shin": 0.4, "spec": 0.3},
    "bowl":   {"rgba": "0.93 0.91 0.87 1", "shin": 0.45, "spec": 0.3},
    "apple":  {"rgba": "0.82 0.12 0.10 1", "shin": 0.55, "spec": 0.35},
    "cup":    {"rgba": "0.95 0.93 0.90 1", "shin": 0.5, "spec": 0.35},
    "can":    {"rgba": "0.75 0.15 0.15 1", "shin": 0.65, "spec": 0.5},
    "pen":    {"rgba": "0.12 0.12 0.14 1", "shin": 0.5, "spec": 0.3},
    "pencil": {"rgba": "0.85 0.75 0.15 1", "shin": 0.2, "spec": 0.1},
    "remote_control": {"rgba": "0.18 0.18 0.20 1", "shin": 0.35, "spec": 0.2},
    "box":    {"rgba": "0.62 0.44 0.24 1", "shin": 0.12, "spec": 0.05},
    "book":   {"rgba": "0.22 0.32 0.62 1", "shin": 0.15, "spec": 0.08},
}

# Physical properties per material type
FRICTION = {
    "plate": "0.6 0.005 0.001", "mug": "0.6 0.005 0.001",
    "bowl": "0.6 0.005 0.001", "cup": "0.6 0.005 0.001",
    "apple": "0.7 0.005 0.001", "can": "0.5 0.005 0.001",
    "pen": "0.5 0.005 0.001", "pencil": "0.5 0.005 0.001",
    "remote_control": "0.5 0.005 0.001", "box": "0.65 0.005 0.001",
    "book": "0.7 0.005 0.001",
}

# Target dimensions
DIMS = {
    "plate": [0.25, 0.25, 0.02], "mug": [0.08, 0.08, 0.10],
    "bowl": [0.15, 0.15, 0.08], "apple": [0.08, 0.08, 0.08],
    "cup": [0.08, 0.08, 0.10], "can": [0.06, 0.06, 0.12],
    "pen": [0.01, 0.15, 0.01], "pencil": [0.01, 0.18, 0.01],
    "remote_control": [0.05, 0.18, 0.02], "box": [0.15, 0.10, 0.10],
    "book": [0.15, 0.22, 0.03],
}

def compute_correct_scale(cat):
    """Compute the correct scale with axis permutation."""
    if cat not in BEST_ASSETS:
        return None, None, None
    
    info = BEST_ASSETS[cat]
    stl_path = STL_DIR / info["file"]
    if not stl_path.exists():
        return None, None, None
    
    mesh = trimesh.load(str(stl_path))
    actual = mesh.bounds[1] - mesh.bounds[0]
    target = np.array(DIMS.get(cat, [0.1, 0.1, 0.1]))
    perm = info["perm"]
    
    # Reorder actual dims according to best permutation
    reordered = np.array([actual[p] for p in perm])
    scales = target / np.maximum(reordered, 0.001)
    
    # For MuJoCo mesh scale: we need to map from STL axes to target axes
    # The permutation says: target[i] corresponds to STL axis perm[i]
    # So scale for STL axis perm[i] = scales[i]
    # We need scale for STL axis [0,1,2], so:
    mj_scale = [0, 0, 0]
    for i in range(3):
        mj_scale[perm[i]] = scales[i]
    
    return mj_scale, info["file"], str(stl_path)


def build_scene_xml(layout, table_w=0.80, table_d=0.60, table_h=0.75):
    """Build MuJoCo XML with correct axis mapping and real meshes only."""
    
    mesh_defs = []
    mat_defs = []
    body_defs = []
    stl_files = set()
    skipped = []
    
    for i, obj in enumerate(layout["objects"]):
        cat = obj["category"]
        pos = obj["position"]
        rot = obj.get("rotation", 0)
        
        scale, stl_file, stl_path = compute_correct_scale(cat)
        if scale is None:
            skipped.append(cat)
            continue
        
        stl_files.add(stl_file)
        
        mat_info = MATERIALS.get(cat, {"rgba": "0.6 0.6 0.6 1", "shin": 0.3, "spec": 0.2})
        fric = FRICTION.get(cat, "0.6 0.005 0.001")
        target = DIMS.get(cat, [0.1, 0.1, 0.1])
        mass = max(target[0] * target[1] * target[2] * 1500, 0.005)
        
        mesh_name = "mesh_%d_%s" % (i, cat)
        mat_name = "mat_%d_%s" % (i, cat)
        
        rot_rad = math.radians(rot)
        qw = math.cos(rot_rad / 2)
        qz = math.sin(rot_rad / 2)
        
        mesh_defs.append(
            '<mesh name="%s" file="%s" scale="%.5f %.5f %.5f"/>' % (
                mesh_name, stl_file, scale[0], scale[1], scale[2]))
        
        mat_defs.append(
            '<material name="%s" rgba="%s" shininess="%.2f" specular="%.2f"/>' % (
                mat_name, mat_info["rgba"], mat_info["shin"], mat_info["spec"]))
        
        z = table_h + 0.02  # table surface
        
        body_defs.append("""
        <body name="obj_%d_%s" pos="%.4f %.4f %.3f" quat="%.4f 0 0 %.4f">
            <freejoint name="fj_%d"/>
            <geom name="geom_%d" type="mesh" mesh="%s" material="%s" 
                  mass="%.4f" friction="%s"
                  solimp="0.95 0.95 0.01" solref="0.02 1"/>
        </body>""" % (i, cat, pos[0], pos[1], z, qw, qz, i, i, mesh_name, mat_name, mass, fric))
    
    if skipped:
        print("  Skipped (no good mesh): %s" % skipped, flush=True)
    
    mesh_xml = "\n        ".join(mesh_defs)
    mat_xml = "\n        ".join(mat_defs)
    body_xml = "\n".join(body_defs)
    
    # Panda arm
    robot_y = -table_d/2 - 0.18
    
    xml = """<mujoco model="realistic_tabletop_v2">
    <compiler angle="degree" meshdir="%s"/>
    
    <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>
    
    <visual>
        <global offwidth="1920" offheight="1440"/>
        <headlight diffuse="0.55 0.55 0.55" ambient="0.35 0.35 0.35" specular="0.4 0.4 0.4"/>
        <quality shadowsize="4096"/>
    </visual>
    
    <asset>
        %s
        
        <texture name="wood_tex" type="2d" builtin="checker" rgb1="0.62 0.42 0.24" rgb2="0.58 0.38 0.20" width="256" height="256"/>
        <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.88 0.86 0.84" rgb2="0.82 0.80 0.78" width="512" height="512"/>
        
        <material name="table_mat" texture="wood_tex" shininess="0.35" specular="0.2" reflectance="0.03"/>
        <material name="floor_mat" texture="floor_tex" shininess="0.12"/>
        <material name="robot_white" rgba="0.95 0.95 0.95 1" shininess="0.3" specular="0.2"/>
        <material name="robot_dark" rgba="0.22 0.22 0.25 1" shininess="0.4" specular="0.3"/>
        <material name="finger_mat" rgba="0.50 0.50 0.53 1" shininess="0.5" specular="0.4"/>
        
        %s
    </asset>
    
    <worldbody>
        <light name="key" pos="0.3 -0.5 2.0" dir="-0.15 0.25 -1" diffuse="0.85 0.82 0.78" castshadow="true"/>
        <light name="fill" pos="-0.5 0.3 1.5" dir="0.25 -0.15 -1" diffuse="0.40 0.40 0.42"/>
        <light name="rim" pos="0 0.5 1.8" dir="0 -0.3 -1" diffuse="0.25 0.25 0.28"/>
        
        <geom name="floor" type="plane" size="3 3 0.01" material="floor_mat"/>
        <geom name="back_wall" type="box" size="2 0.01 1.5" pos="0 1.2 0.75" rgba="0.94 0.92 0.89 1"/>
        
        <!-- Table -->
        <body name="table" pos="0 0 %.3f">
            <geom name="tabletop" type="box" size="%.3f %.3f 0.02" material="table_mat" mass="15"/>
        </body>
        <geom name="tl1" type="cylinder" size="0.022 %.3f" pos="%.3f %.3f %.3f" rgba="0.52 0.35 0.20 1"/>
        <geom name="tl2" type="cylinder" size="0.022 %.3f" pos="%.3f %.3f %.3f" rgba="0.52 0.35 0.20 1"/>
        <geom name="tl3" type="cylinder" size="0.022 %.3f" pos="%.3f %.3f %.3f" rgba="0.52 0.35 0.20 1"/>
        <geom name="tl4" type="cylinder" size="0.022 %.3f" pos="%.3f %.3f %.3f" rgba="0.52 0.35 0.20 1"/>
        
        <!-- Robot arm -->
        <body name="robot_base" pos="0 %.3f 0">
            <geom name="base0" type="cylinder" size="0.07 0.015" material="robot_dark" mass="5"/>
            <geom name="base1" type="cylinder" size="0.06 0.08" pos="0 0 0.095" material="robot_white" mass="3"/>
            <body name="shoulder" pos="0 0 0.175">
                <joint name="J1" type="hinge" axis="0 0 1" range="-166 166" damping="8" armature="0.1"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.19" size="0.045" material="robot_white" mass="2.5"/>
                <body pos="0 0 0.19">
                    <joint name="J2" type="hinge" axis="0 1 0" range="-101 101" damping="6" armature="0.1"/>
                    <geom type="capsule" fromto="0 0 0 0 0 0.19" size="0.04" material="robot_white" mass="2"/>
                    <body pos="0 0 0.19">
                        <joint name="J3" type="hinge" axis="0 0 1" range="-166 166" damping="5" armature="0.08"/>
                        <geom type="capsule" fromto="0 0 0 0 0 0.16" size="0.035" material="robot_white" mass="1.5"/>
                        <body pos="0 0 0.16">
                            <joint name="J4" type="hinge" axis="0 -1 0" range="-176 4" damping="4" armature="0.06"/>
                            <geom type="capsule" fromto="0 0 0 0 0 0.14" size="0.03" material="robot_white" mass="1"/>
                            <body pos="0 0 0.14">
                                <joint name="J5" type="hinge" axis="0 0 1" range="-166 166" damping="2" armature="0.04"/>
                                <geom type="capsule" fromto="0 0 0 0 0 0.08" size="0.025" material="robot_white" mass="0.5"/>
                                <body pos="0 0 0.08">
                                    <joint name="J6" type="hinge" axis="0 1 0" range="-166 166" damping="1.5" armature="0.03"/>
                                    <geom type="cylinder" size="0.03 0.01" material="robot_dark" mass="0.3"/>
                                    <body pos="0 0 0.02">
                                        <geom type="box" size="0.035 0.025 0.008" material="robot_dark" mass="0.15"/>
                                        <body pos="-0.035 0 0.02">
                                            <joint name="gl" type="slide" axis="1 0 0" range="0 0.04" damping="2"/>
                                            <geom type="box" size="0.006 0.012 0.03" material="finger_mat" mass="0.04" friction="1 0.005 0.001"/>
                                        </body>
                                        <body pos="0.035 0 0.02">
                                            <joint name="gr" type="slide" axis="-1 0 0" range="0 0.04" damping="2"/>
                                            <geom type="box" size="0.006 0.012 0.03" material="finger_mat" mass="0.04" friction="1 0.005 0.001"/>
                                        </body>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
        
        %s
    </worldbody>
    
    <actuator>
        <motor joint="J1" ctrlrange="-87 87" gear="80"/>
        <motor joint="J2" ctrlrange="-87 87" gear="80"/>
        <motor joint="J3" ctrlrange="-87 87" gear="60"/>
        <motor joint="J4" ctrlrange="-87 87" gear="50"/>
        <motor joint="J5" ctrlrange="-12 12" gear="25"/>
        <motor joint="J6" ctrlrange="-12 12" gear="20"/>
        <motor joint="gl" ctrlrange="-1 1" gear="15"/>
        <motor joint="gr" ctrlrange="-1 1" gear="15"/>
    </actuator>
</mujoco>""" % (
        STL_DIR,
        mesh_xml, mat_xml,
        table_h, table_w/2, table_d/2,
        table_h/2, table_w/2-0.06, table_d/2-0.06, table_h/2,
        table_h/2, -(table_w/2-0.06), table_d/2-0.06, table_h/2,
        table_h/2, table_w/2-0.06, -(table_d/2-0.06), table_h/2,
        table_h/2, -(table_w/2-0.06), -(table_d/2-0.06), table_h/2,
        robot_y,
        body_xml)
    
    return xml


def render(xml_str, out_dir, table_h=0.75):
    """Render high-quality views."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_str)
        xml_path = f.name
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        
        renderer = mujoco.Renderer(model, width=1920, height=1440)
        
        cams = {
            "main":    {"l": [0, -0.03, table_h+0.06], "d": 1.45, "e": -28, "a": 145},
            "side":    {"l": [0, -0.03, table_h+0.06], "d": 1.35, "e": -22, "a": 215},
            "top":     {"l": [0, 0, table_h+0.04], "d": 1.0, "e": -85, "a": 90},
            "close":   {"l": [0.05, 0.03, table_h+0.08], "d": 0.70, "e": -30, "a": 130},
        }
        
        for name, p in cams.items():
            cam = mujoco.MjvCamera()
            cam.lookat[:] = p["l"]
            cam.distance = p["d"]
            cam.elevation = p["e"]
            cam.azimuth = p["a"]
            renderer.update_scene(data, cam)
            img = renderer.render()
            Image.fromarray(img).save(str(out_dir / ("%s.png" % name)))
            print("  saved %s.png" % name, flush=True)
        
        # Physics test
        for _ in range(1000):
            mujoco.mj_step(model, data)
        
        cam2 = mujoco.MjvCamera()
        cam2.lookat[:] = [0, -0.03, table_h+0.06]
        cam2.distance = 1.45
        cam2.elevation = -28
        cam2.azimuth = 145
        renderer.update_scene(data, cam2)
        img2 = renderer.render()
        Image.fromarray(img2).save(str(out_dir / "after_physics.png"))
        
        stable = 0
        fallen = 0
        for i in range(model.nbody):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if nm and nm.startswith("obj_"):
                if data.xpos[i][2] < table_h - 0.1:
                    fallen += 1
                    print("  FALLEN: %s z=%.2f" % (nm, data.xpos[i][2]), flush=True)
                else:
                    stable += 1
        
        renderer.close()
        print("  Physics: %d stable, %d fallen" % (stable, fallen), flush=True)
        return stable, fallen
    finally:
        os.unlink(xml_path)


if __name__ == "__main__":
    print("Good objects (ratio < 2.0):", flush=True)
    for cat in sorted(GOOD_OBJECTS.keys()):
        info = GOOD_OBJECTS[cat]
        print("  %s: v%d, scale ratio=%.1f, verts=%d" % (cat, info["idx"], info["max_ratio"], info["n_verts"]), flush=True)
    
    # A carefully designed breakfast scene
    layout = {"objects": [
        # Plate centered, slightly forward
        {"category": "plate",  "position": [0.00, 0.03], "rotation": 0},
        # Mug to the upper right of the plate
        {"category": "mug",    "position": [0.20, 0.14], "rotation": 10},
        # Bowl to the upper left  
        {"category": "bowl",   "position": [-0.18, 0.12], "rotation": 0},
        # Apple on the plate (slightly offset)
        {"category": "apple",  "position": [0.02, 0.05], "rotation": 0},
        # Pen next to the bowl
        {"category": "pen",    "position": [-0.25, -0.03], "rotation": 15},
        # Can to the far right
        {"category": "can",    "position": [0.30, 0.03], "rotation": 0},
        # Pencil near the pen
        {"category": "pencil", "position": [-0.22, -0.12], "rotation": -8},
        # Box to the left rear
        {"category": "box",    "position": [-0.28, 0.20], "rotation": 5},
    ]}
    
    print("\nBuilding scene with %d objects..." % len(layout["objects"]), flush=True)
    xml = build_scene_xml(layout)
    
    out = OUTPUT_DIR / "fixed_breakfast"
    print("Rendering...", flush=True)
    render(xml, out)
    
    with open(out / "scene.xml", 'w') as f:
        f.write(xml)
    with open(out / "layout.json", 'w') as f:
        json.dump(layout, f, indent=2)
    
    print("\nDone! Check %s/" % out, flush=True)
