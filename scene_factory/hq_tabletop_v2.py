#!/usr/bin/env python3
"""
High-Quality Tabletop Scene Renderer v2
- NO overlapping objects (zero z_off stacking)
- AABB collision check before placement
- Physics settle before render
- Dense but collision-free layouts
"""
import os, json, math, tempfile
import numpy as np
from pathlib import Path
from PIL import Image

os.environ['MUJOCO_GL'] = 'osmesa'
import mujoco
import trimesh

SCENE_GEN = Path(__file__).parent.parent
STL_DIR = SCENE_GEN / "data" / "assets" / "_mujoco_stl"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "hq_v2"

with open(SCENE_GEN / "data" / "best_tabletop_assets.json") as f:
    BEST = json.load(f)


def get_mesh_scale(cat, target_dims):
    if cat not in BEST:
        return None, None
    info = BEST[cat]
    stl_path = STL_DIR / info["file"]
    if not stl_path.exists():
        return None, None
    m = trimesh.load(str(stl_path))
    ext = m.bounds[1] - m.bounds[0]
    perm = info["perm"]
    reord = np.array([ext[p] for p in perm])
    sc = np.array(target_dims) / np.maximum(reord, 0.001)
    mj = [0, 0, 0]
    for i in range(3):
        mj[perm[i]] = sc[i]
    return mj, str(stl_path)


def check_collision(new_pos, new_radius, placed):
    """Check if new object collides with any placed object (2D XY AABB)."""
    for p, r in placed:
        dx = abs(new_pos[0] - p[0])
        dy = abs(new_pos[1] - p[1])
        min_dist = new_radius + r + 0.01  # 1cm margin
        if dx < min_dist and dy < min_dist:
            return True
    return False


def build_xml(scene):
    objects = scene["objects"]
    table = scene.get("table", {"w": 1.00, "d": 0.70, "h": 0.75})
    tw, td, th = table["w"], table["d"], table["h"]

    meshes = []
    mats = []
    bodies = []
    placed = []  # (pos, radius) for collision check
    idx = 0

    for obj in objects:
        cat = obj["cat"]
        dims = obj["dims"]
        pos = obj["pos"]
        rot = obj.get("rot", 0)
        color = obj.get("color", "0.8 0.8 0.8 1")
        shin = obj.get("shin", 0.3)
        spec = obj.get("spec", 0.2)
        refl = obj.get("refl", 0.0)

        scale, stl = get_mesh_scale(cat, dims)
        if scale is None:
            print(f"  SKIP {cat} (no mesh)")
            continue

        # Collision check
        radius = max(dims[0], dims[1]) / 2
        if check_collision(pos, radius, placed):
            # Try small offsets
            fixed = False
            for dx, dy in [(0.03, 0), (-0.03, 0), (0, 0.03), (0, -0.03),
                           (0.05, 0.02), (-0.05, 0.02), (0.02, -0.05)]:
                new_pos = [pos[0]+dx, pos[1]+dy]
                if not check_collision(new_pos, radius, placed):
                    pos = new_pos
                    fixed = True
                    break
            if not fixed:
                print(f"  COLLISION: {cat} at {pos}, skipping")
                continue

        placed.append((pos, radius))

        mn = f"m{idx}_{cat}"
        matn = f"mat{idx}_{cat}"
        rot_rad = math.radians(rot)
        qw = math.cos(rot_rad / 2)
        qz = math.sin(rot_rad / 2)
        z = th + 0.022  # tabletop surface, NO z_off

        mass = max(dims[0] * dims[1] * dims[2] * 1200, 0.01)

        meshes.append(f'    <mesh name="{mn}" file="{stl}" scale="{scale[0]:.5f} {scale[1]:.5f} {scale[2]:.5f}"/>')
        mats.append(f'    <material name="{matn}" rgba="{color}" shininess="{shin}" specular="{spec}" reflectance="{refl}"/>')
        bodies.append(f'''
    <body name="obj{idx}_{cat}" pos="{pos[0]:.4f} {pos[1]:.4f} {z:.4f}" quat="{qw:.5f} 0 0 {qz:.5f}">
      <freejoint name="fj{idx}"/>
      <geom type="mesh" mesh="{mn}" material="{matn}" mass="{mass:.4f}"
            friction="0.8 0.005 0.001" solimp="0.95 0.95 0.01" solref="0.02 1"/>
    </body>''')
        idx += 1

    mesh_block = "\n".join(meshes)
    mat_block = "\n".join(mats)
    body_block = "\n".join(bodies)

    lx = tw / 2 - 0.06
    ly = td / 2 - 0.06
    lh = th / 2

    xml = f"""<mujoco model="hq_tabletop_v2">
  <compiler angle="degree"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>

  <visual>
    <global offwidth="1920" offheight="1440"/>
    <headlight diffuse="0.45 0.44 0.42" ambient="0.32 0.32 0.32" specular="0.35 0.35 0.35"/>
    <quality shadowsize="8192"/>
    <map znear="0.01" zfar="50"/>
  </visual>

  <asset>
{mesh_block}

    <texture name="wood_tex" type="2d" builtin="gradient" rgb1="0.58 0.40 0.22" rgb2="0.50 0.34 0.18"
             width="512" height="512" mark="random" markrgb="0.52 0.36 0.20" random="0.03"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.92 0.90 0.87" rgb2="0.86 0.84 0.81"
             width="512" height="512"/>
    <texture name="wall_tex" type="2d" builtin="flat" rgb1="0.95 0.93 0.90" width="64" height="64"/>

    <material name="table_mat" texture="wood_tex" shininess="0.30" specular="0.15" reflectance="0.02" texrepeat="3 2"/>
    <material name="floor_mat" texture="floor_tex" shininess="0.08" texrepeat="8 8"/>
    <material name="wall_mat" texture="wall_tex" shininess="0.02"/>
    <material name="leg_mat" rgba="0.48 0.32 0.18 1" shininess="0.25" specular="0.12"/>

{mat_block}
  </asset>

  <worldbody>
    <light name="key" pos="0.4 -0.6 1.8" dir="-0.2 0.3 -0.8" diffuse="0.80 0.76 0.70"
           specular="0.5 0.5 0.5" castshadow="true" cutoff="60"/>
    <light name="fill" pos="-0.5 0.2 1.5" dir="0.3 -0.1 -0.9" diffuse="0.30 0.32 0.36"/>
    <light name="rim" pos="0 0.6 1.6" dir="0 -0.3 -0.9" diffuse="0.20 0.20 0.22"/>
    <light name="bounce" pos="0 0 0.1" dir="0 0 1" diffuse="0.06 0.06 0.06" specular="0 0 0"/>

    <geom name="floor" type="plane" size="3 3 0.01" material="floor_mat"/>
    <geom name="wall" type="box" size="2 0.01 1.2" pos="0 1.0 0.6" material="wall_mat"/>

    <body name="table" pos="0 0 {th}">
      <geom name="tabletop" type="box" size="{tw/2} {td/2} 0.02" material="table_mat" mass="20"/>
    </body>
    <geom type="box" size="0.025 0.025 {lh}" pos="{lx} {ly} {lh}" material="leg_mat"/>
    <geom type="box" size="0.025 0.025 {lh}" pos="{-lx} {ly} {lh}" material="leg_mat"/>
    <geom type="box" size="0.025 0.025 {lh}" pos="{lx} {-ly} {lh}" material="leg_mat"/>
    <geom type="box" size="0.025 0.025 {lh}" pos="{-lx} {-ly} {lh}" material="leg_mat"/>

{body_block}
  </worldbody>
</mujoco>"""
    return xml, idx


def render_scene(xml_str, out_dir, name, th=0.75):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    xml_path = out / f"{name}_scene.xml"
    with open(xml_path, 'w') as f:
        f.write(xml_str)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Physics settle FIRST — let everything drop and stabilize
    print("  Settling physics (500 steps)...")
    for _ in range(500):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, width=1920, height=1440)

    views = {
        "perspective": {"l": [0.02, 0.0, th+0.08], "d": 1.40, "e": -30, "a": 145},
        "front":       {"l": [0.0, 0.0, th+0.06], "d": 1.30, "e": -25, "a": 180},
        "closeup":     {"l": [0.05, 0.05, th+0.10], "d": 0.65, "e": -28, "a": 135},
        "topdown":     {"l": [0.0, 0.0, th+0.04], "d": 1.10, "e": -88, "a": 90},
        "side":        {"l": [0.0, 0.0, th+0.06], "d": 1.30, "e": -22, "a": 230},
    }

    for vname, p in views.items():
        cam = mujoco.MjvCamera()
        cam.lookat[:] = p["l"]
        cam.distance = p["d"]
        cam.elevation = p["e"]
        cam.azimuth = p["a"]
        renderer.update_scene(data, cam)
        img = renderer.render()
        fname = f"{name}_{vname}.png"
        Image.fromarray(img).save(str(out / fname))
        print(f"  {fname}")

    # Additional physics (total 2000 steps)
    for _ in range(1500):
        mujoco.mj_step(model, data)

    cam2 = mujoco.MjvCamera()
    cam2.lookat[:] = views["perspective"]["l"]
    cam2.distance = views["perspective"]["d"]
    cam2.elevation = views["perspective"]["e"]
    cam2.azimuth = views["perspective"]["a"]
    renderer.update_scene(data, cam2)
    img2 = renderer.render()
    Image.fromarray(img2).save(str(out / f"{name}_after_physics.png"))

    stable = fallen = 0
    for i in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if nm and nm.startswith("obj"):
            z = data.xpos[i][2]
            if z < th - 0.1:
                fallen += 1
                print(f"  FALLEN: {nm} z={z:.3f}")
            else:
                stable += 1
    print(f"  Physics: {stable} stable, {fallen} fallen")

    renderer.close()
    return stable, fallen


# ============================================================
# Scene definitions — dense but ZERO overlap
# ============================================================

def breakfast_scene():
    """Dense breakfast, every object on its own spot, no stacking."""
    return {
        "name": "breakfast",
        "table": {"w": 1.00, "d": 0.70, "h": 0.75},
        "objects": [
            # Main dinner plate center-right
            {"cat": "plate", "dims": [0.26, 0.26, 0.02], "pos": [0.05, 0.06],
             "color": "0.97 0.96 0.93 1", "shin": 0.50, "spec": 0.35},
            # Cereal bowl to the left
            {"cat": "bowl", "dims": [0.15, 0.15, 0.08], "pos": [-0.18, 0.10],
             "color": "0.95 0.93 0.90 1", "shin": 0.45, "spec": 0.30},
            # Coffee mug upper-right
            {"cat": "mug", "dims": [0.08, 0.08, 0.10], "pos": [0.26, 0.18], "rot": -15,
             "color": "0.82 0.80 0.76 1", "shin": 0.40, "spec": 0.25},
            # Teacup far right
            {"cat": "cup", "dims": [0.07, 0.07, 0.08], "pos": [0.36, 0.06], "rot": 5,
             "color": "0.93 0.91 0.88 1", "shin": 0.50, "spec": 0.35},
            # Red apple — next to the plate, NOT on it
            {"cat": "apple", "dims": [0.07, 0.07, 0.07], "pos": [0.20, 0.02],
             "color": "0.80 0.15 0.10 1", "shin": 0.55, "spec": 0.30},
            # Green apple — on the other side
            {"cat": "apple", "dims": [0.065, 0.065, 0.065], "pos": [-0.04, -0.06],
             "color": "0.25 0.65 0.15 1", "shin": 0.50, "spec": 0.28},
            # Juice can
            {"cat": "can", "dims": [0.06, 0.06, 0.12], "pos": [0.38, -0.10],
             "color": "0.90 0.55 0.10 1", "shin": 0.65, "spec": 0.50},
            # Cereal box at back
            {"cat": "box", "dims": [0.12, 0.08, 0.18], "pos": [-0.35, 0.22], "rot": 8,
             "color": "0.85 0.70 0.20 1", "shin": 0.10, "spec": 0.05},
            # Bread plate (small) lower-left
            {"cat": "plate", "dims": [0.17, 0.17, 0.015], "pos": [-0.22, -0.10],
             "color": "0.96 0.95 0.92 1", "shin": 0.48, "spec": 0.32},
            # Pen
            {"cat": "pen", "dims": [0.01, 0.14, 0.01], "pos": [0.15, -0.16], "rot": 12,
             "color": "0.10 0.10 0.12 1", "shin": 0.50, "spec": 0.30},
            # Pencil
            {"cat": "pencil", "dims": [0.008, 0.17, 0.008], "pos": [0.10, -0.18], "rot": 8,
             "color": "0.82 0.72 0.12 1", "shin": 0.15, "spec": 0.08},
            # Remote control
            {"cat": "remote_control", "dims": [0.05, 0.17, 0.02], "pos": [-0.38, -0.15], "rot": -5,
             "color": "0.15 0.15 0.18 1", "shin": 0.35, "spec": 0.20},
        ]
    }


def study_desk_scene():
    """Study desk — no stacking books, everything flat on surface."""
    return {
        "name": "study_desk",
        "table": {"w": 1.20, "d": 0.70, "h": 0.75},
        "objects": [
            # Open book center
            {"cat": "book", "dims": [0.17, 0.24, 0.025], "pos": [0.0, 0.05],
             "color": "0.95 0.93 0.88 1", "shin": 0.10, "spec": 0.05},
            # Second book to the left (NOT stacked)
            {"cat": "book", "dims": [0.15, 0.22, 0.03], "pos": [-0.28, 0.15], "rot": 3,
             "color": "0.20 0.30 0.60 1", "shin": 0.12, "spec": 0.06},
            # Mug upper-right
            {"cat": "mug", "dims": [0.08, 0.08, 0.10], "pos": [0.32, 0.18], "rot": -20,
             "color": "0.25 0.25 0.28 1", "shin": 0.40, "spec": 0.25},
            # Pen 1
            {"cat": "pen", "dims": [0.01, 0.14, 0.01], "pos": [0.22, -0.05], "rot": 2,
             "color": "0.08 0.08 0.10 1", "shin": 0.50, "spec": 0.30},
            # Pen 2
            {"cat": "pen", "dims": [0.01, 0.14, 0.01], "pos": [0.26, -0.08], "rot": -5,
             "color": "0.15 0.15 0.55 1", "shin": 0.50, "spec": 0.30},
            # Pencil
            {"cat": "pencil", "dims": [0.008, 0.17, 0.008], "pos": [0.24, -0.02], "rot": 5,
             "color": "0.82 0.72 0.12 1", "shin": 0.15, "spec": 0.08},
            # Small snack plate
            {"cat": "plate", "dims": [0.16, 0.16, 0.015], "pos": [0.38, -0.10],
             "color": "0.96 0.94 0.90 1", "shin": 0.45, "spec": 0.30},
            # Apple next to snack plate
            {"cat": "apple", "dims": [0.07, 0.07, 0.07], "pos": [0.38, 0.04],
             "color": "0.78 0.12 0.10 1", "shin": 0.55, "spec": 0.30},
            # Remote control
            {"cat": "remote_control", "dims": [0.05, 0.17, 0.02], "pos": [-0.42, -0.12], "rot": -10,
             "color": "0.18 0.18 0.20 1", "shin": 0.35, "spec": 0.20},
            # Energy drink can
            {"cat": "can", "dims": [0.055, 0.055, 0.13], "pos": [-0.38, 0.12],
             "color": "0.10 0.35 0.65 1", "shin": 0.65, "spec": 0.50},
            # Tissue box
            {"cat": "box", "dims": [0.12, 0.08, 0.06], "pos": [0.45, 0.22], "rot": 5,
             "color": "0.70 0.65 0.55 1", "shin": 0.08, "spec": 0.03},
            # Bowl (pen holder proxy)
            {"cat": "bowl", "dims": [0.10, 0.10, 0.06], "pos": [0.20, 0.15],
             "color": "0.55 0.50 0.45 1", "shin": 0.30, "spec": 0.15},
        ]
    }


def tea_ceremony_scene():
    """Tea ceremony — compact, symmetrical."""
    return {
        "name": "tea_ceremony",
        "table": {"w": 0.90, "d": 0.60, "h": 0.75},
        "objects": [
            # Teapot (mug proxy, large)
            {"cat": "mug", "dims": [0.10, 0.10, 0.10], "pos": [0.0, 0.08],
             "color": "0.40 0.25 0.15 1", "shin": 0.35, "spec": 0.20, "refl": 0.02},
            # 4 teacups in arc
            {"cat": "cup", "dims": [0.06, 0.06, 0.06], "pos": [-0.16, -0.06],
             "color": "0.92 0.90 0.85 1", "shin": 0.50, "spec": 0.35},
            {"cat": "cup", "dims": [0.06, 0.06, 0.06], "pos": [-0.06, -0.12],
             "color": "0.92 0.90 0.85 1", "shin": 0.50, "spec": 0.35},
            {"cat": "cup", "dims": [0.06, 0.06, 0.06], "pos": [0.06, -0.12],
             "color": "0.92 0.90 0.85 1", "shin": 0.50, "spec": 0.35},
            {"cat": "cup", "dims": [0.06, 0.06, 0.06], "pos": [0.16, -0.06],
             "color": "0.92 0.90 0.85 1", "shin": 0.50, "spec": 0.35},
            # Sweets plate behind teapot
            {"cat": "plate", "dims": [0.20, 0.20, 0.018], "pos": [0.0, 0.24],
             "color": "0.94 0.92 0.88 1", "shin": 0.45, "spec": 0.30},
            # Matcha bowl left
            {"cat": "bowl", "dims": [0.12, 0.12, 0.07], "pos": [-0.25, 0.12],
             "color": "0.35 0.55 0.30 1", "shin": 0.30, "spec": 0.15},
            # Small bowl right
            {"cat": "bowl", "dims": [0.10, 0.10, 0.05], "pos": [0.25, 0.12],
             "color": "0.85 0.80 0.70 1", "shin": 0.30, "spec": 0.15},
            # Individual serving plates
            {"cat": "plate", "dims": [0.12, 0.12, 0.012], "pos": [-0.20, -0.18],
             "color": "0.90 0.88 0.82 1", "shin": 0.40, "spec": 0.25},
            {"cat": "plate", "dims": [0.12, 0.12, 0.012], "pos": [0.20, -0.18],
             "color": "0.90 0.88 0.82 1", "shin": 0.40, "spec": 0.25},
            # Tea caddy
            {"cat": "box", "dims": [0.08, 0.08, 0.10], "pos": [0.32, 0.22], "rot": 15,
             "color": "0.45 0.22 0.12 1", "shin": 0.30, "spec": 0.15, "refl": 0.02},
        ]
    }


if __name__ == "__main__":
    scenes = [breakfast_scene(), study_desk_scene(), tea_ceremony_scene()]

    for scene in scenes:
        name = scene["name"]
        print(f"\n{'='*60}")
        print(f"Rendering: {name} ({len(scene['objects'])} objects)")
        print(f"{'='*60}")

        xml, n_placed = build_xml(scene)
        print(f"  Placed: {n_placed}/{len(scene['objects'])}")
        out_dir = OUTPUT_DIR / name
        s, f_ = render_scene(xml, out_dir, name, scene["table"]["h"])

    print(f"\nAll done! → {OUTPUT_DIR}/")
