#!/usr/bin/env python3
"""
High-Quality Tabletop Scene Renderer
Goal: Industrial-quality MuJoCo rendering approaching TabletopGen visual fidelity.
Uses real STL meshes + advanced MuJoCo materials + dense realistic layout.
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
OUTPUT_DIR = Path(__file__).parent / "outputs" / "hq_scenes"

with open(SCENE_GEN / "data" / "best_tabletop_assets.json") as f:
    BEST = json.load(f)


def get_mesh_scale(cat, target_dims):
    """Compute MuJoCo scale with correct axis permutation."""
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


# Rich material library with texture patterns
def build_xml(scene):
    """Build HQ MuJoCo XML for a tabletop scene."""
    objects = scene["objects"]
    table = scene.get("table", {"w": 1.00, "d": 0.70, "h": 0.75})
    tw, td, th = table["w"], table["d"], table["h"]

    meshes = []
    mats = []
    bodies = []
    idx = 0

    for obj in objects:
        cat = obj["cat"]
        dims = obj["dims"]
        pos = obj["pos"]  # [x, y] on table surface
        rot = obj.get("rot", 0)
        color = obj.get("color", "0.8 0.8 0.8 1")
        shin = obj.get("shin", 0.3)
        spec = obj.get("spec", 0.2)
        refl = obj.get("refl", 0.0)
        z_off = obj.get("z_off", 0)  # extra z offset (e.g. on plate)

        scale, stl = get_mesh_scale(cat, dims)
        if scale is None:
            continue

        mn = f"m{idx}_{cat}"
        matn = f"mat{idx}_{cat}"
        rot_rad = math.radians(rot)
        qw = math.cos(rot_rad / 2)
        qz = math.sin(rot_rad / 2)
        z = th + 0.022 + z_off  # tabletop surface

        mass = max(dims[0] * dims[1] * dims[2] * 1200, 0.005)

        meshes.append(f'    <mesh name="{mn}" file="{stl}" scale="{scale[0]:.5f} {scale[1]:.5f} {scale[2]:.5f}"/>')
        mats.append(f'    <material name="{matn}" rgba="{color}" shininess="{shin}" specular="{spec}" reflectance="{refl}"/>')
        bodies.append(f'''
    <body name="obj{idx}_{cat}" pos="{pos[0]:.4f} {pos[1]:.4f} {z:.4f}" quat="{qw:.5f} 0 0 {qz:.5f}">
      <freejoint name="fj{idx}"/>
      <geom type="mesh" mesh="{mn}" material="{matn}" mass="{mass:.4f}"
            friction="0.7 0.005 0.001" solimp="0.95 0.95 0.01" solref="0.02 1"/>
    </body>''')
        idx += 1

    mesh_block = "\n".join(meshes)
    mat_block = "\n".join(mats)
    body_block = "\n".join(bodies)

    # Table leg positions
    lx = tw / 2 - 0.06
    ly = td / 2 - 0.06
    lh = th / 2
    lz = lh

    xml = f"""<mujoco model="hq_tabletop">
  <compiler angle="degree"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>

  <visual>
    <global offwidth="1920" offheight="1440"/>
    <headlight diffuse="0.45 0.44 0.42" ambient="0.30 0.30 0.30" specular="0.35 0.35 0.35"/>
    <quality shadowsize="8192"/>
    <map znear="0.01" zfar="50"/>
  </visual>

  <asset>
{mesh_block}

    <!-- Procedural textures -->
    <texture name="wood_tex" type="2d" builtin="gradient" rgb1="0.58 0.40 0.22" rgb2="0.50 0.34 0.18"
             width="512" height="512" mark="random" markrgb="0.52 0.36 0.20" random="0.03"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.92 0.90 0.87" rgb2="0.86 0.84 0.81"
             width="512" height="512"/>
    <texture name="wall_tex" type="2d" builtin="flat" rgb1="0.95 0.93 0.90" width="64" height="64"/>
    <texture name="cloth_tex" type="2d" builtin="gradient" rgb1="0.96 0.94 0.92" rgb2="0.93 0.90 0.87"
             width="128" height="128"/>

    <material name="table_mat" texture="wood_tex" shininess="0.30" specular="0.15" reflectance="0.02" texrepeat="3 2"/>
    <material name="floor_mat" texture="floor_tex" shininess="0.08" texrepeat="8 8"/>
    <material name="wall_mat" texture="wall_tex" shininess="0.02"/>
    <material name="cloth_mat" texture="cloth_tex" shininess="0.05" specular="0.02"/>
    <material name="leg_mat" rgba="0.48 0.32 0.18 1" shininess="0.25" specular="0.12"/>

{mat_block}
  </asset>

  <worldbody>
    <!-- Warm key light from upper-left front -->
    <light name="key" pos="0.4 -0.6 1.8" dir="-0.2 0.3 -0.8" diffuse="0.80 0.76 0.70"
           specular="0.5 0.5 0.5" castshadow="true" cutoff="60"/>
    <!-- Cool fill from right -->
    <light name="fill" pos="-0.5 0.2 1.5" dir="0.3 -0.1 -0.9" diffuse="0.30 0.32 0.36"/>
    <!-- Rim / back light -->
    <light name="rim" pos="0 0.6 1.6" dir="0 -0.3 -0.9" diffuse="0.20 0.20 0.22"/>
    <!-- Ambient bounce from below -->
    <light name="bounce" pos="0 0 0.1" dir="0 0 1" diffuse="0.06 0.06 0.06" specular="0 0 0"/>

    <!-- Floor -->
    <geom name="floor" type="plane" size="3 3 0.01" material="floor_mat"/>

    <!-- Back wall -->
    <geom name="wall" type="box" size="2 0.01 1.2" pos="0 1.0 0.6" material="wall_mat"/>

    <!-- Table -->
    <body name="table" pos="0 0 {th}">
      <geom name="tabletop" type="box" size="{tw/2} {td/2} 0.02" material="table_mat" mass="20"/>
    </body>
    <!-- Table legs -->
    <geom type="box" size="0.025 0.025 {lh}" pos="{lx} {ly} {lz}" material="leg_mat"/>
    <geom type="box" size="0.025 0.025 {lh}" pos="{-lx} {ly} {lz}" material="leg_mat"/>
    <geom type="box" size="0.025 0.025 {lh}" pos="{lx} {-ly} {lz}" material="leg_mat"/>
    <geom type="box" size="0.025 0.025 {lh}" pos="{-lx} {-ly} {lz}" material="leg_mat"/>

    <!-- Optional: tablecloth area (subtle) -->
    <geom name="cloth" type="box" size="0.30 0.22 0.001" pos="0 0.05 {th+0.022}" material="cloth_mat"/>

    <!-- Scene objects -->
{body_block}
  </worldbody>
</mujoco>"""
    return xml


def render_scene(xml_str, out_dir, name, th=0.75):
    """Render HQ views."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, dir=str(out)) as f:
        f.write(xml_str)
        xp = f.name

    try:
        model = mujoco.MjModel.from_xml_path(xp)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

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

        # Physics stability test
        for _ in range(1000):
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
                if data.xpos[i][2] < th - 0.1:
                    fallen += 1
                    print(f"  FALLEN: {nm}")
                else:
                    stable += 1
        print(f"  Physics: {stable} stable, {fallen} fallen")

        renderer.close()
        return stable, fallen
    finally:
        os.unlink(xp)


# ============================================================
# Scene definitions — dense, realistic layouts
# ============================================================

def breakfast_scene():
    """Dense breakfast for one — 12+ objects, tightly grouped."""
    return {
        "name": "breakfast",
        "table": {"w": 1.00, "d": 0.70, "h": 0.75},
        "objects": [
            # Center: main plate
            {"cat": "plate", "dims": [0.26, 0.26, 0.02], "pos": [0.0, 0.06],
             "color": "0.97 0.96 0.93 1", "shin": 0.50, "spec": 0.35},
            # Bowl upper-left (cereal)
            {"cat": "bowl", "dims": [0.15, 0.15, 0.08], "pos": [-0.20, 0.15],
             "color": "0.95 0.93 0.90 1", "shin": 0.45, "spec": 0.30},
            # Mug upper-right (coffee)
            {"cat": "mug", "dims": [0.08, 0.08, 0.10], "pos": [0.22, 0.16], "rot": -15,
             "color": "0.85 0.83 0.80 1", "shin": 0.40, "spec": 0.25},
            # Cup for tea, further right
            {"cat": "cup", "dims": [0.07, 0.07, 0.08], "pos": [0.32, 0.08], "rot": 5,
             "color": "0.93 0.91 0.88 1", "shin": 0.50, "spec": 0.35},
            # Apple on the plate
            {"cat": "apple", "dims": [0.07, 0.07, 0.07], "pos": [0.03, 0.08],
             "color": "0.80 0.15 0.10 1", "shin": 0.55, "spec": 0.30, "z_off": 0.02},
            # Second apple nearby
            {"cat": "apple", "dims": [0.065, 0.065, 0.065], "pos": [-0.06, 0.04],
             "color": "0.25 0.65 0.15 1", "shin": 0.50, "spec": 0.28, "z_off": 0.02},
            # Can of juice to the right
            {"cat": "can", "dims": [0.06, 0.06, 0.12], "pos": [0.35, -0.05],
             "color": "0.90 0.55 0.10 1", "shin": 0.65, "spec": 0.50},
            # Box (cereal box) at the back
            {"cat": "box", "dims": [0.12, 0.08, 0.18], "pos": [-0.32, 0.20], "rot": 8,
             "color": "0.85 0.70 0.20 1", "shin": 0.10, "spec": 0.05},
            # Pen on the right side
            {"cat": "pen", "dims": [0.01, 0.14, 0.01], "pos": [0.18, -0.08], "rot": 12,
             "color": "0.10 0.10 0.12 1", "shin": 0.50, "spec": 0.30},
            # Remote control on the far left
            {"cat": "remote_control", "dims": [0.05, 0.17, 0.02], "pos": [-0.35, -0.08], "rot": -5,
             "color": "0.15 0.15 0.18 1", "shin": 0.35, "spec": 0.20},
            # Small plate (bread plate) upper-left
            {"cat": "plate", "dims": [0.17, 0.17, 0.015], "pos": [-0.20, -0.05],
             "color": "0.96 0.95 0.92 1", "shin": 0.48, "spec": 0.32},
            # Pencil near pen
            {"cat": "pencil", "dims": [0.008, 0.17, 0.008], "pos": [0.16, -0.12], "rot": 8,
             "color": "0.82 0.72 0.12 1", "shin": 0.15, "spec": 0.08},
        ]
    }


def study_desk_scene():
    """Study desk with books, pens, cup, and organized items."""
    return {
        "name": "study_desk",
        "table": {"w": 1.20, "d": 0.70, "h": 0.75},
        "objects": [
            # Open book center
            {"cat": "book", "dims": [0.17, 0.24, 0.025], "pos": [0.0, 0.05],
             "color": "0.95 0.93 0.88 1", "shin": 0.10, "spec": 0.05},
            # Stacked books upper-left
            {"cat": "book", "dims": [0.15, 0.22, 0.03], "pos": [-0.28, 0.18], "rot": 3,
             "color": "0.20 0.30 0.60 1", "shin": 0.12, "spec": 0.06},
            {"cat": "book", "dims": [0.16, 0.23, 0.025], "pos": [-0.27, 0.18], "rot": -2,
             "color": "0.60 0.15 0.15 1", "shin": 0.12, "spec": 0.06, "z_off": 0.03},
            # Mug with coffee upper-right
            {"cat": "mug", "dims": [0.08, 0.08, 0.10], "pos": [0.30, 0.15], "rot": -20,
             "color": "0.25 0.25 0.28 1", "shin": 0.40, "spec": 0.25},
            # Pen holder area — pens and pencils
            {"cat": "pen", "dims": [0.01, 0.14, 0.01], "pos": [0.22, -0.03], "rot": 2,
             "color": "0.08 0.08 0.10 1", "shin": 0.50, "spec": 0.30},
            {"cat": "pen", "dims": [0.01, 0.14, 0.01], "pos": [0.24, -0.04], "rot": -3,
             "color": "0.15 0.15 0.55 1", "shin": 0.50, "spec": 0.30},
            {"cat": "pencil", "dims": [0.008, 0.17, 0.008], "pos": [0.23, -0.01], "rot": 5,
             "color": "0.82 0.72 0.12 1", "shin": 0.15, "spec": 0.08},
            # Small plate (snack plate)
            {"cat": "plate", "dims": [0.16, 0.16, 0.015], "pos": [0.32, -0.10],
             "color": "0.96 0.94 0.90 1", "shin": 0.45, "spec": 0.30},
            # Apple on snack plate
            {"cat": "apple", "dims": [0.07, 0.07, 0.07], "pos": [0.32, -0.10],
             "color": "0.78 0.12 0.10 1", "shin": 0.55, "spec": 0.30, "z_off": 0.015},
            # Remote control (desk gadget)
            {"cat": "remote_control", "dims": [0.05, 0.17, 0.02], "pos": [-0.38, -0.10], "rot": -10,
             "color": "0.18 0.18 0.20 1", "shin": 0.35, "spec": 0.20},
            # Can (energy drink)
            {"cat": "can", "dims": [0.055, 0.055, 0.13], "pos": [-0.35, 0.10],
             "color": "0.10 0.35 0.65 1", "shin": 0.65, "spec": 0.50},
            # Box (tissue box or small package)
            {"cat": "box", "dims": [0.12, 0.08, 0.06], "pos": [0.40, 0.20], "rot": 5,
             "color": "0.70 0.65 0.55 1", "shin": 0.08, "spec": 0.03},
        ]
    }


def tea_ceremony_scene():
    """Tea ceremony — bowls, cups, plate, very organized."""
    return {
        "name": "tea_ceremony",
        "table": {"w": 0.90, "d": 0.60, "h": 0.75},
        "objects": [
            # Central teapot (use mug as proxy, larger)
            {"cat": "mug", "dims": [0.10, 0.10, 0.10], "pos": [0.0, 0.05],
             "color": "0.40 0.25 0.15 1", "shin": 0.35, "spec": 0.20, "refl": 0.02},
            # 4 teacups arranged in arc
            {"cat": "cup", "dims": [0.06, 0.06, 0.06], "pos": [-0.15, -0.08], "rot": 0,
             "color": "0.92 0.90 0.85 1", "shin": 0.50, "spec": 0.35},
            {"cat": "cup", "dims": [0.06, 0.06, 0.06], "pos": [-0.05, -0.12], "rot": 0,
             "color": "0.92 0.90 0.85 1", "shin": 0.50, "spec": 0.35},
            {"cat": "cup", "dims": [0.06, 0.06, 0.06], "pos": [0.05, -0.12], "rot": 0,
             "color": "0.92 0.90 0.85 1", "shin": 0.50, "spec": 0.35},
            {"cat": "cup", "dims": [0.06, 0.06, 0.06], "pos": [0.15, -0.08], "rot": 0,
             "color": "0.92 0.90 0.85 1", "shin": 0.50, "spec": 0.35},
            # Plate for sweets
            {"cat": "plate", "dims": [0.20, 0.20, 0.018], "pos": [0.0, 0.20],
             "color": "0.94 0.92 0.88 1", "shin": 0.45, "spec": 0.30},
            # Small bowl (matcha)
            {"cat": "bowl", "dims": [0.12, 0.12, 0.07], "pos": [-0.22, 0.15],
             "color": "0.35 0.55 0.30 1", "shin": 0.30, "spec": 0.15},
            # Another small bowl
            {"cat": "bowl", "dims": [0.10, 0.10, 0.05], "pos": [0.22, 0.15],
             "color": "0.85 0.80 0.70 1", "shin": 0.30, "spec": 0.15},
            # Small plate for individual serving
            {"cat": "plate", "dims": [0.12, 0.12, 0.012], "pos": [-0.15, 0.04],
             "color": "0.90 0.88 0.82 1", "shin": 0.40, "spec": 0.25},
            {"cat": "plate", "dims": [0.12, 0.12, 0.012], "pos": [0.15, 0.04],
             "color": "0.90 0.88 0.82 1", "shin": 0.40, "spec": 0.25},
            # Box (tea caddy)
            {"cat": "box", "dims": [0.08, 0.08, 0.10], "pos": [0.30, 0.20], "rot": 15,
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

        xml = build_xml(scene)
        out_dir = OUTPUT_DIR / name

        os.makedirs(out_dir, exist_ok=True)
        with open(out_dir / f"{name}_scene.xml", 'w') as f:
            f.write(xml)

        s, f_ = render_scene(xml, out_dir, name, scene["table"]["h"])
        print(f"  → {s} stable, {f_} fallen")

    print(f"\nAll done! Check {OUTPUT_DIR}/")
