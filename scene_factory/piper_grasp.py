#!/usr/bin/env python3
"""
Piper 机械臂抓取苹果规划模块
使用 mujoco_menagerie 的真实 Piper arm + 真实 STL 苹果
Mocap body + weld constraint 末端跟踪
"""

import os, sys, json, math, tempfile
import numpy as np
from pathlib import Path
from PIL import Image

os.environ['MUJOCO_GL'] = 'osmesa'
import mujoco

SCENE_DIR = Path(__file__).parent
PIPER_DIR = SCENE_DIR / "mujoco_menagerie" / "agilex_piper"
STL_DIR = SCENE_DIR.parent / "data" / "assets" / "_mujoco_stl"
OUTPUT_DIR = SCENE_DIR / "outputs" / "piper_grasp"

TABLE_H = 0.75
TABLE_W = 0.80
TABLE_D = 0.60


def build_scene_xml(apple_pos, plate_pos, task_name="task"):
    """Build XML with real Piper arm + real STL apple + table."""

    # Load best asset info
    best_path = SCENE_DIR.parent / "data" / "best_tabletop_assets.json"
    with open(best_path) as f:
        best = json.load(f)

    apple_info = best["apple"]
    plate_info = best["plate"]

    # Compute apple scale
    import trimesh
    am = trimesh.load(str(STL_DIR / apple_info["file"]))
    a_ext = am.bounds[1] - am.bounds[0]
    a_target = np.array([0.07, 0.07, 0.07])
    a_perm = apple_info["perm"]
    a_reord = np.array([a_ext[p] for p in a_perm])
    a_sc = a_target / np.maximum(a_reord, 0.001)
    apple_scale = [0,0,0]
    for i in range(3):
        apple_scale[a_perm[i]] = a_sc[i]

    # Compute plate scale
    pm = trimesh.load(str(STL_DIR / plate_info["file"]))
    p_ext = pm.bounds[1] - pm.bounds[0]
    p_target = np.array([0.22, 0.22, 0.02])
    p_perm = plate_info["perm"]
    p_reord = np.array([p_ext[p] for p in p_perm])
    p_sc = p_target / np.maximum(p_reord, 0.001)
    plate_scale = [0,0,0]
    for i in range(3):
        plate_scale[p_perm[i]] = p_sc[i]

    # Piper arm position: on a pedestal at the table front edge
    # Piper reaches ~0.5m in its forward direction (X at rest)
    # We rotate base 90° so it reaches into +Y direction (onto table)
    piper_base_y = -(TABLE_D / 2 + 0.05)

    xml = """<mujoco model="piper_grasp_scene">
  <compiler angle="radian" meshdir="{piper_assets}" autolimits="true"/>
  
  <option integrator="implicitfast" cone="elliptic" impratio="10" timestep="0.002"
          gravity="0 0 -9.81"/>
  
  <visual>
    <global offwidth="1920" offheight="1440"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0.3 0.3 0.3"/>
    <quality shadowsize="4096"/>
  </visual>

  <default>
    <default class="piper">
      <joint frictionloss="0.3" armature="0.005"/>
      <position inheritrange="1" forcerange="-100 100"/>
      <default class="finger">
        <joint frictionloss="0" type="slide"/>
        <position forcerange="-10 10"/>
      </default>
    </default>
  </default>

  <asset>
    <!-- Piper mesh assets -->
    <mesh name="base_link" file="base_link.stl"/>
    <mesh name="link1" file="link1.stl"/>
    <mesh name="link2" file="link2.stl"/>
    <mesh name="link2_gray" file="link2_gray.stl"/>
    <mesh name="link3" file="link3.stl"/>
    <mesh name="link4" file="link4.stl"/>
    <mesh name="link5" file="link5.stl"/>
    <mesh name="link6" file="link6.stl"/>
    <mesh name="link7" file="link7.stl"/>
    <mesh name="link8" file="link8.stl"/>

    <!-- Object meshes -->
    <mesh name="apple_mesh" file="{apple_stl}" scale="{a_s0:.5f} {a_s1:.5f} {a_s2:.5f}"/>
    <mesh name="plate_mesh" file="{plate_stl}" scale="{p_s0:.5f} {p_s1:.5f} {p_s2:.5f}"/>

    <!-- Textures & materials -->
    <texture name="wood_tex" type="2d" builtin="checker" rgb1="0.62 0.42 0.24" rgb2="0.58 0.38 0.20"
             width="256" height="256"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.88 0.86 0.84" rgb2="0.82 0.80 0.78"
             width="512" height="512"/>

    <material name="table_mat" texture="wood_tex" shininess="0.35" specular="0.2" reflectance="0.03"/>
    <material name="floor_mat" texture="floor_tex" shininess="0.12"/>
    <material name="apple_mat" rgba="0.82 0.12 0.10 1" shininess="0.55" specular="0.35"/>
    <material name="plate_mat" rgba="0.97 0.96 0.94 1" shininess="0.5" specular="0.35"/>
    
    <!-- Piper materials -->
    <material name="gray_mat" rgba="0.59 0.59 0.59 1"/>
    <material name="light_gray_mat" rgba="0.98 0.98 0.98 1"/>
    <material name="light_medium_gray_mat" rgba="0.85 0.85 0.85 1"/>
    <material name="dark_gray_mat" rgba="0.086 0.086 0.086 1"/>
    <material name="darker_gray_mat" rgba="0.14 0.14 0.14 1"/>
    <material name="white_mat" rgba="1 1 1 1"/>
    <material name="red_mat" rgba="0.82 0.15 0.15 1"/>
    <material name="black_mat" rgba="0 0 0 1"/>
    <material name="light_blue" rgba="0.79 0.82 0.93 1"/>
    <material name="target_vis" rgba="1 0 0 0.3"/>
  </asset>

  <worldbody>
    <light name="key" pos="0.3 -0.5 2.0" dir="-0.15 0.25 -1" diffuse="0.85 0.82 0.78" castshadow="true"/>
    <light name="fill" pos="-0.5 0.3 1.5" dir="0.25 -0.15 -1" diffuse="0.40 0.40 0.42"/>
    <light name="rim" pos="0 0.5 1.8" dir="0 -0.3 -1" diffuse="0.25 0.25 0.28"/>

    <geom name="floor" type="plane" size="3 3 0.01" material="floor_mat"/>

    <!-- Table -->
    <body name="table" pos="0 0 {th}">
      <geom name="tabletop" type="box" size="{tw2} {td2} 0.02" material="table_mat" mass="15"/>
    </body>
    <geom name="tl1" type="cylinder" size="0.022 {lh}" pos="{lx} {ly} {lz}" rgba="0.52 0.35 0.20 1"/>
    <geom name="tl2" type="cylinder" size="0.022 {lh}" pos="-{lx} {ly} {lz}" rgba="0.52 0.35 0.20 1"/>
    <geom name="tl3" type="cylinder" size="0.022 {lh}" pos="{lx} -{ly} {lz}" rgba="0.52 0.35 0.20 1"/>
    <geom name="tl4" type="cylinder" size="0.022 {lh}" pos="-{lx} -{ly} {lz}" rgba="0.52 0.35 0.20 1"/>

    <!-- Plate (static) -->
    <body name="plate" pos="{px} {py} {pz}">
      <geom name="plate_geom" type="mesh" mesh="plate_mesh" material="plate_mat"
            mass="0.3" friction="0.7 0.005 0.001"/>
    </body>

    <!-- Apple (freejoint, graspable) -->
    <body name="apple" pos="{ax} {ay} {az}">
      <freejoint name="apple_fj"/>
      <geom name="apple_geom" type="mesh" mesh="apple_mesh" material="apple_mat"
            mass="0.15" friction="0.8 0.005 0.001"
            solimp="0.95 0.95 0.01" solref="0.02 1"/>
    </body>

    <!-- Piper arm on pedestal -->
    <body name="pedestal" pos="0 {ped_y} 0">
      <geom name="ped_base" type="cylinder" size="0.08 0.02" rgba="0.2 0.2 0.2 1"/>
      <geom name="ped_col" type="cylinder" size="0.05 {ped_h2}" pos="0 0 {ped_h2}" rgba="0.25 0.25 0.28 1"/>
      <geom name="ped_top" type="cylinder" size="0.08 0.015" pos="0 0 {ped_top}" rgba="0.2 0.2 0.2 1"/>

      <!-- Rotated 90° around Z so arm reaches into +Y (onto table) -->
      <body name="base_link" pos="0 0 {piper_z}" quat="0.707107 0 0 0.707107" childclass="piper" gravcomp="1">
        <light name="spotlight" mode="targetbodycom" target="link8_body" pos="0 0 1"/>
        <inertial mass="0.738" pos="0 0 0.0485" diaginertia="0.001 0.001 0.0005"/>
        <geom type="mesh" mesh="base_link" material="darker_gray_mat"/>

        <body name="link1_body" pos="0 0 0.123" quat="0.707105 0 0 -0.707108" gravcomp="1">
          <joint name="joint1" axis="0 0 1" range="-2.618 2.618"/>
          <inertial mass="0.408" pos="0 0 0" diaginertia="0.0003 0.0003 0.0002"/>
          <geom type="mesh" mesh="link1" material="light_gray_mat"/>

          <body name="link2_body" quat="0.499998 0.5 -0.500002 -0.5" gravcomp="1">
            <joint name="joint2" axis="0 0 1" range="0 3.14"/>
            <inertial mass="1.474" pos="0.13 0.02 0" diaginertia="0.01 0.01 0.001"/>
            <geom type="mesh" mesh="link2" material="white_mat"/>
            <geom type="mesh" mesh="link2_gray" material="light_medium_gray_mat"/>

            <body name="link3_body" pos="0.28358 0.028726 0" quat="0.998726 0 0 0.0504536" gravcomp="1">
              <joint name="joint3" pos="0 0 0" axis="0 0 1" range="-2.697 0"/>
              <inertial mass="0.630" pos="-0.12 0.035 0" diaginertia="0.005 0.005 0.0005"/>
              <geom type="mesh" mesh="link3" material="white_mat"/>

              <body name="link4_body" pos="-0.24221 0.068514 0" quat="0.544767 -0.544769 -0.450809 0.450808" gravcomp="1">
                <joint name="joint4" axis="0 0 1" range="-1.832 1.832"/>
                <inertial mass="0.202" pos="0 0 0.075" diaginertia="0.001 0.001 0.0002"/>
                <geom type="mesh" mesh="link4" material="white_mat"/>

                <body name="link5_body" pos="0 0 0.15" gravcomp="1">
                  <joint name="joint5" axis="0 0 1" range="-1.22 1.22"/>
                  <inertial mass="0.081" pos="0 0 0.01" diaginertia="0.0002 0.0002 0.00005"/>
                  <geom type="mesh" mesh="link5" material="white_mat"/>

                  <body name="link6_body" pos="0 0 0" quat="0.707107 0.707107 0 0" gravcomp="1">
                    <joint name="joint6" axis="0 0 1" range="-3.14 3.14"/>
                    <inertial mass="0.165" pos="0 0 0" diaginertia="0.0001 0.0001 0.00005"/>
                    <geom type="mesh" mesh="link6" material="white_mat"/>

                    <body name="link7_body" pos="0 0.044 0" quat="0 -0.707107 0 0.707107" gravcomp="1">
                      <inertial mass="0.069" pos="0 0 0.026" diaginertia="0.00005 0.00005 0.00002"/>
                      <geom type="mesh" mesh="link7" material="dark_gray_mat"/>

                      <!-- Left finger -->
                      <body name="link8_body" pos="-0.0135 0 0.06435" gravcomp="1">
                        <joint name="joint7" axis="0 0 -1" range="0 0.035"/>
                        <inertial mass="0.024" pos="0 0 0.012" diaginertia="0.00001 0.00001 0.000005"/>
                        <geom type="mesh" mesh="link8" material="dark_gray_mat"/>
                        <geom name="finger_left_col" type="box" size="0.006 0.009 0.0145"
                              pos="-0.0025 0 0.022" rgba="0 0 0 0" friction="1.5 0.01 0.001"/>
                      </body>

                      <!-- Right finger -->
                      <body name="link9_body" pos="0.0135 0 0.06435" quat="0 0 0 1" gravcomp="1">
                        <joint name="joint8" axis="0 0 1" range="-0.035 0"/>
                        <inertial mass="0.024" pos="0 0 0.012" diaginertia="0.00001 0.00001 0.000005"/>
                        <geom type="mesh" mesh="link8" material="dark_gray_mat"/>
                        <geom name="finger_right_col" type="box" size="0.006 0.009 0.0145"
                              pos="0.0025 0 0.022" rgba="0 0 0 0" friction="1.5 0.01 0.001"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Mocap target for end-effector tracking -->
    <body name="mocap_target" mocap="true" pos="0 {ped_y} {piper_z_plus}">
      <geom type="sphere" size="0.012" material="target_vis" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <equality>
    <joint joint1="joint8" joint2="joint7" polycoef="0 -1 0 0 0"/>
    <weld body1="mocap_target" body2="link7_body" solref="0.01 1" solimp="0.95 0.99 0.001"/>
  </equality>

  <contact>
    <exclude body1="base_link" body2="link1_body"/>
  </contact>

  <actuator>
    <position name="act1" joint="joint1" class="piper" kp="80" kv="5"/>
    <position name="act2" joint="joint2" class="piper" kp="80" kv="5"/>
    <position name="act3" joint="joint3" class="piper" kp="80" kv="5"/>
    <position name="act4" joint="joint4" class="piper" kp="40" kv="5"/>
    <position name="act5" joint="joint5" class="piper" kp="10" kv="1.5"/>
    <position name="act6" joint="joint6" class="piper" kp="10" kv="1.5"/>
    <position name="grip" joint="joint7" class="finger" kp="40" kv="5"/>
  </actuator>
</mujoco>""".format(
        piper_assets=str(PIPER_DIR / "assets"),
        apple_stl=str(STL_DIR / apple_info["file"]),
        plate_stl=str(STL_DIR / plate_info["file"]),
        a_s0=apple_scale[0], a_s1=apple_scale[1], a_s2=apple_scale[2],
        p_s0=plate_scale[0], p_s1=plate_scale[1], p_s2=plate_scale[2],
        th=TABLE_H, tw2=TABLE_W/2, td2=TABLE_D/2,
        lh=TABLE_H/2, lx=TABLE_W/2-0.06, ly=TABLE_D/2-0.06, lz=TABLE_H/2,
        px=plate_pos[0], py=plate_pos[1], pz=plate_pos[2],
        ax=apple_pos[0], ay=apple_pos[1], az=apple_pos[2],
        ped_y=piper_base_y,
        ped_h2=TABLE_H/2 - 0.015,
        ped_top=TABLE_H - 0.030,
        piper_z=TABLE_H - 0.015,
        piper_z_plus=TABLE_H + 0.30,
    )
    return xml


class PiperController:
    """Control Piper arm via mocap body tracking."""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.renderer = mujoco.Renderer(model, width=1920, height=1440)
        self.frames = []
        self.keyframes = {}

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def move_to(self, target, steps=200, record_every=10):
        start = self.data.mocap_pos[0].copy()
        target = np.array(target, dtype=np.float64)
        for i in range(steps):
            t = (i + 1) / steps
            # Smooth cubic interpolation
            s = 3*t*t - 2*t*t*t
            self.data.mocap_pos[0] = start + s * (target - start)
            self.step()
            if record_every and (i % record_every == 0):
                self.frames.append(self._capture())

    def open_gripper(self, steps=80):
        self.data.ctrl[6] = 0.035
        for i in range(steps):
            self.step()
            if i % 10 == 0:
                self.frames.append(self._capture())

    def close_gripper(self, steps=120):
        self.data.ctrl[6] = 0.0
        for i in range(steps):
            self.step()
            if i % 10 == 0:
                self.frames.append(self._capture())

    def snapshot(self, name):
        img = self._capture()
        self.keyframes[name] = img
        self.frames.append(img)

    def _capture(self):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, -0.05, TABLE_H + 0.05]
        cam.distance = 1.6
        cam.elevation = -25
        cam.azimuth = 150
        self.renderer.update_scene(self.data, cam)
        return self.renderer.render().copy()

    def get_body_pos(self, name):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[bid].copy()

    def save(self, out_dir, task_name):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save keyframes
        for name, img in self.keyframes.items():
            Image.fromarray(img).save(str(out / f"{task_name}_{name}.png"))
            print(f"  saved {task_name}_{name}.png")

        # Save GIF
        if self.frames:
            pil_frames = [Image.fromarray(f).resize((960, 720)) for f in self.frames]
            pil_frames[0].save(
                str(out / f"{task_name}.gif"),
                save_all=True, append_images=pil_frames[1:],
                duration=50, loop=0)
            print(f"  saved {task_name}.gif ({len(pil_frames)} frames)")

    def close(self):
        self.renderer.close()


def run_task(task_name, apple_pos, plate_pos, target_pos):
    """Run a pick-and-place task."""
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"  Apple: {apple_pos}")
    print(f"  Plate: {plate_pos}")
    print(f"  Target: {target_pos}")
    print(f"{'='*60}")

    xml = build_scene_xml(apple_pos, plate_pos)

    # Save XML for debugging
    xml_path = OUTPUT_DIR / f"{task_name}_scene.xml"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(xml_path, 'w') as f:
        f.write(xml)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ctrl = PiperController(model, data)

    # Let scene settle
    print("  Settling physics...")
    ctrl.step(500)

    # Get real apple position after settling
    real_apple = ctrl.get_body_pos("apple")
    print(f"  Apple settled at: {real_apple}")

    # Waypoints
    above_apple = real_apple.copy()
    above_apple[2] += 0.12

    at_apple = real_apple.copy()
    at_apple[2] += 0.03  # fingertip offset

    above_target = np.array(target_pos, dtype=np.float64)
    above_target[2] += 0.12

    at_target = np.array(target_pos, dtype=np.float64)
    at_target[2] += 0.03

    # Execute sequence
    print("  Phase 1: Open gripper + move above apple")
    ctrl.open_gripper(steps=60)
    ctrl.move_to(above_apple, steps=250)
    ctrl.snapshot("01_above_apple")

    print("  Phase 2: Descend to apple")
    ctrl.move_to(at_apple, steps=150)
    ctrl.snapshot("02_at_apple")

    print("  Phase 3: Close gripper (grasp)")
    ctrl.close_gripper(steps=150)
    ctrl.snapshot("03_grasped")
    apple_after_grasp = ctrl.get_body_pos("apple")
    print(f"    Apple pos after grasp: {apple_after_grasp}")

    print("  Phase 4: Lift")
    ctrl.move_to(above_apple, steps=200)
    ctrl.snapshot("04_lifted")
    apple_lifted = ctrl.get_body_pos("apple")
    lifted = apple_lifted[2] > real_apple[2] + 0.03
    print(f"    Lifted: {lifted} (z: {real_apple[2]:.3f} → {apple_lifted[2]:.3f})")

    print("  Phase 5: Move to target")
    ctrl.move_to(above_target, steps=250)
    ctrl.snapshot("05_above_target")

    print("  Phase 6: Descend to target")
    ctrl.move_to(at_target, steps=150)
    ctrl.snapshot("06_at_target")

    print("  Phase 7: Release")
    ctrl.open_gripper(steps=100)
    ctrl.step(200)  # let apple settle
    ctrl.snapshot("07_released")

    print("  Phase 8: Retreat")
    ctrl.move_to(above_target, steps=150)
    ctrl.snapshot("08_done")

    # Final apple position
    final_apple = ctrl.get_body_pos("apple")
    dist_to_target = np.linalg.norm(final_apple[:2] - np.array(target_pos[:2]))
    print(f"  Final apple: {final_apple}")
    print(f"  Distance to target: {dist_to_target:.4f}m")
    success = dist_to_target < 0.10
    print(f"  SUCCESS: {success}")

    # Save outputs
    ctrl.save(OUTPUT_DIR, task_name)

    # Save trajectory info
    info = {
        "task": task_name,
        "apple_start": apple_pos.tolist() if hasattr(apple_pos, 'tolist') else list(apple_pos),
        "plate_pos": plate_pos.tolist() if hasattr(plate_pos, 'tolist') else list(plate_pos),
        "target": target_pos.tolist() if hasattr(target_pos, 'tolist') else list(target_pos),
        "apple_final": final_apple.tolist(),
        "dist_to_target": float(dist_to_target),
        "lifted": bool(lifted),
        "success": bool(success),
    }
    with open(OUTPUT_DIR / f"{task_name}_result.json", 'w') as f:
        json.dump(info, f, indent=2)

    ctrl.close()
    return success


def main():
    print("=" * 60)
    print("Piper 机械臂抓取规划 (Real Piper + Real STL)")
    print("=" * 60)

    surface_z = TABLE_H + 0.022  # tabletop surface (top of 2cm thick tabletop)

    # Objects near Y=0 (center of table), within Piper reach (~0.35-0.45m from base)
    # Piper base is at Y = -(TABLE_D/2 + 0.05) ≈ -0.35
    # So Y=0 is about 0.35m from base — within reach

    # Task 1: Pick apple from table, place on plate
    plate1 = [0.10, 0.0, surface_z]
    apple1 = [-0.10, 0.0, surface_z + 0.035]
    target1 = [0.10, 0.0, surface_z + 0.025 + 0.04]  # on plate

    s1 = run_task("place_on_plate", apple1, plate1, target1)

    # Task 2: Pick apple from plate area, place aside
    plate2 = [0.10, 0.0, surface_z]
    apple2 = [0.10, 0.0, surface_z + 0.025 + 0.05]  # above plate
    target2 = [-0.10, 0.0, surface_z + 0.035]

    s2 = run_task("pick_from_plate", apple2, plate2, target2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Task 1 (pick from plate): {'✅ SUCCESS' if s1 else '❌ FAIL'}")
    print(f"  Task 2 (place on plate):  {'✅ SUCCESS' if s2 else '❌ FAIL'}")

    summary = f"""Piper 机械臂抓取规划结果
{'='*50}
Task 1 (从盘子拿苹果): {'成功' if s1 else '失败'}
Task 2 (放苹果到盘子): {'成功' if s2 else '失败'}

Pipeline:
- 真实 Piper 6-DOF arm (mujoco_menagerie)
- 真实 STL 苹果/盘子 (Objaverse)
- Mocap body + weld constraint 末端跟踪
- 8 phase pick-and-place sequence
- GIF + 8 keyframe snapshots per task
"""
    with open(OUTPUT_DIR / "summary.txt", 'w') as f:
        f.write(summary)

    print(f"\nOutputs: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
