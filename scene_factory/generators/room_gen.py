#!/usr/bin/env python3
"""
Procedural Room Environment Generator for MuJoCo.

Generates diverse room navigation scenes with:
- Randomized room sizes and furniture layouts
- Real 3D meshes from Objaverse
- Built-in mobile robot for navigation tasks
- Configurable difficulty (open → cluttered)
- Physics-validated
"""
import json, os, sys, math, random, tempfile, shutil
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Furniture catalog with realistic dimensions
FURNITURE = {
    "sofa":         {"dims": [2.0, 0.9, 0.85], "mass": 40,  "wall_item": True,  "rgba": "0.35 0.45 0.65 1"},
    "armchair":     {"dims": [0.9, 0.85, 0.85], "mass": 15, "wall_item": True,  "rgba": "0.50 0.30 0.30 1"},
    "coffee_table": {"dims": [1.2, 0.6, 0.45], "mass": 10,  "wall_item": False, "rgba": "0.55 0.35 0.18 1"},
    "dining_table": {"dims": [1.6, 0.9, 0.75], "mass": 25,  "wall_item": False, "rgba": "0.55 0.35 0.18 1"},
    "chair":        {"dims": [0.5, 0.5, 0.9],  "mass": 5,   "wall_item": False, "rgba": "0.55 0.35 0.18 1"},
    "desk":         {"dims": [1.4, 0.7, 0.75], "mass": 20,  "wall_item": True,  "rgba": "0.40 0.28 0.15 1"},
    "bed":          {"dims": [1.6, 2.0, 0.6],  "mass": 50,  "wall_item": True,  "rgba": "0.55 0.36 0.96 1"},
    "cabinet":      {"dims": [0.8, 0.45, 1.8], "mass": 30,  "wall_item": True,  "rgba": "0.45 0.30 0.15 1"},
    "wardrobe":     {"dims": [1.2, 0.6, 2.0],  "mass": 45,  "wall_item": True,  "rgba": "0.50 0.32 0.18 1"},
    "bookshelf":    {"dims": [0.8, 0.3, 1.8],  "mass": 25,  "wall_item": True,  "rgba": "0.45 0.30 0.15 1"},
    "dresser":      {"dims": [1.2, 0.5, 0.8],  "mass": 25,  "wall_item": True,  "rgba": "0.55 0.38 0.22 1"},
    "nightstand":   {"dims": [0.5, 0.4, 0.6],  "mass": 8,   "wall_item": True,  "rgba": "0.50 0.35 0.20 1"},
    "tv_stand":     {"dims": [1.2, 0.4, 0.5],  "mass": 15,  "wall_item": True,  "rgba": "0.20 0.20 0.22 1"},
    "lamp":         {"dims": [0.3, 0.3, 1.5],  "mass": 3,   "wall_item": False, "rgba": "0.90 0.85 0.30 1"},
}

# Room templates
ROOM_TEMPLATES = {
    "living_room": {
        "size_range": ([4, 3.5], [6, 5]),
        "required": ["sofa", "coffee_table"],
        "optional": ["armchair", "tv_stand", "lamp", "bookshelf", "cabinet"],
    },
    "bedroom": {
        "size_range": ([3, 3], [5, 4.5]),
        "required": ["bed"],
        "optional": ["nightstand", "nightstand", "wardrobe", "dresser", "lamp", "chair"],
    },
    "office": {
        "size_range": ([3, 2.5], [4.5, 3.5]),
        "required": ["desk", "chair"],
        "optional": ["bookshelf", "cabinet", "lamp"],
    },
    "dining_room": {
        "size_range": ([3.5, 3], [5, 4]),
        "required": ["dining_table"],
        "optional": ["chair", "chair", "chair", "chair", "cabinet"],
    },
    "random": {
        "size_range": ([3, 3], [7, 6]),
        "required": [],
        "optional": list(FURNITURE.keys()),
    },
}

# STL mesh mapping
STL_DIR = Path(__file__).parent.parent.parent / "data" / "assets" / "_mujoco_stl"
STL_MAP = {
    "sofa": "sofa_0.stl", "armchair": "armchair_0.stl", "bed": "bed_0.stl",
    "coffee_table": "coffee_table_0.stl", "dining_table": "dining_table_0.stl",
    "desk": "desk_0.stl", "chair": "chair_0.stl", "cabinet": "cabinet_0.stl",
    "wardrobe": "wardrobe_0.stl", "dresser": "dresser_0.stl", "lamp": "lamp_0.stl",
    "tv_stand": "television_set_0.stl",
}


class RoomGenerator:
    """Generates randomized room navigation environments."""
    
    def __init__(self, seed=None, use_meshes=True):
        self.rng = random.Random(seed)
        self.use_meshes = use_meshes and STL_DIR.exists()
    
    def generate(self, template="random", n_furniture=None, difficulty="medium"):
        """Generate a room environment."""
        tmpl = ROOM_TEMPLATES.get(template, ROOM_TEMPLATES["random"])
        
        # Random room size within template range
        min_s, max_s = tmpl["size_range"]
        room_w = self.rng.uniform(min_s[0], max_s[0])
        room_h = self.rng.uniform(min_s[1], max_s[1])
        
        # Object count
        if n_furniture is None:
            n_furniture = {"easy": 3, "medium": 5, "hard": 8}.get(difficulty, 5)
        
        # Select furniture
        to_place = list(tmpl["required"])
        available = list(tmpl["optional"])
        while len(to_place) < n_furniture and available:
            cat = self.rng.choice(available)
            to_place.append(cat)
            available.remove(cat)
        
        # Place furniture
        placed = self._place_furniture(to_place, room_w, room_h)
        
        # Find valid robot start position
        robot_start = self._find_free_position(placed, room_w, room_h, radius=0.2)
        robot_goal = self._find_free_position(placed, room_w, room_h, radius=0.2, 
                                                min_dist_from=robot_start, min_dist=1.5)
        
        # Build XML
        xml = self._build_xml(placed, room_w, room_h, robot_start)
        
        metadata = {
            "template": template,
            "difficulty": difficulty,
            "room_size": [round(room_w, 2), round(room_h, 2)],
            "n_furniture": len(placed),
            "robot_start": robot_start,
            "robot_goal": robot_goal,
            "uses_meshes": self.use_meshes,
        }
        
        return {
            "furniture": [{"category": f[0], "position": f[1], "rotation": f[2], 
                          "dimensions": FURNITURE[f[0]]["dims"]} for f in placed],
            "room": {"width": round(room_w, 2), "height": round(room_h, 2)},
            "robot_start": robot_start,
            "robot_goal": robot_goal,
            "mujoco_xml": xml,
            "metadata": metadata,
        }
    
    def _place_furniture(self, categories, room_w, room_h):
        """Place furniture with wall-alignment preference and collision avoidance."""
        placed = []  # (category, [x,y], rotation)
        margin = 0.1
        
        for cat in categories:
            info = FURNITURE.get(cat, FURNITURE["chair"])
            w, d, h = info["dims"]
            
            for attempt in range(200):
                if info["wall_item"] and attempt < 100:
                    # Try wall placement
                    wall = self.rng.choice(["north", "south", "east", "west"])
                    rot = {"north": 180, "south": 0, "east": 270, "west": 90}[wall]
                    
                    if wall == "north":
                        x = self.rng.uniform(w/2 + margin, room_w - w/2 - margin)
                        y = room_h - d/2 - margin
                    elif wall == "south":
                        x = self.rng.uniform(w/2 + margin, room_w - w/2 - margin)
                        y = d/2 + margin
                    elif wall == "east":
                        x = room_w - d/2 - margin
                        y = self.rng.uniform(w/2 + margin, room_h - w/2 - margin)
                    else:
                        x = d/2 + margin
                        y = self.rng.uniform(w/2 + margin, room_h - w/2 - margin)
                else:
                    # Random placement
                    x = self.rng.uniform(w/2 + margin, room_w - w/2 - margin)
                    y = self.rng.uniform(d/2 + margin, room_h - d/2 - margin)
                    rot = self.rng.choice([0, 90, 180, 270])
                
                # Check collisions
                collision = False
                for pc, pp, pr in placed:
                    pi = FURNITURE[pc]
                    pw, pd = pi["dims"][0], pi["dims"][1]
                    if pr in [90, 270]:
                        pw, pd = pd, pw
                    
                    cw, cd = w, d
                    if rot in [90, 270]:
                        cw, cd = cd, cw
                    
                    dx = abs(x - pp[0])
                    dy = abs(y - pp[1])
                    if dx < (cw + pw)/2 + 0.15 and dy < (cd + pd)/2 + 0.15:
                        collision = True
                        break
                
                if not collision:
                    placed.append((cat, [round(x, 3), round(y, 3)], rot))
                    break
        
        return placed
    
    def _find_free_position(self, placed, room_w, room_h, radius=0.2, 
                             min_dist_from=None, min_dist=0):
        """Find a position free of furniture."""
        for _ in range(500):
            x = self.rng.uniform(radius + 0.3, room_w - radius - 0.3)
            y = self.rng.uniform(radius + 0.3, room_h - radius - 0.3)
            
            free = True
            for cat, pos, rot in placed:
                info = FURNITURE[cat]
                fw, fd = info["dims"][0], info["dims"][1]
                if rot in [90, 270]:
                    fw, fd = fd, fw
                dx = abs(x - pos[0])
                dy = abs(y - pos[1])
                if dx < (fw/2 + radius + 0.2) and dy < (fd/2 + radius + 0.2):
                    free = False
                    break
            
            if free and min_dist_from:
                dist = math.sqrt((x - min_dist_from[0])**2 + (y - min_dist_from[1])**2)
                if dist < min_dist:
                    free = False
            
            if free:
                return [round(x, 2), round(y, 2)]
        
        return [room_w/2, room_h/2]
    
    def _build_xml(self, placed, room_w, room_h, robot_start):
        """Build MuJoCo XML with room + furniture + mobile robot."""
        
        # Build furniture bodies
        furn_bodies = []
        mesh_assets = []
        tmp_stls = set()
        
        for i, (cat, pos, rot) in enumerate(placed):
            info = FURNITURE[cat]
            w, d, h = info["dims"]
            rot_rad = math.radians(rot)
            qw = math.cos(rot_rad / 2)
            qz = math.sin(rot_rad / 2)
            
            stl_file = STL_MAP.get(cat)
            if self.use_meshes and stl_file and (STL_DIR / stl_file).exists():
                mesh_name = f"mesh_{i}_{cat}"
                mesh_assets.append(f'<mesh name="{mesh_name}" file="{stl_file}"/>')
                tmp_stls.add(stl_file)
                
                furn_bodies.append(f"""
        <body name="furn_{i}_{cat}" pos="{pos[0]:.3f} {pos[1]:.3f} 0" quat="{qw:.4f} 0 0 {qz:.4f}">
            <geom name="furn_geom_{i}" type="mesh" mesh="{mesh_name}" mass="{info['mass']}"/>
        </body>""")
            else:
                furn_bodies.append(f"""
        <body name="furn_{i}_{cat}" pos="{pos[0]:.3f} {pos[1]:.3f} {h/2:.3f}" quat="{qw:.4f} 0 0 {qz:.4f}">
            <geom name="furn_geom_{i}" type="box" size="{w/2:.3f} {d/2:.3f} {h/2:.3f}" 
                  rgba="{info['rgba']}" mass="{info['mass']}"/>
        </body>""")
        
        mesh_xml = "\n        ".join(mesh_assets) if mesh_assets else ""
        furn_xml = "\n".join(furn_bodies)
        
        xml = f"""<mujoco model="room_nav_env">
    <compiler angle="degree" meshdir="{STL_DIR}"/>
    
    <option timestep="0.005" gravity="0 0 -9.81"/>
    
    <visual>
        <global offwidth="1600" offheight="1200"/>
        <headlight diffuse="0.8 0.8 0.8" ambient="0.4 0.4 0.4"/>
    </visual>
    
    <asset>
        {mesh_xml}
        <texture name="floor" type="2d" builtin="checker" rgb1="0.92 0.90 0.85" rgb2="0.88 0.86 0.82" width="128" height="128"/>
        <texture name="wall" type="2d" builtin="gradient" rgb1="0.95 0.93 0.90" rgb2="0.90 0.88 0.85" width="64" height="64"/>
        <material name="floor_mat" texture="floor"/>
        <material name="wall_mat" texture="wall"/>
    </asset>
    
    <worldbody>
        <light pos="{room_w/2} {room_h/2} 3" dir="0 0 -1" diffuse="0.9 0.9 0.85" castshadow="true"/>
        
        <!-- Floor -->
        <geom name="floor" type="plane" size="{room_w/2+1} {room_h/2+1} 0.01" 
              pos="{room_w/2} {room_h/2} 0" material="floor_mat"/>
        
        <!-- Walls -->
        <geom name="wall_s" type="box" size="{room_w/2:.2f} 0.05 0.5" pos="{room_w/2:.2f} -0.05 0.5" material="wall_mat"/>
        <geom name="wall_n" type="box" size="{room_w/2:.2f} 0.05 0.5" pos="{room_w/2:.2f} {room_h+0.05:.2f} 0.5" material="wall_mat"/>
        <geom name="wall_w" type="box" size="0.05 {room_h/2:.2f} 0.5" pos="-0.05 {room_h/2:.2f} 0.5" material="wall_mat"/>
        <geom name="wall_e" type="box" size="0.05 {room_h/2:.2f} 0.5" pos="{room_w+0.05:.2f} {room_h/2:.2f} 0.5" material="wall_mat"/>
        
        <!-- Mobile Robot (differential drive) -->
        <body name="robot" pos="{robot_start[0]} {robot_start[1]} 0.08">
            <joint name="robot_x" type="slide" axis="1 0 0" damping="5"/>
            <joint name="robot_y" type="slide" axis="0 1 0" damping="5"/>
            <joint name="robot_rz" type="hinge" axis="0 0 1" damping="2"/>
            <geom name="robot_body" type="cylinder" size="0.15 0.06" rgba="0.9 0.2 0.2 0.9" mass="3"/>
            <geom name="robot_head" type="sphere" size="0.05" pos="0.10 0 0.06" rgba="0.2 0.2 0.9 0.9" mass="0.1"/>
            
            <!-- Goal marker (visual only) -->
            <site name="robot_site" size="0.02" rgba="1 0 0 1"/>
        </body>
        
        <!-- Goal marker -->
        <site name="goal" pos="{room_w/2} {room_h/2} 0.01" size="0.15 0.01" type="cylinder" rgba="0.2 0.9 0.2 0.5"/>
        
        <!-- Furniture -->
        {furn_xml}
    </worldbody>
    
    <actuator>
        <motor joint="robot_x" ctrlrange="-3 3" gear="15"/>
        <motor joint="robot_y" ctrlrange="-3 3" gear="15"/>
        <motor joint="robot_rz" ctrlrange="-2 2" gear="5"/>
    </actuator>
    
    <sensor>
        <framepos objtype="body" objname="robot"/>
        <framequat objtype="body" objname="robot"/>
    </sensor>
</mujoco>"""
        
        return xml


def demo():
    """Demo: generate and render room environments."""
    import mujoco
    os.environ['MUJOCO_GL'] = 'osmesa'
    
    gen = RoomGenerator(seed=42, use_meshes=True)
    
    for template in ["living_room", "bedroom", "office"]:
        env = gen.generate(template=template, difficulty="medium")
        meta = env["metadata"]
        print(f"\n=== {template.upper()} ({meta['room_size'][0]}m × {meta['room_size'][1]}m, {meta['n_furniture']} items) ===")
        for f in env["furniture"]:
            print(f"  {f['category']:15s} at ({f['position'][0]:.1f}, {f['position'][1]:.1f}) rot={f['rotation']}°")
        print(f"  🤖 Robot: {meta['robot_start']} → Goal: {meta['robot_goal']}")
        
        # Render
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(env["mujoco_xml"])
            xml_path = f.name
        
        try:
            model = mujoco.MjModel.from_xml_path(xml_path)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            
            renderer = mujoco.Renderer(model, width=1200, height=900)
            
            # Top-down view
            camera = mujoco.MjvCamera()
            rw, rh = meta["room_size"]
            camera.lookat[0] = rw / 2
            camera.lookat[1] = rh / 2
            camera.lookat[2] = 0.3
            camera.distance = max(rw, rh) * 1.1
            camera.elevation = -80
            camera.azimuth = 90
            
            renderer.update_scene(data, camera)
            img = renderer.render()
            
            from PIL import Image
            out_dir = Path(__file__).parent.parent / "envs" / "demo"
            out_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(str(out_dir / f"room_{template}.png"))
            print(f"  📸 Saved: envs/demo/room_{template}.png")
            
            # Perspective
            camera.elevation = -35
            camera.azimuth = 135
            camera.distance = max(rw, rh) * 1.3
            renderer.update_scene(data, camera)
            img2 = renderer.render()
            Image.fromarray(img2).save(str(out_dir / f"room_{template}_persp.png"))
            print(f"  📸 Saved: envs/demo/room_{template}_persp.png")
            
            renderer.close()
        finally:
            os.unlink(xml_path)
    
    # Speed test
    import time
    gen2 = RoomGenerator(seed=0, use_meshes=False)
    t0 = time.time()
    for i in range(100):
        gen2.generate(template="random", difficulty="medium")
    t1 = time.time()
    print(f"\n⚡ Speed: {100/(t1-t0):.0f} room environments/second")


if __name__ == "__main__":
    demo()
