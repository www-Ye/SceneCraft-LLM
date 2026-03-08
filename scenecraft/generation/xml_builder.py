#!/usr/bin/env python3
"""
MuJoCo XML Builder for SceneCraft.

Builds MuJoCo XML scenes with separate visual and collision meshes.
"""
import math
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MuJoCoXMLBuilder:
    """Builds MuJoCo XML scenes with proper visual/collision separation."""
    
    def __init__(self, asset_manager):
        self.asset_manager = asset_manager
    
    def build_xml(self, objects, table_config, include_lighting=True):
        """Build complete MuJoCo XML scene."""
        meshes = []
        materials = []
        bodies = []
        
        tw, td, th = table_config["w"], table_config["d"], table_config["h"]
        
        # Process each object
        for idx, obj in enumerate(objects):
            cat = obj["cat"]
            dims = obj["dims"]
            pos = obj.get("final_pos", obj["pos"] + [th + 0.02])  # Use computed position
            rot = obj.get("rot", 0)
            color = obj.get("color", "0.8 0.8 0.8 1")
            shin = obj.get("shin", 0.3)
            spec = obj.get("spec", 0.2)
            
            # Get mesh and material information
            scale_info, asset_info = self.asset_manager.get_mesh_scale(cat, dims)
            if scale_info is None:
                logger.warning(f"Skipping {cat} - no mesh available")
                continue
            
            material_props = self.asset_manager.get_material_properties(cat)
            
            # Use custom color if provided, otherwise material default
            if color != "0.8 0.8 0.8 1":  # Custom color specified
                rgba = color
            else:
                rgba = material_props["rgba"]
            
            # Mesh names
            visual_mesh_name = f"visual_{idx}_{cat}"
            collision_mesh_name = f"collision_{idx}_{cat}"
            material_name = f"mat_{idx}_{cat}"
            
            # Rotation quaternion
            rot_rad = math.radians(rot)
            qw = math.cos(rot_rad / 2)
            qz = math.sin(rot_rad / 2)
            
            # Visual mesh (high detail)
            visual_scale = scale_info["visual_scale"]
            visual_mesh_file = Path(scale_info["visual_mesh"]).name
            meshes.append(
                f'    <mesh name="{visual_mesh_name}" file="{visual_mesh_file}" '
                f'scale="{visual_scale[0]:.5f} {visual_scale[1]:.5f} {visual_scale[2]:.5f}"/>'
            )
            
            # Collision mesh (simplified)
            collision_scale = scale_info["collision_scale"] 
            collision_mesh_file = Path(scale_info["collision_mesh"]).name
            meshes.append(
                f'    <mesh name="{collision_mesh_name}" file="{collision_mesh_file}" '
                f'scale="{collision_scale[0]:.5f} {collision_scale[1]:.5f} {collision_scale[2]:.5f}"/>'
            )
            
            # Material
            materials.append(
                f'    <material name="{material_name}" rgba="{rgba}" '
                f'shininess="{shin}" specular="{spec}"/>'
            )
            
            # Body with separate visual and collision geoms
            mass = scale_info["mass"]
            friction = material_props["friction"]
            
            bodies.append(f'''
    <body name="obj_{idx}_{cat}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" quat="{qw:.5f} 0 0 {qz:.5f}">
      <freejoint name="fj_{idx}"/>
      <!-- Visual geom (high detail, no collision) -->
      <geom name="visual_{idx}" type="mesh" mesh="{visual_mesh_name}" material="{material_name}"
            contype="0" conaffinity="0"/>
      <!-- Collision geom (simplified, invisible) -->
      <geom name="collision_{idx}" type="mesh" mesh="{collision_mesh_name}" 
            mass="{mass:.4f}" friction="{friction}" 
            solimp="0.95 0.95 0.01" solref="0.02 1" rgba="0 0 0 0"/>
    </body>''')
        
        # Combine all sections
        mesh_block = "\n".join(meshes)
        material_block = "\n".join(materials)
        body_block = "\n".join(bodies)
        
        # Lighting
        lighting_xml = ""
        if include_lighting:
            lighting_xml = """
    <light name="key" pos="0.4 -0.6 1.8" dir="-0.2 0.3 -0.8" diffuse="0.80 0.76 0.70"
           specular="0.5 0.5 0.5" castshadow="true" cutoff="60"/>
    <light name="fill" pos="-0.5 0.2 1.5" dir="0.3 -0.1 -0.9" diffuse="0.30 0.32 0.36"/>
    <light name="rim" pos="0 0.6 1.6" dir="0 -0.3 -0.9" diffuse="0.20 0.20 0.22"/>
    <light name="bounce" pos="0 0 0.1" dir="0 0 1" diffuse="0.06 0.06 0.06" specular="0 0 0"/>"""
        
        # Table legs positions
        lx = tw / 2 - 0.06
        ly = td / 2 - 0.06
        lh = th / 2
        
        # STL directory for mesh files
        stl_dir = self.asset_manager.stl_dir
        
        xml = f"""<mujoco model="scenecraft_tabletop">
  <compiler angle="degree" meshdir="{stl_dir}"/>
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

{material_block}
  </asset>

  <worldbody>{lighting_xml}

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
        
        return xml