#!/usr/bin/env python3
"""
简单测试MuJoCo和Piper模型
"""

import os
import mujoco
import numpy as np

# 设置headless渲染
os.environ['MUJOCO_GL'] = 'osmesa'

def test_simple_scene():
    """测试简单场景"""
    
    xml_string = '''
    <mujoco model="simple_test">
      <compiler angle="radian"/>
      
      <option integrator="implicitfast" timestep="0.002"/>
      
      <asset>
        <material name="white" rgba="1 1 1 1"/>
        <material name="red" rgba="1 0 0 1"/>
      </asset>
      
      <worldbody>
        <light directional="true" pos="0 0 4" dir="0 0 -1"/>
        
        <!-- Ground -->
        <geom name="floor" size="2 2 0.1" pos="0 0 -0.1" type="box" material="white"/>
        
        <!-- Simple box -->
        <body name="box" pos="0 0 1">
          <joint type="free"/>
          <geom name="box_geom" type="box" size="0.1 0.1 0.1" material="red"/>
          <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
        </body>
        
        <!-- Mocap target -->
        <body name="target" mocap="true" pos="0 0 1.5">
          <geom name="target_viz" type="sphere" size="0.05" rgba="0 1 0 0.5" 
                contype="0" conaffinity="0"/>
        </body>
      </worldbody>
      
    </mujoco>
    '''
    
    print("Creating simple scene...")
    try:
        model = mujoco.MjModel.from_xml_string(xml_string)
        data = mujoco.MjData(model)
        print("Simple model loaded successfully!")
        
        # 简单步进
        for i in range(100):
            mujoco.mj_step(model, data)
            
        print("Simulation steps completed!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_simple_scene()