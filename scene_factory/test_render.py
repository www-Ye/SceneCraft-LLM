#!/usr/bin/env python3
"""
测试MuJoCo渲染功能
"""

import os
import mujoco
import numpy as np
from PIL import Image

# 设置headless渲染
os.environ['MUJOCO_GL'] = 'osmesa'

def test_rendering():
    """测试渲染功能"""
    
    xml_string = '''
    <mujoco model="render_test">
      <compiler angle="radian"/>
      
      <option integrator="implicitfast" timestep="0.002"/>
      
      <visual>
        <global azimuth="120" elevation="-20" offwidth="640" offheight="480"/>
      </visual>
      
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
          <geom name="box_geom" type="box" size="0.1 0.1 0.1" material="red"/>
        </body>
      </worldbody>
      
    </mujoco>
    '''
    
    print("Creating render test scene...")
    try:
        model = mujoco.MjModel.from_xml_string(xml_string)
        data = mujoco.MjData(model)
        print("Model loaded successfully!")
        
        # 测试渲染
        print("Testing renderer...")
        renderer = mujoco.Renderer(model, height=480, width=640)
        renderer.update_scene(data)
        pixels = renderer.render()
        image = Image.fromarray(pixels)
        
        # 保存测试图像
        os.makedirs("outputs/test", exist_ok=True)
        image.save("outputs/test/render_test.png")
        print("Render test successful! Image saved to outputs/test/render_test.png")
        
        renderer.close()
        return True
        
    except Exception as e:
        print(f"Render error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_rendering()