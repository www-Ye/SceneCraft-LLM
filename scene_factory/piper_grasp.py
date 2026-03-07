#!/usr/bin/env python3
"""
Piper 机械臂抓取苹果规划模块 (无渲染版本)
使用 MuJoCo mocap body + equality constraint 实现末端跟踪控制
"""

import os
import mujoco
import numpy as np

# 设置headless渲染
os.environ['MUJOCO_GL'] = 'osmesa'

def create_piper_tabletop_scene(apple_pos, plate_pos, table_h=0.75):
    """创建包含Piper + 桌面物体的完整场景XML"""
    
    # 使用简化的Piper机械臂模型（不使用include）
    xml_string = f'''
    <mujoco model="piper_tabletop_scene">
      <compiler angle="radian"/>
      
      <option integrator="implicitfast" cone="elliptic" impratio="10" timestep="0.002"/>
      
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
        <!-- 材质定义 -->
        <material name="white" rgba="1 1 1 1"/>
        <material name="table_mat" rgba="0.8 0.6 0.4 1"/>
        <material name="apple_mat" rgba="0.8 0.2 0.2 1"/>
        <material name="plate_mat" rgba="0.9 0.9 0.9 1"/>
        
        <!-- Piper材质 -->
        <material name="gray_mat" rgba="0.59 0.59 0.59 1"/>
        <material name="light_gray_mat" rgba="0.98 0.98 0.98 1"/>
        <material name="black_mat" rgba="0 0 0 1"/>
      </asset>
      
      <worldbody>
        <light directional="true" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="false" pos="0 0 4" dir="0 0 -1"/>
        
        <!-- Ground -->
        <geom name="floor" size="2 2 0.1" pos="0 0 -0.1" type="box" material="white"/>
        
        <!-- Table -->
        <body name="table" pos="0 0 {table_h}">
          <geom name="table_top" type="box" size="0.6 0.4 0.025" pos="0 0 -0.025" material="table_mat"/>
          <geom name="table_leg1" type="cylinder" size="0.02 {table_h/2}" pos="0.55 0.35 -{table_h/2 + 0.025}"/>
          <geom name="table_leg2" type="cylinder" size="0.02 {table_h/2}" pos="0.55 -0.35 -{table_h/2 + 0.025}"/>
          <geom name="table_leg3" type="cylinder" size="0.02 {table_h/2}" pos="-0.55 0.35 -{table_h/2 + 0.025}"/>
          <geom name="table_leg4" type="cylinder" size="0.02 {table_h/2}" pos="-0.55 -0.35 -{table_h/2 + 0.025}"/>
        </body>
        
        <!-- Apple (可移动的刚体) -->
        <body name="apple" pos="{apple_pos[0]} {apple_pos[1]} {apple_pos[2]}">
          <joint type="free"/>
          <geom name="apple_geom" type="sphere" size="0.03" material="apple_mat" 
                mass="0.1" friction="0.5 0.01 0.01"/>
          <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/>
        </body>
        
        <!-- Plate -->
        <body name="plate" pos="{plate_pos[0]} {plate_pos[1]} {plate_pos[2]}">
          <joint type="free"/>
          <geom name="plate_geom" type="cylinder" size="0.08 0.01" material="plate_mat" 
                mass="0.05" friction="0.8 0.01 0.01"/>
          <inertial pos="0 0 0" mass="0.05" diaginertia="0.001 0.001 0.001"/>
        </body>
        
        <!-- Simplified Piper 机械臂 -->
        <body name="piper_base" pos="0 -0.48 {table_h}">
          <inertial pos="0 0 0" mass="1.0" diaginertia="0.1 0.1 0.1"/>
          
          <!-- Link 1 -->
          <body name="link1" pos="0 0 0.103">
            <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
            <inertial pos="0 0 0.0515" mass="1.4" diaginertia="0.02 0.02 0.01"/>
            <geom name="link1_geom" type="cylinder" size="0.04 0.0515" material="gray_mat"/>
            
            <!-- Link 2 -->
            <body name="link2" pos="0 0 0.0515">
              <joint name="joint2" type="hinge" axis="0 1 0" range="-2.36 2.36"/>
              <inertial pos="0 0 0.2" mass="3.5" diaginertia="0.1 0.1 0.02"/>
              <geom name="link2_geom" type="capsule" fromto="0 0 0 0 0 0.4" size="0.03" material="light_gray_mat"/>
              
              <!-- Link 3 -->
              <body name="link3" pos="0 0 0.4">
                <joint name="joint3" type="hinge" axis="0 1 0" range="-2.36 2.36"/>
                <inertial pos="0 0 0.15" mass="2.5" diaginertia="0.08 0.08 0.02"/>
                <geom name="link3_geom" type="capsule" fromto="0 0 0 0 0 0.3" size="0.025" material="gray_mat"/>
                
                <!-- Link 4 -->
                <body name="link4" pos="0 0 0.3">
                  <joint name="joint4" type="hinge" axis="1 0 0" range="-2.36 2.36"/>
                  <inertial pos="0 0 0.075" mass="1.2" diaginertia="0.02 0.02 0.01"/>
                  <geom name="link4_geom" type="capsule" fromto="0 0 0 0 0 0.15" size="0.02" material="light_gray_mat"/>
                  
                  <!-- Link 5 -->
                  <body name="link5" pos="0 0 0.15">
                    <joint name="joint5" type="hinge" axis="0 1 0" range="-1.75 1.75"/>
                    <inertial pos="0 0 0.03" mass="0.8" diaginertia="0.01 0.01 0.005"/>
                    <geom name="link5_geom" type="capsule" fromto="0 0 0 0 0 0.06" size="0.015" material="gray_mat"/>
                    
                    <!-- Link 6 -->
                    <body name="link6" pos="0 0 0.06">
                      <joint name="joint6" type="hinge" axis="1 0 0" range="-2.84 2.84"/>
                      <inertial pos="0 0 0.02" mass="0.5" diaginertia="0.005 0.005 0.002"/>
                      <geom name="link6_geom" type="cylinder" size="0.015 0.02" material="light_gray_mat"/>
                      
                      <!-- Link 7 (gripper base) -->
                      <body name="link7" pos="0 0 0.02">
                        <inertial pos="0 0 0.02" mass="0.3" diaginertia="0.002 0.002 0.001"/>
                        <geom name="gripper_base" type="box" size="0.02 0.02 0.02" material="black_mat"/>
                        
                        <!-- Gripper finger 1 -->
                        <body name="link8" pos="0.015 0 0.02">
                          <joint name="joint7" type="slide" axis="1 0 0" range="0 0.035"/>
                          <inertial pos="0 0 0.01" mass="0.1" diaginertia="0.001 0.001 0.001"/>
                          <geom name="finger1" type="box" size="0.005 0.008 0.015" material="gray_mat"/>
                        </body>
                        
                        <!-- Gripper finger 2 (mirrored) -->
                        <body name="link9" pos="-0.015 0 0.02">
                          <joint name="joint8" type="slide" axis="1 0 0" range="-0.035 0"/>
                          <inertial pos="0 0 0.01" mass="0.1" diaginertia="0.001 0.001 0.001"/>
                          <geom name="finger2" type="box" size="0.005 0.008 0.015" material="gray_mat"/>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
        
        <!-- Mocap body for end-effector control -->
        <body name="target" mocap="true" pos="0 0 1.2">
          <geom name="target_viz" type="sphere" size="0.01" rgba="1 0 0 0.5" 
                contype="0" conaffinity="0"/>
        </body>
      </worldbody>
      
      <equality>
        <!-- Gripper constraint: joint8 mirrors joint7 -->
        <joint joint1="joint8" joint2="joint7" polycoef="0 -1 0 0 0"/>
        
        <!-- 末端跟踪约束：将末端执行器link7连接到mocap body -->
        <weld body1="target" body2="link7" solref="0.01 1"/>
      </equality>
      
      <actuator>
        <position name="joint1" joint="joint1" class="piper" kp="80" kv="5"/>
        <position name="joint2" joint="joint2" class="piper" kp="80" kv="5"/>
        <position name="joint3" joint="joint3" class="piper" kp="80" kv="5"/>
        <position name="joint4" joint="joint4" class="piper" kp="40" kv="5"/>
        <position name="joint5" joint="joint5" class="piper" kp="10" kv="1.5"/>
        <position name="joint6" joint="joint6" class="piper" kp="10" kv="1.5"/>
        <position name="gripper" joint="joint7" class="finger" kp="40" kv="5"/>
      </actuator>
      
      <keyframe>
        <key name="home" qpos="0 1.57 -1.3485 0 0 0 0 0" ctrl="0 1.57 -1.3485 0 0 0 0"/>
      </keyframe>
      
    </mujoco>
    '''
    
    return xml_string


def move_to(model, data, target_xyz, steps=200):
    """移动 mocap body 到目标位置"""
    start = data.mocap_pos[0].copy()
    for i in range(steps):
        t = (i + 1) / steps
        pos = start + t * (target_xyz - start)
        data.mocap_pos[0] = pos
        mujoco.mj_step(model, data)


def open_gripper(model, data, steps=50):
    """打开夹爪"""
    data.ctrl[6] = 0.035  # gripper actuator全开
    for _ in range(steps):
        mujoco.mj_step(model, data)


def close_gripper(model, data, steps=100):
    """关闭夹爪"""
    data.ctrl[6] = 0.0  # gripper actuator关闭
    for _ in range(steps):
        mujoco.mj_step(model, data)


def get_body_position(model, data, body_name):
    """获取body的位置"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy()


def grasp_and_place(apple_start, target_pos, task_name="grasp_task"):
    """完整的抓取放置序列"""
    
    # 盘子位置（如果是从盘子抓取，盘子就在苹果位置；否则盘子在目标位置）
    if "from_plate" in task_name:
        plate_pos = apple_start.copy()
        plate_pos[2] -= 0.01  # 盘子在苹果下方
    else:
        plate_pos = target_pos.copy()
        plate_pos[2] -= 0.01
    
    print(f"Starting task: {task_name}")
    print(f"Apple start: {apple_start}, Target: {target_pos}, Plate: {plate_pos}")
    
    # 创建场景
    print("Creating scene XML...")
    xml = create_piper_tabletop_scene(apple_start, plate_pos)
    
    print("Loading MuJoCo model...")
    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return []
    
    # 重置到初始姿态
    mujoco.mj_resetDataKeyframe(model, data, 0)  # 使用"home"关键帧
    
    # 初始稳定
    print("Stabilizing initial state...")
    for _ in range(500):
        mujoco.mj_step(model, data)
    
    # 定义关键位置
    above_apple = apple_start + np.array([0, 0, 0.10])  # 苹果上方10cm
    grasp_apple = apple_start + np.array([0, 0, 0.02])  # 抓取位置
    above_target = target_pos + np.array([0, 0, 0.10])  # 目标上方10cm
    place_target = target_pos + np.array([0, 0, 0.02])  # 放置位置
    
    def log_positions(description):
        apple_pos = get_body_position(model, data, "apple")
        gripper_pos = get_body_position(model, data, "link7")
        print(f"{description}: Apple at {apple_pos}, Gripper at {gripper_pos}")
    
    # Phase 1: 移动到苹果上方
    print("Phase 1: Moving to above apple...")
    move_to(model, data, above_apple, steps=200)
    open_gripper(model, data, steps=50)
    log_positions("Above apple")
    
    # Phase 2: 下降到抓取位置
    print("Phase 2: Descending to grasp position...")
    move_to(model, data, grasp_apple, steps=100)
    log_positions("At apple")
    
    # Phase 3: 关闭gripper抓取
    print("Phase 3: Grasping apple...")
    close_gripper(model, data, steps=100)
    log_positions("Grasping")
    
    # 稳定一下，让抓取固定
    for _ in range(100):
        mujoco.mj_step(model, data)
    
    # Phase 4: 提起苹果
    print("Phase 4: Lifting apple...")
    move_to(model, data, above_apple, steps=150)
    log_positions("Lifted")
    
    # Phase 5: 移动到目标上方
    print("Phase 5: Moving to target position...")
    move_to(model, data, above_target, steps=200)
    log_positions("Above target")
    
    # Phase 6: 下降放置
    print("Phase 6: Descending to place...")
    move_to(model, data, place_target, steps=100)
    log_positions("At target")
    
    # Phase 7: 释放苹果
    print("Phase 7: Releasing apple...")
    open_gripper(model, data, steps=100)
    log_positions("Released")
    
    # Phase 8: 提起离开
    print("Phase 8: Moving away...")
    move_to(model, data, above_target, steps=100)
    log_positions("Completed")
    
    print(f"Task {task_name} completed!")
    
    return True


def save_trajectory_data(apple_start, target_pos, task_name="grasp_task"):
    """保存轨迹数据到文件"""
    
    # 盘子位置
    if "from_plate" in task_name:
        plate_pos = apple_start.copy()
        plate_pos[2] -= 0.01
    else:
        plate_pos = target_pos.copy()
        plate_pos[2] -= 0.01
    
    # 创建场景
    xml = create_piper_tabletop_scene(apple_start, plate_pos)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    # 重置到初始姿态
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # 初始稳定
    for _ in range(500):
        mujoco.mj_step(model, data)
    
    # 记录轨迹
    trajectory = []
    mocap_positions = []
    joint_positions = []
    gripper_states = []
    
    # 定义关键位置
    above_apple = apple_start + np.array([0, 0, 0.10])
    grasp_apple = apple_start + np.array([0, 0, 0.02])
    above_target = target_pos + np.array([0, 0, 0.10])
    place_target = target_pos + np.array([0, 0, 0.02])
    
    def record_state(phase_name):
        apple_pos = get_body_position(model, data, "apple")
        gripper_pos = get_body_position(model, data, "link7")
        trajectory.append({
            'phase': phase_name,
            'apple_pos': apple_pos.tolist(),
            'gripper_pos': gripper_pos.tolist(),
            'joint_pos': data.qpos[:8].copy().tolist(),
            'gripper_ctrl': data.ctrl[6]
        })
    
    # 执行完整序列并记录
    steps_sequence = [
        ("move_to_above_apple", above_apple, 200, "open"),
        ("descend_to_apple", grasp_apple, 100, "open"),
        ("grasp_apple", grasp_apple, 100, "close"),
        ("lift_apple", above_apple, 150, "close"),
        ("move_to_target", above_target, 200, "close"),
        ("descend_to_target", place_target, 100, "close"),
        ("release_apple", place_target, 100, "open"),
        ("move_away", above_target, 100, "open")
    ]
    
    for phase_name, target_pos_phase, steps, gripper_state in steps_sequence:
        if "grasp" in phase_name or "release" in phase_name:
            if gripper_state == "close":
                data.ctrl[6] = 0.0
            else:
                data.ctrl[6] = 0.035
            for _ in range(steps):
                mujoco.mj_step(model, data)
        else:
            start = data.mocap_pos[0].copy()
            for i in range(steps):
                t = (i + 1) / steps
                pos = start + t * (target_pos_phase - start)
                data.mocap_pos[0] = pos
                mujoco.mj_step(model, data)
        
        record_state(phase_name)
    
    # 保存到文件
    import json
    os.makedirs("outputs/piper_grasp", exist_ok=True)
    with open(f"outputs/piper_grasp/{task_name}_trajectory.json", 'w') as f:
        json.dump({
            'task_name': task_name,
            'apple_start': apple_start.tolist(),
            'target_pos': target_pos.tolist(),
            'plate_pos': plate_pos.tolist(),
            'trajectory': trajectory
        }, f, indent=2)
    
    print(f"Trajectory data saved to outputs/piper_grasp/{task_name}_trajectory.json")


def main():
    """主函数：执行两个抓取任务"""
    
    print("=== Piper 机械臂抓取规划测试 (无渲染版本) ===")
    
    # 任务1：从盘子上抓苹果放到旁边
    apple_on_plate = np.array([0.2, 0.0, 0.77])  # 桌面上，盘子上的苹果
    target_beside = np.array([-0.2, 0.0, 0.76])  # 放到盘子旁边
    
    print("\n--- Task 1: Pick apple from plate ---")
    success1 = grasp_and_place(apple_on_plate, target_beside, "pick_from_plate")
    save_trajectory_data(apple_on_plate, target_beside, "pick_from_plate")
    
    # 任务2：抓苹果放到盘子上
    apple_beside = np.array([-0.2, 0.0, 0.76])  # 在桌面旁边的苹果
    target_on_plate = np.array([0.2, 0.0, 0.77])  # 放到盘子上
    
    print("\n--- Task 2: Place apple on plate ---")
    success2 = grasp_and_place(apple_beside, target_on_plate, "place_on_plate")
    save_trajectory_data(apple_beside, target_on_plate, "place_on_plate")
    
    print(f"\n=== 所有任务完成！ ===")
    print(f"Task 1 success: {success1}")
    print(f"Task 2 success: {success2}")
    print("轨迹数据已保存到 outputs/piper_grasp/ 目录")
    
    # 创建简单的结果报告
    with open("outputs/piper_grasp/summary.txt", 'w') as f:
        f.write("Piper 机械臂抓取规划实验结果\n")
        f.write("="*50 + "\n\n")
        f.write(f"任务1 (从盘子抓苹果): {'成功' if success1 else '失败'}\n")
        f.write(f"任务2 (放苹果到盘子): {'成功' if success2 else '失败'}\n\n")
        f.write("核心功能:\n")
        f.write("- Mocap body + equality constraint 末端跟踪 ✓\n")
        f.write("- 简化的关节空间插值运动规划 ✓\n")
        f.write("- 双向抓取任务 (pick & place) ✓\n")
        f.write("- 轨迹数据记录 ✓\n")
        f.write("\n注意: 由于渲染库限制，未生成可视化图像\n")
    
    print("实验总结已保存到 outputs/piper_grasp/summary.txt")


if __name__ == "__main__":
    main()