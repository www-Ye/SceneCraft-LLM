# Piper 机械臂抓取规划模块

## 概述
这个模块实现了基于 MuJoCo 的 Piper 6DOF 机械臂抓取苹果的运动规划。使用 mocap body + equality constraint 方法实现末端执行器跟踪控制。

## 文件结构
```
piper_grasp.py          # 主要实现文件
outputs/piper_grasp/    # 输出目录
├── pick_from_plate_trajectory.json    # 任务1轨迹数据
├── place_on_plate_trajectory.json     # 任务2轨迹数据
└── summary.txt                        # 实验结果总结
```

## 核心特性

### 1. 机械臂模型
- **Piper 6DOF 机械臂**: 简化的几何模型，6个旋转关节 + 1个gripper
- **Mocap 控制**: 使用 mocap body 和 equality constraint 实现末端跟踪
- **位置控制**: PD控制器，kp=80/40/10，根据关节重要性调节

### 2. 运动规划
采用简单的**关节空间插值**方法：
- Phase 1: 移动到目标上方 (预抓取位姿)
- Phase 2: 下降到抓取位置
- Phase 3: 闭合gripper抓取
- Phase 4: 提起物体
- Phase 5: 移动到放置位置上方
- Phase 6: 下降放置
- Phase 7: 打开gripper释放
- Phase 8: 抬起离开

### 3. 任务场景
- **桌面高度**: 0.75m
- **机械臂底座**: 位于桌子前方 (y=-0.48m)，放置在与桌面同高的底座上
- **物体**: 球形苹果 (简化STL为sphere) + 圆形盘子
- **两个任务**:
  1. **Pick from plate**: 从盘子上抓苹果放到旁边
  2. **Place on plate**: 从旁边抓苹果放到盘子上

## 使用方法

### 运行测试
```bash
cd /path/to/scene_factory
MUJOCO_GL=osmesa python3 piper_grasp.py
```

### 输出文件
- `summary.txt`: 实验结果总结
- `*_trajectory.json`: 详细轨迹数据，包含每个阶段的：
  - 苹果位置
  - gripper位置  
  - 关节角度
  - gripper控制信号

## 技术细节

### XML 场景构建
```xml
<equality>
  <!-- Gripper mirroring -->
  <joint joint1="joint8" joint2="joint7" polycoef="0 -1 0 0 0"/>
  
  <!-- End-effector tracking -->
  <weld body1="target" body2="link7" solref="0.01 1"/>
</equality>
```

### 控制循环
```python
def move_to(model, data, target_xyz, steps=200):
    start = data.mocap_pos[0].copy()
    for i in range(steps):
        t = (i + 1) / steps
        pos = start + t * (target_xyz - start)
        data.mocap_pos[0] = pos
        mujoco.mj_step(model, data)
```

### 抓取控制
```python
def close_gripper(model, data, steps=100):
    data.ctrl[6] = 0.0  # 关闭
    
def open_gripper(model, data, steps=50):
    data.ctrl[6] = 0.035  # 全开
```

## 限制与改进方向

### 当前限制
1. **渲染**: 由于OpenGL环境限制，无法生成可视化图像
2. **简化模型**: 使用基本几何体代替真实STL网格
3. **运动规划**: 简单插值，无碰撞检测和路径优化

### 可能改进
1. **视觉反馈**: 集成摄像头数据进行视觉伺服
2. **高级规划**: RRT/RRT* 路径规划算法
3. **力控**: 增加力/触觉传感器反馈
4. **学习控制**: 强化学习或模仿学习方法

## 验证结果
✅ **任务1**: 从盘子抓苹果 - 成功  
✅ **任务2**: 放苹果到盘子 - 成功  
✅ **Mocap跟踪**: 末端执行器准确跟踪目标位置  
✅ **Gripper控制**: 开合动作正确执行  
✅ **数据记录**: 完整轨迹数据保存  

## 依赖项
- Python 3.x
- MuJoCo 3.5.0+
- NumPy
- PIL (用于图像处理，虽然当前未使用)