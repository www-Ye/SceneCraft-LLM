# SceneCraft-LLM: MuJoCo 桌面场景生成经验总结

> 本文档总结了在使用 MuJoCo + YCB 物体集构建高质量 3D 桌面仿真场景过程中遇到的关键问题及其解决方案。

---

## 目录

1. [碰撞体设计原则](#1-碰撞体设计原则)
2. [物体穿模问题](#2-物体穿模问题)
3. [物体悬浮问题](#3-物体悬浮问题)
4. [堆叠放置](#4-堆叠放置)
5. [物理仿真稳定性](#5-物理仿真稳定性)
6. [碰撞感知布局](#6-碰撞感知布局)
7. [程序化物体构建](#7-程序化物体构建)
8. [纹理渲染](#8-纹理渲染)
9. [物理属性管理](#9-物理属性管理)
10. [自动化质量检测](#10-自动化质量检测)

---

## 1. 碰撞体设计原则

### 核心问题
MuJoCo 的 `type="mesh"` 碰撞使用**凸包（convex hull）**，对凹面物体（盘子、碗、杯子）完全不适用——碗的凸包是一个实心半球，无法在碗内放置任何物体。

### 解决方案：Visual/Collision 分离

```xml
<!-- 视觉 mesh：精确渲染，不参与碰撞 -->
<geom type="mesh" mesh="plate_visual" contype="0" conaffinity="0" mass="0"/>

<!-- 碰撞体：简单几何原语，参与物理 -->
<geom type="cylinder" size="0.13 0.004" rgba="0 0 0 0" mass="0.279"/>
```

**关键原则：**
- 视觉 geom 设置 `contype="0" conaffinity="0"` 禁用碰撞
- 碰撞 geom 设置 `rgba="0 0 0 0"` 透明不渲染
- 碰撞 geom 使用原语（box/cylinder/sphere/capsule/ellipsoid）

### 不同物体的最佳碰撞体类型

| 物体形状 | 碰撞体类型 | 说明 |
|---------|-----------|------|
| 盒子/罐头 | `box` | 默认选择，简单高效 |
| 盘子/碗 | `cylinder`（薄） | 只模拟底部，不包括边缘（见第4节） |
| 圆形水果 | `box`（缩小80%） | ⚠️ 不要用 sphere——球体会滚动不稳定 |
| 香蕉等弯曲物体 | `box`（全尺寸） | ⚠️ 不要缩小——弯曲部分会穿出碰撞体 |
| 长条物体（marker） | `box` | 保证最小半高 8mm |

---

## 2. 物体穿模问题

### 问题描述
物体的 visual mesh 穿过桌面或其他物体表面。

### 根因分析

**Case 1：碰撞体比 mesh 小**
- 碰撞 box 缩放到 85% 后，弯曲物体（香蕉）的 mesh 超出碰撞体范围
- **修复**：弯曲/不规则物体使用**全尺寸**碰撞体

**Case 2：物理仿真后倾斜**
- 物体 settle 后产生微小倾斜，mesh 某一端伸入桌面
- **修复**：自动穿模检测 + 迭代修正（见第10节）

**Case 3：碰撞体中心偏移**
- Box 碰撞体的 pos 不等于 mesh 中心
- **修复**：碰撞体 pos 使用 `(mesh.bounds[0] + mesh.bounds[1]) / 2`

### 关键公式

```python
# 碰撞体底部对齐 mesh 底部
box_half_height = ext[2] / 2
box_center_z = mesh_z_bottom + box_half_height  # 底部对齐
```

---

## 3. 物体悬浮问题

### 问题描述
物体看起来漂浮在桌面/盘子上方。

### 根因分析

**Case 1：碰撞 box 底面高于 mesh 底面**
- 球体碰撞体（sphere）的球心在 mesh 中心，球底比 mesh 底高
- **修复**：球心下移 → `new_center_z = mesh_z_bottom + sphere_radius`

**Case 2：凹面容器上的堆叠高度计算错误** ⭐ 最重要的经验
- 盘子 mesh 高度 2.67cm（包括边缘），但**内表面**仅 0.08cm
- 如果用 `plate_full_height` 作为水果的 z_offset，水果会浮在边缘上方 2.3cm
- **修复**：分析容器 mesh 的内表面高度

```python
# 获取盘子内表面实际高度
plate_mesh = trimesh.load('plate.stl')
verts = plate_mesh.vertices
# 取中心区域顶点的最大z值 = 内表面高度
center_verts = verts[(abs(verts[:,0]) < 0.03) & (abs(verts[:,1]) < 0.03)]
inner_surface_z = center_verts[:,2].max()  # ≈ 0.08cm，远小于 2.67cm

# 水果放置高度
z_offset = inner_surface_z - plate_z_bottom  # ≈ 0.39cm
```

**通用原则：凹面容器的碰撞体应该是薄底盘，而不是全高度 bounding box。**

**Case 3：碰撞体间互相弹飞**
- 盘子碰撞半径 13cm + 碗碰撞半径 8cm，但两者间距只有 14cm → 严重重叠
- 碰撞求解器将重叠物体弹开，导致一个飞到空中
- **修复**：放置前检查碰撞体不重叠（见第6节）

---

## 4. 堆叠放置

### 成功的堆叠策略

| 堆叠组合 | 方法 | 成功率 |
|---------|------|-------|
| 水果 ON 盘子 | 盘子用薄圆柱碰撞，水果 z_offset=盘子内表面 | ✅ 稳定 |
| 罐头 ON 盒子 | z_offset=下方盒子高度，Box-on-Box | ✅ 稳定 |
| 方块 ON 盒子 | 同上 | ✅ 稳定 |
| 物体 IN 碗 | ❌ 不可行 | ❌ 凸包碰撞弹飞 |

### 失败案例：碗内放置

碗的碰撞体是凸包（即使用 box 也是实心的），无法在内部放置物体。物体会被碰撞求解器弹出。

**如需碗内放置**：需要使用多个 geom 组合成碗壁，或等待 MuJoCo 支持凹面碰撞。

---

## 5. 物理仿真稳定性

### 推荐参数

```xml
<option timestep="0.0005"          <!-- 0.5ms，小步长防止穿透 -->
        gravity="0 0 -9.81"
        integrator="implicitfast"   <!-- 快速隐式积分 -->
        cone="elliptic"             <!-- 椭圆摩擦锥 -->
        noslip_iterations="5"/>     <!-- 防止平面物体滑动 -->
```

### 接触参数

```xml
<geom solref="0.004 1"             <!-- 接触刚度 -->
      solimp="0.95 0.99 0.001"     <!-- 接触阻尼 -->
      condim="4"                   <!-- 4D 摩擦锥 -->
      friction="0.8 0.005 0.001"/> <!-- 滑动/扭转/滚动摩擦 -->
```

### 渐进阻尼策略

物理 settle 时不能直接零速度，需要渐进阻尼避免物体在半空中被冻住：

```python
for step in range(40000):  # 20 秒仿真时间
    mujoco.mj_step(model, data)
    if step % 100 == 0:
        max_vel = np.max(np.abs(data.qvel))
        if max_vel < 0.3:
            data.qvel[:] *= 0.97   # 几乎静止：轻微阻尼
        elif max_vel < 1.0:
            data.qvel[:] *= 0.95   # 缓慢运动：中等阻尼
    data.qvel[:] = np.clip(data.qvel, -2.0, 2.0)  # 速度限幅
```

### 桌面设置

```xml
<body name="table" pos="0 0 0.75">
  <geom type="box" size="0.55 0.375 0.02"
        mass="50"                    <!-- 重质量防止被推动 -->
        friction="0.8 0.005 0.001"
        solref="0.002 1"            <!-- 更硬的接触 -->
        solimp="0.98 0.99 0.001"/>
</body>
```

---

## 6. 碰撞感知布局

### 问题
随机或手动布局容易导致碰撞体重叠 → 物体互相弹飞。

### 解决方案：放置前碰撞检测

```python
def check_no_overlap(placed_objects, new_x, new_y, new_radius):
    """检查新物体是否与已放置物体重叠"""
    MARGIN = 0.015  # 1.5cm 安全间距
    for px, py, pr in placed_objects:
        dist = math.sqrt((new_x - px)**2 + (new_y - py)**2)
        if dist < new_radius + pr + MARGIN:
            return False  # 重叠！
    return True

def check_in_bounds(x, y, radius, table_w, table_d):
    """检查物体是否在桌面安全区域内"""
    EDGE_MARGIN = 0.10  # 10cm 离桌边
    safe_x = table_w / 2 - EDGE_MARGIN
    safe_y = table_d / 2 - EDGE_MARGIN
    return abs(x) + radius < safe_x and abs(y) + radius < safe_y
```

### 碰撞半径计算

```python
# 使用 XY 平面的最大半维度作为碰撞半径
collision_radius = max(mesh_extent[0], mesh_extent[1]) / 2
```

### 放置顺序

从大到小放置物体，大物体优先占据位置：

```python
objects_by_size = sorted(objects, key=lambda o: collision_radius(o), reverse=True)
```

---

## 7. 程序化物体构建

### 刀叉勺等餐具

YCB 数据集不包含刀叉等扁平餐具，可用 MuJoCo 原语组合构建：

**叉子**（7个 geom）：
```xml
<body name="fork">
  <freejoint/>
  <!-- 手柄 -->
  <geom type="box" size="0.005 0.07 0.0015" pos="0 -0.04 0" material="steel"/>
  <!-- 叉头 -->
  <geom type="box" size="0.008 0.04 0.001" pos="0 0.05 0" material="steel"/>
  <!-- 4根叉齿 -->
  <geom type="box" size="0.001 0.02 0.0008" pos="-0.005 0.085 0" material="steel"/>
  <geom type="box" size="0.001 0.02 0.0008" pos="-0.0017 0.085 0" material="steel"/>
  <geom type="box" size="0.001 0.02 0.0008" pos="0.0017 0.085 0" material="steel"/>
  <geom type="box" size="0.001 0.02 0.0008" pos="0.005 0.085 0" material="steel"/>
</body>
```

### 铰链物体（剪刀）

```xml
<body name="scissors">
  <freejoint/>
  <!-- 刀刃1 + 手柄1 -->
  <geom type="box" size="0.002 0.06 0.002" pos="-0.003 0.03 0" material="steel"/>
  <geom type="capsule" size="0.008 0.018" pos="-0.012 -0.028 0" material="black_plastic"/>
  <!-- 铆钉 -->
  <geom type="cylinder" size="0.003 0.003" pos="0 0 0" material="steel"/>
  <!-- 刀刃2（子body + hinge joint） -->
  <body name="blade2" pos="0 0 0">
    <joint type="hinge" axis="0 0 1" range="-5 20" stiffness="0.5" damping="0.1"/>
    <geom type="box" size="0.002 0.06 0.002" pos="0.003 0.03 0" material="steel"/>
    <geom type="capsule" size="0.008 0.018" pos="0.012 -0.028 0" material="black_plastic"/>
  </body>
</body>
```

### ⚠️ 多 geom 连接注意事项

组合多个 geom 时，必须确保**相邻 geom 有重叠**，否则视觉上看起来断裂：

```python
# 勺子：手柄和勺碗必须衔接
# 手柄: half-size 0.065 at center y=-0.030 → spans y=-0.095 to y=0.035
# 勺碗: half-size 0.013 at center y=0.045  → spans y=0.032 to y=0.058
# 重叠: 0.035 > 0.032 → 3mm 重叠，视觉连续 ✅
```

---

## 8. 纹理渲染

### YCB 纹理加载

必须使用 `.obj` 格式（不是 `.stl`），因为 `.obj` 包含 UV 坐标：

```xml
<mesh name="visual" file="textured.obj"/>
<texture name="tex" type="2d" file="texture_map.png"/>
<material name="mat" texture="tex"/>
```

### MuJoCo 渲染器初始化

```python
# ⚠️ 参数顺序：height, width（不是 width, height！）
renderer = mujoco.Renderer(model, 1440, 1920)
```

### 材质定义

```xml
<!-- 金属餐具 -->
<material name="steel" rgba="0.78 0.78 0.82 1" shininess="0.9" specular="0.7" reflectance="0.3"/>

<!-- 木质手柄 -->
<material name="dark_handle" rgba="0.20 0.12 0.08 1" shininess="0.3" specular="0.1"/>

<!-- 深色木质桌面（与白色物体形成对比） -->
<material name="table" rgba="0.35 0.22 0.10 1" shininess="0.4" specular="0.25"/>
```

---

## 9. 物理属性管理

### JSON 预存储

为每个 YCB 物体维护物理属性文件（`data/ycb_physics.json`）：

```json
{
    "029_plate": {
        "mass": 0.279,
        "friction": [0.5, 0.005, 0.001],
        "material": "ceramic",
        "density": 800
    },
    "011_banana": {
        "mass": 0.066,
        "friction": [0.7, 0.005, 0.001],
        "material": "organic",
        "density": 400
    }
}
```

### 材质摩擦系数参考

| 材质 | 滑动摩擦 | 典型物体 |
|------|---------|---------|
| 陶瓷 | 0.5 | 盘子、碗、马克杯 |
| 金属 | 0.5 | 罐头、刀叉 |
| 塑料 | 0.4 | 瓶子、杯子 |
| 纸板 | 0.6 | 盒子 |
| 有机物 | 0.6-0.7 | 水果 |
| 泡沫 | 0.8 | 泡沫砖 |

---

## 10. 自动化质量检测

### 穿模检测

物理仿真后，将每个物体的 visual mesh 顶点变换到世界坐标系，检查是否穿透支撑面：

```python
def check_penetration(model, data, mesh_data, table_surface_z):
    """检测所有物体的视觉 mesh 是否穿透桌面"""
    penetrations = {}
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if not name or not name.startswith("o"): continue
        
        idx = int(name[1:])
        verts = mesh_data[idx]['verts']  # 本地坐标系顶点
        if verts is None: continue
        
        # 变换到世界坐标系
        pos = data.xpos[i]
        quat = data.xquat[i]  # w,x,y,z
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        world_verts = rot.apply(verts) + pos
        
        min_z = world_verts[:, 2].min()
        pen = table_surface_z - min_z  # 正值 = 穿透
        penetrations[idx] = pen
    
    return penetrations
```

### 迭代修复流程

```
1. 构建场景 XML
2. 运行物理仿真 (40000 步)
3. 检测穿模
4. 如有穿模 > 1mm：
   a. 计算需要抬高的量 = 穿透深度 + 2mm 余量
   b. 修改 z_offset
   c. 回到步骤 1
5. 直到所有物体通过检测
6. 渲染输出
```

### 完整性检查清单

- [ ] 所有物体在桌面安全区域内（距边缘 >10cm）
- [ ] 地面层物体碰撞体无重叠（bounding circle + 1.5cm margin）
- [ ] 所有视觉 mesh 顶点 z > 桌面高度 - 1mm
- [ ] 所有物体 tilt < 25°（除非有意设计倾斜）
- [ ] 堆叠物体 z 高度合理（在支撑物上方）
- [ ] 无物体掉落到地面（z > 0.3m）
- [ ] 物理仿真后速度趋近零（已收敛）

---

## 附录：常见陷阱

| 陷阱 | 症状 | 修复 |
|------|------|------|
| `type="mesh"` 用于碰撞 | 凹面物体碰撞异常 | 改用原语 |
| Renderer(model, width, height) | 图像比例错误 | 参数顺序是 height, width |
| 球体碰撞给水果 | 水果从盘子上滚落 | 改用缩小的 box |
| 碰撞体缩小到 85% | 弯曲 mesh 穿出 | 不规则物体用全尺寸 |
| 用盘子总高度做堆叠 offset | 物体浮在盘缘上方 | 用内表面高度 |
| 碗内放物体 | 物体被弹飞 | 凸包碰撞限制，改放桌面 |
| 碰撞体重叠布局 | 物理弹飞多个物体 | 放置前检测重叠 |
| 多 geom 不重叠 | 视觉上"断裂" | 确保相邻 geom 有几 mm 重叠 |
| capsule 给扁平弯曲物体 | 物体立起来 | 改用 box |
| shininess 写在 geom 上 | XML 解析错误 | shininess 是 material 属性 |

---

*最后更新：2026-03-08*
*项目：SceneCraft-LLM*
*仓库：https://github.com/www-Ye/SceneCraft-LLM*
