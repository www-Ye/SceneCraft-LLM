#!/usr/bin/env python3
"""
桌面小物体3D资产下载和转换脚本
使用objaverse下载LVIS标注的3D模型，归一化尺寸后导出为STL格式
"""

import objaverse
import trimesh
import json
import os
import numpy as np
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 桌面小物体归一化尺寸（米）
TABLETOP_STANDARD_DIMS = {
    "mug": {"width": 0.08, "depth": 0.08, "height": 0.10},
    "cup": {"width": 0.08, "depth": 0.08, "height": 0.10},
    "bowl": {"width": 0.15, "depth": 0.15, "height": 0.08},
    "plate": {"width": 0.25, "depth": 0.25, "height": 0.02},
    "dish": {"width": 0.25, "depth": 0.25, "height": 0.02},
    "bottle": {"width": 0.07, "depth": 0.07, "height": 0.25},
    "can": {"width": 0.06, "depth": 0.06, "height": 0.12},
    "apple": {"width": 0.08, "depth": 0.08, "height": 0.08},
    "banana": {"width": 0.04, "depth": 0.18, "height": 0.04},
    "orange": {"width": 0.08, "depth": 0.08, "height": 0.08},
    "book": {"width": 0.15, "depth": 0.22, "height": 0.03},
    "pen": {"width": 0.01, "depth": 0.15, "height": 0.01},
    "pencil": {"width": 0.01, "depth": 0.18, "height": 0.01},
    "box": {"width": 0.15, "depth": 0.10, "height": 0.10},
    "keyboard": {"width": 0.44, "depth": 0.14, "height": 0.03},
    "mouse": {"width": 0.06, "depth": 0.10, "height": 0.04},
    "remote_control": {"width": 0.05, "depth": 0.18, "height": 0.02},
    "scissors": {"width": 0.03, "depth": 0.18, "height": 0.01},
    "tape_dispenser": {"width": 0.08, "depth": 0.08, "height": 0.05},
    "stapler": {"width": 0.06, "depth": 0.15, "height": 0.05},
}

# LVIS类别到标准化名称的映射
LVIS_CATEGORY_MAPPING = {
    "mug": ["mug", "cup"],
    "cup": ["cup", "mug"],  
    "bowl": ["bowl"],
    "plate": ["plate", "dish"],
    "dish": ["dish", "plate"],
    "bottle": ["bottle"],
    "can": ["can"],
    "apple": ["apple"],
    "banana": ["banana"],
    "orange": ["orange"],
    "book": ["book"],
    "pen": ["pen", "pencil"],
    "pencil": ["pencil", "pen"],
    "box": ["box"],
    "keyboard": ["keyboard"],
    "mouse": ["mouse"],
    "remote_control": ["remote_control", "remote"],
    "scissors": ["scissors"],
    "tape_dispenser": ["tape_dispenser"],
    "stapler": ["stapler"],
}

def normalize_mesh_axis_agnostic(mesh, target_dims):
    """
    Axis-agnostic mesh normalization
    测试所有6种轴排列，找最匹配目标尺寸的
    """
    if not mesh.is_volume:
        logger.warning("Mesh is not watertight, trying to fix...")
        mesh = mesh.convex_hull
    
    # 居中到原点
    mesh.vertices -= mesh.bounds.mean(axis=0)
    
    # 当前边界框
    current_dims = mesh.bounds[1] - mesh.bounds[0]
    target_arr = np.array([target_dims['width'], target_dims['depth'], target_dims['height']])
    
    # 测试所有6种轴排列 (x,y,z), (x,z,y), (y,x,z), (y,z,x), (z,x,y), (z,y,x)
    permutations = [
        [0, 1, 2],  # x,y,z
        [0, 2, 1],  # x,z,y  
        [1, 0, 2],  # y,x,z
        [1, 2, 0],  # y,z,x
        [2, 0, 1],  # z,x,y
        [2, 1, 0]   # z,y,x
    ]
    
    best_error = float('inf')
    best_perm = None
    
    for perm in permutations:
        perm_dims = current_dims[perm]
        # 计算缩放因子和误差
        scale_factors = target_arr / perm_dims
        min_scale = scale_factors.min()
        scaled_dims = perm_dims * min_scale
        error = np.sum((scaled_dims - target_arr) ** 2)
        
        if error < best_error:
            best_error = error
            best_perm = perm
    
    # 应用最佳轴排列
    if best_perm != [0, 1, 2]:
        vertices = mesh.vertices.copy()
        mesh.vertices = vertices[:, best_perm]
        logger.info(f"Applied axis permutation: {best_perm}")
    
    # 重新计算边界框并缩放
    current_dims = mesh.bounds[1] - mesh.bounds[0]
    scale_factors = target_arr / current_dims
    min_scale = scale_factors.min()
    mesh.apply_scale(min_scale)
    
    # 平移使bottom_z = 0
    mesh.vertices[:, 2] -= mesh.bounds[0, 2]
    
    return mesh

def find_category_uids(lvis_annotations, category_name, max_objects=2):
    """在LVIS标注中查找类别的UIDs"""
    possible_names = LVIS_CATEGORY_MAPPING.get(category_name, [category_name])
    
    for name in possible_names:
        if name in lvis_annotations:
            uids = lvis_annotations[name]
            logger.info(f"Found {len(uids)} objects for category '{name}' (searching for '{category_name}')")
            return uids[:max_objects]
    
    logger.warning(f"Category '{category_name}' not found in LVIS annotations")
    return []

def process_model(uid, glb_path, category, output_dir, catalog_data, model_index):
    """处理单个3D模型"""
    try:
        # 加载GLB文件
        logger.info(f"Processing {category} model {model_index}: {uid}")
        scene_or_mesh = trimesh.load(glb_path, force='scene')
        
        # 如果是场景，合并所有geometry
        if hasattr(scene_or_mesh, 'geometry'):
            if len(scene_or_mesh.geometry) == 0:
                logger.warning(f"Empty scene for {uid}")
                return False
            
            # 合并所有几何体
            meshes = [geom for geom in scene_or_mesh.geometry.values() 
                     if isinstance(geom, trimesh.Trimesh)]
            if not meshes:
                logger.warning(f"No valid meshes in scene for {uid}")
                return False
                
            if len(meshes) == 1:
                mesh = meshes[0]
            else:
                mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = scene_or_mesh
        
        # 检查mesh有效性
        if not isinstance(mesh, trimesh.Trimesh):
            logger.warning(f"Invalid mesh type for {uid}")
            return False
            
        if len(mesh.vertices) == 0:
            logger.warning(f"Empty mesh for {uid}")
            return False
        
        # 归一化尺寸
        target_dims = TABLETOP_STANDARD_DIMS[category]
        normalized_mesh = normalize_mesh_axis_agnostic(mesh, target_dims)
        
        # 验证mesh是否有效（使用is_watertight作为替代）
        if not normalized_mesh.is_watertight:
            logger.warning(f"Mesh validation failed for {uid}, attempting to repair...")
            normalized_mesh = normalized_mesh.convex_hull
        
        # 生成输出文件名
        output_filename = f"{category}_{model_index}.stl"
        output_path = output_dir / output_filename
        
        # 导出STL
        normalized_mesh.export(str(output_path))
        logger.info(f"Exported STL: {output_filename}")
        
        # 更新目录数据
        if category not in catalog_data:
            catalog_data[category] = []
        
        actual_dims = normalized_mesh.bounds[1] - normalized_mesh.bounds[0]
        catalog_data[category].append({
            "uid": uid,
            "file": output_filename,
            "dims": actual_dims.tolist()
        })
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to process model {uid}: {str(e)}")
        return False

def main():
    # 设置路径
    base_dir = Path("/root/.openclaw/workspace-lab/scene-gen")
    data_dir = base_dir / "data"
    assets_dir = data_dir / "assets"
    stl_dir = assets_dir / "_mujoco_stl"
    catalog_path = data_dir / "tabletop_asset_catalog.json"
    
    # 确保目录存在
    stl_dir.mkdir(exist_ok=True)
    
    logger.info("Loading LVIS annotations...")
    try:
        lvis_annotations = objaverse.load_lvis_annotations()
        logger.info(f"Loaded LVIS annotations with {len(lvis_annotations)} categories")
    except Exception as e:
        logger.error(f"Failed to load LVIS annotations: {e}")
        return
    
    # 需要下载的类别
    target_categories = [
        "mug", "cup", "bowl", "plate", "dish", "bottle", "can",
        "apple", "banana", "orange", "book", "pen", "pencil", "box",
        "keyboard", "mouse", "remote_control", "scissors", "tape_dispenser", "stapler"
    ]
    
    catalog_data = {}
    all_uids = []
    uid_to_category = {}
    
    # 收集所有需要下载的UIDs
    for category in target_categories:
        uids = find_category_uids(lvis_annotations, category, max_objects=2)
        if uids:
            all_uids.extend(uids)
            for uid in uids:
                uid_to_category[uid] = category
        else:
            logger.warning(f"Skipping category '{category}' - no models found")
    
    if not all_uids:
        logger.error("No models found to download!")
        return
    
    logger.info(f"Downloading {len(all_uids)} models...")
    
    # 下载所有模型
    try:
        objects = objaverse.load_objects(uids=all_uids)
        logger.info(f"Downloaded {len(objects)} objects successfully")
    except Exception as e:
        logger.error(f"Failed to download objects: {e}")
        return
    
    # 处理每个下载的模型
    category_counters = {}
    
    for uid, glb_path in objects.items():
        if uid not in uid_to_category:
            continue
            
        category = uid_to_category[uid]
        
        # 计数器
        if category not in category_counters:
            category_counters[category] = 0
        
        model_index = category_counters[category]
        category_counters[category] += 1
        
        # 处理模型
        success = process_model(uid, glb_path, category, stl_dir, catalog_data, model_index)
        if success:
            logger.info(f"Successfully processed {category} model {model_index}")
        else:
            logger.warning(f"Failed to process {category} model {model_index}")
    
    # 保存目录文件
    with open(catalog_path, 'w', encoding='utf-8') as f:
        json.dump(catalog_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved catalog to {catalog_path}")
    
    # 打印总结
    logger.info("=== Download Summary ===")
    total_models = sum(len(models) for models in catalog_data.values())
    logger.info(f"Total categories processed: {len(catalog_data)}")
    logger.info(f"Total models converted: {total_models}")
    
    for category, models in catalog_data.items():
        logger.info(f"  {category}: {len(models)} models")

if __name__ == "__main__":
    main()