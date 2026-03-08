#!/usr/bin/env python3
"""
桌面小物体3D资产下载和转换脚本 (FIXED)
使用objaverse下载LVIS标注的3D模型，归一化尺寸后导出为STL格式

BUGFIX: Removed destructive convex hull operations that destroyed mesh quality.
Instead uses gentler mesh repair and separate visual/collision meshes in MuJoCo.
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

def normalize_mesh_axis_agnostic(mesh, target_dims, preserve_geometry=True):
    """
    Per-axis scaling normalization with improved axis permutation scoring.
    
    BUGFIX: Uses independent per-axis scaling instead of uniform scaling.
    This ensures each dimension matches the target exactly.
    """
    logger.info(f"Original mesh: {len(mesh.vertices)} vertices, watertight: {mesh.is_watertight}")
    
    # Simplify very high-poly meshes for MuJoCo compatibility
    if len(mesh.vertices) > 50000 or len(mesh.faces) > 100000:
        logger.warning(f"Mesh too high-poly ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces), simplifying...")
        try:
            # Calculate reduction ratio to get ~20k vertices and <100k faces
            vertex_ratio = 20000 / len(mesh.vertices) if len(mesh.vertices) > 20000 else 1.0
            face_ratio = 80000 / len(mesh.faces) if len(mesh.faces) > 80000 else 1.0
            reduction_ratio = min(vertex_ratio, face_ratio, 0.9)  # Cap at 90% reduction
            
            mesh = mesh.simplify_quadric_decimation(1.0 - reduction_ratio)
            logger.info(f"Simplified to {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        except Exception as e:
            logger.warning(f"Simplification failed: {e}, trying vertex clustering...")
            try:
                mesh = mesh.simplify_vertex_clustering(0.01)  # 1cm clusters
                logger.info(f"Vertex clustering reduced to {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            except Exception as e2:
                logger.warning(f"All simplification failed: {e2}")
    
    # Only try gentle repairs if mesh has issues
    if not mesh.is_volume and preserve_geometry:
        logger.warning("Mesh is not a volume, attempting gentle repair...")
        
        # Try gentle repair methods first
        try:
            # Fill small holes
            repaired = mesh.copy()
            repaired.fill_holes()
            if repaired.is_volume:
                mesh = repaired
                logger.info("Successfully filled holes")
            else:
                # Try fixing normals
                mesh.fix_normals()
                logger.info("Fixed normals")
        except Exception as e:
            logger.warning(f"Gentle repair failed: {e}")
    
    # If preserve_geometry is False or we still have issues, fallback to convex hull
    if not mesh.is_volume and not preserve_geometry:
        logger.warning("Using convex hull as fallback (geometry will be simplified)")
        mesh = mesh.convex_hull
    
    # Center mesh
    mesh.vertices -= mesh.bounds.mean(axis=0)
    current_dims = mesh.bounds[1] - mesh.bounds[0]
    target_arr = np.array([target_dims['width'], target_dims['depth'], target_dims['height']])
    
    # Find best axis permutation
    # Score: minimize the variance of scale ratios (most uniform scaling = least distortion)
    perms = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
    best_score = float('inf')
    best_perm = [0,1,2]
    
    for perm in perms:
        perm_dims = current_dims[perm]
        ratios = target_arr / np.maximum(perm_dims, 1e-6)
        # Score: variance of log-ratios (penalize non-uniform scaling)
        log_ratios = np.log(ratios)
        score = np.var(log_ratios)
        if score < best_score:
            best_score = score
            best_perm = perm
    
    # Apply permutation
    if best_perm != [0, 1, 2]:
        mesh.vertices = mesh.vertices[:, best_perm]
        logger.info(f"Applied axis permutation: {best_perm}")
    
    # PER-AXIS scaling (not uniform!)
    current_dims = mesh.bounds[1] - mesh.bounds[0]
    for axis in range(3):
        scale = target_arr[axis] / max(current_dims[axis], 1e-6)
        mesh.vertices[:, axis] *= scale
        logger.info(f"Axis {axis}: scaled by {scale:.4f} (from {current_dims[axis]:.4f} to {target_arr[axis]:.4f})")
    
    # Bottom at z=0
    mesh.vertices[:, 2] -= mesh.bounds[0, 2]
    # Center XY
    mesh.vertices[:, 0] -= (mesh.bounds[0][0] + mesh.bounds[1][0]) / 2
    mesh.vertices[:, 1] -= (mesh.bounds[0][1] + mesh.bounds[1][1]) / 2
    
    # Log final dimensions for verification
    final_dims = mesh.bounds[1] - mesh.bounds[0]
    logger.info(f"Final mesh dimensions: [{final_dims[0]:.4f}, {final_dims[1]:.4f}, {final_dims[2]:.4f}]")
    logger.info(f"Target dimensions: [{target_arr[0]:.4f}, {target_arr[1]:.4f}, {target_arr[2]:.4f}]")
    
    return mesh


# Categories where the opening/wide part should face UP (z+)
CONTAINER_CATEGORIES = {"mug", "cup", "bowl", "plate", "dish"}

def fix_container_orientation(mesh, category):
    """
    For container objects (mug, cup, bowl, plate, dish), ensure the
    opening / wider part faces z+ (up).
    
    Heuristic: compare the XY spread of vertices in the bottom 20% vs top 20%.
    If the bottom is wider than the top, flip the mesh 180° around X.
    """
    if category not in CONTAINER_CATEGORIES:
        return mesh
    
    ext = mesh.bounds[1] - mesh.bounds[0]
    h = ext[2]
    if h < 1e-6:
        return mesh
    
    z_min, z_max = mesh.bounds[0][2], mesh.bounds[1][2]
    
    # Get vertices in bottom 20% and top 20%
    bottom_mask = mesh.vertices[:, 2] < z_min + h * 0.2
    top_mask = mesh.vertices[:, 2] > z_max - h * 0.2
    
    bottom_verts = mesh.vertices[bottom_mask]
    top_verts = mesh.vertices[top_mask]
    
    if len(bottom_verts) < 3 or len(top_verts) < 3:
        logger.warning(f"Not enough vertices to check orientation for {category}")
        return mesh
    
    # Compute XY spread (max - min) for each group
    bottom_spread = (bottom_verts[:, :2].max(axis=0) - bottom_verts[:, :2].min(axis=0)).mean()
    top_spread = (top_verts[:, :2].max(axis=0) - top_verts[:, :2].min(axis=0)).mean()
    
    logger.info(f"Orientation check ({category}): bottom_spread={bottom_spread:.4f}, top_spread={top_spread:.4f}")
    
    if top_spread < bottom_spread * 0.85:  # Top is significantly narrower → upside down
        logger.warning(f"  → {category} is UPSIDE DOWN! Flipping 180° around X axis.")
        # Flip: negate Z, then shift so bottom is at z=0 again
        mesh.vertices[:, 2] = -mesh.vertices[:, 2]
        mesh.vertices[:, 2] -= mesh.bounds[0, 2]
        
        # Verify
        bottom_verts2 = mesh.vertices[mesh.vertices[:, 2] < mesh.bounds[0][2] + h * 0.2]
        top_verts2 = mesh.vertices[mesh.vertices[:, 2] > mesh.bounds[1][2] - h * 0.2]
        if len(bottom_verts2) > 0 and len(top_verts2) > 0:
            bs2 = (bottom_verts2[:, :2].max(axis=0) - bottom_verts2[:, :2].min(axis=0)).mean()
            ts2 = (top_verts2[:, :2].max(axis=0) - top_verts2[:, :2].min(axis=0)).mean()
            logger.info(f"  → After flip: bottom_spread={bs2:.4f}, top_spread={ts2:.4f} ✓")
        
        # Also fix face winding (flipping Z inverts normals)
        mesh.faces = mesh.faces[:, ::-1]
        mesh.fix_normals()
    else:
        logger.info(f"  → {category} orientation is correct ✓")
    
    return mesh

def create_collision_mesh(visual_mesh):
    """
    Create a collision mesh (convex hull) from the visual mesh.
    This preserves the original visual geometry while providing a simpler collision shape.
    """
    collision_mesh = visual_mesh.convex_hull
    
    # Ensure collision mesh is not too complex for physics
    if len(collision_mesh.vertices) > 5000 or len(collision_mesh.faces) > 10000:
        try:
            vertex_ratio = 2000 / len(collision_mesh.vertices) if len(collision_mesh.vertices) > 2000 else 1.0
            face_ratio = 8000 / len(collision_mesh.faces) if len(collision_mesh.faces) > 8000 else 1.0
            reduction_ratio = min(vertex_ratio, face_ratio, 0.9)
            
            collision_mesh = collision_mesh.simplify_quadric_decimation(1.0 - reduction_ratio)
            logger.info(f"Simplified collision mesh to {len(collision_mesh.vertices)} vertices, {len(collision_mesh.faces)} faces")
        except:
            try:
                collision_mesh = collision_mesh.simplify_vertex_clustering(0.02)  # 2cm clusters for collision
                logger.info(f"Vertex clustering reduced collision to {len(collision_mesh.vertices)} vertices, {len(collision_mesh.faces)} faces")
            except:
                logger.warning("Could not simplify collision mesh, using convex hull")
    
    logger.info(f"Collision mesh: {len(collision_mesh.vertices)} vertices (simplified from {len(visual_mesh.vertices)})")
    return collision_mesh

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
    """处理单个3D模型，导出visual和collision两个版本"""
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
        
        # 归一化尺寸 - PRESERVE original geometry for visual mesh
        target_dims = TABLETOP_STANDARD_DIMS[category]
        visual_mesh = normalize_mesh_axis_agnostic(mesh, target_dims, preserve_geometry=True)
        
        # Fix orientation for containers (opening should face UP)
        visual_mesh = fix_container_orientation(visual_mesh, category)
        
        # Create separate collision mesh (simplified)
        collision_mesh = create_collision_mesh(visual_mesh)
        
        # 生成输出文件名
        visual_filename = f"{category}_{model_index}_visual.stl"
        collision_filename = f"{category}_{model_index}_collision.stl"
        visual_path = output_dir / visual_filename
        collision_path = output_dir / collision_filename
        
        # 导出STL files
        visual_mesh.export(str(visual_path))
        collision_mesh.export(str(collision_path))
        logger.info(f"Exported visual STL: {visual_filename}")
        logger.info(f"Exported collision STL: {collision_filename}")
        
        # 更新目录数据
        if category not in catalog_data:
            catalog_data[category] = []
        
        actual_dims = visual_mesh.bounds[1] - visual_mesh.bounds[0]
        catalog_data[category].append({
            "uid": uid,
            "visual_file": visual_filename,
            "collision_file": collision_filename,
            "dims": actual_dims.tolist(),
            "visual_vertices": len(visual_mesh.vertices),
            "collision_vertices": len(collision_mesh.vertices)
        })
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to process model {uid}: {str(e)}")
        return False

class TabletopAssetDownloader:
    """Asset downloader with improved mesh processing."""
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.assets_dir = self.data_dir / "assets"
        self.stl_dir = self.assets_dir / "_mujoco_stl"
        self.catalog_path = self.data_dir / "tabletop_asset_catalog.json"
        
        # 确保目录存在
        self.stl_dir.mkdir(parents=True, exist_ok=True)
    
    def download_assets(self, target_categories=None, max_objects=2):
        """Download and process assets with improved mesh handling."""
        if target_categories is None:
            target_categories = [
                "mug", "cup", "bowl", "plate", "dish", "bottle", "can",
                "apple", "banana", "orange", "book", "pen", "pencil", "box",
                "keyboard", "mouse", "remote_control", "scissors", "tape_dispenser", "stapler"
            ]
        
        logger.info("Loading LVIS annotations...")
        try:
            lvis_annotations = objaverse.load_lvis_annotations()
            logger.info(f"Loaded LVIS annotations with {len(lvis_annotations)} categories")
        except Exception as e:
            logger.error(f"Failed to load LVIS annotations: {e}")
            return False
        
        catalog_data = {}
        all_uids = []
        uid_to_category = {}
        
        # 收集所有需要下载的UIDs
        for category in target_categories:
            uids = find_category_uids(lvis_annotations, category, max_objects)
            if uids:
                all_uids.extend(uids)
                for uid in uids:
                    uid_to_category[uid] = category
            else:
                logger.warning(f"Skipping category '{category}' - no models found")
        
        if not all_uids:
            logger.error("No models found to download!")
            return False
        
        logger.info(f"Downloading {len(all_uids)} models...")
        
        # 下载所有模型
        try:
            objects = objaverse.load_objects(uids=all_uids)
            logger.info(f"Downloaded {len(objects)} objects successfully")
        except Exception as e:
            logger.error(f"Failed to download objects: {e}")
            return False
        
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
            success = process_model(uid, glb_path, category, self.stl_dir, catalog_data, model_index)
            if success:
                logger.info(f"Successfully processed {category} model {model_index}")
            else:
                logger.warning(f"Failed to process {category} model {model_index}")
        
        # 保存目录文件
        with open(self.catalog_path, 'w', encoding='utf-8') as f:
            json.dump(catalog_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved catalog to {self.catalog_path}")
        
        # 打印总结
        logger.info("=== Download Summary ===")
        total_models = sum(len(models) for models in catalog_data.values())
        logger.info(f"Total categories processed: {len(catalog_data)}")
        logger.info(f"Total models converted: {total_models}")
        
        for category, models in catalog_data.items():
            logger.info(f"  {category}: {len(models)} models")
        
        return True

def main():
    """Main function for standalone use."""
    downloader = TabletopAssetDownloader()
    downloader.download_assets()

if __name__ == "__main__":
    main()