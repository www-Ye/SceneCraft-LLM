#!/usr/bin/env python3
"""
TableTop-Bench: Desktop object layout evaluation sub-benchmark.
Evaluates small-scale object placement on tabletop surfaces.
"""

import json
import math
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Any, Tuple, Optional
import tempfile
import xml.etree.ElementTree as ET
import mujoco

# Set MuJoCo to headless mode
os.environ['MUJOCO_GL'] = 'osmesa'

# Tabletop object dimensions (width, depth, height) in meters
TABLETOP_DIMS = {
    "plate": [0.25, 0.25, 0.02],
    "cup": [0.08, 0.08, 0.10], 
    "bowl": [0.15, 0.15, 0.08],
    "fork": [0.02, 0.18, 0.01],
    "knife": [0.02, 0.20, 0.02],
    "napkin": [0.15, 0.15, 0.005],
    "laptop": [0.35, 0.25, 0.02],
    "book": [0.20, 0.15, 0.03],
    "pen_holder": [0.08, 0.08, 0.12],
    "phone": [0.07, 0.15, 0.008],
    "lamp": [0.15, 0.15, 0.35],
    "glass": [0.07, 0.07, 0.12],
    "condiment": [0.10, 0.10, 0.15],
    "toolbox": [0.30, 0.15, 0.15],
    "screwdriver": [0.03, 0.20, 0.03],
    "wrench": [0.04, 0.18, 0.02],
    "parts_tray": [0.20, 0.12, 0.04],
    "goggles": [0.15, 0.08, 0.06],
    "teapot": [0.15, 0.12, 0.15],
    "tea_canister": [0.08, 0.08, 0.12],
}

class TabletopBench:
    """Evaluate tabletop object layouts."""
    
    def __init__(self, prompts_path: str = None):
        """Initialize with prompts file path."""
        if prompts_path is None:
            prompts_path = os.path.join(
                os.path.dirname(__file__), 
                "../prompts/tabletop_prompts.json"
            )
        self.prompts_path = prompts_path
        self.prompts = self._load_prompts()
        
    def _load_prompts(self) -> List[Dict]:
        """Load tabletop prompts from JSON file."""
        try:
            with open(self.prompts_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Prompts file not found: {self.prompts_path}")
            return []
    
    def mock_llm_generate(self, prompt: str, surface_size: List[float]) -> List[Dict]:
        """Mock LLM layout generation for testing purposes."""
        # Simple grid-based placement for basic functionality testing
        objects = []
        obj_names = self._extract_objects_from_prompt(prompt)
        
        # Place objects in a simple grid
        rows = math.ceil(math.sqrt(len(obj_names)))
        cols = math.ceil(len(obj_names) / rows)
        
        cell_w = surface_size[0] / cols
        cell_h = surface_size[1] / rows
        
        for i, obj_name in enumerate(obj_names):
            row = i // cols
            col = i % cols
            
            # Center position in each cell
            x = (col + 0.5) * cell_w - surface_size[0] / 2
            y = (row + 0.5) * cell_h - surface_size[1] / 2
            
            # Get object dimensions
            dims = TABLETOP_DIMS.get(obj_name, [0.1, 0.1, 0.1])
            
            objects.append({
                "name": obj_name,
                "position": [x, y, dims[2] / 2],  # z at half-height
                "dimensions": dims,
                "rotation": 0.0
            })
        
        return objects
    
    def _extract_objects_from_prompt(self, prompt: str) -> List[str]:
        """Extract object names from prompt text (simple keyword matching)."""
        objects = []
        for obj_name in TABLETOP_DIMS.keys():
            # Count occurrences to handle duplicates (like multiple cups)
            count = prompt.lower().count(obj_name.replace('_', ' '))
            if count == 0:
                count = prompt.lower().count(obj_name)
            objects.extend([obj_name] * count)
        
        return objects
    
    def evaluate_layout(self, layout: List[Dict], surface_size: List[float], 
                       prompt_data: Dict) -> Dict[str, Any]:
        """Evaluate a tabletop layout with multiple metrics."""
        results = {
            "collision_score": self._check_collisions(layout),
            "boundary_score": self._check_boundaries(layout, surface_size),
            "gripper_clearance": self._check_gripper_clearance(layout),
            "functional_grouping": self._check_functional_grouping(layout, prompt_data),
            "coverage_efficiency": self._check_coverage_efficiency(layout, surface_size)
        }
        
        # Overall score (weighted average)
        weights = {
            "collision_score": 0.3,
            "boundary_score": 0.2, 
            "gripper_clearance": 0.25,
            "functional_grouping": 0.15,
            "coverage_efficiency": 0.1
        }
        
        results["overall_score"] = sum(
            results[key] * weights[key] for key in weights.keys()
        )
        
        return results
    
    def _check_collisions(self, layout: List[Dict]) -> float:
        """Check for object-object collisions. Returns score 0-1 (1=no collisions)."""
        if len(layout) <= 1:
            return 1.0
        
        collisions = 0
        total_pairs = 0
        
        for i, obj1 in enumerate(layout):
            for j, obj2 in enumerate(layout[i+1:], i+1):
                total_pairs += 1
                
                # Get 2D bounding boxes (ignoring height)
                bbox1 = self._get_2d_bbox(obj1)
                bbox2 = self._get_2d_bbox(obj2)
                
                # Check overlap
                if self._boxes_overlap(bbox1, bbox2):
                    collisions += 1
        
        return max(0.0, 1.0 - collisions / total_pairs) if total_pairs > 0 else 1.0
    
    def _get_2d_bbox(self, obj: Dict) -> Tuple[float, float, float, float]:
        """Get 2D bounding box (x_min, y_min, x_max, y_max)."""
        pos = obj["position"]
        dims = obj["dimensions"]
        
        x_min = pos[0] - dims[0] / 2
        x_max = pos[0] + dims[0] / 2
        y_min = pos[1] - dims[1] / 2
        y_max = pos[1] + dims[1] / 2
        
        return (x_min, y_min, x_max, y_max)
    
    def _boxes_overlap(self, bbox1: Tuple, bbox2: Tuple) -> bool:
        """Check if two 2D bounding boxes overlap."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        return not (x1_max <= x2_min or x2_max <= x1_min or 
                   y1_max <= y2_min or y2_max <= y1_min)
    
    def _check_boundaries(self, layout: List[Dict], surface_size: List[float]) -> float:
        """Check if objects stay within table boundaries. Returns score 0-1."""
        if not layout:
            return 1.0
        
        violations = 0
        
        # Table boundaries (centered at origin)
        table_x_min = -surface_size[0] / 2
        table_x_max = surface_size[0] / 2
        table_y_min = -surface_size[1] / 2  
        table_y_max = surface_size[1] / 2
        
        for obj in layout:
            bbox = self._get_2d_bbox(obj)
            x_min, y_min, x_max, y_max = bbox
            
            # Check if object extends beyond table
            if (x_min < table_x_min or x_max > table_x_max or 
                y_min < table_y_min or y_max > table_y_max):
                violations += 1
        
        return max(0.0, 1.0 - violations / len(layout))
    
    def _check_gripper_clearance(self, layout: List[Dict], 
                                clearance_radius: float = 0.03) -> float:
        """Check if each object has sufficient clearance for robotic gripper."""
        if not layout:
            return 1.0
        
        valid_objects = 0
        
        for i, target_obj in enumerate(layout):
            has_clearance = True
            target_pos = np.array(target_obj["position"][:2])
            
            # Check clearance against all other objects
            for j, other_obj in enumerate(layout):
                if i == j:
                    continue
                
                other_pos = np.array(other_obj["position"][:2])
                distance = np.linalg.norm(target_pos - other_pos)
                
                # Required distance = object radii + clearance
                target_radius = max(target_obj["dimensions"][:2]) / 2
                other_radius = max(other_obj["dimensions"][:2]) / 2
                required_distance = target_radius + other_radius + clearance_radius
                
                if distance < required_distance:
                    has_clearance = False
                    break
            
            if has_clearance:
                valid_objects += 1
        
        return valid_objects / len(layout)
    
    def _check_functional_grouping(self, layout: List[Dict], 
                                  prompt_data: Dict) -> float:
        """Check if functionally related objects are appropriately grouped."""
        # Define functional groups for different scenes
        scene_groups = {
            "breakfast": [["plate", "fork", "knife"], ["cup", "napkin"]],
            "study_desk": [["laptop", "book"], ["phone", "pen_holder"]],
            "dining_for_two": [["plate", "fork", "knife"], ["glass"]],
            "workshop": [["toolbox", "screwdriver", "wrench"], ["parts_tray", "goggles"]],
            "tea_ceremony": [["teapot", "cup"], ["tea_canister"]]
        }
        
        scene = prompt_data.get("scene", "")
        groups = scene_groups.get(scene, [])
        
        if not groups:
            return 1.0  # No specific grouping requirements
        
        group_scores = []
        
        for group in groups:
            # Find objects in this functional group that exist in layout
            group_objects = [obj for obj in layout if obj["name"] in group]
            
            if len(group_objects) < 2:
                continue  # Need at least 2 objects to evaluate grouping
            
            # Calculate average pairwise distance within group
            positions = [np.array(obj["position"][:2]) for obj in group_objects]
            distances = []
            
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    distances.append(np.linalg.norm(positions[i] - positions[j]))
            
            avg_distance = np.mean(distances) if distances else 0
            
            # Score based on compactness (closer = better, up to reasonable limit)
            optimal_distance = 0.2  # 20cm is ideal group spacing
            score = max(0, 1.0 - abs(avg_distance - optimal_distance) / optimal_distance)
            group_scores.append(score)
        
        return np.mean(group_scores) if group_scores else 1.0
    
    def _check_coverage_efficiency(self, layout: List[Dict], 
                                  surface_size: List[float]) -> float:
        """Check how efficiently the layout uses available table space."""
        if not layout:
            return 0.0
        
        # Calculate total object area
        total_object_area = sum(obj["dimensions"][0] * obj["dimensions"][1] 
                               for obj in layout)
        
        # Table area
        table_area = surface_size[0] * surface_size[1]
        
        # Ideal coverage is 30-60% (not too sparse, not too cluttered)
        coverage_ratio = total_object_area / table_area
        
        if 0.3 <= coverage_ratio <= 0.6:
            return 1.0
        elif coverage_ratio < 0.3:
            return coverage_ratio / 0.3  # Penalty for sparse layout
        else:
            return max(0, 1.0 - (coverage_ratio - 0.6) / 0.4)  # Penalty for clutter
    
    def render_2d_topview(self, layout: List[Dict], surface_size: List[float], 
                         output_path: str = None) -> str:
        """Render 2D top-view of the tabletop layout."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Draw table surface
        table_rect = patches.Rectangle(
            (-surface_size[0]/2, -surface_size[1]/2),
            surface_size[0], surface_size[1],
            linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3
        )
        ax.add_patch(table_rect)
        
        # Draw objects
        colors = plt.cm.Set3(np.linspace(0, 1, len(layout)))
        
        for i, obj in enumerate(layout):
            pos = obj["position"]
            dims = obj["dimensions"]
            
            # Create rectangle for object
            rect = patches.Rectangle(
                (pos[0] - dims[0]/2, pos[1] - dims[1]/2),
                dims[0], dims[1],
                linewidth=1, edgecolor='darkblue', 
                facecolor=colors[i], alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(pos[0], pos[1], obj["name"], 
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Set equal aspect ratio and limits
        ax.set_xlim(-surface_size[0]/2 * 1.2, surface_size[0]/2 * 1.2)
        ax.set_ylim(-surface_size[1]/2 * 1.2, surface_size[1]/2 * 1.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Tabletop Layout - Top View')
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            # Save to temporary file
            temp_path = tempfile.mktemp(suffix='.png')
            plt.savefig(temp_path, dpi=150, bbox_inches='tight')
            plt.close()
            return temp_path
    
    def simulate_mujoco(self, layout: List[Dict], surface_size: List[float], 
                       sim_time: float = 2.0) -> Dict[str, Any]:
        """Run MuJoCo physics simulation with gravity."""
        # Build MuJoCo XML
        xml_content = self._build_mujoco_xml(layout, surface_size)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            xml_path = f.name
        
        try:
            # Load and simulate
            model = mujoco.MjModel.from_xml_path(xml_path)
            data = mujoco.MjData(model)
            
            # Record initial positions
            initial_positions = {}
            for i in range(model.nbody):
                if model.body(i).name and model.body(i).name.startswith('obj_'):
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, model.body(i).name)
                    initial_positions[model.body(i).name] = data.xpos[body_id].copy()
            
            # Simulate
            n_steps = int(sim_time / model.opt.timestep)
            for _ in range(n_steps):
                mujoco.mj_step(model, data)
            
            # Record final positions and check stability
            final_positions = {}
            movements = {}
            
            for name in initial_positions:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                final_positions[name] = data.xpos[body_id].copy()
                movements[name] = np.linalg.norm(
                    final_positions[name] - initial_positions[name]
                )
            
            return {
                "stability_score": self._calculate_stability_score(movements),
                "movements": movements,
                "simulation_success": True
            }
            
        except Exception as e:
            return {
                "stability_score": 0.0,
                "error": str(e),
                "simulation_success": False
            }
        finally:
            # Clean up
            if os.path.exists(xml_path):
                os.unlink(xml_path)
    
    def _build_mujoco_xml(self, layout: List[Dict], surface_size: List[float]) -> str:
        """Build MuJoCo XML for tabletop simulation."""
        xml_parts = [
            '<?xml version="1.0"?>',
            '<mujoco>',
            '  <option timestep="0.005" gravity="0 0 -9.81"/>',
            '  <asset>',
            '    <material name="table_mat" rgba="0.8 0.6 0.4 1"/>',
            '    <material name="object_mat" rgba="0.2 0.4 0.8 1"/>',
            '  </asset>',
            '  <worldbody>',
            '    <!-- Table surface -->',
            f'    <geom name="table" type="box" size="{surface_size[0]/2} {surface_size[1]/2} 0.05"',
            '          pos="0 0 -0.05" material="table_mat"/>',
        ]
        
        # Add objects
        for i, obj in enumerate(layout):
            pos = obj["position"]
            dims = obj["dimensions"]
            name = obj["name"]
            
            xml_parts.extend([
                f'    <body name="obj_{name}_{i}" pos="{pos[0]} {pos[1]} {pos[2]}">',
                f'      <joint type="free"/>',
                f'      <geom name="geom_{name}_{i}" type="box"',
                f'            size="{dims[0]/2} {dims[1]/2} {dims[2]/2}"',
                f'            material="object_mat" mass="0.5"/>',
                f'    </body>',
            ])
        
        xml_parts.extend([
            '  </worldbody>',
            '</mujoco>'
        ])
        
        return '\n'.join(xml_parts)
    
    def _calculate_stability_score(self, movements: Dict[str, float]) -> float:
        """Calculate stability score based on object movements during simulation."""
        if not movements:
            return 1.0
        
        # Objects should move less than 5cm to be considered stable
        stable_threshold = 0.05
        stable_count = sum(1 for movement in movements.values() 
                          if movement < stable_threshold)
        
        return stable_count / len(movements)
    
    def run_benchmark(self, use_mock: bool = True) -> Dict[str, Any]:
        """Run the full tabletop benchmark on all prompts."""
        results = {}
        
        for prompt_data in self.prompts:
            prompt_id = prompt_data["id"]
            surface_size = prompt_data["surface_size"]
            prompt_text = prompt_data["prompt"]
            
            print(f"Processing {prompt_id}: {prompt_data['scene']}")
            
            try:
                # Generate layout (using mock for now)
                if use_mock:
                    layout = self.mock_llm_generate(prompt_text, surface_size)
                else:
                    # TODO: Integrate with actual LLM
                    layout = self.mock_llm_generate(prompt_text, surface_size)
                
                # Evaluate layout
                evaluation = self.evaluate_layout(layout, surface_size, prompt_data)
                
                # Render 2D view
                output_dir = os.path.join(
                    os.path.dirname(__file__), 
                    "../results", "tabletop", prompt_id
                )
                os.makedirs(output_dir, exist_ok=True)
                
                render_path = os.path.join(output_dir, "layout_2d.png")
                self.render_2d_topview(layout, surface_size, render_path)
                
                # Run MuJoCo simulation
                sim_results = self.simulate_mujoco(layout, surface_size)
                
                # Combine results
                results[prompt_id] = {
                    "scene": prompt_data["scene"],
                    "layout": layout,
                    "evaluation": evaluation,
                    "simulation": sim_results,
                    "render_path": render_path
                }
                
            except Exception as e:
                print(f"Error processing {prompt_id}: {e}")
                results[prompt_id] = {
                    "error": str(e),
                    "scene": prompt_data["scene"]
                }
        
        return results

def main():
    """Main function for standalone testing."""
    print("TableTop-Bench: Desktop Object Layout Evaluation")
    print("=" * 50)
    
    # Initialize benchmark
    bench = TabletopBench()
    
    # Run benchmark
    results = bench.run_benchmark(use_mock=True)
    
    # Print summary
    print("\nBenchmark Results Summary:")
    print("-" * 30)
    
    for prompt_id, result in results.items():
        if "error" in result:
            print(f"{prompt_id}: ERROR - {result['error']}")
        else:
            eval_score = result["evaluation"]["overall_score"]
            sim_success = result["simulation"]["simulation_success"]
            stability = result["simulation"].get("stability_score", 0)
            
            print(f"{prompt_id} ({result['scene']}):")
            print(f"  Evaluation Score: {eval_score:.3f}")
            print(f"  Simulation: {'✓' if sim_success else '✗'}")
            print(f"  Stability Score: {stability:.3f}")
    
    # Save results
    output_path = os.path.join(
        os.path.dirname(__file__), 
        "../results", "tabletop_bench_results.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for k, v in results.items():
            json_results[k] = v
            if "layout" in v:
                for obj in v["layout"]:
                    if isinstance(obj.get("position"), np.ndarray):
                        obj["position"] = obj["position"].tolist()
                    if isinstance(obj.get("dimensions"), np.ndarray):
                        obj["dimensions"] = obj["dimensions"].tolist()
        
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()