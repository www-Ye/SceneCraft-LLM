#!/usr/bin/env python3
"""
Full Benchmark Runner - Integrates all evaluation dimensions.
Combines rule-based evaluation, robot navigation, and VLM judging.
"""

import json
import os
import sys
import argparse
import time
from typing import Dict, List, Any, Optional
import numpy as np

# Import local modules
try:
    from evaluator import PhysSceneEvaluator
    from tabletop_bench import TabletopBench
    from robot_eval import RobotNavigator
    from vlm_judge import vlm_judge
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

class FullBenchmarkRunner:
    """Runs complete evaluation suite across all dimensions."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with configuration."""
        self.config = config or self._default_config()
        
        # Initialize evaluators
        self.rule_evaluator = PhysSceneEvaluator()
        self.tabletop_bench = TabletopBench()
        # VLM judge is just a function, not a class
        
        print("Full Benchmark Runner Initialized")
        print(f"Configuration: {self.config}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            "use_rule_based": True,
            "use_robot_nav": True,
            "use_tabletop": True,
            "use_vlm": True,
            "robot_nav_trials": 10,
            "output_dir": "../results",
            "save_intermediate": True,
            "verbose": True
        }
    
    def evaluate_room_layout(self, layout: List[Dict], room_size: List[float], 
                            prompt_data: Dict) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a room layout.
        
        Args:
            layout: List of furniture objects
            room_size: [width, height] of room
            prompt_data: Original prompt information
        
        Returns:
            Combined evaluation results
        """
        results = {
            "layout_id": prompt_data.get("id", "unknown"),
            "room_type": prompt_data.get("room_type", "unknown"),
            "timestamp": time.time()
        }
        
        print(f"\nEvaluating layout {results['layout_id']}: {results['room_type']}")
        print("-" * 50)
        
        # 1. Rule-based evaluation
        if self.config["use_rule_based"]:
            print("Running rule-based evaluation...")
            try:
                expected_objects = prompt_data.get("expected_objects", [])
                functional_checks = prompt_data.get("functional_checks", [])
                rule_results = self.rule_evaluator.evaluate_layout(
                    {"objects": layout}, room_size, expected_objects, functional_checks
                )
                results["rule_based"] = rule_results
                
                if self.config["verbose"]:
                    print(f"  Overall score: {rule_results.get('overall_score', 0):.3f}")
                    print(f"  Physics score: {rule_results.get('physics_score', 0):.3f}")
                    print(f"  Layout score: {rule_results.get('layout_score', 0):.3f}")
                    
            except Exception as e:
                print(f"  Error in rule-based evaluation: {e}")
                results["rule_based"] = {"error": str(e)}
        
        # 2. Robot navigation evaluation
        if self.config["use_robot_nav"]:
            print("Running robot navigation evaluation...")
            try:
                navigator = RobotNavigator(layout, room_size)
                nav_results = navigator.evaluate_navigability(
                    n_trials=self.config["robot_nav_trials"]
                )
                results["robot_nav"] = nav_results
                navigator.cleanup()
                
                if self.config["verbose"]:
                    print(f"  Success rate: {nav_results.get('success_rate', 0):.3f}")
                    print(f"  Avg efficiency: {nav_results.get('avg_path_efficiency', 0):.3f}")
                    print(f"  Avg clearance: {nav_results.get('avg_clearance', 0):.3f}m")
                    
            except Exception as e:
                print(f"  Error in robot navigation evaluation: {e}")
                results["robot_nav"] = {"error": str(e)}
        
        # 3. VLM Judge evaluation  
        if self.config["use_vlm"]:
            print("Running VLM judge evaluation...")
            try:
                # First render the layout for VLM
                from render_layout import render_floorplan
                
                output_dir = os.path.join(self.config["output_dir"], "temp")
                os.makedirs(output_dir, exist_ok=True)
                
                render_path = os.path.join(output_dir, f"{results['layout_id']}_layout.png")
                render_floorplan(layout, room_size, render_path)
                
                vlm_results = vlm_judge(render_path, prompt_data)
                results["vlm_judge"] = vlm_results
                
                if self.config["verbose"]:
                    print(f"  VLM score: {vlm_results.get('overall_score', 0):.3f}")
                    
            except Exception as e:
                print(f"  Error in VLM judge evaluation: {e}")
                results["vlm_judge"] = {"error": str(e)}
        
        # Calculate combined score
        results["combined_score"] = self._calculate_combined_score(results)
        
        return results
    
    def evaluate_tabletop_scene(self, prompt_data: Dict) -> Dict[str, Any]:
        """
        Evaluate a tabletop scene.
        
        Args:
            prompt_data: Tabletop prompt information
        
        Returns:
            Tabletop evaluation results
        """
        if not self.config["use_tabletop"]:
            return {"skipped": True}
        
        print(f"\nEvaluating tabletop scene {prompt_data['id']}: {prompt_data['scene']}")
        print("-" * 50)
        
        try:
            # Generate layout (using mock for now)
            layout = self.tabletop_bench.mock_llm_generate(
                prompt_data["prompt"], 
                prompt_data["surface_size"]
            )
            
            # Evaluate
            evaluation = self.tabletop_bench.evaluate_layout(
                layout, 
                prompt_data["surface_size"], 
                prompt_data
            )
            
            # Run simulation
            sim_results = self.tabletop_bench.simulate_mujoco(
                layout, 
                prompt_data["surface_size"]
            )
            
            results = {
                "tabletop_id": prompt_data["id"],
                "scene": prompt_data["scene"],
                "layout": layout,
                "evaluation": evaluation,
                "simulation": sim_results,
                "timestamp": time.time()
            }
            
            if self.config["verbose"]:
                print(f"  Overall score: {evaluation.get('overall_score', 0):.3f}")
                print(f"  Stability: {sim_results.get('stability_score', 0):.3f}")
            
            return results
            
        except Exception as e:
            print(f"Error evaluating tabletop scene: {e}")
            return {"error": str(e)}
    
    def _calculate_combined_score(self, results: Dict) -> float:
        """Calculate weighted combination of all evaluation scores."""
        weights = {
            "rule_based": 0.4,
            "robot_nav": 0.3,
            "vlm_judge": 0.3
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        # Rule-based score
        if "rule_based" in results and "error" not in results["rule_based"]:
            score = results["rule_based"].get("overall_score", 0)
            total_score += score * weights["rule_based"]
            total_weight += weights["rule_based"]
        
        # Robot navigation score (combine metrics)
        if "robot_nav" in results and "error" not in results["robot_nav"]:
            nav = results["robot_nav"]
            # Weighted combination of navigation metrics
            nav_score = (
                nav.get("success_rate", 0) * 0.4 +
                nav.get("avg_path_efficiency", 0) * 0.3 +
                min(1.0, nav.get("avg_clearance", 0) / 0.5) * 0.2 +
                max(0, 1.0 - nav.get("avg_collisions", 0) / 5.0) * 0.1
            )
            total_score += nav_score * weights["robot_nav"]
            total_weight += weights["robot_nav"]
        
        # VLM judge score
        if "vlm_judge" in results and "error" not in results["vlm_judge"]:
            score = results["vlm_judge"].get("overall_score", 0)
            total_score += score * weights["vlm_judge"]
            total_weight += weights["vlm_judge"]
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def run_room_benchmark(self, prompts_path: str = None) -> Dict[str, Any]:
        """Run benchmark on room layout prompts."""
        if prompts_path is None:
            prompts_path = os.path.join(
                os.path.dirname(__file__),
                "../prompts/layout_prompts.json"
            )
        
        print(f"Loading room prompts from: {prompts_path}")
        
        try:
            with open(prompts_path, 'r') as f:
                prompts = json.load(f)
        except FileNotFoundError:
            print(f"Error: Prompts file not found: {prompts_path}")
            return {"error": "prompts_not_found"}
        
        results = {"room_layouts": {}, "summary": {}}
        
        for prompt_data in prompts:
            # For this demo, we'll use mock layout generation
            # In real implementation, this would call your layout generator
            layout = self._generate_mock_layout(prompt_data)
            room_size = prompt_data["room_size"]
            
            # Evaluate layout
            layout_results = self.evaluate_room_layout(layout, room_size, prompt_data)
            results["room_layouts"][prompt_data["id"]] = layout_results
            
            # Save intermediate results if requested
            if self.config["save_intermediate"]:
                self._save_intermediate_results(prompt_data["id"], layout_results)
        
        # Generate summary
        results["summary"] = self._generate_summary(results["room_layouts"])
        
        return results
    
    def run_tabletop_benchmark(self) -> Dict[str, Any]:
        """Run benchmark on tabletop prompts."""
        if not self.config["use_tabletop"]:
            return {"skipped": True}
        
        tabletop_results = self.tabletop_bench.run_benchmark(use_mock=True)
        
        # Convert to our format
        converted_results = {}
        for prompt_id, result in tabletop_results.items():
            if "error" not in result:
                converted_results[prompt_id] = self.evaluate_tabletop_scene({
                    "id": prompt_id,
                    "scene": result["scene"],
                    "surface_size": [0.8, 0.6]  # Default size
                })
        
        return {"tabletop_scenes": converted_results}
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("=" * 60)
        print("PHYSSCENE-BENCH: FULL EVALUATION SUITE")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run room layout benchmark
        print("\n1. ROOM LAYOUT BENCHMARK")
        print("=" * 30)
        room_results = self.run_room_benchmark()
        
        # Run tabletop benchmark
        print("\n2. TABLETOP BENCHMARK")
        print("=" * 30)
        tabletop_results = self.run_tabletop_benchmark()
        
        # Combine results
        full_results = {
            "benchmark_version": "1.0",
            "timestamp": time.time(),
            "duration": time.time() - start_time,
            "config": self.config,
            "room_benchmark": room_results,
            "tabletop_benchmark": tabletop_results
        }
        
        # Save full results
        output_path = self._save_full_results(full_results)
        
        print(f"\n" + "=" * 60)
        print("BENCHMARK COMPLETED")
        print(f"Results saved to: {output_path}")
        print(f"Total duration: {full_results['duration']:.1f} seconds")
        print("=" * 60)
        
        return full_results
    
    def _generate_mock_layout(self, prompt_data: Dict) -> List[Dict]:
        """Generate mock layout for testing purposes."""
        # This is a simple mock - in real use, integrate with your layout generator
        room_size = prompt_data["room_size"]
        expected_objects = prompt_data.get("expected_objects", [])
        
        # Simple grid placement
        layout = []
        n_objects = len(expected_objects)
        
        if n_objects == 0:
            return layout
        
        # Grid dimensions
        cols = int(np.sqrt(n_objects)) + 1
        rows = (n_objects + cols - 1) // cols
        
        cell_w = room_size[0] / cols
        cell_h = room_size[1] / rows
        
        for i, obj_name in enumerate(expected_objects):
            row = i // cols
            col = i % cols
            
            # Position in room (centered at origin)
            x = (col + 0.5) * cell_w - room_size[0] / 2
            y = (row + 0.5) * cell_h - room_size[1] / 2
            
            # Get standard dimensions
            dims = self.rule_evaluator.standard_dims.get(obj_name, {
                "width": 0.6, "depth": 0.6, "height": 0.8
            })
            
            layout.append({
                "name": obj_name,
                "category": obj_name,  # Add category field for evaluator
                "position": [x, y, dims["height"] / 2],
                "dimensions": [dims["width"], dims["depth"], dims["height"]],
                "rotation": 0
            })
        
        return layout
    
    def _save_intermediate_results(self, layout_id: str, results: Dict):
        """Save intermediate results for a single layout."""
        output_dir = os.path.join(self.config["output_dir"], "intermediate")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{layout_id}_results.json")
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _save_full_results(self, results: Dict) -> str:
        """Save complete benchmark results."""
        output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"full_benchmark_{timestamp}.json")
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return output_path
    
    def _generate_summary(self, room_results: Dict) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            "total_layouts": len(room_results),
            "successful_evaluations": 0,
            "avg_combined_score": 0.0,
            "avg_rule_based_score": 0.0,
            "avg_robot_nav_success_rate": 0.0,
            "avg_vlm_score": 0.0
        }
        
        valid_results = [r for r in room_results.values() if "error" not in r]
        summary["successful_evaluations"] = len(valid_results)
        
        if valid_results:
            # Combined scores
            combined_scores = [r.get("combined_score", 0) for r in valid_results]
            summary["avg_combined_score"] = np.mean(combined_scores)
            
            # Rule-based scores
            rule_scores = [r["rule_based"].get("overall_score", 0) 
                          for r in valid_results 
                          if "rule_based" in r and "error" not in r["rule_based"]]
            summary["avg_rule_based_score"] = np.mean(rule_scores) if rule_scores else 0.0
            
            # Robot nav success rates
            nav_rates = [r["robot_nav"].get("success_rate", 0)
                        for r in valid_results
                        if "robot_nav" in r and "error" not in r["robot_nav"]]
            summary["avg_robot_nav_success_rate"] = np.mean(nav_rates) if nav_rates else 0.0
            
            # VLM scores
            vlm_scores = [r["vlm_judge"].get("overall_score", 0)
                         for r in valid_results
                         if "vlm_judge" in r and "error" not in r["vlm_judge"]]
            summary["avg_vlm_score"] = np.mean(vlm_scores) if vlm_scores else 0.0
        
        return summary

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Run PhysScene-Bench full evaluation suite")
    
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--output-dir", type=str, default="../results", 
                       help="Output directory for results")
    parser.add_argument("--skip-rule", action="store_true", 
                       help="Skip rule-based evaluation")
    parser.add_argument("--skip-robot", action="store_true",
                       help="Skip robot navigation evaluation") 
    parser.add_argument("--skip-tabletop", action="store_true",
                       help="Skip tabletop evaluation")
    parser.add_argument("--skip-vlm", action="store_true",
                       help="Skip VLM judge evaluation")
    parser.add_argument("--robot-trials", type=int, default=10,
                       help="Number of robot navigation trials")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override config with command line arguments
    config.update({
        "output_dir": args.output_dir,
        "use_rule_based": not args.skip_rule,
        "use_robot_nav": not args.skip_robot,
        "use_tabletop": not args.skip_tabletop,
        "use_vlm": not args.skip_vlm,
        "robot_nav_trials": args.robot_trials,
        "verbose": args.verbose,
        "save_intermediate": True  # Default to True
    })
    
    # Run benchmark
    runner = FullBenchmarkRunner(config)
    results = runner.run_full_benchmark()
    
    # Print summary
    if "room_benchmark" in results and "summary" in results["room_benchmark"]:
        summary = results["room_benchmark"]["summary"]
        print(f"\nFINAL SUMMARY:")
        print(f"  Total layouts evaluated: {summary['total_layouts']}")
        print(f"  Successful evaluations: {summary['successful_evaluations']}")
        print(f"  Average combined score: {summary['avg_combined_score']:.3f}")
        print(f"  Average rule-based score: {summary['avg_rule_based_score']:.3f}")
        print(f"  Average robot nav success: {summary['avg_robot_nav_success_rate']:.3f}")
        print(f"  Average VLM score: {summary['avg_vlm_score']:.3f}")

if __name__ == "__main__":
    main()