#!/usr/bin/env python3
"""
PhysScene-Bench Report Generator
Generates comprehensive evaluation reports with visualizations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

class PhysSceneReportGenerator:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir
        
    def load_results(self) -> Dict[str, Any]:
        """Load all evaluation results"""
        summary_file = self.results_dir / "benchmark_summary.json"
        
        if not summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_file}")
        
        with open(summary_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def create_model_comparison_table(self, results: List[Dict]) -> str:
        """Create markdown comparison table"""
        # Aggregate results by model
        model_stats = {}
        
        for result in results:
            if not result["success"]:
                continue
                
            model = result["model"].split("/")[-1].split(":")[0]  # Clean model name
            
            if model not in model_stats:
                model_stats[model] = {
                    "results": [],
                    "semantic_scores": [],
                    "physical_scores": [],
                    "functional_scores": [],
                    "overall_scores": []
                }
            
            eval_data = result["evaluation"]
            model_stats[model]["results"].append(result)
            model_stats[model]["semantic_scores"].append(eval_data["layer_scores"]["semantic_fidelity"])
            model_stats[model]["physical_scores"].append(eval_data["layer_scores"]["physical_plausibility"])
            model_stats[model]["functional_scores"].append(eval_data["layer_scores"]["functional_affordance"])
            model_stats[model]["overall_scores"].append(eval_data["overall_score"])
        
        # Create table
        table_lines = [
            "# PhysScene-Bench Results",
            "",
            "## Model Performance Comparison",
            "",
            "| Model | Overall | Semantic | Physical | Functional | Success Rate |",
            "|-------|---------|----------|----------|------------|--------------|"
        ]
        
        # Sort models by overall score
        sorted_models = sorted(model_stats.items(), 
                             key=lambda x: np.mean(x[1]["overall_scores"]), 
                             reverse=True)
        
        for model, stats in sorted_models:
            overall_avg = np.mean(stats["overall_scores"])
            semantic_avg = np.mean(stats["semantic_scores"])
            physical_avg = np.mean(stats["physical_scores"])
            functional_avg = np.mean(stats["functional_scores"])
            success_rate = len(stats["results"]) / 10  # 10 total prompts
            
            table_lines.append(
                f"| {model} | {overall_avg:.3f} | {semantic_avg:.3f} | "
                f"{physical_avg:.3f} | {functional_avg:.3f} | {success_rate:.1%} |"
            )
        
        table_lines.extend([
            "",
            "## Score Breakdown",
            "",
            "- **Overall**: Weighted average of all three layers (30% semantic, 40% physical, 30% functional)",
            "- **Semantic**: How well the layout matches expected object categories and counts", 
            "- **Physical**: Collision-free placement, room boundaries, and physics stability",
            "- **Functional**: Spatial relationships, facing directions, and walkability",
            "- **Success Rate**: Percentage of prompts successfully completed",
            ""
        ])
        
        return "\n".join(table_lines)
    
    def create_radar_charts(self, results: List[Dict]):
        """Create radar charts for model performance"""
        # Aggregate results by model
        model_stats = {}
        
        for result in results:
            if not result["success"]:
                continue
                
            model = result["model"].split("/")[-1].split(":")[0]  # Clean model name
            
            if model not in model_stats:
                model_stats[model] = {
                    "semantic": [],
                    "physical": [], 
                    "functional": []
                }
            
            eval_data = result["evaluation"]
            model_stats[model]["semantic"].append(eval_data["layer_scores"]["semantic_fidelity"])
            model_stats[model]["physical"].append(eval_data["layer_scores"]["physical_plausibility"])
            model_stats[model]["functional"].append(eval_data["layer_scores"]["functional_affordance"])
        
        # Create radar chart
        categories = ['Semantic\nFidelity', 'Physical\nPlausibility', 'Functional\nAffordance']
        n_cats = len(categories)
        
        # Calculate angles for radar chart
        angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
        angles += angles[:1]  # Complete the circle
        
        # Set up the plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_stats)))
        
        for idx, (model, stats) in enumerate(model_stats.items()):
            if idx >= 6:  # Limit to 6 models for layout
                break
                
            ax = axes[idx]
            
            # Calculate average scores
            values = [
                np.mean(stats["semantic"]),
                np.mean(stats["physical"]),
                np.mean(stats["functional"])
            ]
            values += values[:1]  # Complete the circle
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])
            ax.set_ylim(0, 1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_title(model, size=12, fontweight='bold', pad=20)
            ax.grid(True)
            
            # Add value labels
            for angle, value in zip(angles[:-1], values[:-1]):
                ax.text(angle, value + 0.05, f'{value:.2f}', ha='center', va='center')
        
        # Hide unused subplots
        for idx in range(len(model_stats), 6):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('PhysScene-Bench Model Performance by Layer', size=16, y=0.98)
        
        # Save chart
        chart_file = self.output_dir / "model_performance_radar.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_file
    
    def create_detailed_analysis(self, results: List[Dict]) -> str:
        """Create detailed analysis section"""
        # Success/failure analysis
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        analysis = [
            "## Detailed Analysis",
            "",
            f"### Success Rate: {len(successful)}/{len(results)} ({len(successful)/len(results):.1%})",
            ""
        ]
        
        if failed:
            analysis.extend([
                "#### Failed Evaluations:",
                ""
            ])
            for fail in failed:
                analysis.append(f"- **{fail['model']}** on {fail['prompt_id']}: {fail.get('error', 'Unknown error')}")
            analysis.append("")
        
        # Score distribution analysis
        if successful:
            overall_scores = [r["evaluation"]["overall_score"] for r in successful]
            semantic_scores = [r["evaluation"]["layer_scores"]["semantic_fidelity"] for r in successful]
            physical_scores = [r["evaluation"]["layer_scores"]["physical_plausibility"] for r in successful]
            functional_scores = [r["evaluation"]["layer_scores"]["functional_affordance"] for r in successful]
            
            analysis.extend([
                "### Score Distribution:",
                "",
                f"- **Overall**: Mean={np.mean(overall_scores):.3f}, Std={np.std(overall_scores):.3f}",
                f"- **Semantic**: Mean={np.mean(semantic_scores):.3f}, Std={np.std(semantic_scores):.3f}",
                f"- **Physical**: Mean={np.mean(physical_scores):.3f}, Std={np.std(physical_scores):.3f}",
                f"- **Functional**: Mean={np.mean(functional_scores):.3f}, Std={np.std(functional_scores):.3f}",
                ""
            ])
        
        # Common failure modes
        analysis.extend([
            "### Key Findings:",
            "",
            "1. **Semantic Layer**: Most models successfully identify required furniture categories",
            "2. **Physical Layer**: Collision avoidance and boundary constraints are challenging", 
            "3. **Functional Layer**: Spatial relationships and walkability vary significantly",
            "4. **Model Performance**: Free-tier models show varying capabilities in spatial reasoning",
            ""
        ])
        
        return "\n".join(analysis)
    
    def create_score_heatmap(self, results: List[Dict]):
        """Create heatmap of scores across prompts and models"""
        # Prepare data for heatmap
        models = []
        prompts = []
        scores = []
        
        for result in results:
            if not result["success"]:
                continue
                
            model = result["model"].split("/")[-1].split(":")[0]
            prompt = result["prompt_id"]
            score = result["evaluation"]["overall_score"]
            
            models.append(model)
            prompts.append(prompt)
            scores.append(score)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Model': models,
            'Prompt': prompts, 
            'Score': scores
        })
        
        # Pivot for heatmap
        heatmap_data = df.pivot(index='Model', columns='Prompt', values='Score')
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   vmin=0, vmax=1, cbar_kws={'label': 'Overall Score'})
        plt.title('PhysScene-Bench: Overall Scores by Model and Prompt')
        plt.xlabel('Prompt ID')
        plt.ylabel('Model')
        plt.tight_layout()
        
        # Save heatmap
        heatmap_file = self.output_dir / "score_heatmap.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return heatmap_file
    
    def visualize_scene_layouts(self, results: List[Dict], max_examples: int = 6):
        """Create layout visualizations for top-scoring examples"""
        # Sort by score and take top examples
        successful_results = [r for r in results if r["success"]]
        top_results = sorted(successful_results, 
                           key=lambda x: x["evaluation"]["overall_score"], 
                           reverse=True)[:max_examples]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, result in enumerate(top_results):
            if idx >= 6:
                break
                
            ax = axes[idx]
            
            # Get layout data
            layout = result["layout"]
            room_size = result["metadata"]["room_size"]
            model = result["model"].split("/")[-1].split(":")[0]
            score = result["evaluation"]["overall_score"]
            
            # Draw room boundaries
            room_rect = Rectangle((0, 0), room_size[0], room_size[1], 
                                linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
            ax.add_patch(room_rect)
            
            # Color map for object categories
            categories = set(obj["category"] for obj in layout["objects"])
            colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
            category_colors = dict(zip(categories, colors))
            
            # Draw objects
            for obj in layout["objects"]:
                pos = obj["position"]
                dims = obj["dimensions"]
                rotation = obj.get("rotation", 0)
                
                # Create rectangle (simplified, no rotation visualization)
                rect = Rectangle(
                    (pos[0] - dims["width"]/2, pos[1] - dims["depth"]/2),
                    dims["width"], dims["depth"],
                    linewidth=1, edgecolor='black',
                    facecolor=category_colors[obj["category"]], alpha=0.7
                )
                ax.add_patch(rect)
                
                # Add label
                ax.text(pos[0], pos[1], obj["category"][:4], 
                       ha='center', va='center', fontsize=8, fontweight='bold')
            
            ax.set_xlim(-0.5, room_size[0] + 0.5)
            ax.set_ylim(-0.5, room_size[1] + 0.5)
            ax.set_aspect('equal')
            ax.set_title(f'{model}\nScore: {score:.3f}', fontsize=10)
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
        
        # Hide unused subplots
        for idx in range(len(top_results), 6):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Top-Scoring Scene Layouts', size=16, y=0.98)
        
        # Save layout visualization
        layout_file = self.output_dir / "top_layouts.png"
        plt.savefig(layout_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return layout_file
    
    def generate_report(self):
        """Generate complete evaluation report"""
        print("Loading results...")
        data = self.load_results()
        results = data["results"]
        
        print("Creating model comparison table...")
        table_markdown = self.create_model_comparison_table(results)
        
        print("Creating detailed analysis...")
        analysis_markdown = self.create_detailed_analysis(results)
        
        print("Generating radar charts...")
        radar_file = self.create_radar_charts(results)
        
        print("Creating score heatmap...")
        heatmap_file = self.create_score_heatmap(results)
        
        print("Visualizing top layouts...")
        layout_file = self.visualize_scene_layouts(results)
        
        # Combine all content
        full_report = [
            table_markdown,
            "",
            analysis_markdown,
            "",
            "## Visualizations",
            "",
            f"### Model Performance by Evaluation Layer",
            f"![Model Performance Radar Charts]({radar_file.name})",
            "",
            f"### Score Heatmap Across All Evaluations", 
            f"![Score Heatmap]({heatmap_file.name})",
            "",
            f"### Top-Scoring Layout Examples",
            f"![Top Layout Visualizations]({layout_file.name})",
            "",
            "---",
            "",
            f"**Report generated from {len(results)} evaluation runs**",
            f"**Success rate: {len([r for r in results if r['success']])}/{len(results)} "
            f"({len([r for r in results if r['success']])/len(results):.1%})**"
        ]
        
        # Save report
        report_file = self.output_dir / "evaluation_report.md"
        with open(report_file, 'w') as f:
            f.write("\n".join(full_report))
        
        print(f"Report generated: {report_file}")
        print(f"Visualizations saved:")
        print(f"  - {radar_file}")
        print(f"  - {heatmap_file}")
        print(f"  - {layout_file}")
        
        return report_file

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate PhysScene-Bench evaluation report")
    parser.add_argument("--results", default="../results", help="Results directory")
    
    args = parser.parse_args()
    
    generator = PhysSceneReportGenerator(args.results)
    generator.generate_report()

if __name__ == "__main__":
    main()