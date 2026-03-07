"""
ScenePlanner: LLM-Agent that orchestrates the full scene generation pipeline.

Pipeline:
  1. Parse description → structured scene spec
  2. Retrieve 3D assets matching the spec
  3. Plan spatial layout via LLM reasoning
  4. Optimize layout with constraint solver
  5. Iterative refinement with LLM feedback
"""

import json
import logging
from typing import Optional

import yaml

from .prompt_templates import PromptTemplates
from ..layout.constraint_solver import ConstraintSolver
from ..layout.scene_graph import SceneGraph
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ScenePlanner:
    """Main orchestrator for LLM-driven scene generation."""

    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.llm = LLMClient(
            provider=self.config["llm"]["provider"],
            model=self.config["llm"]["model"],
            temperature=self.config["llm"]["temperature"],
            max_tokens=self.config["llm"]["max_tokens"],
        )
        self.prompts = PromptTemplates()
        self.solver = ConstraintSolver(self.config["layout"])
        self.max_refinement_rounds = 3

    def generate(
        self,
        description: str,
        room_dims: tuple[float, float, float] = (5.0, 4.0, 2.8),
        door_position: str = "bottom-center",
        window_positions: Optional[list[str]] = None,
    ) -> dict:
        """
        Generate a complete scene from a text description.

        Args:
            description: Natural language room description
            room_dims: (width, length, height) in meters
            door_position: Door location description
            window_positions: List of window location descriptions

        Returns:
            Complete scene specification dict
        """
        width, length, height = room_dims
        window_positions = window_positions or ["right-center"]

        logger.info(f"Generating scene: {description[:80]}...")

        # Stage 1: Scene Understanding
        logger.info("Stage 1: Understanding scene description...")
        scene_spec = self._understand_scene(description, width, length, height)

        # Stage 2: Asset Retrieval (get sizes for layout planning)
        logger.info("Stage 2: Retrieving assets...")
        objects_with_sizes = self._retrieve_assets(scene_spec)

        # Stage 3: Layout Planning
        logger.info("Stage 3: Planning layout...")
        layout = self._plan_layout(
            objects_with_sizes,
            scene_spec,
            width,
            length,
            door_position,
            window_positions,
        )

        # Stage 4: Constraint Optimization
        logger.info("Stage 4: Optimizing layout...")
        scene_graph = SceneGraph.from_layout(layout, width, length)
        optimized = self.solver.optimize(scene_graph)

        # Stage 5: Iterative Refinement
        logger.info("Stage 5: Refining layout...")
        final_layout = self._refine_layout(optimized, width, length)

        # Assemble final scene
        scene = {
            "description": description,
            "room": {
                "width": width,
                "length": length,
                "height": height,
                "door": door_position,
                "windows": window_positions,
            },
            "spec": scene_spec,
            "objects": final_layout.to_object_list(),
            "scene_graph": final_layout.to_dict(),
        }

        logger.info(
            f"Scene generated with {len(scene['objects'])} objects."
        )
        return scene

    def _understand_scene(
        self, description: str, width: float, length: float, height: float
    ) -> dict:
        """Stage 1: Parse description into structured scene specification."""
        prompt = self.prompts.SCENE_UNDERSTANDING.format(
            description=description,
            width=width,
            length=length,
            height=height,
        )
        response = self.llm.chat(prompt)
        return self._parse_json_response(response)

    def _retrieve_assets(self, scene_spec: dict) -> list[dict]:
        """Stage 2: For each required object, estimate size via LLM."""
        objects_with_sizes = []
        style = scene_spec.get("style", "modern")

        for obj in scene_spec.get("required_objects", []):
            for i in range(obj.get("quantity", 1)):
                prompt = self.prompts.ASSET_QUERY.format(
                    description=obj["description"],
                    style=style,
                    category=obj["category"],
                )
                response = self.llm.chat(prompt)
                asset_info = self._parse_json_response(response)

                objects_with_sizes.append(
                    {
                        "id": f"{obj['category']}_{i}",
                        "category": obj["category"],
                        "description": obj["description"],
                        "size": asset_info.get(
                            "size_estimate",
                            {"width": 0.8, "depth": 0.8, "height": 0.8},
                        ),
                        "priority": obj.get("priority", "must_have"),
                    }
                )

        return objects_with_sizes

    def _plan_layout(
        self,
        objects: list[dict],
        scene_spec: dict,
        width: float,
        length: float,
        door_position: str,
        window_positions: list[str],
    ) -> SceneGraph:
        """Stage 3: Generate initial layout via LLM spatial reasoning."""
        objects_list = "\n".join(
            [
                f"- {obj['id']}: {obj['category']} ({obj['description']}), "
                f"size: {obj['size']['width']:.1f}m x {obj['size']['depth']:.1f}m x {obj['size']['height']:.1f}m"
                for obj in objects
            ]
        )
        spatial_constraints = "\n".join(
            [
                f"- {c}"
                for c in scene_spec.get("spatial_constraints", [])
            ]
        )

        prompt = self.prompts.LAYOUT_PLANNING.format(
            width=width,
            length=length,
            door_position=door_position,
            window_positions=", ".join(window_positions),
            objects_list=objects_list,
            spatial_constraints=spatial_constraints or "None specified",
            collision_margin=self.config["layout"]["collision_margin"],
            wall_margin=self.config["layout"]["wall_margin"],
        )

        response = self.llm.chat(prompt)
        layout_data = self._parse_json_response(response)

        # Build scene graph from LLM output
        scene_graph = SceneGraph(width=width, length=length)
        for placement in layout_data.get("placements", []):
            obj_info = next(
                (o for o in objects if o["id"] == placement["object_id"]),
                None,
            )
            if obj_info:
                scene_graph.add_object(
                    object_id=placement["object_id"],
                    category=obj_info["category"],
                    position=tuple(placement["position"]),
                    rotation=placement.get("rotation", 0),
                    size=obj_info["size"],
                )

        return scene_graph

    def _refine_layout(
        self, scene_graph: SceneGraph, width: float, length: float
    ) -> SceneGraph:
        """Stage 5: Iterative LLM-based refinement."""
        for round_idx in range(self.max_refinement_rounds):
            violations = scene_graph.check_violations()
            if not violations:
                logger.info(
                    f"No violations after round {round_idx}. Layout is valid."
                )
                break

            logger.info(
                f"Refinement round {round_idx + 1}: "
                f"{len(violations)} violations found."
            )

            prompt = self.prompts.LAYOUT_REFINEMENT.format(
                current_layout=json.dumps(
                    scene_graph.to_dict(), indent=2
                ),
                violations=json.dumps(violations, indent=2),
            )
            response = self.llm.chat(prompt)
            adjustments = self._parse_json_response(response)

            for adj in adjustments.get("adjustments", []):
                scene_graph.update_object(
                    adj["object_id"],
                    position=tuple(adj["new_position"]),
                    rotation=adj.get("new_rotation"),
                )

        return scene_graph

    @staticmethod
    def _parse_json_response(response: str) -> dict:
        """Extract JSON from LLM response (handles markdown code blocks)."""
        # Try to find JSON in code blocks
        if "```json" in response:
            start = response.index("```json") + 7
            end = response.index("```", start)
            json_str = response[start:end].strip()
        elif "```" in response:
            start = response.index("```") + 3
            end = response.index("```", start)
            json_str = response[start:end].strip()
        else:
            json_str = response.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from LLM response.")
            return {}
