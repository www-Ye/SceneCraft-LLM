"""
Prompt templates for LLM-based scene planning.

The planner operates in three stages:
  1. Scene Understanding: Parse user description into structured requirements
  2. Layout Planning: Generate object placements with spatial reasoning
  3. Refinement: Iteratively fix constraint violations
"""

from dataclasses import dataclass


@dataclass
class PromptTemplates:
    """Collection of prompt templates for the scene planning pipeline."""

    # Stage 1: Scene Understanding
    SCENE_UNDERSTANDING = """You are an expert interior designer and spatial reasoner.
Given a natural language description of a room, extract structured information.

## Input Description
{description}

## Room Dimensions
- Width: {width}m, Length: {length}m, Height: {height}m

## Task
Extract the following information in JSON format:

```json
{{
  "room_type": "<room category>",
  "style": "<design style, e.g., modern, traditional, minimalist>",
  "required_objects": [
    {{
      "category": "<furniture category>",
      "description": "<specific description>",
      "quantity": <number>,
      "priority": "<must_have | nice_to_have>"
    }}
  ],
  "spatial_constraints": [
    "<e.g., sofa should face TV>",
    "<e.g., desk near window>"
  ],
  "functional_zones": [
    {{
      "name": "<zone name>",
      "objects": ["<object categories in this zone>"],
      "approximate_area_ratio": <0.0-1.0>
    }}
  ],
  "atmosphere": "<overall mood/feeling>"
}}
```

Be thorough but realistic. Only include objects that make sense for the room type and description."""

    # Stage 2: Layout Planning
    LAYOUT_PLANNING = """You are a spatial layout planner for 3D indoor scenes.
Given room dimensions and a list of objects with their sizes, plan their placement.

## Room
- Dimensions: {width}m x {length}m (floor plan, origin at bottom-left corner)
- Walls: bottom (y=0), top (y={length}), left (x=0), right (x={width})
- Door position: {door_position}
- Window positions: {window_positions}

## Objects to Place
{objects_list}

## Spatial Constraints
{spatial_constraints}

## Rules
1. No overlapping objects (maintain at least {collision_margin}m gap)
2. Objects must be within room boundaries (at least {wall_margin}m from walls)
3. Maintain walkable paths (at least 0.6m wide)
4. Respect functional groupings (e.g., dining table with chairs)
5. Consider facing directions (e.g., sofa faces TV, desk faces away from door)
6. Heavy/large furniture against walls when possible

## Output Format
For each object, provide placement in JSON:

```json
{{
  "placements": [
    {{
      "object_id": "<id>",
      "category": "<category>",
      "position": [<x>, <y>],
      "rotation": <degrees, 0-360>,
      "facing": "<direction the front faces: north/south/east/west>",
      "reasoning": "<brief explanation of why placed here>"
    }}
  ]
}}
```

Think step by step. First identify functional zones, then place anchor objects, then fill in secondary objects."""

    # Stage 3: Refinement
    LAYOUT_REFINEMENT = """You are a layout quality inspector for 3D indoor scenes.

## Current Layout
{current_layout}

## Detected Issues
{violations}

## Task
Fix the layout issues while maintaining the overall design intent.
For each problematic object, suggest a new position and rotation.

Output only the objects that need to be moved:

```json
{{
  "adjustments": [
    {{
      "object_id": "<id>",
      "old_position": [<x>, <y>],
      "new_position": [<x>, <y>],
      "new_rotation": <degrees>,
      "fix_reason": "<what issue this fixes>"
    }}
  ]
}}
```

Make minimal adjustments — move objects as little as possible to resolve issues."""

    # Asset Retrieval Query
    ASSET_QUERY = """Given the following furniture description, generate a search query
for retrieving a matching 3D model from a furniture database.

Description: {description}
Room style: {style}
Category: {category}

Output a JSON with:
```json
{{
  "search_query": "<concise search query>",
  "style_keywords": ["<style descriptors>"],
  "size_estimate": {{
    "width": <meters>,
    "depth": <meters>,
    "height": <meters>
  }},
  "material_preferences": ["<preferred materials>"]
}}
```"""
