"""
Scene Graph Builder for SceneCraft-LLM.

Builds a structured scene graph with:
  - Nodes: furniture objects with properties (category, size, position, etc.)
  - Edges: spatial/semantic relationships between objects
    - Directional: left_of, right_of, in_front_of, behind
    - Proximity: near, adjacent_to
    - Support: on_top_of, supported_by
    - Facing: facing, facing_away
    - Containment: inside, surrounding
    - Functional: used_with, paired_with

The scene graph drives layout optimization and enables
natural language querying of spatial relationships.
"""

import json
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum


class RelationType(str, Enum):
    """Spatial relationship types between objects."""
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    NEAR = "near"
    ADJACENT_TO = "adjacent_to"
    ON_TOP_OF = "on_top_of"
    SUPPORTED_BY = "supported_by"
    FACING = "facing"
    FACING_AWAY = "facing_away"
    ALIGNED_WITH = "aligned_with"
    CENTER_OF = "center_of"
    AGAINST_WALL = "against_wall"


@dataclass
class SceneNode:
    """A node in the scene graph representing a furniture object."""
    id: str
    category: str
    position: List[float]  # [x, y, z]
    rotation: float  # degrees
    dimensions: Dict[str, float]  # width, depth, height
    properties: Dict = field(default_factory=dict)
    # Additional metadata
    wall_aligned: bool = False
    floor_object: bool = True  # vs. wall-mounted, ceiling, etc.
    functional_zone: str = ""  # e.g., "seating", "work", "storage", "entertainment"


@dataclass 
class SceneEdge:
    """An edge in the scene graph representing a relationship."""
    source: str  # source node id
    target: str  # target node id
    relation: str  # RelationType value
    weight: float = 1.0  # relationship strength/importance
    distance: float = 0.0  # actual distance between objects
    properties: Dict = field(default_factory=dict)


@dataclass
class SceneGraphData:
    """Complete scene graph with nodes and edges."""
    room: Dict  # room dimensions and properties
    nodes: List[SceneNode] = field(default_factory=list)
    edges: List[SceneEdge] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


# Functional zone definitions
FUNCTIONAL_ZONES = {
    'seating': ['sofa', 'armchair', 'chair', 'stool', 'bench', 'rocking_chair'],
    'entertainment': ['television_set', 'tv', 'speaker'],
    'work': ['desk', 'office_chair', 'bookshelf'],
    'dining': ['dining_table', 'chair', 'stool'],
    'sleeping': ['bed', 'nightstand', 'dresser'],
    'storage': ['cabinet', 'wardrobe', 'dresser', 'bookshelf', 'shelf'],
    'lighting': ['lamp', 'table_lamp', 'floor_lamp'],
    'surface': ['coffee_table', 'side_table', 'end_table'],
    'floor_covering': ['runner_(carpet)', 'rug', 'carpet', 'mat'],
}

# Semantic pairing rules: objects that should be near each other
SEMANTIC_PAIRS = [
    ('sofa', 'coffee_table', 1.2),
    ('sofa', 'television_set', 3.5),
    ('sofa', 'table_lamp', 1.5),
    ('bed', 'nightstand', 0.5),
    ('desk', 'chair', 0.8),
    ('dining_table', 'chair', 0.8),
    ('armchair', 'coffee_table', 1.2),
    ('armchair', 'table_lamp', 1.2),
]

# Wall alignment categories
WALL_ALIGNED = ['sofa', 'cabinet', 'wardrobe', 'dresser', 'television_set', 'bookshelf', 'bed']


def get_functional_zone(category: str) -> str:
    """Get the functional zone for a furniture category."""
    for zone, cats in FUNCTIONAL_ZONES.items():
        if category in cats:
            return zone
    return "other"


class SceneGraphBuilder:
    """
    Build a structured scene graph from furniture objects.
    Infers spatial relationships and semantic connections.
    """
    
    def __init__(self, room_width: float, room_depth: float, room_height: float = 2.8):
        self.room_width = room_width
        self.room_depth = room_depth
        self.room_height = room_height
        self.graph = SceneGraphData(
            room={
                'width': room_width,
                'depth': room_depth,
                'height': room_height,
                'area': room_width * room_depth,
            }
        )
    
    def add_object(self, obj: Dict) -> SceneNode:
        """Add a furniture object as a node in the scene graph."""
        cat = obj['category']
        node = SceneNode(
            id=obj.get('id', f"{cat}_{len(self.graph.nodes)}"),
            category=cat,
            position=obj['position'],
            rotation=obj.get('rotation', 0.0),
            dimensions=obj['dimensions'],
            properties=obj.get('properties', {}),
            wall_aligned=cat in WALL_ALIGNED,
            functional_zone=get_functional_zone(cat),
        )
        self.graph.nodes.append(node)
        return node
    
    def build_relationships(self):
        """Infer all spatial relationships between objects."""
        nodes = self.graph.nodes
        self.graph.edges = []  # Reset edges
        
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i >= j:
                    continue
                
                # Compute spatial metrics
                dx = b.position[0] - a.position[0]
                dy = b.position[1] - a.position[1]
                dist = math.sqrt(dx**2 + dy**2)
                
                # 1. Directional relationships (left/right/front/behind)
                self._add_directional_relations(a, b, dx, dy, dist)
                
                # 2. Proximity
                self._add_proximity_relations(a, b, dist)
                
                # 3. Facing relationships
                self._add_facing_relations(a, b, dx, dy, dist)
                
                # 4. Support relationships (on_top_of)
                self._add_support_relations(a, b)
                
                # 5. Alignment
                self._add_alignment_relations(a, b, dx, dy)
        
        # 6. Wall relationships
        for node in nodes:
            self._add_wall_relations(node)
        
        # 7. Center-of-room relationship
        for node in nodes:
            self._add_center_relations(node)
    
    def _add_directional_relations(self, a: SceneNode, b: SceneNode, 
                                     dx: float, dy: float, dist: float):
        """Add left/right/front/behind relationships."""
        if dist < 0.1:
            return
        
        threshold = 0.3  # Minimum offset to establish directionality
        
        # X-axis: left/right
        if abs(dx) > threshold:
            if dx > 0:
                self.graph.edges.append(SceneEdge(
                    source=a.id, target=b.id,
                    relation=RelationType.LEFT_OF,
                    distance=dist, weight=abs(dx) / dist
                ))
                self.graph.edges.append(SceneEdge(
                    source=b.id, target=a.id,
                    relation=RelationType.RIGHT_OF,
                    distance=dist, weight=abs(dx) / dist
                ))
            else:
                self.graph.edges.append(SceneEdge(
                    source=a.id, target=b.id,
                    relation=RelationType.RIGHT_OF,
                    distance=dist, weight=abs(dx) / dist
                ))
                self.graph.edges.append(SceneEdge(
                    source=b.id, target=a.id,
                    relation=RelationType.LEFT_OF,
                    distance=dist, weight=abs(dx) / dist
                ))
        
        # Y-axis: in_front_of / behind
        if abs(dy) > threshold:
            if dy > 0:
                self.graph.edges.append(SceneEdge(
                    source=a.id, target=b.id,
                    relation=RelationType.IN_FRONT_OF,
                    distance=dist, weight=abs(dy) / dist
                ))
                self.graph.edges.append(SceneEdge(
                    source=b.id, target=a.id,
                    relation=RelationType.BEHIND,
                    distance=dist, weight=abs(dy) / dist
                ))
            else:
                self.graph.edges.append(SceneEdge(
                    source=a.id, target=b.id,
                    relation=RelationType.BEHIND,
                    distance=dist, weight=abs(dy) / dist
                ))
                self.graph.edges.append(SceneEdge(
                    source=b.id, target=a.id,
                    relation=RelationType.IN_FRONT_OF,
                    distance=dist, weight=abs(dy) / dist
                ))
    
    def _add_proximity_relations(self, a: SceneNode, b: SceneNode, dist: float):
        """Add near/adjacent_to relationships."""
        # Adjacent: within touching distance + margin
        a_radius = max(a.dimensions['width'], a.dimensions['depth']) / 2
        b_radius = max(b.dimensions['width'], b.dimensions['depth']) / 2
        touch_dist = a_radius + b_radius + 0.3
        
        if dist < touch_dist:
            self.graph.edges.append(SceneEdge(
                source=a.id, target=b.id,
                relation=RelationType.ADJACENT_TO,
                distance=dist, weight=1.0 - dist / touch_dist
            ))
        elif dist < touch_dist * 2:
            self.graph.edges.append(SceneEdge(
                source=a.id, target=b.id,
                relation=RelationType.NEAR,
                distance=dist, weight=1.0 - dist / (touch_dist * 2)
            ))
    
    def _add_facing_relations(self, a: SceneNode, b: SceneNode,
                                dx: float, dy: float, dist: float):
        """Add facing/facing_away relationships based on rotation."""
        if dist < 0.1:
            return
        
        # Check if A is facing B
        angle_to_b = math.degrees(math.atan2(dy, dx))
        a_facing = (a.rotation + 90) % 360  # Front direction (assuming front = +Y local)
        
        angle_diff = abs((angle_to_b - a_facing + 180) % 360 - 180)
        
        if angle_diff < 45:  # Facing within 45 degrees
            self.graph.edges.append(SceneEdge(
                source=a.id, target=b.id,
                relation=RelationType.FACING,
                distance=dist, weight=1.0 - angle_diff / 45
            ))
        elif angle_diff > 135:  # Facing away
            self.graph.edges.append(SceneEdge(
                source=a.id, target=b.id,
                relation=RelationType.FACING_AWAY,
                distance=dist, weight=(angle_diff - 135) / 45
            ))
    
    def _add_support_relations(self, a: SceneNode, b: SceneNode):
        """Add on_top_of / supported_by relationships."""
        # Check if one is on top of the other based on Z position and dimensions
        a_top = a.position[2] + a.dimensions['height'] if len(a.position) > 2 else a.dimensions['height']
        b_bottom = b.position[2] if len(b.position) > 2 else 0
        b_top = b.position[2] + b.dimensions['height'] if len(b.position) > 2 else b.dimensions['height']
        a_bottom = a.position[2] if len(a.position) > 2 else 0
        
        # Check horizontal overlap
        dx = abs(a.position[0] - b.position[0])
        dy = abs(a.position[1] - b.position[1])
        overlap_x = (a.dimensions['width'] + b.dimensions['width']) / 2 - dx
        overlap_y = (a.dimensions['depth'] + b.dimensions['depth']) / 2 - dy
        
        if overlap_x > 0 and overlap_y > 0:
            # b on top of a?
            if abs(b_bottom - a_top) < 0.1 and b.dimensions['width'] <= a.dimensions['width'] * 1.2:
                self.graph.edges.append(SceneEdge(
                    source=b.id, target=a.id,
                    relation=RelationType.ON_TOP_OF,
                    weight=1.0
                ))
                self.graph.edges.append(SceneEdge(
                    source=a.id, target=b.id,
                    relation=RelationType.SUPPORTED_BY,
                    weight=1.0
                ))
    
    def _add_alignment_relations(self, a: SceneNode, b: SceneNode,
                                   dx: float, dy: float):
        """Add alignment relationships (same X or same Y line)."""
        threshold = 0.2  # meters
        
        if abs(dx) < threshold or abs(dy) < threshold:
            self.graph.edges.append(SceneEdge(
                source=a.id, target=b.id,
                relation=RelationType.ALIGNED_WITH,
                weight=1.0 - min(abs(dx), abs(dy)) / threshold
            ))
    
    def _add_wall_relations(self, node: SceneNode):
        """Add wall proximity relationships."""
        hw = node.dimensions['width'] / 2
        hd = node.dimensions['depth'] / 2
        
        wall_dists = {
            'south': node.position[1] - hd,
            'north': self.room_depth - node.position[1] - hd,
            'west': node.position[0] - hw,
            'east': self.room_width - node.position[0] - hw,
        }
        
        min_wall = min(wall_dists, key=wall_dists.get)
        min_dist = wall_dists[min_wall]
        
        if min_dist < 0.5:  # Within 50cm of wall
            self.graph.edges.append(SceneEdge(
                source=node.id, target=f"wall_{min_wall}",
                relation=RelationType.AGAINST_WALL,
                distance=min_dist,
                weight=1.0 - min_dist / 0.5,
                properties={'wall': min_wall}
            ))
    
    def _add_center_relations(self, node: SceneNode):
        """Check if object is at the center of the room."""
        cx = self.room_width / 2
        cy = self.room_depth / 2
        
        dx = abs(node.position[0] - cx)
        dy = abs(node.position[1] - cy)
        dist_to_center = math.sqrt(dx**2 + dy**2)
        
        room_diag = math.sqrt(self.room_width**2 + self.room_depth**2) / 2
        
        if dist_to_center < room_diag * 0.25:  # Within 25% of room diagonal from center
            self.graph.edges.append(SceneEdge(
                source=node.id, target="room_center",
                relation=RelationType.CENTER_OF,
                distance=dist_to_center,
                weight=1.0 - dist_to_center / (room_diag * 0.25)
            ))
    
    def get_node_by_id(self, node_id: str) -> Optional[SceneNode]:
        """Get a node by its ID."""
        for node in self.graph.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_edges_for_node(self, node_id: str) -> List[SceneEdge]:
        """Get all edges involving a node."""
        return [e for e in self.graph.edges if e.source == node_id or e.target == node_id]
    
    def get_relationships(self, source_id: str, target_id: str) -> List[SceneEdge]:
        """Get all relationships between two specific nodes."""
        return [e for e in self.graph.edges 
                if e.source == source_id and e.target == target_id]
    
    def to_dict(self) -> Dict:
        """Convert scene graph to dictionary (JSON-serializable)."""
        return {
            'room': self.graph.room,
            'nodes': [asdict(n) for n in self.graph.nodes],
            'edges': [asdict(e) for e in self.graph.edges],
            'metadata': {
                'n_nodes': len(self.graph.nodes),
                'n_edges': len(self.graph.edges),
                'relation_types': list(set(e.relation for e in self.graph.edges)),
                'functional_zones': list(set(n.functional_zone for n in self.graph.nodes)),
            }
        }
    
    def to_json(self, path: str, indent: int = 2):
        """Save scene graph to JSON file."""
        data = self.to_dict()
        with open(path, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        return path
    
    def summary(self) -> str:
        """Generate a human-readable summary of the scene graph."""
        lines = []
        lines.append(f"=== Scene Graph Summary ===")
        lines.append(f"Room: {self.room_width}m × {self.room_depth}m × {self.room_height}m")
        lines.append(f"Objects: {len(self.graph.nodes)}")
        lines.append(f"Relationships: {len(self.graph.edges)}")
        
        # Node summary
        lines.append(f"\n--- Nodes ---")
        for node in self.graph.nodes:
            lines.append(f"  [{node.id}] {node.category} "
                        f"pos=({node.position[0]:.2f}, {node.position[1]:.2f}) "
                        f"rot={node.rotation:.0f}° "
                        f"zone={node.functional_zone}")
        
        # Edge summary by type
        lines.append(f"\n--- Relationships ---")
        rel_counts = {}
        for edge in self.graph.edges:
            rel_counts[edge.relation] = rel_counts.get(edge.relation, 0) + 1
        
        for rel, count in sorted(rel_counts.items()):
            lines.append(f"  {rel}: {count}")
        
        # Key spatial relationships
        lines.append(f"\n--- Key Spatial Relations ---")
        for edge in self.graph.edges:
            if edge.weight > 0.5 and edge.relation in [
                RelationType.FACING, RelationType.ADJACENT_TO,
                RelationType.ON_TOP_OF, RelationType.AGAINST_WALL,
                RelationType.CENTER_OF
            ]:
                lines.append(f"  {edge.source} --[{edge.relation}]--> {edge.target} "
                           f"(w={edge.weight:.2f}, d={edge.distance:.2f}m)")
        
        return '\n'.join(lines)


def build_scene_graph(objects: List[Dict], room_width: float, room_depth: float,
                      output_path: Optional[str] = None, verbose: bool = True) -> SceneGraphBuilder:
    """
    Convenience function to build a complete scene graph.
    
    Args:
        objects: List of furniture dicts with category, position, rotation, dimensions
        room_width, room_depth: Room dimensions
        output_path: Optional JSON output path
        verbose: Print summary
        
    Returns:
        SceneGraphBuilder instance
    """
    builder = SceneGraphBuilder(room_width, room_depth)
    
    for i, obj in enumerate(objects):
        obj_with_id = {**obj, 'id': obj.get('id', f"{obj['category']}_{i}")}
        builder.add_object(obj_with_id)
    
    builder.build_relationships()
    
    if verbose:
        print(builder.summary())
    
    if output_path:
        builder.to_json(output_path)
        print(f"\nScene graph saved to: {output_path}")
    
    return builder
