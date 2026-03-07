#!/usr/bin/env python3
"""
Test the evaluator with a mock layout to ensure it works correctly.
"""

import json
from evaluator import PhysSceneEvaluator

def create_mock_layout():
    """Create a mock layout for testing"""
    return {
        "objects": [
            {
                "category": "sofa",
                "position": [2.0, 1.0],
                "rotation": 0,
                "dimensions": {"width": 2.0, "depth": 0.9, "height": 0.85}
            },
            {
                "category": "coffee_table", 
                "position": [2.0, 2.0],
                "rotation": 0,
                "dimensions": {"width": 1.2, "depth": 0.6, "height": 0.4}
            },
            {
                "category": "television",
                "position": [2.0, 3.5],
                "rotation": 180,
                "dimensions": {"width": 1.2, "depth": 0.2, "height": 0.7}
            },
            {
                "category": "armchair",
                "position": [3.5, 1.5],
                "rotation": 270,
                "dimensions": {"width": 0.8, "depth": 0.8, "height": 0.85}
            }
        ]
    }

def test_evaluator():
    """Test the evaluator with mock data"""
    print("Testing PhysScene Evaluator...")
    
    evaluator = PhysSceneEvaluator()
    layout = create_mock_layout()
    room_size = [5.0, 4.0]
    expected_objects = ["sofa", "television", "coffee_table", "armchair", "lamp", "cabinet", "rug"]
    functional_checks = ["sofa_faces_tv", "coffee_table_reachable_from_sofa", "walkable_path_exists"]
    
    print("Mock layout:")
    print(json.dumps(layout, indent=2))
    
    print("\nRunning evaluation...")
    results = evaluator.evaluate_layout(
        layout=layout,
        room_size=room_size,
        expected_objects=expected_objects,
        functional_checks=functional_checks
    )
    
    print("\nEvaluation Results:")
    print(f"Overall Score: {results['overall_score']:.3f}")
    print("\nLayer Scores:")
    for layer, score in results["layer_scores"].items():
        print(f"  {layer}: {score:.3f}")
    
    print("\nDetailed Scores:")
    for category, scores in results["detailed_scores"].items():
        print(f"  {category}:")
        for metric, score in scores.items():
            print(f"    {metric}: {score:.3f}")
    
    # Test individual components
    print("\n=== Testing Individual Components ===")
    
    print("\n1. Testing collision detection...")
    obj1 = layout["objects"][0]  # sofa
    obj2 = layout["objects"][1]  # coffee_table
    collision = evaluator.check_collision(obj1, obj2)
    print(f"Sofa-CoffeeTable collision: {collision}")
    
    print("\n2. Testing boundary checks...")
    for i, obj in enumerate(layout["objects"]):
        inside = evaluator.is_inside_room(obj, room_size)
        print(f"Object {i} ({obj['category']}) inside room: {inside}")
    
    print("\n3. Testing bounding boxes...")
    for i, obj in enumerate(layout["objects"]):
        bbox = evaluator.get_object_bbox(obj)
        print(f"Object {i} ({obj['category']}) bbox: {bbox}")
    
    print("\nEvaluator test completed successfully!")

if __name__ == "__main__":
    test_evaluator()