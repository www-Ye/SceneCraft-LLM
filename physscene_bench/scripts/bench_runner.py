#!/usr/bin/env python3
"""
PhysScene-Bench Main Evaluation Runner
Evaluates LLM-generated 3D indoor scene layouts across multiple dimensions.
"""

import json
import os
import time
import requests
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from evaluator import PhysSceneEvaluator

# Constants
MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-next-80b-a3b-instruct:free", 
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "z-ai/glm-4.5-air:free",
]

SYSTEM_PROMPT = """You are a 3D indoor scene layout designer. Given a room description, generate a furniture layout as a JSON object.
Output ONLY valid JSON with this structure:
{
  "objects": [
    {
      "category": "sofa",
      "position": [x, y],
      "rotation": 0,
      "dimensions": {"width": 2.0, "depth": 0.9, "height": 0.85}
    }
  ]
}
Rules:
- All coordinates in meters. Origin (0,0) is the southwest corner.
- rotation is in degrees (0=facing north/+Y, 90=facing east/+X, 180=facing south, 270=facing west)
- Objects must be within room boundaries
- Dimensions should be realistic for the furniture type
- Consider walkability and functional placement"""

class PhysSceneBenchRunner:
    def __init__(self, prompts_file: str, results_dir: str):
        self.prompts_file = Path(prompts_file)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load API key
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            print("Warning: OPENROUTER_API_KEY not found in environment variables")
        
        # Initialize evaluator
        self.evaluator = PhysSceneEvaluator()
        
        # Load prompts
        with open(self.prompts_file, 'r') as f:
            self.prompts = json.load(f)
        
        print(f"Loaded {len(self.prompts)} prompts from {self.prompts_file}")
        print(f"Results will be saved to {self.results_dir}")
    
    def clean_json_response(self, response_text: str) -> str:
        """Extract JSON from LLM response that might contain markdown code blocks"""
        # Remove markdown code blocks if present
        text = response_text.strip()
        
        # Look for JSON within code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # Look for raw JSON
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return text
    
    def call_llm(self, model: str, prompt: str, max_retries: int = 3) -> Optional[Dict]:
        """Call OpenRouter API to generate scene layout"""
        # Mock mode for testing without API key
        if not self.api_key:
            print("No API key found, using mock response...")
            return self._generate_mock_layout(prompt)
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT}, 
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                    },
                    timeout=120,
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # Clean and parse JSON
                    json_str = self.clean_json_response(content)
                    layout = json.loads(json_str)
                    
                    return layout
                    
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Rate limited, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"API error {response.status_code}: {response.text}")
                    
            except json.JSONDecodeError as e:
                print(f"JSON parse error for {model}: {e}")
                print(f"Raw response: {response.json()['choices'][0]['message']['content'][:200]}...")
            except Exception as e:
                print(f"Error calling {model}: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2)  # Rate limiting
        
        return None
    
    def _generate_mock_layout(self, prompt: str) -> Dict:
        """Generate a mock layout for testing purposes"""
        import random
        
        # Simple rule-based mock generation for testing
        objects = []
        
        if "living room" in prompt.lower():
            objects = [
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
                },
                {
                    "category": "lamp",
                    "position": [4.0, 0.5],
                    "rotation": 0,
                    "dimensions": {"width": 0.3, "depth": 0.3, "height": 1.5}
                }
            ]
        elif "bedroom" in prompt.lower():
            objects = [
                {
                    "category": "bed",
                    "position": [1.5, 1.0],
                    "rotation": 0,
                    "dimensions": {"width": 1.5, "depth": 2.0, "height": 0.6}
                },
                {
                    "category": "nightstand",
                    "position": [0.5, 1.0],
                    "rotation": 0,
                    "dimensions": {"width": 0.5, "depth": 0.4, "height": 0.6}
                },
                {
                    "category": "nightstand",
                    "position": [2.5, 1.0],
                    "rotation": 0,
                    "dimensions": {"width": 0.5, "depth": 0.4, "height": 0.6}
                },
                {
                    "category": "wardrobe",
                    "position": [3.5, 3.0],
                    "rotation": 0,
                    "dimensions": {"width": 1.0, "depth": 0.6, "height": 2.0}
                }
            ]
        else:
            # Generic furniture
            objects = [
                {
                    "category": "chair",
                    "position": [1.0, 1.0],
                    "rotation": 0,
                    "dimensions": {"width": 0.6, "depth": 0.6, "height": 0.85}
                },
                {
                    "category": "table",
                    "position": [2.0, 2.0],
                    "rotation": 0,
                    "dimensions": {"width": 1.5, "depth": 0.8, "height": 0.75}
                }
            ]
        
        return {"objects": objects}
    
    def run_single_evaluation(self, prompt_data: Dict, model: str) -> Dict[str, Any]:
        """Run evaluation for one prompt with one model"""
        prompt_id = prompt_data["id"]
        room_size = prompt_data["room_size"]
        prompt_text = prompt_data["prompt"]
        expected_objects = prompt_data["expected_objects"]
        functional_checks = prompt_data["functional_checks"]
        
        print(f"Evaluating {model} on {prompt_id}...")
        
        # Create model-specific result directory
        model_dir = self.results_dir / model.replace("/", "_").replace(":", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        prompt_dir = model_dir / prompt_id
        prompt_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate layout
        layout = self.call_llm(model, prompt_text)
        if layout is None:
            return {
                "prompt_id": prompt_id,
                "model": model,
                "success": False,
                "error": "Failed to generate layout"
            }
        
        # Save generated layout
        layout_file = prompt_dir / "generated_layout.json"
        with open(layout_file, 'w') as f:
            json.dump(layout, f, indent=2)
        
        # Run evaluation
        try:
            evaluation_results = self.evaluator.evaluate_layout(
                layout=layout,
                room_size=room_size,
                expected_objects=expected_objects,
                functional_checks=functional_checks
            )
            
            # Compile results
            results = {
                "prompt_id": prompt_id,
                "model": model,
                "success": True,
                "layout": layout,
                "evaluation": evaluation_results,
                "metadata": {
                    "room_type": prompt_data["room_type"],
                    "room_size": room_size,
                    "expected_objects": expected_objects,
                    "functional_checks": functional_checks
                }
            }
            
            # Save results
            results_file = prompt_dir / "evaluation_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            return results
            
        except Exception as e:
            print(f"Evaluation error for {model} on {prompt_id}: {e}")
            return {
                "prompt_id": prompt_id,
                "model": model,
                "success": False,
                "error": str(e),
                "layout": layout
            }
    
    def run_benchmark(self, test_mode: bool = False, demo_mode: bool = False):
        """Run the complete benchmark evaluation"""
        if test_mode:
            print("Running in test mode (1 prompt, 1 model)")
            prompts_to_test = self.prompts[:1]
            models_to_test = MODELS[:1]
        elif demo_mode:
            print("Running in demo mode (3 prompts, 3 models)")
            prompts_to_test = self.prompts[:3]
            models_to_test = MODELS[:3]
        else:
            print("Running full benchmark (10 prompts × 6 models)")
            prompts_to_test = self.prompts
            models_to_test = MODELS
        
        all_results = []
        total_runs = len(prompts_to_test) * len(models_to_test)
        current_run = 0
        
        for prompt_data in prompts_to_test:
            for model in models_to_test:
                current_run += 1
                print(f"Progress: {current_run}/{total_runs}")
                
                result = self.run_single_evaluation(prompt_data, model)
                all_results.append(result)
                
                # Rate limiting between API calls
                time.sleep(2)
        
        # Save consolidated results
        summary_file = self.results_dir / "benchmark_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "metadata": {
                    "total_prompts": len(prompts_to_test),
                    "total_models": len(models_to_test),
                    "total_runs": len(all_results),
                    "success_rate": sum(1 for r in all_results if r["success"]) / len(all_results)
                },
                "results": all_results
            }, f, indent=2)
        
        print(f"Benchmark complete! Results saved to {summary_file}")
        return all_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run PhysScene-Bench evaluation")
    parser.add_argument("--test", action="store_true", help="Run test mode (1 prompt, 1 model)")
    parser.add_argument("--demo", action="store_true", help="Run demo mode (3 prompts, 3 models)")
    parser.add_argument("--prompts", default="../prompts/layout_prompts.json", help="Path to prompts file")
    parser.add_argument("--results", default="../results", help="Results directory")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = PhysSceneBenchRunner(args.prompts, args.results)
    
    # Run benchmark
    results = runner.run_benchmark(test_mode=args.test, demo_mode=args.demo)
    
    # Print summary
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"\nBenchmark Summary:")
    print(f"Total runs: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success rate: {successful/total*100:.1f}%")

if __name__ == "__main__":
    main()