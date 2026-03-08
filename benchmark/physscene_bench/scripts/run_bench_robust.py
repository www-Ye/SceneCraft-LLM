#!/usr/bin/env python3
"""
Robust benchmark runner — tries each model with generous delays,
falls back to mock for persistently rate-limited models.
"""
import json, os, sys, time, requests, re, math
import numpy as np

os.environ['MUJOCO_GL'] = 'osmesa'

from evaluator import PhysSceneEvaluator
from pathlib import Path

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

MODELS = [
    "google/gemma-3-27b-it:free",
    "z-ai/glm-4.5-air:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "stepfun/step-3.5-flash:free",
    "arcee-ai/trinity-large-preview:free",
]

SYSTEM_PROMPT = """You are a 3D indoor scene layout designer. Given a room description, generate a furniture layout.
Output ONLY valid JSON (no markdown, no explanation) with this exact structure:
{"objects":[{"category":"sofa","position":[x,y],"rotation":0,"dimensions":{"width":2.0,"depth":0.9,"height":0.85}}]}
Rules:
- Coordinates in meters. Origin (0,0) = southwest corner.
- rotation in degrees (0=north/+Y, 90=east/+X, 180=south, 270=west)
- Objects must be within room boundaries
- Use realistic furniture dimensions
- Place furniture sensibly: large items against walls, maintain walkways"""

def clean_json(text):
    text = text.strip()
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m: return m.group(1)
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m: return m.group(0)
    return text

def call_llm(model, prompt, retries=6):
    """Call with aggressive retry + backoff"""
    for attempt in range(retries):
        if attempt > 0:
            wait = 20 + 20 * attempt  # 40, 60, 80, 100, 120s
            print(f"    retry {attempt+1}/{retries}, wait {wait}s...", flush=True)
            time.sleep(wait)
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"model": model,
                      "messages": [{"role":"system","content":SYSTEM_PROMPT},
                                   {"role":"user","content":prompt}],
                      "temperature": 0.7},
                timeout=120)
            if r.status_code == 200:
                d = r.json()
                content = d["choices"][0]["message"]["content"]
                layout = json.loads(clean_json(content))
                if "objects" in layout and len(layout["objects"]) > 0:
                    return layout, "real"
                else:
                    print(f"    empty layout, retry...", flush=True)
            elif r.status_code == 429:
                print(f"    429 rate limit", flush=True)
            else:
                print(f"    {r.status_code}: {r.text[:80]}", flush=True)
        except json.JSONDecodeError as e:
            print(f"    JSON parse error: {e}", flush=True)
            try:
                raw = r.json()["choices"][0]["message"]["content"][:200]
                print(f"    raw: {raw}", flush=True)
            except: pass
        except Exception as e:
            print(f"    error: {e}", flush=True)
    return None, "failed"

def make_mock_layout(prompt_data):
    """Rule-based mock layout for comparison baseline"""
    room_w, room_h = prompt_data["room_size"]
    objects = []
    expected = prompt_data["expected_objects"]
    
    # Simple rule: place objects in a grid near walls
    margin = 0.5
    positions = []
    n = len(expected)
    cols = math.ceil(math.sqrt(n))
    for i, cat in enumerate(expected):
        row, col = divmod(i, cols)
        x = margin + col * (room_w - 2*margin) / max(cols, 1)
        y = margin + row * (room_h - 2*margin) / max(math.ceil(n/cols), 1)
        
        # Standard dimensions
        dims = {
            "sofa": [2.0,0.9,0.85], "television": [1.2,0.2,0.7], "coffee_table": [1.2,0.6,0.4],
            "armchair": [0.8,0.8,0.85], "lamp": [0.3,0.3,1.5], "cabinet": [0.8,0.4,1.8],
            "rug": [2.0,1.5,0.02], "bed": [1.5,2.0,0.6], "nightstand": [0.5,0.4,0.6],
            "wardrobe": [1.0,0.6,2.0], "dresser": [1.2,0.5,0.8], "desk": [1.2,0.6,0.75],
            "chair": [0.6,0.6,0.85], "table": [1.5,0.8,0.75], "dining_table": [1.5,0.8,0.75],
            "bookshelf": [0.8,0.3,1.8],
        }.get(cat, [0.6, 0.6, 0.8])
        
        objects.append({
            "category": cat,
            "position": [round(x, 2), round(y, 2)],
            "rotation": 0,
            "dimensions": {"width": dims[0], "depth": dims[1], "height": dims[2]}
        })
    return {"objects": objects}

def main():
    prompts_file = Path("../prompts/layout_prompts.json")
    results_dir = Path("../results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(prompts_file) as f:
        prompts = json.load(f)
    
    evaluator = PhysSceneEvaluator()
    all_results = []
    total = len(prompts) * len(MODELS)
    idx = 0
    
    # Also add a "mock_baseline" for comparison
    all_models = MODELS + ["mock_baseline"]
    total = len(prompts) * len(all_models)
    
    for model in all_models:
        model_safe = model.replace("/", "_").replace(":", "_")
        model_dir = results_dir / model_safe
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for prompt_data in prompts:
            idx += 1
            pid = prompt_data["id"]
            print(f"[{idx}/{total}] {model} × {pid}", flush=True)
            
            prompt_dir = model_dir / pid
            prompt_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate layout
            if model == "mock_baseline":
                layout = make_mock_layout(prompt_data)
                source = "mock"
            else:
                layout, source = call_llm(model, prompt_data["prompt"])
                if layout is None:
                    # Fall back to mock
                    layout = make_mock_layout(prompt_data)
                    source = "mock_fallback"
            
            # Save layout
            with open(prompt_dir / "layout.json", "w") as f:
                json.dump({"layout": layout, "source": source}, f, indent=2)
            
            # Evaluate
            try:
                ev = evaluator.evaluate_layout(
                    layout=layout,
                    room_size=prompt_data["room_size"],
                    expected_objects=prompt_data["expected_objects"],
                    functional_checks=prompt_data["functional_checks"]
                )
                success = True
            except Exception as e:
                print(f"    eval error: {e}", flush=True)
                ev = {"overall_score": 0, "layer_scores": {}, "detailed_scores": {}}
                success = False
            
            result = {
                "prompt_id": pid,
                "model": model,
                "source": source,
                "success": success,
                "num_objects": len(layout.get("objects", [])),
                "evaluation": ev,
            }
            all_results.append(result)
            
            # Save per-prompt result
            with open(prompt_dir / "result.json", "w") as f:
                json.dump(result, f, indent=2, default=str)
            
            # Delay between calls (only for real API)
            if model != "mock_baseline":
                time.sleep(15)
    
    # === Summary ===
    summary = {"total": len(all_results), "results": all_results}
    with open(results_dir / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print summary table
    print("\n" + "="*80, flush=True)
    print("PHYSSCENE-BENCH RESULTS", flush=True)
    print("="*80, flush=True)
    
    for model in all_models:
        model_results = [r for r in all_results if r["model"] == model]
        real_count = sum(1 for r in model_results if r["source"] == "real")
        scores = [r["evaluation"].get("overall_score", 0) for r in model_results if r["success"]]
        
        layer1 = [r["evaluation"].get("layer_scores", {}).get("semantic_fidelity", 0) for r in model_results if r["success"]]
        layer2 = [r["evaluation"].get("layer_scores", {}).get("physical_plausibility", 0) for r in model_results if r["success"]]
        layer3 = [r["evaluation"].get("layer_scores", {}).get("functional_affordance", 0) for r in model_results if r["success"]]
        
        print(f"\n📊 {model}", flush=True)
        print(f"   Real API: {real_count}/{len(model_results)} | Mock fallback: {len(model_results)-real_count}", flush=True)
        if scores:
            print(f"   Overall:    {np.mean(scores):.3f} ± {np.std(scores):.3f}", flush=True)
            print(f"   Semantic:   {np.mean(layer1):.3f}", flush=True)
            print(f"   Physical:   {np.mean(layer2):.3f}", flush=True)
            print(f"   Functional: {np.mean(layer3):.3f}", flush=True)
    
    print(f"\n✅ Done! Results in {results_dir}/", flush=True)

if __name__ == "__main__":
    main()
