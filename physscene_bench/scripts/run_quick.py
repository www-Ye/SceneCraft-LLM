#!/usr/bin/env python3
"""Quick benchmark: try each model once per prompt with 60s delay, auto-fallback."""
import json, os, sys, time, requests, re, math, traceback
import numpy as np

os.environ['MUJOCO_GL'] = 'osmesa'
sys.path.insert(0, os.path.dirname(__file__))
from evaluator import PhysSceneEvaluator
from pathlib import Path

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Models that tested OK
MODELS = [
    "z-ai/glm-4.5-air:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "stepfun/step-3.5-flash:free",
    "google/gemma-3-27b-it:free",
    "arcee-ai/trinity-large-preview:free",
]

SYSTEM_PROMPT = """You are a 3D indoor scene layout designer. Given a room description, output ONLY valid JSON.
Format: {"objects":[{"category":"sofa","position":[x,y],"rotation":0,"dimensions":{"width":2.0,"depth":0.9,"height":0.85}}]}
Coordinates in meters, origin (0,0) = bottom-left. rotation in degrees. Keep objects inside room. Use realistic dimensions."""

STANDARD_DIMS = {
    "sofa": [2.0,0.9,0.85], "television": [1.2,0.2,0.7], "coffee_table": [1.2,0.6,0.4],
    "armchair": [0.8,0.8,0.85], "lamp": [0.3,0.3,1.5], "cabinet": [0.8,0.4,1.8],
    "rug": [2.0,1.5,0.02], "bed": [1.5,2.0,0.6], "nightstand": [0.5,0.4,0.6],
    "wardrobe": [1.0,0.6,2.0], "dresser": [1.2,0.5,0.8], "desk": [1.2,0.6,0.75],
    "chair": [0.6,0.6,0.85], "table": [1.5,0.8,0.75], "dining_table": [1.5,0.8,0.75],
    "bookshelf": [0.8,0.3,1.8],
}

def clean_json(text):
    text = text.strip()
    # Try to find JSON in code blocks
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m: return m.group(1)
    # Try to find raw JSON  
    m = re.search(r'\{[^{}]*"objects"[^{}]*\[.*?\]\s*\}', text, re.DOTALL)
    if m: return m.group(0)
    # Last resort: first { to last }
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        return text[start:end+1]
    return text

def call_llm(model, prompt):
    """Single attempt with generous timeout"""
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": model,
                  "messages": [{"role":"system","content":SYSTEM_PROMPT},
                               {"role":"user","content":prompt}],
                  "temperature": 0.7, "max_tokens": 2000},
            timeout=120)
        if r.status_code == 200:
            d = r.json()
            if "choices" in d and d["choices"]:
                content = d["choices"][0]["message"].get("content") or ""
                if not content:
                    print(f"    empty content from model", flush=True)
                    return None, "failed"
                cleaned = clean_json(content)
                layout = json.loads(cleaned)
                if "objects" in layout and len(layout["objects"]) > 0:
                    return layout, "real"
                print(f"    parsed but empty/invalid objects", flush=True)
        else:
            errm = r.text[:100] if r.text else str(r.status_code)
            print(f"    HTTP {r.status_code}: {errm}", flush=True)
    except json.JSONDecodeError as e:
        print(f"    JSON error: {e}", flush=True)
    except Exception as e:
        print(f"    error: {e}", flush=True)
    return None, "failed"

def make_mock(prompt_data):
    """Simple grid placement baseline"""
    room_w, room_h = prompt_data["room_size"]
    expected = prompt_data["expected_objects"]
    objects = []
    n = len(expected)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = max(1, math.ceil(n / cols))
    for i, cat in enumerate(expected):
        row, col = divmod(i, cols)
        x = 0.8 + col * (room_w - 1.6) / max(cols - 1, 1)
        y = 0.8 + row * (room_h - 1.6) / max(rows - 1, 1)
        dims = STANDARD_DIMS.get(cat, [0.6, 0.6, 0.8])
        objects.append({
            "category": cat,
            "position": [round(x, 2), round(y, 2)],
            "rotation": 0,
            "dimensions": {"width": dims[0], "depth": dims[1], "height": dims[2]}
        })
    return {"objects": objects}

def main():
    prompts_file = Path(__file__).parent / "../prompts/layout_prompts.json"
    results_dir = Path(__file__).parent / "../results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(prompts_file) as f:
        prompts = json.load(f)
    
    evaluator = PhysSceneEvaluator()
    all_results = []
    
    all_models = MODELS + ["mock_baseline"]
    total = len(prompts) * len(all_models)
    idx = 0
    
    for model in all_models:
        model_safe = model.replace("/","_").replace(":","_")
        model_dir = results_dir / model_safe
        model_dir.mkdir(parents=True, exist_ok=True)
        
        successes = 0
        for prompt_data in prompts:
            idx += 1
            pid = prompt_data["id"]
            print(f"[{idx}/{total}] {model} x {pid}", flush=True)
            
            prompt_dir = model_dir / pid
            prompt_dir.mkdir(parents=True, exist_ok=True)
            
            if model == "mock_baseline":
                layout = make_mock(prompt_data)
                source = "mock"
            else:
                layout, source = call_llm(model, prompt_data["prompt"])
                if layout is None:
                    layout = make_mock(prompt_data)
                    source = "mock_fallback"
                else:
                    successes += 1
            
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
                traceback.print_exc()
                ev = {"overall_score": 0, "layer_scores": {"semantic_fidelity":0,"physical_plausibility":0,"functional_affordance":0}, "detailed_scores": {}}
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
            
            with open(prompt_dir / "result.json", "w") as f:
                json.dump(result, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
            
            # Generous delay for free API
            if model != "mock_baseline":
                time.sleep(30)
        
        if model != "mock_baseline":
            print(f"  => {model}: {successes}/10 real API successes", flush=True)
    
    # Save summary
    with open(results_dir / "benchmark_summary.json", "w") as f:
        json.dump({"total": len(all_results), "results": all_results}, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    
    # Print results table
    print("\n" + "="*80, flush=True)
    print("PHYSSCENE-BENCH RESULTS SUMMARY", flush=True)
    print("="*80, flush=True)
    
    for model in all_models:
        mr = [r for r in all_results if r["model"] == model]
        real = sum(1 for r in mr if r["source"] == "real")
        
        scores = [r["evaluation"].get("overall_score", 0) for r in mr if r["success"]]
        l1 = [r["evaluation"].get("layer_scores",{}).get("semantic_fidelity",0) for r in mr if r["success"]]
        l2 = [r["evaluation"].get("layer_scores",{}).get("physical_plausibility",0) for r in mr if r["success"]]
        l3 = [r["evaluation"].get("layer_scores",{}).get("functional_affordance",0) for r in mr if r["success"]]
        
        name = model.split("/")[-1] if "/" in model else model
        print(f"\n{name}", flush=True)
        print(f"  API success: {real}/10", flush=True)
        if scores:
            print(f"  Overall:    {np.mean(scores):.3f} (+/- {np.std(scores):.3f})", flush=True)
            print(f"  Semantic:   {np.mean(l1):.3f}", flush=True)
            print(f"  Physical:   {np.mean(l2):.3f}", flush=True)
            print(f"  Functional: {np.mean(l3):.3f}", flush=True)
    
    print(f"\nDone! Results saved to {results_dir}/benchmark_summary.json", flush=True)

if __name__ == "__main__":
    main()
