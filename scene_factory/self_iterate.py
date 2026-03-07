#!/usr/bin/env python3
"""
Self-Iterating LLM Pipeline for Tabletop Environment Generation.

The LLM:
1. Designs a layout from a scene description
2. The system renders it in MuJoCo 
3. The LLM reviews the top-down image and physics results
4. The LLM refines the layout
5. Repeat until convergence

Uses OpenRouter API (free models or paid).
"""
import json, os, sys, math, re, time, base64, tempfile, shutil
import numpy as np
from pathlib import Path
from datetime import datetime
import requests

os.environ['MUJOCO_GL'] = 'osmesa'
from llm_tabletop_pipeline import (
    build_tabletop_xml, simulate_and_render, 
    TABLETOP_DIMS, STL_DIR, OUTPUT_DIR
)

# === Config ===
API_KEY = ""  # Will be loaded from openclaw config
LLM_MODEL = "anthropic/claude-sonnet-4"  # For layout design
VLM_MODEL = "qwen/qwen3-vl-30b-a3b-thinking"  # Free VLM for image review

DESIGN_PROMPT = """You are an expert at designing realistic tabletop arrangements for robot manipulation training in MuJoCo.

Design a {scene_type} scene on a table ({table_w}m x {table_d}m). The table center is at (0, 0).
Valid x range: [{x_min:.2f}, {x_max:.2f}], valid y range: [{y_min:.2f}, {y_max:.2f}]

A robot arm is positioned at the -Y side of the table (below the table in top-down view).

Requirements:
- Place objects realistically (as a human would arrange them)
- Objects must not overlap (maintain at least 2cm gap)
- Objects must be within table boundaries
- Consider functional grouping (utensils near plate, etc.)
- Leave some clear workspace for the robot arm to operate

Available objects and their sizes (width x depth x height in meters):
{available_objects}

Output ONLY valid JSON:
{{"objects": [{{"category": "plate", "position": [0.0, 0.05], "rotation": 0}}, ...]}}

Position is [x, y] relative to table center. Rotation in degrees."""

REVIEW_PROMPT = """You are reviewing a tabletop scene rendered in MuJoCo for robot manipulation training.

Scene description: {scene_type}
Table size: {table_w}m x {table_d}m

The image shows a top-down view of the table with objects arranged on it.
A robot arm is at the bottom of the image.

Physics simulation results:
- Stable objects: {n_stable}
- Fallen objects: {n_fallen} {fallen_list}

Previous layout:
{layout_json}

Please evaluate and suggest improvements. Consider:
1. Are objects realistically arranged? (e.g., fork left of plate, knife right)
2. Is there enough space between objects for robot gripper access?
3. Are any objects too close to the table edge?
4. Does the arrangement make functional sense?
5. Is there clear workspace for the robot?

If the scene is good (score >= 8/10), respond with:
{{"score": N, "feedback": "looks good", "action": "accept"}}

If improvements needed, respond with the COMPLETE revised layout:
{{"score": N, "feedback": "description of issues", "action": "revise", "objects": [...]}}

Output ONLY valid JSON."""


def load_api_key():
    """Load OpenRouter API key from openclaw config."""
    global API_KEY
    try:
        with open(os.path.expanduser("~/.openclaw/openclaw.json")) as f:
            cfg = json.load(f)
        API_KEY = cfg["models"]["providers"]["openrouter"]["apiKey"]
    except:
        pass
    return API_KEY


def call_llm(prompt, model=None):
    """Call LLM via OpenRouter."""
    model = model or LLM_MODEL
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": model,
              "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.4, "max_tokens": 2000},
        timeout=120)
    
    if r.status_code != 200:
        raise Exception(f"LLM API error {r.status_code}: {r.text[:200]}")
    
    content = r.json()["choices"][0]["message"]["content"]
    return content


def call_vlm(prompt, image_path, model=None):
    """Call VLM with an image via OpenRouter."""
    model = model or VLM_MODEL
    
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": model,
              "messages": [{
                  "role": "user",
                  "content": [
                      {"type": "text", "text": prompt},
                      {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                  ]
              }],
              "temperature": 0.3, "max_tokens": 2000},
        timeout=120)
    
    if r.status_code != 200:
        raise Exception(f"VLM API error {r.status_code}: {r.text[:200]}")
    
    content = r.json()["choices"][0]["message"]["content"]
    return content


def extract_json(text):
    """Extract JSON from LLM response."""
    text = text.strip()
    # Code block
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Raw JSON
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"No JSON found in: {text[:200]}")


def run_self_iterate(scene_type, table_size=[0.80, 0.60], max_iter=3, use_vlm=True):
    """Run the full self-iterating pipeline."""
    load_api_key()
    
    tw, td = table_size
    x_min, x_max = -(tw/2 - 0.05), (tw/2 - 0.05)
    y_min, y_max = -(td/2 - 0.05), (td/2 - 0.05)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scene_dir = OUTPUT_DIR / f"iterate_{timestamp}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    # Format available objects
    obj_list = "\n".join([f"  {cat}: {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f}m" 
                          for cat, dims in sorted(TABLETOP_DIMS.items())])
    
    print(f"🎯 Scene: {scene_type}", flush=True)
    print(f"📐 Table: {tw}m x {td}m", flush=True)
    print(f"🔄 Max iterations: {max_iter}", flush=True)
    print(f"📁 Output: {scene_dir}", flush=True)
    
    # === Step 1: Initial layout design ===
    print(f"\n--- Iteration 0: Initial Design ---", flush=True)
    
    design_prompt = DESIGN_PROMPT.format(
        scene_type=scene_type,
        table_w=tw, table_d=td,
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        available_objects=obj_list,
    )
    
    print("  🤖 Calling LLM for initial layout...", flush=True)
    response = call_llm(design_prompt)
    layout = extract_json(response)
    
    n_objects = len(layout.get("objects", []))
    print(f"  📦 Generated {n_objects} objects: {[o['category'] for o in layout['objects']]}", flush=True)
    
    # Save
    iter_dir = scene_dir / "iter_0"
    iter_dir.mkdir(exist_ok=True)
    with open(iter_dir / "layout.json", "w") as f:
        json.dump(layout, f, indent=2)
    
    # === Iteration loop ===
    best_layout = layout
    best_score = 0
    
    for iteration in range(max_iter):
        iter_dir = scene_dir / f"iter_{iteration}"
        iter_dir.mkdir(exist_ok=True)
        
        # Step 2: Build and render
        print(f"\n  🔨 Building MuJoCo scene...", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            for f in STL_DIR.glob("*.stl"):
                shutil.copy2(f, tmp)
            xml = build_tabletop_xml(layout, table_size=table_size, tmp_dir=tmp)
            
            with open(iter_dir / "scene.xml", "w") as f:
                f.write(xml)
            
            print(f"  🎮 Simulating physics...", flush=True)
            sim_results = simulate_and_render(xml, iter_dir)
        
        n_stable = len(sim_results["stable"])
        n_fallen = len(sim_results["fallen"])
        print(f"  📊 Stable: {n_stable}, Fallen: {n_fallen}", flush=True)
        
        # Step 3: Review with VLM (or LLM if VLM unavailable)
        print(f"  👁️ Reviewing scene...", flush=True)
        
        review_prompt = REVIEW_PROMPT.format(
            scene_type=scene_type,
            table_w=tw, table_d=td,
            n_stable=n_stable,
            n_fallen=n_fallen,
            fallen_list=str(sim_results["fallen"]) if sim_results["fallen"] else "(none)",
            layout_json=json.dumps(layout, indent=2),
        )
        
        try:
            if use_vlm:
                review_response = call_vlm(review_prompt, str(iter_dir / "topdown.png"))
            else:
                review_response = call_llm(review_prompt)
        except Exception as e:
            print(f"  ⚠️ Review failed: {e}, trying text-only...", flush=True)
            review_response = call_llm(review_prompt)
        
        try:
            review = extract_json(review_response)
        except:
            print(f"  ⚠️ Could not parse review, using default", flush=True)
            review = {"score": 5, "feedback": "could not parse", "action": "accept"}
        
        score = review.get("score", 5)
        feedback = review.get("feedback", "")
        action = review.get("action", "accept")
        
        print(f"  ⭐ Score: {score}/10 — {feedback}", flush=True)
        
        # Save review
        with open(iter_dir / "review.json", "w") as f:
            json.dump(review, f, indent=2)
        
        if score > best_score:
            best_score = score
            best_layout = layout
        
        # Step 4: Accept or revise
        if action == "accept" or score >= 8 or iteration == max_iter - 1:
            print(f"\n✅ Accepted! Final score: {score}/10", flush=True)
            break
        
        # Revise layout
        if "objects" in review:
            layout = {"objects": review["objects"]}
            print(f"  🔄 Revised layout: {len(layout['objects'])} objects", flush=True)
        else:
            # Ask LLM to revise based on feedback
            revise_prompt = f"""Revise this tabletop layout based on feedback.

Current layout:
{json.dumps(layout, indent=2)}

Feedback: {feedback}

Table size: {tw}m x {td}m. Center at (0,0).
x range: [{x_min:.2f}, {x_max:.2f}], y range: [{y_min:.2f}, {y_max:.2f}]

Output the COMPLETE revised layout as JSON:
{{"objects": [...]}}"""
            
            print(f"  🤖 Calling LLM for revised layout...", flush=True)
            revise_response = call_llm(revise_prompt)
            layout = extract_json(revise_response)
            print(f"  📦 Revised: {[o['category'] for o in layout['objects']]}", flush=True)
        
        time.sleep(5)  # Rate limiting
    
    # Save final result
    final_dir = scene_dir / "final"
    final_dir.mkdir(exist_ok=True)
    with open(final_dir / "layout.json", "w") as f:
        json.dump(best_layout, f, indent=2)
    
    # Final render of best layout
    with tempfile.TemporaryDirectory() as tmp:
        for f in STL_DIR.glob("*.stl"):
            shutil.copy2(f, tmp)
        xml = build_tabletop_xml(best_layout, table_size=table_size, tmp_dir=tmp)
        with open(final_dir / "scene.xml", "w") as f:
            f.write(xml)
        simulate_and_render(xml, final_dir)
    
    print(f"\n📁 All outputs in: {scene_dir}", flush=True)
    print(f"🏆 Best score: {best_score}/10", flush=True)
    
    return {
        "scene_dir": str(scene_dir),
        "best_score": best_score,
        "best_layout": best_layout,
        "iterations": iteration + 1,
    }


if __name__ == "__main__":
    # Run a few example scenes
    scenes = [
        "A breakfast setting for one person: plate in center, mug of coffee to the right, bowl of cereal to the left, fork and knife beside the plate, and a glass of juice",
        "A cluttered study desk: laptop in center, books stacked to the left, pen and phone on the right, mug near the top right corner",
    ]
    
    for scene in scenes:
        print(f"\n{'='*60}", flush=True)
        result = run_self_iterate(scene, max_iter=2, use_vlm=True)
        print(f"\nResult: score={result['best_score']}, iterations={result['iterations']}", flush=True)
        time.sleep(10)
