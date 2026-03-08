#!/usr/bin/env python3
"""
VLM-as-Judge: Use a Vision-Language Model to evaluate rendered scene layouts.
Scores each layout on multiple dimensions by analyzing the floorplan image.
"""
import json, os, sys, time, base64, requests
from pathlib import Path

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Free VLM model
VLM_MODEL = "qwen/qwen3-vl-30b-a3b-thinking"

JUDGE_PROMPT = """You are an expert interior designer evaluating a 3D indoor scene layout.

The image shows a 2D floor plan of a room. Each colored rectangle represents a piece of furniture, with labels showing the category. Red arrows show the facing direction.

Room description: {prompt}
Room size: {room_w}m × {room_h}m
Expected objects: {expected}

Please evaluate this layout on the following dimensions. For each, give a score from 0-10 and a brief justification:

1. **Object Completeness** (0-10): Are all expected objects present? Are there extras?
2. **Spatial Reasonableness** (0-10): Are objects placed in sensible locations? (e.g., bed against wall, TV facing seating)
3. **No Collisions** (0-10): Do objects overlap or clip into each other?
4. **Walkability** (0-10): Is there enough open space to walk through the room?
5. **Functional Layout** (0-10): Can furniture be used as intended? (e.g., chairs face tables, sofa faces TV, nightstands next to bed)
6. **Wall Utilization** (0-10): Are large furniture items properly placed against walls?
7. **Overall Quality** (0-10): Overall impression of the layout quality.

Output ONLY valid JSON like this example (replace N with 0-10 scores):
{{"completeness":N,"spatial":N,"no_collisions":N,"walkability":N,"functional":N,"wall_use":N,"overall":N,"reasoning":"brief text"}}
"""

def encode_image(image_path):
    """Base64 encode an image file."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def vlm_judge(image_path, prompt_data):
    """Use VLM to judge a layout from its floorplan image."""
    if not API_KEY:
        return None, "no_api_key"

    img_b64 = encode_image(image_path)

    user_prompt = JUDGE_PROMPT.format(
        prompt=prompt_data.get("prompt", ""),
        room_w=prompt_data.get("room_size", [5, 4])[0],
        room_h=prompt_data.get("room_size", [5, 4])[1],
        expected=", ".join(prompt_data.get("expected_objects", []))
    )

    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": VLM_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]
                }],
                "temperature": 0.3,
                "max_tokens": 500,
            },
            timeout=120)

        if r.status_code == 200:
            d = r.json()
            content = d["choices"][0]["message"].get("content", "")
            if not content:
                return None, "empty_response"
            
            # Extract JSON from response
            import re
            # Try code block
            m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if m:
                scores = json.loads(m.group(1))
            else:
                # Try raw JSON
                m = re.search(r'\{[^{}]*"overall"[^{}]*\}', content, re.DOTALL)
                if m:
                    scores = json.loads(m.group(0))
                else:
                    return None, f"no_json_in: {content[:200]}"

            return scores, "success"
        else:
            return None, f"http_{r.status_code}"

    except Exception as e:
        return None, str(e)

def run_vlm_evaluation(results_dir):
    """Run VLM evaluation on all rendered floorplans."""
    results_dir = Path(results_dir)
    prompts_file = results_dir.parent / "prompts" / "layout_prompts.json"

    with open(prompts_file) as f:
        prompts = {p['id']: p for p in json.load(f)}

    all_scores = []

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name in ('__pycache__',):
            continue
        model_name = model_dir.name

        for prompt_dir in sorted(model_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue
            pid = prompt_dir.name

            fp_path = prompt_dir / "floorplan.png"
            vlm_path = prompt_dir / "vlm_scores.json"

            if not fp_path.exists():
                continue
            if vlm_path.exists():
                # Already evaluated
                with open(vlm_path) as f:
                    existing = json.load(f)
                all_scores.append(existing)
                continue

            prompt_data = prompts.get(pid, {})

            # Check source — only evaluate real API results and mock baseline
            layout_file = prompt_dir / "layout.json"
            if layout_file.exists():
                with open(layout_file) as f:
                    ld = json.load(f)
                source = ld.get("source", "unknown")
            else:
                source = "unknown"

            print(f"  VLM judging: {model_name}/{pid} (source={source})", flush=True)

            scores, status = vlm_judge(str(fp_path), prompt_data)

            result = {
                "model": model_name,
                "prompt_id": pid,
                "source": source,
                "vlm_status": status,
                "vlm_scores": scores,
            }

            with open(vlm_path, "w") as f:
                json.dump(result, f, indent=2)

            all_scores.append(result)
            time.sleep(10)  # Rate limit

    # Summary
    print("\n" + "="*70, flush=True)
    print("VLM JUDGE RESULTS", flush=True)
    print("="*70, flush=True)

    model_scores = {}
    for s in all_scores:
        m = s["model"]
        if m not in model_scores:
            model_scores[m] = []
        if s.get("vlm_scores"):
            model_scores[m].append(s["vlm_scores"])

    for model, scores_list in model_scores.items():
        if not scores_list:
            print(f"\n{model}: no VLM scores", flush=True)
            continue

        dims = ["completeness", "spatial", "no_collisions", "walkability", "functional", "wall_use", "overall"]
        print(f"\n{model} ({len(scores_list)} scored):", flush=True)
        for dim in dims:
            vals = [s.get(dim, 0) for s in scores_list if isinstance(s.get(dim), (int, float))]
            if vals:
                import numpy as np
                print(f"  {dim:16s}: {np.mean(vals):.1f}/10", flush=True)

    # Save summary
    with open(results_dir / "vlm_judge_summary.json", "w") as f:
        json.dump(all_scores, f, indent=2)

    print(f"\nDone! Saved to {results_dir}/vlm_judge_summary.json", flush=True)

if __name__ == "__main__":
    results_dir = Path(__file__).parent / "../results"
    run_vlm_evaluation(results_dir)
