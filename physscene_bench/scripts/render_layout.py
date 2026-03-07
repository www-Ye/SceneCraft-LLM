#!/usr/bin/env python3
"""
Render 2D floorplan + MuJoCo 3D top-down view for each generated layout.
Saves: floorplan.png + mujoco_topdown.png per result folder.
"""
import json, os, sys, math, tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

os.environ['MUJOCO_GL'] = 'osmesa'

# Category colors
CATEGORY_COLORS = {
    'sofa': '#4A90D9', 'couch': '#4A90D9',
    'bed': '#8B5CF6', 'television': '#10B981', 'tv': '#10B981', 'tv_stand': '#10B981',
    'coffee_table': '#F59E0B', 'dining_table': '#F59E0B', 'table': '#F59E0B',
    'chair': '#EF4444', 'armchair': '#DC2626',
    'desk': '#6366F1', 'cabinet': '#8B4513', 'bookshelf': '#8B4513', 'wardrobe': '#A0522D',
    'nightstand': '#D2691E', 'dresser': '#CD853F',
    'lamp': '#FFD700', 'floor_lamp': '#FFD700',
    'rug': '#90EE90', 'carpet': '#90EE90',
}

def get_color(category):
    cat_lower = category.lower().replace(' ', '_')
    for key, color in CATEGORY_COLORS.items():
        if key in cat_lower:
            return color
    return '#999999'

def render_floorplan(layout, room_size, save_path, title=""):
    """Render 2D floorplan with labeled furniture and facing arrows."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    room_w, room_h = room_size

    # Draw room
    room_rect = patches.Rectangle((0, 0), room_w, room_h, linewidth=2,
                                   edgecolor='black', facecolor='#F5F5F0')
    ax.add_patch(room_rect)

    # Grid
    for x in range(int(room_w) + 1):
        ax.axvline(x, color='#E0E0E0', linewidth=0.5, linestyle='--')
    for y in range(int(room_h) + 1):
        ax.axhline(y, color='#E0E0E0', linewidth=0.5, linestyle='--')

    objects = layout.get('objects', [])
    for i, obj in enumerate(objects):
        pos = obj['position']
        dims = obj.get('dimensions', {'width': 0.5, 'depth': 0.5, 'height': 0.5})
        rotation = obj.get('rotation', 0)
        cat = obj['category']
        color = get_color(cat)

        w, d = dims['width'], dims['depth']
        rot_rad = math.radians(rotation)

        # Draw rotated rectangle
        import matplotlib.transforms as mtransforms
        rect = patches.Rectangle((-w/2, -d/2), w, d,
                                  linewidth=1.5, edgecolor='black',
                                  facecolor=color, alpha=0.7)
        t = mtransforms.Affine2D().rotate_deg(rotation).translate(pos[0], pos[1])
        rect.set_transform(t + ax.transData)
        ax.add_patch(rect)

        # Label
        ax.text(pos[0], pos[1], f"{cat}\n{i}", ha='center', va='center',
                fontsize=6, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.3))

        # Facing arrow (rotation=0 means facing +Y)
        arrow_len = min(w, d) * 0.4
        dx = arrow_len * math.sin(rot_rad)
        dy = arrow_len * math.cos(rot_rad)
        ax.annotate('', xy=(pos[0]+dx, pos[1]+dy), xytext=(pos[0], pos[1]),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # Setup axes
    margin = 0.3
    ax.set_xlim(-margin, room_w + margin)
    ax.set_ylim(-margin, room_h + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(title or 'Floor Plan', fontsize=12, fontweight='bold')

    # Legend
    seen = set()
    legend_handles = []
    for obj in objects:
        cat = obj['category']
        if cat not in seen:
            seen.add(cat)
            legend_handles.append(patches.Patch(color=get_color(cat), label=cat))
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def render_mujoco_topdown(layout, room_size, save_path, title=""):
    """Render MuJoCo 3D top-down view."""
    import mujoco
    import xml.etree.ElementTree as ET

    room_w, room_h = room_size
    objects = layout.get('objects', [])

    # Build MuJoCo XML
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="degree")

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1200", offheight="900")
    headlight = ET.SubElement(visual, "headlight", diffuse="0.8 0.8 0.8", ambient="0.3 0.3 0.3")

    worldbody = ET.SubElement(root, "worldbody")

    # Light
    ET.SubElement(worldbody, "light", pos=f"{room_w/2} {room_h/2} 5",
                  dir="0 0 -1", diffuse="0.9 0.9 0.9")

    # Floor
    ET.SubElement(worldbody, "geom", name="floor", type="plane",
                  size=f"{room_w/2+1} {room_h/2+1} 0.01",
                  pos=f"{room_w/2} {room_h/2} 0",
                  rgba="0.95 0.93 0.88 1")

    # Low walls
    wall_h = 0.15
    walls = [
        ("wall_south", f"{room_w/2} 0.025 {wall_h/2}", f"{room_w/2} 0.025 {wall_h/2}"),
        ("wall_north", f"{room_w/2} {room_h-0.025} {wall_h/2}", f"{room_w/2} 0.025 {wall_h/2}"),
        ("wall_west", f"0.025 {room_h/2} {wall_h/2}", f"0.025 {room_h/2} {wall_h/2}"),
        ("wall_east", f"{room_w-0.025} {room_h/2} {wall_h/2}", f"0.025 {room_h/2} {wall_h/2}"),
    ]
    for name, pos, size in walls:
        ET.SubElement(worldbody, "geom", name=name, type="box",
                      pos=pos, size=size, rgba="0.85 0.82 0.78 1")

    # Furniture
    cat_rgba = {
        'sofa': '0.29 0.56 0.85 1', 'bed': '0.55 0.36 0.96 1',
        'television': '0.06 0.73 0.51 1', 'tv_stand': '0.06 0.73 0.51 1', 'tv': '0.06 0.73 0.51 1',
        'coffee_table': '0.96 0.62 0.04 1', 'dining_table': '0.96 0.62 0.04 1', 'table': '0.96 0.62 0.04 1',
        'chair': '0.94 0.27 0.27 1', 'armchair': '0.86 0.15 0.15 1',
        'desk': '0.39 0.40 0.95 1', 'cabinet': '0.55 0.27 0.07 1',
        'wardrobe': '0.63 0.32 0.18 1', 'nightstand': '0.82 0.41 0.12 1',
        'dresser': '0.80 0.52 0.25 1', 'lamp': '1.0 0.84 0.0 1',
        'rug': '0.56 0.93 0.56 0.5', 'bookshelf': '0.55 0.27 0.07 1',
    }

    for i, obj in enumerate(objects):
        pos = obj['position']
        dims = obj.get('dimensions', {'width':0.5, 'depth':0.5, 'height':0.5})
        rotation = obj.get('rotation', 0)
        cat = obj['category'].lower().replace(' ', '_')

        rgba = cat_rgba.get(cat, '0.6 0.6 0.6 1')
        h = dims['height']

        # Rotation quaternion (around Z axis)
        rot_rad = math.radians(rotation)
        qw = math.cos(rot_rad / 2)
        qz = math.sin(rot_rad / 2)

        body = ET.SubElement(worldbody, "body",
                             name=f"obj_{i}_{obj['category']}",
                             pos=f"{pos[0]} {pos[1]} {h/2}",
                             quat=f"{qw} 0 0 {qz}")
        ET.SubElement(body, "geom", name=f"geom_{i}", type="box",
                      size=f"{dims['width']/2} {dims['depth']/2} {h/2}",
                      rgba=rgba)

    xml_str = ET.tostring(root, encoding='unicode')

    # Render
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_str)
        xml_path = f.name

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        renderer = mujoco.Renderer(model, width=1200, height=900)

        # Top-down camera
        renderer.update_scene(data)
        cam = renderer.scene.camera[0]

        # Manual camera setup
        scene = renderer.scene
        scene.camera[0].pos[0] = room_w / 2
        scene.camera[0].pos[1] = room_h / 2
        scene.camera[0].pos[2] = max(room_w, room_h) * 1.2

        # Use mujoco camera
        camera = mujoco.MjvCamera()
        camera.lookat[0] = room_w / 2
        camera.lookat[1] = room_h / 2
        camera.lookat[2] = 0
        camera.distance = max(room_w, room_h) * 1.1
        camera.elevation = -90  # Top-down
        camera.azimuth = 0

        renderer.update_scene(data, camera)
        img = renderer.render()

        from PIL import Image
        pil_img = Image.fromarray(img)
        if title:
            # Add title text
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()
            draw.text((10, 10), title, fill=(0,0,0), font=font)

        pil_img.save(save_path)
        renderer.close()
    finally:
        os.unlink(xml_path)

def render_all_results(results_dir):
    """Render all layouts in the results directory."""
    results_dir = Path(results_dir)
    prompts_file = results_dir.parent / "prompts" / "layout_prompts.json"

    with open(prompts_file) as f:
        prompts = {p['id']: p for p in json.load(f)}

    rendered = 0
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == '__pycache__':
            continue
        model_name = model_dir.name

        for prompt_dir in sorted(model_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue
            pid = prompt_dir.name

            layout_file = prompt_dir / "layout.json"
            if not layout_file.exists():
                continue

            with open(layout_file) as f:
                data = json.load(f)

            layout = data.get('layout', data)
            source = data.get('source', 'unknown')
            prompt_data = prompts.get(pid, {})
            room_size = prompt_data.get('room_size', [5, 4])
            room_type = prompt_data.get('room_type', pid)

            title = f"{model_name} | {pid} ({room_type}) | source={source}"

            # Render 2D floorplan
            fp_path = prompt_dir / "floorplan.png"
            if not fp_path.exists():
                try:
                    render_floorplan(layout, room_size, str(fp_path), title)
                    print(f"  ✅ {model_name}/{pid}/floorplan.png", flush=True)
                    rendered += 1
                except Exception as e:
                    print(f"  ❌ {model_name}/{pid}/floorplan.png: {e}", flush=True)

            # Render MuJoCo 3D top-down
            mj_path = prompt_dir / "mujoco_topdown.png"
            if not mj_path.exists():
                try:
                    render_mujoco_topdown(layout, room_size, str(mj_path), title)
                    print(f"  ✅ {model_name}/{pid}/mujoco_topdown.png", flush=True)
                    rendered += 1
                except Exception as e:
                    print(f"  ❌ {model_name}/{pid}/mujoco_topdown.png: {e}", flush=True)

    print(f"\nRendered {rendered} images total.", flush=True)

if __name__ == "__main__":
    results_dir = Path(__file__).parent / "../results"
    print("Rendering all layouts...", flush=True)
    render_all_results(results_dir)
