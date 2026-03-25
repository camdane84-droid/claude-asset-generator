# ──────────────────────────────────────────────────────────────────────
#  Image-to-3D Asset Generator  —  Blender Add-on  v3.0
#  Works with Blender 3.x / 4.x / 5.x
#  Uses AI depth estimation (Depth Anything V2) for real surface detail.
# ──────────────────────────────────────────────────────────────────────

bl_info = {
    "name": "Image to 3D Asset Generator",
    "author": "Claude Asset Generator",
    "version": (3, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Img2Mesh",
    "description": "Generate high-detail 3D meshes from 2D images using AI depth estimation",
    "category": "Object",
}

import sys
import os
import subprocess
import math
import struct
import tempfile
import time

import bpy
import bmesh
from bpy.props import (
    StringProperty,
    IntProperty,
    EnumProperty,
    FloatProperty,
    BoolProperty,
)
from bpy.types import (
    Operator,
    Panel,
    PropertyGroup,
)
from mathutils import Vector, Matrix

# ── Path to companion depth script ───────────────────────────────────
_ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
_DEPTH_SCRIPT = os.path.join(_ADDON_DIR, "generate_depth.py")

# Add Pillow paths
_extra_paths = []
_appdata = os.environ.get("APPDATA", "")
if _appdata:
    _extra_paths.append(os.path.join(_appdata, "Python", "Python311", "site-packages"))
_extra_paths.append(r"C:\Users\cdall\AppData\Roaming\Python\Python311\site-packages")
_extra_paths.append(r"C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\lib\site-packages")
try:
    import site as _site
    _sp = _site.getusersitepackages()
    if isinstance(_sp, str):
        _extra_paths.append(_sp)
    elif isinstance(_sp, list):
        _extra_paths.extend(_sp)
    _extra_paths.extend(_site.getsitepackages() if hasattr(_site, 'getsitepackages') else [])
except Exception:
    pass
for _p in _extra_paths:
    if isinstance(_p, str) and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def _ensure_pillow():
    try:
        if 'PIL' in sys.modules and sys.modules['PIL'] is None:
            del sys.modules['PIL']
        from PIL import Image
        return True
    except ImportError:
        try:
            import importlib
            importlib.invalidate_caches()
            from PIL import Image
            return True
        except ImportError:
            return False


def _find_system_python():
    """Find system Python (not Blender's) for running the depth script."""
    candidates = [
        r"C:\Users\cdall\AppData\Local\Python\pythoncore-3.14-64\python.exe",
        r"C:\Users\cdall\AppData\Local\Microsoft\WindowsApps\python.exe",
        r"C:\Users\cdall\AppData\Local\Python\bin\python.exe",
    ]
    # Also try PATH
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Fallback: try 'python' from PATH
    return "python"


# ── Stage 1: Generate AI Depth Map ───────────────────────────────────

def generate_ai_depth(image_path, output_path=None):
    """
    Run the Depth Anything V2 companion script via system Python.
    Returns path to the generated depth map PNG.
    """
    if output_path is None:
        base, _ = os.path.splitext(image_path)
        output_path = base + "_depth.png"

    # Check if depth map already exists and is newer than the image
    if os.path.isfile(output_path):
        if os.path.getmtime(output_path) >= os.path.getmtime(image_path):
            print(f"Using cached depth map: {output_path}")
            return output_path

    python = _find_system_python()
    script = _DEPTH_SCRIPT

    if not os.path.isfile(script):
        raise FileNotFoundError(
            f"Depth generation script not found: {script}\n"
            f"Place generate_depth.py next to this add-on file."
        )

    print(f"Running depth estimation with: {python}")
    print(f"Script: {script}")
    print(f"Input: {image_path}")

    result = subprocess.run(
        [python, script, image_path, output_path],
        capture_output=True,
        text=True,
        timeout=300,  # 5 min timeout
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(
            f"Depth estimation failed (exit code {result.returncode}):\n"
            f"{result.stderr}"
        )

    if not os.path.isfile(output_path):
        raise RuntimeError(f"Depth map was not created: {output_path}")

    return output_path


# ── Stage 2: Load & Process Maps ─────────────────────────────────────

def load_maps(image_path, depth_path):
    """
    Load the source image and AI depth map.
    Returns (img_w, img_h, alpha_mask_2d, depth_map_2d)
    """
    from PIL import Image, ImageFilter

    img = Image.open(image_path).convert("RGBA")
    w, h = img.size

    # ── Alpha mask ───────────────────────────────────────────────────
    alpha = img.split()[3]
    alpha_data = list(alpha.getdata())
    has_alpha = any(p < 240 for p in alpha_data) and any(p > 15 for p in alpha_data)

    if has_alpha:
        # Image has real transparency — use it directly
        mask_flat = [p / 255.0 for p in alpha_data]
    else:
        # No alpha channel — build silhouette from the depth map itself.
        # The AI depth map already segments the subject: background pixels
        # have low/consistent depth, subject pixels have varied/higher depth.
        # Also use luminance-based background detection as a fallback.
        print("  No alpha channel — building silhouette from depth map + background detection")

        gray = img.convert("L")

        # Sample corners to detect background color
        margin = 5
        corner_pixels = []
        for cy in [0, h - 1]:
            for cx in [0, w - 1]:
                for dy in range(margin):
                    for dx in range(margin):
                        px = min(max(cx + dx if cx == 0 else cx - dx, 0), w - 1)
                        py = min(max(cy + dy if cy == 0 else cy - dy, 0), h - 1)
                        corner_pixels.append(gray.getpixel((px, py)))

        bg_val = sum(corner_pixels) / len(corner_pixels)
        bg_std = (sum((p - bg_val) ** 2 for p in corner_pixels) / len(corner_pixels)) ** 0.5
        # Threshold: pixels that differ from bg by more than bg_std + tolerance
        tolerance = max(bg_std * 2, 25)

        gray_data = list(gray.getdata())
        mask_flat = []
        for p in gray_data:
            diff = abs(p - bg_val)
            if diff > tolerance:
                mask_flat.append(1.0)
            elif diff > tolerance * 0.5:
                # Soft edge
                mask_flat.append((diff - tolerance * 0.5) / (tolerance * 0.5))
            else:
                mask_flat.append(0.0)

        # Also use the depth map to refine: if depth is very low, it's background
        depth_img_for_mask = Image.open(depth_path).convert("L")
        if depth_img_for_mask.size != (w, h):
            depth_img_for_mask = depth_img_for_mask.resize((w, h), Image.LANCZOS)
        depth_for_mask = list(depth_img_for_mask.getdata())
        d_min_m = min(depth_for_mask)
        d_max_m = max(depth_for_mask)
        d_range_m = max(d_max_m - d_min_m, 1)

        # Combine: pixel must pass BOTH luminance and depth checks
        # Low depth = far away = likely background
        depth_threshold = 0.08  # bottom 8% of depth range = background
        for i in range(w * h):
            d_norm = (depth_for_mask[i] - d_min_m) / d_range_m
            if d_norm < depth_threshold:
                mask_flat[i] *= d_norm / depth_threshold  # fade out
            # Reinforce: if depth is strong, trust it even if luminance says bg
            if d_norm > 0.15:
                mask_flat[i] = max(mask_flat[i], min(1.0, d_norm * 1.5))

        # Smooth the mask to reduce noise
        mask_img = Image.new("L", (w, h))
        mask_img.putdata([int(max(0, min(255, v * 255))) for v in mask_flat])
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=3))
        # Slight dilate then erode to fill small holes
        mask_img = mask_img.filter(ImageFilter.MaxFilter(5))
        mask_img = mask_img.filter(ImageFilter.MinFilter(3))
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2))
        mask_flat = [p / 255.0 for p in list(mask_img.getdata())]

    print(f"  Mask coverage: {sum(1 for m in mask_flat if m > 0.5)}/{w * h} pixels "
          f"({100 * sum(1 for m in mask_flat if m > 0.5) / (w * h):.0f}%)")

    # ── Depth map (AI-generated) ─────────────────────────────────────
    depth_img = Image.open(depth_path).convert("L")
    if depth_img.size != (w, h):
        depth_img = depth_img.resize((w, h), Image.LANCZOS)

    depth_data = list(depth_img.getdata())
    d_min = min(depth_data)
    d_max = max(depth_data)
    d_range = max(d_max - d_min, 1)
    depth_flat = [(d - d_min) / d_range for d in depth_data]

    # Apply mask to depth
    for i in range(w * h):
        depth_flat[i] *= mask_flat[i]

    # Convert to 2D
    alpha_2d = [mask_flat[y * w:(y + 1) * w] for y in range(h)]
    depth_2d = [depth_flat[y * w:(y + 1) * w] for y in range(h)]

    return w, h, alpha_2d, depth_2d


# ── Stage 3: High-Poly Mesh from Depth Map ───────────────────────────

def create_mesh_from_depth(alpha_mask, depth_map, img_w, img_h,
                           depth_factor=0.5, grid_resolution=512,
                           back_mode='mirror'):
    """
    Create a high-polygon 3D mesh by displacing a dense grid using the
    AI-generated depth map. This is where the magic happens.

    For 5M+ polygons, grid_resolution=1024 gives ~2M quads = ~4M tris
    on a full-coverage image (more with front+back+sides).
    """
    # Determine grid dimensions maintaining aspect ratio
    aspect = img_w / img_h
    if aspect >= 1.0:
        grid_x = grid_resolution
        grid_y = max(4, int(grid_resolution / aspect))
    else:
        grid_y = grid_resolution
        grid_x = max(4, int(grid_resolution * aspect))

    # World size — roughly 2 units on the long axis
    size_x = 2.0 if aspect >= 1.0 else 2.0 * aspect
    size_y = 2.0 / aspect if aspect >= 1.0 else 2.0

    print(f"Grid: {grid_x}x{grid_y} = {grid_x * grid_y} vertices per face")

    # ── Delete existing generated object if any ──────────────────────
    old = bpy.data.objects.get("Img2Mesh_Generated")
    if old:
        bpy.data.objects.remove(old, do_unlink=True)

    mesh = bpy.data.meshes.new("Img2Mesh_Generated")
    obj = bpy.data.objects.new("Img2Mesh_Generated", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bm = bmesh.new()

    # ── Pre-compute all samples ──────────────────────────────────────
    mask_threshold = 0.05
    front_verts = {}
    back_verts = {}

    t0 = time.time()

    for gy in range(grid_y + 1):
        v = gy / grid_y
        fy = v * (img_h - 1)
        y0 = int(fy)
        y1 = min(y0 + 1, img_h - 1)
        sy = fy - y0

        for gx in range(grid_x + 1):
            u = gx / grid_x
            fx = u * (img_w - 1)
            x0 = int(fx)
            x1 = min(x0 + 1, img_w - 1)
            sx = fx - x0

            # Bilinear sample mask
            m00 = alpha_mask[y0][x0]
            m10 = alpha_mask[y0][x1]
            m01 = alpha_mask[y1][x0]
            m11 = alpha_mask[y1][x1]
            mask_val = (m00 * (1 - sx) + m10 * sx) * (1 - sy) + \
                       (m01 * (1 - sx) + m11 * sx) * sy

            if mask_val < mask_threshold:
                continue

            # Bilinear sample depth
            d00 = depth_map[y0][x0]
            d10 = depth_map[y0][x1]
            d01 = depth_map[y1][x0]
            d11 = depth_map[y1][x1]
            depth_val = (d00 * (1 - sx) + d10 * sx) * (1 - sy) + \
                        (d01 * (1 - sx) + d11 * sx) * sy

            # World position
            wx = (u - 0.5) * size_x
            wy = -(v - 0.5) * size_y

            # Front: displaced outward by depth
            fz = depth_val * depth_factor
            front_verts[(gx, gy)] = bm.verts.new((wx, wy, fz))

            # Back face
            if back_mode == 'mirror':
                bz = -depth_val * depth_factor
            elif back_mode == 'half':
                bz = -depth_val * depth_factor * 0.3
            else:
                bz = -depth_factor * 0.05

            back_verts[(gx, gy)] = bm.verts.new((wx, wy, bz))

    bm.verts.ensure_lookup_table()
    print(f"  Vertices created: {len(bm.verts)} in {time.time() - t0:.1f}s")

    if len(front_verts) < 4:
        bm.free()
        raise ValueError("Not enough visible area to generate mesh.")

    # ── Create faces ─────────────────────────────────────────────────
    t0 = time.time()
    face_count = 0

    for gy in range(grid_y):
        for gx in range(grid_x):
            # Front face quad
            k00 = (gx, gy)
            k10 = (gx + 1, gy)
            k11 = (gx + 1, gy + 1)
            k01 = (gx, gy + 1)

            fv = [front_verts.get(k) for k in (k00, k10, k11, k01)]
            if all(fv):
                try:
                    bm.faces.new(fv)
                    face_count += 1
                except Exception:
                    pass

            # Back face (reversed winding)
            bv = [back_verts.get(k) for k in (k00, k10, k11, k01)]
            if all(bv):
                try:
                    bm.faces.new([bv[0], bv[3], bv[2], bv[1]])
                    face_count += 1
                except Exception:
                    pass

    # ── Stitch boundary edges ────────────────────────────────────────
    _stitch_boundary(bm, front_verts, back_verts, grid_x, grid_y)

    print(f"  Faces created: {face_count} in {time.time() - t0:.1f}s")

    # ── Triangulate ──────────────────────────────────────────────────
    t0 = time.time()
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    print(f"  Triangulated: {len(bm.faces)} tris in {time.time() - t0:.1f}s")

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    return obj


def _stitch_boundary(bm, front_verts, back_verts, grid_x, grid_y):
    """Connect front and back mesh along the silhouette boundary edges."""
    def _has(gx, gy):
        return (gx, gy) in front_verts

    # Horizontal edges
    for gy in range(grid_y + 1):
        for gx in range(grid_x):
            if not (_has(gx, gy) and _has(gx + 1, gy)):
                continue
            above = (gy > 0 and _has(gx, gy - 1) and _has(gx + 1, gy - 1))
            below = (gy < grid_y and _has(gx, gy + 1) and _has(gx + 1, gy + 1))
            if gy == 0: above = False
            if gy == grid_y: below = False

            if not above or not below:
                a, b = (gx, gy), (gx + 1, gy)
                fa, fb = front_verts[a], front_verts[b]
                ba, bb = back_verts[a], back_verts[b]
                try:
                    if not above:
                        bm.faces.new([fa, fb, bb, ba])
                    else:
                        bm.faces.new([fb, fa, ba, bb])
                except Exception:
                    pass

    # Vertical edges
    for gy in range(grid_y):
        for gx in range(grid_x + 1):
            if not (_has(gx, gy) and _has(gx, gy + 1)):
                continue
            left = (gx > 0 and _has(gx - 1, gy) and _has(gx - 1, gy + 1))
            right = (gx < grid_x and _has(gx + 1, gy) and _has(gx + 1, gy + 1))
            if gx == 0: left = False
            if gx == grid_x: right = False

            if not left or not right:
                a, b = (gx, gy), (gx, gy + 1)
                fa, fb = front_verts[a], front_verts[b]
                ba, bb = back_verts[a], back_verts[b]
                try:
                    if not left:
                        bm.faces.new([fb, fa, ba, bb])
                    else:
                        bm.faces.new([fa, fb, bb, ba])
                except Exception:
                    pass


# ── Stage 4: Optimize ────────────────────────────────────────────────

def optimize_mesh(obj, target_tris=0, apply_smooth=True):
    """
    Clean up mesh. Only decimate if target_tris > 0 and mesh exceeds it.
    """
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Clean geometry
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Decimate only if over budget
    current_tris = len(obj.data.polygons)
    if target_tris > 0 and current_tris > target_tris:
        ratio = target_tris / current_tris
        dec = obj.modifiers.new(name="Decimate", type='DECIMATE')
        dec.decimate_type = 'COLLAPSE'
        dec.ratio = ratio
        _apply_modifier(obj, dec.name)

    # Light smooth
    if apply_smooth:
        sm = obj.modifiers.new(name="Smooth", type='SMOOTH')
        sm.factor = 0.3
        sm.iterations = 2
        _apply_modifier(obj, sm.name)

    # Weighted normals
    try:
        wn = obj.modifiers.new(name="WeightedNormal", type='WEIGHTED_NORMAL')
        wn.weight = 50
        wn.keep_sharp = True
        _apply_modifier(obj, wn.name)
    except Exception:
        pass

    # Shade smooth
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    if hasattr(bpy.ops.object, 'shade_smooth_by_angle'):
        bpy.ops.object.shade_smooth_by_angle()
    elif hasattr(bpy.ops.object, 'shade_smooth'):
        bpy.ops.object.shade_smooth()


def _apply_modifier(obj, mod_name):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        bpy.ops.object.modifier_apply(modifier=mod_name)
    except Exception:
        if mod_name in obj.modifiers:
            obj.modifiers.remove(obj.modifiers[mod_name])


# ── Stage 5: UV & Texture ───────────────────────────────────────────

def apply_texture(obj, image_path, img_w, img_h):
    """Planar UV projection + material from source image."""
    mesh = obj.data
    aspect = img_w / img_h
    size_x = 2.0 if aspect >= 1.0 else 2.0 * aspect
    size_y = 2.0 / aspect if aspect >= 1.0 else 2.0

    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")
    uv_layer = mesh.uv_layers.active

    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            vert = mesh.vertices[mesh.loops[loop_idx].vertex_index]
            co = vert.co
            u = (co.x / size_x) + 0.5
            v = (-co.y / size_y) + 0.5
            uv_layer.data[loop_idx].uv = (
                max(0.0, min(1.0, u)),
                max(0.0, min(1.0, v)),
            )

    # Load image
    img_name = os.path.basename(image_path)
    if img_name in bpy.data.images:
        ref_img = bpy.data.images[img_name]
    else:
        ref_img = bpy.data.images.load(image_path)

    # Material
    mat = bpy.data.materials.new(name="Img2Mesh_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in nodes:
        nodes.remove(n)

    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Roughness'].default_value = 0.7

    tex = nodes.new(type='ShaderNodeTexImage')
    tex.location = (-400, 0)
    tex.image = ref_img

    out = nodes.new(type='ShaderNodeOutputMaterial')
    out.location = (300, 0)

    links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    try:
        links.new(tex.outputs['Alpha'], bsdf.inputs['Alpha'])
    except Exception:
        pass
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


# ── Stage 6: Export ──────────────────────────────────────────────────

def export_asset(obj, filepath, format='FBX'):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    if format == 'FBX':
        if not filepath.lower().endswith('.fbx'):
            filepath += '.fbx'
        bpy.ops.export_scene.fbx(
            filepath=filepath, use_selection=True,
            embed_textures=True, path_mode='COPY', mesh_smooth_type='FACE',
        )
    elif format == 'GLB':
        if not filepath.lower().endswith('.glb'):
            filepath += '.glb'
        bpy.ops.export_scene.gltf(
            filepath=filepath, use_selection=True,
            export_format='GLB', export_image_format='AUTO',
        )
    elif format == 'OBJ':
        if not filepath.lower().endswith('.obj'):
            filepath += '.obj'
        try:
            bpy.ops.wm.obj_export(filepath=filepath, export_selected_objects=True)
        except AttributeError:
            bpy.ops.export_scene.obj(filepath=filepath, use_selection=True)


# ════════════════════════════════════════════════════════════════════
#  OPERATORS
# ════════════════════════════════════════════════════════════════════

class IMG2MESH_OT_generate_depth(Operator):
    """Generate AI depth map from the image (runs outside Blender)"""
    bl_idname = "img2mesh.generate_depth"
    bl_label = "Generate AI Depth Map"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.img2mesh_props
        if not props.image_path or not os.path.isfile(props.image_path):
            self.report({'ERROR'}, "Select a valid image file first.")
            return {'CANCELLED'}

        self.report({'INFO'}, "Running Depth Anything V2 (this may take 15-60s first time)...")

        try:
            depth_path = generate_ai_depth(props.image_path)
            props.depth_path = depth_path
            self.report({'INFO'}, f"Depth map saved: {depth_path}")
        except Exception as e:
            self.report({'ERROR'}, f"Depth generation failed: {e}")
            return {'CANCELLED'}

        return {'FINISHED'}


class IMG2MESH_OT_generate(Operator):
    """Generate a high-detail 3D mesh from image + depth map"""
    bl_idname = "img2mesh.generate"
    bl_label = "Generate 3D Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.img2mesh_props

        if not props.image_path or not os.path.isfile(props.image_path):
            self.report({'ERROR'}, "Select a valid image file.")
            return {'CANCELLED'}

        # ── Depth map: generate if missing ───────────────────────────
        depth_path = props.depth_path
        if not depth_path or not os.path.isfile(depth_path):
            # Auto-check for existing depth map
            base, _ = os.path.splitext(props.image_path)
            auto_path = base + "_depth.png"
            if os.path.isfile(auto_path):
                depth_path = auto_path
                props.depth_path = auto_path
            else:
                self.report({'ERROR'},
                    "No depth map found. Click 'Generate AI Depth Map' first.")
                return {'CANCELLED'}

        # ── Load maps ────────────────────────────────────────────────
        self.report({'INFO'}, "Loading image and depth map...")
        try:
            img_w, img_h, alpha_mask, depth_map = load_maps(
                props.image_path, depth_path
            )
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load maps: {e}")
            return {'CANCELLED'}

        # ── Build mesh ───────────────────────────────────────────────
        self.report({'INFO'},
            f"Building mesh (grid {props.grid_resolution}x...)...")
        try:
            obj = create_mesh_from_depth(
                alpha_mask, depth_map, img_w, img_h,
                depth_factor=props.depth_factor,
                grid_resolution=props.grid_resolution,
                back_mode=props.back_mode,
            )
        except Exception as e:
            self.report({'ERROR'}, f"Mesh generation failed: {e}")
            return {'CANCELLED'}

        # ── Optimize ─────────────────────────────────────────────────
        self.report({'INFO'}, "Optimizing mesh...")
        try:
            optimize_mesh(
                obj,
                target_tris=props.target_tris,
                apply_smooth=props.apply_smooth,
            )
        except Exception as e:
            self.report({'WARNING'}, f"Optimization issue: {e}")

        # ── Texture ──────────────────────────────────────────────────
        self.report({'INFO'}, "Applying texture...")
        try:
            apply_texture(obj, props.image_path, img_w, img_h)
        except Exception as e:
            self.report({'WARNING'}, f"Texture issue: {e}")

        # Stats
        tri_count = len(obj.data.polygons)
        vert_count = len(obj.data.vertices)
        self.report({'INFO'},
            f"Done! {vert_count:,} verts, {tri_count:,} tris")

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        try:
            bpy.ops.view3d.view_selected()
        except Exception:
            pass

        return {'FINISHED'}


class IMG2MESH_OT_export(Operator):
    """Export the generated mesh"""
    bl_idname = "img2mesh.export"
    bl_label = "Export Asset"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.img2mesh_props
        obj = bpy.context.active_object
        if obj is None or obj.type != 'MESH':
            obj = bpy.data.objects.get("Img2Mesh_Generated")
            if obj is None:
                self.report({'ERROR'}, "No mesh found. Generate one first.")
                return {'CANCELLED'}

        if not props.export_path:
            base = os.path.splitext(props.image_path)[0] if props.image_path else "export"
            props.export_path = base + f".{props.export_format.lower()}"

        try:
            export_asset(obj, props.export_path, format=props.export_format)
            self.report({'INFO'}, f"Exported: {props.export_path}")
        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}


# ════════════════════════════════════════════════════════════════════
#  PROPERTIES
# ════════════════════════════════════════════════════════════════════

class IMG2MESH_Properties(PropertyGroup):
    image_path: StringProperty(
        name="Image",
        description="Path to the reference image",
        default="",
        subtype='FILE_PATH',
    )  # type: ignore

    depth_path: StringProperty(
        name="Depth Map",
        description="Path to AI-generated depth map (auto-generated)",
        default="",
        subtype='FILE_PATH',
    )  # type: ignore

    grid_resolution: IntProperty(
        name="Grid Resolution",
        description="Vertices along longest axis. 512=~500k tris, 1024=~2M tris, 2048=~8M tris",
        default=512,
        min=64,
        max=2048,
        step=64,
    )  # type: ignore

    target_tris: IntProperty(
        name="Max Triangles",
        description="Maximum triangle budget (0 = no limit, keep all detail)",
        default=0,
        min=0,
        max=10000000,
        step=100000,
    )  # type: ignore

    depth_factor: FloatProperty(
        name="Depth Strength",
        description="How much the depth map displaces the surface",
        default=0.5,
        min=0.05,
        max=3.0,
    )  # type: ignore

    back_mode: EnumProperty(
        name="Back Face",
        description="How to generate the back of the mesh",
        items=[
            ('mirror', 'Mirror', 'Mirror front depth to back (fully 3D)'),
            ('half', 'Partial', 'Back has 30% of front depth'),
            ('flat', 'Flat', 'Flat back (relief style)'),
        ],
        default='mirror',
    )  # type: ignore

    apply_smooth: BoolProperty(
        name="Smooth",
        description="Apply light smoothing pass",
        default=True,
    )  # type: ignore

    export_format: EnumProperty(
        name="Format",
        items=[
            ('FBX', 'FBX', 'Autodesk FBX'),
            ('GLB', 'GLB', 'glTF Binary'),
            ('OBJ', 'OBJ', 'Wavefront OBJ'),
        ],
        default='FBX',
    )  # type: ignore

    export_path: StringProperty(
        name="Export Path",
        default="",
        subtype='FILE_PATH',
    )  # type: ignore


# ════════════════════════════════════════════════════════════════════
#  UI PANEL
# ════════════════════════════════════════════════════════════════════

class IMG2MESH_PT_main_panel(Panel):
    bl_label = "Image → 3D Asset v3"
    bl_idname = "IMG2MESH_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Img2Mesh"

    def draw(self, context):
        layout = self.layout
        props = context.scene.img2mesh_props

        # ── Input ────────────────────────────────────────────────────
        box = layout.box()
        box.label(text="1. Source Image", icon='IMAGE_DATA')
        box.prop(props, "image_path", text="")

        # ── AI Depth ─────────────────────────────────────────────────
        box = layout.box()
        box.label(text="2. AI Depth Map", icon='OUTLINER_OB_FORCE_FIELD')
        row = box.row(align=True)
        row.scale_y = 1.5
        row.operator("img2mesh.generate_depth", icon='MOD_OCEAN')
        if props.depth_path and os.path.isfile(props.depth_path):
            box.label(text=f"Ready: {os.path.basename(props.depth_path)}", icon='CHECKMARK')
        else:
            # Check for auto-detected depth map
            if props.image_path:
                base, _ = os.path.splitext(props.image_path)
                auto = base + "_depth.png"
                if os.path.isfile(auto):
                    box.label(text=f"Found: {os.path.basename(auto)}", icon='CHECKMARK')
                    props.depth_path = auto
                else:
                    box.label(text="No depth map yet — generate one above", icon='INFO')
        box.prop(props, "depth_path", text="Override")

        # ── Mesh Settings ────────────────────────────────────────────
        box = layout.box()
        box.label(text="3. Mesh Settings", icon='PREFERENCES')
        box.prop(props, "grid_resolution")
        col = box.column(align=True)
        if props.grid_resolution >= 1024:
            col.label(text=f"~{(props.grid_resolution ** 2 * 4) // 1000000}M+ triangles", icon='ERROR')
        else:
            col.label(text=f"~{(props.grid_resolution ** 2 * 4) // 1000}k triangles")
        box.prop(props, "target_tris")
        if props.target_tris == 0:
            box.label(text="No tri limit (full detail)")
        box.prop(props, "depth_factor")
        box.prop(props, "back_mode")
        box.prop(props, "apply_smooth")

        # ── Generate ─────────────────────────────────────────────────
        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 2.0
        row.operator("img2mesh.generate", icon='MESH_MONKEY')

        # ── Export ───────────────────────────────────────────────────
        layout.separator()
        box = layout.box()
        box.label(text="Export", icon='EXPORT')
        box.prop(props, "export_format", text="Format")
        box.prop(props, "export_path", text="")
        box.operator("img2mesh.export", icon='FILE_TICK')

        # ── Stats ────────────────────────────────────────────────────
        obj = context.active_object
        if obj and obj.type == 'MESH':
            layout.separator()
            box = layout.box()
            box.label(text="Mesh Stats", icon='INFO')
            box.label(text=f"Vertices: {len(obj.data.vertices):,}")
            box.label(text=f"Faces: {len(obj.data.polygons):,}")
            box.label(text=f"Materials: {len(obj.data.materials)}")


# ════════════════════════════════════════════════════════════════════
#  REGISTRATION
# ════════════════════════════════════════════════════════════════════

classes = (
    IMG2MESH_Properties,
    IMG2MESH_OT_generate_depth,
    IMG2MESH_OT_generate,
    IMG2MESH_OT_export,
    IMG2MESH_PT_main_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.img2mesh_props = bpy.props.PointerProperty(type=IMG2MESH_Properties)


def unregister():
    del bpy.types.Scene.img2mesh_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
