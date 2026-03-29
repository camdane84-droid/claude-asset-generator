#!/usr/bin/env python3
"""
3D Asset Generator — Web-based GUI with 3D viewer.
Run: python app.py
Then open http://localhost:5000 in your browser.
"""

import sys
import os
import json
import time
import uuid
import threading

from flask import Flask, request, jsonify, send_from_directory, send_file

# Add TripoSR to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRIPOSR_DIR = os.path.join(SCRIPT_DIR, "TripoSR")
if TRIPOSR_DIR not in sys.path:
    sys.path.insert(0, TRIPOSR_DIR)

app = Flask(__name__, static_folder="static")

# Working directory for uploads and outputs
WORK_DIR = os.path.join(SCRIPT_DIR, "workspace")
os.makedirs(WORK_DIR, exist_ok=True)

# Global state for generation jobs
jobs = {}


@app.route("/")
def index():
    return send_file(os.path.join(SCRIPT_DIR, "static", "index.html"))


@app.route("/workspace/<path:filename>")
def serve_workspace(filename):
    return send_from_directory(WORK_DIR, filename)


@app.route("/api/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400

    # Save with a unique prefix to avoid collisions
    base_name = os.path.splitext(f.filename)[0]
    ext = os.path.splitext(f.filename)[1]
    safe_name = "".join(c for c in base_name if c.isalnum() or c in "-_")[:50]
    filename = f"{safe_name}{ext}"
    filepath = os.path.join(WORK_DIR, filename)
    f.save(filepath)

    return jsonify({
        "filename": filename,
        "path": filepath,
        "url": f"/workspace/{filename}",
    })


@app.route("/api/generate", methods=["POST"])
def generate_mesh():
    data = request.json
    filename = data.get("filename")
    quality = data.get("quality", "medium")  # low, medium, high
    base_point = data.get("base_point")  # {x: 0-1, y: 0-1} or None

    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    image_path = os.path.join(WORK_DIR, filename)
    if not os.path.isfile(image_path):
        return jsonify({"error": f"File not found: {filename}"}), 404

    # Map quality to marching cubes resolution
    quality_map = {
        "low": {"mc_resolution": 128, "chunk_size": 8192},
        "medium": {"mc_resolution": 256, "chunk_size": 4096},
        "high": {"mc_resolution": 512, "chunk_size": 2048},
    }
    settings = quality_map.get(quality, quality_map["medium"])
    settings["base_point"] = base_point

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "running",
        "progress": "Starting...",
        "started": time.time(),
        "result": None,
        "progressive": [],  # list of intermediate mesh URLs
        "progressive_version": 0,
    }

    # Run generation in background thread
    thread = threading.Thread(
        target=_run_generation,
        args=(job_id, image_path, filename, settings),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


def _fix_orientation(mesh):
    """
    Fix TripoSR mesh to standard Y-up.

    Empirically verified: raw TripoSR output has X as the tallest axis (height).
    Swap X↔Y so height goes into Y (standard Y-up for Three.js/Blender).
    The swap is an odd permutation → flip face winding to fix normals.
    """
    mesh.vertices = mesh.vertices[:, [1, 0, 2]]
    mesh.faces = mesh.faces[:, ::-1]
    return mesh


def _apply_base_point(mesh, base_point, image_path):
    """
    Reposition mesh so the user's base point maps to the origin (bottom center).

    base_point: {x: 0-1, y: 0-1} — normalized image coordinates where the user
    clicked to indicate the bottom of the object.

    The mesh (after _fix_orientation) is Y-up with the front facing -Z.
    We project the mesh bounding box onto the image plane (X=horizontal, Y=vertical)
    and shift so the base point maps to Y=0 (ground).
    """
    import numpy as np

    if not base_point:
        # Default: center horizontally, sit on ground (Y=0)
        bbox_min = mesh.vertices.min(axis=0)
        bbox_max = mesh.vertices.max(axis=0)
        center_x = (bbox_min[0] + bbox_max[0]) / 2
        center_z = (bbox_min[2] + bbox_max[2]) / 2
        mesh.vertices[:, 0] -= center_x
        mesh.vertices[:, 1] -= bbox_min[1]  # bottom at Y=0
        mesh.vertices[:, 2] -= center_z
        return mesh

    bp_x = base_point.get("x", 0.5)
    bp_y = base_point.get("y", 1.0)

    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    bbox_size = (bbox_max - bbox_min).max()

    # The image maps to the mesh's front projection:
    # Image X (0=left, 1=right) → mesh X (left to right)
    # Image Y (0=top, 1=bottom) → mesh Y (top to bottom, inverted)
    # Base point in mesh coords:
    target_x = bbox_min[0] + bp_x * (bbox_max[0] - bbox_min[0])
    target_y = bbox_max[1] - bp_y * (bbox_max[1] - bbox_min[1])  # Y inverted

    center_z = (bbox_min[2] + bbox_max[2]) / 2

    # Shift so base point is at origin (X=0, Y=0, Z=0)
    mesh.vertices[:, 0] -= target_x
    mesh.vertices[:, 1] -= target_y
    mesh.vertices[:, 2] -= center_z

    return mesh


def _rasterize_texture_gpu(moderngl, verts, faces, uvs, vert_colors,
                           proj_u, proj_v, front_weights,
                           img_array, img_w, img_h, tex_res):
    """GPU-accelerated texture baking using moderngl."""
    import numpy as np

    ctx = moderngl.create_context(standalone=True)

    # Upload input image as texture
    img_uint8 = (np.clip(img_array[:, :, :3], 0, 1) * 255).astype(np.uint8)
    img_tex = ctx.texture((img_w, img_h), 3, img_uint8.tobytes())
    img_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    img_tex.use(location=0)

    # Has alpha?
    if img_array.shape[2] == 4:
        alpha_data = (np.clip(img_array[:, :, 3], 0, 1) * 255).astype(np.uint8)
        alpha_tex = ctx.texture((img_w, img_h), 1, alpha_data.tobytes())
    else:
        alpha_data = np.full((img_h, img_w), 255, dtype=np.uint8)
        alpha_tex = ctx.texture((img_w, img_h), 1, alpha_data.tobytes())
    alpha_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    alpha_tex.use(location=1)

    prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec2 in_uv;
            in vec2 in_proj;
            in vec3 in_vcol;
            in float in_front_weight;
            out vec2 v_proj;
            out vec3 v_vcol;
            out float v_fw;
            void main() {
                v_proj = in_proj;
                v_vcol = in_vcol;
                v_fw = in_front_weight;
                gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
            }
        """,
        fragment_shader="""
            #version 330
            uniform sampler2D u_image;
            uniform sampler2D u_alpha;
            in vec2 v_proj;
            in vec3 v_vcol;
            in float v_fw;
            out vec4 o_col;
            void main() {
                vec2 uv = clamp(v_proj, 0.0, 1.0);
                vec3 front_color = texture(u_image, uv).rgb;
                float alpha = texture(u_alpha, uv).r;
                float blend = v_fw * alpha;
                vec3 color = mix(v_vcol, front_color, blend);
                o_col = vec4(color, 1.0);
            }
        """,
    )
    prog['u_image'].value = 0
    prog['u_alpha'].value = 1

    # Build per-vertex-of-face data (3 entries per triangle)
    face_verts_idx = faces.flatten()  # indices into remapped vertices
    num_indices = len(face_verts_idx)

    # UVs are per-vertex in the remapped mesh (xatlas output)
    uv_data = uvs[face_verts_idx].flatten().astype('f4')
    proj_data = np.column_stack([proj_u, proj_v])[face_verts_idx].flatten().astype('f4')
    vcol_data = vert_colors[face_verts_idx].flatten().astype('f4')
    fw_expanded = np.repeat(front_weights, 3).astype('f4')

    vbo_uv = ctx.buffer(uv_data)
    vbo_proj = ctx.buffer(proj_data)
    vbo_vcol = ctx.buffer(vcol_data)
    vbo_fw = ctx.buffer(fw_expanded)
    ibo = ctx.buffer(np.arange(num_indices, dtype='i4'))

    vao = ctx.vertex_array(prog, [
        vbo_uv.bind('in_uv', layout='2f'),
        vbo_proj.bind('in_proj', layout='2f'),
        vbo_vcol.bind('in_vcol', layout='3f'),
        vbo_fw.bind('in_front_weight', layout='1f'),
    ], ibo)

    fbo = ctx.framebuffer(
        color_attachments=[ctx.texture((tex_res, tex_res), 4, dtype='f4')]
    )
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 0.0)
    vao.render()

    data = fbo.color_attachments[0].read()
    texture = np.frombuffer(data, dtype='f4').reshape(tex_res, tex_res, 4).copy()

    ctx.release()
    return texture


def _rasterize_texture_cpu(verts, faces, uvs, vert_colors,
                           proj_u, proj_v, front_weights,
                           img_array, img_w, img_h, tex_res):
    """CPU fallback texture rasterization — per-face scanline."""
    import numpy as np

    texture = np.zeros((tex_res, tex_res, 4), dtype=np.float32)

    for fi in range(len(faces)):
        f = faces[fi]
        uv0, uv1, uv2 = uvs[f[0]], uvs[f[1]], uvs[f[2]]
        pu = np.array([[proj_u[f[j]], proj_v[f[j]]] for j in range(3)])
        vc = vert_colors[f]
        fw = front_weights[fi]

        uv_px = np.array([uv0, uv1, uv2]) * tex_res
        min_px = np.clip(np.floor(uv_px.min(axis=0)).astype(int), 0, tex_res - 1)
        max_px = np.clip(np.ceil(uv_px.max(axis=0)).astype(int), 0, tex_res - 1)

        e1 = np.array(uv1) - np.array(uv0)
        e2 = np.array(uv2) - np.array(uv0)
        denom = e1[0] * e2[1] - e1[1] * e2[0]
        if abs(denom) < 1e-10:
            continue

        for py in range(min_px[1], max_px[1] + 1):
            for px in range(min_px[0], max_px[0] + 1):
                p = np.array([(px + 0.5) / tex_res, (py + 0.5) / tex_res]) - np.array(uv0)
                bv = (p[0] * e2[1] - p[1] * e2[0]) / denom
                bw = (e1[0] * p[1] - e1[1] * p[0]) / denom
                bu = 1.0 - bv - bw
                if bu < -0.01 or bv < -0.01 or bw < -0.01:
                    continue
                bu, bv, bw = max(0,bu), max(0,bv), max(0,bw)
                s = bu+bv+bw
                if s > 0: bu/=s; bv/=s; bw/=s

                img_uv = pu[0]*bu + pu[1]*bv + pu[2]*bw
                ix = int(np.clip(img_uv[0]*img_w, 0, img_w-1))
                iy = int(np.clip(img_uv[1]*img_h, 0, img_h-1))
                front_c = img_array[iy, ix, :3]
                front_a = img_array[iy, ix, 3] if img_array.shape[2]==4 else 1.0
                back_c = vc[0]*bu + vc[1]*bv + vc[2]*bw
                blend = fw * front_a
                color = front_c * blend + back_c * (1.0 - blend)
                ty = tex_res - 1 - py
                if 0 <= ty < tex_res:
                    texture[ty, px, :3] = color
                    texture[ty, px, 3] = 1.0

    return texture


def _bake_projected_texture(mesh, image_path, texture_resolution=2048):
    """
    Project the original input image onto the mesh as a UV texture.

    1. UV unwrap the mesh with xatlas
    2. For each texel in the UV atlas, find which face it belongs to
    3. Project that 3D position onto the front camera view
    4. Sample the input image at that projected position
    5. For back-facing geometry, use TripoSR vertex colors as fallback

    Returns a trimesh with proper UV texture applied.
    """
    import numpy as np
    import xatlas
    import trimesh as _trimesh
    from PIL import Image

    print("  [Texture] Loading input image...")
    input_img = Image.open(image_path).convert("RGBA")
    img_w, img_h = input_img.size
    img_array = np.array(input_img).astype(np.float32) / 255.0

    # Get vertex colors as fallback (for back faces)
    has_vertex_colors = hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None
    if has_vertex_colors:
        vc = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
    else:
        vc = np.full((len(mesh.vertices), 3), 0.5, dtype=np.float32)

    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()

    # Step 1: UV unwrap with xatlas
    print("  [Texture] UV unwrapping with xatlas...")
    atlas = xatlas.Atlas()
    atlas.add_mesh(verts, faces)
    options = xatlas.PackOptions()
    options.resolution = texture_resolution
    options.padding = max(2, texture_resolution // 256)
    options.bilinear = True
    atlas.generate(pack_options=options)
    vmapping, new_faces, uvs = atlas[0]

    # Remap vertices
    new_verts = verts[vmapping]
    new_vc = vc[vmapping]

    # Step 2: Compute face normals to determine front vs back facing
    # Raw TripoSR: X=height, Y=width, Z=depth. Camera looks down -Z.
    # Winding is pre-flip (flipped by _fix_orientation later),
    # so normals point inward → front-facing = POSITIVE Z component.
    print("  [Texture] Computing face visibility...")

    v0 = new_verts[new_faces[:, 0]]
    v1 = new_verts[new_faces[:, 1]]
    v2 = new_verts[new_faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v1)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    face_normals = face_normals / norms
    front_weight_per_face = np.clip(face_normals[:, 2], 0, 1)

    # Step 3: Project vertices onto image plane
    # Raw: X=height(up), Y=width(right), Z=depth(back)
    # Image mapping: Y → U (horizontal), X → V (vertical, flipped)
    print("  [Texture] Projecting image onto mesh...")
    bbox_min = new_verts.min(axis=0)
    bbox_max = new_verts.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = (bbox_max - bbox_min).max()

    # Y=width → horizontal (U), X=height → vertical (V)
    proj_u = (new_verts[:, 1] - bbox_center[1]) / bbox_size + 0.5
    proj_v = (new_verts[:, 0] - bbox_center[0]) / bbox_size + 0.5
    # Flip V because image Y goes top-down but mesh X goes bottom-up
    proj_v = 1.0 - proj_v

    # Step 4: Rasterize texture atlas using moderngl (GPU-accelerated)
    print(f"  [Texture] Baking {texture_resolution}x{texture_resolution} texture...")

    try:
        import moderngl
        texture = _rasterize_texture_gpu(
            moderngl, new_verts, new_faces, uvs, new_vc,
            proj_u, proj_v, front_weight_per_face,
            img_array, img_w, img_h, texture_resolution
        )
    except Exception as gpu_err:
        print(f"  [Texture] GPU rasterize failed ({gpu_err}), using CPU fallback...")
        texture = _rasterize_texture_cpu(
            new_verts, new_faces, uvs, new_vc,
            proj_u, proj_v, front_weight_per_face,
            img_array, img_w, img_h, texture_resolution
        )

    # Fill any remaining black pixels with nearby colors (padding)
    print("  [Texture] Padding texture seams...")
    alpha = texture[:, :, 3]
    from scipy.ndimage import maximum_filter, uniform_filter
    for _ in range(3):
        mask = alpha == 0
        if not mask.any():
            break
        for c in range(3):
            chan = texture[:, :, c]
            filled = uniform_filter(chan * alpha, size=3) / np.maximum(
                uniform_filter(alpha, size=3), 1e-6
            )
            chan[mask] = filled[mask]
            texture[:, :, c] = chan
        alpha_filled = maximum_filter(alpha, size=3)
        alpha = np.where(mask, alpha_filled, alpha)
        texture[:, :, 3] = alpha

    # Convert to PIL image
    texture_uint8 = (np.clip(texture[:, :, :3], 0, 1) * 255).astype(np.uint8)
    texture_img = Image.fromarray(texture_uint8)

    # Build the textured trimesh
    print("  [Texture] Building textured mesh...")
    textured_mesh = _trimesh.Trimesh(
        vertices=new_verts,
        faces=new_faces,
        process=False,
    )

    # Apply UV texture
    material = _trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture_img,
        metallicFactor=0.0,
        roughnessFactor=0.7,
    )
    visuals = _trimesh.visual.TextureVisuals(uv=uvs, material=material)
    textured_mesh.visual = visuals

    return textured_mesh, texture_img


def _run_generation(job_id, image_path, filename, settings):
    try:
        import numpy as np
        import torch
        from PIL import Image

        from tsr.system import TSR
        from tsr.utils import remove_background, resize_foreground

        base_name = os.path.splitext(filename)[0]
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Step 1: Load model
        jobs[job_id]["progress"] = "Loading TripoSR model..."
        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(settings["chunk_size"])
        model.to(device)
        model.eval()

        # Step 2: Preprocess
        jobs[job_id]["progress"] = "Removing background & preprocessing..."
        raw_image = Image.open(image_path)

        import rembg
        rembg_session = rembg.new_session()
        image = remove_background(raw_image, rembg_session)
        image = resize_foreground(image, 0.85)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = image_np[:, :, :3] * image_np[:, :, 3:4] + (1 - image_np[:, :, 3:4]) * 0.5
        processed = Image.fromarray((image_np * 255.0).astype(np.uint8))

        preview_path = os.path.join(WORK_DIR, f"{base_name}_processed.png")
        processed.save(preview_path)

        # Step 3: Inference
        jobs[job_id]["progress"] = "Running 3D reconstruction (neural net)..."
        with torch.no_grad():
            scene_codes = model([processed], device=device)

        # Step 4: Progressive mesh extraction
        # Build at increasing resolutions so the user sees it refine
        final_res = settings["mc_resolution"]
        progressive_resolutions = []
        if final_res >= 128:
            progressive_resolutions.append(64)
        if final_res >= 256:
            progressive_resolutions.append(128)
        progressive_resolutions.append(final_res)

        final_mesh = None
        for i, mc_res in enumerate(progressive_resolutions):
            stage = f"Extracting mesh ({mc_res}) — stage {i+1}/{len(progressive_resolutions)}"
            jobs[job_id]["progress"] = stage

            meshes = model.extract_mesh(
                scene_codes,
                has_vertex_color=True,
                resolution=mc_res,
            )
            mesh = meshes[0]
            final_mesh = mesh

            # Export this progressive stage
            prog_filename = f"{base_name}_prog_{mc_res}.glb"
            prog_path = os.path.join(WORK_DIR, prog_filename)
            # Clone mesh before orientation fix (so we don't double-rotate)
            import trimesh as _trimesh
            prog_mesh = _trimesh.Trimesh(
                vertices=mesh.vertices.copy(),
                faces=mesh.faces.copy(),
                vertex_colors=mesh.visual.vertex_colors.copy() if hasattr(mesh.visual, 'vertex_colors') else None,
            )
            _fix_orientation(prog_mesh)
            _apply_base_point(prog_mesh, settings.get("base_point"), image_path)
            prog_mesh.export(prog_path)

            jobs[job_id]["progressive"].append({
                "glb_url": f"/workspace/{prog_filename}",
                "resolution": mc_res,
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
            })
            jobs[job_id]["progressive_version"] += 1

        # Step 4b: Apply base point repositioning if specified
        base_point = settings.get("base_point")

        # Step 5: Bake neural texture and export final mesh
        jobs[job_id]["progress"] = "Baking neural texture (2048x2048)..."
        import trimesh as _trimesh
        from PIL import Image as PILImage

        glb_filename = f"{base_name}_3d.glb"
        obj_filename = f"{base_name}_3d.obj"
        glb_path = os.path.join(WORK_DIR, glb_filename)
        obj_path = os.path.join(WORK_DIR, obj_filename)

        try:
            from tsr.bake_texture import bake_texture

            texture_result = bake_texture(
                final_mesh, model, scene_codes[0],
                texture_resolution=2048,
            )

            vmapping = texture_result["vmapping"]
            indices = texture_result["indices"]
            uvs = texture_result["uvs"]
            colors_rgba = texture_result["colors"]

            # Convert float RGBA to uint8 RGB image
            # Flip vertically: OpenGL framebuffer origin is bottom-left
            colors_rgb = (np.clip(colors_rgba[:, :, :3], 0, 1) * 255).astype(np.uint8)
            texture_img = PILImage.fromarray(np.flipud(colors_rgb))

            # Save texture for debugging/export
            tex_filename = f"{base_name}_3d_texture.png"
            tex_path = os.path.join(WORK_DIR, tex_filename)
            texture_img.save(tex_path)

            # Build mesh with xatlas-remapped vertices/faces
            new_vertices = final_mesh.vertices[vmapping]
            textured_mesh = _trimesh.Trimesh(
                vertices=new_vertices,
                faces=indices,
                process=False,
            )

            # Apply PBR material with baked texture
            material = _trimesh.visual.material.PBRMaterial(
                baseColorTexture=texture_img,
                metallicFactor=0.0,
                roughnessFactor=0.7,
            )
            textured_mesh.visual = _trimesh.visual.TextureVisuals(
                uv=uvs, material=material
            )

            _fix_orientation(textured_mesh)
            _apply_base_point(textured_mesh, base_point, image_path)

            textured_mesh.export(glb_path)
            textured_mesh.export(obj_path)
            final_verts = len(textured_mesh.vertices)
            final_faces = len(textured_mesh.faces)
            print(f"  [Texture] Neural texture bake complete: {final_verts} verts, {final_faces} faces")

        except Exception as tex_err:
            print(f"  [Texture] Neural bake failed ({tex_err}), falling back to vertex colors")
            import traceback
            traceback.print_exc()

            # Fallback: export with vertex colors only
            fallback = _trimesh.Trimesh(
                vertices=final_mesh.vertices.copy(),
                faces=final_mesh.faces.copy(),
                vertex_colors=final_mesh.visual.vertex_colors.copy() if hasattr(final_mesh.visual, 'vertex_colors') else None,
            )
            _fix_orientation(fallback)
            _apply_base_point(fallback, base_point, image_path)
            fallback.export(glb_path)
            fallback.export(obj_path)
            final_verts = len(fallback.vertices)
            final_faces = len(fallback.faces)

        # Free GPU memory
        del model, scene_codes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - jobs[job_id]["started"]

        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = "Complete!"
        jobs[job_id]["result"] = {
            "glb_url": f"/workspace/{glb_filename}",
            "obj_url": f"/workspace/{obj_filename}",
            "glb_path": glb_path,
            "obj_path": obj_path,
            "vertices": final_verts,
            "faces": final_faces,
            "elapsed": round(elapsed, 1),
            "preview_url": f"/workspace/{base_name}_processed.png",
        }

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["progress"] = f"Error: {str(e)}"
        import traceback
        traceback.print_exc()


@app.route("/api/job/<job_id>")
def get_job_status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(jobs[job_id])


@app.route("/api/export", methods=["POST"])
def export_asset():
    """Export the current mesh — converts format and serves as browser download."""
    data = request.json
    source_glb = data.get("glb_path")
    export_format = data.get("format", "glb")

    if not source_glb or not os.path.isfile(source_glb):
        return jsonify({"error": "Source mesh not found"}), 404

    import trimesh

    base_name = os.path.splitext(os.path.basename(source_glb))[0]
    out_name = f"{base_name}.{export_format}"
    out_path = os.path.join(WORK_DIR, out_name)

    # If requesting same format as source and file exists, just serve it
    if source_glb == out_path or (export_format == "glb" and os.path.isfile(source_glb)):
        out_path = source_glb
    else:
        mesh = trimesh.load(source_glb)
        if isinstance(mesh, trimesh.Scene):
            combined = trimesh.util.concatenate(
                [g for g in mesh.geometry.values()]
            )
            combined.export(out_path)
        else:
            mesh.export(out_path)

    return send_file(
        out_path,
        as_attachment=True,
        download_name=out_name,
    )


# ── Mesh undo history ────────────────────────────────────
mesh_history = []  # stack of GLB paths for undo


@app.route("/api/edit-mesh", methods=["POST"])
def edit_mesh():
    """Apply a modification command to the current mesh."""
    import trimesh
    import numpy as np

    data = request.json
    glb_path = data.get("glb_path")
    command = data.get("command", "").strip().lower()

    if not glb_path or not os.path.isfile(glb_path):
        return jsonify({"error": "Mesh file not found"}), 404

    if not command:
        return jsonify({"error": "No command provided"}), 400

    # Parse the command
    parts = command.split()
    action = parts[0]

    try:
        # Load current mesh
        loaded = trimesh.load(glb_path)
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [g for g in loaded.geometry.values()]
            )
        else:
            mesh = loaded

        # Save to undo stack before modifying
        mesh_history.append(glb_path)

        message = ""

        # ── SMOOTH ───────────────────────────────────
        if action == "smooth":
            iterations = int(parts[1]) if len(parts) > 1 else 1
            iterations = max(1, min(iterations, 20))
            # Laplacian smoothing
            from scipy.sparse import lil_matrix
            for _ in range(iterations):
                adj = lil_matrix((len(mesh.vertices), len(mesh.vertices)))
                for face in mesh.faces:
                    for i in range(3):
                        for j in range(3):
                            if i != j:
                                adj[face[i], face[j]] = 1
                adj = adj.tocsr()
                row_sums = np.array(adj.sum(axis=1)).flatten()
                row_sums[row_sums == 0] = 1
                new_verts = mesh.vertices.copy()
                for vi in range(len(mesh.vertices)):
                    neighbors = adj[vi].nonzero()[1]
                    if len(neighbors) > 0:
                        avg = mesh.vertices[neighbors].mean(axis=0)
                        new_verts[vi] = mesh.vertices[vi] * 0.5 + avg * 0.5
                mesh.vertices = new_verts
            message = f"Smoothed ({iterations} iteration{'s' if iterations > 1 else ''})"

        # ── SUBDIVIDE ────────────────────────────────
        elif action == "subdivide":
            # Simple midpoint subdivision
            new_verts = list(mesh.vertices)
            new_faces = []
            edge_midpoints = {}

            def get_midpoint(a, b):
                key = (min(a, b), max(a, b))
                if key not in edge_midpoints:
                    mid = (mesh.vertices[a] + mesh.vertices[b]) / 2
                    edge_midpoints[key] = len(new_verts)
                    new_verts.append(mid)
                return edge_midpoints[key]

            for face in mesh.faces:
                a, b, c = face
                ab = get_midpoint(a, b)
                bc = get_midpoint(b, c)
                ca = get_midpoint(c, a)
                new_faces.extend([
                    [a, ab, ca],
                    [ab, b, bc],
                    [ca, bc, c],
                    [ab, bc, ca],
                ])

            mesh = trimesh.Trimesh(
                vertices=np.array(new_verts),
                faces=np.array(new_faces),
                process=False,
            )
            message = f"Subdivided → {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces"

        # ── SCALE ────────────────────────────────────
        elif action == "scale":
            if len(parts) == 3:
                # axis-specific: scale y 2.0
                axis = parts[1]
                factor = float(parts[2])
                axis_map = {"x": 0, "y": 1, "z": 2}
                if axis in axis_map:
                    mesh.vertices[:, axis_map[axis]] *= factor
                    message = f"Scaled {axis.upper()} by {factor}"
                else:
                    return jsonify({"error": f"Unknown axis: {axis}. Use x, y, or z"}), 400
            elif len(parts) == 2:
                # uniform: scale 1.5
                factor = float(parts[1])
                mesh.vertices *= factor
                message = f"Scaled uniformly by {factor}"
            else:
                return jsonify({"error": "Usage: scale 1.5  or  scale y 2.0"}), 400

        # ── ROTATE ───────────────────────────────────
        elif action == "rotate":
            if len(parts) < 3:
                return jsonify({"error": "Usage: rotate x 45"}), 400
            axis = parts[1]
            angle_deg = float(parts[2])
            axis_vec = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}.get(axis)
            if not axis_vec:
                return jsonify({"error": f"Unknown axis: {axis}. Use x, y, or z"}), 400
            rot = trimesh.transformations.rotation_matrix(
                np.radians(angle_deg), axis_vec
            )
            mesh.apply_transform(rot)
            message = f"Rotated {angle_deg}° around {axis.upper()}"

        # ── MIRROR ───────────────────────────────────
        elif action == "mirror":
            if len(parts) < 2:
                return jsonify({"error": "Usage: mirror x"}), 400
            axis = parts[1]
            axis_map = {"x": 0, "y": 1, "z": 2}
            if axis not in axis_map:
                return jsonify({"error": f"Unknown axis: {axis}. Use x, y, or z"}), 400
            ai = axis_map[axis]
            mirrored_verts = mesh.vertices.copy()
            mirrored_verts[:, ai] *= -1
            mirrored_faces = mesh.faces[:, ::-1]  # flip winding
            combined_verts = np.vstack([mesh.vertices, mirrored_verts])
            combined_faces = np.vstack([
                mesh.faces,
                mirrored_faces + len(mesh.vertices)
            ])
            mesh = trimesh.Trimesh(
                vertices=combined_verts,
                faces=combined_faces,
                process=True,
            )
            message = f"Mirrored across {axis.upper()} → {len(mesh.vertices):,} verts"

        # ── REMESH ───────────────────────────────────
        elif action == "remesh":
            # Uniform remesh — simplify then subdivide
            target_faces = int(parts[1]) if len(parts) > 1 else len(mesh.faces)
            mesh = mesh.simplify_quadric_decimation(target_faces)
            message = f"Remeshed → {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces"

        # ── DECIMATE ─────────────────────────────────
        elif action in ("decimate", "simplify"):
            if len(parts) < 2:
                return jsonify({"error": "Usage: decimate 5000 (target face count)"}), 400
            target = int(parts[1])
            mesh = mesh.simplify_quadric_decimation(target)
            message = f"Decimated → {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces"

        # ── UNDO ─────────────────────────────────────
        elif action == "undo":
            if not mesh_history:
                return jsonify({"error": "Nothing to undo"}), 400
            prev_path = mesh_history.pop()
            # Just point back to the previous file
            base_name = os.path.splitext(os.path.basename(prev_path))[0]
            prev_mesh = trimesh.load(prev_path)
            if isinstance(prev_mesh, trimesh.Scene):
                prev_mesh = trimesh.util.concatenate(
                    [g for g in prev_mesh.geometry.values()]
                )
            glb_filename = os.path.basename(prev_path)
            obj_filename = base_name + ".obj"
            return jsonify({
                "message": "Undone — reverted to previous mesh",
                "glb_url": f"/workspace/{glb_filename}",
                "glb_path": prev_path,
                "obj_url": f"/workspace/{obj_filename}",
                "obj_path": os.path.join(WORK_DIR, obj_filename),
                "vertices": len(prev_mesh.vertices),
                "faces": len(prev_mesh.faces),
            })

        else:
            return jsonify({
                "error": f"Unknown command: {action}. Try: smooth, subdivide, scale, rotate, mirror, remesh, decimate, undo"
            }), 400

        # Save the modified mesh
        base_name = os.path.splitext(os.path.basename(glb_path))[0]
        # Strip old _edit suffixes to keep name clean
        if "_edit" in base_name:
            base_name = base_name.split("_edit")[0]
        edit_id = str(uuid.uuid4())[:6]
        new_glb_name = f"{base_name}_edit_{edit_id}.glb"
        new_obj_name = f"{base_name}_edit_{edit_id}.obj"
        new_glb_path = os.path.join(WORK_DIR, new_glb_name)
        new_obj_path = os.path.join(WORK_DIR, new_obj_name)

        mesh.export(new_glb_path)
        mesh.export(new_obj_path)

        return jsonify({
            "message": message,
            "glb_url": f"/workspace/{new_glb_name}",
            "glb_path": new_glb_path,
            "obj_url": f"/workspace/{new_obj_name}",
            "obj_path": new_obj_path,
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("  3D Asset Generator")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 50)
    app.run(host="127.0.0.1", port=5000, debug=False)
