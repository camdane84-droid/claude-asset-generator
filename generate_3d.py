#!/usr/bin/env python3
"""
Image → 3D Asset Generator (Standalone)
No Blender needed. Uses Depth Anything V2 for AI depth estimation,
then builds a high-poly displacement mesh and exports OBJ/GLB.

Usage:
    python generate_3d.py image.png [--grid 1024] [--depth 0.5] [--format obj]
    python generate_3d.py image.png --grid 2048 --depth 0.8 --format glb
"""

import sys
import os
import argparse
import time
import numpy as np
from PIL import Image, ImageFilter


def generate_depth_map(image_path):
    """Run Depth Anything V2 on the image. Returns (orig_image, depth_np_normalized)."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    print(f"\n[DEPTH] Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    print(f"[DEPTH] Size: {orig_w}x{orig_h}")

    print("[DEPTH] Loading Depth Anything V2 (first run downloads ~350MB)...")
    t0 = time.time()

    model_id = "depth-anything/Depth-Anything-V2-Base-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.eval()
    print(f"[DEPTH] Model loaded in {time.time() - t0:.1f}s")

    print("[DEPTH] Running inference...")
    t0 = time.time()

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Upscale to original resolution
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(orig_h, orig_w),
        mode="bicubic",
        align_corners=False,
    )
    depth_np = prediction.squeeze().cpu().numpy()
    print(f"[DEPTH] Inference done in {time.time() - t0:.1f}s")

    # Normalize 0..1
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max - d_min > 0:
        depth_np = (depth_np - d_min) / (d_max - d_min)
    else:
        depth_np = np.zeros_like(depth_np)

    # Save preview
    base, _ = os.path.splitext(image_path)
    preview = (depth_np * 255).astype(np.uint8)
    Image.fromarray(preview, mode='L').save(base + "_depth_preview.png")
    print(f"[DEPTH] Preview saved: {base}_depth_preview.png")

    return img, depth_np


def build_silhouette(img_rgba, depth_np):
    """
    Build a silhouette mask. Uses alpha if available, otherwise
    uses depth map + background detection.
    Returns numpy array [h, w] of floats 0..1.
    """
    w, h = img_rgba.size

    alpha = np.array(img_rgba.split()[3], dtype=np.float32) / 255.0
    has_alpha = (alpha.min() < 0.94) and (alpha.max() > 0.06)

    if has_alpha:
        print("[MASK] Using alpha channel")
        return alpha

    print("[MASK] No alpha — building mask from depth + background detection")

    gray = np.array(img_rgba.convert("L"), dtype=np.float32)

    # Sample corners for background color
    margin = 10
    corners = np.concatenate([
        gray[:margin, :margin].flatten(),
        gray[:margin, -margin:].flatten(),
        gray[-margin:, :margin].flatten(),
        gray[-margin:, -margin:].flatten(),
    ])
    bg_val = corners.mean()
    bg_std = max(corners.std(), 5.0)
    tolerance = max(bg_std * 2.5, 30)

    # Luminance-based mask
    diff = np.abs(gray - bg_val)
    lum_mask = np.clip((diff - tolerance * 0.3) / (tolerance * 0.7), 0, 1)

    # Depth-based mask: low depth = background
    d_threshold = np.percentile(depth_np, 15)  # bottom 15% is background
    d_range = depth_np.max() - d_threshold
    if d_range > 0:
        depth_mask = np.clip((depth_np - d_threshold) / d_range, 0, 1)
    else:
        depth_mask = np.ones_like(depth_np)

    # Combine: either signal can indicate foreground
    mask = np.maximum(lum_mask, depth_mask * 0.9)

    # Smooth
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=3))
    mask_img = mask_img.filter(ImageFilter.MaxFilter(5))
    mask_img = mask_img.filter(ImageFilter.MinFilter(3))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2))

    result = np.array(mask_img, dtype=np.float32) / 255.0

    coverage = (result > 0.5).sum() / result.size * 100
    print(f"[MASK] Coverage: {coverage:.0f}%")

    return result


def build_mesh(depth_np, mask, img_w, img_h, grid_res=512,
               depth_factor=0.5, back_mode='mirror'):
    """
    Build a high-poly mesh from depth map + silhouette mask.
    Returns (vertices, faces, uvs) as numpy arrays.
    """
    print(f"\n[MESH] Building grid {grid_res}x...")
    t0 = time.time()

    aspect = img_w / img_h
    if aspect >= 1.0:
        gx = grid_res
        gy = max(4, int(grid_res / aspect))
    else:
        gy = grid_res
        gx = max(4, int(grid_res * aspect))

    size_x = 2.0 if aspect >= 1.0 else 2.0 * aspect
    size_y = 2.0 / aspect if aspect >= 1.0 else 2.0

    print(f"[MESH] Grid: {gx}x{gy} = {(gx+1)*(gy+1)} verts per face")

    # ── Sample mask and depth at each grid point ─────────────────────
    # Create grid UV coordinates
    us = np.linspace(0, 1, gx + 1)
    vs = np.linspace(0, 1, gy + 1)
    uu, vv = np.meshgrid(us, vs, indexing='xy')  # [gy+1, gx+1]

    # Sample from images using bilinear interpolation
    from scipy.ndimage import map_coordinates

    # Pixel coordinates for sampling
    px = uu * (img_w - 1)
    py = vv * (img_h - 1)

    mask_sampled = map_coordinates(mask, [py, px], order=1, mode='nearest')
    depth_sampled = map_coordinates(depth_np, [py, px], order=1, mode='nearest')

    # ── Build vertex positions ───────────────────────────────────────
    # World XY from UV
    wx = (uu - 0.5) * size_x
    wy = -(vv - 0.5) * size_y

    # Front Z: depth * factor, masked
    fz = depth_sampled * depth_factor * mask_sampled

    # Back Z
    if back_mode == 'mirror':
        bz = -depth_sampled * depth_factor * mask_sampled
    elif back_mode == 'half':
        bz = -depth_sampled * depth_factor * mask_sampled * 0.3
    else:
        bz = np.full_like(fz, -depth_factor * 0.05)
        bz *= (mask_sampled > 0.05).astype(float)

    # Mask threshold — which vertices to keep
    threshold = 0.05
    active = mask_sampled > threshold  # [gy+1, gx+1] bool

    # ── Build vertex arrays ──────────────────────────────────────────
    # Vertex index maps: -1 if inactive
    front_idx = np.full((gy + 1, gx + 1), -1, dtype=np.int64)
    back_idx = np.full((gy + 1, gx + 1), -1, dtype=np.int64)

    # Collect active vertices
    active_coords = np.argwhere(active)  # [N, 2] — (row=gy, col=gx)

    n_active = len(active_coords)
    if n_active < 4:
        raise ValueError("Not enough visible area in the image.")

    print(f"[MESH] Active vertices: {n_active} / {(gx+1)*(gy+1)}")

    # Front vertices
    front_verts = np.zeros((n_active, 3), dtype=np.float64)
    front_uvs = np.zeros((n_active, 2), dtype=np.float64)
    for i, (row, col) in enumerate(active_coords):
        front_idx[row, col] = i
        front_verts[i] = [wx[row, col], wy[row, col], fz[row, col]]
        front_uvs[i] = [uu[row, col], 1.0 - vv[row, col]]

    # Back vertices (offset indices by n_active)
    back_verts = np.zeros((n_active, 3), dtype=np.float64)
    back_uvs = np.zeros((n_active, 2), dtype=np.float64)
    for i, (row, col) in enumerate(active_coords):
        back_idx[row, col] = n_active + i
        back_verts[i] = [wx[row, col], wy[row, col], bz[row, col]]
        back_uvs[i] = [uu[row, col], 1.0 - vv[row, col]]

    all_verts = np.vstack([front_verts, back_verts])
    all_uvs = np.vstack([front_uvs, back_uvs])

    print(f"[MESH] Total vertices: {len(all_verts):,}")

    # ── Build faces ──────────────────────────────────────────────────
    faces = []

    for row in range(gy):
        for col in range(gx):
            # Front face: two triangles per quad
            f00 = front_idx[row, col]
            f10 = front_idx[row, col + 1]
            f01 = front_idx[row + 1, col]
            f11 = front_idx[row + 1, col + 1]

            if f00 >= 0 and f10 >= 0 and f11 >= 0 and f01 >= 0:
                faces.append([f00, f10, f11])
                faces.append([f00, f11, f01])

            # Back face: reversed winding
            b00 = back_idx[row, col]
            b10 = back_idx[row, col + 1]
            b01 = back_idx[row + 1, col]
            b11 = back_idx[row + 1, col + 1]

            if b00 >= 0 and b10 >= 0 and b11 >= 0 and b01 >= 0:
                faces.append([b00, b11, b10])
                faces.append([b00, b01, b11])

    # ── Stitch boundary ──────────────────────────────────────────────
    # Find boundary edges and connect front to back
    for row in range(gy + 1):
        for col in range(gx):
            if front_idx[row, col] < 0 or front_idx[row, col + 1] < 0:
                continue
            above = (row > 0 and front_idx[row - 1, col] >= 0
                     and front_idx[row - 1, col + 1] >= 0)
            below = (row < gy and front_idx[row + 1, col] >= 0
                     and front_idx[row + 1, col + 1] >= 0)
            if row == 0: above = False
            if row == gy: below = False

            if not above or not below:
                fa = front_idx[row, col]
                fb = front_idx[row, col + 1]
                ba = back_idx[row, col]
                bb = back_idx[row, col + 1]
                if not above:
                    faces.append([fa, fb, bb])
                    faces.append([fa, bb, ba])
                else:
                    faces.append([fb, fa, ba])
                    faces.append([fb, ba, bb])

    for row in range(gy):
        for col in range(gx + 1):
            if front_idx[row, col] < 0 or front_idx[row + 1, col] < 0:
                continue
            left = (col > 0 and front_idx[row, col - 1] >= 0
                    and front_idx[row + 1, col - 1] >= 0)
            right = (col < gx and front_idx[row, col + 1] >= 0
                     and front_idx[row + 1, col + 1] >= 0)
            if col == 0: left = False
            if col == gx: right = False

            if not left or not right:
                fa = front_idx[row, col]
                fb = front_idx[row + 1, col]
                ba = back_idx[row, col]
                bb = back_idx[row + 1, col]
                if not left:
                    faces.append([fb, fa, ba])
                    faces.append([fb, ba, bb])
                else:
                    faces.append([fa, fb, bb])
                    faces.append([fa, bb, ba])

    faces = np.array(faces, dtype=np.int64)
    elapsed = time.time() - t0
    print(f"[MESH] Faces: {len(faces):,} triangles")
    print(f"[MESH] Built in {elapsed:.1f}s")

    return all_verts, faces, all_uvs


def export_obj(verts, faces, uvs, texture_path, output_path):
    """Export mesh as OBJ with MTL material file referencing the texture."""
    print(f"\n[EXPORT] Writing OBJ: {output_path}")
    t0 = time.time()

    base, _ = os.path.splitext(output_path)
    mtl_path = base + ".mtl"
    mtl_name = os.path.basename(base)

    # Write MTL
    with open(mtl_path, 'w') as f:
        f.write(f"# Material for {mtl_name}\n")
        f.write(f"newmtl {mtl_name}_mat\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write("d 1.0\n")
        f.write(f"map_Kd {os.path.basename(texture_path)}\n")

    # Write OBJ
    with open(output_path, 'w') as f:
        f.write(f"# Image-to-3D Asset Generator\n")
        f.write(f"# Vertices: {len(verts):,}\n")
        f.write(f"# Faces: {len(faces):,}\n")
        f.write(f"mtllib {os.path.basename(mtl_path)}\n")
        f.write(f"usemtl {mtl_name}_mat\n\n")

        # Vertices
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")

        # UVs
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        f.write("\n")

        # Faces (1-indexed, with UV indices)
        for face in faces:
            i0, i1, i2 = face + 1  # OBJ is 1-indexed
            f.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}\n")

    elapsed = time.time() - t0
    print(f"[EXPORT] Done in {elapsed:.1f}s")
    print(f"[EXPORT] OBJ: {output_path}")
    print(f"[EXPORT] MTL: {mtl_path}")
    print(f"[EXPORT] Texture: {texture_path}")

    return output_path


def export_glb(verts, faces, uvs, texture_path, output_path):
    """Export mesh as GLB using trimesh."""
    print(f"\n[EXPORT] Writing GLB: {output_path}")
    t0 = time.time()

    import trimesh
    from trimesh.visual import TextureVisuals

    # Load texture
    tex_img = Image.open(texture_path).convert("RGBA")

    # Create trimesh with UV
    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        process=False,
    )

    # Create UV visual
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=tex_img,
        metallicFactor=0.0,
        roughnessFactor=0.7,
    )

    # Build face UV indices (same as vertex indices for our mesh)
    face_uvs = faces  # our UVs are per-vertex, same indexing
    mesh.visual = TextureVisuals(uv=uvs, material=material)

    mesh.export(output_path)

    elapsed = time.time() - t0
    print(f"[EXPORT] Done in {elapsed:.1f}s")
    print(f"[EXPORT] GLB: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Image → 3D Asset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_3d.py character.png
  python generate_3d.py character.png --grid 1024 --depth 0.8
  python generate_3d.py character.png --grid 2048 --format glb
  python generate_3d.py character.png --depth-map character_depth.png
        """,
    )
    parser.add_argument("image", help="Input image path")
    parser.add_argument("-o", "--output", default=None, help="Output file path")
    parser.add_argument("--grid", type=int, default=512,
                        help="Grid resolution (512=~500k tris, 1024=~2M, 2048=~8M)")
    parser.add_argument("--depth", type=float, default=0.5,
                        help="Depth strength (0.1-3.0, default 0.5)")
    parser.add_argument("--back", choices=['mirror', 'half', 'flat'], default='mirror',
                        help="Back face mode (default: mirror)")
    parser.add_argument("--format", choices=['obj', 'glb'], default='obj',
                        help="Output format (default: obj)")
    parser.add_argument("--depth-map", default=None,
                        help="Pre-generated depth map (skip AI inference)")
    parser.add_argument("--skip-depth", action='store_true',
                        help="Use existing _depth_preview.png if available")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: File not found: {args.image}")
        sys.exit(1)

    base, _ = os.path.splitext(args.image)

    # ── Step 1: Depth map ────────────────────────────────────────────
    if args.depth_map and os.path.isfile(args.depth_map):
        print(f"[DEPTH] Using provided depth map: {args.depth_map}")
        img = Image.open(args.image).convert("RGBA")
        depth_np = np.array(Image.open(args.depth_map).convert("L"), dtype=np.float32)
        # Resize to match image if needed
        if depth_np.shape != (img.size[1], img.size[0]):
            depth_pil = Image.fromarray(depth_np.astype(np.uint8), mode='L')
            depth_pil = depth_pil.resize(img.size, Image.LANCZOS)
            depth_np = np.array(depth_pil, dtype=np.float32)
        depth_np = (depth_np - depth_np.min()) / max(depth_np.max() - depth_np.min(), 1)
    elif args.skip_depth and os.path.isfile(base + "_depth_preview.png"):
        print(f"[DEPTH] Using cached depth: {base}_depth_preview.png")
        img = Image.open(args.image).convert("RGBA")
        depth_np = np.array(
            Image.open(base + "_depth_preview.png").convert("L"),
            dtype=np.float32
        )
        if depth_np.shape != (img.size[1], img.size[0]):
            depth_pil = Image.fromarray(depth_np.astype(np.uint8), mode='L')
            depth_pil = depth_pil.resize(img.size, Image.LANCZOS)
            depth_np = np.array(depth_pil, dtype=np.float32)
        depth_np = (depth_np - depth_np.min()) / max(depth_np.max() - depth_np.min(), 1)
    else:
        img_rgb, depth_np = generate_depth_map(args.image)
        img = Image.open(args.image).convert("RGBA")

    img_w, img_h = img.size

    # ── Step 2: Silhouette mask ──────────────────────────────────────
    mask = build_silhouette(img, depth_np)

    # ── Step 3: Build mesh ───────────────────────────────────────────
    verts, faces, uvs = build_mesh(
        depth_np, mask, img_w, img_h,
        grid_res=args.grid,
        depth_factor=args.depth,
        back_mode=args.back,
    )

    # ── Step 4: Export ───────────────────────────────────────────────
    if args.output:
        output_path = args.output
    else:
        output_path = base + f"_3d.{args.format}"

    if args.format == 'obj':
        export_obj(verts, faces, uvs, args.image, output_path)
    elif args.format == 'glb':
        export_glb(verts, faces, uvs, args.image, output_path)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DONE!")
    print(f"  Vertices:  {len(verts):,}")
    print(f"  Triangles: {len(faces):,}")
    print(f"  Output:    {output_path}")
    print(f"  Import into Blender: File > Import > {'Wavefront (.obj)' if args.format == 'obj' else 'glTF (.glb)'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
