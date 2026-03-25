#!/usr/bin/env python3
"""
3D Asset Generator — True 3D reconstruction from a single image.

Uses TripoSR (by Stability AI / Tripo AI) for neural 3D reconstruction
with marching cubes mesh extraction. Includes a built-in 3D viewer.

Usage:
    python asset_generator.py image.png                          # Generate + view
    python asset_generator.py image.png --resolution 512         # Higher detail
    python asset_generator.py image.png --format glb             # GLB export
    python asset_generator.py image.png --no-view                # Skip viewer
    python asset_generator.py image.png --bake-texture           # Baked texture atlas
    python asset_generator.py view model.obj                     # View existing mesh
"""

import sys
import os
import argparse
import time

# Add TripoSR to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRIPOSR_DIR = os.path.join(SCRIPT_DIR, "TripoSR")
if TRIPOSR_DIR not in sys.path:
    sys.path.insert(0, TRIPOSR_DIR)


def generate_3d_asset(
    image_path,
    output_dir=None,
    mc_resolution=256,
    chunk_size=4096,
    foreground_ratio=0.85,
    bake_texture=False,
    texture_resolution=2048,
    export_format="obj",
    no_remove_bg=False,
    show_viewer=True,
):
    """
    Generate a 3D mesh from a single image using TripoSR.

    Returns dict with paths to generated files.
    """
    import numpy as np
    import torch
    from PIL import Image

    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(image_path))
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Device: {device}]")

    # ── Step 1: Load model ────────────────────────────────────────────
    print("[1/5] Loading TripoSR model...")
    t0 = time.time()
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(chunk_size)
    model.to(device)
    model.eval()
    print(f"       Model loaded in {time.time() - t0:.1f}s")

    # ── Step 2: Preprocess image ──────────────────────────────────────
    print("[2/5] Preprocessing image...")
    t0 = time.time()

    raw_image = Image.open(image_path)
    print(f"       Input: {raw_image.size[0]}x{raw_image.size[1]}, mode={raw_image.mode}")

    if no_remove_bg:
        image = np.array(raw_image.convert("RGB"))
        processed = Image.fromarray(image)
    else:
        import rembg
        rembg_session = rembg.new_session()
        image = remove_background(raw_image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image_np = np.array(image).astype(np.float32) / 255.0
        # Composite onto gray background
        image_np = image_np[:, :, :3] * image_np[:, :, 3:4] + (1 - image_np[:, :, 3:4]) * 0.5
        processed = Image.fromarray((image_np * 255.0).astype(np.uint8))

    # Save preprocessed input for reference
    input_preview_path = os.path.join(output_dir, f"{base_name}_input_processed.png")
    processed.save(input_preview_path)
    print(f"       Preprocessed in {time.time() - t0:.1f}s")
    print(f"       Saved preview: {input_preview_path}")

    # ── Step 3: Run TripoSR inference ─────────────────────────────────
    print("[3/5] Running TripoSR 3D reconstruction...")
    t0 = time.time()
    with torch.no_grad():
        scene_codes = model([processed], device=device)
    print(f"       Inference done in {time.time() - t0:.1f}s")

    # ── Step 4: Extract mesh ──────────────────────────────────────────
    print(f"[4/5] Extracting mesh (marching cubes resolution={mc_resolution})...")
    t0 = time.time()
    meshes = model.extract_mesh(
        scene_codes,
        has_vertex_color=not bake_texture,
        resolution=mc_resolution,
    )
    mesh = meshes[0]
    print(f"       Mesh extracted in {time.time() - t0:.1f}s")
    print(f"       Vertices: {len(mesh.vertices):,}")
    print(f"       Faces: {len(mesh.faces):,}")

    # ── Step 5: Export ────────────────────────────────────────────────
    print(f"[5/5] Exporting as {export_format.upper()}...")
    t0 = time.time()
    result_files = {}

    if bake_texture:
        import xatlas
        from tsr.bake_texture import bake_texture as do_bake_texture

        print(f"       Baking texture atlas ({texture_resolution}x{texture_resolution})...")
        bake_output = do_bake_texture(mesh, model, scene_codes[0], texture_resolution)

        mesh_path = os.path.join(output_dir, f"{base_name}_3d.{export_format}")
        texture_path = os.path.join(output_dir, f"{base_name}_3d_texture.png")

        xatlas.export(
            mesh_path,
            mesh.vertices[bake_output["vmapping"]],
            bake_output["indices"],
            bake_output["uvs"],
            mesh.vertex_normals[bake_output["vmapping"]],
        )
        Image.fromarray(
            (bake_output["colors"] * 255.0).astype(np.uint8)
        ).transpose(Image.FLIP_TOP_BOTTOM).save(texture_path)

        result_files["mesh"] = mesh_path
        result_files["texture"] = texture_path
        print(f"       Texture saved: {texture_path}")
    else:
        mesh_path = os.path.join(output_dir, f"{base_name}_3d.{export_format}")
        mesh.export(mesh_path)
        result_files["mesh"] = mesh_path

    print(f"       Export done in {time.time() - t0:.1f}s")
    print(f"       Mesh saved: {mesh_path}")

    # Also export GLB if primary format is OBJ (for easy viewing)
    if export_format == "obj" and not bake_texture:
        glb_path = os.path.join(output_dir, f"{base_name}_3d.glb")
        mesh.export(glb_path)
        result_files["glb"] = glb_path
        print(f"       Also saved: {glb_path}")

    print(f"\n       === DONE ===")
    print(f"       Vertices: {len(mesh.vertices):,}")
    print(f"       Faces:    {len(mesh.faces):,}")
    for key, path in result_files.items():
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"       {key}: {path} ({size_mb:.1f} MB)")

    # ── Show 3D viewer ────────────────────────────────────────────────
    if show_viewer:
        show_mesh(mesh_path)

    return result_files


def show_mesh(mesh_path):
    """Open a 3D viewer window to inspect a mesh."""
    import trimesh

    print(f"\n       Opening 3D viewer for: {mesh_path}")
    print("       Controls: Left-drag=rotate, Right-drag=pan, Scroll=zoom, Q=quit")

    mesh = trimesh.load(mesh_path)

    if isinstance(mesh, trimesh.Scene):
        # If it loaded as a scene, just show it
        mesh.show()
    else:
        # Create a scene with the mesh
        scene = trimesh.Scene(mesh)
        scene.show()


def main():
    parser = argparse.ArgumentParser(
        description="3D Asset Generator — True 3D reconstruction from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.png                         Generate 3D model and view it
  %(prog)s image.png --resolution 512        Higher detail (slower)
  %(prog)s image.png --format glb            Export as GLB
  %(prog)s image.png --bake-texture          Bake texture atlas
  %(prog)s image.png --output-dir ./output   Custom output directory
  %(prog)s image.png --no-view               Skip 3D viewer
  %(prog)s --view model.obj                  View an existing mesh file
""",
    )

    parser.add_argument("image", nargs="?", help="Input image path")
    parser.add_argument(
        "--view",
        type=str,
        default=None,
        metavar="MESH_FILE",
        help="View an existing mesh file (OBJ, GLB, STL, etc.)",
    )
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        default=256,
        help="Marching cubes resolution (128/256/512). Higher = more detail but slower. Default: 256",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Processing chunk size. Lower = less memory. Default: 4096",
    )
    parser.add_argument(
        "--foreground-ratio",
        type=float,
        default=0.85,
        help="Foreground size ratio (0.5-1.0). Default: 0.85",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["obj", "glb"],
        default="obj",
        help="Export format. Default: obj",
    )
    parser.add_argument(
        "--bake-texture",
        action="store_true",
        help="Bake a UV texture atlas instead of vertex colors",
    )
    parser.add_argument(
        "--texture-resolution",
        type=int,
        default=2048,
        help="Texture atlas resolution (only with --bake-texture). Default: 2048",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory. Default: same as input image",
    )
    parser.add_argument(
        "--no-remove-bg",
        action="store_true",
        help="Don't auto-remove background (input must have gray bg already)",
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Skip opening the 3D viewer after generation",
    )

    args = parser.parse_args()

    # Handle --view mode
    if args.view:
        if not os.path.isfile(args.view):
            print(f"Error: File not found: {args.view}")
            sys.exit(1)
        show_mesh(args.view)
        return

    # Handle generate (default)
    if not args.image:
        parser.print_help()
        sys.exit(1)

    if not os.path.isfile(args.image):
        print(f"Error: File not found: {args.image}")
        sys.exit(1)

    print("=" * 60)
    print("  3D Asset Generator (TripoSR)")
    print("=" * 60)
    print(f"  Input:      {args.image}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Format:     {args.format}")
    print(f"  Texture:    {'baked atlas' if args.bake_texture else 'vertex colors'}")
    print("=" * 60)

    generate_3d_asset(
        image_path=args.image,
        output_dir=args.output_dir,
        mc_resolution=args.resolution,
        chunk_size=args.chunk_size,
        foreground_ratio=args.foreground_ratio,
        bake_texture=args.bake_texture,
        texture_resolution=args.texture_resolution,
        export_format=args.format,
        no_remove_bg=args.no_remove_bg,
        show_viewer=not args.no_view,
    )


if __name__ == "__main__":
    main()
