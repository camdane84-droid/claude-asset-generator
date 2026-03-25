#!/usr/bin/env python3
"""
Depth map generator using Depth Anything V2.
Standalone script — runs outside Blender using system Python + PyTorch.

Usage:
    python generate_depth.py input_image.png [output_depth.png]
"""

import sys
import os
import argparse
import time


def generate_depth_map(image_path, output_path=None):
    """
    Generate a high-quality depth map using Depth Anything V2.
    Returns the output path.
    """
    print(f"[1/4] Loading image: {image_path}")

    from PIL import Image
    import torch
    import numpy as np

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    print(f"       Image size: {orig_w}x{orig_h}")

    # ── Load model ───────────────────────────────────────────────────
    print("[2/4] Loading Depth Anything V2 model (first run downloads ~350MB)...")
    t0 = time.time()

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    # Use the HF-compatible version of Depth Anything V2
    model_id = "depth-anything/Depth-Anything-V2-Base-hf"

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.eval()

    print(f"       Model loaded in {time.time() - t0:.1f}s")

    # ── Run inference ────────────────────────────────────────────────
    print("[3/4] Running depth estimation...")
    t0 = time.time()

    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original image size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(orig_h, orig_w),
        mode="bicubic",
        align_corners=False,
    )
    depth_np = prediction.squeeze().cpu().numpy()

    print(f"       Inference done in {time.time() - t0:.1f}s")

    # ── Post-process & save ──────────────────────────────────────────
    print("[4/4] Saving depth map...")

    # Normalize to 0..1
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max - d_min > 0:
        depth_np = (depth_np - d_min) / (d_max - d_min)
    else:
        depth_np = np.zeros_like(depth_np)

    # Save as 16-bit grayscale PNG for maximum precision
    depth_16 = (depth_np * 65535).astype(np.uint16)
    depth_pil = Image.fromarray(depth_16, mode='I;16')

    if output_path is None:
        base, _ = os.path.splitext(image_path)
        output_path = base + "_depth.png"

    depth_pil.save(output_path)

    # Also save an 8-bit preview for easy viewing
    preview_path = output_path.replace("_depth.png", "_depth_preview.png")
    depth_8 = (depth_np * 255).astype(np.uint8)
    Image.fromarray(depth_8, mode='L').save(preview_path)

    print(f"       Saved: {output_path}")
    print(f"       Preview: {preview_path}")
    print(f"       Size: {orig_w}x{orig_h}")
    print(f"       Done!")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate depth map using Depth Anything V2")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("output", nargs="?", default=None, help="Output depth map path")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: File not found: {args.image}")
        sys.exit(1)

    generate_depth_map(args.image, args.output)


if __name__ == "__main__":
    main()
