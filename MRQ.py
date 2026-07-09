#!/usr/bin/env python3
"""
generic_blender_mrq_v4.py
Version: 2026-07-10-v4-write-check

Scene-agnostic Blender command-line render driver for UE-MRQ-like jittered output.
This version deliberately avoids OPEN_EXR_MULTILAYER because some Blender builds
only expose OPEN_EXR in scene.render.image_settings.file_format.

Run:
  blender -b scene.blend -P generic_blender_mrq_v4.py -- --out C:/tmp/mrq_out --start 1 --end 1 --samples 64 --jitter pmj --save-all-subsamples
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import bpy

VERSION = "2026-07-10-v4-write-check"

# -----------------------------------------------------------------------------
# UE-like jitter helpers
# -----------------------------------------------------------------------------

def _u32(v: int) -> int:
    return v & 0xFFFFFFFF


def halton(index: int, base: int) -> float:
    result = 0.0
    inv_base = 1.0 / float(base)
    fraction = inv_base
    while index > 0:
        result += float(index % base) * fraction
        index //= base
        fraction *= inv_base
    return result


def halton23_signed(i: int, samples_per_frame: int) -> Tuple[float, float]:
    h = (i % samples_per_frame) + 1
    return halton(h, 2) - 0.5, halton(h, 3) - 0.5


def pcg_hash(v: int) -> int:
    v = _u32(v)
    v ^= _u32(v * 0x6C50B47C)
    v = _u32(v)
    v ^= _u32(v * 0xB82F1E52)
    v = _u32(v)
    v ^= _u32(v * 0xC7AFE638)
    v = _u32(v)
    v ^= _u32(v * 0x8D22F6E6)
    return _u32(v)


def permute_pmj(i: int, l: int, seed: int) -> int:
    # Matches the uploaded UE code's 64-sample PMJ permutation logic.
    w = l - 1
    i = _u32(i ^ seed)
    i = _u32(i * 0xE170893D)
    i ^= seed >> 16
    i ^= (i & w) >> 4
    i ^= seed >> 8
    i = _u32(i * 0x0929EB3F)
    i ^= seed >> 23
    i ^= (i & w) >> 1
    i = _u32(i * (1 | (seed >> 27)))
    i = _u32(i * 0x6935FA69)
    i ^= (i & w) >> 11
    i = _u32(i * 0x74DCB303)
    i ^= (i & w) >> 2
    i = _u32(i * 0x9E501CC3)
    i ^= (i & w) >> 2
    i = _u32(i * 0xC860A3DF)
    i &= w
    i ^= i >> 5
    return (i + seed) & w


def pmj01(i: int, l: int) -> Tuple[float, float]:
    # The UE code hardcodes M=8 for 64 samples. This script supports other square counts too,
    # but 64 is the intended path.
    m = int(round(math.sqrt(l)))
    if m * m != l:
        m = 8
    ix = i & (m - 1)
    iy = i >> int(round(math.log2(m)))
    ix = permute_pmj(ix, m, 0x51633E2D ^ iy)
    iy = permute_pmj(iy, m, 0x68BC21EB ^ ix)
    jx = float(pcg_hash(i) & 0xFFFF) / 65536.0
    jy = float(pcg_hash(i ^ 0x9E3779B9) & 0xFFFF) / 65536.0
    return (float(ix) + jx) / float(m), (float(iy) + jy) / float(m)


def pmj_signed(i: int, samples_per_frame: int) -> Tuple[float, float]:
    x, y = pmj01(i, samples_per_frame)
    return x - 0.5, y - 0.5


def signed_jitter(i: int, samples_per_frame: int, mode: str) -> Tuple[float, float]:
    if mode == "pmj":
        return pmj_signed(i, samples_per_frame)
    if mode == "halton23":
        return halton23_signed(i, samples_per_frame)
    if mode == "mixed":
        # Useful for matching pipelines that use PMJ as the underlying pixel jitter
        # and Halton 2,3 as the subpixel/accumulator offset.
        px, py = pmj_signed(i, samples_per_frame)
        hx, hy = halton23_signed(i, samples_per_frame)
        return 0.5 * (px + hx), 0.5 * (py + hy)
    raise ValueError(mode)


@dataclass
class SampleRecord:
    frame: int
    sample: int
    current_sub_index: int
    chosen_sub_index: int
    jitter_x_pixels: float
    jitter_y_pixels: float
    camera_shift_x_delta: float
    camera_shift_y_delta: float
    path: str
    exists: bool
    bytes: int
    is_chosen: bool


def enum_values(owner, prop_name: str) -> set[str]:
    try:
        return {item.identifier for item in owner.bl_rna.properties[prop_name].enum_items}
    except Exception:
        return set()


def parse_args(argv: List[str]) -> argparse.Namespace:
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser(description="Generic Blender MRQ-like jittered render driver v4")
    p.add_argument("--out", default="//mrq_out")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--camera", default=None)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--jitter", choices=["pmj", "halton23", "mixed"], default="pmj")
    p.add_argument("--chosen", type=int, default=None)
    p.add_argument("--chosen-mode", choices=["fixed", "center_nearest"], default="center_nearest")
    p.add_argument("--resolution-x", type=int, default=None)
    p.add_argument("--resolution-y", type=int, default=None)
    p.add_argument("--engine", choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"], default=None)
    p.add_argument("--cycles-render-samples", type=int, default=1)
    p.add_argument("--save-all-subsamples", action="store_true")
    p.add_argument("--keep-temp", action="store_true", help="Alias for --save-all-subsamples")
    p.add_argument("--view-transform", default="Standard")
    p.add_argument("--look", default="None")
    p.add_argument("--exr-codec", default="ZIP")
    p.add_argument("--half", action="store_true", default=True)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Do not render, just write manifest and print planned paths")
    return p.parse_args(argv)


def blender_abspath(path: str) -> str:
    return os.path.abspath(bpy.path.abspath(path))


def configure_scene(scene: bpy.types.Scene, args: argparse.Namespace) -> None:
    if args.engine:
        scene.render.engine = args.engine
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = max(1, args.cycles_render_samples)
        scene.cycles.use_denoising = False
        if hasattr(scene.cycles, "use_animated_seed"):
            scene.cycles.use_animated_seed = False

    if args.resolution_x:
        scene.render.resolution_x = args.resolution_x
    if args.resolution_y:
        scene.render.resolution_y = args.resolution_y
    scene.render.resolution_percentage = 100

    # Color management: keep as close to linear output as Blender allows without scene-specific assumptions.
    try:
        scene.view_settings.view_transform = args.view_transform
    except Exception:
        print(f"[MRQ v4] warning: view_transform {args.view_transform!r} unavailable; keeping {scene.view_settings.view_transform!r}")
    try:
        scene.view_settings.look = args.look
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    settings = scene.render.image_settings
    settings.file_format = "OPEN_EXR"
    if "color_depth" in settings.bl_rna.properties:
        vals = enum_values(settings, "color_depth")
        if args.half and "16" in vals:
            settings.color_depth = "16"
        elif "32" in vals:
            settings.color_depth = "32"
    if hasattr(settings, "exr_codec"):
        vals = enum_values(settings, "exr_codec")
        if not vals or args.exr_codec in vals:
            settings.exr_codec = args.exr_codec
    scene.render.use_file_extension = False  # We pass explicit .exr filenames and verify exact path.

    # Enable passes where available. With plain OPEN_EXR, not every build will include them in the file,
    # but enabling them is harmless and helps if the build writes layer/pass data.
    for vl in scene.view_layers:
        if hasattr(vl, "use_pass_z"):
            vl.use_pass_z = True
        if hasattr(vl, "use_pass_vector"):
            vl.use_pass_vector = True


def choose_camera(scene: bpy.types.Scene, name: str | None) -> bpy.types.Object:
    if name:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "CAMERA":
            raise RuntimeError(f"Camera not found or not a camera: {name}")
        scene.camera = obj
        return obj
    if scene.camera and scene.camera.type == "CAMERA":
        return scene.camera
    cams = [o for o in bpy.data.objects if o.type == "CAMERA"]
    if not cams:
        raise RuntimeError("No camera found. Use --camera CameraName.")
    scene.camera = cams[0]
    return cams[0]


def choose_subsample(sample_count: int, args: argparse.Namespace) -> int:
    if args.chosen is not None:
        if not (0 <= args.chosen < sample_count):
            raise RuntimeError(f"--chosen must be in [0,{sample_count})")
        return args.chosen
    if args.chosen_mode == "fixed":
        return 0
    best_i, best_d2 = 0, 1e30
    for i in range(sample_count):
        x, y = signed_jitter(i, sample_count, args.jitter)
        d2 = x * x + y * y
        if d2 < best_d2:
            best_i, best_d2 = i, d2
    return best_i


def possible_written_paths(requested: str, frame: int) -> List[str]:
    requested = os.path.abspath(requested)
    root, ext = os.path.splitext(requested)
    ext = ext or ".exr"
    candidates = [requested]
    candidates.append(root + ext)
    candidates.append(requested + ".exr")
    candidates.append(f"{root}{frame:04d}{ext}")
    candidates.append(f"{root}_{frame:04d}{ext}")
    candidates.append(f"{root}{frame:06d}{ext}")
    candidates.append(f"{root}_{frame:06d}{ext}")
    out, seen = [], set()
    for p in candidates:
        ap = os.path.abspath(p)
        if ap not in seen:
            out.append(ap)
            seen.add(ap)
    return out


def check_or_save_render_result(scene: bpy.types.Scene, requested: str, frame: int) -> str:
    requested = os.path.abspath(requested)
    for p in possible_written_paths(requested, frame):
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            if p != requested:
                os.makedirs(os.path.dirname(requested), exist_ok=True)
                shutil.move(p, requested)
            return requested

    img = bpy.data.images.get("Render Result")
    if img is None:
        raise RuntimeError(f"No Render Result after render. Expected {requested}")
    os.makedirs(os.path.dirname(requested), exist_ok=True)
    img.save_render(requested, scene=scene)
    if os.path.isfile(requested) and os.path.getsize(requested) > 0:
        return requested

    raise RuntimeError(
        "Render finished, but no EXR was written.\n"
        f"Requested: {requested}\n"
        "Checked:\n  " + "\n  ".join(possible_written_paths(requested, frame)) + "\n"
        "This usually means Blender failed during render, the output directory is not writable, or this scene has no active camera."
    )


def render_sample(scene: bpy.types.Scene, cam: bpy.types.Object, path: str, frame: int, jx: float, jy: float, args: argparse.Namespace) -> Tuple[bool, int]:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
    old_shift_x = cam.data.shift_x
    old_shift_y = cam.data.shift_y
    old_filepath = scene.render.filepath
    try:
        dx = jx / float(width)
        dy = -jy / float(height)
        cam.data.shift_x = old_shift_x + dx
        cam.data.shift_y = old_shift_y + dy
        scene.render.filepath = path
        print(f"[MRQ v4] render frame={frame} sample_path={path} jitter=({jx:+.8f},{jy:+.8f}) cam_shift_delta=({dx:+.10f},{dy:+.10f})", flush=True)
        if not args.dry_run:
            bpy.ops.render.render(write_still=True)
            check_or_save_render_result(scene, path, frame)
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        return exists, size
    finally:
        cam.data.shift_x = old_shift_x
        cam.data.shift_y = old_shift_y
        scene.render.filepath = old_filepath


def main() -> None:
    args = parse_args(sys.argv)
    if args.keep_temp:
        args.save_all_subsamples = True
    scene = bpy.context.scene
    configure_scene(scene, args)
    cam = choose_camera(scene, args.camera)

    start = args.start if args.start is not None else scene.frame_start
    end = args.end if args.end is not None else scene.frame_end
    if end < start:
        raise RuntimeError(f"Invalid frame range: start={start}, end={end}")
    if args.samples < 1:
        raise RuntimeError("--samples must be >= 1")

    out_dir = blender_abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[MRQ v4] VERSION={VERSION}", flush=True)
    print(f"[MRQ v4] blend={bpy.data.filepath}", flush=True)
    print(f"[MRQ v4] output_dir={out_dir}", flush=True)
    print(f"[MRQ v4] engine={scene.render.engine} format={scene.render.image_settings.file_format} camera={cam.name}", flush=True)
    print(f"[MRQ v4] resolution={scene.render.resolution_x}x{scene.render.resolution_y} frames={start}-{end} samples={args.samples}", flush=True)

    chosen = choose_subsample(args.samples, args)
    records: List[SampleRecord] = []

    for frame in range(start, end + 1):
        scene.frame_set(frame)
        frame_dir = os.path.join(out_dir, f"frame_{frame:04d}")
        os.makedirs(frame_dir, exist_ok=True)
        for sample in range(args.samples):
            jx, jy = signed_jitter(sample, args.samples, args.jitter)
            is_chosen = sample == chosen
            if args.save_all_subsamples:
                filename = f"sample_{sample:04d}_jx_{jx:+.8f}_jy_{jy:+.8f}.exr"
            elif is_chosen:
                filename = f"chosen_sample_{sample:04d}_jx_{jx:+.8f}_jy_{jy:+.8f}.exr"
            else:
                # Render non-chosen to a temp file only if future accumulation needs it.
                # In this driver-only version we skip non-chosen unless --save-all-subsamples.
                continue
            path = os.path.join(frame_dir, filename)
            exists, size = render_sample(scene, cam, path, frame, jx, jy, args)
            width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
            height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
            records.append(SampleRecord(
                frame=frame,
                sample=sample,
                current_sub_index=sample,
                chosen_sub_index=chosen,
                jitter_x_pixels=jx,
                jitter_y_pixels=jy,
                camera_shift_x_delta=jx / float(width),
                camera_shift_y_delta=-jy / float(height),
                path=path,
                exists=exists,
                bytes=size,
                is_chosen=is_chosen,
            ))

    manifest = {
        "version": VERSION,
        "blend": bpy.data.filepath,
        "out_dir": out_dir,
        "camera": cam.name,
        "engine": scene.render.engine,
        "file_format": scene.render.image_settings.file_format,
        "frames": [start, end],
        "samples": args.samples,
        "jitter": args.jitter,
        "chosen_sub_index": chosen,
        "records": [asdict(r) for r in records],
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[MRQ v4] manifest={manifest_path}", flush=True)
    produced = [r for r in records if r.exists and r.bytes > 0]
    print(f"[MRQ v4] done. produced_files={len(produced)}", flush=True)
    if len(produced) == 0 and not args.dry_run:
        raise RuntimeError("No files produced. See paths printed above and manifest.json.")


if __name__ == "__main__":
    main()
