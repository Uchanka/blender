#!/usr/bin/env python3
"""
generic_blender_mrq_v19_frame_major_accum.py
Version: 2026-07-12-v19-frame-major-accum-ndc

Scene-agnostic Blender command-line render driver that mimics UE5 MRQ spatial
subsample output for super-resolution / frame-interpolation datasets.

Per frame it produces:
  1. 64 (--samples) spatial subsamples, each jittered by a signed subpixel
     offset from Halton(2,3) (in pixels), applied via camera shift.
  2. accumulated.exr          - the average of all subsamples (no MV / depth).
  3. chosen_sample_XXXX_*.exr - the one chosen subsample's beauty.
  4. chosen_depth_ndc.exr     - depth of the SAME chosen subsample, in NDC.
  5. chosen_mv_ndc.exr        - geometric motion vector of the SAME chosen
                                subsample, converted from Blender pixel-space
                                to NDC/UV space.
  6. chosen_mvdepth_ndc.exr   - packed R=mv.x, G=mv.y, B=ndc_depth, A=1
                                (UE-style sidecar).

Design changes vs the older v18 script:
  * frame-major loop (all samples of frame N, then frame N+1).
  * Compositor File Output nodes are attached ONLY for the chosen subsample
    (1 compositor render per frame instead of 64). This sidesteps the Blender 5
    compositor/File Output state bug that previously forced sample-major order
    and the --split-samples multi-process workaround; both are removed.
  * Real accumulation: every subsample is rendered to a temp EXR, loaded with
    numpy, summed, and averaged into accumulated.exr.
  * NDC conversion of depth/vector is done in numpy post-processing with
    explicit, calibratable conventions (see --depth-ndc-mode / --mv-*).
  * Default jitter is halton23 to match the user's UE pipeline.

Run:
  blender -b scene.blend -P generic_blender_mrq_v19_frame_major_accum.py -- \
      --out C:/tmp/mrq_out --start 1 --end 24 --samples 64 --jitter halton23

CALIBRATION WARNING:
  Blender's Vector pass component order and sign, and its depth definition,
  vary by engine/version and do NOT automatically match UE. Render a simple
  scene with known motion once and compare against your UE MRQ output, then
  fix --mv-components / --mv-sign-x / --mv-sign-y / --depth-ndc-mode.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import shutil
import struct
import sys
import zlib
from dataclasses import asdict, dataclass, field
from typing import List, Tuple

import bpy

try:
    import numpy as np
except Exception as exc:  # Blender bundles numpy; this should not happen.
    raise RuntimeError("numpy is required (bundled with Blender)") from exc

VERSION = "2026-07-13-v19.3-point-filter"
TEMP_NODE_PREFIX = "__MRQ_TEMP_PASS__"

# -----------------------------------------------------------------------------
# Jitter sequences (signed, in pixels, centered on 0)
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
    if mode == "halton23":
        return halton23_signed(i, samples_per_frame)
    if mode == "pmj":
        return pmj_signed(i, samples_per_frame)
    if mode == "mixed":
        px, py = pmj_signed(i, samples_per_frame)
        hx, hy = halton23_signed(i, samples_per_frame)
        return 0.5 * (px + hx), 0.5 * (py + hy)
    raise ValueError(mode)


FLT_MAX = 3.402823466e38
 

SUBPIXEL_OFFSETS: Tuple[Tuple[float, float], ...] = (
    (-0.5, -0.5),
    (-0.5,  0.5),
    ( 0.5, -0.5),
    ( 0.5,  0.5),
)


def to_1b2_offset(jitter: Tuple[float, float], j: int) -> Tuple[float, float]:
    """对应 To1b2Offset。"""
    ox, oy = SUBPIXEL_OFFSETS[j]
    return ((jitter[0] - ox) * 0.5, (jitter[1] - oy) * 0.5)


def _dist_squared(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy
 
 
def get_optimal_seq_seg_indices(
    ref_jitter: Tuple[float, float], length: int = 64
) -> Tuple[int, int]:
    min_dist = FLT_MAX
    best_indices = (0, 0)
    for seq_idx in range(length):
        for seg_idx in range(4):
            halt_jitter = signed_jitter(seq_idx, length, "halton23")
            trial_jitter = to_1b2_offset(halt_jitter, seg_idx)
            dist = _dist_squared(trial_jitter, ref_jitter)
            if dist < min_dist:
                min_dist = dist
                best_indices = (seq_idx, seg_idx)
    return best_indices


def get_optimal_seg_index(
    ref_jitter: Tuple[float, float], seq: int, length: int = 64
) -> int:
    """对应 GetOptimalSegIndex，固定 seq，只在 4 个 segment 里找最近的。"""
    min_dist = FLT_MAX
    best_index = 0
    for seg_idx in range(4):
        halt_jitter = signed_jitter(seq, length, "halton23")
        trial_jitter = to_1b2_offset(halt_jitter, seg_idx)
        dist = _dist_squared(trial_jitter, ref_jitter)
        if dist < min_dist:
            min_dist = dist
            best_index = seg_idx
    return best_index


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser(description="Blender MRQ-like frame-major jittered render driver v19")
    p.add_argument("--out", default="//mrq_out")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--frame-offset", type=int, default=0, help="Add an offset to the frame number sequence offset")
    p.add_argument("--camera", default=None)
    p.add_argument("--samples", type=int, default=64, help="Spatial subsamples per frame (jitter sequence length)")
    p.add_argument("--jitter", choices=["halton23", "pmj", "mixed"], default="halton23")
    #p.add_argument("--chosen", type=int, default=None, help="Force this subsample index as the chosen one")
    p.add_argument("--chosen-mode", choices=["fixed", "center_nearest"], default="center_nearest",
                   help="fixed: index 0; center_nearest: the jitter closest to the pixel center")
    p.add_argument("--resolution-x", type=int, default=None)
    p.add_argument("--resolution-y", type=int, default=None)
    p.add_argument("--engine", choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_WORKBENCH"], default=None)
    p.add_argument("--cycles-render-samples", type=int, default=1)
    p.add_argument("--save-all-subsamples", action="store_true",
                   help="Keep every subsample EXR on disk (named sample_XXXX_...). Default: temp subsamples are deleted after accumulation.")
    p.add_argument("--keep-raw-passes", action="store_true",
                   help="Keep the raw (pre-NDC) chosen depth/vector EXRs written by the compositor")
    p.add_argument("--no-accumulate", action="store_true", help="Skip accumulated.exr")
    p.add_argument("--no-passes", action="store_true", help="Do not write depth/vector for the chosen subsample")
    p.add_argument("--view-transform", default="Standard")
    p.add_argument("--look", default="None")
    p.add_argument("--exr-codec", default="ZIP")
    p.add_argument("--half", action="store_true", default=True)
    p.add_argument("--pass-color-depth", choices=["16", "32"], default="32",
                   help="OpenEXR bit depth for raw depth/vector pass files (default 32; depth precision matters)")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    # --- Stability / performance knobs -----------------------------------------
    p.add_argument("--keep-texture-cache", action="store_true",
                   help="Do NOT disable the Blender 5.2 Cycles Texture Cache. By default the script "
                        "disables it because its tiled-EXR streaming has known race-condition crashes "
                        "(access violation in TiledInputFile::readTile) that our 64-renders-per-frame "
                        "workload triggers readily.")
    p.add_argument("--no-persistent-data", action="store_true",
                   help="Do NOT enable Render > Persistent Data. By default it is enabled so Cycles keeps "
                        "scene/texture data alive across the 64 subsample renders per frame (much faster, "
                        "and avoids reloading textures 64x).")
    p.add_argument("--compositor-device", choices=["CPU", "GPU", "keep"], default="CPU",
                   help="Execution device for the compositor. Default CPU (avoids the 'Render size too "
                        "large for GPU' fallback path). 'keep' leaves the scene setting untouched.")
    # --- Sampling semantics -----------------------------------------------------
    p.add_argument("--pixel-filter", choices=["point", "keep"], default="point",
                   help="point (default): shrink the render pixel filter to ~a point (Cycles BOX filter, "
                        "width 0.01; render.filter_width 0.01 for EEVEE) so every internal render sample "
                        "lands at the jittered subpixel position. This matches UE MRQ spatial samples "
                        "with AA off: geometry stays point-sampled (aliased) per subsample while spp only "
                        "converges shading noise, and the 64-sample accumulation becomes the box-filtered "
                        "AA ground truth. keep: leave the scene's filter untouched (note: Cycles default "
                        "is a 1.5px Blackman-Harris, which pre-blurs every subsample).")
    p.add_argument("--eevee-taa-samples", type=int, default=None,
                   help="Set EEVEE taa_render_samples for each subsample render. With --pixel-filter point "
                        "these internal samples all hit the same subpixel position, so higher values "
                        "converge EEVEE's stochastic shadows/GI without re-anti-aliasing the image.")
    # --- NDC conversion knobs -------------------------------------------------
    p.add_argument("--depth-ndc-mode", choices=["ue_reversed_z", "d3d01", "gl", "raw"], default="ue_reversed_z",
                   help="ue_reversed_z: DeviceZ=near/z (UE reversed-Z, infinite far); d3d01: [0,1] forward Z; gl: [-1,1]; raw: keep linear depth")
    p.add_argument("--ray-depth", choices=["auto", "on", "off"], default="auto",
                   help="Correct ray-length depth to planar Z before NDC conversion. Cycles outputs ray length; Eevee outputs planar Z. auto = on for CYCLES only.")
    p.add_argument("--mv-components", choices=["rg", "ba"], default="rg",
                   help="Which pair of the 4-component Vector pass to use as the motion vector. CALIBRATE against UE.")
    p.add_argument("--mv-sign-x", type=float, default=1.0)
    p.add_argument("--mv-sign-y", type=float, default=-1.0,
                   help="Default -1: Blender image Y is up, UE screen Y is down. CALIBRATE.")
    p.add_argument("--mv-scale", choices=["ndc", "uv"], default="ndc",
                   help="ndc: pixels * 2/resolution (NDC spans [-1,1]); uv: pixels / resolution")
    p.add_argument("--flip-y", action="store_true",
                   help="Vertically flip the NDC depth/mv/mvdepth arrays before saving (match a top-left-origin consumer)")
    return p.parse_args(argv)


# -----------------------------------------------------------------------------
# Scene / camera setup
# -----------------------------------------------------------------------------

def enum_values(owner, prop_name: str) -> set:
    try:
        return {item.identifier for item in owner.bl_rna.properties[prop_name].enum_items}
    except Exception:
        return set()


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

        # Blender 5.2's Cycles Texture Cache streams tiled EXR/tx textures via
        # OIIO. It has known race-condition crashes (tiles freed while other
        # threads read them -> access violation in TiledInputFile::readTile),
        # and rendering 64 stills back-to-back per frame in one process is the
        # worst case for it. Disable it unless explicitly kept; textures then
        # load fully into memory (pre-5.2 behavior).
        if not args.keep_texture_cache:
            disabled = []
            for attr in ("use_texture_cache", "texture_cache", "use_image_cache"):
                if hasattr(scene.cycles, attr):
                    try:
                        setattr(scene.cycles, attr, False)
                        disabled.append(attr)
                    except Exception:
                        pass
            if disabled:
                print(f"[MRQ v19] Cycles texture cache disabled via {disabled} "
                      "(known crash source in 5.2; use --keep-texture-cache to override)", flush=True)
            elif args.debug:
                tex_props = [p for p in dir(scene.cycles) if "texture" in p.lower() or "cache" in p.lower()]
                print(f"[MRQ v19] no texture-cache property found to disable; cycles props: {tex_props}", flush=True)

    # Keep Cycles scene/texture data alive across the many still renders per
    # frame: massively faster, and it avoids the texture load/free churn that
    # aggravates the texture-cache races.
    if not args.no_persistent_data and hasattr(scene.render, "use_persistent_data"):
        try:
            scene.render.use_persistent_data = True
            print("[MRQ v19] use_persistent_data=True (disable with --no-persistent-data)", flush=True)
        except Exception:
            pass

    # Pin the compositor execution device (avoids 'Render size too large for
    # GPU, use CPU compositor instead' fallback churn on big frames).
    if args.compositor_device != "keep" and hasattr(scene.render, "compositor_device"):
        try:
            scene.render.compositor_device = args.compositor_device
            print(f"[MRQ v19] compositor_device={args.compositor_device}", flush=True)
        except Exception:
            pass

    # Point pixel filter: make each subsample a true point sample at the jitter
    # position (UE-MRQ semantics), instead of a pre-anti-aliased image.
    if args.pixel_filter == "point":
        applied = []
        if scene.render.engine == "CYCLES" and hasattr(scene, "cycles"):
            if hasattr(scene.cycles, "pixel_filter_type"):
                try:
                    scene.cycles.pixel_filter_type = "BOX"
                    applied.append("cycles.pixel_filter_type=BOX")
                except Exception:
                    pass
            if hasattr(scene.cycles, "filter_width"):
                try:
                    scene.cycles.filter_width = 0.01
                    applied.append("cycles.filter_width=0.01")
                except Exception:
                    pass
        if hasattr(scene.render, "filter_width"):
            try:
                scene.render.filter_width = 0.01
                applied.append("render.filter_width=0.01")
            except Exception:
                pass
        if applied:
            print(f"[MRQ v19] point pixel filter: {', '.join(applied)} "
                  "(use --pixel-filter keep to preserve the scene's filter)", flush=True)

    if args.eevee_taa_samples is not None and hasattr(scene, "eevee"):
        if hasattr(scene.eevee, "taa_render_samples"):
            try:
                scene.eevee.taa_render_samples = max(1, args.eevee_taa_samples)
                print(f"[MRQ v19] eevee.taa_render_samples={scene.eevee.taa_render_samples}", flush=True)
            except Exception:
                pass

    if args.resolution_x:
        scene.render.resolution_x = args.resolution_x
    if args.resolution_y:
        scene.render.resolution_y = args.resolution_y
    scene.render.resolution_percentage = 100

    try:
        scene.view_settings.view_transform = args.view_transform
    except Exception:
        print(f"[MRQ v19] warning: view_transform {args.view_transform!r} unavailable; keeping {scene.view_settings.view_transform!r}")
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
    scene.render.use_file_extension = False

    for vl in scene.view_layers:
        if hasattr(vl, "use_pass_z"):
            vl.use_pass_z = True
        if hasattr(vl, "use_pass_vector"):
            vl.use_pass_vector = True
        for attr in ("use_pass_motion", "use_pass_motion_vector"):
            if hasattr(vl, attr):
                try:
                    setattr(vl, attr, True)
                except Exception:
                    pass


def choose_camera(scene: bpy.types.Scene, name):
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


# -----------------------------------------------------------------------------
# Compositor helpers (Blender 4.x scene.node_tree and 5.x compositing_node_group)
# -----------------------------------------------------------------------------

def get_compositor_tree(scene: bpy.types.Scene, create: bool = True):
    if hasattr(scene, "node_tree"):
        if create and hasattr(scene, "use_nodes"):
            try:
                scene.use_nodes = True
            except Exception:
                pass
        return getattr(scene, "node_tree", None)
    if hasattr(scene, "compositing_node_group"):
        tree = getattr(scene, "compositing_node_group", None)
        if tree is None and create:
            tree = bpy.data.node_groups.new(name=f"{scene.name}_MRQ_Compositor", type="CompositorNodeTree")
            scene.compositing_node_group = tree
        return tree
    return None


def _enable_compositor(scene: bpy.types.Scene) -> None:
    if hasattr(scene, "use_nodes"):
        try:
            scene.use_nodes = True
        except Exception:
            pass
    if hasattr(scene.render, "use_compositing"):
        try:
            scene.render.use_compositing = True
        except Exception:
            pass


def _remove_temp_compositor_nodes(scene: bpy.types.Scene) -> None:
    tree = get_compositor_tree(scene, create=False)
    if tree is None:
        return
    for node in list(tree.nodes):
        if node.name.startswith(TEMP_NODE_PREFIX) or node.label.startswith(TEMP_NODE_PREFIX):
            tree.nodes.remove(node)


def _begin_isolated_compositor(scene: bpy.types.Scene, args: argparse.Namespace):
    """On Blender 5, swap in a fresh temporary compositor node group so we never
    mutate (or depend on) the user's group. No-op on Blender 4.x."""
    if not hasattr(scene, "compositing_node_group"):
        return None, None
    old_group = getattr(scene, "compositing_node_group", None)
    temp_group = None
    try:
        temp_group = bpy.data.node_groups.new(
            name=f"MRQ_TMP_Compositor_{os.getpid()}_{id(scene)}",
            type="CompositorNodeTree",
        )
        scene.compositing_node_group = temp_group
        if args.debug:
            print(f"[MRQ v19] isolated compositor group created: {temp_group.name}", flush=True)
        return old_group, temp_group
    except Exception as exc:
        print(f"[MRQ v19] warning: could not create isolated compositor group: {exc}", flush=True)
        try:
            scene.compositing_node_group = old_group
        except Exception:
            pass
        if temp_group is not None:
            try:
                bpy.data.node_groups.remove(temp_group)
            except Exception:
                pass
        return None, None


def _end_isolated_compositor(scene: bpy.types.Scene, old_group, temp_group) -> None:
    if temp_group is None or not hasattr(scene, "compositing_node_group"):
        return
    try:
        scene.compositing_node_group = old_group
    except Exception:
        pass
    try:
        bpy.data.node_groups.remove(temp_group)
    except Exception:
        pass


def _find_output_socket(node, names: List[str]):
    wanted = {n.lower() for n in names}
    for sock in node.outputs:
        if sock.name.lower() in wanted:
            return sock
    for sock in node.outputs:
        low = sock.name.lower()
        if any(n.lower() in low for n in names):
            return sock
    return None


def _socket_by_names(sockets, names):
    wanted = {n.lower() for n in names}
    for sock in sockets:
        if sock.name.lower() in wanted:
            return sock
    for sock in sockets:
        low = sock.name.lower()
        if any(w in low for w in wanted):
            return sock
    return sockets[0] if len(sockets) else None


def _apply_image_format_settings(fmt, args: argparse.Namespace, color_mode: str) -> None:
    if fmt is None:
        return
    if hasattr(fmt, "file_format"):
        try:
            fmt.file_format = "OPEN_EXR"
        except Exception:
            pass
    if hasattr(fmt, "color_depth"):
        if args.pass_color_depth in enum_values(fmt, "color_depth"):
            try:
                fmt.color_depth = args.pass_color_depth
            except Exception:
                pass
    if hasattr(fmt, "exr_codec"):
        vals = enum_values(fmt, "exr_codec")
        if not vals or args.exr_codec in vals:
            try:
                fmt.exr_codec = args.exr_codec
            except Exception:
                pass
    if hasattr(fmt, "color_mode"):
        if color_mode in enum_values(fmt, "color_mode"):
            try:
                fmt.color_mode = color_mode
            except Exception:
                pass


def _set_file_output_format(node, args: argparse.Namespace, color_mode: str) -> None:
    _apply_image_format_settings(getattr(node, "format", None), args, color_mode)
    items = getattr(node, "file_output_items", None)
    if items is not None:
        for item in items:
            _apply_image_format_settings(getattr(item, "format", None), args, color_mode)
    if args.debug:
        fmts = [getattr(getattr(node, "format", None), "file_format", None)]
        if items is not None:
            fmts += [getattr(getattr(it, "format", None), "file_format", None) for it in items]
        print(f"[MRQ v19] File Output {node.name!r} effective formats: {fmts}", flush=True)


def _set_file_output_directory(node, frame_dir: str) -> None:
    if hasattr(node, "base_path"):
        node.base_path = frame_dir
    elif hasattr(node, "directory"):
        node.directory = frame_dir
    else:
        print("[MRQ v19] warning: File Output node has neither base_path nor directory", flush=True)


def _set_file_output_prefix(node, prefix_without_extension: str, color_mode: str) -> int:
    """Configure one File Output node across Blender 4/5; return input index to link."""
    if hasattr(node, "file_slots") and len(node.file_slots) > 0:
        node.file_slots[0].path = prefix_without_extension
        return 0

    if hasattr(node, "file_name"):
        try:
            node.file_name = prefix_without_extension
        except Exception as exc:
            print(f"[MRQ v19] warning: could not set file_name={prefix_without_extension!r}: {exc}", flush=True)

    items = getattr(node, "file_output_items", None)
    if items is not None:
        try:
            while len(items) > 0:
                items.remove(items[0])
        except Exception as exc:
            print(f"[MRQ v19] warning: could not clear default file_output_items: {exc}", flush=True)
        item_name = "Z" if color_mode == "BW" else "Image"
        socket_candidates = (
            ("VALUE", "NodeSocketFloat", "RGBA", "NodeSocketColor") if color_mode == "BW"
            else ("RGBA", "NodeSocketColor", "VALUE", "NodeSocketFloat")
        )
        last_exc = None
        for socket_type in socket_candidates:
            try:
                items.new(socket_type, item_name)
                break
            except Exception as exc:
                last_exc = exc
        if len(node.inputs) == 0:
            raise RuntimeError(f"Could not create File Output item for {prefix_without_extension!r}: {last_exc}")
        try:
            if len(items) > 0:
                items[0].name = item_name
                if hasattr(items[0], "file_name"):
                    items[0].file_name = item_name
        except Exception:
            pass
        try:
            node.active_item_index = 0
        except Exception:
            pass
        return 0
    return 0


def _delete_stale_pass_files(frame_dir: str, stem: str, suffix: str) -> None:
    exact = os.path.join(frame_dir, f"{stem}{suffix}.exr")
    for f in glob.glob(os.path.join(frame_dir, f"{stem}{suffix}*.exr")) + [exact]:
        try:
            if os.path.isfile(f):
                os.remove(f)
        except Exception:
            pass


def setup_compositor_pass_outputs(scene: bpy.types.Scene, frame_dir: str, stem: str, args: argparse.Namespace) -> dict:
    """Create temporary File Output nodes for raw Depth/Z and Vector.
    Only ever called for the chosen subsample of each frame."""
    outputs = {}
    _enable_compositor(scene)
    tree = get_compositor_tree(scene, create=True)
    if tree is None:
        print("[MRQ v19] warning: no compositor node tree; depth/vector will not be written", flush=True)
        return outputs
    _remove_temp_compositor_nodes(scene)

    rlayers = tree.nodes.new(type="CompositorNodeRLayers")
    rlayers.name = TEMP_NODE_PREFIX + "RenderLayers"
    rlayers.label = TEMP_NODE_PREFIX + "RenderLayers"
    try:
        rlayers.layer = scene.view_layers[0].name
    except Exception:
        pass

    # Blender 5: compositor node groups need an explicit NodeGroupOutput with an
    # interface socket, otherwise the group may never execute in background
    # renders and File Output nodes silently write nothing.
    try:
        img_sock = _find_output_socket(rlayers, ["Image", "Combined", "Color"])
        if img_sock is not None and not hasattr(scene, "node_tree"):
            group_out = tree.nodes.new(type="NodeGroupOutput")
            group_out.name = TEMP_NODE_PREFIX + "GroupOutputKeepAlive"
            group_out.label = TEMP_NODE_PREFIX + "GroupOutputKeepAlive"
            if hasattr(tree, "interface") and hasattr(tree.interface, "new_socket"):
                try:
                    tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
                except Exception:
                    pass
            out_sock = _socket_by_names(group_out.inputs, ["Image", "Output", "Color"])
            if out_sock is not None:
                tree.links.new(img_sock, out_sock)
    except Exception as exc:
        print(f"[MRQ v19] warning: could not create group-output keepalive: {exc}", flush=True)

    if args.debug:
        try:
            print("[MRQ v19] Render Layers outputs: " + ", ".join(s.name for s in rlayers.outputs), flush=True)
        except Exception:
            pass

    def add_output(socket_names: List[str], suffix: str, color_mode: str) -> None:
        socket = _find_output_socket(rlayers, socket_names)
        exact = os.path.join(frame_dir, f"{stem}{suffix}.exr")
        outputs[suffix.lstrip("_")] = {"exact": exact, "socket": "" if socket is None else socket.name}
        if socket is None:
            print(f"[MRQ v19] warning: no compositor socket for {socket_names}; pass {suffix} skipped", flush=True)
            return
        _delete_stale_pass_files(frame_dir, stem, suffix)
        out = tree.nodes.new(type="CompositorNodeOutputFile")
        out.name = TEMP_NODE_PREFIX + suffix
        out.label = TEMP_NODE_PREFIX + suffix
        _set_file_output_directory(out, frame_dir)
        input_index = _set_file_output_prefix(out, f"{stem}{suffix}_", color_mode)
        _set_file_output_format(out, args, color_mode)
        tree.links.new(socket, out.inputs[input_index])
        if args.debug:
            print(f"[MRQ v19] compositor pass {suffix}: socket={socket.name!r} -> {exact}", flush=True)

    add_output(["Depth", "Z"], "_depth", "BW")
    add_output(["Vector", "Motion Vector", "MotionVector"], "_vector", "RGBA")
    return outputs


def collect_compositor_pass_output(frame_dir: str, stem: str, suffix: str) -> Tuple[str, bool, int]:
    """File Output appends frame numbers; glob and rename to the exact path."""
    exact = os.path.join(frame_dir, f"{stem}{suffix}.exr")
    patterns = [
        os.path.join(frame_dir, f"{stem}{suffix}.exr"),
        os.path.join(frame_dir, f"{stem}{suffix}_*.exr"),
        os.path.join(frame_dir, f"{stem}{suffix}*.exr"),
        os.path.join(frame_dir, "**", f"{stem}{suffix}*.exr"),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat, recursive=True))
    candidates = [c for c in candidates if os.path.isfile(c) and os.path.getsize(c) > 0]
    candidates.sort(key=lambda c: os.path.getmtime(c), reverse=True)
    if candidates:
        src = candidates[0]
        if os.path.abspath(src) != os.path.abspath(exact):
            try:
                if os.path.exists(exact):
                    os.remove(exact)
                shutil.move(src, exact)
            except Exception:
                shutil.copy2(src, exact)
    ok = os.path.isfile(exact)
    return exact, ok, os.path.getsize(exact) if ok else 0


# -----------------------------------------------------------------------------
# Minimal pure-Python EXR reader (numpy + zlib)
#
# Why this exists: Blender 5's redesigned File Output node can write multilayer
# EXRs regardless of the format we request, and bpy's Python image API cannot
# read pixels from multilayer EXRs (it reports size 0x0 and empty pixels).
# This reader parses single-part scanline EXRs directly: compression
# NONE/ZIP/ZIPS, pixel types HALF/FLOAT/UINT, any channel names, so it handles
# both plain and Blender-multilayer files.
# -----------------------------------------------------------------------------

_EXR_PT_SIZES = {0: 4, 1: 2, 2: 4}          # UINT, HALF, FLOAT
_EXR_PT_DTYPES = {0: np.uint32, 1: np.float16, 2: np.float32}


def _exr_read_null_str(f) -> str:
    bs = bytearray()
    while True:
        c = f.read(1)
        if not c:
            raise RuntimeError("Unexpected EOF while reading EXR header string")
        if c == b"\x00":
            return bs.decode("utf-8", "replace")
        bs += c
        if len(bs) > 4096:
            raise RuntimeError("EXR header string too long (corrupt file?)")


def read_exr_channels(path: str) -> Tuple[dict, int, int]:
    """Return ({channel_name: (H, W) float32 array}, width, height).
    Row order is top-down (EXR native)."""
    with open(path, "rb") as f:
        magic, version = struct.unpack("<ii", f.read(8))
        if magic != 20000630:
            raise RuntimeError(f"Not an EXR file: {path}")
        if version & 0x200:
            raise RuntimeError(f"Tiled EXR not supported by built-in reader: {path}")
        if version & 0x800 or version & 0x1000:
            raise RuntimeError(f"Deep/multi-part EXR not supported by built-in reader: {path}")

        channels: List[Tuple[str, int]] = []
        compression = None
        data_window = None
        while True:
            name = _exr_read_null_str(f)
            if name == "":
                break
            attr_type = _exr_read_null_str(f)
            (size,) = struct.unpack("<i", f.read(4))
            data = f.read(size)
            if name == "channels" and attr_type == "chlist":
                cf = io.BytesIO(data)
                while True:
                    cname = _exr_read_null_str(cf)
                    if cname == "":
                        break
                    (ptype,) = struct.unpack("<i", cf.read(4))
                    cf.read(4)  # pLinear + 3 reserved bytes
                    xs, ys = struct.unpack("<ii", cf.read(8))
                    if xs != 1 or ys != 1:
                        raise RuntimeError(f"Subsampled EXR channels not supported: {path}")
                    if ptype not in _EXR_PT_SIZES:
                        raise RuntimeError(f"Unknown EXR pixel type {ptype} in {path}")
                    channels.append((cname, ptype))
            elif name == "compression":
                compression = data[0]
            elif name == "dataWindow":
                data_window = struct.unpack("<iiii", data)

        if data_window is None or compression is None or not channels:
            raise RuntimeError(f"EXR missing required header attributes: {path}")
        xmin, ymin, xmax, ymax = data_window
        width = xmax - xmin + 1
        height = ymax - ymin + 1
        if width <= 0 or height <= 0:
            raise RuntimeError(f"EXR has empty data window {data_window}: {path}")

        if compression == 0:          # NONE
            lines_per_block = 1
        elif compression == 2:        # ZIPS
            lines_per_block = 1
        elif compression == 3:        # ZIP
            lines_per_block = 16
        else:
            raise RuntimeError(
                f"EXR compression id {compression} not supported by built-in reader "
                f"({path}); re-render with --exr-codec ZIP")

        n_blocks = (height + lines_per_block - 1) // lines_per_block
        offsets = struct.unpack(f"<{n_blocks}Q", f.read(8 * n_blocks))
        bytes_per_line = sum(_EXR_PT_SIZES[pt] for _, pt in channels) * width
        out = {cname: np.zeros((height, width), dtype=np.float32) for cname, _ in channels}

        for off in offsets:
            f.seek(off)
            y, dsize = struct.unpack("<ii", f.read(8))
            raw = f.read(dsize)
            block_lines = min(lines_per_block, ymax - y + 1)
            expected = bytes_per_line * block_lines
            if compression in (2, 3) and dsize != expected:
                buf = np.frombuffer(zlib.decompress(raw), dtype=np.uint8)
                # Undo delta predictor: t[i] = t[i-1] + d[i] - 128 (mod 256)
                t = ((np.cumsum(buf.astype(np.int64)) - 128 * np.arange(buf.size, dtype=np.int64)) & 0xFF).astype(np.uint8)
                # Undo byte interleave: even bytes from first half, odd from second.
                half = (buf.size + 1) // 2
                un = np.empty(buf.size, dtype=np.uint8)
                un[0::2] = t[:half]
                un[1::2] = t[half:]
                dec = un.tobytes()
            else:
                dec = raw
            if len(dec) != expected:
                raise RuntimeError(f"EXR block size mismatch in {path}: got {len(dec)}, expected {expected}")
            pos = 0
            for line in range(block_lines):
                row = y - ymin + line
                for cname, pt in channels:
                    nbytes = _EXR_PT_SIZES[pt] * width
                    vals = np.frombuffer(dec, dtype=_EXR_PT_DTYPES[pt], count=width, offset=pos)
                    pos += nbytes
                    if 0 <= row < height:
                        out[cname][row] = vals.astype(np.float32)
        return out, width, height


def _exr_pick_depth(channels: dict) -> np.ndarray:
    """Pick the depth channel from a raw depth EXR (single-layer or multilayer)."""
    names = list(channels.keys())
    for pred in (
        lambda n: n == "Z" or n.endswith(".Z"),
        lambda n: "depth" in n.lower(),
        lambda n: n in ("Y", "V") or n.endswith(".Y") or n.endswith(".V"),
        lambda n: n == "R" or n.endswith(".R"),
    ):
        for n in names:
            if pred(n):
                return channels[n]
    if not names:
        raise RuntimeError("Depth EXR contains no channels")
    return channels[names[0]]


def _exr_pick_rgba(channels: dict) -> np.ndarray:
    """Assemble an (H, W, 4) array from R/G/B/A or X/Y/Z/W channel groups.
    Falls back to replicating a single channel into RGB."""
    groups: dict = {}
    for n, arr in channels.items():
        if "." in n:
            prefix, comp = n.rsplit(".", 1)
        else:
            prefix, comp = "", n
        groups.setdefault(prefix, {})[comp.upper()] = arr
    for comps in groups.values():
        for order in (("R", "G", "B", "A"), ("X", "Y", "Z", "W")):
            if order[0] in comps and order[1] in comps:
                h, w = comps[order[0]].shape
                res = np.zeros((h, w, 4), dtype=np.float32)
                res[..., 3] = 1.0
                for i, c in enumerate(order):
                    if c in comps:
                        res[..., i] = comps[c]
                return res
    # Single-channel fallback.
    first = next(iter(channels.values()))
    h, w = first.shape
    res = np.zeros((h, w, 4), dtype=np.float32)
    res[..., 0] = res[..., 1] = res[..., 2] = first
    res[..., 3] = 1.0
    return res


# -----------------------------------------------------------------------------
# EXR IO via bpy image API (works in background mode; no extra deps)
# -----------------------------------------------------------------------------

def load_exr_rgba(path: str, non_color: bool = False) -> np.ndarray:
    """Load an EXR into an (H, W, 4) float32 array (bottom-up row order, as bpy
    stores pixels). Falls back to the built-in pure-Python reader when bpy
    cannot provide pixels (e.g. multilayer EXRs report size 0x0)."""
    img = bpy.data.images.load(path, check_existing=False)
    try:
        if non_color:
            try:
                img.colorspace_settings.name = "Non-Color"
            except Exception:
                pass
        w, h = img.size
        try:
            npix = len(img.pixels)
        except Exception:
            npix = 0
        ch = img.channels if img.channels else 4
        if w > 0 and h > 0 and npix == w * h * ch:
            buf = np.empty(npix, dtype=np.float32)
            img.pixels.foreach_get(buf)
            arr = buf.reshape(h, w, ch)
            if ch == 4:
                return arr
            out = np.zeros((h, w, 4), dtype=np.float32)
            out[..., :ch] = arr
            out[..., 3] = 1.0
            return out
        print(f"[MRQ v19] bpy could not read {os.path.basename(path)} "
              f"(size={w}x{h}, pixels={npix}, type={getattr(img, 'type', '?')}); using built-in EXR reader", flush=True)
    finally:
        try:
            bpy.data.images.remove(img)
        except Exception:
            pass
    channels, _, _ = read_exr_channels(path)
    return _exr_pick_rgba(channels)[::-1].copy()  # top-down -> bottom-up


def save_exr_rgba(path: str, arr: np.ndarray) -> None:
    """Save an (H, W, 4) float array as a linear float EXR (bottom-up row order expected)."""
    h, w = arr.shape[:2]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    name = f"__MRQ_SAVE__{os.getpid()}"
    img = bpy.data.images.new(name, width=w, height=h, alpha=True, float_buffer=True)
    try:
        try:
            img.colorspace_settings.name = "Non-Color"
        except Exception:
            pass
        img.pixels.foreach_set(np.ascontiguousarray(arr, dtype=np.float32).ravel())
        img.filepath_raw = path
        img.file_format = "OPEN_EXR"
        img.save()
    finally:
        try:
            bpy.data.images.remove(img)
        except Exception:
            pass
    if not (os.path.isfile(path) and os.path.getsize(path) > 0):
        raise RuntimeError(f"Failed to write EXR: {path}")


# -----------------------------------------------------------------------------
# NDC post-processing
# -----------------------------------------------------------------------------

def _planar_correction_field(cam, width: int, height: int, pixel_aspect: float) -> np.ndarray:
    """Per-pixel factor converting Cycles ray-length depth to planar Z: z = t * factor.
    Approximation: ignores subpixel jitter shift and lens distortion; assumes perspective camera."""
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Planar depth correction got an empty image ({width}x{height}); "
                           "the depth pass EXR failed to load")
    data = cam.data
    if getattr(data, "type", "PERSP") != "PERSP":
        return np.ones((height, width), dtype=np.float32)
    try:
        tan_x = math.tan(data.angle_x * 0.5)
    except Exception:
        tan_x = math.tan(getattr(data, "angle", math.radians(50.0)) * 0.5)
    # Render-aspect-consistent vertical tangent.
    tan_y = tan_x * (float(height) / float(width)) / max(pixel_aspect, 1e-8)
    xs = ((np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 - 1.0)
    ys = ((np.arange(height, dtype=np.float64) + 0.5) / height * 2.0 - 1.0)
    # Include the camera's base shift (jitter deltas are sub-pixel; negligible here).
    xs = xs + 2.0 * float(getattr(data, "shift_x", 0.0))
    ys = ys + 2.0 * float(getattr(data, "shift_y", 0.0))
    gx, gy = np.meshgrid(xs * tan_x, ys * tan_y)
    factor = 1.0 / np.sqrt(1.0 + gx * gx + gy * gy)
    return factor.astype(np.float32)


def depth_to_ndc(depth: np.ndarray, cam, mode: str) -> np.ndarray:
    near = max(float(cam.data.clip_start), 1e-8)
    far = max(float(cam.data.clip_end), near + 1e-6)
    z = np.maximum(depth.astype(np.float64), 1e-12)
    if mode == "raw":
        ndc = z
    elif mode == "ue_reversed_z":
        # UE reversed-Z with infinite far plane: DeviceZ = Near / SceneDepth.
        ndc = near / z
    elif mode == "d3d01":
        ndc = (far / (far - near)) * (1.0 - near / z)
    elif mode == "gl":
        ndc = ((far + near) - 2.0 * far * near / z) / (far - near)
    else:
        raise ValueError(mode)
    return ndc.astype(np.float32)


def postprocess_chosen_passes(frame_dir: str, raw_depth_path: str, raw_vector_path: str,
                              cam, scene: bpy.types.Scene, args: argparse.Namespace) -> dict:
    """Convert the chosen sample's raw Blender passes to NDC and write:
    chosen_depth_ndc.exr, chosen_mv_ndc.exr, chosen_mvdepth_ndc.exr."""
    result = {}
    width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
    pixel_aspect = float(scene.render.pixel_aspect_x) / max(float(scene.render.pixel_aspect_y), 1e-8)

    depth_ndc = None
    if raw_depth_path and os.path.isfile(raw_depth_path):
        depth_channels, dw, dh = read_exr_channels(raw_depth_path)
        if args.debug:
            print(f"[MRQ v19] depth EXR channels: {sorted(depth_channels.keys())} ({dw}x{dh})", flush=True)
        depth = _exr_pick_depth(depth_channels)[::-1].copy()  # top-down -> bottom-up
        if depth.shape != (height, width):
            print(f"[MRQ v19] warning: depth EXR is {depth.shape[1]}x{depth.shape[0]}, "
                  f"render is {width}x{height}", flush=True)
        # Cycles Depth pass = ray length from camera; Eevee = planar Z.
        do_planar = (args.ray_depth == "on") or (args.ray_depth == "auto" and scene.render.engine == "CYCLES")
        if do_planar:
            depth = depth * _planar_correction_field(cam, depth.shape[1], depth.shape[0], pixel_aspect)
        depth_ndc = depth_to_ndc(depth, cam, args.depth_ndc_mode)
        out = np.zeros((*depth_ndc.shape, 4), dtype=np.float32)
        out[..., 0] = out[..., 1] = out[..., 2] = depth_ndc
        out[..., 3] = 1.0
        if args.flip_y:
            out = out[::-1]
        p = os.path.join(frame_dir, "chosen_depth_ndc.exr")
        save_exr_rgba(p, out)
        result["depth_ndc"] = p
    else:
        print(f"[MRQ v19] warning: raw depth missing, cannot write chosen_depth_ndc.exr ({raw_depth_path})", flush=True)

    mv_ndc = None
    if raw_vector_path and os.path.isfile(raw_vector_path):
        vec_channels, vw, vh = read_exr_channels(raw_vector_path)
        if args.debug:
            print(f"[MRQ v19] vector EXR channels: {sorted(vec_channels.keys())} ({vw}x{vh})", flush=True)
        vec = _exr_pick_rgba(vec_channels)[::-1].copy()  # top-down -> bottom-up
        if args.mv_components == "rg":
            mvx_px, mvy_px = vec[..., 0], vec[..., 1]
        else:
            mvx_px, mvy_px = vec[..., 2], vec[..., 3]
        scale = 2.0 if args.mv_scale == "ndc" else 1.0
        mv_ndc = np.zeros((*mvx_px.shape, 2), dtype=np.float32)
        mv_ndc[..., 0] = mvx_px * (scale / width) * args.mv_sign_x
        mv_ndc[..., 1] = mvy_px * (scale / height) * args.mv_sign_y
        out = np.zeros((*mvx_px.shape, 4), dtype=np.float32)
        out[..., 0] = mv_ndc[..., 0]
        out[..., 1] = mv_ndc[..., 1]
        out[..., 3] = 1.0
        if args.flip_y:
            out = out[::-1]
        p = os.path.join(frame_dir, "chosen_mv_ndc.exr")
        save_exr_rgba(p, out)
        result["mv_ndc"] = p
    else:
        print(f"[MRQ v19] warning: raw vector missing, cannot write chosen_mv_ndc.exr ({raw_vector_path})", flush=True)

    if depth_ndc is not None and mv_ndc is not None:
        out = np.zeros((*depth_ndc.shape, 4), dtype=np.float32)
        out[..., 0] = mv_ndc[..., 0]
        out[..., 1] = mv_ndc[..., 1]
        out[..., 2] = depth_ndc
        out[..., 3] = 1.0
        if args.flip_y:
            out = out[::-1]
        p = os.path.join(frame_dir, "chosen_mvdepth_ndc.exr")
        save_exr_rgba(p, out)
        result["mvdepth_ndc"] = p
    return result


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def possible_written_paths(requested: str, frame: int) -> List[str]:
    requested = os.path.abspath(requested)
    root, ext = os.path.splitext(requested)
    ext = ext or ".exr"
    candidates = [requested, root + ext, requested + ".exr",
                  f"{root}{frame:04d}{ext}", f"{root}_{frame:04d}{ext}",
                  f"{root}{frame:06d}{ext}", f"{root}_{frame:06d}{ext}"]
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
    raise RuntimeError(f"Render finished but no EXR was written: {requested}")


def render_subsample(scene: bpy.types.Scene, cam, path: str, frame: int,
                     jx: float, jy: float, with_passes: bool, args: argparse.Namespace) -> dict:
    """Render one jittered subsample. Only when with_passes is True are the
    temporary compositor Depth/Vector File Output nodes attached."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
    old_shift_x = cam.data.shift_x
    old_shift_y = cam.data.shift_y
    old_filepath = scene.render.filepath
    old_use_comp = getattr(scene.render, "use_compositing", None)
    old_group = temp_group = None
    raw_passes = {}
    try:
        dx = jx / float(width)
        dy = -jy / float(height)
        cam.data.shift_x = old_shift_x + dx
        cam.data.shift_y = old_shift_y + dy
        scene.render.filepath = path
        stem = os.path.splitext(os.path.basename(path))[0]

        if with_passes:
            old_group, temp_group = _begin_isolated_compositor(scene, args)
            setup_compositor_pass_outputs(scene, os.path.dirname(path), stem, args)
        else:
            # Plain beauty render: no compositor at all (deterministic, and avoids
            # the Blender 5 File Output state bug entirely for the 63 other samples).
            if hasattr(scene.render, "use_compositing"):
                scene.render.use_compositing = False

        if args.debug or with_passes:
            print(f"[MRQ v19] render frame={frame} passes={with_passes} path={os.path.basename(path)} "
                  f"jitter=({jx:+.8f},{jy:+.8f})px shift_delta=({dx:+.10f},{dy:+.10f})", flush=True)

        if not args.dry_run:
            bpy.ops.render.render(write_still=True)
            check_or_save_render_result(scene, path, frame)
            if with_passes:
                for suffix in ("_depth", "_vector"):
                    pth, ok, nbytes = collect_compositor_pass_output(os.path.dirname(path), stem, suffix)
                    raw_passes[suffix.lstrip("_")] = {"path": pth, "exists": ok, "bytes": nbytes}
                    if not ok and not args.no_passes:
                        nearby = sorted(glob.glob(os.path.join(os.path.dirname(path), "*.exr")))[:20]
                        print(f"[MRQ v19] warning: pass {suffix} not written for {path}; nearby={nearby}", flush=True)
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        return {"path": path, "exists": exists, "bytes": size, "raw_passes": raw_passes}
    finally:
        cam.data.shift_x = old_shift_x
        cam.data.shift_y = old_shift_y
        scene.render.filepath = old_filepath
        if old_use_comp is not None and hasattr(scene.render, "use_compositing"):
            scene.render.use_compositing = old_use_comp
        if temp_group is not None:
            _end_isolated_compositor(scene, old_group, temp_group)
        else:
            _remove_temp_compositor_nodes(scene)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

@dataclass
class FrameRecord:
    frame: int
    chosen_sub_index: int
    jitters_pixels: List[List[float]] = field(default_factory=list)
    chosen_beauty: str = ""
    accumulated: str = ""
    depth_ndc: str = ""
    mv_ndc: str = ""
    mvdepth_ndc: str = ""
    raw_depth: str = ""
    raw_vector: str = ""
    subsample_files: List[str] = field(default_factory=list)


def main() -> None:
    args = parse_args(sys.argv)
    scene = bpy.context.scene
    configure_scene(scene, args)
    cam = choose_camera(scene, args.camera)

    start = args.start if args.start is not None else scene.frame_start
    end = args.end if args.end is not None else scene.frame_end
    if end < start:
        raise RuntimeError(f"Invalid frame range: start={start}, end={end}")
    #if args.samples < 1:
    #    raise RuntimeError("--samples must be >= 1")

    out_dir = blender_abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[MRQ v19] VERSION={VERSION}", flush=True)
    print(f"[MRQ v19] parsed_args={vars(args)}", flush=True)
    print(f"[MRQ v19] blend={bpy.data.filepath}", flush=True)
    print(f"[MRQ v19] out={out_dir} engine={scene.render.engine} camera={cam.name}", flush=True)
    print(f"[MRQ v19] resolution={scene.render.resolution_x}x{scene.render.resolution_y} "
          f"frames={start}-{end} jitter={args.jitter}", flush=True)

    records: List[FrameRecord] = []

    for frame in range(start, end + 1):
        scene.frame_set(frame)
        try:
            bpy.context.view_layer.update()
        except Exception:
            pass
        frame_dir = os.path.join(out_dir, f"frame_{frame:04d}")
        os.makedirs(frame_dir, exist_ok=True)
        
        adjusted_frame = frame + args.frame_offset
        ref_jitter = signed_jitter(adjusted_frame, args.samples, "pmj")
        Seq, Seg = get_optimal_seq_seg_indices(ref_jitter, args.samples)
        chosen_jitter = signed_jitter(Seq, args.samples, "halton23")

        rec = FrameRecord(frame=frame, chosen_sub_index=Seq)
        acc_sum = None
        acc_count = 0
        raw_depth_path = ""
        raw_vector_path = ""
        
        filename = f"NPP_beauty_{adjusted_frame:04d}_{Seq:04d}_{Seg:01d}_{chosen_jitter[0]:+.8f}_{chosen_jitter[1]:+.8f}.exr"
        path = os.path.join(frame_dir, filename)
        
        with_passes = not args.no_passes
        info = render_subsample(scene, cam, path, frame, chosen_jitter[0], chosen_jitter[1], with_passes, args)
        
        if not (info["exists"] and info["bytes"] > 0):
            raise RuntimeError(f"Chosen jittered sample render produced no file: {path}")
        rec.chosen_beauty = path
        raw_depth_path = info["raw_passes"].get("depth", {}).get("path", "")
        raw_vector_path = info["raw_passes"].get("vector", {}).get("path", "")

        if not args.no_passes:
            ndc = postprocess_chosen_passes(frame_dir, raw_depth_path, raw_vector_path, cam, scene, args)
            rec.depth_ndc = ndc.get("depth_ndc", "")
            rec.mv_ndc = ndc.get("mv_ndc", "")
            rec.mvdepth_ndc = ndc.get("mvdepth_ndc", "")
            if args.keep_raw_passes:
                rec.raw_depth = raw_depth_path
                rec.raw_vector = raw_vector_path
            else:
                for p in (raw_depth_path, raw_vector_path):
                    if p and os.path.isfile(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass

        records.append(rec)
        print(f"[MRQ v19] frame {frame} done: chosen={os.path.basename(rec.chosen_beauty) if rec.chosen_beauty else '-'} "
              f"depth_ndc={'ok' if rec.depth_ndc else '-'} mv_ndc={'ok' if rec.mv_ndc else '-'}", flush=True)

    manifest = {
        "version": VERSION,
        "blend": bpy.data.filepath,
        "out_dir": out_dir,
        "camera": cam.name,
        "engine": scene.render.engine,
        "frames": [start, end],
        "samples": args.samples,
        "jitter": args.jitter,
        "depth_ndc_mode": args.depth_ndc_mode,
        "ray_depth_correction": args.ray_depth,
        "mv_components": args.mv_components,
        "mv_sign": [args.mv_sign_x, args.mv_sign_y],
        "mv_scale": args.mv_scale,
        "flip_y": args.flip_y,
        "notes": (
            "Jitter is applied as camera shift; the chosen beauty and its NDC depth/MV come from the same "
            "jittered render (identical jitter). accumulated.exr is the plain average of all subsamples. "
            "Vector-pass component order/sign and depth convention must be calibrated against your UE output; "
            "adjust --mv-components/--mv-sign-x/--mv-sign-y/--depth-ndc-mode after a known-motion test scene."
        ),
        "records": [asdict(r) for r in records],
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[MRQ v19] manifest={manifest_path}", flush=True)

    if not args.dry_run:
        missing = [r.frame for r in records if not r.chosen_beauty]
        if missing:
            raise RuntimeError(f"Frames missing chosen beauty output: {missing}")
    print("[MRQ v19] done.", flush=True)


if __name__ == "__main__":
    main()