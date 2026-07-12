#!/usr/bin/env python3
"""
generic_blender_mrq_v18_split_tmp_merge.py
Version: 2026-07-10-v18-split-samples-tempdir-merge

Scene-agnostic Blender command-line render driver for UE-MRQ-like jittered output.
This version keeps the beauty render as regular OPEN_EXR and uses temporary
Compositor File Output nodes to save Depth/Z and Vector passes as separate EXRs.

Run:
  blender -b scene.blend -P generic_blender_mrq_v6_passes.py -- --out C:/tmp/mrq_out --start 1 --end 1 --samples 64 --jitter pmj --save-all-subsamples
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import bpy

VERSION = "2026-07-10-v18-split-samples-tempdir-merge"

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
    depth_path: str = ""
    depth_exists: bool = False
    depth_bytes: int = 0
    vector_path: str = ""
    vector_exists: bool = False
    vector_bytes: int = 0
    mvdepth_path: str = ""
    mvdepth_exists: bool = False
    mvdepth_bytes: int = 0


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
    p = argparse.ArgumentParser(description="Generic Blender MRQ-like jittered render driver v18 split-samples tempdir merge")
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
    p.add_argument("--output-mode", choices=["all", "chosen", "both"], default="all",
                   help="all: write sample_XXXX for every subsample; chosen: only write chosen_sample; both: write all samples and an extra chosen_sample copy")
    p.add_argument("--loop-order", choices=["sample-major", "frame-major"], default="sample-major",
                   help="sample-major renders sample 0 across all frames, then sample 1, etc. This avoids a Blender 5 compositor/File Output state bug where 64 same-frame still renders can make passes stop after the first frame. frame-major matches the older order.")
    p.add_argument("--save-all-subsamples", action="store_true",
                   help="Compatibility alias: forces --output-mode all unless --output-mode both was explicitly supplied")
    p.add_argument("--only-chosen", action="store_true", help="Compatibility alias for --output-mode chosen")
    p.add_argument("--keep-temp", action="store_true", help="Alias for --save-all-subsamples")
    p.add_argument("--view-transform", default="Standard")
    p.add_argument("--look", default="None")
    p.add_argument("--exr-codec", default="ZIP")
    p.add_argument("--half", action="store_true", default=True)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--no-passes", action="store_true", help="Do not create compositor outputs for depth/vector passes")
    p.add_argument("--pass-color-depth", choices=["16", "32"], default="16", help="OpenEXR bit depth for depth/vector pass files")
    p.add_argument("--dry-run", action="store_true", help="Do not render, just write manifest and print planned paths")
    p.add_argument("--single-sample", type=int, default=None,
                   help="Render exactly this subsample index while keeping --samples as the jitter sequence length. This is the stable Blender 5 path when called once per sample.")
    p.add_argument("--sample-start", type=int, default=None,
                   help="First subsample index to render, inclusive. Defaults to 0 unless --single-sample is used.")
    p.add_argument("--sample-end", type=int, default=None,
                   help="Last subsample index to render, inclusive. Defaults to samples-1 unless --single-sample is used.")
    p.add_argument("--split-samples", action="store_true",
                   help="Parent mode: launch one fresh background Blender process per subsample. This avoids Blender 5 compositor/File Output state bugs observed when rendering 64 samples in one process.")
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
        print(f"[MRQ v18] warning: view_transform {args.view_transform!r} unavailable; keeping {scene.view_settings.view_transform!r}")
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
        # Some Blender versions expose motion-vector-related toggles under different names.
        for attr in ("use_pass_motion", "use_pass_motion_vector"):
            if hasattr(vl, attr):
                try:
                    setattr(vl, attr, True)
                except Exception:
                    pass


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



# -----------------------------------------------------------------------------
# Temporary compositor pass output helpers
# -----------------------------------------------------------------------------

TEMP_NODE_PREFIX = "__MRQ_TEMP_PASS__"


def get_compositor_tree(scene: bpy.types.Scene, create: bool = True):
    """Return the compositor node tree across Blender 4.x and 5.x.

    Blender 5 removed scene.node_tree and replaced it with
    scene.compositing_node_group. Blender 4.x still uses scene.node_tree and
    creates it when scene.use_nodes is enabled.
    """
    # Blender 4.x / legacy path.
    if hasattr(scene, "node_tree"):
        if create and hasattr(scene, "use_nodes"):
            try:
                scene.use_nodes = True
            except Exception:
                pass
        return getattr(scene, "node_tree", None)

    # Blender 5.x path.
    if hasattr(scene, "compositing_node_group"):
        tree = getattr(scene, "compositing_node_group", None)
        if tree is None and create:
            # In Blender 5 the compositor node tree is an explicit datablock.
            tree = bpy.data.node_groups.new(name=f"{scene.name}_MRQ_Compositor", type="CompositorNodeTree")
            scene.compositing_node_group = tree
        return tree

    return None


def _enable_compositor(scene: bpy.types.Scene) -> None:
    # Blender 4.x property; Blender 5 deprecates/ignores it, so guard it.
    if hasattr(scene, "use_nodes"):
        try:
            scene.use_nodes = True
        except Exception:
            pass
    # Blender 5 uses Output Properties > Post Processing > Compositing.
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
    """Use a fresh compositor node group for this render sample on Blender 5.

    Blender 5 keeps compositor node groups as datablocks. Reusing and mutating
    the same group across hundreds of still renders can leave stale interface
    sockets / File Output state. This helper assigns a brand-new temporary group
    before setup_compositor_pass_outputs() and restores the user's original
    compositor group after the render. Blender 4.x has scene.node_tree and uses
    the old path, so this is a no-op there.
    """
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
        if hasattr(scene.render, "use_compositing"):
            scene.render.use_compositing = True
        if getattr(args, "debug", False):
            print(f"[MRQ v18] isolated compositor group created: {temp_group.name}", flush=True)
        return old_group, temp_group
    except Exception as exc:
        print(f"[MRQ v18] warning: could not create isolated compositor group; using existing compositor tree: {exc}", flush=True)
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


def _find_output_socket(node: bpy.types.Node, names: List[str]):
    wanted = {n.lower() for n in names}
    for sock in node.outputs:
        if sock.name.lower() in wanted:
            return sock
    # Loose fallback for localized/minor name differences.
    for sock in node.outputs:
        low = sock.name.lower()
        if any(n.lower() in low for n in names):
            return sock
    return None


def _apply_image_format_settings(fmt, args: argparse.Namespace, color_mode: str) -> None:
    if fmt is None:
        return
    if hasattr(fmt, "file_format"):
        try:
            fmt.file_format = "OPEN_EXR"
        except Exception:
            pass
    if hasattr(fmt, "color_depth"):
        vals = enum_values(fmt, "color_depth")
        if args.pass_color_depth in vals:
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
        vals = enum_values(fmt, "color_mode")
        if color_mode in vals:
            try:
                fmt.color_mode = color_mode
            except Exception:
                pass


def _set_file_output_format(node: bpy.types.Node, args: argparse.Namespace, color_mode: str) -> None:
    # Blender 4.x exposes node.format. Blender 5.x can expose per-item formats.
    _apply_image_format_settings(getattr(node, "format", None), args, color_mode)
    items = getattr(node, "file_output_items", None)
    if items is not None:
        for item in items:
            _apply_image_format_settings(getattr(item, "format", None), args, color_mode)


def _set_file_output_directory(node: bpy.types.Node, frame_dir: str) -> None:
    # Blender <=4.x uses base_path; Blender 5.x renamed it to directory.
    if hasattr(node, "base_path"):
        node.base_path = frame_dir
    elif hasattr(node, "directory"):
        node.directory = frame_dir
    else:
        print("[MRQ v18] warning: File Output node has neither base_path nor directory", flush=True)


def _set_file_output_prefix(node: bpy.types.Node, prefix_without_extension: str, color_mode: str) -> int:
    """Configure one File Output node and return the input index to link.

    v13 fixes the v12 failure mode shown by inputs=[''] and no pass files. In
    Blender 5, the default File Output input can exist but not be backed by a
    real file_output_item. We remove every default item, create exactly one new
    item, and link input 0.
    """
    if hasattr(node, "file_slots") and len(node.file_slots) > 0:
        node.file_slots[0].path = prefix_without_extension
        print(f"[MRQ v18] File Output legacy slot: path={prefix_without_extension!r} input_index=0", flush=True)
        return 0

    if hasattr(node, "file_name"):
        try:
            node.file_name = prefix_without_extension
        except Exception as exc:
            print(f"[MRQ v18] warning: could not set File Output file_name={prefix_without_extension!r}: {exc}", flush=True)

    items = getattr(node, "file_output_items", None)
    if items is not None:
        # Delete Blender-created default blank items. In your v12 log the input
        # list was [''], and Blender wrote no pass files. A fresh explicit item
        # avoids that half-configured state.
        try:
            while len(items) > 0:
                items.remove(items[0])
        except Exception as exc:
            print(f"[MRQ v18] warning: could not clear default file_output_items: {exc}", flush=True)

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
            raise RuntimeError(f"Could not create Blender 5 File Output item for {prefix_without_extension!r}: {last_exc}")
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
        try:
            input_names = [inp.name for inp in node.inputs]
        except Exception:
            input_names = []
        try:
            item_names = [it.name for it in items]
        except Exception:
            item_names = []
        print(f"[MRQ v18] File Output recreated item/input0: file_name={getattr(node, 'file_name', None)!r} color_mode={color_mode!r} inputs={input_names!r} items={item_names!r}", flush=True)
        return 0

    return 0

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


def _new_separate_rgba_node(tree):
    """Return (node, r_socket, g_socket). Works on Blender 4 and 5."""
    # Blender 5 removed SepRGBA; use Separate Color.
    for node_type in ("CompositorNodeSeparateColor", "CompositorNodeSepRGBA"):
        try:
            node = tree.nodes.new(type=node_type)
            node.name = TEMP_NODE_PREFIX + "SeparateVector"
            node.label = TEMP_NODE_PREFIX + "SeparateVector"
            try:
                if hasattr(node, "mode"):
                    node.mode = "RGB"
            except Exception:
                pass
            r = _socket_by_names(node.outputs, ["Red", "R"])
            g = _socket_by_names(node.outputs, ["Green", "G"])
            if r is not None and g is not None:
                return node, r, g
        except Exception:
            continue
    raise RuntimeError("No Separate RGBA/Color compositor node available")


def _new_combine_rgba_node(tree):
    """Return (node, r_input, g_input, b_input, a_input, image_output). Works on Blender 4 and 5."""
    for node_type in ("CompositorNodeCombineColor", "CompositorNodeCombRGBA"):
        try:
            node = tree.nodes.new(type=node_type)
            node.name = TEMP_NODE_PREFIX + "CombineMVDepth"
            node.label = TEMP_NODE_PREFIX + "CombineMVDepth"
            try:
                if hasattr(node, "mode"):
                    node.mode = "RGB"
            except Exception:
                pass
            r = _socket_by_names(node.inputs, ["Red", "R"])
            g = _socket_by_names(node.inputs, ["Green", "G"])
            b = _socket_by_names(node.inputs, ["Blue", "B"])
            a = _socket_by_names(node.inputs, ["Alpha", "A"])
            out = _socket_by_names(node.outputs, ["Image", "RGBA", "Color"])
            if r is not None and g is not None and b is not None and out is not None:
                return node, r, g, b, a, out
        except Exception:
            continue
    raise RuntimeError("No Combine RGBA/Color compositor node available")

def _delete_stale_pass_files(frame_dir: str, stem: str, suffix: str) -> None:
    import glob
    exact = os.path.join(frame_dir, f"{stem}{suffix}.exr")
    for f in glob.glob(os.path.join(frame_dir, f"{stem}{suffix}*.exr")) + [exact]:
        try:
            if os.path.isfile(f):
                os.remove(f)
        except Exception:
            pass


def setup_compositor_pass_outputs(scene: bpy.types.Scene, frame_dir: str, stem: str, args: argparse.Namespace) -> dict:
    """Create temporary File Output nodes for Depth/Z and Vector.

    Returns a dict with requested exact output paths. Blender's File Output node appends
    frame numbers, so after rendering we glob and rename to these exact paths.
    """
    outputs = {}
    if args.no_passes:
        return outputs

    # Enable compositor without changing the scene's Composite/Viewer output.
    _enable_compositor(scene)

    tree = get_compositor_tree(scene, create=True)
    if tree is None:
        print("[MRQ v18] warning: no compositor node tree available; depth/vector passes will not be written", flush=True)
        return outputs
    _remove_temp_compositor_nodes(scene)

    rlayers = tree.nodes.new(type="CompositorNodeRLayers")
    rlayers.name = TEMP_NODE_PREFIX + "RenderLayers"
    rlayers.label = TEMP_NODE_PREFIX + "RenderLayers"
    try:
        rlayers.layer = scene.view_layers[0].name
    except Exception:
        pass

    # Blender 5 removed the old Composite output node from compositor node groups.
    # A compositor group now needs an explicit NodeGroupOutput, and its sockets
    # must be created on the tree.interface before linking. Without this, the
    # group can exist but never execute during a background render, so File Output
    # nodes silently write nothing.
    try:
        img_sock = _find_output_socket(rlayers, ["Image", "Combined", "Color"])
        if img_sock is not None:
            group_out = tree.nodes.new(type="NodeGroupOutput")
            group_out.name = TEMP_NODE_PREFIX + "GroupOutputKeepAlive"
            group_out.label = TEMP_NODE_PREFIX + "GroupOutputKeepAlive"
            # Blender 5 path: sockets are created at node-tree interface level.
            if hasattr(tree, "interface") and hasattr(tree.interface, "new_socket"):
                try:
                    tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
                except Exception:
                    # Socket may already exist if Blender reused the group datablock.
                    pass
            out_sock = _socket_by_names(group_out.inputs, ["Image", "Output", "Color"])
            if out_sock is not None:
                tree.links.new(img_sock, out_sock)
                print("[MRQ v18] compositor keepalive: RenderLayers.Image -> NodeGroupOutput.Image", flush=True)
            else:
                print("[MRQ v18] warning: NodeGroupOutput has no usable Image input", flush=True)
    except Exception as exc:
        print(f"[MRQ v18] warning: could not create compositor group-output keepalive: {exc}", flush=True)

    if args.debug:
        try:
            print("[MRQ v18] Render Layers outputs: " + ", ".join([sock.name for sock in rlayers.outputs]), flush=True)
        except Exception:
            pass

    def add_output(socket_names: List[str], suffix: str, color_mode: str) -> None:
        socket = _find_output_socket(rlayers, socket_names)
        exact = os.path.join(frame_dir, f"{stem}{suffix}.exr")
        outputs[suffix.lstrip("_")] = {"exact": exact, "socket": "" if socket is None else socket.name}
        if socket is None:
            print(f"[MRQ v18] warning: no compositor socket found for {socket_names}; pass {suffix} will not be written", flush=True)
            return
        _delete_stale_pass_files(frame_dir, stem, suffix)
        out = tree.nodes.new(type="CompositorNodeOutputFile")
        out.name = TEMP_NODE_PREFIX + suffix
        out.label = TEMP_NODE_PREFIX + suffix
        _set_file_output_directory(out, frame_dir)
        # File Output appends frame number to this prefix. No extension here.
        input_index = _set_file_output_prefix(out, f"{stem}{suffix}_", color_mode)
        _set_file_output_format(out, args, color_mode)
        tree.links.new(socket, out.inputs[input_index])
        print(f"[MRQ v18] compositor pass {suffix}: socket={socket.name!r} -> {exact}", flush=True)

    # Depth/Z is scalar. Vector pass is generally RGBA-like in Blender; exact channel semantics vary by engine/version.
    add_output(["Depth", "Z"], "_depth", "BW")
    add_output(["Vector", "Motion Vector", "MotionVector"], "_vector", "RGBA")

    # Best-effort UE-style packed MVD sidecar: R=Vector.R, G=Vector.G, B=Depth, A=1.
    # This is intentionally raw Blender vector data; sign/scale still needs calibration against UE.
    depth_socket = _find_output_socket(rlayers, ["Depth", "Z"])
    vector_socket = _find_output_socket(rlayers, ["Vector", "Motion Vector", "MotionVector"])
    exact = os.path.join(frame_dir, f"{stem}_mvdepth.exr")
    outputs["mvdepth"] = {"exact": exact, "socket": "Vector+Depth" if depth_socket and vector_socket else ""}
    if depth_socket is not None and vector_socket is not None:
        try:
            _delete_stale_pass_files(frame_dir, stem, "_mvdepth")
            sep, sep_r, sep_g = _new_separate_rgba_node(tree)
            comb, comb_r, comb_g, comb_b, comb_a, comb_out = _new_combine_rgba_node(tree)
            out = tree.nodes.new(type="CompositorNodeOutputFile")
            out.name = TEMP_NODE_PREFIX + "_mvdepth"
            out.label = TEMP_NODE_PREFIX + "_mvdepth"
            _set_file_output_directory(out, frame_dir)
            input_index = _set_file_output_prefix(out, f"{stem}_mvdepth_", "RGBA")
            _set_file_output_format(out, args, "RGBA")
            tree.links.new(vector_socket, sep.inputs[0])
            tree.links.new(sep_r, comb_r)      # R = Vector.R
            tree.links.new(sep_g, comb_g)      # G = Vector.G
            tree.links.new(depth_socket, comb_b)  # B = Depth/Z
            if comb_a is not None:
                try:
                    comb_a.default_value = 1.0
                except Exception:
                    pass
            tree.links.new(comb_out, out.inputs[input_index])
            print(f"[MRQ v18] compositor pass _mvdepth: sockets=({vector_socket.name!r},{depth_socket.name!r}) -> {exact}", flush=True)
        except Exception as exc:
            print(f"[MRQ v18] warning: could not create combined _mvdepth output: {exc}", flush=True)
    else:
        print("[MRQ v18] warning: cannot create _mvdepth because Depth/Z or Vector socket is missing", flush=True)
    return outputs


def collect_compositor_pass_outputs(frame_dir: str, stem: str, suffix: str, frame: int) -> Tuple[str, bool, int]:
    import glob
    exact = os.path.join(frame_dir, f"{stem}{suffix}.exr")
    candidates = []
    patterns = [
        os.path.join(frame_dir, f"{stem}{suffix}.exr"),
        os.path.join(frame_dir, f"{stem}{suffix}_*.exr"),
        os.path.join(frame_dir, f"{stem}{suffix}*.exr"),
        os.path.join(frame_dir, f"{stem}{suffix}_Image*.exr"),
        os.path.join(frame_dir, f"{stem}{suffix}_Z*.exr"),
        os.path.join(frame_dir, "**", f"{stem}{suffix}.exr"),
        os.path.join(frame_dir, "**", f"{stem}{suffix}_*.exr"),
        os.path.join(frame_dir, "**", f"{stem}{suffix}*.exr"),
    ]
    for pat in patterns:
        candidates.extend(glob.glob(pat, recursive=True))
    # Prefer newest non-empty file.
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
        return exact, os.path.isfile(exact), os.path.getsize(exact) if os.path.isfile(exact) else 0
    return exact, os.path.isfile(exact), os.path.getsize(exact) if os.path.isfile(exact) else 0

def render_sample(scene: bpy.types.Scene, cam: bpy.types.Object, path: str, frame: int, jx: float, jy: float, args: argparse.Namespace) -> Tuple[bool, int, dict]:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
    old_shift_x = cam.data.shift_x
    old_shift_y = cam.data.shift_y
    old_filepath = scene.render.filepath
    old_comp_group = None
    temp_comp_group = None
    
    '''
    for ob in bpy.data.objects:
        if ob.type in {'MESH'}:# {'MESH', 'CURVES', 'POINTCLOUD'}:
            ob.hide_render = True
    for ps in ob.modifiers:
        if ps.type == 'PARTICLE_SYSTEM':
            ps.show_render = False
    '''
    
    try:
        old_comp_group, temp_comp_group = _begin_isolated_compositor(scene, args)
        dx = jx / float(width)
        dy = -jy / float(height)
        cam.data.shift_x = old_shift_x + dx
        cam.data.shift_y = old_shift_y + dy
        scene.render.filepath = path
        stem = os.path.splitext(os.path.basename(path))[0]
        pass_requests = setup_compositor_pass_outputs(scene, os.path.dirname(path), stem, args)
        print(f"[MRQ v18] render frame={frame} sample_path={path} jitter=({jx:+.8f},{jy:+.8f}) cam_shift_delta=({dx:+.10f},{dy:+.10f})", flush=True)
        pass_outputs = {}
        if not args.dry_run:
            bpy.ops.render.render(write_still=True)
            check_or_save_render_result(scene, path, frame)
            for suffix in ("_depth", "_vector", "_mvdepth"):
                pth, ok, nbytes = collect_compositor_pass_outputs(os.path.dirname(path), stem, suffix, frame)
                pass_outputs[suffix.lstrip("_")] = {"path": pth, "exists": ok, "bytes": nbytes}
                if ok:
                    print(f"[MRQ v18] wrote pass {suffix}: {pth} bytes={nbytes}", flush=True)
                elif not args.no_passes:
                    print(f"[MRQ v18] warning: pass {suffix} was not written for {path}", flush=True)
                    try:
                        import glob as _glob
                        nearby = sorted(_glob.glob(os.path.join(os.path.dirname(path), "*.exr")))[:20]
                        print(f"[MRQ v18] nearby exr files in frame dir: {nearby}", flush=True)
                    except Exception:
                        pass
        else:
            for suffix in ("_depth", "_vector", "_mvdepth"):
                pth = os.path.join(os.path.dirname(path), f"{stem}{suffix}.exr")
                pass_outputs[suffix.lstrip("_")] = {"path": pth, "exists": False, "bytes": 0}
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        return exists, size, pass_outputs
    finally:
        cam.data.shift_x = old_shift_x
        cam.data.shift_y = old_shift_y
        scene.render.filepath = old_filepath
        if temp_comp_group is not None:
            _end_isolated_compositor(scene, old_comp_group, temp_comp_group)
        else:
            _remove_temp_compositor_nodes(scene)


def _arg_list_for_child(args: argparse.Namespace, start: int, end: int, sample: int, out_dir: str, scene: bpy.types.Scene, cam: bpy.types.Object) -> List[str]:
    child = [
        "--out", out_dir,
        "--start", str(start),
        "--end", str(end),
        "--samples", str(args.samples),
        "--single-sample", str(sample),
        "--jitter", args.jitter,
        "--chosen-mode", args.chosen_mode,
        "--output-mode", args.output_mode,
        "--loop-order", "sample-major",
        "--cycles-render-samples", str(args.cycles_render_samples),
        "--pass-color-depth", args.pass_color_depth,
        "--view-transform", args.view_transform,
        "--look", args.look,
        "--exr-codec", args.exr_codec,
    ]
    if args.engine:
        child += ["--engine", args.engine]
    if args.camera:
        child += ["--camera", args.camera]
    elif cam is not None:
        child += ["--camera", cam.name]
    if args.resolution_x:
        child += ["--resolution-x", str(args.resolution_x)]
    if args.resolution_y:
        child += ["--resolution-y", str(args.resolution_y)]
    if args.only_chosen:
        child.append("--only-chosen")
    if args.save_all_subsamples:
        child.append("--save-all-subsamples")
    if args.keep_temp:
        child.append("--keep-temp")
    if args.no_passes:
        child.append("--no-passes")
    if args.debug:
        child.append("--debug")
    if args.dry_run:
        child.append("--dry-run")
    return child


def _run_split_samples_parent(args: argparse.Namespace, start: int, end: int, out_dir: str, scene: bpy.types.Scene, cam: bpy.types.Object) -> None:
    """Spawn one fresh Blender process per subsample.

    This is slower to launch but much more reliable on Blender 5: we already
    observed that --samples 1 across all frames writes Depth/Vector correctly,
    while 64 samples in one process can leave later samples or frames with only
    beauty output. Splitting keeps the same PMJ/Halton sequence because each
    child receives --samples N and --single-sample i.
    """
    script_path = os.path.abspath(__file__)
    blend_path = bpy.data.filepath
    blender_bin = bpy.app.binary_path
    if not blend_path:
        raise RuntimeError("--split-samples requires the .blend to be saved on disk")

    s0 = 0 if args.sample_start is None else args.sample_start
    s1 = args.samples - 1 if args.sample_end is None else args.sample_end
    if args.single_sample is not None:
        s0 = s1 = args.single_sample
    if s0 < 0 or s1 < s0 or s1 >= args.samples:
        raise RuntimeError(f"Invalid split sample range {s0}-{s1} for --samples {args.samples}")

    print(f"[MRQ v18] split_samples parent: blender={blender_bin}", flush=True)
    print(f"[MRQ v18] split_samples parent: blend={blend_path}", flush=True)
    print(f"[MRQ v18] split_samples parent: samples={s0}-{s1} frames={start}-{end}", flush=True)

    # Render each subsample into its own private temp output root, then merge/copy
    # the resulting frame_* directories into the final out_dir. This avoids two
    # Blender-5 failure modes we have observed in production scenes:
    #   1) child processes or compositor file-output nodes touching stale files
    #      in the shared frame directory;
    #   2) users seeing only the last child manifest/sample after sequential runs.
    split_tmp_root = os.path.join(out_dir, "_split_tmp")
    os.makedirs(split_tmp_root, exist_ok=True)

    failures = []
    for sample in range(s0, s1 + 1):
        child_out_dir = os.path.join(split_tmp_root, f"sample_{sample:04d}")
        os.makedirs(child_out_dir, exist_ok=True)
        child_args = _arg_list_for_child(args, start, end, sample, child_out_dir, scene, cam)
        cmd = [blender_bin, "-b", blend_path, "-P", script_path, "--"] + child_args
        print(f"[MRQ v18] launching sample {sample:04d}/{args.samples-1:04d}: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd)
        print(f"[MRQ v18] child sample {sample:04d} returncode={proc.returncode}", flush=True)
        if proc.returncode != 0:
            failures.append((sample, proc.returncode))
            print(f"[MRQ v18] ERROR sample {sample:04d} failed with code {proc.returncode}", flush=True)
            break

        # Merge this sample's outputs into the final shared frame directories.
        # We copy, not move, so the temp directory remains available for debugging.
        for frame in range(start, end + 1):
            src_frame_dir = os.path.join(child_out_dir, f"frame_{frame:04d}")
            dst_frame_dir = os.path.join(out_dir, f"frame_{frame:04d}")
            os.makedirs(dst_frame_dir, exist_ok=True)
            if os.path.isdir(src_frame_dir):
                for name in os.listdir(src_frame_dir):
                    src = os.path.join(src_frame_dir, name)
                    dst = os.path.join(dst_frame_dir, name)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)

        # Copy child manifest to the final root using the sample-specific name.
        child_manifest = os.path.join(child_out_dir, f"manifest_sample_{sample:04d}.json")
        if os.path.isfile(child_manifest):
            shutil.copy2(child_manifest, os.path.join(out_dir, f"manifest_sample_{sample:04d}.json"))

    if failures:
        raise RuntimeError(f"Split-sample render failed: {failures}")

    # Children write manifest_sample_XXXX.json so they do not overwrite each other.
    # Merge them here and also verify actual files on disk by glob, which prevents
    # confusing a single child manifest with missing sample files.
    merged_records = []
    child_manifests = []
    for sample in range(s0, s1 + 1):
        mp = os.path.join(out_dir, f"manifest_sample_{sample:04d}.json")
        if os.path.isfile(mp):
            child_manifests.append(mp)
            try:
                with open(mp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                merged_records.extend(data.get("records", []))
            except Exception as exc:
                print(f"[MRQ v18] warning: failed to read child manifest {mp}: {exc}", flush=True)
        else:
            print(f"[MRQ v18] warning: missing child manifest {mp}", flush=True)

    frame_counts = {}
    for frame in range(start, end + 1):
        frame_dir = os.path.join(out_dir, f"frame_{frame:04d}")
        import glob as _glob
        beauty = [x for x in _glob.glob(os.path.join(frame_dir, "sample_*.exr"))
                  if not (x.endswith("_depth.exr") or x.endswith("_vector.exr") or x.endswith("_mvdepth.exr"))]
        depth = _glob.glob(os.path.join(frame_dir, "sample_*_depth.exr"))
        vector = _glob.glob(os.path.join(frame_dir, "sample_*_vector.exr"))
        mvdepth = _glob.glob(os.path.join(frame_dir, "sample_*_mvdepth.exr"))
        frame_counts[str(frame)] = {
            "beauty": len(beauty),
            "depth": len(depth),
            "vector": len(vector),
            "mvdepth": len(mvdepth),
        }
        print(f"[MRQ v18] verify frame={frame}: beauty={len(beauty)} depth={len(depth)} vector={len(vector)} mvdepth={len(mvdepth)}", flush=True)

    merged = {
        "version": VERSION,
        "blend": bpy.data.filepath,
        "out_dir": out_dir,
        "frames": [start, end],
        "samples": args.samples,
        "rendered_sample_indices": list(range(s0, s1 + 1)),
        "child_manifests": child_manifests,
        "frame_counts": frame_counts,
        "records": merged_records,
    }
    merged_path = os.path.join(out_dir, "manifest.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"[MRQ v18] merged_manifest={merged_path}", flush=True)
    print("[MRQ v18] split_samples parent done", flush=True)


def main() -> None:
    args = parse_args(sys.argv)
    if args.keep_temp:
        args.save_all_subsamples = True
    if args.only_chosen:
        args.output_mode = "chosen"
    elif args.save_all_subsamples and args.output_mode != "both":
        args.output_mode = "all"
    scene = bpy.context.scene
    configure_scene(scene, args)
    cam = choose_camera(scene, args.camera)

    start = args.start if args.start is not None else scene.frame_start
    end = args.end if args.end is not None else scene.frame_end
    if end < start:
        raise RuntimeError(f"Invalid frame range: start={start}, end={end}")
    if args.samples < 1:
        raise RuntimeError("--samples must be >= 1")
    script_tail = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--samples" not in script_tail and not any(a.startswith("--samples=") for a in script_tail):
        print("[MRQ v18] warning: --samples was not found after Blender's -- separator; using default samples=64", flush=True)

    out_dir = blender_abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # Parent mode: do not render inside this process; spawn one clean Blender
    # process per subsample. Children receive --single-sample and render all frames.
    if args.split_samples and args.single_sample is None:
        print(f"[MRQ v18] VERSION={VERSION}", flush=True)
        print(f"[MRQ v18] parsed_args={vars(args)}", flush=True)
        print(f"[MRQ v18] raw_argv={sys.argv}", flush=True)
        _run_split_samples_parent(args, start, end, out_dir, scene, cam)
        return

    print(f"[MRQ v18] VERSION={VERSION}", flush=True)
    print(f"[MRQ v18] parsed_args={vars(args)}", flush=True)
    print(f"[MRQ v18] raw_argv={sys.argv}", flush=True)
    print(f"[MRQ v18] blend={bpy.data.filepath}", flush=True)
    print(f"[MRQ v18] output_dir={out_dir}", flush=True)
    print(f"[MRQ v18] engine={scene.render.engine} format={scene.render.image_settings.file_format} camera={cam.name}", flush=True)
    print(f"[MRQ v18] output_mode={args.output_mode}", flush=True)
    print(f"[MRQ v18] resolution={scene.render.resolution_x}x{scene.render.resolution_y} frames={start}-{end} samples={args.samples}", flush=True)

    chosen = choose_subsample(args.samples, args)
    records: List[SampleRecord] = []

    def render_one(frame: int, sample: int) -> None:
        scene.frame_set(frame)
        # Force depsgraph evaluation at the current frame. This is especially important
        # for Blender 5 compositor/File Output passes when rendering many stills.
        try:
            bpy.context.view_layer.update()
        except Exception:
            pass
        frame_dir = os.path.join(out_dir, f"frame_{frame:04d}")
        os.makedirs(frame_dir, exist_ok=True)
        jx, jy = signed_jitter(sample, args.samples, args.jitter)
        is_chosen = sample == chosen

        should_render_sample = args.output_mode in {"all", "both"} or (args.output_mode == "chosen" and is_chosen)
        if not should_render_sample:
            return

        if args.output_mode == "chosen":
            filename = f"chosen_sample_{sample:04d}_jx_{jx:+.8f}_jy_{jy:+.8f}.exr"
        else:
            filename = f"sample_{sample:04d}_jx_{jx:+.8f}_jy_{jy:+.8f}.exr"

        path = os.path.join(frame_dir, filename)
        exists, size, pass_outputs = render_sample(scene, cam, path, frame, jx, jy, args)
        depth_info = pass_outputs.get("depth", {"path": "", "exists": False, "bytes": 0})
        vector_info = pass_outputs.get("vector", {"path": "", "exists": False, "bytes": 0})
        mvdepth_info = pass_outputs.get("mvdepth", {"path": "", "exists": False, "bytes": 0})
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
            depth_path=depth_info.get("path", ""),
            depth_exists=bool(depth_info.get("exists", False)),
            depth_bytes=int(depth_info.get("bytes", 0)),
            vector_path=vector_info.get("path", ""),
            vector_exists=bool(vector_info.get("exists", False)),
            vector_bytes=int(vector_info.get("bytes", 0)),
            mvdepth_path=mvdepth_info.get("path", ""),
            mvdepth_exists=bool(mvdepth_info.get("exists", False)),
            mvdepth_bytes=int(mvdepth_info.get("bytes", 0)),
        ))

        if args.output_mode == "both" and is_chosen and exists and size > 0:
            chosen_filename = f"chosen_sample_{sample:04d}_jx_{jx:+.8f}_jy_{jy:+.8f}.exr"
            chosen_path = os.path.join(frame_dir, chosen_filename)
            if os.path.abspath(chosen_path) != os.path.abspath(path):
                shutil.copy2(path, chosen_path)
                chosen_exists = os.path.isfile(chosen_path)
                chosen_size = os.path.getsize(chosen_path) if chosen_exists else 0
                chosen_depth_path = ""
                chosen_vector_path = ""
                chosen_mvdepth_path = ""
                chosen_depth_exists = False
                chosen_vector_exists = False
                chosen_mvdepth_exists = False
                chosen_depth_size = 0
                chosen_vector_size = 0
                chosen_mvdepth_size = 0
                for pass_name in ("depth", "vector", "mvdepth"):
                    src_info = pass_outputs.get(pass_name, {})
                    src_pass = src_info.get("path", "")
                    if src_pass and os.path.isfile(src_pass):
                        dst_pass = os.path.splitext(chosen_path)[0] + f"_{pass_name}.exr"
                        shutil.copy2(src_pass, dst_pass)
                        if pass_name == "depth":
                            chosen_depth_path = dst_pass; chosen_depth_exists = True; chosen_depth_size = os.path.getsize(dst_pass)
                        elif pass_name == "vector":
                            chosen_vector_path = dst_pass; chosen_vector_exists = True; chosen_vector_size = os.path.getsize(dst_pass)
                        else:
                            chosen_mvdepth_path = dst_pass; chosen_mvdepth_exists = True; chosen_mvdepth_size = os.path.getsize(dst_pass)
                records.append(SampleRecord(
                    frame=frame,
                    sample=sample,
                    current_sub_index=sample,
                    chosen_sub_index=chosen,
                    jitter_x_pixels=jx,
                    jitter_y_pixels=jy,
                    camera_shift_x_delta=jx / float(width),
                    camera_shift_y_delta=-jy / float(height),
                    path=chosen_path,
                    exists=chosen_exists,
                    bytes=chosen_size,
                    is_chosen=True,
                    depth_path=chosen_depth_path,
                    depth_exists=chosen_depth_exists,
                    depth_bytes=chosen_depth_size,
                    vector_path=chosen_vector_path,
                    vector_exists=chosen_vector_exists,
                    vector_bytes=chosen_vector_size,
                    mvdepth_path=chosen_mvdepth_path,
                    mvdepth_exists=chosen_mvdepth_exists,
                    mvdepth_bytes=chosen_mvdepth_size,
                ))

    # Determine which subsamples this process owns. In --split-samples child mode
    # this will be exactly one sample, but --sample-start/--sample-end are useful
    # for manual chunking too. --samples still means the full jitter sequence length.
    if args.single_sample is not None:
        sample_indices = [args.single_sample]
    else:
        s0 = 0 if args.sample_start is None else args.sample_start
        s1 = args.samples - 1 if args.sample_end is None else args.sample_end
        if s0 < 0 or s1 < s0 or s1 >= args.samples:
            raise RuntimeError(f"Invalid sample range {s0}-{s1} for --samples {args.samples}")
        sample_indices = list(range(s0, s1 + 1))

    print(f"[MRQ v18] loop_order={args.loop_order} sample_indices={sample_indices[:8]}{'...' if len(sample_indices) > 8 else ''} count={len(sample_indices)}", flush=True)
    if args.loop_order == "sample-major":
        for sample in sample_indices:
            for frame in range(start, end + 1):
                render_one(frame, sample)
    else:
        for frame in range(start, end + 1):
            for sample in sample_indices:
                render_one(frame, sample)

    manifest = {
        "version": VERSION,
        "blend": bpy.data.filepath,
        "out_dir": out_dir,
        "camera": cam.name,
        "engine": scene.render.engine,
        "file_format": scene.render.image_settings.file_format,
        "frames": [start, end],
        "samples": args.samples,
        "rendered_sample_indices": sample_indices,
        "jitter": args.jitter,
        "chosen_sub_index": chosen,
        "output_mode": args.output_mode,
        "loop_order": args.loop_order,
        "passes_enabled": not args.no_passes,
        "pass_note": "Depth, Vector, and best-effort packed MVDepth (R=Vector.R, G=Vector.G, B=Depth) are written as sidecar EXRs via compositor File Output nodes; vector channel semantics depend on Blender engine/version.",
        "records": [asdict(r) for r in records],
    }
    if args.single_sample is not None:
        manifest_path = os.path.join(out_dir, f"manifest_sample_{args.single_sample:04d}.json")
    else:
        manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[MRQ v18] manifest={manifest_path}", flush=True)
    produced = [r for r in records if r.exists and r.bytes > 0]
    print(f"[MRQ v18] done. produced_files={len(produced)}", flush=True)
    if len(produced) == 0 and not args.dry_run:
        raise RuntimeError("No files produced. See paths printed above and manifest.json.")


if __name__ == "__main__":
    main()
