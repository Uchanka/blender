# ============================================================================
#  Blender port of custom UE MRQ pipeline  (rev 2)
#
#  Faithful translation of:
#    - MovieRenderOverlappedImage.cpp  (Sacht-Nehab quadratic quasi-interpolant
#      accumulator: prefilter pentadiagonal solve + 3-tap separable gather)
#    - MoviePipelineUtils.cpp          (Halton / shuffled-PMJ jitter, uint32-exact)
#
#  Compatible with Blender 4.x AND Blender 5.0+ (main branch):
#    - scene.compositing_node_group vs scene.use_nodes/node_tree
#    - File Output node: directory/file_name/file_output_items vs
#      base_path/file_slots; per-item format override vs node-level format
#    - temp file discovery via glob (5.0 filename decoration is not assumed)
#
#  Usage:  blender -b your.blend -P MRQ.py
#
#  Outputs per frame (float16 + ZIP(zlib) OpenEXR):
#    final.####.exr       64-sample Sacht-Nehab accumulated beauty
#    subsample.####.exr   the chosen single subsample
#    mvdepth.####.exr     R=MV.x  G=MV.y  B=depth, same jitter as chosen sample
# ============================================================================

import bpy, os, glob
import numpy as np

# ============================== CONFIG ======================================
OUT_DIR = r"C:\Users\grill\Desktop\BlenderMTSS\out"
TMP_DIR = r"C:\Users\grill\Desktop\BlenderMTSS\tmp"

SPATIAL_SAMPLE_COUNT = 64
ACCUMULATION_GAMMA   = 1.0     # FImageOverlappedAccumulator::AccumulationGamma
IS_NPP               = False   # isNPP -> SafeLog1pOrZero on RGB before accumulation

# How the chosen subsample relates to the two jitter sequences.
#   'RENDER_HALTON' : the chosen subsample is *rendered* with the Halton23
#                     runtime-TAA jitter (GetSubPixelJitter); the other 63 use
#                     shuffled PMJ over the global sample index
#                     (GetUnderlyingPixelJitter).  <-- default guess
#   'NEAREST_PMJ'   : all 64 samples use PMJ; the chosen one is the PMJ sample
#                     closest to this frame's Halton23 jitter.
# NOTE: the call sites of GetSubPixelJitter / GetUnderlyingPixelJitter were
# not in the uploaded files; edit get_frame_jitters()/choose_subsample() if
# your MoviePipelineRendering logic differs.
CHOSEN_JITTER_MODE = 'RENDER_HALTON'

# --- Camera shift signs (DERIVED, not guessed) -------------------------------
# UE: DefaultJitter = (jx * 2/W, jy * -2/H) added to the projection matrix
#     => content moves (+jx px right, +jy px down) in top-down image space;
#     accumulator receives SubpixelOffset = 0.5 - jitter.
# Blender: +shift_x moves the film right  => content moves LEFT;
#          +shift_y moves the film up     => content moves DOWN (top-down).
# Hence X flips, Y does not:
SHIFT_SIGN_X = -1.0
SHIFT_SIGN_Y = +1.0

# --- Motion vector output convention ----------------------------------------
# Cycles 'Vector' pass is 4 channels in pixel units. Verify once with a
# uniformly-moving object which pair is "towards previous frame" and its
# sign convention, then adjust below (see VERIFICATION NOTES).
MV_PREV_CHANNELS = (0, 1)      # indices into the Vector pass RGBA
MV_SIGN_X, MV_SIGN_Y = 1.0, -1.0

U32 = 0xFFFFFFFF

# ============================================================================
#  SECTION 1 -- Jitter sequences (bit-exact ports from MoviePipelineUtils.cpp,
#               namespace MTSSDevOpTrainingSet)
# ============================================================================

def halton(index, base):
    """float32-exact port of MTSSDevOpTrainingSet::Halton."""
    result   = np.float32(0.0)
    inv_base = np.float32(1.0) / np.float32(base)
    fraction = inv_base
    while index > 0:
        result   = np.float32(result + np.float32(index % base) * fraction)
        index    = index // base
        fraction = np.float32(fraction * inv_base)
    return float(result)

def get_halton23_subpixel_jitter(frame_index, samples_per_frame):
    """GetHalton23SubPixelJitter: repeats identically each output frame."""
    halton_index = (frame_index % samples_per_frame) + 1
    return (np.float32(halton(halton_index, 2) - 0.5),
            np.float32(halton(halton_index, 3) - 0.5))

def pcg_hash(v):
    """pcgHash: 32-bit xor-multiply chain."""
    v &= U32
    v ^= (v * 0x6c50b47c) & U32; v &= U32
    v ^= (v * 0xb82f1e52) & U32; v &= U32
    v ^= (v * 0xc7afe638) & U32; v &= U32
    v ^= (v * 0x8d22f6e6) & U32; v &= U32
    return v

def permute_pmj(i, l, seed):
    """permutePMJ, uint32-exact. l must be a power of two (8 here)."""
    w = (l - 1) & U32
    i = (i ^ seed) & U32
    i = (i * 0xe170893d) & U32
    i = (i ^ (seed >> 16)) & U32
    i ^= (i & w) >> 4
    i = (i ^ (seed >> 8)) & U32
    i = (i * 0x0929eb3f) & U32
    i = (i ^ (seed >> 23)) & U32
    i ^= (i & w) >> 1
    i = (i * ((1 | (seed >> 27)) & U32)) & U32
    i = (i * 0x6935fa69) & U32
    i ^= (i & w) >> 11
    i = (i * 0x74dcb303) & U32
    i ^= (i & w) >> 2
    i = (i * 0x9e501cc3) & U32
    i ^= (i & w) >> 2
    i = (i * 0xc860a3df) & U32
    i &= w
    i ^= i >> 5
    return (i + seed) & w

def pmj(i, l):
    """pmj(i, l) -> point in [0,1)^2.  M hardcoded to 8 as in source (the
    original carries a FIXME that it should be sqrt(l))."""
    i &= U32
    M  = 8
    ix = i & 7
    iy = (i >> 3) & U32
    ix = permute_pmj(ix, M, (0x51633e2d ^ iy) & U32)
    iy = permute_pmj(iy, M, (0x68bc21eb ^ ix) & U32)   # note: uses updated ix
    jx = float(pcg_hash(i) & 0xffff) * (1.0 / 65536.0)
    jy = float(pcg_hash((i ^ 0x9e3779b9) & U32) & 0xffff) * (1.0 / 65536.0)
    return (np.float32((float(ix) + jx) / float(M)),
            np.float32((float(iy) + jy) / float(M)))

def get_subpixel_jitter(frame_index, samples_per_frame):
    """UE::MoviePipeline::GetSubPixelJitter (active branch: Halton23)."""
    return get_halton23_subpixel_jitter(frame_index, samples_per_frame)

def get_underlying_pixel_jitter(sample_index, samples_per_frame):
    """UE::MoviePipeline::GetUnderlyingPixelJitter (active branch: pmj - 0.5).
    sample_index is the global counter: 'sample sequence num = spatial*temporal'."""
    px, py = pmj(sample_index, samples_per_frame)
    return (np.float32(px - np.float32(0.5)), np.float32(py - np.float32(0.5)))

# ---- selection + mixing policy (INFERRED -- see CHOSEN_JITTER_MODE note) ----

def choose_subsample(frame, jitters):
    """Which of the 64 subsamples is saved this frame."""
    if CHOSEN_JITTER_MODE == 'NEAREST_PMJ':
        hx, hy = get_subpixel_jitter(frame, SPATIAL_SAMPLE_COUNT)
        d = [(jx - hx) ** 2 + (jy - hy) ** 2 for (jx, jy) in jitters]
        return int(np.argmin(d))
    return frame % SPATIAL_SAMPLE_COUNT          # runtime TAA cycles 0..63

def get_frame_jitters(frame):
    """All 64 subsample jitters for this output frame, in pixels [-0.5, 0.5),
    image space +Y down."""
    spp = SPATIAL_SAMPLE_COUNT
    jitters = [get_underlying_pixel_jitter(frame * spp + si, spp)
               for si in range(spp)]
    if CHOSEN_JITTER_MODE == 'RENDER_HALTON':
        jitters[frame % spp] = get_subpixel_jitter(frame, spp)
    return jitters

# ============================================================================
#  SECTION 2 -- Sacht-Nehab accumulator
#               (port of namespace SachtNehabQuadratic + FImageOverlapped*)
# ============================================================================

# Blu-basis decomposition coefficients (Appendix A, quadratic case N=2, W=3)
B00 = np.float32( 0.75627421)
B01 = np.float32( 0.11798097)
B11 = np.float32( 0.11798097)
C00 = np.float32( 0.01588197)
C10 = np.float32(-0.02400002)
C20 = np.float32( 0.01588197)
# Prefilter pentadiagonal stencil E = [E2 E1 E0 E1 E2]
E0  = np.float32( 0.65314970)
E1  = np.float32( 0.17889730)
E2  = np.float32(-0.00547216)

LUT_SIZE = 1024

def _beta0_nc(x):
    return 1.0 if (0.0 <= x < 1.0) else 0.0

def _beta1_nc(x):
    if 0.0 <= x < 1.0: return x
    if 1.0 <= x < 2.0: return 2.0 - x
    return 0.0

def _beta2_nc(x):
    if 0.0 <= x < 1.0: return 0.5 * x * x
    if 1.0 <= x < 2.0: return -x * x + 3.0 * x - 1.5
    if 2.0 <= x < 3.0:
        s = 3.0 - x
        return 0.5 * s * s
    return 0.0

def _eval_generator(x):
    t = x + 1.5
    return float(B00) * _beta2_nc(t) \
         + float(B01) * _beta1_nc(t) \
         + float(B11) * _beta1_nc(t - 1.0) \
         + float(C00) * _beta0_nc(t) \
         + float(C10) * _beta0_nc(t - 1.0) \
         + float(C20) * _beta0_nc(t - 2.0)

# Weights[i] = generator at (1-t, -t, -1-t) for t = i/LUT_SIZE
_KERNEL_LUT = np.array(
    [[_eval_generator(1.0 - t), _eval_generator(-t), _eval_generator(-1.0 - t)]
     for t in (np.arange(LUT_SIZE + 1) / float(LUT_SIZE))],
    dtype=np.float32)

def get_cached_axis_kernel(subpixel_offset):
    """GetCachedAxisKernel: offset in [0,1] -> (start_bias, 3 weights).
    FMath::RoundToInt == floor(x + 0.5).
    NOTE: faithfully reproduces the original's dropped phi(2-t) tap on the
    bias == -1 side (offset < 0.5); the weight plane uses the same truncated
    kernel so DC content normalizes exactly. Keep in sync if the UE side is
    ever changed."""
    assert 0.0 <= subpixel_offset <= 1.0
    start_bias = 0 if subpixel_offset >= 0.5 else -1
    t = (subpixel_offset + 0.5) % 1.0                       # FMath::Frac
    b = int(np.clip(np.floor(t * LUT_SIZE + 0.5), 0, LUT_SIZE))
    return start_bias, _KERNEL_LUT[b]

def safe_pow_or_zero(values, exponent):
    """Vectorized SafePowOrZero (only used when AccumulationGamma != 1)."""
    if exponent <= 0.0 or not np.isfinite(exponent):
        return np.zeros_like(values)
    is_int_exp = abs(exponent - round(exponent)) < 1e-4     # KINDA_SMALL_NUMBER
    with np.errstate(invalid='ignore'):
        out = np.power(values, np.float32(exponent))
    bad = ~np.isfinite(values) | ~np.isfinite(out)
    if not is_int_exp:
        bad |= (values < 0.0)
    out[bad] = 0.0
    return out.astype(np.float32)

class Symmetric5InversePlan:
    """FSymmetric5InversePlan: LU factorization of the symmetric pentadiagonal
    prefilter matrix with reflect ([-i-1 / 2N-i-1]) boundary folding."""

    def __init__(self, N):
        self.N = N
        if N <= 4:
            # bUseDense path: build reflected dense matrix, keep it for solve
            A = np.zeros((N, N), np.float32)
            for row in range(N):
                A[row, row] += E0
                A[row, self._reflect(row - 1, N)] += E1
                A[row, self._reflect(row + 1, N)] += E1
                A[row, self._reflect(row - 2, N)] += E2
                A[row, self._reflect(row + 2, N)] += E2
            self.dense = A
            return
        self.dense = None
        Dm2 = np.zeros(N, np.float32); Dm1 = np.zeros(N, np.float32)
        D0v = np.zeros(N, np.float32); Dp1 = np.zeros(N, np.float32)
        Dp2 = np.zeros(N, np.float32)
        for row in range(N):
            for col, val in ((row, E0),
                             (self._reflect(row - 1, N), E1),
                             (self._reflect(row + 1, N), E1),
                             (self._reflect(row - 2, N), E2),
                             (self._reflect(row + 2, N), E2)):
                d = col - row
                if   d == -2: Dm2[row] += val
                elif d == -1: Dm1[row] += val
                elif d ==  0: D0v[row] += val
                elif d ==  1: Dp1[row] += val
                elif d ==  2: Dp2[row] += val
                else: raise AssertionError("stencil outside band")
        FM1 = np.zeros(N, np.float32); FM2 = np.zeros(N, np.float32)
        D0Inv = np.zeros(N, np.float32)
        for K in range(N):
            pivot = D0v[K]
            assert abs(pivot) >= 1e-6
            inv = np.float32(1.0) / pivot
            D0Inv[K] = inv
            if K + 1 < N and Dm1[K + 1] != 0.0:
                F = np.float32(Dm1[K + 1] * inv)
                FM1[K + 1] = F; Dm1[K + 1] = 0.0
                D0v[K + 1] = np.float32(D0v[K + 1] - F * Dp1[K])
                Dp1[K + 1] = np.float32(Dp1[K + 1] - F * Dp2[K])
            if K + 2 < N and Dm2[K + 2] != 0.0:
                F = np.float32(Dm2[K + 2] * inv)
                FM2[K + 2] = F; Dm2[K + 2] = 0.0
                Dm1[K + 2] = np.float32(Dm1[K + 2] - F * Dp1[K])
                D0v[K + 2] = np.float32(D0v[K + 2] - F * Dp2[K])
        self.FM1, self.FM2 = FM1, FM2
        self.D0Inv, self.Dp1, self.Dp2 = D0Inv, Dp1, Dp2

    @staticmethod
    def _reflect(i, n):
        while i < 0 or i >= n:
            i = (-i - 1) if i < 0 else (2 * n - i - 1)
        return i

    def solve_along(self, X, axis):
        """In-place solve E * out = X along `axis`. X is float32."""
        N = self.N
        if self.dense is not None:
            Xm = np.moveaxis(X, axis, 0)
            flat = Xm.reshape(N, -1).astype(np.float64)
            Xm.reshape(N, -1)[...] = np.linalg.solve(
                self.dense.astype(np.float64), flat).astype(np.float32)
            return
        Xm = np.moveaxis(X, axis, 0)          # view -> in-place ops propagate
        FM1, FM2 = self.FM1, self.FM2
        D0Inv, Dp1, Dp2 = self.D0Inv, self.Dp1, self.Dp2
        for i in range(1, N):                 # forward elimination
            Xm[i] -= FM1[i] * Xm[i - 1]
            if i >= 2:
                Xm[i] -= FM2[i] * Xm[i - 2]
        Xm[N - 1] *= D0Inv[N - 1]             # back substitution
        if N >= 2:
            Xm[N - 2] = (Xm[N - 2] - Dp1[N - 2] * Xm[N - 1]) * D0Inv[N - 2]
        for i in range(N - 3, -1, -1):
            Xm[i] = (Xm[i] - Dp1[i] * Xm[i + 1] - Dp2[i] * Xm[i + 2]) * D0Inv[i]

_plan_cache = {}
def get_or_create_plan(N):
    if N not in _plan_cache:
        _plan_cache[N] = Symmetric5InversePlan(N)
    return _plan_cache[N]

class ImageOverlappedAccumulator:
    """FImageOverlappedAccumulator, specialized for the single full-frame tile
    (1x1 tiles, no overlap => WeightDataX/Y == 1.0, InTileOffset == 0,
    SubRect == full frame), which is what the Blender pipeline produces."""

    def __init__(self, width, height, num_channels=4,
                 accumulation_gamma=1.0, is_npp=False):
        self.W, self.H, self.NC = width, height, num_channels
        self.gamma = accumulation_gamma
        self.is_npp = is_npp
        self.planes = np.zeros((num_channels, height, width), np.float32)
        self.weight = np.zeros((height, width), np.float32)
        self.row_plan = get_or_create_plan(width)    # RowPlan  (along X)
        self.col_plan = get_or_create_plan(height)   # ColPlan  (along Y)
        # SolveWeightVector(ones): identical for every sample -> cache
        wx = np.ones(width, np.float32);  self.row_plan.solve_along(wx, 0)
        wy = np.ones(height, np.float32); self.col_plan.solve_along(wy, 0)
        self.solved_wx, self.solved_wy = wx, wy

    def zero_planes(self):
        self.planes[...] = 0.0
        self.weight[...] = 0.0

    @staticmethod
    def _axis_gather_indices(start_bias, N):
        # ActualDst0/1 = clamp(Start + {0, N}, 0, N); Curr = Dst - Start
        dst0 = min(max(start_bias, 0), N)
        dst1 = min(max(start_bias + N, 0), N)
        curr = np.arange(dst0, dst1) - start_bias
        idx = tuple(np.clip(curr + d, 0, N - 1) for d in (-1, 0, 1))
        return dst0, dst1, idx

    def accumulate_pixel_data(self, channels, subpx_offset_x, subpx_offset_y):
        """AccumulatePixelData for Float32 RGBA input.
        channels: (NC, H, W) float32; subpx offsets in [0, 1]."""
        assert channels.shape == (self.NC, self.H, self.W)
        assert 0.0 <= subpx_offset_x <= 1.0 and 0.0 <= subpx_offset_y <= 1.0
        data = channels.astype(np.float32, copy=True)

        # -- unpack stage: optional log1p, NaN-only RGB zeroing (Inf passes) --
        if self.is_npp:
            for c in range(min(3, self.NC)):
                v = data[c]
                bad = ~np.isfinite(v) | (v <= -1.0)
                with np.errstate(invalid='ignore', divide='ignore'):
                    lg = np.log1p(v)
                lg[bad | ~np.isfinite(lg)] = 0.0
                data[c] = lg
        nan_mask = np.isnan(data[0]) | np.isnan(data[1]) | np.isnan(data[2])
        data[0][nan_mask] = 0.0
        data[1][nan_mask] = 0.0
        data[2][nan_mask] = 0.0

        if self.gamma != 1.0:
            data = safe_pow_or_zero(data, self.gamma)

        # -- PrefilterWeightedInputBatch: RHS = data * Wx * Wy = data (ones) --
        pre = data
        self.row_plan.solve_along(pre, axis=2)   # rows first (along width)
        self.col_plan.solve_along(pre, axis=1)   # then columns (along height)

        # -- AccumulatePrefilteredPlane: 3x3 gather, edge clamp --------------
        bias_x, KX = get_cached_axis_kernel(subpx_offset_x)
        bias_y, KY = get_cached_axis_kernel(subpx_offset_y)
        x0, x1, ix = self._axis_gather_indices(bias_x, self.W)
        y0, y1, iy = self._axis_gather_indices(bias_y, self.H)
        if x1 <= x0 or y1 <= y0:
            return
        for c in range(self.NC):
            P = pre[c]
            acc = np.zeros((y1 - y0, x1 - x0), np.float32)
            for a in range(3):
                rows = P[iy[a]]
                acc += KY[a] * (KX[0] * rows[:, ix[0]]
                              + KX[1] * rows[:, ix[1]]
                              + KX[2] * rows[:, ix[2]])
            self.planes[c, y0:y1, x0:x1] += acc

        # -- AccumulateSolvedWeightPlane: rank-1 solved-weight gather --------
        row_w = (KY[0] * self.solved_wy[iy[0]]
               + KY[1] * self.solved_wy[iy[1]]
               + KY[2] * self.solved_wy[iy[2]])
        col_w = (KX[0] * self.solved_wx[ix[0]]
               + KX[1] * self.solved_wx[ix[1]]
               + KX[2] * self.solved_wx[ix[2]])
        self.weight[y0:y1, x0:x1] += np.outer(row_w, col_w).astype(np.float32)

    def fetch_full_image(self):
        """FetchFullImageValue for every pixel -> (H, W, NC) float32."""
        scale = np.float32(1.0) / np.maximum(self.weight, np.float32(1e-4))
        out = (self.planes * scale[None, :, :]).astype(np.float32)
        if self.gamma != 1.0 and self.gamma > 0.0:
            out = safe_pow_or_zero(out, 1.0 / self.gamma)
        return np.moveaxis(out, 0, -1)

# ============================================================================
#  SECTION 3 -- Blender driver (4.x / 5.0 compatible)
# ============================================================================

def _get_compositor_tree(scene):
    if hasattr(scene, "compositing_node_group"):         # Blender 5.0+
        nt = scene.compositing_node_group
        if nt is None:
            nt = bpy.data.node_groups.new("MRQ_Compositor",
                                          'CompositorNodeTree')
            scene.compositing_node_group = nt
        return nt
    scene.use_nodes = True                               # Blender <= 4.x
    return scene.node_tree

def _config_exr_float32(fo):
    """OPEN_EXR / RGBA / float32 / no codec for temp files, on whatever level
    this Blender version exposes the format."""
    try:
        fo.format.file_format = 'OPEN_EXR'               # <= 4.x: node level
        fmt = fo.format
    except TypeError:
        # 5.0+: node-level format is multilayer-only; use per-item format
        it = fo.file_output_items[0]
        if hasattr(it, 'override_node_format'):
            it.override_node_format = True
        elif hasattr(it, 'use_node_format'):
            it.use_node_format = False
        it.format.file_format = 'OPEN_EXR'
        fmt = it.format
    fmt.color_mode = 'RGBA'
    fmt.color_depth = '32'
    fmt.exr_codec = 'NONE'

def _ensure_output_item(fo, name):
    """Blender 5.0+: File Output nodes are created with ZERO inputs;
    an item must be added explicitly. The .new() signature isn't stable
    across builds, so probe the plausible ones."""
    items = fo.file_output_items
    if len(items):
        items[0].name = name
        return items[0]
    last_err = None
    for args in ((name,), ('RGBA', name), ('COLOR', name),
                 ('NodeSocketColor', name)):
        try:
            items.new(*args)
            items[0].name = name
            return items[0]
        except TypeError as e:
            last_err = e
    # all guesses failed -> dump the real signature and bail
    try:
        params = items.bl_rna.functions['new'].parameters
        sig = [(p.identifier, p.rna_type.identifier) for p in params]
    except Exception:
        sig = "unavailable"
    raise RuntimeError(
        f"file_output_items.new() signature mismatch, last error: {last_err}; "
        f"actual parameters: {sig}")

def _new_file_output(nt, rl, pass_socket, prefix0):
    """One File Output node per pass (5.0 removed multi-slot single-file
    outputs from a single node in the way 4.x did them)."""
    fo = nt.nodes.new('CompositorNodeOutputFile')
    if hasattr(fo, 'base_path'):                         # Blender <= 4.x
        fo.base_path = TMP_DIR
        fo.file_slots[0].path = prefix0
    else:                                                # Blender 5.0+
        fo.directory = TMP_DIR
        fo.file_name = ""                # item 名本身就会进文件名，见下
        _ensure_output_item(fo, prefix0)
    _config_exr_float32(fo)
    nt.links.new(rl.outputs[pass_socket], fo.inputs[0])
    return fo

def _set_prefix(fo, prefix):
    if hasattr(fo, 'base_path'):
        fo.file_slots[0].path = prefix
    else:
        fo.file_output_items[0].name = prefix

def _find_output(prefix):
    """Locate the temp EXR by prefix; 5.0's exact filename decoration
    (frame padding, item-name insertion) is not assumed."""
    pat = os.path.join(TMP_DIR, glob.escape(prefix) + "*.exr")
    files = glob.glob(pat)
    assert len(files) == 1, f"expected exactly 1 file for {pat}, got {files}"
    return files[0]

def setup_scene(scene):
    scene.render.engine = 'CYCLES'
    scene.render.use_persistent_data = True      # critical: no BVH rebuild x64
    scene.render.use_motion_blur = False         # Vector pass requirement
    scene.render.dither_intensity = 0.0
    scene.render.use_compositing = True
    vl = scene.view_layers[0]
    vl.use_pass_z = True
    vl.use_pass_vector = True
    cy = scene.cycles
    cy.pixel_filter_type = 'BOX'
    cy.filter_width = 0.01                       # ~= point sampling at center
    cy.use_denoising = False

    nt = _get_compositor_tree(scene)
    nt.nodes.clear()
    rl = nt.nodes.new('CompositorNodeRLayers')
    fos = {
        'rgba':   _new_file_output(nt, rl, 'Image',  'rgba_'),
        'vector': _new_file_output(nt, rl, 'Vector', 'vector_'),
        'depth':  _new_file_output(nt, rl, 'Depth',  'depth_'),
    }
    return fos

def load_exr_topdown(path, W, H):
    """Load EXR via bpy and flip to top-down (UE image space, +Y down)."""
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = 'Non-Color'
    buf = np.empty(W * H * 4, np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    return buf.reshape(H, W, 4)[::-1].copy()     # bottom-up -> top-down

def save_exr_half_zip(scene, path, arr_topdown):
    """float16 + zlib(ZIP) OpenEXR. Input is top-down (H, W, 3|4)."""
    H, W = arr_topdown.shape[:2]
    a = np.ones((H, W, 4), np.float32)
    a[..., :arr_topdown.shape[2]] = arr_topdown
    a = a[::-1]                                   # back to bottom-up for bpy
    img = bpy.data.images.new('_tmp_save', W, H, alpha=True,
                              float_buffer=True, is_data=True)
    img.pixels.foreach_set(np.ascontiguousarray(a).ravel())
    ist = scene.render.image_settings
    old = (ist.file_format, ist.color_mode, ist.color_depth, ist.exr_codec)
    ist.file_format, ist.color_mode = 'OPEN_EXR', 'RGB'
    ist.color_depth, ist.exr_codec = '16', 'ZIP'  # float16 + zlib
    img.save_render(path, scene=scene)
    (ist.file_format, ist.color_mode, ist.color_depth, ist.exr_codec) = old
    bpy.data.images.remove(img)

def render_all():
    scene = bpy.context.scene
    fos = setup_scene(scene)
    cam = scene.camera
    base_sx, base_sy = cam.data.shift_x, cam.data.shift_y
    pct = scene.render.resolution_percentage / 100.0
    W = int(scene.render.resolution_x * pct)
    H = int(scene.render.resolution_y * pct)
    m = max(W, H)                                # Blender shift unit = long edge
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    accum = ImageOverlappedAccumulator(W, H, 4, ACCUMULATION_GAMMA, IS_NPP)

    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        jitters = get_frame_jitters(frame)
        chosen  = choose_subsample(frame, jitters)
        accum.zero_planes()
        picked = {}

        for si, (jx, jy) in enumerate(jitters):
            # SpatialShift (px, +Y down) -> camera shift (long-edge units)
            cam.data.shift_x = base_sx + SHIFT_SIGN_X * float(jx) / m
            cam.data.shift_y = base_sy + SHIFT_SIGN_Y * float(jy) / m
            scene.cycles.seed = frame * SPATIAL_SAMPLE_COUNT + si
            for name, node in fos.items():
                _set_prefix(node, f"{name}_s{si:03d}_")
            bpy.ops.render.render(write_still=False)

            paths = {name: _find_output(f"{name}_s{si:03d}_")
                     for name in ('rgba', 'vector', 'depth')}
            rgba = load_exr_topdown(paths['rgba'], W, H)

            # OverlappedSubpixelShift = 0.5 - SpatialShift (utils line ~1546)
            accum.accumulate_pixel_data(
                np.moveaxis(rgba, -1, 0),
                float(np.float32(0.5) - jx),
                float(np.float32(0.5) - jy))

            if si == chosen:
                picked['rgba']  = rgba
                picked['vec']   = load_exr_topdown(paths['vector'], W, H)
                picked['depth'] = load_exr_topdown(paths['depth'],  W, H)
            for p in paths.values():
                os.remove(p)

        cam.data.shift_x, cam.data.shift_y = base_sx, base_sy

        final = accum.fetch_full_image()
        save_exr_half_zip(scene, os.path.join(OUT_DIR, f"final.{frame:04d}.exr"),
                          final)
        save_exr_half_zip(scene, os.path.join(OUT_DIR, f"subsample.{frame:04d}.exr"),
                          picked['rgba'])

        mvd = np.empty((H, W, 3), np.float32)
        mvd[..., 0] = MV_SIGN_X * picked['vec'][..., MV_PREV_CHANNELS[0]]
        mvd[..., 1] = MV_SIGN_Y * picked['vec'][..., MV_PREV_CHANNELS[1]]
        mvd[..., 2] = picked['depth'][..., 0]
        save_exr_half_zip(scene, os.path.join(OUT_DIR, f"mvdepth.{frame:04d}.exr"),
                          mvd)
        print(f"frame {frame}: done, chosen subsample = {chosen}, "
              f"jitter = ({float(jitters[chosen][0]):+.5f}, "
              f"{float(jitters[chosen][1]):+.5f})")

render_all()

# ============================================================================
#  VERIFICATION NOTES
# ============================================================================
#  1. Shift signs: derived from the confirmed UE code
#     (DefaultJitter = (jx*2/W, jy*-2/H)); a checkerboard smoke test is still
#     worthwhile after Blender version bumps: force jitter (+0.5, 0) then
#     (0, +0.5) vs zero -- content must move +x right / +y down (top-down).
#  2. Motion vectors: render 2 frames of an object moving +x at known px/frame
#     and inspect which Vector channels/sign encode "towards previous frame";
#     set MV_PREV_CHANNELS / MV_SIGN_X/Y. Divide by (W, H) if your UE pipeline
#     stores MV in UV space rather than pixels.
#  3. Depth: Cycles Z is ray distance (euclidean); UE SceneDepth is planar.
#     If you need planar depth, multiply per-pixel by cos(theta) derived from
#     the camera intrinsics before writing channel B.
#  4. Jitter mixing: get_frame_jitters()/choose_subsample() encode the
#     inferred policy (see CHOSEN_JITTER_MODE); confirm against your
#     MoviePipelineRendering call sites.
#  5. Known quirk reproduced on purpose: GetCachedAxisKernel drops the
#     phi(2-t) tap when offset < 0.5 (kernel mass up to ~0.169 missing,
#     discontinuity at offset = 0.5). The weight plane compensates for DC.
#     If the UE side ever fixes this, update get_cached_axis_kernel too.
# ============================================================================