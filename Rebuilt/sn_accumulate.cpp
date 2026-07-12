// sn_accumulate.cpp
// Offline Sacht-Nehab accumulator for Blender subsample EXRs.
//
// This is a standalone C++17/OpenEXR implementation of the core accumulator logic
// in the supplied UE MovieRenderOverlappedImage.cpp, adapted to full-frame Blender
// subsample files named like:
//   frame_2404/sample_0000_jx_-0.50000000_jy_+0.17351723.exr
//
// Scope:
// - Accumulates color/beauty sample_*.exr files only.
// - Ignores *_depth.exr, *_vector.exr, *_mvdepth.exr.
// - Full-frame, single-tile weights are assumed (WeightX = WeightY = 1).
// - Writes final_####.exr as half-float ZIP RGBA.
//
// Build example with CMakeLists.txt included next to this file.

#include <ImfRgbaFile.h>
#include <ImfArray.h>
#include <ImfHeader.h>
#include <ImathBox.h>
#include <ImathVec.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;
namespace ImfNs = OPENEXR_IMF_NAMESPACE;
namespace ImathNs = IMATH_NAMESPACE;

static constexpr float B00 = 0.75627421f;
static constexpr float B01 = 0.11798097f;
static constexpr float B11 = 0.11798097f;
static constexpr float C00 = 0.01588197f;
static constexpr float C10 = -0.02400002f;
static constexpr float C20 = 0.01588197f;

static constexpr float E0 = 0.65314970f;
static constexpr float E1 = 0.17889730f;
static constexpr float E2 = -0.00547216f;
static constexpr int LUT_SIZE = 1024;

struct ImageRGBA {
    int width = 0;
    int height = 0;
    std::vector<float> rgba; // interleaved RGBA, size width*height*4
};

struct SampleInfo {
    fs::path path;
    int index = 0;
    float jx_centered = 0.0f;
    float jy_centered = 0.0f;
    float subpixel_x() const { return jx_centered + 0.5f; }
    float subpixel_y() const { return jy_centered + 0.5f; }
};

struct AxisKernel {
    int start_bias = 0;
    float w[3] = {0.0f, 0.0f, 0.0f};
};

static inline int reflect_index(int idx, int count) {
    if (count <= 0) throw std::runtime_error("reflect_index count <= 0");
    while (idx < 0 || idx >= count) {
        idx = (idx < 0) ? (-idx - 1) : (2 * count - idx - 1);
    }
    return idx;
}

static inline float beta0_nc(float x) {
    return (x >= 0.0f && x < 1.0f) ? 1.0f : 0.0f;
}

static inline float beta1_nc(float x) {
    if (x >= 0.0f && x < 1.0f) return x;
    if (x >= 1.0f && x < 2.0f) return 2.0f - x;
    return 0.0f;
}

static inline float beta2_nc(float x) {
    if (x >= 0.0f && x < 1.0f) return 0.5f * x * x;
    if (x >= 1.0f && x < 2.0f) return -x * x + 3.0f * x - 1.5f;
    if (x >= 2.0f && x < 3.0f) {
        float s = 3.0f - x;
        return 0.5f * s * s;
    }
    return 0.0f;
}

static inline float eval_generator(float x) {
    float t = x + 1.5f;
    return B00 * beta2_nc(t)
        + B01 * beta1_nc(t)
        + B11 * beta1_nc(t - 1.0f)
        + C00 * beta0_nc(t)
        + C10 * beta0_nc(t - 1.0f)
        + C20 * beta0_nc(t - 2.0f);
}

struct WeightLUT {
    float weights[LUT_SIZE + 1][3];
    WeightLUT() {
        for (int i = 0; i <= LUT_SIZE; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(LUT_SIZE);
            weights[i][0] = eval_generator(1.0f - t);
            weights[i][1] = eval_generator(-t);
            weights[i][2] = eval_generator(-1.0f - t);
        }
    }
};

static const WeightLUT& get_weight_lut() {
    static WeightLUT lut;
    return lut;
}

static AxisKernel get_axis_kernel(float subpixel_offset) {
    if (!(subpixel_offset >= 0.0f && subpixel_offset <= 1.0f)) {
        std::ostringstream oss;
        oss << "subpixel offset out of range [0,1]: " << subpixel_offset;
        throw std::runtime_error(oss.str());
    }
    AxisKernel k;
    k.start_bias = (subpixel_offset >= 0.5f) ? 0 : -1;
    float t = subpixel_offset + 0.5f;
    t -= std::floor(t);
    int bin = static_cast<int>(std::lround(t * static_cast<float>(LUT_SIZE)));
    bin = std::max(0, std::min(LUT_SIZE, bin));
    const auto& lut = get_weight_lut();
    k.w[0] = lut.weights[bin][0];
    k.w[1] = lut.weights[bin][1];
    k.w[2] = lut.weights[bin][2];
    return k;
}

class Symmetric5InversePlan {
public:
    explicit Symmetric5InversePlan(int n) : N(n), use_dense(n <= 4) {
        if (N <= 0) throw std::runtime_error("Symmetric5InversePlan N <= 0");
        if (use_dense) init_dense(); else init_banded();
    }

    int size() const { return N; }

    void solve_one(const float* rhs, float* out) const {
        std::copy(rhs, rhs + N, out);
        if (use_dense) {
            for (int k = 0; k < N; ++k) {
                for (int i = k + 1; i < N; ++i) {
                    float factor = dense[i * N + k];
                    if (factor != 0.0f) out[i] -= factor * out[k];
                }
            }
            for (int i = N - 1; i >= 0; --i) {
                float pivot = dense[i * N + i];
                if (std::abs(pivot) < 1e-6f) throw std::runtime_error("dense backsolve singular");
                float sum = out[i];
                for (int j = i + 1; j < N; ++j) sum -= dense[i * N + j] * out[j];
                out[i] = sum / pivot;
            }
            return;
        }

        for (int i = 1; i < N; ++i) {
            out[i] -= forward_mul1[i] * out[i - 1];
            if (i >= 2) out[i] -= forward_mul2[i] * out[i - 2];
        }
        out[N - 1] *= d0_inv[N - 1];
        if (N >= 2) out[N - 2] = (out[N - 2] - dp1[N - 2] * out[N - 1]) * d0_inv[N - 2];
        for (int i = N - 3; i >= 0; --i) {
            out[i] = (out[i] - dp1[i] * out[i + 1] - dp2[i] * out[i + 2]) * d0_inv[i];
        }
    }

private:
    int N = 0;
    bool use_dense = false;
    std::vector<float> dense;
    std::vector<float> d0_inv, dp1, dp2, forward_mul1, forward_mul2;

    void init_dense() {
        dense.assign(static_cast<size_t>(N) * N, 0.0f);
        auto add = [&](int r, int c, float v) { dense[static_cast<size_t>(r) * N + c] += v; };
        for (int row = 0; row < N; ++row) {
            add(row, row, E0);
            add(row, reflect_index(row - 1, N), E1);
            add(row, reflect_index(row + 1, N), E1);
            add(row, reflect_index(row - 2, N), E2);
            add(row, reflect_index(row + 2, N), E2);
        }
        for (int k = 0; k < N; ++k) {
            float pivot = dense[static_cast<size_t>(k) * N + k];
            if (std::abs(pivot) < 1e-6f) throw std::runtime_error("dense plan singular");
            float inv_pivot = 1.0f / pivot;
            for (int i = k + 1; i < N; ++i) {
                float factor = dense[static_cast<size_t>(i) * N + k] * inv_pivot;
                if (factor == 0.0f) continue;
                dense[static_cast<size_t>(i) * N + k] = factor;
                for (int j = k + 1; j < N; ++j) {
                    dense[static_cast<size_t>(i) * N + j] -= factor * dense[static_cast<size_t>(k) * N + j];
                }
            }
        }
    }

    void init_banded() {
        std::vector<float> dm2(N, 0.0f), dm1(N, 0.0f), d0(N, 0.0f);
        dp1.assign(N, 0.0f);
        dp2.assign(N, 0.0f);
        forward_mul1.assign(N, 0.0f);
        forward_mul2.assign(N, 0.0f);
        d0_inv.assign(N, 0.0f);

        auto add = [&](int row, int col, float value) {
            int delta = col - row;
            switch (delta) {
                case -2: dm2[row] += value; break;
                case -1: dm1[row] += value; break;
                case 0: d0[row] += value; break;
                case 1: dp1[row] += value; break;
                case 2: dp2[row] += value; break;
                default: throw std::runtime_error("unexpected band delta");
            }
        };

        for (int row = 0; row < N; ++row) {
            add(row, row, E0);
            add(row, reflect_index(row - 1, N), E1);
            add(row, reflect_index(row + 1, N), E1);
            add(row, reflect_index(row - 2, N), E2);
            add(row, reflect_index(row + 2, N), E2);
        }

        for (int k = 0; k < N; ++k) {
            float pivot = d0[k];
            if (std::abs(pivot) < 1e-6f) throw std::runtime_error("banded plan singular");
            float inv_pivot = 1.0f / pivot;
            d0_inv[k] = inv_pivot;

            if (k + 1 < N && dm1[k + 1] != 0.0f) {
                float factor = dm1[k + 1] * inv_pivot;
                forward_mul1[k + 1] = factor;
                dm1[k + 1] = 0.0f;
                d0[k + 1] -= factor * dp1[k];
                dp1[k + 1] -= factor * dp2[k];
            }
            if (k + 2 < N && dm2[k + 2] != 0.0f) {
                float factor = dm2[k + 2] * inv_pivot;
                forward_mul2[k + 2] = factor;
                dm2[k + 2] = 0.0f;
                dm1[k + 2] -= factor * dp1[k];
                d0[k + 2] -= factor * dp2[k];
            }
        }
    }
};

static const Symmetric5InversePlan& get_plan(int n) {
    static std::unordered_map<int, std::unique_ptr<Symmetric5InversePlan>> cache;
    auto it = cache.find(n);
    if (it != cache.end()) return *it->second;
    auto plan = std::make_unique<Symmetric5InversePlan>(n);
    const Symmetric5InversePlan& ref = *plan;
    cache.emplace(n, std::move(plan));
    return ref;
}

static ImageRGBA read_rgba_exr(const fs::path& path) {
    ImfNs::RgbaInputFile file(path.string().c_str());
    ImathNs::Box2i dw = file.dataWindow();
    int w = dw.max.x - dw.min.x + 1;
    int h = dw.max.y - dw.min.y + 1;
    std::vector<ImfNs::Rgba> px(static_cast<size_t>(w) * h);
    file.setFrameBuffer(px.data() - dw.min.x - static_cast<ptrdiff_t>(dw.min.y) * w, 1, w);
    file.readPixels(dw.min.y, dw.max.y);

    ImageRGBA img;
    img.width = w;
    img.height = h;
    img.rgba.resize(static_cast<size_t>(w) * h * 4);
    for (size_t i = 0; i < px.size(); ++i) {
        img.rgba[i * 4 + 0] = static_cast<float>(px[i].r);
        img.rgba[i * 4 + 1] = static_cast<float>(px[i].g);
        img.rgba[i * 4 + 2] = static_cast<float>(px[i].b);
        img.rgba[i * 4 + 3] = static_cast<float>(px[i].a);
    }
    return img;
}

static void write_rgba_exr(const fs::path& path, int w, int h, const std::vector<float>& rgba) {
    fs::create_directories(path.parent_path());
    std::vector<ImfNs::Rgba> px(static_cast<size_t>(w) * h);
    for (size_t i = 0; i < px.size(); ++i) {
        px[i].r = rgba[i * 4 + 0];
        px[i].g = rgba[i * 4 + 1];
        px[i].b = rgba[i * 4 + 2];
        px[i].a = rgba[i * 4 + 3];
    }
    ImfNs::RgbaOutputFile out(
        path.string().c_str(),
        w,
        h,
        ImfNs::WRITE_RGBA,
        1.0f,
        ImathNs::V2f(0.0f, 0.0f),
        1.0f,
        ImfNs::INCREASING_Y,
        ImfNs::ZIP_COMPRESSION);
    out.setFrameBuffer(px.data(), 1, w);
    out.writePixels(h);
}

static std::vector<float> solve_weight_vector(const std::vector<float>& weights) {
    std::vector<float> solved(weights.size());
    get_plan(static_cast<int>(weights.size())).solve_one(weights.data(), solved.data());
    return solved;
}

// channels[ch][y*w+x] -> coeffs[ch][y*w+x]
static void prefilter_weighted_input_batch(
    const std::vector<std::vector<float>>& channels,
    int w,
    int h,
    const std::vector<float>& weight_x,
    const std::vector<float>& weight_y,
    std::vector<std::vector<float>>& coeffs) {

    const int num_channels = static_cast<int>(channels.size());
    const size_t pixel_count = static_cast<size_t>(w) * h;
    std::vector<std::vector<float>> temp(num_channels, std::vector<float>(pixel_count));
    coeffs.assign(num_channels, std::vector<float>(pixel_count));

    const auto& row_plan = get_plan(w);
    const auto& col_plan = get_plan(h);

    #pragma omp parallel for if(h >= 64) schedule(static)
    for (int y = 0; y < h; ++y) {
        std::vector<float> rhs(w), sol(w);
        for (int ch = 0; ch < num_channels; ++ch) {
            const float wy = weight_y[y];
            const float* src = channels[ch].data() + static_cast<size_t>(y) * w;
            for (int x = 0; x < w; ++x) rhs[x] = src[x] * weight_x[x] * wy;
            row_plan.solve_one(rhs.data(), sol.data());
            std::copy(sol.begin(), sol.end(), temp[ch].begin() + static_cast<size_t>(y) * w);
        }
    }

    #pragma omp parallel for if(w >= 64) schedule(static)
    for (int x = 0; x < w; ++x) {
        std::vector<float> rhs(h), sol(h);
        for (int ch = 0; ch < num_channels; ++ch) {
            for (int y = 0; y < h; ++y) rhs[y] = temp[ch][static_cast<size_t>(y) * w + x];
            col_plan.solve_one(rhs.data(), sol.data());
            for (int y = 0; y < h; ++y) coeffs[ch][static_cast<size_t>(y) * w + x] = sol[y];
        }
    }
}

static void accumulate_prefiltered_plane(
    std::vector<float>& dst,
    const std::vector<float>& src,
    int w,
    int h,
    float subpixel_x,
    float subpixel_y) {

    AxisKernel kx = get_axis_kernel(subpixel_x);
    AxisKernel ky = get_axis_kernel(subpixel_y);
    int start_x = kx.start_bias;
    int start_y = ky.start_bias;

    int x0d = std::clamp(start_x, 0, w);
    int x1d = std::clamp(start_x + w, 0, w);
    int y0d = std::clamp(start_y, 0, h);
    int y1d = std::clamp(start_y + h, 0, h);
    if (x1d <= x0d || y1d <= y0d) return;

    #pragma omp parallel for if((y1d-y0d) >= 64) schedule(static)
    for (int dst_y = y0d; dst_y < y1d; ++dst_y) {
        int curr_y = dst_y - start_y;
        int sy0 = std::clamp(curr_y - 1, 0, h - 1);
        int sy1 = std::clamp(curr_y, 0, h - 1);
        int sy2 = std::clamp(curr_y + 1, 0, h - 1);
        const float* row0 = src.data() + static_cast<size_t>(sy0) * w;
        const float* row1 = src.data() + static_cast<size_t>(sy1) * w;
        const float* row2 = src.data() + static_cast<size_t>(sy2) * w;
        float* out = dst.data() + static_cast<size_t>(dst_y) * w;

        for (int dst_x = x0d; dst_x < x1d; ++dst_x) {
            int curr_x = dst_x - start_x;
            int sx0 = std::clamp(curr_x - 1, 0, w - 1);
            int sx1 = std::clamp(curr_x, 0, w - 1);
            int sx2 = std::clamp(curr_x + 1, 0, w - 1);
            out[dst_x] += ky.w[0] * (kx.w[0] * row0[sx0] + kx.w[1] * row0[sx1] + kx.w[2] * row0[sx2])
                        + ky.w[1] * (kx.w[0] * row1[sx0] + kx.w[1] * row1[sx1] + kx.w[2] * row1[sx2])
                        + ky.w[2] * (kx.w[0] * row2[sx0] + kx.w[1] * row2[sx1] + kx.w[2] * row2[sx2]);
        }
    }
}

static void accumulate_solved_weight_plane(
    std::vector<float>& dst,
    int w,
    int h,
    const std::vector<float>& solved_x,
    const std::vector<float>& solved_y,
    float subpixel_x,
    float subpixel_y) {

    AxisKernel kx = get_axis_kernel(subpixel_x);
    AxisKernel ky = get_axis_kernel(subpixel_y);
    int start_x = kx.start_bias;
    int start_y = ky.start_bias;

    int x0d = std::clamp(start_x, 0, w);
    int x1d = std::clamp(start_x + w, 0, w);
    int y0d = std::clamp(start_y, 0, h);
    int y1d = std::clamp(start_y + h, 0, h);
    if (x1d <= x0d || y1d <= y0d) return;

    #pragma omp parallel for if((y1d-y0d) >= 64) schedule(static)
    for (int dst_y = y0d; dst_y < y1d; ++dst_y) {
        int curr_y = dst_y - start_y;
        int sy0 = std::clamp(curr_y - 1, 0, h - 1);
        int sy1 = std::clamp(curr_y, 0, h - 1);
        int sy2 = std::clamp(curr_y + 1, 0, h - 1);
        float row_weight = ky.w[0] * solved_y[sy0] + ky.w[1] * solved_y[sy1] + ky.w[2] * solved_y[sy2];
        float* out = dst.data() + static_cast<size_t>(dst_y) * w;
        for (int dst_x = x0d; dst_x < x1d; ++dst_x) {
            int curr_x = dst_x - start_x;
            int sx0 = std::clamp(curr_x - 1, 0, w - 1);
            int sx1 = std::clamp(curr_x, 0, w - 1);
            int sx2 = std::clamp(curr_x + 1, 0, w - 1);
            float col_weight = kx.w[0] * solved_x[sx0] + kx.w[1] * solved_x[sx1] + kx.w[2] * solved_x[sx2];
            out[dst_x] += row_weight * col_weight;
        }
    }
}

static std::optional<SampleInfo> parse_sample_path(const fs::path& p) {
    static const std::regex re(R"(^sample_(\d+)_jx_([+-]?\d+(?:\.\d+)?)_jy_([+-]?\d+(?:\.\d+)?)\.exr$)");
    const std::string name = p.filename().string();
    if (name.find("_depth.exr") != std::string::npos ||
        name.find("_vector.exr") != std::string::npos ||
        name.find("_mvdepth.exr") != std::string::npos) {
        return std::nullopt;
    }
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    SampleInfo s;
    s.path = p;
    s.index = std::stoi(m[1].str());
    s.jx_centered = std::stof(m[2].str());
    s.jy_centered = std::stof(m[3].str());
    return s;
}

static std::vector<SampleInfo> collect_samples(const fs::path& frame_dir) {
    std::vector<SampleInfo> samples;
    if (!fs::exists(frame_dir)) return samples;
    for (const auto& e : fs::directory_iterator(frame_dir)) {
        if (!e.is_regular_file()) continue;
        if (e.path().extension() != ".exr") continue;
        auto s = parse_sample_path(e.path());
        if (s) samples.push_back(*s);
    }
    std::sort(samples.begin(), samples.end(), [](const SampleInfo& a, const SampleInfo& b) {
        return a.index < b.index;
    });
    return samples;
}

static int parse_frame_number(const fs::path& frame_dir) {
    const std::string name = frame_dir.filename().string();
    const std::string prefix = "frame_";
    if (name.rfind(prefix, 0) != 0) throw std::runtime_error("invalid frame dir name: " + name);
    return std::stoi(name.substr(prefix.size()));
}

struct Args {
    fs::path root;
    fs::path out_dir;
    int start = -1;
    int end = -1;
    int samples_expected = 0;
    bool debug = false;
};

static std::string zero_pad4(int n) {
    std::ostringstream oss;
    oss << std::setw(4) << std::setfill('0') << n;
    return oss.str();
}

static void usage() {
    std::cout << "Usage:\n"
              << "  sn_accumulate --root <mrq_out> [--out <final_dir>] [--start N --end M] [--samples 64] [--debug]\n\n"
              << "Example:\n"
              << "  sn_accumulate --root E:/BlenderScenes/mrq_out_v18 --out E:/BlenderScenes/final --start 2404 --end 2475 --samples 64\n";
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (k == "--root") a.root = need("--root");
        else if (k == "--out") a.out_dir = need("--out");
        else if (k == "--start") a.start = std::stoi(need("--start"));
        else if (k == "--end") a.end = std::stoi(need("--end"));
        else if (k == "--samples") a.samples_expected = std::stoi(need("--samples"));
        else if (k == "--debug") a.debug = true;
        else if (k == "--help" || k == "-h") { usage(); std::exit(0); }
        else throw std::runtime_error("unknown argument: " + k);
    }
    if (a.root.empty()) throw std::runtime_error("--root is required");
    if (a.out_dir.empty()) a.out_dir = a.root / "_accum_cpp";
    if (a.start >= 0 && a.end < a.start) throw std::runtime_error("--end must be >= --start");
    return a;
}

static std::vector<fs::path> find_frame_dirs(const Args& args) {
    std::vector<fs::path> dirs;
    if (!fs::exists(args.root)) throw std::runtime_error("root does not exist: " + args.root.string());
    for (const auto& e : fs::directory_iterator(args.root)) {
        if (!e.is_directory()) continue;
        const std::string name = e.path().filename().string();
        if (name.rfind("frame_", 0) != 0) continue;
        int n = parse_frame_number(e.path());
        if (args.start >= 0 && n < args.start) continue;
        if (args.end >= 0 && n > args.end) continue;
        dirs.push_back(e.path());
    }
    std::sort(dirs.begin(), dirs.end(), [](const fs::path& a, const fs::path& b) {
        return parse_frame_number(a) < parse_frame_number(b);
    });
    return dirs;
}

static void reconstruct_frame(const fs::path& frame_dir, const fs::path& out_path, int expected_samples, bool debug) {
    const int frame_number = parse_frame_number(frame_dir);
    auto samples = collect_samples(frame_dir);
    if (samples.empty()) throw std::runtime_error("no color samples in " + frame_dir.string());
    if (expected_samples > 0 && static_cast<int>(samples.size()) != expected_samples) {
        std::cerr << "[sn_accumulate] warning frame=" << frame_number << " expected samples=" << expected_samples
                  << " found=" << samples.size() << "\n";
    }

    ImageRGBA first = read_rgba_exr(samples[0].path);
    const int w = first.width;
    const int h = first.height;
    const size_t pixel_count = static_cast<size_t>(w) * h;

    std::vector<float> weight_x(w, 1.0f), weight_y(h, 1.0f);
    std::vector<float> solved_x = solve_weight_vector(weight_x);
    std::vector<float> solved_y = solve_weight_vector(weight_y);

    std::vector<std::vector<float>> accum(4, std::vector<float>(pixel_count, 0.0f));
    std::vector<float> weight_plane(pixel_count, 0.0f);

    for (size_t si = 0; si < samples.size(); ++si) {
        const SampleInfo& s = samples[si];
        ImageRGBA img = (si == 0) ? std::move(first) : read_rgba_exr(s.path);
        if (img.width != w || img.height != h) throw std::runtime_error("sample resolution mismatch: " + s.path.string());

        std::vector<std::vector<float>> channels(4, std::vector<float>(pixel_count));
        #pragma omp parallel for if(pixel_count >= 65536) schedule(static)
        for (int64_t p = 0; p < static_cast<int64_t>(pixel_count); ++p) {
            channels[0][p] = img.rgba[static_cast<size_t>(p) * 4 + 0];
            channels[1][p] = img.rgba[static_cast<size_t>(p) * 4 + 1];
            channels[2][p] = img.rgba[static_cast<size_t>(p) * 4 + 2];
            channels[3][p] = img.rgba[static_cast<size_t>(p) * 4 + 3];
        }

        std::vector<std::vector<float>> coeffs;
        prefilter_weighted_input_batch(channels, w, h, weight_x, weight_y, coeffs);

        const float subx = s.subpixel_x();
        const float suby = s.subpixel_y();
        for (int ch = 0; ch < 4; ++ch) {
            accumulate_prefiltered_plane(accum[ch], coeffs[ch], w, h, subx, suby);
        }
        accumulate_solved_weight_plane(weight_plane, w, h, solved_x, solved_y, subx, suby);

        if (debug) {
            std::cout << "[sn_accumulate] frame=" << frame_number << " sample=" << std::setw(4) << std::setfill('0') << s.index
                      << std::setfill(' ') << " j=(" << s.jx_centered << "," << s.jy_centered << ")"
                      << " sub=(" << subx << "," << suby << ")\n";
        }
    }

    std::vector<float> out(pixel_count * 4, 0.0f);
    #pragma omp parallel for if(pixel_count >= 65536) schedule(static)
    for (int64_t p = 0; p < static_cast<int64_t>(pixel_count); ++p) {
        float scale = 1.0f / std::max(weight_plane[p], 0.0001f);
        out[static_cast<size_t>(p) * 4 + 0] = accum[0][p] * scale;
        out[static_cast<size_t>(p) * 4 + 1] = accum[1][p] * scale;
        out[static_cast<size_t>(p) * 4 + 2] = accum[2][p] * scale;
        out[static_cast<size_t>(p) * 4 + 3] = accum[3][p] * scale;
    }

    write_rgba_exr(out_path, w, h, out);
    std::cout << "[sn_accumulate] wrote " << out_path.string() << " samples=" << samples.size() << " size=" << w << "x" << h << "\n";
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        fs::create_directories(args.out_dir);
        auto frames = find_frame_dirs(args);
        if (frames.empty()) throw std::runtime_error("no frame_* dirs found in " + args.root.string());

        std::cout << "[sn_accumulate] root=" << fs::absolute(args.root).string() << "\n";
        std::cout << "[sn_accumulate] out=" << fs::absolute(args.out_dir).string() << "\n";
        std::cout << "[sn_accumulate] frames=" << frames.size() << "\n";
        #ifdef _OPENMP
        std::cout << "[sn_accumulate] openmp threads=" << omp_get_max_threads() << "\n";
        #endif

        for (const fs::path& frame_dir : frames) {
            int frame = parse_frame_number(frame_dir);
            fs::path out_path = args.out_dir / ("final_" + zero_pad4(frame) + ".exr");
            std::cout << "[sn_accumulate] reconstruct frame=" << frame << "\n";
            reconstruct_frame(frame_dir, out_path, args.samples_expected, args.debug);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[sn_accumulate] ERROR: " << e.what() << "\n";
        return 1;
    }
}
