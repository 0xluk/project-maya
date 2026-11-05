// Decode-only in-box tensor parallel for the GLM layer split - see glm_tp_decode.hpp.
#include "glm_tp_decode.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>

namespace gf = strata::kernels::glmf;

namespace strata::core::glmtp {
namespace {

constexpr int kBlocks = 4;                 // exchange blocks a transfer (one flag each)
constexpr int kThreads = 512;
constexpr int kFlagStride = 16;            // flags 64 B apart
constexpr size_t kInMax = 32 << 10;        // largest broadcast (q8_1 of 16384 values: 18 KB)
constexpr size_t kOutMax = 32 << 10;       // largest helper output (3 x 2048 floats: 24 KB)
constexpr unsigned long long kSpinNs = 3000000000ull;

__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// thread 0 of the block waits for flag == seq (or times out and raises *err)
__device__ __forceinline__ void wait_flag(const unsigned* flag, unsigned seq, unsigned* err) {
    if (threadIdx.x == 0) {
        const volatile unsigned* f = flag;
        const unsigned long long t0 = gtime();
        while (*f != seq) {
            __nanosleep(32);
            if (gtime() - t0 > kSpinNs) {
                *(volatile unsigned*) err = 1u;
                break;
            }
        }
    }
    __syncthreads();
}

// device -> host slot: block b writes its word range and then its flag
__global__ void k_put(const uint32_t* __restrict__ src, int words, uint32_t* host, unsigned* flags, unsigned seq) {
    const int b = blockIdx.x;
    const int w0 = (int) ((long long) words * b / kBlocks), w1 = (int) ((long long) words * (b + 1) / kBlocks);
    volatile uint32_t* h = host;
    for (int i = w0 + (int) threadIdx.x; i < w1; i += blockDim.x) h[i] = src[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile unsigned*) (flags + b * kFlagStride) = seq;
}

// host slot -> device: block b waits for its flag, then reads its word range
__global__ void k_fetch(const uint32_t* host, int words, uint32_t* __restrict__ dst, unsigned* flags, unsigned seq,
                        unsigned* err) {
    const int b = blockIdx.x;
    wait_flag(flags + b * kFlagStride, seq, err);
    const int w0 = (int) ((long long) words * b / kBlocks), w1 = (int) ((long long) words * (b + 1) / kBlocks);
    for (int i = w0 + (int) threadIdx.x; i < w1; i += blockDim.x) dst[i] = __ldcv((const unsigned int*) host + i);
}

struct CollectJobs {
    float* dst[3][kMaxJobs];   // [helper][job]: the owner's y + the helper's first row
    int off[3][kMaxJobs + 1];  // [helper]: word offsets of its jobs in its slot
    uint32_t* host[3];
    unsigned* flags[3];
    int nh, nj;
};

// the owner: block (b, h) waits for helper h's flag b and scatters that word range into the jobs' outputs
__global__ void k_collect(CollectJobs c, unsigned seq, unsigned* err) {
    const int b = blockIdx.x, h = blockIdx.y;
    wait_flag(c.flags[h] + b * kFlagStride, seq, err);
    const int words = c.off[h][c.nj];
    const int w0 = (int) ((long long) words * b / kBlocks), w1 = (int) ((long long) words * (b + 1) / kBlocks);
    for (int i = w0 + (int) threadIdx.x; i < w1; i += blockDim.x) {
        int j = 0;
        while (j + 1 < c.nj && i >= c.off[h][j + 1]) ++j;
        const unsigned v = __ldcv((const unsigned int*) c.host[h] + i);
        reinterpret_cast<uint32_t*>(c.dst[h][j])[i - c.off[h][j]] = v;
    }
}

struct Slice {
    void* w = nullptr;     // on the helper
    int64_t r0 = 0, rows = 0;
};
struct GroupState {
    int n = 0;
    int64_t own_rows[kMaxJobs] = {};
    Slice sl[3][kMaxJobs];   // [helper][job]
};
struct Helper {
    int dev = -1;
    cudaStream_t s = nullptr;
    uint8_t* x = nullptr;    // the broadcast input on the helper
    float* y = nullptr;      // its output rows
    uint32_t* host = nullptr;
    unsigned* flags = nullptr;
};
struct Owner {
    int dev = -1;
    unsigned seq = 0;
    uint32_t* bhost = nullptr;   // the broadcast slot
    unsigned* bflags = nullptr;
    std::vector<Helper> helpers;
    std::map<int, GroupState> groups;   // il * kGroups + grp
};
struct Ctx {
    std::vector<std::unique_ptr<Owner>> owners;
    uint8_t* host = nullptr;
    size_t host_bytes = 0;
    unsigned* err = nullptr;
    std::vector<void*> dev_allocs;   // (device, pointer) pairs flattened: dev in the low slot via map below
    std::vector<int> dev_of;
};
Ctx* g_ctx = nullptr;

int64_t split_point(int64_t n, int k, int parts) {
    // 32-row aligned quarters
    const int64_t r = (n * k / parts + 31) / 32 * 32;
    return std::min(r, n);
}

}  // namespace

bool enabled() {
    static const bool on = [] {
        const char* v = getenv("MAYA_GLM_TP_DECODE");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}

size_t reserve_bytes() {
    if (!enabled()) return 0;
    const char* v = getenv("MAYA_GLM_TP_RESERVE_MB");
    return (size_t) (v != nullptr ? std::max(0LL, std::atoll(v)) : 850LL) << 20;
}

bool active(int dev) {
    // MAYA_GLM_TP_DRY=1: everything set up (the same VRAM left to the pool) but nothing split - the A/B twin of TP on
    static const bool dry = getenv("MAYA_GLM_TP_DRY") != nullptr && std::atoi(getenv("MAYA_GLM_TP_DRY")) != 0;
    if (g_ctx == nullptr || dry) return false;
    for (auto& o : g_ctx->owners)
        if (o->dev == dev) return true;
    return false;
}

bool failed() { return g_ctx != nullptr && g_ctx->err != nullptr && *(volatile unsigned*) g_ctx->err != 0; }

void teardown() {
    if (g_ctx == nullptr) return;
    for (auto& o : g_ctx->owners)
        for (auto& h : o->helpers) {
            cudaSetDevice(h.dev);
            if (h.s) cudaStreamSynchronize(h.s), cudaStreamDestroy(h.s);
        }
    for (size_t i = 0; i < g_ctx->dev_allocs.size(); ++i) {
        cudaSetDevice(g_ctx->dev_of[i]);
        cudaFree(g_ctx->dev_allocs[i]);
    }
    if (g_ctx->host) cudaFreeHost(g_ctx->host);
    delete g_ctx;
    g_ctx = nullptr;
}

bool setup(const std::vector<PartDesc>& parts, std::string& err) {
    if (!enabled() || parts.size() < 2) return true;
    int prev = 0;
    cudaGetDevice(&prev);
    auto* C = new Ctx();
    g_ctx = C;
    const int P = (int) parts.size();
    const int NH = P - 1;
    if (NH > 3) { err = "glm tp: at most 4 GPUs"; teardown(); return false; }
    // host-mapped exchange: per owner a broadcast slot and one slot per helper, each with kBlocks flags
    const size_t flag_bytes = (size_t) kBlocks * kFlagStride * sizeof(unsigned);
    const size_t per_owner = kInMax + flag_bytes + (size_t) NH * (kOutMax + flag_bytes);
    C->host_bytes = 4096 + (size_t) P * per_owner;
    if (cudaHostAlloc((void**) &C->host, C->host_bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        err = "glm tp: the host-mapped exchange did not allocate";
        cudaGetLastError();
        teardown();
        return false;
    }
    std::memset(C->host, 0, C->host_bytes);
    C->err = (unsigned*) C->host;
    const auto dalloc = [&](int dev, size_t bytes) -> void* {
        cudaSetDevice(dev);
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        C->dev_allocs.push_back(p);
        C->dev_of.push_back(dev);
        return p;
    };
    std::vector<size_t> used((size_t) P, 0);
    for (int pi = 0; pi < P; ++pi) {
        auto o = std::make_unique<Owner>();
        o->dev = parts[(size_t) pi].dev;
        uint8_t* base = C->host + 4096 + (size_t) pi * per_owner;
        o->bhost = (uint32_t*) base;
        o->bflags = (unsigned*) (base + kInMax);
        base += kInMax + flag_bytes;
        for (int hi = 0, q = 0; q < P; ++q) {
            if (q == pi) continue;
            Helper h;
            h.dev = parts[(size_t) q].dev;
            h.host = (uint32_t*) base;
            h.flags = (unsigned*) (base + kOutMax);
            base += kOutMax + flag_bytes;
            cudaSetDevice(h.dev);
            if (cudaStreamCreateWithFlags(&h.s, cudaStreamNonBlocking) != cudaSuccess) {
                err = "glm tp: helper stream";
                teardown();
                return false;
            }
            h.x = (uint8_t*) dalloc(h.dev, kInMax);
            h.y = (float*) dalloc(h.dev, kOutMax);
            if (!h.x || !h.y) { err = "glm tp: helper buffers"; teardown(); return false; }
            o->helpers.push_back(h);
            ++hi;
        }
        // the row slices
        for (const GroupDesc& gd : parts[(size_t) pi].groups) {
            GroupState gs;
            gs.n = gd.n;
            bool ok = true;
            for (int j = 0; j < gd.n && ok; ++j) {
                const Mat& m = gd.m[j];
                const size_t rb = gf::row_bytes(m.type, m.n_in);
                if (rb == 0 || m.w == nullptr || m.n_out % 32 != 0) { ok = false; break; }
                gs.own_rows[j] = split_point(m.n_out, 1, P);
                for (int h = 0; h < NH; ++h) {
                    const int64_t r0 = split_point(m.n_out, h + 1, P), r1 = split_point(m.n_out, h + 2, P);
                    Slice& s = gs.sl[h][j];
                    s.r0 = r0;
                    s.rows = r1 - r0;
                    const size_t bytes = (size_t) s.rows * rb;
                    s.w = dalloc(o->helpers[(size_t) h].dev, std::max<size_t>(bytes, 256));
                    if (s.w == nullptr) {
                        err = "glm tp: CUDA" + std::to_string(o->helpers[(size_t) h].dev) +
                              " is out of VRAM for the weight slices (raise MAYA_GLM_TP_RESERVE_MB)";
                        teardown();
                        return false;
                    }
                    if (cudaMemcpyPeer(s.w, o->helpers[(size_t) h].dev, (const uint8_t*) m.w + (size_t) r0 * rb, o->dev,
                                       bytes) != cudaSuccess) {
                        err = "glm tp: slice copy";
                        cudaGetLastError();
                        teardown();
                        return false;
                    }
                    for (int q = 0; q < P; ++q)
                        if (parts[(size_t) q].dev == o->helpers[(size_t) h].dev) used[(size_t) q] += bytes;
                }
                // a helper's outputs of one group must fit its slot
            }
            if (!ok) continue;
            int64_t out_max = 0;
            for (int h = 0; h < NH; ++h) {
                int64_t t = 0;
                for (int j = 0; j < gd.n; ++j) t += gs.sl[h][j].rows;
                out_max = std::max(out_max, t);
            }
            if ((size_t) out_max * sizeof(float) > kOutMax) continue;
            o->groups[gd.il * kGroups + gd.grp] = gs;
        }
        C->owners.push_back(std::move(o));
    }
    for (int q = 0; q < P; ++q) {
        cudaSetDevice(parts[(size_t) q].dev);
        cudaDeviceSynchronize();
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        std::fprintf(stderr, "glm tp: CUDA%d holds %.2f GB of the other parts' weight slices (%zu groups own), %.2f GB free\n",
                     parts[(size_t) q].dev, used[(size_t) q] / 1073741824.0, C->owners[(size_t) q]->groups.size(),
                     fr / 1073741824.0);
    }
    cudaSetDevice(prev);
    return true;
}

bool mv(int dev, int il, int grp, const gf::MvJob* split, int n, const gf::MvJob* local, int nl, cudaStream_t s,
        bool& ok) {
    ok = true;
    if (g_ctx == nullptr) return false;
    Owner* o = nullptr;
    for (auto& x : g_ctx->owners)
        if (x->dev == dev) o = x.get();
    if (o == nullptr) return false;
    auto it = o->groups.find(il * kGroups + grp);
    if (it == o->groups.end()) return false;
    const GroupState& gs = it->second;
    if (gs.n != n) return false;
    const void* xq = split[0].xq;
    const int64_t n_in = split[0].n_in;
    for (int j = 0; j < n; ++j)
        if (split[j].xq != xq || split[j].n_in != n_in || split[j].ncol != 1 || split[j].xq == nullptr) return false;
    const size_t in_bytes = (size_t) (n_in / 32) * 36;   // q8_1
    if (in_bytes > kInMax || in_bytes % 4 != 0) return false;
    const unsigned seq = ++o->seq;
    if (seq == 1) std::fprintf(stderr, "glm tp: CUDA%d splits its first GEMV group (layer %d)\n", dev, il);
    if (seq % 100000 == 0 && failed()) std::fprintf(stderr, "glm tp: an exchange timed out - the outputs are wrong\n");
    unsigned* err = g_ctx->err;
    const int words_in = (int) (in_bytes / 4);
    // 1. the owner publishes the q8_1 input
    k_put<<<kBlocks, kThreads, 0, s>>>((const uint32_t*) xq, words_in, o->bhost, o->bflags, seq);
    // 2. the owner's own rows (and its other jobs) in one launch
    gf::MvJob J[gf::kMaxMvJobs];
    int nj = 0;
    for (int j = 0; j < n; ++j) {
        J[nj] = split[j];
        J[nj].n_out = (int) gs.own_rows[j];
        ++nj;
    }
    for (int j = 0; j < nl && nj < gf::kMaxMvJobs; ++j) J[nj++] = local[j];
    if (!gf::mv(J, nj, s)) { ok = false; return true; }
    // 3. the helpers: fetch, their rows, put
    const int* skip = gf::get_skip();
    gf::set_skip(nullptr);   // (the owner's skip flag lives on the owner's device)
    CollectJobs c{};
    c.nh = (int) o->helpers.size();
    c.nj = n;
    for (int h = 0; h < c.nh; ++h) {
        Helper& H = o->helpers[(size_t) h];
        cudaSetDevice(H.dev);
        k_fetch<<<kBlocks, kThreads, 0, H.s>>>(o->bhost, words_in, (uint32_t*) H.x, o->bflags, seq, err);
        gf::MvJob HJ[kMaxJobs];
        int off = 0;
        c.off[h][0] = 0;
        for (int j = 0; j < n; ++j) {
            const Slice& sl = gs.sl[h][j];
            HJ[j] = split[j];
            HJ[j].w = sl.w;
            HJ[j].xq = H.x;
            HJ[j].xf = nullptr;
            HJ[j].bias = nullptr;
            HJ[j].y = H.y + off;
            HJ[j].n_out = (int) sl.rows;
            c.dst[h][j] = split[j].y + sl.r0;
            off += (int) sl.rows;
            c.off[h][j + 1] = off;
        }
        if (!gf::mv(HJ, n, H.s)) ok = false;
        k_put<<<kBlocks, kThreads, 0, H.s>>>((const uint32_t*) H.y, off, H.host, H.flags, seq);
        c.host[h] = H.host;
        c.flags[h] = H.flags;
    }
    gf::set_skip(skip);
    cudaSetDevice(dev);
    // 4. the owner gathers the helpers' rows
    k_collect<<<dim3(kBlocks, c.nh), kThreads, 0, s>>>(c, seq, err);
    return true;
}

}  // namespace strata::core::glmtp
