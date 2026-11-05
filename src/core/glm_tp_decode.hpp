// Decode-only in-box tensor parallel for the GLM layer split (MAYA_GLM_TP_DECODE=1).
//
// The part that owns a layer keeps running it, but its big quantized GEMVs (KDA q/k/v and out, DSA q_b and out, the
// dense FFN) are split by output rows across every GPU of the split: the owner computes the first row range from its
// own weights, the other GPUs ("helpers") compute the rest from row-slice copies made at load time.  The q8_1 input is
// broadcast and the output rows gathered through pinned, host-mapped memory (no peer access needed).  Every row is
// computed by the same kernel from the same input, so the results are bit-identical to the unsplit GEMVs.
#pragma once

#include "strata/kernels/glm_fast.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

namespace strata::core::glmtp {

// Split groups: the GEMVs of one group share their q8_1 input
enum Group { kKdaQkv = 0, kKdaOut = 1, kDsaQb = 2, kDsaOut = 3, kDenseGu = 4, kDenseDown = 5, kGroups = 6 };
constexpr int kMaxJobs = 3;

/// MAYA_GLM_TP_DECODE=1
bool enabled();
/// VRAM each part leaves outside its expert pool for the helpers' weight slices (MAYA_GLM_TP_RESERVE_MB, default 850)
size_t reserve_bytes(int part);
/// MAYA_GLM_TP_HELPERS: whether split part `part` lends its GPU to the other parts (default: every part)
bool helper_part(int part);

struct Mat {
    const void* w = nullptr;
    int type = 0;
    int64_t n_in = 0, n_out = 0;
};
struct GroupDesc {
    int il = 0, grp = 0, n = 0;
    Mat m[kMaxJobs];
};
struct PartDesc {
    int dev = -1;
    std::vector<GroupDesc> groups;
};

/// After every part is loaded: the helpers' slices, streams and the host-mapped exchange.  false + err: TP stays off.
bool setup(const std::vector<PartDesc>& parts, std::string& err);
/// Whether the part on `dev` splits its GEMVs
bool active(int dev);
/// Enqueue the split GEMVs of (il, grp) on the owner's stream s, plus `local` (other jobs of the owner, run in the
/// same launch as its own slice).  false: nothing was enqueued (no such group) - the caller runs gf::mv itself.
bool mv(int dev, int il, int grp, const strata::kernels::glmf::MvJob* split, int n,
        const strata::kernels::glmf::MvJob* local, int nl, cudaStream_t s, bool& ok);
/// A spin timed out somewhere (the results since are garbage): printed once by the caller
bool failed();
void teardown();

}  // namespace strata::core::glmtp
