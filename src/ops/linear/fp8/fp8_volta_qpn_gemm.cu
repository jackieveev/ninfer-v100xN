#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/fp8/fp8_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
using S = Fp8VoltaQpnSchedule;
static_assert(kFp8VoltaQpnRowsPerTile == S::kRowsPerTile,
              "dispatch mirrors the tile height for host code");

__global__ void qpn_cast_activation(const __nv_bfloat16* input, half* output, int count) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) { output[index] = __float2half_rn(__bfloat162float(input[index])); }
}
} // namespace

bool fp8_volta_qpn_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    if (n <= 0 || t <= 0) { return false; }
    // A lane consumes whole 128-byte lines, and the CTA's SPLITK warps split K by whole lines.
    // 8 is the floor every shipped SPLITK choice (8 or 16) divides down to (see the launcher).
    if (k % S::kKPerBlock != 0) { return false; }
    if (k / S::kKPerBlock < 8) { return false; }
    // Two 8-row A tiles cover T=9..16 in one pass over the weights. The W8 sibling rejected its
    // two-tile form because the 32x8 fused route already absorbed those rows for free; FP8 has no
    // 32x8 route, so here the alternative is chunking -- reading the whole weight twice -- and the
    // second tile wins easily.
    //
    // T=1 is admitted for the load-time prepacked layout (and used by every production FP8 layer
    // after bindings prepack). The SIMT decode kernel was a few percent faster on row-major T=1,
    // but cannot consume the resident QPN stream.
    return t >= 1 && t <= 4 * S::kRowsPerTile;
}

bool fp8_volta_qpn_precast_supported(std::int32_t n, std::int32_t k,
                                     std::int32_t tokens) noexcept {
    // Decode and MTP verify widths: one FP16 image feeds every CTA and removes the per-slice
    // BF16->FP16 convert from the inner loop. Prefill's CUTLASS route has its own cast.
    // Vocabulary keeps its existing non-precast QPN path (workspace stays zero for A16).
    if (tokens < 1 || tokens > 8) { return false; }
    return (k == 5120 && (n == 7168 || n == 8192 || n == 8704 || n == 17408 || n == 34816)) ||
           (n == 5120 && (k == 3072 || k == 6144 || k == 8704 || k == 17408));
}

std::size_t fp8_volta_qpn_precast_workspace_bytes(std::int32_t n, std::int32_t k,
                                                  std::int32_t max_tokens) {
    if (max_tokens < 1 || !fp8_volta_qpn_precast_supported(n, k, std::min(max_tokens, 8))) {
        return 0;
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP16, {k, std::min(max_tokens, 8)});
    return layout.peak_bytes(1);
}

void fp8_volta_qpn_cast_activation(const Tensor& input, Tensor& output, cudaStream_t stream) {
    const int count = static_cast<int>(input.numel());
    qpn_cast_activation<<<(count + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data), static_cast<half*>(output.data), count);
    CUDA_CHECK(cudaGetLastError());
}

void launch_fp8_volta_qpn(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n = out.ne[0];
    launch_fp8_volta_qpn_with_output(x, w, Fp8ContiguousOutput{static_cast<__nv_bfloat16*>(out.data),
                                                               n},
                                     n, stream);
}

void launch_fp8_volta_qpn_precast(const Tensor& x, const Weight& w, Tensor& out,
                                  WorkspaceArena& workspace, cudaStream_t stream) {
    launch_fp8_volta_qpn_with_precast_output(
        x, w, Fp8ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), out.ne[0]}, out.ne[0],
        workspace, stream);
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
