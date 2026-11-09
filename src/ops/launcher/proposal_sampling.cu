// Implements: include/ninfer/ops/proposal_sampling.h
// Match: validated BF16 [N,B] drafter logits and caller-owned partial-key workspace.
#include "ops/launcher/proposal_sampling.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kernel/proposal_sampling.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

void proposal_sample_launch(const Tensor& logits, const std::int32_t* id_map,
                            const SamplingConfig* configs, const Tensor& logical_positions,
                            std::int32_t step, Tensor& tokens, Tensor& q_support, Tensor& q_ids,
                            Tensor& q_probs, DeviceSpan workspace, cudaStream_t stream) {
    const std::int32_t rows     = logits.ne[0];
    const std::int32_t batch    = logits.ne[1];
    const std::int32_t partials = div_up(rows, kSamplerPartialTileItems);
    auto* keys                  = static_cast<unsigned long long*>(workspace.data);
    const dim3 partial_grid(static_cast<unsigned int>(partials), static_cast<unsigned int>(batch));
    proposal_sample_partial_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), rows, partials, keys);
    CUDA_CHECK(cudaGetLastError());
    proposal_sample_final_kernel<<<static_cast<unsigned int>(batch), kSamplerGroupBlock, 0,
                                   stream>>>(
        keys, rows, partials, id_map, configs,
        static_cast<const std::int32_t*>(logical_positions.data), step,
        static_cast<std::int32_t*>(tokens.data), static_cast<std::int32_t*>(q_support.data),
        static_cast<std::int32_t*>(q_ids.data), static_cast<float*>(q_probs.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
