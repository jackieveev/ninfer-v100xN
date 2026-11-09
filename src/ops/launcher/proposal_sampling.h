#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void proposal_sample_launch(const Tensor& logits, const std::int32_t* id_map,
                            const SamplingConfig* configs, const Tensor& logical_positions,
                            std::int32_t step, Tensor& tokens, Tensor& q_support, Tensor& q_ids,
                            Tensor& q_probs, DeviceSpan workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
