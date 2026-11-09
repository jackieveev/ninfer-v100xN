#pragma once

// Implements: include/ninfer/ops/proposal_sampling.h
// Match: contiguous BF16 [N,B] drafter logits.
// Algorithm assumptions: a tile partial top-20 per 512 rows, then one block per row merges
// the partial candidates through a carried top-20 and applies the sampling truncation.

#include "ninfer/ops/proposal_sampling.h"
#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

static_assert(kDraftDistributionCap == kSamplerCandidateCap,
              "stored draft distributions use the sampling candidate cap");

__launch_bounds__(kSamplerBlock) __global__ void proposal_sample_partial_kernel(
    const __nv_bfloat16* logits, std::int32_t rows, std::int32_t partials,
    unsigned long long* partial_keys) {
    const int row_b = static_cast<int>(blockIdx.y);
    const int tile  = static_cast<int>(blockIdx.x);
    __shared__ typename SamplingPartialSort::TempStorage sort_storage;
    unsigned long long keys[kSamplerItemsPerThread];
    const __nv_bfloat16* column = logits + static_cast<std::int64_t>(row_b) * rows;
    const int start             = tile * kSamplerPartialTileItems;
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = start + item * blockDim.x + threadIdx.x;
        keys[item]  = v < rows ? sampling_sort_key(__bfloat162float(column[v]), v) : 0ull;
    }
    SamplingPartialSort(sort_storage).Sort(keys, SamplingKeyGreater{});
    unsigned long long* out =
        partial_keys + (static_cast<std::int64_t>(row_b) * partials + tile) * kSamplerCandidateCap;
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int rank = threadIdx.x * kSamplerItemsPerThread + item;
        if (rank < kSamplerCandidateCap) { out[rank] = keys[item]; }
    }
}

__launch_bounds__(kSamplerGroupBlock) __global__ void proposal_sample_final_kernel(
    const unsigned long long* partial_keys, std::int32_t rows, std::int32_t partials,
    const std::int32_t* id_map, const SamplingConfig* configs,
    const std::int32_t* logical_positions, std::int32_t step, std::int32_t* tokens,
    std::int32_t* q_support, std::int32_t* q_ids, float* q_probs) {
    constexpr int kCap          = kSamplerCandidateCap;
    constexpr int kPerIteration = (kSamplerGroupTileItems - kCap) / kCap;
    const int row_b             = static_cast<int>(blockIdx.x);
    const int tid               = threadIdx.x;
    __shared__ typename SamplingGroupSort::TempStorage sort_storage;
    __shared__ unsigned long long carry[kCap];
    __shared__ float cand_val[kCap];
    __shared__ int cand_idx[kCap];
    __shared__ float prob[kCap];
    __shared__ int n_support;

    if (tid < kCap) { carry[tid] = 0ull; }
    const unsigned long long* row_keys =
        partial_keys + static_cast<std::int64_t>(row_b) * partials * kCap;
    for (int first = 0; first < partials; first += kPerIteration) {
        const int count = min(kPerIteration, partials - first);
        __syncthreads();
        unsigned long long keys[kSamplerGroupItemsPerThread];
#pragma unroll
        for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
            const int p = item * blockDim.x + tid;
            if (p < kCap) {
                keys[item] = carry[p];
            } else if (p - kCap < count * kCap) {
                keys[item] = row_keys[static_cast<std::int64_t>(first) * kCap + (p - kCap)];
            } else {
                keys[item] = 0ull;
            }
        }
        SamplingGroupSort(sort_storage).Sort(keys, SamplingKeyGreater{});
#pragma unroll
        for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
            const int rank = tid * kSamplerGroupItemsPerThread + item;
            if (rank < kCap) { carry[rank] = keys[item]; }
        }
    }
    __syncthreads();
    if (tid < kCap) {
        cand_val[tid] = sampling_key_float(carry[tid]);
        cand_idx[tid] = sampling_key_index(carry[tid]);
    }
    __syncthreads();

    const SamplingConfig cfg = configs[row_b];
    const auto map = [&](int v) { return id_map != nullptr ? id_map[v] : v; };
    if (!(cfg.temperature > 0.0f)) {
        if (tid == 0) {
            tokens[row_b]    = map(cand_idx[0]);
            q_support[row_b] = 0;
        }
        if (tid < kCap) {
            q_ids[row_b * kCap + tid]   = -1;
            q_probs[row_b * kCap + tid] = 0.0f;
        }
        return;
    }
    sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support,
                               sampling_candidate_cap(cfg, rows));
    const int n = n_support;
    if (tid < kCap) {
        q_ids[row_b * kCap + tid]   = tid < n ? map(cand_idx[tid]) : -1;
        q_probs[row_b * kCap + tid] = tid < n ? prob[tid] : 0.0f;
    }
    if (tid == 0) {
        const float u    = sampling_uniform(cfg.seed, logical_positions[row_b],
                                            kSamplePurposeDraftProposal,
                                            static_cast<unsigned int>(step));
        tokens[row_b]    = map(sampling_pick_from_support(cand_idx, prob, n, -1, u));
        q_support[row_b] = n;
    }
}

} // namespace ninfer::ops
