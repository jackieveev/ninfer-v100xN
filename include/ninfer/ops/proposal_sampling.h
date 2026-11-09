#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Maximum support of one stored draft distribution; equal to the sampling candidate cap.
inline constexpr std::int32_t kDraftDistributionCap = 20;

// Caller-owned transient capacity for proposal_sample() at every batch in [1,max_batch].
[[nodiscard]] std::size_t proposal_sample_workspace_capacity_bytes(std::int32_t proposal_rows,
                                                                   std::int32_t max_batch);

/**
 * Op: proposal_sample
 *
 * Draws one speculative proposal per row from the drafter's own truncated distribution and
 * publishes that distribution, so verification can use exact speculative rejection sampling
 * with the true proposal probabilities instead of a one-hot draft.
 *
 * Math:
 *   For row b let x_v = float(logits[v,b]) for v in [0,N). Candidates are the
 *   min(20,N) largest x_v, ordered by value with the lower v breaking ties.
 *   configs[b].temperature <= 0:
 *     tokens[b] = map(argmax), q_support[b] = 0 (a one-hot proposal).
 *   Otherwise the candidates are truncated and normalized exactly as sample() does without
 *   penalties (top_k cap, temperature, min_p, top_p, renormalization) into support
 *   q_0..q_{n-1}; one candidate is drawn with counter-based RNG key
 *   (configs[b].seed, logical_positions[b], kSamplePurposeDraftProposal, step);
 *   tokens[b] = map(drawn), q_support[b] = n, q_ids[j,b] = map(candidate j),
 *   q_probs[j,b] = q_j for j < n, and q_ids = -1 / q_probs = 0 for j >= n.
 *   map(v) = id_map[v] when id_map is non-null, otherwise v.
 *
 * Shapes:
 *   logits contiguous BF16 [N,B] with N >= 20 and 1 <= B <= 16; logical_positions, tokens and
 *   q_support contiguous I32 [B]; q_ids I32 and q_probs FP32 contiguous [20,B]. configs is a
 *   device SamplingConfig[B]; id_map is null or a device I32[N]. Outputs do not overlap inputs.
 *
 * Effects:
 *   Writes every output element. No token_counts update and no other persistent state.
 */
void proposal_sample(const Tensor& logits, const std::int32_t* id_map,
                     const SamplingConfig* configs, const Tensor& logical_positions,
                     std::int32_t step, Tensor& tokens, Tensor& q_support, Tensor& q_ids,
                     Tensor& q_probs, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
