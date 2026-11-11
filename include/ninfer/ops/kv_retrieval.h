#pragma once

#include "core/paged_kv_cache.h"
#include "ninfer/ops/rope.h"

#include <cstdint>
#include <limits>

namespace ninfer::ops {

// KVMem statistics capture for one sequence. These Ops do not modify KV.
// All geometry is head-local: D=256, H=2 or 4 KV heads, six grouped query heads per KV head.

// query_rows: copy represented BF16 post-RMSNorm, pre-RoPE Q[D,6H,T] columns whose original
// position p lies in span[begin,end) and p >= end-R into rows[D,6H,R] at row p-(end-R).
// Other columns and untouched rows are left unchanged. positions are original I32 positions.
void kv_retrieval_query_rows(const Tensor& q, const Tensor& positions, const Tensor& span,
                             Tensor rows, cudaStream_t stream);

inline constexpr std::int32_t kKvRetrievalNoRopeOffset = std::numeric_limits<std::int32_t>::min();

// page_means: for logical pages [first_page, first_page+P), decode the ORIGINAL stored BF16 or
// INT8-G64/FP16-scaled K through the logical block table, undo its Text RoPE (64 rotary
// dimensions, theta=1e7, optional YaRN override) at rope position page*64+t+rope_offsets[i],
// and write the FP32 mean over the page's 64 tokens to means[D,H,P]. A page whose offset is
// kKvRetrievalNoRopeOffset (it contains non-text MRoPE positions) is written as zeros.
// Every summarized page must be complete and resident.
void kv_retrieval_page_means(PagedKVBatchLayerView cache, std::uint32_t first_page,
                             const Tensor& rope_offsets, Tensor means,
                             RopeFrequencyOverride frequency, cudaStream_t stream);

} // namespace ninfer::ops
