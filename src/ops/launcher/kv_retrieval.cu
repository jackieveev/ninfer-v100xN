#include "ninfer/ops/kv_retrieval.h"
#include "core/device.h"
#include "ops/kernel/rope.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

namespace ninfer::ops {
namespace {

void tensor(const Tensor& t, DType dtype, const char* label) {
    if (!t.data || t.dtype != dtype || !t.is_contiguous()) {
        throw std::invalid_argument(label);
    }
}

__global__ void query_rows_kernel(const __nv_bfloat16* q, const int* pos, const int* span,
                                  __nv_bfloat16* rows, int heads, int capacity) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int first = max(span[0], span[1] - capacity);
    const int p = pos[t];
    if (p < first || p >= span[1]) { return; }
    const int row = p - (span[1] - capacity);
    rows[(static_cast<long long>(row) * heads + h) * 256 + threadIdx.x] =
        q[(static_cast<long long>(t) * heads + h) * 256 + threadIdx.x];
}

template <bool Quantized>
__device__ float key_value(PagedKVBatchLayerView cache, int physical, int token, int h, int d) {
    const char* base = static_cast<const char*>(cache.k_pages.data) +
        physical * cache.k_pages.nb[3] + h * cache.k_pages.nb[2] + token * cache.k_pages.nb[1];
    if constexpr (!Quantized) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(base)[d]);
    } else {
        const char* scales = static_cast<const char*>(cache.k_scale_pages.data) +
            physical * cache.k_scale_pages.nb[3] + h * cache.k_scale_pages.nb[2] +
            token * cache.k_scale_pages.nb[1];
        return static_cast<float>(reinterpret_cast<const signed char*>(base)[d]) *
               __half2float(reinterpret_cast<const __half*>(scales)[d / 64]);
    }
}

template <bool Quantized>
__global__ void page_means_kernel(PagedKVBatchLayerView cache, int first_page,
                                  const int* offsets, float* means,
                                  RopeFrequencyOverride frequency) {
    const int i = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    float* out = means + (static_cast<long long>(i) * cache.num_kv_heads + h) * 256 + d;
    const int p = first_page + i;
    const int physical = static_cast<const int*>(cache.block_tables.data)[p];
    const int offset = offsets[i];
    if (offset == kKvRetrievalNoRopeOffset || physical < 0) { *out = 0; return; }
    float sum = 0;
    for (int t = 0; t < 64; ++t) {
        float x = key_value<Quantized>(cache, physical, t, h, d);
        if (d < 64) {
            const int pair = d % 32;
            const int partner = d < 32 ? d + 32 : d - 32;
            const float other = key_value<Quantized>(cache, physical, t, h, partner);
            const float freq = frequency.inv_frequency ? frequency.inv_frequency[pair]
                                                       : kTextRopeInvFrequency[pair];
            float sine, cosine;
            sincosf(static_cast<float>(p * 64 + t + offset) * freq, &sine, &cosine);
            const float scale = frequency.inv_frequency ? frequency.mscale : 1.0f;
            x = (x * cosine + (d < 32 ? other : -other) * sine) / scale;
        }
        sum += x;
    }
    *out = sum / 64;
}

} // namespace

void kv_retrieval_query_rows(const Tensor& q, const Tensor& positions, const Tensor& span,
                             Tensor rows, cudaStream_t stream) {
    tensor(q, DType::BF16, "KV retrieval Q must be contiguous BF16");
    tensor(positions, DType::I32, "KV retrieval positions must be I32");
    tensor(span, DType::I32, "KV retrieval query span must be I32");
    tensor(rows, DType::BF16, "KV retrieval query rows must be contiguous BF16");
    if (q.ne[0] != 256 || (q.ne[1] != 12 && q.ne[1] != 24) || q.ne[3] != 1 ||
        rows.ne[0] != 256 || rows.ne[1] != q.ne[1] || rows.ne[2] <= 0 || rows.ne[3] != 1 ||
        positions.numel() != q.ne[2] || span.numel() != 2) {
        throw std::invalid_argument("KV retrieval query geometry is invalid");
    }
    query_rows_kernel<<<dim3(q.ne[2], q.ne[1]), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const int*>(positions.data),
        static_cast<const int*>(span.data), static_cast<__nv_bfloat16*>(rows.data), q.ne[1],
        rows.ne[2]);
    CUDA_CHECK(cudaGetLastError());
}

void kv_retrieval_page_means(PagedKVBatchLayerView cache, std::uint32_t first_page,
                             const Tensor& rope_offsets, Tensor means,
                             RopeFrequencyOverride frequency, cudaStream_t stream) {
    tensor(rope_offsets, DType::I32, "KV retrieval RoPE offsets must be I32");
    tensor(means, DType::FP32, "KV retrieval key means must be FP32");
    const auto pages = static_cast<std::uint32_t>(means.ne[2]);
    if (cache.head_dim != 256 || (cache.num_kv_heads != 2 && cache.num_kv_heads != 4) ||
        means.ne[0] != 256 || means.ne[1] != cache.num_kv_heads || pages == 0 ||
        rope_offsets.numel() < pages || cache.block_tables.ne[1] != 1 ||
        first_page + pages > static_cast<std::uint32_t>(cache.block_tables.ne[0]) ||
        (cache.dtype != DType::BF16 && cache.dtype != DType::I8) ||
        (cache.dtype == DType::I8 && cache.quant_group != 64)) {
        throw std::invalid_argument("KV retrieval key geometry is invalid");
    }
    const dim3 grid(pages, cache.num_kv_heads);
    const auto* offsets = static_cast<const int*>(rope_offsets.data);
    auto* out = static_cast<float*>(means.data);
    if (cache.dtype == DType::I8) {
        page_means_kernel<true><<<grid, 256, 0, stream>>>(cache, static_cast<int>(first_page),
                                                          offsets, out, frequency);
    } else {
        page_means_kernel<false><<<grid, 256, 0, stream>>>(cache, static_cast<int>(first_page),
                                                           offsets, out, frequency);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
