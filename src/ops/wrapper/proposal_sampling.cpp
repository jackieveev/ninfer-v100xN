#include "ninfer/ops/proposal_sampling.h"
#include "ops/common/math.h"
#include "ops/common/sampling_workspace.h"
#include "ops/launcher/proposal_sampling.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr const char* kOp = "proposal_sample";

void require_tensor(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t n1,
                    const char* name) {
    if (t.dtype != dtype || !t.is_contiguous() || t.data == nullptr || t.ne[0] != n0 ||
        t.ne[1] != n1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(kOp) + ": invalid " + name);
    }
}

} // namespace

std::size_t proposal_sample_workspace_capacity_bytes(std::int32_t proposal_rows,
                                                     std::int32_t max_batch) {
    if (proposal_rows < kDraftDistributionCap || max_batch < 1 ||
        max_batch > kSamplerMaxColumns) {
        throw std::invalid_argument("proposal_sample workspace: invalid domain");
    }
    const auto partials = static_cast<std::size_t>(div_up(proposal_rows, kSamplerPartialTileItems));
    return partials * static_cast<std::size_t>(max_batch) * kSamplerCandidateCap *
           sizeof(unsigned long long);
}

void proposal_sample(const Tensor& logits, const std::int32_t* id_map,
                     const SamplingConfig* configs, const Tensor& logical_positions,
                     std::int32_t step, Tensor& tokens, Tensor& q_support, Tensor& q_ids,
                     Tensor& q_probs, WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t rows  = logits.ne[0];
    const std::int32_t batch = logits.ne[1];
    if (rows < kDraftDistributionCap || batch < 1 || batch > kSamplerMaxColumns || step < 0) {
        throw std::invalid_argument("proposal_sample: invalid logits shape or step");
    }
    require_tensor(logits, DType::BF16, rows, batch, "logits");
    require_tensor(logical_positions, DType::I32, batch, 1, "logical_positions");
    require_tensor(tokens, DType::I32, batch, 1, "tokens");
    require_tensor(q_support, DType::I32, batch, 1, "q_support");
    require_tensor(q_ids, DType::I32, kDraftDistributionCap, batch, "q_ids");
    require_tensor(q_probs, DType::FP32, kDraftDistributionCap, batch, "q_probs");
    if (configs == nullptr) { throw std::invalid_argument("proposal_sample: configs is null"); }
    auto scope = workspace.scope();
    const DeviceSpan scratch =
        workspace.alloc_bytes(proposal_sample_workspace_capacity_bytes(rows, batch));
    detail::proposal_sample_launch(logits, id_map, configs, logical_positions, step, tokens,
                                   q_support, q_ids, q_probs, scratch, stream);
}

} // namespace ninfer::ops
