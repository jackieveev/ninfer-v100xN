#include "product/kvmem_options/kvmem_options.h"

#include "kvmem/cli_options.h"
#include "kvmem/kvmem_store.hpp"
#include "kvmem/launch_contract.hpp"

#include <stdexcept>

namespace ninfer::product {
namespace {

// The ninfer-v100 backend maps one KVMem block onto one 64-token KV page.
constexpr std::uint32_t kKvMemBlockTokens = 64;

} // namespace

KvMemCommandLine::KvMemCommandLine() : options_(std::make_unique<kvmem::CliOptions>()) {}
KvMemCommandLine::~KvMemCommandLine() = default;

bool KvMemCommandLine::parse(const std::string& arg,
                             const std::function<const char*(const char*)>& value) {
    return options_->parse(arg, [&](const char* flag) { return value(flag); });
}

KvMemOptions KvMemCommandLine::resolve() {
    kvmem::CliOptions& cli = *options_;
    cli.resolve();
    if (!cli.enabled) { return {}; }
    if (cli.payload_disk_bytes != 0 || !cli.payload_disk_dir.empty()) {
        throw std::invalid_argument("--kvmem-disk-* is unavailable on the ninfer-v100 backend");
    }
    if (cli.sessions_set) {
        throw std::invalid_argument("--kvmem-sessions is unavailable on the ninfer-v100 backend");
    }
    if (cli.recall_strategy != kvmem::RecallStrategy::Legacy) {
        throw std::invalid_argument(
            "--kvmem-recall-strategy layerwise is unavailable on the ninfer-v100 backend");
    }
    if (cli.graph_alpha_set && cli.graph_alpha != 0.0) {
        throw std::invalid_argument(
            "the ninfer-v100 backend collects no attention graph; use --kvmem-graph-alpha 0");
    }
    const auto keep = kvmem::resolve_kvmem_keep_allocation(
        kKvMemBlockTokens, cli.budget, -1, -1,
        cli.sink_tokens_set ? static_cast<std::int64_t>(cli.sink_tokens) : -1,
        cli.recent_tokens_set ? static_cast<std::int64_t>(cli.recent_tokens) : -1);
    KvMemOptions out;
    out.budget_tokens      = cli.budget;
    out.gen_reserve_tokens = cli.gen_reserve;
    out.sink_tokens        = keep.sink_effective_tokens;
    out.recent_tokens      = keep.recent_effective_tokens;
    out.host_bytes         = cli.payload_host_bytes;
    out.query_scorer       = cli.query_scorer == kvmem::QueryScorer::MeanQCosine
                                 ? KvMemQueryScorer::MeanQCosine
                                 : KvMemQueryScorer::PageSoftmax;
    out.score_aggregation  = cli.score_agg == 1 ? KvMemScoreAggregation::Max
                                                : KvMemScoreAggregation::Mean;
    out.lrfu               = cli.lrfu;
    out.lrfu_lambda        = cli.lrfu_lambda;
    out.lrfu_top_fraction  = cli.lrfu_top_fraction;
    return out;
}

std::string kvmem_capabilities_json() { return kvmem::launch_capabilities(kKvMemBackendName); }

std::string kvmem_usage_text() {
    return "       --kvmem enables KVMem block-sparse long-context memory (default off; SM70 27B "
           "TP1/TP2, one request, Text/Vision None/MTP).\n"
           "       --kvmem-budget B (default 32768) selected history tokens; --kvmem-gen-reserve R "
           "(default 1024) unselected appends; the GPU KV window is B+R.\n"
           "       --kvmem-host-mib H (default 16384) bounds the packed host KV archive for "
           "--max-context.\n"
           "       --kvmem-sink-tokens N / --kvmem-recent-tokens N pin the head/tail (default "
           "1%/8% of B, clamped to 1024..2048/4096..16384); "
           "--kvmem-query-scorer page-softmax|meanq-cosine; --kvmem-score-agg mean|max;\n"
           "       --kvmem-lrfu / --kvmem-lrfu-lambda F / --kvmem-lrfu-top-frac F; "
           "--kvmem-capabilities prints the backend contract.\n";
}

} // namespace ninfer::product
