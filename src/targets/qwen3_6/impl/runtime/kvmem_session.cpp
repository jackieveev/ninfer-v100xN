#include "targets/qwen3_6/impl/runtime/kvmem_session.h"

#include "kvmem/graph_recall.hpp"
#include "kvmem/kvmem_store.hpp"
#include "kvmem/page_softmax_cuda.hpp"
#include "kvmem/query_policy.hpp"
#include "kvmem/raw_kv_store.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail {
namespace {

std::uint32_t pages_for(std::uint64_t tokens) {
    return static_cast<std::uint32_t>((tokens + kKvMemBlockTokens - 1) / kKvMemBlockTokens);
}

bool trace() {
    static const bool enabled = std::getenv("NINFER_KVMEM_TRACE") != nullptr;
    return enabled;
}

} // namespace

KvMemSession::KvMemSession(const KvMemOptions& options, KvMemGeometry geometry, int scorer_device)
    : options_(options), geometry_(geometry), window_(options.window_tokens()),
      sink_blocks_(pages_for(options.sink_tokens)),
      recent_blocks_(pages_for(options.recent_tokens)) {
    kvmem::KvMemStoreConfig config;
    config.lrfu              = options.lrfu;
    config.lrfu_lambda       = options.lrfu_lambda;
    config.lrfu_top_fraction = options.lrfu_top_fraction;
    config.block_tokens      = kKvMemBlockTokens;
    config.select_budget     = options.budget_tokens;
    config.gen_budget        = options.gen_reserve_tokens;
    config.sink_blocks       = sink_blocks_;
    config.recent_blocks     = recent_blocks_;
    store_ = std::make_unique<kvmem::KvMemStore>(config);
    const auto width = geometry.kv_heads * geometry.head_dim;
    statistics_ = std::make_unique<kvmem::RawKvStore>(kvmem::RawKvStoreConfig{
        .n_layer = geometry.layers, .n_embd_k = width, .n_embd_v = width,
        .block_tokens = kKvMemBlockTokens});
    if (options.query_scorer == KvMemQueryScorer::PageSoftmax) {
        scorer_ = std::make_unique<kvmem::PageSoftmaxCuda>(scorer_device);
    }
}

KvMemSession::~KvMemSession() = default;

void KvMemSession::rewind(std::uint32_t frontier) {
    store_->truncate_to(frontier);
    statistics_->truncate_to(frontier);
    summarized_pages_ = std::min(summarized_pages_, frontier / kKvMemBlockTokens);
    if (!active_) { return; }
    const auto pages = pages_for(frontier);
    std::erase_if(selected_, [&](std::uint32_t id) { return id >= pages; });
    frontier_ = std::min(frontier_, frontier);
}

void KvMemSession::set_media(std::vector<std::pair<std::uint32_t, std::uint32_t>> token_ranges) {
    media_pages_.clear();
    for (const auto& [begin, end] : token_ranges) {
        for (auto page = begin / kKvMemBlockTokens; page < pages_for(end); ++page) {
            media_pages_.push_back(page);
        }
    }
    std::sort(media_pages_.begin(), media_pages_.end());
    media_pages_.erase(std::unique(media_pages_.begin(), media_pages_.end()), media_pages_.end());
    if (media_pages_.size() + sink_blocks_ + recent_blocks_ > options_.budget_tokens / kKvMemBlockTokens) {
        throw std::invalid_argument("KVMem budget cannot keep every media block resident");
    }
    store_->set_media_ranges(std::move(token_ranges));
}

void KvMemSession::write_page_means(std::uint32_t first_page, std::uint32_t pages,
                                    std::span<const float> means) {
    const std::size_t width = static_cast<std::size_t>(geometry_.kv_heads) * geometry_.head_dim;
    if (first_page != summarized_pages_ ||
        means.size() != static_cast<std::size_t>(geometry_.layers) * pages * width) {
        throw std::logic_error("KVMem statistics must extend the summarized prefix");
    }
    std::vector<float> sum(width);
    for (std::uint32_t layer = 0; layer < geometry_.layers; ++layer) {
        for (std::uint32_t page = 0; page < pages; ++page) {
            const float* mean = means.data() + (static_cast<std::size_t>(layer) * pages + page) * width;
            for (std::size_t d = 0; d < width; ++d) {
                if (!std::isfinite(mean[d])) { throw std::runtime_error("Nonfinite KVMem key mean"); }
                sum[d] = mean[d] * static_cast<float>(kKvMemBlockTokens);
            }
            statistics_->write_layer_mean_sum((first_page + page) * kKvMemBlockTokens,
                                              kKvMemBlockTokens, layer, sum.data());
        }
    }
    summarized_pages_ += pages;
}

void KvMemSession::register_to(std::uint32_t tokens) {
    if (tokens < store_->total_tokens()) {
        throw std::logic_error("KVMem block catalog cannot register a shorter history");
    }
    if (tokens > store_->total_tokens()) { store_->register_append(tokens - store_->total_tokens()); }
}

std::uint64_t KvMemSession::resident_tokens(std::uint32_t end) const noexcept {
    if (!active_) { return end; }
    const auto pinned = frontier_ / kKvMemBlockTokens * kKvMemBlockTokens;
    return static_cast<std::uint64_t>(selected_.size()) * kKvMemBlockTokens + (end - pinned);
}

std::vector<std::uint32_t> KvMemSession::required(std::uint32_t frontier) const {
    return kvmem::query_prefix_required(frontier, kKvMemBlockTokens, sink_blocks_, recent_blocks_,
                                        media_pages_);
}

std::vector<std::uint32_t> KvMemSession::pick(std::uint32_t frontier) {
    const auto pages = pages_for(frontier);
    std::vector<std::uint32_t> candidates(pages);
    std::iota(candidates.begin(), candidates.end(), 0U);
    std::vector<double> scores(pages);
    for (std::uint32_t id = 0; id < pages; ++id) { scores[id] = store_->selection_score(id); }
    return store_->pick_scored_blocks(candidates, scores, required(frontier),
                                      options_.budget_tokens / kKvMemBlockTokens, true);
}

std::vector<std::uint32_t> KvMemSession::pressure_selection() {
    return store_->pick_prefill_pressure_blocks(media_pages_);
}

std::vector<std::uint32_t> KvMemSession::query_selection(std::span<const float> rows,
                                                         std::uint32_t row_count) {
    const std::uint32_t accepted = summarized_pages_ * kKvMemBlockTokens;
    const std::size_t width = static_cast<std::size_t>(geometry_.query_heads) * geometry_.head_dim;
    if (row_count == 0 || rows.size() != static_cast<std::size_t>(geometry_.layers) * row_count * width) {
        throw std::logic_error("KVMem query rows do not match the model geometry");
    }
    if (accepted != 0) {
        const auto budget = options_.budget_tokens / kKvMemBlockTokens;
        const auto frame = kvmem::query_score_frame(accepted, kKvMemBlockTokens, budget,
                                                    sink_blocks_, recent_blocks_, required(accepted));
        statistics_->ensure_blocks(summarized_pages_);
        const std::vector<std::uint32_t> layer_rows(geometry_.layers, row_count);
        const kvmem::PageQueryView query{rows.data(), layer_rows.data(), geometry_.layers, row_count,
                                         geometry_.query_heads, geometry_.kv_heads, geometry_.head_dim};
        const auto aggregation = options_.score_aggregation == KvMemScoreAggregation::Max
                                     ? kvmem::GqaScoreAggregation::Max
                                     : kvmem::GqaScoreAggregation::Mean;
        const auto scores = scorer_
            ? scorer_->score(*statistics_, query, frame, aggregation)
            : kvmem::mean_query_cosine_scores(*statistics_, query, summarized_pages_, aggregation);
        store_->begin_retrieval_round();
        store_->set_retrieval_scores(scores);
        store_->commit_retrieval_round();
    }
    return pick(store_->total_tokens());
}

std::vector<std::uint32_t> KvMemSession::decode_selection() { return pick(store_->total_tokens()); }

void KvMemSession::commit(std::vector<std::uint32_t> selected, std::uint32_t frontier) {
    selected_ = std::move(selected);
    frontier_ = frontier;
    active_   = true;
    ++selections_;
    if (trace()) {
        std::fprintf(stderr, "KVMEM_SELECT frontier=%u summarized=%u selected=%zu window=%u\n",
                     frontier, summarized_pages_, selected_.size(), window_);
    }
}

} // namespace ninfer::targets::qwen3_6::detail
