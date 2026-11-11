#pragma once

#include <ninfer/types.h>

#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace kvmem {
class KvMemStore;
class RawKvStore;
class PageSoftmaxCuda;
} // namespace kvmem

namespace ninfer::targets::qwen3_6::detail {

inline constexpr std::uint32_t kKvMemBlockTokens = 64;

struct KvMemGeometry {
    std::uint32_t layers      = 0;
    std::uint32_t kv_heads    = 0;
    std::uint32_t query_heads = 0;
    std::uint32_t head_dim    = 0;
};

// One request lane's KVMem policy state: the KVMem block catalog (selection, LRFU history),
// the host Mean-K statistics index, and the selection currently applied to its paged KV.
// Logical KVMem blocks are exactly one 64-token KV page. Payload stays in the RAM archive of
// the paged allocation; this object never owns or moves KV bytes.
class KvMemSession {
public:
    KvMemSession(const KvMemOptions& options, KvMemGeometry geometry, int scorer_device);
    ~KvMemSession();
    KvMemSession(const KvMemSession&)            = delete;
    KvMemSession& operator=(const KvMemSession&) = delete;

    // Rewind statistics and the catalog to a retained frontier. Selection is kept: retained
    // resident pages remain valid, later pages were trimmed with the allocation.
    void rewind(std::uint32_t frontier);
    void set_media(std::vector<std::pair<std::uint32_t, std::uint32_t>> token_ranges);

    [[nodiscard]] std::uint32_t summarized_pages() const noexcept { return summarized_pages_; }
    // means: [layer][page][kv_head * head_dim] FP32 Mean-K of complete pages.
    void write_page_means(std::uint32_t first_page, std::uint32_t pages,
                          std::span<const float> means);
    void register_to(std::uint32_t tokens);

    [[nodiscard]] bool selection_active() const noexcept { return active_; }
    [[nodiscard]] std::uint32_t selection_frontier() const noexcept { return frontier_; }
    [[nodiscard]] const std::vector<std::uint32_t>& selected() const noexcept { return selected_; }
    // Tokens resident after the next append under the current selection.
    [[nodiscard]] std::uint64_t resident_tokens(std::uint32_t end) const noexcept;
    [[nodiscard]] std::uint32_t window_tokens() const noexcept { return window_; }
    [[nodiscard]] std::uint64_t selections() const noexcept { return selections_; }

    // Sink plus newest tail within B, over the registered history.
    std::vector<std::uint32_t> pressure_selection();
    // Scores the summarized history with the captured query rows and selects within B.
    // rows: [layer][row][query_head * head_dim] FP32 post-RMSNorm, pre-RoPE Q.
    std::vector<std::uint32_t> query_selection(std::span<const float> rows, std::uint32_t row_count);
    // Re-selects within B from retained scores when generation exhausts R.
    std::vector<std::uint32_t> decode_selection();
    void commit(std::vector<std::uint32_t> selected, std::uint32_t frontier);

private:
    std::vector<std::uint32_t> required(std::uint32_t frontier) const;
    std::vector<std::uint32_t> pick(std::uint32_t frontier);

    KvMemOptions options_;
    KvMemGeometry geometry_;
    std::uint32_t window_ = 0;
    std::uint32_t sink_blocks_ = 0;
    std::uint32_t recent_blocks_ = 0;
    std::unique_ptr<kvmem::KvMemStore> store_;
    std::unique_ptr<kvmem::RawKvStore> statistics_;
    std::unique_ptr<kvmem::PageSoftmaxCuda> scorer_;
    std::vector<std::uint32_t> media_pages_;
    std::uint32_t summarized_pages_ = 0;
    std::vector<std::uint32_t> selected_;
    std::uint32_t frontier_ = 0;
    bool active_ = false;
    std::uint64_t selections_ = 0;
};

} // namespace ninfer::targets::qwen3_6::detail
