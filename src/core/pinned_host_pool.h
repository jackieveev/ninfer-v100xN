#pragma once

#include <cstddef>
#include <memory>
#include <thread>

namespace ninfer {

/**
 * Recycled fixed-size page-locked host blocks for stream-ordered device/host images.
 *
 * A block may be released while copies into or out of it are still queued. Reissuing it is
 * ordered only on the stream that queued those copies, so a pool serves one stream, or a fixed
 * stream per byte range of its blocks. A background thread page-locks blocks ahead of demand;
 * when none is free, acquire() returns pageable memory, for which device-to-host copies complete
 * before cudaMemcpyAsync returns.
 */
class PinnedHostPool {
public:
    explicit PinnedHostPool(std::size_t block_bytes);
    ~PinnedHostPool();

    PinnedHostPool(const PinnedHostPool&)            = delete;
    PinnedHostPool& operator=(const PinnedHostPool&) = delete;
    PinnedHostPool(PinnedHostPool&&)                 = delete;
    PinnedHostPool& operator=(PinnedHostPool&&)      = delete;

    // Page-locks in the background until `spare` blocks beyond those in use exist, never
    // exceeding `limit` blocks in total. Allocated blocks are kept until destruction.
    void reserve(std::size_t spare, std::size_t limit);
    [[nodiscard]] std::shared_ptr<std::byte[]> acquire(std::size_t bytes);
    [[nodiscard]] std::size_t block_bytes() const noexcept;

private:
    struct State;
    std::shared_ptr<State> state_;
    std::thread worker_;
};

} // namespace ninfer
