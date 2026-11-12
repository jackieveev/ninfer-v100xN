#include "core/pinned_host_pool.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace ninfer {

struct PinnedHostPool::State {
    explicit State(std::size_t bytes) : block_bytes(bytes) {}
    ~State() {
        for (std::byte* block : free) { (void)cudaFreeHost(block); }
    }

    const std::size_t block_bytes;
    std::mutex mutex;
    std::condition_variable wake;
    std::vector<std::byte*> free;
    std::size_t allocated = 0;
    std::size_t target    = 0;
    bool stop             = false;
    bool exhausted        = false;
};

PinnedHostPool::PinnedHostPool(std::size_t block_bytes)
    : state_(std::make_shared<State>(block_bytes)) {
    if (block_bytes == 0) { throw std::invalid_argument("PinnedHostPool block size must be nonzero"); }
}

PinnedHostPool::~PinnedHostPool() {
    {
        const std::lock_guard lock(state_->mutex);
        state_->stop = true;
    }
    state_->wake.notify_all();
    if (worker_.joinable()) { worker_.join(); }
}

void PinnedHostPool::reserve(std::size_t spare, std::size_t limit) {
    {
        const std::lock_guard lock(state_->mutex);
        if (state_->exhausted) { return; }
        const std::size_t in_use = state_->allocated - state_->free.size();
        const std::size_t blocks = std::min(limit, in_use + spare);
        if (blocks <= state_->target) { return; }
        state_->target = blocks;
    }
    state_->wake.notify_all();
    if (worker_.joinable()) { return; }
    worker_ = std::thread([state = state_] {
        std::unique_lock lock(state->mutex);
        for (;;) {
            state->wake.wait(lock, [&] { return state->stop || state->allocated < state->target; });
            if (state->stop) { return; }
            lock.unlock();
            void* block = nullptr;
            const bool ok =
                cudaHostAlloc(&block, state->block_bytes, cudaHostAllocPortable) == cudaSuccess;
            lock.lock();
            if (!ok) {
                (void)cudaGetLastError();
                state->target    = state->allocated; // Keep the pageable fallback from now on.
                state->exhausted = true;
                continue;
            }
            state->free.push_back(static_cast<std::byte*>(block));
            ++state->allocated;
        }
    });
}

std::shared_ptr<std::byte[]> PinnedHostPool::acquire(std::size_t bytes) {
    if (bytes <= state_->block_bytes) {
        const std::lock_guard lock(state_->mutex);
        if (!state_->free.empty()) {
            std::byte* block = state_->free.back();
            state_->free.pop_back();
            return std::shared_ptr<std::byte[]>(block, [state = state_](std::byte* released) {
                const std::lock_guard returned(state->mutex);
                state->free.push_back(released);
            });
        }
    }
    return std::shared_ptr<std::byte[]>(new std::byte[bytes]);
}

std::size_t PinnedHostPool::block_bytes() const noexcept { return state_->block_bytes; }

} // namespace ninfer
