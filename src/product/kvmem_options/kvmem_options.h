#pragma once

#include "ninfer/types.h"

#include <functional>
#include <memory>
#include <string>

namespace kvmem { struct CliOptions; }

namespace ninfer::product {

// Canonical KVMem CLI contract (kvmem::CliOptions v1) resolved for the ninfer-v100 backend.
// Unsupported canonical settings (SSD tier, sessions, layerwise recall, graph propagation) are
// rejected instead of silently ignored.
class KvMemCommandLine {
public:
    KvMemCommandLine();
    ~KvMemCommandLine();
    KvMemCommandLine(const KvMemCommandLine&)            = delete;
    KvMemCommandLine& operator=(const KvMemCommandLine&) = delete;

    // Returns true when `arg` is a KVMem flag; `value(flag)` consumes its argument.
    bool parse(const std::string& arg, const std::function<const char*(const char*)>& value);
    // Disabled options when neither --kvmem nor a positive --kvmem-budget was given.
    KvMemOptions resolve();

private:
    std::unique_ptr<kvmem::CliOptions> options_;
};

inline constexpr const char* kKvMemBackendName = "ninfer-v100";

std::string kvmem_capabilities_json();
std::string kvmem_usage_text();

} // namespace ninfer::product
