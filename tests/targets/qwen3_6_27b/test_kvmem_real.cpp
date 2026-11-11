#include <ninfer/engine.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

using namespace ninfer;

void require(bool valid, const char* message) {
    if (!valid) { throw std::runtime_error(message); }
}

EngineOptions config(const char* artifact, std::uint32_t budget) {
    EngineOptions options;
    options.artifact_path = artifact;
    options.tp = 2; options.devices = {0, 1};
    options.max_context = 32768;
    options.prefill_chunk = 1024;
    options.kv_cache = KvCacheStorage::Int8Group64;
    options.speculative = {SpeculativeBackend::Mtp, 3, ProposalHead::Optimized};
    options.kvmem.budget_tokens = budget;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(
        options.kvmem.enabled() ? options.kvmem.window_tokens() : options.max_context);
    return options;
}

ChatMessage message(ChatRole role, std::string text) {
    ChatMessage value; value.role = role;
    value.parts.push_back({MessagePartKind::Text, std::move(text), {}});
    return value;
}

PromptInput prompt(int functions, std::string question) {
    PromptInput input;
    input.options.enable_thinking = false;
    input.options.preserve_thinking = true;
    std::string archive = "Review this code archive.\n";
    for (int i = 0; i < functions; ++i) {
        if (i == functions / 3) { archive += "# The unique archive PIN is SILVER-PAGODA.\n"; }
        archive += "def add_" + std::to_string(i) + "(value):\n    return value + " +
                   std::to_string(i) + "\n\n";
    }
    input.messages.push_back(message(ChatRole::User, std::move(archive)));
    input.messages.push_back(message(ChatRole::Assistant, "Archive received."));
    input.messages.push_back(message(ChatRole::User, std::move(question)));
    return input;
}

GenerationResult run(Engine& engine, PromptInput input, int count = 64) {
    RequestOptions request;
    request.execution.requested_output_tokens = count;
    request.execution.sampling.temperature = 0;
    request.execution.sampling.presence_penalty = 0;
    request.execution.sampling.frequency_penalty = 0;
    request.stop.include_model_defaults = false;
    return engine.generate(engine.prepare(std::move(input)), request);
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_KVMEM_REAL_WEIGHTS");
    if (!artifact || !*artifact) { return 77; }
    try {
        const std::string question = "Implement a bounded FIFO queue in Python with push/pop and tests.";
        {
            // A working set that never exceeds B+R selects nothing and leaves attention complete.
            std::vector<TokenId> baseline;
            {
                Engine engine(config(artifact, 0));
                baseline = run(engine, prompt(120, question)).generated_token_ids;
            }
            Engine engine(config(artifact, 31744));
            const auto identity = run(engine, prompt(120, question));
            require(identity.generated_token_ids == baseline, "unpressured KVMem changed tokens");
            require(engine.memory_summary().kvmem_selections == 0, "unpressured KVMem selected");
            std::cout << "PASS unpressured KVMem parity\n";
        }
        {
            Engine engine(config(artifact, 8192));
            engine.debug_enable_peer_egress_check(true);
            auto query = prompt(1600, "Return only the unique archive PIN, exactly as written.");
            const auto tokens = engine.prepare(query).debug_token_ids().size();
            require(tokens > 9216, "test did not exceed the KVMem window");
            const auto answer = run(engine, query, 24);
            const auto memory = engine.memory_summary();
            require(memory.kvmem_selections > 0 && memory.kvmem_transfer_bytes > 0,
                    "KVMem did not select and archive");
            require(answer.content.find("SILVER-PAGODA") != std::string::npos,
                    "query selection lost the mid-archive fact");
            const auto first = run(engine, prompt(1600, question), 65);
            require(first.generated_token_ids.size() == 65, "MTP output commit count changed");
            const auto replay = run(engine, prompt(1600, question), 65);
            require(replay.reused_prompt_tokens > 0, "KVMem prefix did not reuse");
            auto [rounds, mismatches] = engine.debug_peer_egress_check_counts();
            require(rounds != 0 && mismatches == 0, "KVMem MTP peers diverged");
            std::cout << "PASS KVMem query recall (" << tokens << " tokens), MTP commit and prefix "
                      << "reuse; selections=" << engine.memory_summary().kvmem_selections
                      << " transfers=" << engine.memory_summary().kvmem_transfer_bytes << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL KVMem real: " << error.what() << '\n';
        return 1;
    }
}
