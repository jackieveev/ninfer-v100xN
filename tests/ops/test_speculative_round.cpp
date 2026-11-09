#include "ninfer/ops/speculative_round.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

template <typename T>
void initialize(GuardedDeviceBuffer& buffer, const std::vector<T>& values) {
    buffer.copy_from_host(values.data(), values.size() * sizeof(T));
}

template <typename T>
std::vector<T> read(const GuardedDeviceBuffer& buffer, std::size_t count) {
    return from_device<T>(buffer.data(), count);
}

DeviceBuffer device_config(const ops::SamplingConfig& config) {
    return to_device(std::vector<ops::SamplingConfig>{config});
}

struct VerifyInputsExpected {
    std::vector<std::int32_t> verify_ids;
    std::vector<std::int32_t> positions;
};

VerifyInputsExpected verify_inputs_oracle(std::int32_t token,
                                          const std::vector<std::int32_t>& drafts,
                                          std::int32_t length) {
    VerifyInputsExpected expected{
        .verify_ids = std::vector<std::int32_t>(drafts.size() + 1),
        .positions  = std::vector<std::int32_t>(drafts.size() + 1),
    };
    expected.verify_ids[0] = token;
    for (std::size_t i = 0; i < drafts.size(); ++i) expected.verify_ids[i + 1] = drafts[i];
    for (std::size_t i = 0; i < expected.positions.size(); ++i) {
        expected.positions[i] = length + static_cast<std::int32_t>(i);
    }
    return expected;
}

int prepare_verify_case(int k) {
    const std::int32_t token_value  = 70000 + k;
    const std::int32_t length_value = 1000 - k;
    std::vector<std::int32_t> drafts(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) drafts[static_cast<std::size_t>(i)] = 37 + 7919 * i;
    const auto expected = verify_inputs_oracle(token_value, drafts, length_value);

    DeviceBuffer d_token  = to_device<std::int32_t>({token_value});
    DeviceBuffer d_drafts = to_device(drafts);
    DeviceBuffer d_length = to_device<std::int32_t>({length_value});
    DeviceBuffer d_extent = to_device<std::int32_t>({k});
    GuardedDeviceBuffer d_verify(expected.verify_ids.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_positions(expected.positions.size() * sizeof(std::int32_t));
    d_verify.fill(0xcd);
    d_positions.fill(0xef);

    Tensor token(d_token.p, DType::I32, {1});
    Tensor draft_tensor(d_drafts.p, DType::I32, {k});
    Tensor length(d_length.p, DType::I32, {1});
    Tensor extent(d_extent.p, DType::I32, {1});
    Tensor verify(d_verify.data(), DType::I32, {k + 1});
    Tensor positions(d_positions.data(), DType::I32, {k + 1});
    ops::speculative_prepare_verify_inputs(token, draft_tensor, length, extent, verify, positions,
                                           nullptr);
    cuda_synchronize();

    const std::string label = "speculative prepare K=" + std::to_string(k);
    int failures =
        verify_exact((label + " verify ids").c_str(),
                     read<std::int32_t>(d_verify, expected.verify_ids.size()), expected.verify_ids);
    failures += verify_exact((label + " positions").c_str(),
                             read<std::int32_t>(d_positions, expected.positions.size()),
                             expected.positions);
    failures += verify_exact((label + " token unchanged").c_str(),
                             from_device<std::int32_t>(d_token, 1), {token_value});
    failures += verify_exact((label + " drafts unchanged").c_str(),
                             from_device<std::int32_t>(d_drafts, drafts.size()), drafts);
    failures += verify_exact((label + " length unchanged").c_str(),
                             from_device<std::int32_t>(d_length, 1), {length_value});
    failures += d_verify.verify_guards((label + " verify guards").c_str());
    failures += d_positions.verify_guards((label + " positions guards").c_str());
    return failures;
}

struct AcceptExpected {
    std::vector<std::int32_t> sampled;
    std::int32_t num_sampled;
    std::int32_t accepted;
    std::int32_t length;
    std::int32_t token;
};

AcceptExpected accept_state_oracle(const std::vector<std::int32_t>& drafts, std::int32_t accepted,
                                   std::int32_t terminal_token, std::int32_t initial_length) {
    const int k = static_cast<int>(drafts.size());
    AcceptExpected expected{
        .sampled     = std::vector<std::int32_t>(static_cast<std::size_t>(k + 1), 0),
        .num_sampled = accepted + 1,
        .accepted    = accepted,
        .length      = initial_length + accepted + 1,
        .token       = terminal_token,
    };
    for (int i = 0; i < accepted; ++i) {
        expected.sampled[static_cast<std::size_t>(i)] = drafts[static_cast<std::size_t>(i)];
    }
    expected.sampled[static_cast<std::size_t>(accepted)] = terminal_token;
    return expected;
}

int execute_accept_case(const std::string& label, const std::vector<std::int32_t>& target_tokens,
                        const std::vector<std::uint16_t>& logits_bits, int physical_rows,
                        const std::vector<std::int32_t>& drafts, std::int32_t initial_length,
                        int token_domain, ops::SamplingConfig config,
                        const std::vector<std::int32_t>& initial_token_counts,
                        const AcceptExpected& expected, bool greedy_only = false,
                        int current_extent = -1) {
    const int k              = static_cast<int>(drafts.size());
    DeviceBuffer d_targets   = to_device(target_tokens);
    DeviceBuffer d_logits    = to_device(logits_bits);
    DeviceBuffer d_drafts    = to_device(drafts);
    DeviceBuffer d_counts    = to_device(initial_token_counts);
    config.token_counts      = static_cast<std::int32_t*>(d_counts.p);
    DeviceBuffer d_config    = device_config(config);
    const auto config_before = from_device<std::uint8_t>(d_config, sizeof(ops::SamplingConfig));

    GuardedDeviceBuffer d_length(sizeof(std::int32_t));
    GuardedDeviceBuffer d_token(sizeof(std::int32_t));
    GuardedDeviceBuffer d_sampled(static_cast<std::size_t>(k + 1) * sizeof(std::int32_t));
    GuardedDeviceBuffer d_num(sizeof(std::int32_t));
    GuardedDeviceBuffer d_accepted(sizeof(std::int32_t));
    const std::vector<std::int32_t> extents{current_extent < 0 ? k : current_extent};
    DeviceBuffer d_extent = to_device(extents);
    initialize(d_length, std::vector<std::int32_t>{initial_length});
    initialize(d_token, std::vector<std::int32_t>{-1234567});
    d_sampled.fill(0x9d);
    initialize(d_num, std::vector<std::int32_t>{-11});
    initialize(d_accepted, std::vector<std::int32_t>{-13});

    Tensor targets(d_targets.p, DType::I32, {k + 1});
    Tensor logits(d_logits.p, DType::BF16, {physical_rows, k + 1});
    Tensor draft_tensor(d_drafts.p, DType::I32, {k});
    Tensor extent(d_extent.p, DType::I32, {1});
    Tensor length(d_length.data(), DType::I32, {1});
    Tensor token(d_token.data(), DType::I32, {1});
    Tensor sampled(d_sampled.data(), DType::I32, {k + 1});
    Tensor num_sampled(d_num.data(), DType::I32, {1});
    Tensor accepted(d_accepted.data(), DType::I32, {1});
    const std::size_t workspace_bytes =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k, 1, 1);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    int failures = 0;
    if (greedy_only) {
        const auto invoke = [&](cudaStream_t stream) {
            ops::speculative_accept_greedy_tokens(
                targets, draft_tensor, extent, length, token, sampled, num_sampled, accepted,
                token_domain, static_cast<const ops::SamplingConfig*>(d_config.p), stream);
        };
        invoke(nullptr);
        cuda_synchronize();
        const auto check = [&] {
            int errors = verify_exact((label + " licensed").c_str(),
                                       read<std::int32_t>(d_sampled, k + 1), expected.sampled);
            errors += verify_exact((label + " count").c_str(), read<std::int32_t>(d_num, 1),
                                    {expected.num_sampled});
            errors += verify_exact((label + " accepted").c_str(), read<std::int32_t>(d_accepted, 1),
                                    {expected.accepted});
            errors += verify_exact((label + " frontier").c_str(), read<std::int32_t>(d_length, 1),
                                    {expected.length});
            errors += verify_exact((label + " anchor").c_str(), read<std::int32_t>(d_token, 1),
                                    {expected.token});
            return errors;
        };
        failures += check();
        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        cuda_check(cudaStreamCreate(&stream), "greedy accept stream");
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "greedy accept capture");
        invoke(stream);
        cuda_check(cudaStreamEndCapture(stream, &graph), "greedy accept capture end");
        cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "greedy accept graph");
        for (int replay = 0; replay < 2; ++replay) {
            initialize(d_length, std::vector<std::int32_t>{initial_length});
            cuda_check(cudaGraphLaunch(executable, stream), "greedy accept replay");
            cuda_synchronize(stream);
            failures += check();
        }
        cuda_check(cudaGraphExecDestroy(executable), "greedy accept graph destroy");
        cuda_check(cudaGraphDestroy(graph), "greedy accept definition destroy");
        cuda_check(cudaStreamDestroy(stream), "greedy accept stream destroy");
    } else {
        ops::speculative_accept_greedy_drafts(
            targets, logits, draft_tensor, extent, length, token, sampled, num_sampled, accepted,
            token_domain, static_cast<const ops::SamplingConfig*>(d_config.p), ops::DraftDistributionView{}, workspace,
            nullptr);
    }
    cuda_synchronize();

    failures += verify_exact((label + " sampled").c_str(), read<std::int32_t>(d_sampled, k + 1),
                              expected.sampled);
    failures += verify_exact((label + " num sampled").c_str(), read<std::int32_t>(d_num, 1),
                             {expected.num_sampled});
    failures += verify_exact((label + " accepted").c_str(), read<std::int32_t>(d_accepted, 1),
                             {expected.accepted});
    failures += verify_exact((label + " length").c_str(), read<std::int32_t>(d_length, 1),
                             {expected.length});
    failures +=
        verify_exact((label + " token").c_str(), read<std::int32_t>(d_token, 1), {expected.token});

    failures +=
        verify_exact((label + " target tokens unchanged").c_str(),
                     from_device<std::int32_t>(d_targets, target_tokens.size()), target_tokens);
    failures += verify_exact((label + " logits unchanged").c_str(),
                             from_device<std::uint16_t>(d_logits, logits_bits.size()), logits_bits);
    failures += verify_exact((label + " drafts unchanged").c_str(),
                             from_device<std::int32_t>(d_drafts, drafts.size()), drafts);
    failures += verify_exact((label + " extent unchanged").c_str(),
                             from_device<std::int32_t>(d_extent, 1), extents);
    failures += verify_exact((label + " config unchanged").c_str(),
                             from_device<std::uint8_t>(d_config, sizeof(ops::SamplingConfig)),
                             config_before);

    auto expected_counts = initial_token_counts;
    if (config.temperature > 0.0f) {
        for (int i = 0; i < expected.num_sampled; ++i) {
            ++expected_counts[static_cast<std::size_t>(
                expected.sampled[static_cast<std::size_t>(i)])];
        }
    }
    failures +=
        verify_exact((label + " token counts").c_str(),
                     from_device<std::int32_t>(d_counts, expected_counts.size()), expected_counts);

    failures += d_length.verify_guards((label + " length guards").c_str());
    failures += d_token.verify_guards((label + " token guards").c_str());
    failures += d_sampled.verify_guards((label + " sampled guards").c_str());
    failures += d_num.verify_guards((label + " num guards").c_str());
    failures += d_accepted.verify_guards((label + " accepted guards").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != (greedy_only ? 0 : workspace_bytes)) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int greedy_accept_case(int k, int accepted_count, int token_domain = 64, int extent = -1) {
    std::vector<std::int32_t> targets(static_cast<std::size_t>(k + 1));
    std::vector<std::int32_t> drafts(static_cast<std::size_t>(k));
    for (int i = 0; i <= k; ++i) {
        targets[static_cast<std::size_t>(i)] = 3 + 2 * i;
        if (i < k) drafts[static_cast<std::size_t>(i)] = targets[static_cast<std::size_t>(i)];
    }
    if (accepted_count < k) {
        drafts[static_cast<std::size_t>(accepted_count)] =
            targets[static_cast<std::size_t>(accepted_count)] + 1;
    }
    const std::int32_t initial_length = 200 + k;
    const auto expected               = accept_state_oracle(
        drafts, accepted_count, targets[static_cast<std::size_t>(accepted_count)], initial_length);
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(token_domain) * (k + 1));
    for (std::size_t i = 0; i < logits.size(); ++i) {
        logits[i] = static_cast<std::uint16_t>(0x3f00u + (i % 127u));
    }
    std::vector<std::int32_t> token_counts(token_domain);
    for (int i = 0; i < token_domain; ++i) token_counts[static_cast<std::size_t>(i)] = i % 5;
    const std::string label = "speculative greedy K=" + std::to_string(k) +
                              " A=" + std::to_string(accepted_count);
    ops::SamplingConfig config{};
    config.presence_penalty = 0.5F;
    config.frequency_penalty = 0.3F;
    return execute_accept_case(label, targets, logits, token_domain, drafts, initial_length,
                               token_domain, config, token_counts, expected, false, extent) +
           execute_accept_case(label + " tokens-only", targets, logits, token_domain, drafts,
                               initial_length, token_domain, config, token_counts, expected, true,
                               extent);
}

int deterministic_sampling_case() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    constexpr int k             = 5;
    constexpr int accepted      = 2;
    const std::vector<std::int32_t> drafts{17, 7919, 65537, 131071, 200003};
    std::vector<std::int32_t> targets(static_cast<std::size_t>(k + 1));
    for (int i = 0; i <= k; ++i) targets[static_cast<std::size_t>(i)] = 101 + i;
    constexpr std::int32_t correction = 150001;

    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * (k + 1), -20.0f);
    for (int col = 0; col <= k; ++col) {
        const std::size_t base = static_cast<std::size_t>(col) * physical_rows;
        const int winner =
            col < accepted ? drafts[static_cast<std::size_t>(col)] : correction + col - accepted;
        logits[base + static_cast<std::size_t>(winner)] = 20.0f;
        logits[base + token_domain]                     = 100.0f;
        logits[base + physical_rows - 1]                = 200.0f;
    }
    round_to_bf16(logits);
    std::vector<std::uint16_t> logits_bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) logits_bits[i] = f32_to_bf16(logits[i]);

    const std::int32_t initial_length = 4093;
    const auto expected = accept_state_oracle(drafts, accepted, correction, initial_length);
    std::vector<std::int32_t> token_counts(token_domain, 0);
    token_counts[static_cast<std::size_t>(drafts[0])]  = 3;
    token_counts[static_cast<std::size_t>(drafts[1])]  = 5;
    token_counts[static_cast<std::size_t>(correction)] = 7;

    ops::SamplingConfig config{};
    config.temperature = 1.0f;
    config.top_k       = 1;
    config.top_p       = 0.9f;
    config.min_p       = 0.5f;
    config.seed        = 0x123456789abcdef0ull;
    return execute_accept_case("speculative sampling deterministic support", targets, logits_bits,
                               physical_rows, drafts, initial_length, token_domain, config,
                               token_counts, expected);
}

int batched_sampling_workspace_stride_case() {
    constexpr int physical_rows = 257;
    constexpr int token_domain  = 257;
    constexpr int k             = 3;
    constexpr int batch         = 2;
    constexpr int columns       = k + 1;

    const std::vector<std::int32_t> drafts{10, 11, 12, 30, 31, 32};
    const std::vector<std::int32_t> winners{10, 20, 21, 22, 30, 31, 32, 33};
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns * batch,
                                      f32_to_bf16(-20.0f));
    for (int row = 0; row < batch; ++row) {
        for (int col = 0; col < columns; ++col) {
            const std::size_t base =
                (static_cast<std::size_t>(row) * columns + col) * physical_rows;
            logits[base + static_cast<std::size_t>(winners[row * columns + col])] =
                f32_to_bf16(20.0f);
        }
    }

    DeviceBuffer d_targets = to_device(winners);
    DeviceBuffer d_logits  = to_device(logits);
    DeviceBuffer d_drafts  = to_device(drafts);
    DeviceBuffer d_extents = to_device<std::int32_t>({k, k});
    DeviceBuffer d_lengths = to_device<std::int32_t>({100, 200});
    DeviceBuffer d_anchors = to_device<std::int32_t>({-1, -1});
    DeviceBuffer d_licensed(static_cast<std::size_t>(columns) * batch * sizeof(std::int32_t));
    DeviceBuffer d_counts(static_cast<std::size_t>(batch) * sizeof(std::int32_t));
    DeviceBuffer d_accepted(static_cast<std::size_t>(batch) * sizeof(std::int32_t));

    ops::SamplingConfig config{};
    config.temperature = 1.0f;
    config.top_k       = 1;
    const std::vector<ops::SamplingConfig> configs{config, config};
    DeviceBuffer d_configs = to_device(configs);

    Tensor targets(d_targets.p, DType::I32, {columns, batch});
    Tensor logits_tensor(d_logits.p, DType::BF16, {physical_rows, columns, batch});
    Tensor draft_tensor(d_drafts.p, DType::I32, {k, batch});
    Tensor extents(d_extents.p, DType::I32, {batch});
    Tensor lengths(d_lengths.p, DType::I32, {batch});
    Tensor anchors(d_anchors.p, DType::I32, {batch});
    Tensor licensed(d_licensed.p, DType::I32, {columns, batch});
    Tensor counts(d_counts.p, DType::I32, {batch});
    Tensor accepted(d_accepted.p, DType::I32, {batch});
    const std::size_t workspace_bytes =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k, batch,
                                                                       batch);
    WorkspaceArena workspace(workspace_bytes);
    ops::speculative_accept_greedy_drafts(
        targets, logits_tensor, draft_tensor, extents, lengths, anchors, licensed, counts, accepted,
        token_domain, static_cast<const ops::SamplingConfig*>(d_configs.p), ops::DraftDistributionView{}, workspace,
        nullptr);
    cuda_synchronize();

    int failures = verify_exact("speculative sampling B=2 licensed",
                                from_device<std::int32_t>(d_licensed, columns * batch),
                                {10, 20, 0, 0, 30, 31, 32, 33});
    failures += verify_exact("speculative sampling B=2 counts",
                             from_device<std::int32_t>(d_counts, batch), {2, 4});
    failures += verify_exact("speculative sampling B=2 accepted",
                             from_device<std::int32_t>(d_accepted, batch), {1, 3});
    failures += verify_exact("speculative sampling B=2 lengths",
                             from_device<std::int32_t>(d_lengths, batch), {102, 204});
    failures += verify_exact("speculative sampling B=2 anchors",
                             from_device<std::int32_t>(d_anchors, batch), {20, 33});
    // Reuse the batch layout with a zero-draft budget in one lane and a shortened window in
    // the other. Greedy acceptance must not use physical drafts outside either lane's extent.
    auto greedy_configs = to_device(std::vector<ops::SamplingConfig>(batch));
    auto greedy_lengths = to_device<std::int32_t>({100, 200});
    auto greedy_extents = to_device<std::int32_t>({0, 2});
    Tensor shortened(greedy_extents.p, DType::I32, {batch});
    Tensor frontiers(greedy_lengths.p, DType::I32, {batch});
    ops::speculative_accept_greedy_tokens(
        targets, draft_tensor, shortened, frontiers, anchors, licensed, counts, accepted,
        token_domain, static_cast<const ops::SamplingConfig*>(greedy_configs.p), nullptr);
    cuda_synchronize();
    failures += verify_exact("speculative greedy B=2 licensed",
                             from_device<std::int32_t>(d_licensed, columns * batch),
                             {10, 0, 0, 0, 30, 31, 32, 0});
    failures += verify_exact("speculative greedy B=2 counts",
                             from_device<std::int32_t>(d_counts, batch), {1, 3});
    failures += verify_exact("speculative greedy B=2 accepted",
                             from_device<std::int32_t>(d_accepted, batch), {0, 2});
    failures += verify_exact("speculative greedy B=2 frontiers",
                             from_device<std::int32_t>(greedy_lengths, batch), {101, 203});
    failures += verify_exact("speculative greedy B=2 anchors",
                             from_device<std::int32_t>(d_anchors, batch), {10, 32});
    return failures;
}

int proposal_sample_case() {
    constexpr int rows  = 1024;
    constexpr int batch = 3;
    constexpr int cap   = ops::kDraftDistributionCap;
    constexpr int kMap  = 5000;
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(rows) * batch, f32_to_bf16(-30.0f));
    const auto at = [&](int row, int v) -> std::uint16_t& {
        return logits[static_cast<std::size_t>(row) * rows + static_cast<std::size_t>(v)];
    };
    at(0, 3)   = f32_to_bf16(5.0f);
    at(0, 4)   = f32_to_bf16(4.0f);
    at(1, 777) = f32_to_bf16(5.0f);
    at(1, 778) = f32_to_bf16(4.5f);
    at(2, 100) = f32_to_bf16(3.0f);
    at(2, 600) = f32_to_bf16(2.0f);
    at(2, 900) = f32_to_bf16(2.0f);
    std::vector<std::int32_t> id_map(rows);
    for (int v = 0; v < rows; ++v) id_map[static_cast<std::size_t>(v)] = v + kMap;

    ops::SamplingConfig greedy{};
    ops::SamplingConfig top1{};
    top1.temperature = 1.0f;
    top1.top_k       = 1;
    top1.seed        = 7;
    ops::SamplingConfig sampled{};
    sampled.temperature = 0.7f;
    sampled.top_k       = 20;
    sampled.top_p       = 1.0f;
    sampled.min_p       = 0.01f;
    sampled.seed        = 0x5eedull;
    DeviceBuffer d_logits    = to_device(logits);
    DeviceBuffer d_map       = to_device(id_map);
    DeviceBuffer d_configs   = to_device(std::vector<ops::SamplingConfig>{greedy, top1, sampled});
    DeviceBuffer d_positions = to_device<std::int32_t>({40, 41, 42});
    DeviceBuffer d_tokens(batch * sizeof(std::int32_t));
    DeviceBuffer d_support(batch * sizeof(std::int32_t));
    DeviceBuffer d_ids(static_cast<std::size_t>(cap) * batch * sizeof(std::int32_t));
    DeviceBuffer d_probs(static_cast<std::size_t>(cap) * batch * sizeof(float));
    Tensor logits_tensor(d_logits.p, DType::BF16, {rows, batch});
    Tensor positions(d_positions.p, DType::I32, {batch});
    Tensor tokens(d_tokens.p, DType::I32, {batch});
    Tensor support(d_support.p, DType::I32, {batch});
    Tensor ids(d_ids.p, DType::I32, {cap, batch});
    Tensor probs(d_probs.p, DType::FP32, {cap, batch});
    WorkspaceArena workspace(ops::proposal_sample_workspace_capacity_bytes(rows, batch) + 4096);
    const auto* configs = static_cast<const ops::SamplingConfig*>(d_configs.p);
    const auto* map     = static_cast<const std::int32_t*>(d_map.p);

    ops::proposal_sample(logits_tensor, map, configs, positions, 0, tokens, support, ids, probs,
                         workspace, nullptr);
    cuda_synchronize();
    int failures = 0;
    const auto host_tokens  = from_device<std::int32_t>(d_tokens, batch);
    const auto host_support = from_device<std::int32_t>(d_support, batch);
    const auto host_ids     = from_device<std::int32_t>(d_ids, cap * batch);
    const auto host_probs   = from_device<float>(d_probs, cap * batch);
    failures += verify_exact("proposal_sample greedy/top1 tokens",
                             std::vector<std::int32_t>{host_tokens[0], host_tokens[1]},
                             std::vector<std::int32_t>{3 + kMap, 777 + kMap});
    failures += verify_exact("proposal_sample supports", host_support,
                             std::vector<std::int32_t>{0, 1, 3});

    // FP64 oracle of the truncated proposal distribution from the represented BF16 logits.
    const double x[3] = {3.0, 2.0, 2.0};
    double z          = 0.0;
    for (double v : x) z += std::exp((v - x[0]) / 0.7);
    const std::vector<std::int32_t> want_ids{100 + kMap, 600 + kMap, 900 + kMap};
    for (int j = 0; j < cap; ++j) {
        const std::int32_t want_greedy = -1;
        const std::int32_t want_top1   = j == 0 ? 777 + kMap : -1;
        const std::int32_t want_sample = j < 3 ? want_ids[static_cast<std::size_t>(j)] : -1;
        const double want_prob         = j < 3 ? std::exp((x[j] - x[0]) / 0.7) / z : 0.0;
        if (host_ids[static_cast<std::size_t>(j)] != want_greedy ||
            host_ids[static_cast<std::size_t>(cap + j)] != want_top1 ||
            host_ids[static_cast<std::size_t>(2 * cap + j)] != want_sample ||
            std::abs(host_probs[static_cast<std::size_t>(2 * cap + j)] - want_prob) > 1e-5 ||
            std::abs(host_probs[static_cast<std::size_t>(cap + j)] - (j == 0 ? 1.0 : 0.0)) > 1e-6) {
            std::cerr << "proposal_sample distribution mismatch at slot " << j << "\n";
            ++failures;
            break;
        }
    }

    constexpr int trials = 4000;
    std::vector<int> counts(3, 0);
    for (int step = 0; step < trials; ++step) {
        ops::proposal_sample(logits_tensor, map, configs, positions, step, tokens, support, ids,
                             probs, workspace, nullptr);
        const std::int32_t drawn = from_device<std::int32_t>(d_tokens, batch)[2];
        const auto found         = std::find(want_ids.begin(), want_ids.end(), drawn);
        if (found == want_ids.end()) {
            std::cerr << "proposal_sample drew a token outside its support\n";
            return failures + 1;
        }
        ++counts[static_cast<std::size_t>(found - want_ids.begin())];
    }
    for (int j = 0; j < 3; ++j) {
        const double freq = static_cast<double>(counts[static_cast<std::size_t>(j)]) / trials;
        const double want = std::exp((x[j] - x[0]) / 0.7) / z;
        if (std::abs(freq - want) > 0.03) {
            std::cerr << "proposal_sample draw frequency " << freq << " != " << want << "\n";
            ++failures;
        }
    }
    return failures;
}

// Drafts drawn from a stored distribution q must commit tokens distributed as the target p.
int stored_distribution_accept_case(int physical_rows) {
    constexpr int k   = 1;
    constexpr int cap = ops::kDraftDistributionCap;
    const int token_domain = physical_rows;
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * (k + 1),
                                      f32_to_bf16(-30.0f));
    const std::int32_t p_tokens[3] = {7, 11, 13};
    const double p_logits[3]       = {2.0, 1.0, 0.0};
    for (int col = 0; col <= k; ++col) {
        for (int j = 0; j < 3; ++j) {
            logits[static_cast<std::size_t>(col) * physical_rows +
                   static_cast<std::size_t>(p_tokens[j])] =
                f32_to_bf16(static_cast<float>(p_logits[j]));
        }
    }
    double z = 0.0;
    for (double v : p_logits) z += std::exp(v);
    const std::int32_t q_tokens[3] = {7, 11, 17};
    const float q_probs[3]         = {0.2f, 0.5f, 0.3f};
    std::vector<std::int32_t> q_ids(cap, -1);
    std::vector<float> q_prob_values(cap, 0.0f);
    for (int j = 0; j < 3; ++j) {
        q_ids[static_cast<std::size_t>(j)]         = q_tokens[j];
        q_prob_values[static_cast<std::size_t>(j)] = q_probs[j];
    }

    DeviceBuffer d_targets = to_device<std::int32_t>({7, 7});
    DeviceBuffer d_logits  = to_device(logits);
    DeviceBuffer d_drafts  = to_device<std::int32_t>({0});
    DeviceBuffer d_extents = to_device<std::int32_t>({k});
    DeviceBuffer d_lengths = to_device<std::int32_t>({0});
    DeviceBuffer d_anchors = to_device<std::int32_t>({-1});
    DeviceBuffer d_licensed(static_cast<std::size_t>(k + 1) * sizeof(std::int32_t));
    DeviceBuffer d_counts(sizeof(std::int32_t));
    DeviceBuffer d_accepted(sizeof(std::int32_t));
    DeviceBuffer d_support = to_device<std::int32_t>({3});
    DeviceBuffer d_q_ids   = to_device(q_ids);
    DeviceBuffer d_q_probs = to_device(q_prob_values);
    ops::SamplingConfig config{};
    config.temperature = 1.0f;
    config.top_k       = 20;
    config.top_p       = 1.0f;
    config.min_p       = 0.01f;
    DeviceBuffer d_config = device_config(config);

    Tensor targets(d_targets.p, DType::I32, {k + 1, 1});
    Tensor logits_tensor(d_logits.p, DType::BF16, {physical_rows, k + 1, 1});
    Tensor draft_tensor(d_drafts.p, DType::I32, {k, 1});
    Tensor extents(d_extents.p, DType::I32, {1});
    Tensor lengths(d_lengths.p, DType::I32, {1});
    Tensor anchors(d_anchors.p, DType::I32, {1});
    Tensor licensed(d_licensed.p, DType::I32, {k + 1, 1});
    Tensor counts(d_counts.p, DType::I32, {1});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    const ops::DraftDistributionView proposals{static_cast<const std::int32_t*>(d_support.p),
                                               static_cast<const std::int32_t*>(d_q_ids.p),
                                               static_cast<const float*>(d_q_probs.p)};
    WorkspaceArena workspace(
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k, 1, 1) +
        4096);

    constexpr int trials = 4000;
    std::uint64_t state  = 0x9e3779b97f4a7c15ull;
    std::vector<int> committed(3, 0);
    for (int trial = 0; trial < trials; ++trial) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        const double u = static_cast<double>(state >> 11) * 0x1.0p-53;
        const std::int32_t draft = u < 0.2 ? q_tokens[0] : (u < 0.7 ? q_tokens[1] : q_tokens[2]);
        const std::int32_t length = trial * 4;
        cudaMemcpy(d_drafts.p, &draft, sizeof(draft), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lengths.p, &length, sizeof(length), cudaMemcpyHostToDevice);
        ops::speculative_accept_greedy_drafts(
            targets, logits_tensor, draft_tensor, extents, lengths, anchors, licensed, counts,
            accepted, token_domain, static_cast<const ops::SamplingConfig*>(d_config.p), proposals,
            workspace, nullptr);
        const std::int32_t first = from_device<std::int32_t>(d_licensed, k + 1)[0];
        const auto found         = std::find(std::begin(p_tokens), std::end(p_tokens), first);
        if (found == std::end(p_tokens)) {
            std::cerr << "stored-distribution accept committed a token outside p's support\n";
            return 1;
        }
        ++committed[static_cast<std::size_t>(found - std::begin(p_tokens))];
    }
    int failures = 0;
    for (int j = 0; j < 3; ++j) {
        const double freq = static_cast<double>(committed[static_cast<std::size_t>(j)]) / trials;
        const double want = std::exp(p_logits[j]) / z;
        if (std::abs(freq - want) > 0.03) {
            std::cerr << "stored-distribution accept (rows " << physical_rows << ") frequency "
                      << freq << " != target " << want << "\n";
            ++failures;
        }
    }
    return failures;
}

int select_hidden_case(int rows, int columns, int accepted_value) {
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(rows) * columns);
    for (int col = 0; col < columns; ++col) {
        for (int row = 0; row < rows; ++row) {
            hidden[static_cast<std::size_t>(col) * rows + row] =
                static_cast<std::uint16_t>(0x0100u + ((col * 257 + row * 13) & 0x7fffu));
        }
    }
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(rows));
    std::copy_n(hidden.begin() + static_cast<std::ptrdiff_t>(accepted_value) * rows, rows,
                expected.begin());

    DeviceBuffer d_hidden   = to_device(hidden);
    DeviceBuffer d_accepted = to_device<std::int32_t>({accepted_value});
    GuardedDeviceBuffer d_out(static_cast<std::size_t>(rows) * sizeof(std::uint16_t));
    d_out.fill(0xcd);
    Tensor hidden_tensor(d_hidden.p, DType::BF16, {rows, columns});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor out(d_out.data(), DType::BF16, {rows, 1});
    ops::speculative_select_accepted_hidden(hidden_tensor, accepted, out, nullptr);
    cuda_synchronize();

    const std::string label =
        "speculative select D=" + std::to_string(rows) + " A=" + std::to_string(accepted_value);
    int failures = verify_exact((label + " output").c_str(),
                                read<std::uint16_t>(d_out, expected.size()), expected);
    failures += verify_exact((label + " hidden unchanged").c_str(),
                             from_device<std::uint16_t>(d_hidden, hidden.size()), hidden);
    failures += verify_exact((label + " accepted unchanged").c_str(),
                             from_device<std::int32_t>(d_accepted, 1), {accepted_value});
    failures += d_out.verify_guards((label + " output guards").c_str());
    return failures;
}

int remap_case(int token_count) {
    constexpr int map_size = 131072;
    std::vector<std::int32_t> id_map(map_size);
    for (int i = 0; i < map_size; ++i) {
        const auto value                    = 65537u * static_cast<std::uint32_t>(i) + 17u;
        id_map[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(value & (map_size - 1u));
    }
    std::vector<std::int32_t> proposals(static_cast<std::size_t>(token_count));
    for (int i = 0; i < token_count; ++i) {
        proposals[static_cast<std::size_t>(i)] =
            i == 0 ? 0 : (i == token_count - 1 ? map_size - 1 : (7919 * i) & (map_size - 1));
    }
    std::vector<std::int32_t> expected(proposals.size());
    for (std::size_t i = 0; i < proposals.size(); ++i) {
        expected[i] = id_map[static_cast<std::size_t>(proposals[i])];
    }

    DeviceBuffer d_map = to_device(id_map);
    GuardedDeviceBuffer d_proposals(proposals.size() * sizeof(std::int32_t));
    initialize(d_proposals, proposals);
    Tensor proposal_tensor(d_proposals.data(), DType::I32, {token_count});
    ops::proposal_remap_token_ids(proposal_tensor, static_cast<const std::int32_t*>(d_map.p),
                                  map_size, nullptr);
    cuda_synchronize();

    const std::string label = "proposal remap T=" + std::to_string(token_count);
    int failures            = verify_exact((label + " in-place output").c_str(),
                                           read<std::int32_t>(d_proposals, proposals.size()), expected);
    failures += verify_exact((label + " map unchanged").c_str(),
                             from_device<std::int32_t>(d_map, id_map.size()), id_map);
    failures += d_proposals.verify_guards((label + " guards").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "speculative_round: SKIP (CUDA unavailable)\n";
        return 77;
    }

    int failures = 0;
    const std::size_t k15 =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 15, 15, 1, 1);
    if (k15 == 0 || k15 != ops::sampling_workspace_capacity_bytes(257, 16, 16) ||
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 16, 16, 1, 1) != 0 ||
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 1, 16, 1, 1) != k15 ||
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 15, 15, 1, 2) !=
            2 * k15) {
        std::cerr << "speculative accept workspace did not close over K+1 sampling columns\n";
        ++failures;
    }
    try {
        (void)ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 0, 15, 1, 1);
        std::cerr << "speculative accept workspace accepted an invalid draft interval\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    for (const int k : {1, 5, 15}) failures += prepare_verify_case(k);
    failures += greedy_accept_case(1, 0);
    failures += greedy_accept_case(3, 0, 64, 0);
    failures += greedy_accept_case(3, 2, 64, 2);
    failures += greedy_accept_case(3, 3);
    failures += greedy_accept_case(5, 2);
    failures += greedy_accept_case(5, 5);
    failures += greedy_accept_case(15, 7, 257);
    failures += deterministic_sampling_case();
    failures += batched_sampling_workspace_stride_case();
    failures += proposal_sample_case();
    failures += stored_distribution_accept_case(257);
    failures += stored_distribution_accept_case(248320);
    failures += select_hidden_case(5120, 6, 0);
    failures += select_hidden_case(5120, 6, 5);
    failures += select_hidden_case(2048, 16, 7);
    failures += remap_case(1);
    failures += remap_case(15);
    failures += remap_case(120);

    if (failures != 0) {
        std::cerr << "speculative_round failures=" << failures << '\n';
        return 1;
    }
    std::cout << "speculative_round: PASS\n";
    return 0;
}
