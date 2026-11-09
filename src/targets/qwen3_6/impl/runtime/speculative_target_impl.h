#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::GqaExecutionEnvelope envelope, bool greedy_target) {
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.lanes, envelope,
                                 frame.target_hidden, frame.target_logits, frame.target_tokens,
                                 *frame.feature_sink, greedy_target);
    } else {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.lanes, envelope,
                                 frame.target_hidden, frame.target_logits, frame.target_tokens,
                                 greedy_target);
    }
    if (greedy_target) {
        ops::speculative_accept_greedy_tokens(
            frame.target_tokens, frame.drafts, frame.current_extents, frame.frontiers,
            frame.anchors, frame.licensed_tokens, frame.licensed_counts, frame.accepted_drafts,
            TextConfig::token_domain, frame.sampling, execution.device.stream);
    } else {
        ops::speculative_accept_greedy_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.current_extents,
            frame.frontiers, frame.anchors, frame.licensed_tokens, frame.licensed_counts,
            frame.accepted_drafts, TextConfig::token_domain, frame.sampling, frame.proposals,
            execution.work, execution.device.stream);
    }
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.lanes, continuation_hidden_store,
                 execution.device.stream);
}

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          std::span<const TargetVerifyFrameView> peers,
                          ops::GqaExecutionEnvelope envelope, bool greedy_target) {
    if (execution.peers.empty() || peers.size() != execution.peers.size()) {
        throw std::logic_error("parallel verify requires one frame per peer");
    }
    std::array<TargetVerifyFrameView, kMaximumExecutionDevices> views{};
    views[0] = frame;
    std::copy(peers.begin(), peers.end(), views.begin() + 1);
    for (const auto& peer : peers) {
        if (peer.replay_records == nullptr || peer.feature_sink != nullptr) {
            throw std::logic_error("parallel target verify peer frame is incomplete");
        }
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    const auto field = [&](Tensor TargetVerifyFrameView::* member) {
        return rank_views(execution, [&](int rank) { return views[rank].*member; });
    };
    const auto ids = field(&TargetVerifyFrameView::ids);
    const auto positions = field(&TargetVerifyFrameView::cache_positions);
    const auto rope = field(&TargetVerifyFrameView::rope_positions);
    const auto valid = field(&TargetVerifyFrameView::valid_columns);
    const auto rows = field(&TargetVerifyFrameView::kv_table_rows);
    const auto lanes = field(&TargetVerifyFrameView::lanes);
    const auto hidden = field(&TargetVerifyFrameView::target_hidden);
    const auto logits = field(&TargetVerifyFrameView::target_logits);
    const auto tokens = field(&TargetVerifyFrameView::target_tokens);
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(ids, positions, rope, valid, rows, lanes, envelope,
                                 hidden, logits, tokens, *frame.feature_sink, greedy_target);
    } else {
        card.target_verify_batch(ids, positions, rope, valid, rows, lanes, envelope,
                                 hidden, logits, tokens, greedy_target);
    }
    const auto& ec = *execution.peers[0].execution;
    for_each_rank(ec, [&](int rank) {
        auto& value = views[rank];
        auto& work = rank == 0 ? execution.work : *execution.peers[rank - 1].work;
        const auto stream = ec.dev[rank]->stream;
        if (greedy_target) {
            ops::speculative_accept_greedy_tokens(
                value.target_tokens, value.drafts, value.current_extents, value.frontiers,
                value.anchors, value.licensed_tokens, value.licensed_counts,
                value.accepted_drafts, TextConfig::token_domain, value.sampling, stream);
        } else {
            ops::speculative_accept_greedy_drafts(
                value.target_tokens, value.target_logits, value.drafts, value.current_extents,
                value.frontiers, value.anchors, value.licensed_tokens, value.licensed_counts,
                value.accepted_drafts, TextConfig::token_domain, value.sampling, value.proposals,
                work, stream);
        }
        ops::speculative_select_accepted_hidden(value.target_hidden, value.accepted_drafts,
                                                value.selected_hidden, stream);
    });
    ops::scatter(frame.selected_hidden, frame.lanes, continuation_hidden_store,
                 execution.device.stream);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
