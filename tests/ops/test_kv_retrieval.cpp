#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/kv_retrieval.h"
#include "ops/gqa_attention_fixture.h"

#include <cmath>
#include <cstring>
#include <iostream>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::gqa;

namespace {

// Exact copy oracle: rows[R] receives the columns of span [begin,end) that fall in the last R.
int query_rows_oracle(int heads) {
    const int q_heads = heads * 6, tokens = 193, rows = 64, base = 85019;
    std::vector<int> positions(tokens);
    for (int t = 0; t < tokens; ++t) { positions[t] = base + t; }
    const int span[2] = {base + 13, base + 129};
    const auto q = to_bf16_bits(make_bf16_values(256 * q_heads * tokens, 81, -0.5f, 0.5f));
    std::vector<std::uint16_t> initial(256 * q_heads * rows, 0x7fc0);
    GuardedDeviceBuffer dq(q.size() * 2), dp(tokens * 4), ds(8), dr(initial.size() * 2);
    dq.copy_from_host(q.data(), q.size() * 2);
    dp.copy_from_host(positions.data(), tokens * 4);
    ds.copy_from_host(span, 8);
    dr.copy_from_host(initial.data(), initial.size() * 2);
    ops::kv_retrieval_query_rows(Tensor(dq.data(), DType::BF16, {256, q_heads, tokens}),
                                 Tensor(dp.data(), DType::I32, {tokens}),
                                 Tensor(ds.data(), DType::I32, {2}),
                                 Tensor(dr.data(), DType::BF16, {256, q_heads, rows}), nullptr);
    cuda_synchronize();
    const auto actual = copy_from_guarded<std::uint16_t>(dr, initial.size());
    int failures = 0;
    for (int row = 0; row < rows; ++row) {
        const int t = span[1] - rows + row - base;
        for (int h = 0; h < q_heads; ++h) {
            for (int d = 0; d < 256; ++d) {
                const auto expected = q[(static_cast<std::size_t>(t) * q_heads + h) * 256 + d];
                failures += actual[(static_cast<std::size_t>(row) * q_heads + h) * 256 + d] != expected;
            }
        }
    }
    failures += dr.verify_guards("query rows");
    std::cout << "KVMem query rows exact oracle H=" << heads << " failures=" << failures << '\n';
    return failures;
}

// FP64 oracle: decode the stored K, undo Text RoPE at page*64+t+offset, average the page.
int page_means_oracle(int heads, DType dtype) {
    const Geometry geometry{"retrieval", heads * 6, heads};
    const int pages = 6, tokens = pages * 64;
    auto host = make_cache(geometry, dtype, tokens, 773);
    DeviceCache cache(host, MappingPattern::Fragmented);
    const auto view = cache.batch_view();
    const int first = 1, count = 5;
    const std::int32_t offsets[count] = {0, -37, 85000, ops::kKvRetrievalNoRopeOffset, 0};
    GuardedDeviceBuffer doff(sizeof(offsets)), dm(count * heads * 256 * 4);
    doff.copy_from_host(offsets, sizeof(offsets));
    ops::kv_retrieval_page_means(view, first, Tensor(doff.data(), DType::I32, {count}),
                                 Tensor(dm.data(), DType::FP32, {256, heads, count}), {}, nullptr);
    cuda_synchronize();
    const auto actual = copy_from_guarded<float>(dm, count * heads * 256);
    // The stored K was roped at its cache position; offsets other than 0 describe a different
    // rotation than the fixture used, so the oracle derotates by the caller-declared position.
    double maximum = 0;
    int failures = 0;
    for (int i = 0; i < count; ++i) {
        const int page = first + i;
        for (int h = 0; h < heads; ++h) {
            for (int d = 0; d < 256; ++d) {
                double expected = 0;
                if (offsets[i] != ops::kKvRetrievalNoRopeOffset) {
                    for (int t = 0; t < 64; ++t) {
                        const int position = page * 64 + t;
                        double x = cache_value(host, true, h, position, d);
                        if (d < 64) {
                            const int partner = d < 32 ? d + 32 : d - 32;
                            const double y = cache_value(host, true, h, position, partner);
                            const double angle = static_cast<double>(position + offsets[i]) *
                                                 std::pow(1e7, -2.0 * (d % 32) / 64);
                            x = x * std::cos(angle) + (d < 32 ? y : -y) * std::sin(angle);
                        }
                        expected += x / 64;
                    }
                }
                const double got = actual[(static_cast<std::size_t>(i) * heads + h) * 256 + d];
                if (offsets[i] == ops::kKvRetrievalNoRopeOffset) { failures += got != 0.0; }
                maximum = std::max(maximum, std::abs(got - expected));
            }
        }
    }
    // FP32 sincos of positions near 85K carries ~1e-3 relative angle error on the fastest pair.
    failures += maximum > 3e-4;
    failures += dm.verify_guards("page means");
    std::cout << "KVMem page Mean-K FP64 oracle H=" << heads << ' ' << cache_name(dtype)
              << " max=" << maximum << " failures=" << failures << '\n';
    return failures;
}

// Exact codes/scales survive selection-driven eviction and reload through the host archive.
int selection_roundtrip() {
    LayoutBuilder builder;
    auto layout = plan_paged_kv_pool(builder, {.page_group_count = 80,
        .logical_page_capacity = 200, .table_rows = 1,
        .planes = {{DType::I8, 256, 2}, {DType::I8, 256, 2},
                   {DType::FP16, 4, 2}, {DType::FP16, 4, 2}}, .ram_enabled = true});
    DeviceBuffer storage(builder.finish(256));
    PagedKVPool pool({storage.p, storage.bytes}, layout);
    auto allocation = pool.reserve(80);
    allocation.bind_row(0);
    const auto payload = [](int page, std::size_t plane, std::size_t bytes) {
        std::vector<unsigned char> value(bytes);
        for (std::size_t i = 0; i < bytes; ++i) { value[i] = (page * 17 + plane * 37 + i * 7) % 251; }
        return value;
    };
    int failures = 0;
    const auto verify = [&](int page) {
        const int physical = allocation.page_ids()[page];
        if (physical < 0) { ++failures; return; }
        for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
            auto expected = payload(page, plane, pool.plane(plane).nb[3]);
            std::vector<unsigned char> actual(expected.size());
            CUDA_CHECK(cudaMemcpy(actual.data(), static_cast<char*>(pool.plane(plane).data) +
                physical * actual.size(), actual.size(), cudaMemcpyDeviceToHost));
            failures += actual != expected;
        }
    };
    std::vector<std::int32_t> selected;
    for (int end = 8; end <= 200; end += 8) {
        if (end > 80) {
            // Sink 32 pages plus the 32 pages before the previous frontier; appends are pinned.
            selected.clear();
            for (int p = 0; p < 32; ++p) { selected.push_back(p); }
            for (int p = end - 48; p < end - 16; ++p) { selected.push_back(p); }
            allocation.select_pages(selected, static_cast<std::uint32_t>(end - 8));
        }
        allocation.materialize_pages(end);
        for (int p = end - 8; p < end; ++p) {
            const auto physical = allocation.page_ids()[p];
            if (physical < 0) { throw std::runtime_error("append page was evicted"); }
            for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
                auto bytes = payload(p, plane, pool.plane(plane).nb[3]);
                CUDA_CHECK(cudaMemcpy(static_cast<char*>(pool.plane(plane).data) +
                    physical * bytes.size(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
            }
        }
    }
    // Re-select archived history: every page must return byte-identical.
    selected.clear();
    for (int p = 40; p < 70; ++p) { selected.push_back(p); }
    for (int p = 120; p < 150; ++p) { selected.push_back(p); }
    allocation.select_pages(selected, 200);
    allocation.materialize_pages(200);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int p : selected) { verify(p); }
    for (int p = 0; p < 200; ++p) {
        const bool wanted = (p >= 40 && p < 70) || (p >= 120 && p < 150);
        failures += wanted != (allocation.page_ids()[p] >= 0);
    }
    // A selection larger than the window is rejected only for mandatory pages.
    allocation.select_pages({}, 100);
    bool rejected = false;
    try { allocation.materialize_pages(200); } catch (const std::logic_error&) { rejected = true; }
    failures += !rejected;
    allocation.select_pages(selected, 200);
    allocation.materialize_pages(200);
    PagedKVResize resize{&allocation, allocation.mapped_page_count(), 80};
    resize_paged_kv_bundle(std::span(&resize, 1));
    failures += allocation.mapped_token_capacity() != 200 * 64;
    allocation.trim_tokens(150 * 64 - 3);
    allocation.materialize_tokens(150 * 64 - 3);
    failures += allocation.mapped_token_capacity() != 150 * 64;
    failures += allocation.mapped_page_count() > 80;
    failures += allocation.ram_transfer_bytes() == 0;
    allocation.release();
    failures += pool.free_pages() != 80 || pool.entitled_pages() != 0;
    std::cout << "KVMem selection exact code/scale archive roundtrip: " << failures
              << " failures\n";
    return failures;
}

int sparse_attention(DType dtype, int tokens, bool cached, bool long_attention) {
    const Geometry geometry{"tp2", 12, 2};
    const int base = 1024;
    HostCache host = make_cache(geometry, dtype, base + tokens, 100);
    const auto q = make_bf16_values(256 * 12 * tokens, 20, -0.25f, 0.25f);
    const auto k = make_bf16_values(256 * 2 * tokens, 21, -0.25f, 0.25f);
    const auto v = make_bf16_values(256 * 2 * tokens, 22, -1.0f, 1.0f);
    std::vector<int> positions(tokens);
    for (int t = 0; t < tokens; ++t) { positions[t] = base + t; }
    DeviceCache device_cache(host, MappingPattern::Fragmented);
    if (!cached) { append_cache(host, k, v, positions); }
    std::vector<int> selected{0, 3, 15};
    for (int page = base / 64; page <= (base + tokens - 1) / 64; ++page) { selected.push_back(page); }
    const auto reference = ideal_attention(q, host, positions, selected);
    auto single = device_cache.view();
    std::vector<int> original(single.block_table.ne[0]);
    CUDA_CHECK(cudaMemcpy(original.data(), single.block_table.data, original.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<int> indices(original.size(), -1), table(original.size(), original.back());
    for (std::size_t i = 0; i < selected.size(); ++i) {
        indices[selected[i]] = i;
        table[i] = original[selected[i]];
    }
    GuardedDeviceBuffer di(indices.size() * 4), dt(table.size() * 4);
    di.copy_from_host(indices.data(), indices.size() * 4);
    dt.copy_from_host(table.data(), table.size() * 4);
    single.read_indices = Tensor(di.data(), DType::I32, {int(indices.size())});
    single.read_table = Tensor(dt.data(), DType::I32, {int(table.size())});
    single.read_capacity = selected.size() * 64;
    const auto qb = to_bf16_bits(q), kb = to_bf16_bits(k), vb = to_bf16_bits(v);
    GuardedDeviceBuffer dq(qb.size() * 2), dk(kb.size() * 2), dv(vb.size() * 2),
        dp(positions.size() * 4), dout(qb.size() * 2), drow(4);
    dq.copy_from_host(qb.data(), qb.size() * 2);
    dk.copy_from_host(kb.data(), kb.size() * 2);
    dv.copy_from_host(vb.data(), vb.size() * 2);
    dp.copy_from_host(positions.data(), positions.size() * 4);
    const int row = 0;
    drow.copy_from_host(&row, 4);
    Tensor tq(dq.data(), DType::BF16, {256, 12, tokens}),
        tk(dk.data(), DType::BF16, {256, 2, tokens}), tv(dv.data(), DType::BF16, {256, 2, tokens}),
        tp(dp.data(), DType::I32, {tokens}), out(dout.data(), DType::BF16, {256, 12, tokens}),
        tr(drow.data(), DType::I32, {1});
    const auto visible = static_cast<unsigned>(base + tokens);
    const ops::GqaExecutionEnvelope envelope{cached ? 1U : visible, visible, long_attention};
    const auto capacity = ops::gqa_attention_workspace_capacity_bytes(12, dtype,
        {1, visible, long_attention}, 1, tokens, tokens, true);
    GuardedDeviceBuffer scratch(capacity);
    WorkspaceArena arena({scratch.data(), capacity});
    if (cached) {
        ops::gqa_attention_cached(tq, tp, kAttentionScale, single, envelope, arena, out, nullptr);
    } else {
        auto batch = device_cache.batch_view();
        batch.read_indices = single.read_indices;
        batch.read_tables = single.read_table.view({int(table.size()), 1});
        batch.read_capacity = single.read_capacity;
        ops::gqa_attention(tq, tk, tv, tp, Tensor{}, tr, kAttentionScale, batch, envelope, arena,
                           out, nullptr);
    }
    cuda_synchronize();
    const std::string label = std::string("sparse original-position oracle ") + cache_name(dtype) +
        " T=" + std::to_string(tokens) + (cached ? " cached" : " append") +
        (long_attention ? " long" : "");
    int failures = verify_attention(label, bf16_bits_to_double(copy_from_guarded<std::uint16_t>(dout, qb.size())),
                                     reference, attention_criterion(dtype));
    if (!cached) { failures += verify_cache(label, device_cache.snapshot(), host); }
    failures += scratch.verify_guards(label.c_str());
    failures += verify_positions(label, dp, positions);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) { return 77; }
    int failures = selection_roundtrip();
    for (int heads : {2, 4}) {
        failures += query_rows_oracle(heads);
        for (auto dtype : {DType::BF16, DType::I8}) { failures += page_means_oracle(heads, dtype); }
    }
    for (auto dtype : {DType::BF16, DType::I8}) {
        failures += sparse_attention(dtype, 4, false, false);
        failures += sparse_attention(dtype, 4, true, false);
        failures += sparse_attention(dtype, 64, false, false);
    }
#ifdef NINFER_VOLTA_BUILD
    failures += sparse_attention(DType::I8, 1, true, true);
    failures += sparse_attention(DType::I8, 5, false, true);
    failures += sparse_attention(DType::I8, 512, false, false);
    failures += sparse_attention(DType::I8, 512, false, true);
#endif
    return failures == 0 ? 0 : 1;
}
