# KVMem and PP2 experiments

These startup-fixed experimental routes are separate from the recommended complete-history TP2 configuration. KVMem is approximate attention; PP2 uses serial two-stage decode. See [CLI](cli.md) for options and [paged KV contracts](maintainer/paged-kv-cache.md#15-optional-kvmem-experiment) for retrieval semantics.

## KVMem: 512K and 1M context

Measured on 2026-11-11 on the remote two V100-SXM2-16GB TP2 host, CUDA 12.8 / SM70, with the
QUASAR-derived Coder390 v3 NVFP4 artifact, INT8 group-64 KV, YaRN (factor 4) at 1048576
logical context, long attention, MTP and CUDA Graphs. KVMem budget 131072 + generation reserve
2048 gives a 133120-token GPU window; the host archive limit is 37024 MiB. KVMem is approximate
attention, including during long prefill.

The probe is a synthetic code archive with three one-line secret-code notes at 10%, 50% and 90%
depth, greedy sampling, 4096 output tokens. "Question separate" sends the archive and the question
as two user turns, so the retrieval query is the question alone.

| Prompt tokens | Prompt shape | Prefill tok/s | TTFT s | Committed decode tok/s | MTP accept | Codes recalled |
|---:|---|---:|---:|---:|---:|---:|
| 98528 | question appended (window not exceeded) | 926.3 | 106.8 | 67.8 | 73.6% | 3/3 |
| 197543 | question appended | 726.6 | 272.7 | 70.7 | 76.5% | 3/3 |
| 487384 | question separate | 628.6 | 777.4 | 56.6 | 68.2% | 1/3 |
| 990965 | question separate | 582.5 | 1705.5 | 55.6 | 68.2% | 1/3 |

A repeated 487362-token request restores its turn checkpoint: TTFT 2.2–4.5 s and 70.8–79.5 tok/s
decode. Peak server RSS at 1M is about 36.7 GB. GPU use is about 14.5 GB per card at any depth.
With `--vision`, a 158432-token request with one image before 158K tokens of code exceeds the
window and still answers the image's color correctly (TTFT 202.9 s); media pages stay resident.
Missed codes are not a scoring or reload fault. In a 197554-token diagnosis with budget 32768,
the three needle pages ranked first to third and were all resident at query replay. Archive
reload was byte-exact, and the sparse attention routes pass the FP64 subset oracle. The model
still reported only one code, and which one varied, whether that page was reloaded or never evicted. Keeping
the 10% page resident throughout prefill (`--kvmem-sink-tokens 20480`) recovered it. Pressure
prefill keeps only sink plus newest tail, as upstream KVMem does. Later context therefore
never attends to an evicted needle, and recall falls as budget/context shrinks. These results
describe this probe, not general long-context quality.

## Experimental PP2 layer pipeline

Measured on 2026-10-07 on the local two V100-SXM2-16GB cards, CUDA 12.8 / SM70,
PCIe 3.0 x16 PHB, qualified PCIe P2P and no NVLink. The artifact is
`qwen3.8-27b/nvfp4-full`, at
`/Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer`. This is a startup-fixed
alternative to TP2, not an automatic long-input switch or PP-prefill/TP-decode hybrid.

`--tp 1 --pp 2 --devices 0,1` places layers 0..31 on device 1 and layers 32..63,
final norm, output head and MTP on device 0. Each stage retains complete head/channel
geometry and owns its half of the text KV/GDN layers. `--pipeline-chunks 8` submits
up to eight chronological 1024-token chunks per wave. Source retirement waits only
for the residual copy, permitting the next first-stage chunk to proceed during the
previous second-stage chunk. `--pipeline-chunks 1` is the serialized control.
Decode, including MTP target verification, traverses both stages serially.

The matched sweep fixes capacity at 180000 (180032 allocated), complete INT8 group-64
KV, greedy sampling, CUDA Graphs and a 1024-token prefill chunk. None and optimized
MTP3 each run 512/3072/8192/16384/32768/65536/85000 actual input tokens. All three
layouts use the same saved code-task IDs at each occupancy, with two cold requests,
no prefix reuse or extra request warmup, one prefill output and 256 committed timed
decode outputs. Loading and graph priming are excluded. Active batches have at most
480 seconds before a three-minute cooldown, with an early rest when the next case
would exceed that window; cooldowns are excluded from request measurements.

The PP2 real-model gate checks exact graph/eager commits and sampled routing; all
64 teacher-forced positions agree with the ordinary argmax, with zero emitted-logit
deficit. The prefix gate passes repeated restore, append, checkpoint, partial-stop
and exact-frontier state controls. One cached/cold continuation diverges at output
position 4 with a 0.125 logit deficit; cold re-prefill is not bit-identical. The
24-layer, 48-value-head GDN replay geometry passes its independent recurrent-state
oracle. These checks do not establish source-checkpoint task accuracy or universal
TP2/PP2 token parity. In the occupancy sweep the two layouts can produce different
output IDs and MTP acceptance rates, so cross-layout results are descriptive, not
strict same-output or quality-equivalence claims.

PP2 supports one Text request, None/MTP and prefix reuse on SM70 Qwen3.8 NVFP4/QUASAR.
KVMem, Vision, DFlash, expert storage and concurrent requests are rejected. TP2 remains recommended for decode; PP2 cannot switch back to TP2 at the prefill boundary.

Reproduce the saved-input sweep with the selected Python 3.11 environment:

```bash
LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64" \
  .venv/bin/python tools/v100/bench_pipeline.py \
  --weights /Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
  --bench build-v100-tp4/bench/ninfer_bench \
  --output-dir profiles/bench/pipeline/complete --corpus-dir profiles/bench \
  --backends tp2 pp2-serial pp2 \
  --prompts 512 3072 8192 16384 32768 65536 85000 --drafts 0 3 \
  --capacity 180000 --prefill-chunk 1024 --output-tokens 256 \
  --repetitions 2 --active-seconds 480 --cooldown 180
```

The script saves individual commands/reports, request phases, per-card memory, repetition
checks and PP2 wave/serial equivalence under `profiles/bench/pipeline/complete/`.
Completed cases are reused only if the saved command matches. Raw reports and corpora
are local measurement prerequisites, not committed assets.
