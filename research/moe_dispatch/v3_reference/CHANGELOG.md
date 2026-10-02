# v3_reference CHANGELOG

## v3.1 (review fixes, before hand-off)

- Formats: `Fmt.factor_a` added (A format separate from B). v3 default `FMT_V3` = MXINT4 main, A MXINT4, B BF16, L=8
  (routed 4,970,112 B = BF16/3.481; Shared 9,889,920 B). `mxint8s` = MXINT8 split into two INT4 passes (prepass x2).
- Issue counts: prepass counts A passes; rank tails are lane-only issues (ceil(r_tail/L) per output tile).
- Stream core: Me <= 4 hard cap (cost = inf above); W read once from the pool and reused across nb M blocks; P1 decodes once.
- Placement in `predict_layer` only considers cores with finite cost.
- Workloads: `zipf_me`; mixed T=64 / T=96 cases; bfcl_b16 Me list matches captured distinct-expert and issue counts.
- 3+3 iso-port: G=4, X port 320 B/cycle per core.
- Storage: route state (T*k*16 + E*64 B), FP32 partial U_d, 16 KiB control reserve, pool = 128 banks x 512 B.
  Result: P1 only up to T=96; T=128 only P2 with L=8 (slack 35.7 KiB).
- Pool lower bound margin 64 cycles (largest barrier 52); BW=512 needs 72.5 KiB -> 128 KiB pool, T <= 96.
- Ports: pool aggregate, activation store X/Z/U and aggregate, combine rows.
- Offload cross-core traffic includes Z (U_d = Z A_d): 25.8 KiB at Me=1, 412 KiB at Me=16.
- Timeline: A_d of Z segments 0..S-2 interleaved into the Gate/Up stream on ws_group; barrier = residual SiLU
  (dense Me*16/64; stream Me*I/64) followed by the remaining A_d tiles.
- ref_numerics: `mxint8_split` (codes +-119, hi in [-7,7] at e+4, lo in [-8,7] at e) and `quantize_b_per_segment`.
- Tests: 17 (new: v3 default bytes, stream cap, storage rules, MXINT8 split exactness, per-segment B blocks).

## 2026-10-02 — Hardware numerical contract (§3.8)

Added expert_ffn_hw, combine_hw and fused_rank_lane_gemm_hw. Previously only algorithm-order expert_ffn and generic fused GEMM were supplied. Gold now specifies BF16 X/U/Z storage; ascending-K FP32 segment accumulation; compensation before SiLU; FP32 gate folding before BF16 Z; streamed Down correction after all U_d input slices with explicitly charged lane-only tails; and FP32 scatter merge in externally supplied completion order. Existing reference functions and budget figures are preserved.

## 2026-10-02 — Streamed Down physical-storage correction

WS A prepasses use rank groups of at most 32 columns outside ascending K segments;
each completed group drains to BF16 U. Streamed WS Down cannot retain Me-by-H
FP32 Y in its small private RF. Each main K segment and lane-only rank tail
instead drains an N-group delta atomically into the existing shared Combine
buffer. Added down_partial_contributions_hw and combine_partial_hw to reconstruct
those decoded-operand deltas and replay ordered FP32 atomic drains, with exact
coverage/ownership checks. Host-side gold matrices are verification artifacts,
not extra modeled SRAM. expert_ffn_hw remains the Q1 whole-expert contract.
