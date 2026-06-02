# PLENA Emulator Model/Config Results Matrix

Generated: `2026-05-22`  |  Last updated: `2026-06-02`

Simulator checkout:
```text
simulator repo: /home/khl22/new_plena/PLENA_Simulator
simulator branch: exp/roll-attention-head-batch
simulator HEAD: 45c02b8 (squashed from 59 commits)
compiler submodule branch: exp/roll-attention-head-batch
compiler submodule HEAD: 287f6fb
```

> **Note:** History was squashed on 2026-05-27. All prior sim commits in the
> results table now map to `45c02b8`.

> **⚠️ Lineage / staleness note (2026-06-02):** Every row dated **≤ 2026-05-26** was produced on
> the now-superseded `exp/roll-attention-head-batch` lineage (sim `45c02b8`). `main` has since
> advanced well past it — sub-64 single-batch (PLENA_Compiler #51/#54), native non-packed
> **batch_size>1 incl. true sub-64** (#55 / Simulator #79 / Tools #6), and the accelerator
> refactors (#68/#71/#73/#75). Those rows' **`Repro @HEAD?` is therefore unverified against current
> `main`** and should be treated as stale unless re-run. New rows below are pinned to their actual
> producing commits. (Marked stale rather than re-run: the older matrix is 40+ rows, several
> multi-hour on the shared box.)

Key fixes in this update (2026-05-25):
- **Golden RoPE bug fix**: golden was using X_embed (un-normed) instead of X_norm for RoPE cos term, causing 82% FAIL on SmolLM2/SmolVLM2 due to peaked-softmax amplification. Now 100% at 1L.
- **Q/O projection for mlen=256**: `linear_projection` ISA ops + W_q/W_o weight loading. All 5 decoder models pass 100% at mlen=256 through 10 layers.
- **Per-build TOML**: `HardwareConfig.write_toml(build_dir)` generates per-build plena_settings.toml with correct HBM_WIDTH, HBM_M_Prefetch_Amount, HLEN, BROADCAST_AMOUNT. Global TOML never modified.
- **HBM address alignment**: extract `hbm_addrs` from compiler to match tile-alignment gaps in HBM binary layout.
- **Step-by-step BF16 softmax**: golden attention now quantizes scores→BF16→softmax→BF16 to match emulator ISA path
- **CLI extensions**: --vlen, --num-layers, --batch-size flags; LLaDA-8B-Base added; model config YAMLs + resolve_hardware
- **batch_size threading**: decoder and chain tests support batch_size>1
- HBM alignment: `_allocate_hbm` pads to `mlen*mlen` when MLEN >= 256
- LLaDA bidirectional attention: causal mask skipped for `model_type == "llada"`

## Config axes

| Axis | Meaning | Common values |
|---|---|---|
| **Mode** | `sliced` = sliced decoder test (multi_model_decoder_test.py). `native` = compiler frontend (compile_native_hf_decoder/vision_encoder). Native can run at any mlen. | `sliced`, `native` |
| **MLEN** | Matrix tile length | 64 (default), 128, 256 |
| **VLEN** | Vector tile length | 64 (default), 256 |
| **BLEN** | Batch/vector tile length | 4 (default), 64 |
| **seq_len** | Sequence length (tokens) | 64, 256 |
| **batch_size** | Batch dimension (independent sequences) | 1, 2 |
| **golden_precision** | Reference precision for golden output | `hardware` (MXFP8+BF16), `hf_fp32` |
| **atol / rtol** | allclose tolerances for emulator pass/fail | 0.2 / 0.2 (default) |
| **min_allclose_rate** | Minimum % of elements within tolerance | 90% (default) |

## Results

All rows sorted by date descending. Allclose bar is 90% unless noted.

> **`Sim lat` column = simulated latency** (`executor.now()`, the modeled hardware time — the real
> accelerator metric). For 2026-06-01+ rows this is the emulator's `Simulation completed. Latency …`
> value; host wall time (box-load-dependent, **not** a hardware metric) is noted inline only as
> context. **Earlier rows show host *wall* time in this column** (it was labelled `Wall`) and are
> stale per the lineage note — sim latency was never captured for them because the harness ran the
> emulator at `--log-level warn`, which suppressed the INFO latency line. That harness gap is now
> fixed (`RUST_LOG=warn,transactional_emulator=info`), so `sim_latency_ns` is recorded automatically.

| Date | Sim commit | Comp commit | Mode | Model | Arch | L | MLEN | BLEN | seq | hidden | inter | batch | ISA | Sim lat | Allclose | Pass | MAE | Repro @HEAD? | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-01 | `56ed1e7`‡ | `0f55571`‡ | native | SmolVLM2-256M-text | LlamaForCausalLM | 1 | 16 | 4 | 4 | 576 | 1536 | 2 | 1.24M | 10.03ms | **99.07%** | PASS | 0.037 | YES | **true sub-64** head_dim 64>mlen 16 (4 col-blocks). Was `NotImplementedError` before #55. Multi-batch via kernel batch-loop + rpb comparison (Tools #6). Host ~17s |
| 2026-06-01 | `56ed1e7`‡ | `0f55571`‡ | native | SmolVLM2-256M-text | LlamaForCausalLM | 1 | 32 | 4 | 4 | 576 | 1536 | 2 | 1.16M | 16.42ms | **99.67%** | PASS | 0.034 | YES | true sub-64 head_dim 64>mlen 32 (2 col-blocks). Sim latency normal (cf 16/4 = 10.03ms) — **no pathology**. The original host run took 3.37h **under heavy box contention**; an identical quiet-box re-run was 106s (114× faster). Host wall is unreliable; sim latency is the metric |
| 2026-06-01 | `56ed1e7`‡ | `0f55571`‡ | native | SmolVLM2-256M-text | LlamaForCausalLM | 1 | 64 | 16 | 4 | 576 | 1536 | 2 | 148K | 2.39ms | **97.37%** | PASS | 0.052 | YES | head_dim==mlen batched path. Byte-identical ISA before/after the kernel edit (no-op proof). Was a 29% false-negative before Tools #6. Host ~33s |
| 2026-06-01 | `56ed1e7`‡ | `0f55571`‡ | native | SmolVLM2-256M-text | LlamaForCausalLM | 1 | 64 | 16 | 4 | 576 | 1536 | 1 | 73K | 1.28ms | **99.96%** | PASS | 0.031 | YES | batch=1 baseline (no regression); seq<mlen path. Host ~15s |
| 2026-05-26 | `45c02b8` | `ab4f52c` | sliced | CLM-60M | LlamaForCausalLM | 1 | 256 | 64 | 256 | 256 | 512 | 1 | — | — | **100.00%** | PASS | — | YES | GQA hq=3 hkv=1 (packed 3 Q heads). First fused GQA sliced test |
| 2026-05-26 | `45c02b8` | `ab4f52c` | sliced | SmolLM2-135M | LlamaForCausalLM | 1 | 256 | 64 | 256 | 256 | 512 | 1 | — | — | 95.69% | FAIL | — | N/A | GQA hq=3: packed-head precision sensitivity |
| 2026-05-26 | `45c02b8` | `ab4f52c` | sliced | SmolVLM2-256M-text | LlamaForCausalLM | 1 | 256 | 64 | 256 | 256 | 512 | 1 | — | — | 96.46% | FAIL | — | N/A | GQA hq=3: packed-head precision sensitivity |
| 2026-05-26 | `45c02b8` | `becef79` | native | SmolVLM2-256M-text | LlamaForCausalLM | 1 | 256 | 64 | 256 | 576 | 1536 | 1 | — | — | **99.05%** | PASS | — | YES | Overnight run. Reproduces previous 95-99% range |
| 2026-05-26 | `45c02b8` | `becef79` | native | SmolVLM2-256M-text | LlamaForCausalLM | 5 | 256 | 64 | 256 | 576 | 1536 | 1 | — | — | 96.74% | FAIL | — | N/A | Gradual multi-layer degradation (known) |
| 2026-05-26 | `45c02b8` | `becef79` | native | SmolVLM2-256M-text | LlamaForCausalLM | 10 | 256 | 64 | 256 | 576 | 1536 | 1 | — | — | 94.43% | FAIL | — | N/A | Gradual multi-layer degradation (known) |
| 2026-05-26 | `45c02b8` | `becef79` | native | SmolVLM2-256M-text | LlamaForCausalLM | 30 | 256 | 64 | 256 | 576 | 1536 | 1 | — | — | 93.79% | FAIL | — | N/A | Full model. Matches prev 93.76% |
| 2026-05-26 | `45c02b8` | `ab4f52c` | native | SmolVLM2-256M vision | SigLIP ViT | 0 | 256 | 64 | 256 | 768 | 3072 | 1 | — | — | **100.00%** | PASS | — | YES | hbm_addrs fix. Was 0.06% NaN |
| 2026-05-26 | `45c02b8` | `ab4f52c` | native | SmolVLM2-256M vision | SigLIP ViT | 1 | 256 | 64 | 256 | 768 | 3072 | 1 | — | — | **99.98%** | PASS | — | YES | hbm_addrs fix. Was 0.00% NaN |
| 2026-05-26 | `45c02b8` | `ab4f52c` | native | SmolVLM2-256M connector | Connector | 0 | 256 | 64 | 256 | — | — | 1 | — | — | **99.99%** | PASS | — | YES | hbm_addrs fix. Was 7.91% |
| 2026-05-26 | `45c02b8` | `ab4f52c` | native | SmolVLM2-256M vision | SigLIP ViT | 5 | 256 | 64 | 256 | 768 | 3072 | 1 | — | — | **99.99%** | PASS | — | YES | |
| 2026-05-26 | `45c02b8` | `ab4f52c` | native | SmolVLM2-256M vision | SigLIP ViT | 10 | 256 | 64 | 256 | 768 | 3072 | 1 | — | — | **99.48%** | PASS | — | YES | Graceful multi-layer degradation |
| 2026-05-26 | `45c02b8` | `ab4f52c` | native | SmolVLM2-256M connector | Connector | 1v | 256 | 64 | 256 | — | — | 1 | — | — | **99.74%** | PASS | — | YES | vision 1L + connector |
| 2026-05-26 | `45c02b8` | `ab4f52c` | native | SmolVLM2-256M vlm-e2e | SmolVLM | 1v+1t | 256 | 64 | 256 | — | — | 1 | — | — | **99.74%** | PASS | — | YES | hbm_addrs fix. Was 0.00% NaN |
| 2026-05-25 | `45c02b8` | `becef79` | native | SmolVLM2-256M vlm-e2e | SmolVLM | 1v+1t | 64 | 16 | 64 | 768/576 | 3072/1536 | 1 | — | — | **99.96%** | PASS | — | YES | native pipeline test at mlen=64 (vision→connector→text) |
| 2026-05-25 | `45c02b8` | `becef79` | native | SmolVLM2-256M vision | SigLIP ViT | 1 | 64 | 16 | 64 | 768 | 3072 | 1 | — | — | **100.00%** | PASS | — | YES | native pipeline test at mlen=64 (vision-layers) |
| 2026-05-25 | `45c02b8` | `becef79` | native | SmolVLM2-256M connector | Connector | 0 | 64 | 16 | 64 | — | — | 1 | — | — | **100.00%** | PASS | — | YES | native pipeline test at mlen=64 (connector-only) |
| 2026-05-25 | `45c02b8` | `becef79` | sliced | SmolLM2-135M | LlamaForCausalLM | 10 | 256 | 64 | 256 | 256 | 512 | 1 | — | — | **100.00%** | PASS | — | YES | Q/O projection (head_dim=64). Per-build TOML + hbm_addrs |
| 2026-05-25 | `45c02b8` | `becef79` | sliced | SmolVLM2-256M-text | LlamaForCausalLM | 10 | 256 | 64 | 256 | 256 | 512 | 1 | — | — | **100.00%** | PASS | — | YES | Q/O projection (head_dim=64) |
| 2026-05-25 | `45c02b8` | `becef79` | sliced | CLM-60M | LlamaForCausalLM | 10 | 256 | 64 | 256 | 256 | 512 | 1 | — | — | **100.00%** | PASS | — | YES | Q/O projection (head_dim=64) |
| 2026-05-25 | `45c02b8` | `becef79` | sliced | LLaDA-8B-Instruct | MaskedDiffusion | 10 | 256 | 64 | 256 | 256 | 512 | 1 | — | — | **100.00%** | PASS | — | YES | Q/O projection (head_dim=128) |
| 2026-05-25 | `45c02b8` | `becef79` | sliced | LLaDA-8B-Base | MaskedDiffusion | 10 | 256 | 64 | 256 | 256 | 512 | 1 | — | — | **100.00%** | PASS | — | YES | Q/O projection (head_dim=128) |
| 2026-05-25 | `45c02b8` | `becef79` | sliced | LLaDA-8B-Instruct | MaskedDiffusion | 1 | 128 | 64 | 128 | 128 | 256 | 1 | — | — | **100.00%** | PASS | — | YES | HBM_M_Prefetch_Amount fix |
| 2026-05-25 | `45c02b8` | `becef79` | sliced | SmolLM2-135M | LlamaForCausalLM | 1 | 64 | 16 | 64 | 64 | 128 | 1,2 | — | — | **100.00%** | PASS | 0.007 | YES | Golden RoPE fix. b1+b2 pass |
| 2026-05-25 | `45c02b8` | `becef79` | sliced | SmolVLM2-256M-text | LlamaForCausalLM | 1 | 64 | 16 | 64 | 64 | 128 | 1,2 | — | — | **100.00%** | PASS | 0.007 | YES | Golden RoPE fix. b1+b2 pass |
| 2026-05-25 | `45c02b8` | `becef79` | sliced | CLM-60M | LlamaForCausalLM | 10 | 64 | 16 | 64 | 64 | 128 | 1,2 | — | — | **100.00%** | PASS | — | YES | 1L/5L/10L b1+b2 all pass |
| 2026-05-25 | `45c02b8` | `becef79` | sliced | LLaDA-8B-Instruct | MaskedDiffusion | 10 | 64 | 16 | 64 | 64 | 128 | 1,2 | — | — | **100.00%** | PASS | — | YES | 1L/5L/10L b1+b2 all pass |
| 2026-05-25 | `45c02b8` | `becef79` | sliced | LLaDA-8B-Base | MaskedDiffusion | 10 | 64 | 16 | 64 | 64 | 128 | 1,2 | — | — | **100.00%** | PASS | — | YES | 1L/5L/10L b1+b2 all pass |
| 2026-05-25 | `45c02b8` | `becef79` | sliced | SmolLM2-135M | LlamaForCausalLM | 5 | 64 | 16 | 64 | 64 | 128 | 1,2 | — | — | 81.27% | FAIL | — | N/A | Peaked-softmax accumulation (known: native 30L=94%) |
| 2026-05-25 | `45c02b8` | `becef79` | sliced | SmolVLM2-256M-text | LlamaForCausalLM | 5 | 64 | 16 | 64 | 64 | 128 | 1,2 | — | — | 80.91% | FAIL | — | N/A | Peaked-softmax accumulation (known: native 30L=94%) |
| 2026-05-25 | `45c02b8` | `becef79` | native | SmolVLM2-256M-text | LlamaForCausalLM | 1 | 256 | 64 | 256 | 576 | 1536 | 1 | — | — | **95.89%** | PASS | — | YES | HBM address alignment fix. Was 13.21% before fix |
| 2026-05-25 | `45c02b8` | `4bf353d` | native | LLaDA-8B-Instruct | MaskedDiffusion | 1 | 128 | 64 | 256 | 4096 | 12288 | 1 | 817K | ~46m | **92.29%** | PASS | 0.098 | ? | Bidirectional fix + HBM conditional alignment |
| 2026-05-25 | `45c02b8` | `4bf353d` | sliced | LLaDA-8B-Instruct | MaskedDiffusion | 0 | 128 | 64 | 64 | 128 | 256 | 1 | 4,687 | 3.9s | **100.00%** | PASS | 1.27e-4 | REGRESSED | Rust emu crash at old commit |
| 2026-05-25 | `45c02b8` | `4bf353d` | sliced | SmolLM2-135M | LlamaForCausalLM | 0 | 64 | 4 | 64 | 64 | 128 | 1 | 2,455 | ~4s | **82.52%** | FAIL | 0.188 | FIXED | Golden RoPE bug — now 100% at `45c02b8` |
| 2026-05-22 | `45c02b8` | `1ab4dc9` | native | SmolVLM2-256M-text | LlamaForCausalLM | 30 | 256 | 64 | 256 | 576 | 1536 | 1 | — | 3.2h | 93.76% | PASS | 0.076 | ? | Max err 21.0 |
| 2026-05-22 | `45c02b8` | `1ab4dc9` | native | SmolVLM2-256M-text | LlamaForCausalLM | 5 | 256 | 64 | 256 | 576 | 1536 | 1 | — | 814s | 96.85% | PASS | 0.045 | ? | Max err 15.6 |
| 2026-05-22 | `45c02b8` | `1ab4dc9` | native | SmolVLM2-256M-text | LlamaForCausalLM | 3 | 256 | 64 | 256 | 576 | 1536 | 1 | — | ~350s | 98.53% | PASS | — | ? | Fails 99% bar |
| 2026-05-22 | `45c02b8` | `1ab4dc9` | native | SmolVLM2-256M-text | LlamaForCausalLM | 1 | 256 | 64 | 256 | 576 | 1536 | 1 | 49.7K | 190s | 99.13% | PASS | — | YES (95.89%) | Drops to 95.89% at HEAD with HBM padding fix — still PASS |
| 2026-05-22 | `45c02b8` | `1ab4dc9` | native | LLaDA-8B-Instruct | MaskedDiffusion | 1 | 256 | 64 | 256 | 4096 | 12288 | 1 | 478K | ~190s | 99.13%* | PASS | — | ? | Pre-fix; coincidentally aligned HBM addresses |
| 2026-05-22 | `45c02b8` | `1ab4dc9` | native | LLaDA-8B-Base | MaskedDiffusion | 1 | 256 | 64 | 256 | 4096 | 12288 | 1 | 461K | — | — | CRASH | — | N/A | exit 101: HBM alignment assertion (pre-fix) |
| 2026-05-21 | `45c02b8` | `1ab4dc9` | sliced | SmolLM2-135M | LlamaForCausalLM | 0 | 64 | 4 | 64 | 64 | 128 | 1 | 2,455 | ~160s | ✓ | PASS | — | FIXED | Was golden RoPE bug (82.52%). Fixed at `45c02b8` → 100% |
| 2026-05-21 | `45c02b8` | `1ab4dc9` | sliced | LLaDA-8B-Instruct | MaskedDiffusion | 0 | 64 | 4 | 64 | 64 | 128 | 1 | 2,356 | 158.8s | 100% | PASS | 1.5e-5 | YES | partial_load, trust_remote_code |
| 2026-05-21 | `45c02b8` | `1ab4dc9` | sliced | SmolVLM2-256M-text | SmolVLM | 0 | 64 | 4 | 64 | 64 | 128 | 1 | 2,455 | 162.7s | ~0% | FAIL | 0.17 | N/A | AutoModel fallback; emu diverges |
| 2026-05-21 | `45c02b8` | `1ab4dc9` | sliced | SmolVLM2-256M-vision | SigLIP ViT | 0 | 64 | 4 | 64 | 64 | 128 | 1 | — | 165.3s | 99.98% | PASS | 0.026 | YES | Synthetic conv2d |
| 2026-05-22 | `45c02b8` | `1ab4dc9` | native† | LLaDA-8B-Instruct | MaskedDiffusion | 1 | 256 | 64 | 256 | 4096 | 12288 | 1 | 478K | 286s | — | — | 0.150 | N/A | Golden only; no emulator |
| 2026-05-22 | `45c02b8` | `1ab4dc9` | native† | SmolVLM2-256M | SmolVLM | 1 | 256 | 64 | 256 | 576 | 1536 | 1 | 49.7K | 38s | — | — | 0.151 | N/A | Golden only; no emulator |
| 2026-05-22 | `45c02b8` | `1ab4dc9` | native† | SmolVLM2-256M-vision | SigLIP ViT | 1 | 256 | 64 | 256 | 768 | 3072 | 1 | 1.07M | 71s | — | — | 0.231 | N/A | Golden only; no emulator |

† Golden-only compile — no emulator run.

‡ Multi-batch rows were measured on the content-equivalent local commits (compiler `069930b`,
sim `2ac6b71` on `fix/vision-smolvlm2`); `program_attention.py` / `plena_frontend.py` /
`emulator_runner.py` are byte-identical to the cited PR commits (compiler #55 `0f55571`, sim #79
`56ed1e7`, tools #6 `a6ac9e3`), so the results reproduce there. `Sim lat` = simulated latency
(`executor.now()`) read back from the emulator's `Simulation completed. Latency …ns` line, recovered
by re-running the emulator on the saved build artifacts (the original runs were at `--log-level
warn`, which suppressed it). Inline "Host ~Ns" is uncontended host wall time, for context only.

## How to run tests

### Sliced

```bash
cd /home/khl22/new_plena/PLENA_Simulator
python3 transactional_emulator/testbench/models/multi_model_decoder_test.py <model_key> --mlen <N> --blen <N>
```
Supported model keys: `smollm2-135m`, `llada-8b`, `clm-60m`

### Native

```python
from compiler.aten.plena_frontend import compile_native_hf_decoder
from transformers import AutoModel

model = AutoModel.from_pretrained("<model_id>", trust_remote_code=True, torch_dtype=torch.bfloat16)
result = compile_native_hf_decoder(
    model, seq_len=64, batch_size=1, num_layers=1,
    mlen=256, blen=64, golden_precision="hardware", reference_backend="scheduled",
)
```

### RTL

```bash
source /home/khl22/miniconda3/etc/profile.d/conda.sh && conda activate plena
cd /home/khl22/new_plena/PLENA_RTL
just test-correctness  # SimTop correctness
just test-dfc          # data_flow_control unit test
```

## Known constraints

- `seq_len` >= `MLEN` unless `PLENA_PAD_SEQ_TO_MLEN=1` is set
- `broadcast_amount >= GQA ratio`, `hlen * broadcast == MLEN`, `hlen >= head_dim`
- Full model weights must be loaded (~16 GB for LLaDA-8B)
- Non-standard HF configs (LLaDA `mlp_hidden_size`, `n_kv_heads`) handled in `model_extract.py`
- Vision encoder `seq_len` must be a perfect square

## Supported Models Reference

| Model Key | HF Model ID | Hidden | Inter | Heads | KV Heads | Head Dim | Vocab | Notes |
|---|---|---|---|---|---|---|---|---|
| `smollm2-135m` | HuggingFaceTB/SmolLM2-135M | 576 | 1536 | 9 | 3 | 64 | 49152 | |
| `llada-8b` | GSAI-ML/LLaDA-8B-Instruct | 4096 | 12288 | 32 | 32 | 128 | 126464 | MaskedDiffusion; bidirectional attention; non-standard config fields |
| `clm-60m` | AICrossSim/clm-60m | 384 | 1408 | 6 | 2 | 64 | ? | |
| `smolvlm2` | HuggingFaceTB/SmolVLM2-256M-Instruct | 576 | 1536 | 9 | 3 | 64 | 49152 | VLM: vision encoder + text decoder |

---

*Add new rows to the Results table. Each row must include Date, Sim commit, and Comp commit.*
