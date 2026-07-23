"""Time to hand the KV cache from the prefill chip to the decode chip.

The prefill chip builds the prompt's KV cache and sends it to the decode chip
before generation starts. It writes KV in the decode chip's KV format, so fewer
KV bits means fewer bytes on the wire — KV quantization saves interconnect time
as well as HBM bandwidth and capacity.

Two cases bound the cost added to the decode chip's time-to-first-token (TTFT):

  streamed — each layer is sent while prefill computes the next, so the transfer
             hides behind prefill and adds only ~one layer's KV time.
  bulk     — the whole cache is sent after prefill finishes, so the full
             transfer time is added.

Link presets are per-direction bandwidths of NVLink-/UALink-class fabrics.
"""

from __future__ import annotations

from dataclasses import dataclass

# Aggregate one-direction link bandwidth, bytes/s.
LINK_GENS = {
    "nvlink3": 300e9,   # A100-class: 600 GB/s bidirectional
    "nvlink4": 450e9,   # H100-class: 900 GB/s bidirectional
    "ualink":  400e9,   # UALink-class open-standard target
    "pcie5":   64e9,    # x16 PCIe 5.0, the pessimistic floor
}


@dataclass(frozen=True)
class HandoffTime:
    kv_bytes: float          # bytes on the wire, at the decode chip's KV precision
    link_bw: float           # link bandwidth, bytes/s
    bulk_s: float            # full transfer time (TTFT add in bulk mode)
    streamed_s: float        # per-layer time (TTFT add in streamed mode)

    def ttft_add(self, mode: str) -> float:
        return self.bulk_s if mode == "bulk" else self.streamed_s


def kv_wire_bytes(dims: dict, prec: dict, input_seq: int, batch: int) -> float:
    """Total KV bytes for the prompt at the decode chip's effective KV bits:
    2 (K and V) x kv_heads x head_dim x input_seq x batch x layers."""
    per_layer = 2 * dims["kv_heads"] * dims["head_dim"] * input_seq * batch
    return per_layer * dims["layers"] * prec["kv_bits"] / 8


def handoff_time(
    dims: dict,
    prec: dict,
    input_seq: int,
    batch: int,
    link_gen: str = "nvlink4",
    link_bw: float | None = None,
) -> HandoffTime:
    bw = float(link_bw) if link_bw else LINK_GENS[link_gen]
    total = kv_wire_bytes(dims, prec, input_seq, batch)
    return HandoffTime(
        kv_bytes=total,
        link_bw=bw,
        bulk_s=total / bw,
        streamed_s=(total / dims["layers"]) / bw,
    )


def report(dims: dict, prec: dict, input_seq: int, batch: int,
           link_gen: str = "nvlink4", link_bw: float | None = None) -> str:
    h = handoff_time(dims, prec, input_seq, batch, link_gen, link_bw)
    gb = h.kv_bytes / 1e9
    return (
        f"      KV wire bytes:       {gb:.3f} GB "
        f"(prompt {input_seq} x batch {batch} @ {prec['kv_bits']:.2f} eff bits)\n"
        f"      Link:                {link_gen} = {h.link_bw/1e9:.0f} GB/s per direction\n"
        f"      TTFT add, bulk:      {h.bulk_s*1e3:.3f} ms  (whole cache after prefill)\n"
        f"      TTFT add, streamed:  {h.streamed_s*1e3:.3f} ms  (layer-wise, hidden behind prefill)"
    )
