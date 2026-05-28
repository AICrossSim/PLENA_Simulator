"""FULL single_stream_block via the graph driver + HBM-bin relay.

Data flow (single_stream_block_graph), simplified for segmented execution
(no head-offset / alias tricks from the old SSB system — every tensor gets its
own address; weights reuse one region; concat is a real concat_min step):

  X --layernorm(LN_W)--> LN_Y --modulate(MOD_W)--> MOD_Y
  MOD_Y --linear_q(Wq)--> Q   --qknorm_q(QKN_W)--> QKN_Q --rope_q(ROPE_W)--> ROPE_Q
  MOD_Y --linear_k(Wk)--> K   --qknorm_k(QKN_W)--> QKN_K --rope_k(ROPE_W)--> ROPE_K
  MOD_Y --linear_v(Wv)--> V
  MOD_Y --linear_mlp(Wm)--> MLP --gelu--> GELU_OUT
  flash_attention(ROPE_Q, ROPE_K, V) --> ATTN_OUT
  concat(ATTN_OUT, GELU_OUT) --> CONCAT
  linear2(CONCAT, W2) --> LIN2_OUT
  residual_gate(X, GATE_W, LIN2_OUT) --> BLOCK_OUT

Every HBM hop MX-E4M3 round-trips. Threshold 0.8 (deep chain accumulates MX
quant noise hop by hop).

Run: tools/manager/run.sh _validate_block
"""

import sys
import math
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "transactional_emulator" / "testbench"))

from manager.geometry import load_behavior_settings
from manager.pipeline import Manager
from _tb_cache import mx_roundtrip as mxr


def main() -> int:
    import datetime as _dt
    _start = _dt.datetime.now()
    s = load_behavior_settings()
    MLEN, HLEN = s.mlen, s.hlen
    # Everything geometry-derived from the toml (s.mlen / s.hlen). NOTHING
    # hardcoded: changing plena_settings.toml to hlen=128/mlen=512 flows through
    # here automatically. HEAD = lane_count*2 = (mlen//hlen)*2 (matches the
    # flash_attention test's HEAD_COUNT = (MLEN//HLEN)*2). NSB (seq blocks) is a
    # test knob, not hardware — kept at 2.
    NSB = 4                            # seq blocks: S = NSB*MLEN = 4*1024 = 4096
    HEAD = s.hardware_lane_count * 2   # mlen//hlen * 2 = 16 at mlen=1024/hlen=128
    S = NSB * MLEN
    HD = HEAD * HLEN
    EPS = 1e-6
    HALF = HLEN // 2

    K = "tilelang_tvm_compiler.kernels."
    LN, MOD = K+"layernorm_min:make_layernorm_min", K+"modulate_min:make_modulate_min"
    LIN, GELU = K+"linear_min:make_linear_min", K+"gelu_min:make_gelu_min"
    RMS, ROPE = K+"rmsnorm_min:make_rmsnorm_min", K+"rope_min:make_rope_min"
    FA, CC = K+"flash_attention_min:make_flash_attention_min", K+"concat_min:make_concat_min"
    RG = K+"residual_gate_min:make_residual_gate_min"

    # views (all 128x128 flat unless noted)
    hd1 = [1, S, 1, HD]            # head-packed (1,S,1,128): layernorm/linear/rope/concat-input
    bshd = [1, S, HEAD, HLEN]      # (1,S,8,16): modulate/qknorm/attn/gelu/residual
    wKN = [1, HD, 1, HD]           # linear B (N,K)
    biasMN = [1, S, 1, HD]
    cat = [1, S, 1, 2*HD]          # concat output feature=256
    cat_bshd = [1, S, 2*HEAD, HLEN]

    lin_kw = {"m_blocks": S//MLEN, "n_blocks": HD//MLEN, "k_blocks": HD//MLEN, "with_bias": True}
    norm_geo = {"rows": MLEN, "hlen": HLEN, "head_count": HEAD, "num_s_blocks": NSB, "batch": 1}
    gelu_kw = dict(norm_geo)
    rope_kw = {"rows": MLEN, "hlen": HLEN, "head_count": HEAD, "half_dim": HALF,
               "num_s_blocks": NSB, "batch": 1}
    fa_kw = {"rows": MLEN, "hlen": HLEN, "head_count": HEAD, "active_lane": 2,
             "num_kv_blocks": NSB, "num_q_blocks": NSB}
    cc_kw = {"rows": MLEN, "a_dim": HD, "b_dim": HD, "num_s_blocks": NSB, "batch": 1}
    lin2_kw = {"m_blocks": S//MLEN, "n_blocks": HD//MLEN, "k_blocks": (2*HD)//MLEN, "with_bias": True}

    torch.manual_seed(0)

    # ---------------- weights ----------------
    def W(): return torch.randn(HD, HD) * 0.25
    def Wb(): return torch.randn(HD) * 0.1
    Wq, bq, Wk, bk, Wv, bv, Wm, bm = W(), Wb(), W(), Wb(), W(), Wb(), W(), Wb()
    ln_sc = torch.randn(HD)*0.3+1.0; ln_bi = torch.randn(HD)*0.1
    mod_sc = torch.randn(1, S, HEAD, HLEN)*0.3; mod_sh = torch.randn(1, S, HEAD, HLEN)*0.3
    qkn_sc = torch.randn(HLEN)*0.3+1.0       # rmsnorm scale (HLEN,)
    gate = torch.randn(1, S, HEAD, HLEN)*0.3
    W2 = torch.randn(HD, 2*HD)*0.25; b2 = torch.randn(HD)*0.1

    # rope tables (built like the rope test)
    pos = torch.arange(S, dtype=torch.float32).view(1, S, 1, 1)
    dim = torch.arange(HALF, dtype=torch.float32).view(1, 1, 1, HALF)
    theta = pos * torch.pow(10000.0, -2.0*dim/HLEN)
    cos_h, sin_h = torch.cos(theta), torch.sin(theta)
    cos_f = torch.repeat_interleave(cos_h, 2, -1).expand(1, S, HEAD, HLEN)
    sin_f = torch.repeat_interleave(sin_h, 2, -1).expand(1, S, HEAD, HLEN)
    even = (torch.arange(HLEN) % 2 == 0).view(1, 1, 1, HLEN)
    sgn_f = torch.where(even, -sin_f, sin_f)
    cos_hd = cos_f.reshape(1, S, 1, HD).contiguous()
    sgn_hd = sgn_f.reshape(1, S, 1, HD).contiguous()
    Pm = torch.zeros(MLEN, MLEN); idx = torch.arange(MLEN); Pm[idx, idx ^ 1] = 1.0
    P_hbm = Pm.view(1, MLEN, 1, MLEN)

    # ---------------- golden chain (MX roundtrip per hop) ----------------
    x = torch.randn(1, S, 1, HD)*0.5

    def ling(a_eff_2d, Wm_, b_):  # (M,K) @ (N,K)^T + bias
        return mxr(a_eff_2d @ mxr(Wm_).T + mxr(b_.view(1, HD).expand(S, HD).contiguous()))

    # layernorm
    xe = mxr(x); sce = mxr(ln_sc.view(1,1,1,HD).expand(1,S,1,HD).contiguous())
    bie = mxr(ln_bi.view(1,1,1,HD).expand(1,S,1,HD).contiguous())
    mu = xe.mean(-1, keepdim=True); xc = xe-mu; var = (xc*xc).mean(-1, keepdim=True)
    LN_Y = mxr(xc*torch.rsqrt(var+EPS)*sce+bie)            # (1,S,1,HD)
    # modulate (reshape LN_Y to bshd)
    s1p = mxr((1.0+mod_sc)); shf = mxr(mod_sh)
    MOD_Y = mxr(s1p*mxr(LN_Y.reshape(1,S,HEAD,HLEN))+shf)  # (1,S,8,16)
    mod_2d = mxr(MOD_Y.reshape(S, HD))
    # linears
    Q = ling(mod_2d, Wq, bq); Kk = ling(mod_2d, Wk, bk)
    V = ling(mod_2d, Wv, bv); MLP = ling(mod_2d, Wm, bm)   # each (S,HD)
    # qknorm (rmsnorm per head over hlen) on Q,K
    def rms(t2d):
        t = mxr(t2d.reshape(1, S, HEAD, HLEN)); scq = mxr(qkn_sc.view(1,1,1,HLEN).expand(1,S,HEAD,HLEN).contiguous())
        msq = (t*t).mean(-1, keepdim=True)
        return mxr((t*scq)*torch.rsqrt(msq+EPS))           # (1,S,8,16)
    QKN_Q = rms(Q); QKN_K = rms(Kk)
    # rope (shuffle-matrix) on QKN_Q, QKN_K
    def ropeg(qkn):
        x2d = mxr(qkn.reshape(S, HD)); p2d = mxr(P_hbm.reshape(MLEN, MLEN))
        nb = HD//MLEN
        xs = torch.cat([x2d[:, b*MLEN:(b+1)*MLEN] @ p2d for b in range(nb)], -1)
        return mxr(x2d*mxr(cos_hd.reshape(S,HD))+xs*mxr(sgn_hd.reshape(S,HD)))  # (S,HD)
    ROPE_Q = ropeg(QKN_Q); ROPE_K = ropeg(QKN_K)
    # flash attention (ROPE_Q, ROPE_K, V) per head
    rq = mxr(ROPE_Q.reshape(1,S,HEAD,HLEN)); rk = mxr(ROPE_K.reshape(1,S,HEAD,HLEN)); vv = mxr(V.reshape(1,S,HEAD,HLEN))
    scale = 1.0/math.sqrt(HLEN); attn = torch.empty(1,S,HEAD,HLEN)
    for h in range(HEAD):
        sc = (rq[0,:,h,:] @ rk[0,:,h,:].T)*scale
        attn[0,:,h,:] = torch.softmax(sc, -1) @ vv[0,:,h,:]
    ATTN = mxr(attn)                                       # (1,S,8,16)
    # gelu on MLP  (GELU_g: golden tensor — NOT the kernel-spec constant GELU)
    GELU_g = mxr(torch.nn.functional.gelu(mxr(MLP.reshape(1,S,HEAD,HLEN)), approximate="tanh"))
    # concat(ATTN, GELU) along feature -> (1,S,1,256)
    CONCAT_g = torch.cat([mxr(ATTN).reshape(S,HD), mxr(GELU_g).reshape(S,HD)], -1)  # (S,256)
    CONCAT_g = mxr(CONCAT_g)
    # linear2: (S,256)@(HD,256)^T + b2
    LIN2_g = mxr(mxr(CONCAT_g) @ mxr(W2).T + mxr(b2.view(1,HD).expand(S,HD).contiguous()))  # (S,HD)
    # residual_gate: X + gate*LIN2
    BLOCK = mxr(mxr(x.reshape(1,S,HEAD,HLEN)) + mxr(gate)*mxr(LIN2_g.reshape(1,S,HEAD,HLEN)))

    # ---------------- graph ----------------
    def t(shape, role): return {"shape": shape, "role": role}
    tensors = {
        "X": t(hd1, "io"),
        "LN_W_S": t(hd1, "weight"), "LN_W_B": t(hd1, "weight"),
        "LN_Y": t(bshd, "activation"),
        "MOD_S": t(bshd, "weight"), "MOD_H": t(bshd, "weight"),
        "MOD_Y": t(hd1, "activation"),
        "WQ": t(wKN,"weight"),"BQ": t(biasMN,"weight"),
        "WK": t(wKN,"weight"),"BK": t(biasMN,"weight"),
        "WV": t(wKN,"weight"),"BV": t(biasMN,"weight"),
        "WM": t(wKN,"weight"),"BM": t(biasMN,"weight"),
        "Q": t(bshd,"activation"),"Kk": t(bshd,"activation"),
        "V": t(bshd,"activation"),"MLP": t(bshd,"activation"),
        "QKN_W": t(bshd,"weight"),
        "QKN_Q": t(hd1,"activation"),"QKN_K": t(hd1,"activation"),
        "COS": t(hd1,"weight"),"SGN": t(hd1,"weight"),"P": t([1,MLEN,1,MLEN],"weight"),
        "ROPE_Q": t(bshd,"activation"),"ROPE_K": t(bshd,"activation"),
        "ATTN": t(hd1,"activation"),
        "GELU": t(hd1,"activation"),
        "CONCAT": t([1,S,1,2*HD],"activation"),
        "W2": t([1,HD,1,2*HD],"weight"),"B2": t(biasMN,"weight"),
        "LIN2": t(bshd,"activation"),
        "GATE": t(bshd,"weight"),
        "BLOCK_OUT": t(bshd,"io"),
    }
    nodes = [
        {"name":"layernorm","kernel":LN,"kwargs":{"rows":MLEN,"hidden_size":HD,"num_s_blocks":NSB,"batch":1,"eps":EPS},
         "in":{"X_hbm":"X","SCALE_hbm":"LN_W_S","BIAS_hbm":"LN_W_B"},"out":{"Y_hbm":"LN_Y"}},
        {"name":"modulate","kernel":MOD,"kwargs":norm_geo,
         "in":{"X_hbm":"LN_Y","SCALE1P_hbm":"MOD_S","SHIFT_hbm":"MOD_H"},"out":{"Y_hbm":"MOD_Y"}},
        {"name":"linear_q","kernel":LIN,"kwargs":lin_kw,"in":{"A_hbm":"MOD_Y","B_hbm":"WQ","BIAS_hbm":"BQ"},"out":{"C_hbm":"Q"}},
        {"name":"linear_k","kernel":LIN,"kwargs":lin_kw,"in":{"A_hbm":"MOD_Y","B_hbm":"WK","BIAS_hbm":"BK"},"out":{"C_hbm":"Kk"}},
        {"name":"linear_v","kernel":LIN,"kwargs":lin_kw,"in":{"A_hbm":"MOD_Y","B_hbm":"WV","BIAS_hbm":"BV"},"out":{"C_hbm":"V"}},
        {"name":"linear_mlp","kernel":LIN,"kwargs":lin_kw,"in":{"A_hbm":"MOD_Y","B_hbm":"WM","BIAS_hbm":"BM"},"out":{"C_hbm":"MLP"}},
        {"name":"qknorm_q","kernel":RMS,"kwargs":norm_geo,"in":{"X_hbm":"Q","SCALE_hbm":"QKN_W"},"out":{"Y_hbm":"QKN_Q"}},
        {"name":"qknorm_k","kernel":RMS,"kwargs":norm_geo,"in":{"X_hbm":"Kk","SCALE_hbm":"QKN_W"},"out":{"Y_hbm":"QKN_K"}},
        {"name":"rope_q","kernel":ROPE,"kwargs":rope_kw,"in":{"XQ_hbm":"QKN_Q","COS_hbm":"COS","SGN_SIN_hbm":"SGN","P_hbm":"P"},"out":{"Q_OUT_hbm":"ROPE_Q"}},
        {"name":"rope_k","kernel":ROPE,"kwargs":rope_kw,"in":{"XQ_hbm":"QKN_K","COS_hbm":"COS","SGN_SIN_hbm":"SGN","P_hbm":"P"},"out":{"Q_OUT_hbm":"ROPE_K"}},
        {"name":"gelu","kernel":GELU,"kwargs":gelu_kw,"in":{"X_hbm":"MLP"},"out":{"Y_hbm":"GELU"}},
        {"name":"flash_attention","kernel":FA,"kwargs":fa_kw,"in":{"Q_hbm":"ROPE_Q","K_hbm":"ROPE_K","V_hbm":"V"},"out":{"O_hbm":"ATTN"}},
        {"name":"concat","kernel":CC,"kwargs":cc_kw,"in":{"A_hbm":"ATTN","B_hbm":"GELU"},"out":{"Y_hbm":"CONCAT"}},
        {"name":"linear2","kernel":LIN,"kwargs":lin2_kw,"in":{"A_hbm":"CONCAT","B_hbm":"W2","BIAS_hbm":"B2"},"out":{"C_hbm":"LIN2"}},
        {"name":"residual_gate","kernel":RG,"kwargs":norm_geo,"in":{"X_hbm":"X","GATE_hbm":"GATE","Y_hbm":"LIN2"},"out":{"OUT_hbm":"BLOCK_OUT"}},
    ]
    graph = {"tensors": tensors, "nodes": nodes}

    def b4(b): return b.view(1,1,1,HD).expand(1,S,1,HD).contiguous()
    data = {
        "X": x,
        "LN_W_S": ln_sc.view(1,1,1,HD).expand(1,S,1,HD).contiguous(),
        "LN_W_B": ln_bi.view(1,1,1,HD).expand(1,S,1,HD).contiguous(),
        "MOD_S": (1.0+mod_sc), "MOD_H": mod_sh,
        "WQ": Wq.view(1,HD,1,HD).contiguous(), "BQ": b4(bq),
        "WK": Wk.view(1,HD,1,HD).contiguous(), "BK": b4(bk),
        "WV": Wv.view(1,HD,1,HD).contiguous(), "BV": b4(bv),
        "WM": Wm.view(1,HD,1,HD).contiguous(), "BM": b4(bm),
        "QKN_W": qkn_sc.view(1,1,1,HLEN).expand(1,S,HEAD,HLEN).contiguous(),
        "COS": cos_hd, "SGN": sgn_hd, "P": P_hbm,
        "W2": W2.view(1,HD,1,2*HD).contiguous(), "B2": b4(b2),
        "GATE": gate,
    }
    compare = {
        "LN_Y": LN_Y.reshape(-1).numpy(), "MOD_Y": MOD_Y.reshape(-1).numpy(),
        "Q": Q.reshape(-1).numpy(), "Kk": Kk.reshape(-1).numpy(),
        "V": V.reshape(-1).numpy(), "MLP": MLP.reshape(-1).numpy(),
        "QKN_Q": QKN_Q.reshape(-1).numpy(), "QKN_K": QKN_K.reshape(-1).numpy(),
        "ROPE_Q": ROPE_Q.reshape(-1).numpy(), "ROPE_K": ROPE_K.reshape(-1).numpy(),
        "ATTN": ATTN.reshape(-1).numpy(), "GELU": GELU_g.reshape(-1).numpy(),
        "CONCAT": CONCAT_g.reshape(-1).numpy(), "LIN2": LIN2_g.reshape(-1).numpy(),
        "BLOCK_OUT": BLOCK.reshape(-1).numpy(),
    }

    mgr = Manager(settings=s)
    out = mgr.run_graph(graph, data=data, compare=compare)

    print()
    fails = 0
    for cmp in out["compares"]:
        ok = cmp.ok(cos_thresh=0.8)
        fails += 0 if ok else 1
        print(f"  {'OK ' if ok else 'FAIL'} {cmp.name}: cosine={cmp.cosine:.6f} nrmse={cmp.nrmse:.6f}")
    print(f"\n{'ALL PASS' if fails==0 else f'{fails} FAILURE(S)'}")

    # --- auto-write a standalone report next to KERNEL_REPORT.md ---
    import datetime
    _end = datetime.datetime.now()
    report = _ROOT / "MANAGER_BLOCK_REPORT.md"
    _wall = (_end - _start).total_seconds()
    lines = []
    lines.append(f"# Manager single_stream_block end-to-end report\n")
    lines.append(f"_by tools/manager/_validate_block.py_\n\n")
    lines.append(f"- **开始:** {_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"- **结束:** {_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"- **总墙钟时长:** {_wall:.1f}s ({_wall/60:.1f} min)\n")
    lines.append("\n## Hardware config (plena_settings.toml [BEHAVIOR])\n\n")
    lines.append("| item | value |\n|---|---|\n")
    lines.append(f"| MLEN | {s.mlen} |\n")
    lines.append(f"| HLEN | {s.hlen} |\n")
    lines.append(f"| BLEN | {s.blen} |\n")
    lines.append(f"| VLEN | {s.vlen} |\n")
    lines.append(f"| BROADCAST_AMOUNT (=MLEN//HLEN) | {s.hardware_lane_count} |\n")
    lines.append(f"| HBM_WIDTH | {s.hbm_row_width} bytes |\n")
    lines.append(f"| MX elem | E{s.elem_exp}M{s.elem_man} ({s.elem_bits} bit) |\n")
    lines.append(f"| MX scale | E{s.scale_exp}M{s.scale_man} ({s.scale_bits} bit) |\n")
    lines.append(f"| block_size | {s.block_size} |\n")
    lines.append("\n## Model dims (derived)\n\n")
    lines.append(f"- HEAD_COUNT = MLEN//HLEN*2 = {HEAD}\n")
    lines.append(f"- H*D = HEAD*HLEN = {HD}\n")
    lines.append(f"- NUM_S_BLOCKS = {NSB}, S = NSB*MLEN = {S}\n")
    lines.append(f"- CONCAT_DIM = 2*H*D = {2*HD}\n")
    lines.append("\n## Results (GLOBAL/cumulative cosine vs ideal MX-roundtrip chain)\n\n")
    lines.append("| step | kernel:tensor | cosine | NRMSE | status |\n|---|---|---|---|---|\n")
    for i, cmp in enumerate(out["compares"], 1):
        st = "OK" if cmp.ok(cos_thresh=0.8) else "FAIL"
        lines.append(f"| {i} | {cmp.name} | {cmp.cosine:.6f} | {cmp.nrmse*100:.3f}% | {st} |\n")
    verdict = "ALL PASS" if fails == 0 else f"{fails} FAILURE(S)"
    lines.append(f"\n**Verdict: {verdict}** (threshold cosine >= 0.8).\n")

    # --- timing table (per kernel: compile / write-bin / assemble / emulator) ---
    res = out.get("results", {})
    if res:
        lines.append("\n## Per kernel: wall-clock (seconds) + hardware latency\n\n")
        lines.append("| step | kernel | compile | write(quant) | assemble | emulator | total | HW latency |\n")
        lines.append("|---|---|---|---|---|---|---|---|\n")
        grand = {"compile": 0, "write": 0, "assemble": 0, "emulator": 0, "total": 0}
        for i, (name, r) in enumerate(res.items(), 1):
            tm = r.get("timing", {})
            lat = r.get("latency", "?")
            for k in grand:
                grand[k] += tm.get(k, 0)
            lines.append(f"| {i} | {name} | {tm.get('compile',0):.1f} | "
                        f"{tm.get('write',0):.1f} | {tm.get('assemble',0):.1f} | "
                        f"{tm.get('emulator',0):.1f} | {tm.get('total',0):.1f} | {lat} |\n")
        lines.append(f"| | **TOTAL** | **{grand['compile']:.1f}** | "
                    f"**{grand['write']:.1f}** | **{grand['assemble']:.1f}** | "
                    f"**{grand['emulator']:.1f}** | **{grand['total']:.1f}** | |\n")
        lines.append(f"\n- compile = subprocess into the compiler CLI;  "
                     f"write(quant) = MX-quantize + seek-write weights/fp_sram;  "
                     f"assemble = ISA→machine code;  emulator = the Rust sim run (wall-clock).\n")
        lines.append(f"- **HW latency** = the modeled hardware latency the emulator "
                     f"reports (`Simulation completed. Latency ...`) — i.e. the simulated "
                     f"on-chip cycle/time cost of running that kernel, NOT wall-clock.\n")
    lines.append("\nGLOBAL error: each golden uses the previous step's golden (ideal chain, "
                 "MX-roundtrip per HBM hop); error accumulates kernel by kernel. "
                 "BLOCK_OUT = end-to-end block error. Not local/per-kernel error.\n")
    report.write_text("".join(lines))
    print(f"\nreport -> {report}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
