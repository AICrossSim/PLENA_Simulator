"""FULL MMDiT double_stream_block via the graph driver + HBM-bin relay.

Mirrors Open-Sora's DoubleStreamBlockProcessor (opensora/models/mmdit/
layers.py:195-253). Two streams (img, txt) with INDEPENDENT weights run the
pre-attention chain each; their Q/K/V are concatenated along the SEQUENCE axis
into one joint sequence, a single flash_attention runs over it, then the output
is split back per stream; each stream then runs its own proj-residual and
MLP-residual.

Per stream (img / txt), with its own weights:
  x --norm1(LayerNorm)--> --modulate(mod1)--> --linear_q/k/v--> q,k,v
  q --qknorm_q--rope_q--> Q ;  k --qknorm_k--rope_k--> K ;  v passthrough
Joint:
  Qj = s_concat(txt_q, img_q) ; Kj = s_concat(txt_k, img_k) ; Vj = s_concat(txt_v, img_v)
  ATTN = flash_attention(Qj, Kj, Vj)        # one long sequence S_txt + S_img
  txt_attn, img_attn = s_split(ATTN)
Per stream again:
  x = x + gate1 * proj(attn)                            # proj = linear
  x = x + gate2 * mlp(modulate(mod2, norm2(x)))         # mlp = linear->gelu->linear

Simplifications (faithful structure, not exact hyperparams):
  * mlp_ratio = 1 (mlp_hidden = HD) so every matmul stays HD-square / MLEN-aligned.
  * gate is pre-broadcast to a per-element tile (same as single-block validate).
  * txt and img are symmetric (same seq length NSB each), distinct weights.

Every HBM hop MX-E4M3 round-trips; threshold 0.8 (deep chain accumulates MX
quant noise). Run: tools/manager/run.sh _validate_double_block
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
    NSB = 2                              # seq blocks PER STREAM (symmetric)
    HEAD = s.hardware_lane_count * 2
    S = NSB * MLEN                       # per-stream seq length
    SJ = 2 * S                           # joint seq length (txt + img)
    NSBJ = 2 * NSB
    HD = HEAD * HLEN
    EPS = 1e-6
    HALF = HLEN // 2
    print(f"geometry: mlen={MLEN} hlen={HLEN} HEAD={HEAD} HD={HD}  "
          f"S/stream={S} (nsb={NSB})  joint={SJ}")

    K = "tilelang_tvm_compiler.kernels."
    LN, MOD = K+"layernorm_min:make_layernorm_min", K+"modulate_min:make_modulate_min"
    LIN, GELU = K+"linear_min:make_linear_min", K+"gelu_min:make_gelu_min"
    RMS, ROPE = K+"rmsnorm_min:make_rmsnorm_min", K+"rope_min:make_rope_min"
    FA = K+"flash_attention_min:make_flash_attention_min"
    SC, SS = K+"s_concat_min:make_s_concat_min", K+"s_split_min:make_s_split_min"
    RG = K+"residual_gate_min:make_residual_gate_min"

    hd1 = [1, S, 1, HD]
    bshd = [1, S, HEAD, HLEN]
    wKN = [1, HD, 1, HD]
    biasMN = [1, S, 1, HD]
    hd1j = [1, SJ, 1, HD]
    bshdj = [1, SJ, HEAD, HLEN]

    lin_kw = {"m_blocks": S//MLEN, "n_blocks": HD//MLEN, "k_blocks": HD//MLEN, "with_bias": True}
    norm_geo = {"rows": MLEN, "hlen": HLEN, "head_count": HEAD, "num_s_blocks": NSB, "batch": 1}
    rope_kw = {"rows": MLEN, "hlen": HLEN, "head_count": HEAD, "half_dim": HALF,
               "num_s_blocks": NSB, "batch": 1}
    fa_kw = {"rows": MLEN, "hlen": HLEN, "head_count": HEAD, "active_lane": 2,
             "num_kv_blocks": NSBJ, "num_q_blocks": NSBJ}
    sc_kw = {"hd": HD, "num_s_blocks_a": NSB, "num_s_blocks_b": NSB, "batch": 1}

    torch.manual_seed(0)

    # ---------------- weights (independent per stream) ----------------
    def W(): return torch.randn(HD, HD) * 0.25
    def Wb(): return torch.randn(HD) * 0.1

    class StreamW:
        """One stream's full weight set."""
        def __init__(self):
            self.Wq, self.bq = W(), Wb()
            self.Wk, self.bk = W(), Wb()
            self.Wv, self.bv = W(), Wb()
            self.mod1_sc = torch.randn(1, S, HEAD, HLEN)*0.3
            self.mod1_sh = torch.randn(1, S, HEAD, HLEN)*0.3
            self.mod2_sc = torch.randn(1, S, HEAD, HLEN)*0.3
            self.mod2_sh = torch.randn(1, S, HEAD, HLEN)*0.3
            self.qkn = torch.randn(HLEN)*0.3+1.0
            self.gate1 = torch.randn(1, S, HEAD, HLEN)*0.3
            self.gate2 = torch.randn(1, S, HEAD, HLEN)*0.3
            self.Wproj, self.bproj = W(), Wb()
            self.Wmlp_in, self.bmlp_in = W(), Wb()    # mlp_ratio=1 -> HD x HD
            self.Wmlp_out, self.bmlp_out = W(), Wb()

    img_w, txt_w = StreamW(), StreamW()
    # shared rope tables (positional, stream-agnostic within this validate)
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

    # ---------------- golden helpers ----------------
    def ling(a2d, Wm_, b_):              # (S,K)@(N,K)^T + bias
        return mxr(a2d @ mxr(Wm_).T + mxr(b_.view(1, HD).expand(S, HD).contiguous()))

    def lnorm(xhd):                      # LayerNorm (no affine, eps) on (1,S,1,HD)
        xe = mxr(xhd); mu = xe.mean(-1, keepdim=True); xc = xe-mu
        var = (xc*xc).mean(-1, keepdim=True)
        return mxr(xc*torch.rsqrt(var+EPS))            # (1,S,1,HD)

    def modulate(xhd, sc, sh):           # (1+sc)*x + sh on bshd
        s1p = mxr(1.0+sc); shf = mxr(sh)
        return mxr(s1p*mxr(xhd.reshape(1,S,HEAD,HLEN))+shf)   # (1,S,8,16)

    def rms(t2d, qkn):
        t = mxr(t2d.reshape(1, S, HEAD, HLEN))
        scq = mxr(qkn.view(1,1,1,HLEN).expand(1,S,HEAD,HLEN).contiguous())
        msq = (t*t).mean(-1, keepdim=True)
        return mxr((t*scq)*torch.rsqrt(msq+EPS))

    def ropeg(qkn_t):
        x2d = mxr(qkn_t.reshape(S, HD)); p2d = mxr(P_hbm.reshape(MLEN, MLEN))
        nb = HD//MLEN
        xs = torch.cat([x2d[:, b*MLEN:(b+1)*MLEN] @ p2d for b in range(nb)], -1)
        return mxr(x2d*mxr(cos_hd.reshape(S,HD))+xs*mxr(sgn_hd.reshape(S,HD)))

    def stream_pre(x, w):                # returns (Q,K,V) each (1,S,8,16) eff, plus mod1 used
        ln = lnorm(x)
        mod = modulate(ln, w.mod1_sc, w.mod1_sh)           # (1,S,8,16)
        m2d = mxr(mod.reshape(S, HD))
        Q = ling(m2d, w.Wq, w.bq); Kk = ling(m2d, w.Wk, w.bk); V = ling(m2d, w.Wv, w.bv)
        QKN_Q = rms(Q, w.qkn); QKN_K = rms(Kk, w.qkn)
        RQ = ropeg(QKN_Q); RK = ropeg(QKN_K)               # (S,HD)
        return RQ, RK, V                                   # RQ/RK (S,HD); V (S,HD)

    def stream_post(x, attn_hd, w):      # x:(1,S,1,HD) attn:(S,HD)
        # proj residual
        proj = ling(mxr(attn_hd), w.Wproj, w.bproj)        # (S,HD)
        g1 = mxr(w.gate1)
        x1 = mxr(mxr(x.reshape(1,S,HEAD,HLEN)) + g1*mxr(proj.reshape(1,S,HEAD,HLEN)))
        # mlp residual
        ln2 = lnorm(x1.reshape(1,S,1,HD))
        mod2 = modulate(ln2, w.mod2_sc, w.mod2_sh)
        m2d = mxr(mod2.reshape(S, HD))
        h = ling(m2d, w.Wmlp_in, w.bmlp_in)                # (S,HD)
        h = mxr(torch.nn.functional.gelu(mxr(h.reshape(1,S,HEAD,HLEN)), approximate="tanh"))
        mlp = ling(mxr(h.reshape(S,HD)), w.Wmlp_out, w.bmlp_out)
        g2 = mxr(w.gate2)
        x2 = mxr(x1 + g2*mxr(mlp.reshape(1,S,HEAD,HLEN)))
        return x2                                          # (1,S,8,16)

    # ---------------- golden chain ----------------
    img_x = torch.randn(1, S, 1, HD)*0.5
    txt_x = torch.randn(1, S, 1, HD)*0.5
    iRQ, iRK, iV = stream_pre(img_x, img_w)
    tRQ, tRK, tV = stream_pre(txt_x, txt_w)
    # joint concat along seq (txt first, then img) -> (SJ, HD)
    Qj = mxr(torch.cat([mxr(tRQ), mxr(iRQ)], dim=0))
    Kj = mxr(torch.cat([mxr(tRK), mxr(iRK)], dim=0))
    Vj = mxr(torch.cat([mxr(tV),  mxr(iV)],  dim=0))
    # joint flash attention per head over SJ
    qj = mxr(Qj.reshape(1,SJ,HEAD,HLEN)); kj = mxr(Kj.reshape(1,SJ,HEAD,HLEN)); vj = mxr(Vj.reshape(1,SJ,HEAD,HLEN))
    scale = 1.0/math.sqrt(HLEN); attn = torch.empty(1,SJ,HEAD,HLEN)
    for h in range(HEAD):
        sc = (qj[0,:,h,:] @ kj[0,:,h,:].T)*scale
        attn[0,:,h,:] = torch.softmax(sc, -1) @ vj[0,:,h,:]
    ATTN = mxr(attn)                                       # (1,SJ,8,16)
    attn2d = mxr(ATTN.reshape(SJ, HD))
    t_attn = mxr(attn2d[0:S]); i_attn = mxr(attn2d[S:SJ])  # split: txt first, img second
    img_out = stream_post(img_x, i_attn, img_w)
    txt_out = stream_post(txt_x, t_attn, txt_w)

    # ---------------- graph ----------------
    def t(shape, role): return {"shape": shape, "role": role}

    tensors = {}
    nodes = []

    def add_t(name, shape, role): tensors[name] = t(shape, role)
    def add_n(name, kernel, kwargs, ins, outs):
        nodes.append({"name": name, "kernel": kernel, "kwargs": kwargs, "in": ins, "out": outs})

    data = {}
    compare = {}

    def b4(b): return b.view(1,1,1,HD).expand(1,S,1,HD).contiguous()

    def build_stream_pre(p, x, w, xtensor):
        """Emit pre-attention nodes for stream prefix p ('I'/'T'). Returns
        (rope_q_name, rope_k_name, v_name) tensor names."""
        # weights
        add_t(f"{p}_MOD1S", bshd, "weight"); add_t(f"{p}_MOD1H", bshd, "weight")
        add_t(f"{p}_WQ", wKN, "weight"); add_t(f"{p}_BQ", biasMN, "weight")
        add_t(f"{p}_WK", wKN, "weight"); add_t(f"{p}_BK", biasMN, "weight")
        add_t(f"{p}_WV", wKN, "weight"); add_t(f"{p}_BV", biasMN, "weight")
        add_t(f"{p}_QKN", bshd, "weight")
        add_t("COS", hd1, "weight") if "COS" not in tensors else None
        add_t("SGN", hd1, "weight") if "SGN" not in tensors else None
        add_t("P", [1,MLEN,1,MLEN], "weight") if "P" not in tensors else None
        # activations
        add_t(f"{p}_LN1", bshd, "activation"); add_t(f"{p}_MOD1", hd1, "activation")
        add_t(f"{p}_Q", bshd, "activation"); add_t(f"{p}_K", bshd, "activation"); add_t(f"{p}_V", bshd, "activation")
        add_t(f"{p}_QKNQ", hd1, "activation"); add_t(f"{p}_QKNK", hd1, "activation")
        add_t(f"{p}_RQ", bshd, "activation"); add_t(f"{p}_RK", bshd, "activation")
        # nodes
        add_n(f"{p}_norm1", LN, {"rows":MLEN,"hidden_size":HD,"num_s_blocks":NSB,"batch":1,"eps":EPS,"affine":False} if False else {"rows":MLEN,"hidden_size":HD,"num_s_blocks":NSB,"batch":1,"eps":EPS},
              {"X_hbm":xtensor,"SCALE_hbm":f"{p}_LNS","BIAS_hbm":f"{p}_LNB"}, {"Y_hbm":f"{p}_LN1"})
        add_n(f"{p}_mod1", MOD, norm_geo,
              {"X_hbm":f"{p}_LN1","SCALE1P_hbm":f"{p}_MOD1S","SHIFT_hbm":f"{p}_MOD1H"}, {"Y_hbm":f"{p}_MOD1"})
        add_n(f"{p}_linq", LIN, lin_kw, {"A_hbm":f"{p}_MOD1","B_hbm":f"{p}_WQ","BIAS_hbm":f"{p}_BQ"}, {"C_hbm":f"{p}_Q"})
        add_n(f"{p}_link", LIN, lin_kw, {"A_hbm":f"{p}_MOD1","B_hbm":f"{p}_WK","BIAS_hbm":f"{p}_BK"}, {"C_hbm":f"{p}_K"})
        add_n(f"{p}_linv", LIN, lin_kw, {"A_hbm":f"{p}_MOD1","B_hbm":f"{p}_WV","BIAS_hbm":f"{p}_BV"}, {"C_hbm":f"{p}_V"})
        add_n(f"{p}_qknq", RMS, norm_geo, {"X_hbm":f"{p}_Q","SCALE_hbm":f"{p}_QKN"}, {"Y_hbm":f"{p}_QKNQ"})
        add_n(f"{p}_qknk", RMS, norm_geo, {"X_hbm":f"{p}_K","SCALE_hbm":f"{p}_QKN"}, {"Y_hbm":f"{p}_QKNK"})
        add_n(f"{p}_ropeq", ROPE, rope_kw, {"XQ_hbm":f"{p}_QKNQ","COS_hbm":"COS","SGN_SIN_hbm":"SGN","P_hbm":"P"}, {"Q_OUT_hbm":f"{p}_RQ"})
        add_n(f"{p}_ropek", ROPE, rope_kw, {"XQ_hbm":f"{p}_QKNK","COS_hbm":"COS","SGN_SIN_hbm":"SGN","P_hbm":"P"}, {"Q_OUT_hbm":f"{p}_RK"})
        return f"{p}_RQ", f"{p}_RK", f"{p}_V"

    # LayerNorm here uses scale/bias tensors; for affine-free LN we feed
    # scale=1, bias=0 (layernorm_min always applies y=(norm)*scale+bias).
    def add_ln_affine(p, w):
        add_t(f"{p}_LNS", hd1, "weight"); add_t(f"{p}_LNB", hd1, "weight")
        data[f"{p}_LNS"] = torch.ones(1,S,1,HD)
        data[f"{p}_LNB"] = torch.zeros(1,S,1,HD)

    def stream_data(p, w, xtensor, xval):
        data[xtensor] = xval
        data[f"{p}_MOD1S"] = (1.0+w.mod1_sc); data[f"{p}_MOD1H"] = w.mod1_sh
        data[f"{p}_WQ"] = w.Wq.view(1,HD,1,HD).contiguous(); data[f"{p}_BQ"] = b4(w.bq)
        data[f"{p}_WK"] = w.Wk.view(1,HD,1,HD).contiguous(); data[f"{p}_BK"] = b4(w.bk)
        data[f"{p}_WV"] = w.Wv.view(1,HD,1,HD).contiguous(); data[f"{p}_BV"] = b4(w.bv)
        data[f"{p}_QKN"] = w.qkn.view(1,1,1,HLEN).expand(1,S,HEAD,HLEN).contiguous()

    # shared rope table data (placed once)
    data["COS"] = cos_hd; data["SGN"] = sgn_hd; data["P"] = P_hbm

    # ---- pre-attention for both streams ----
    add_t("IMG_X", hd1, "io"); add_t("TXT_X", hd1, "io")
    add_ln_affine("I", img_w); add_ln_affine("T", txt_w)
    iq, ik, iv = build_stream_pre("I", img_x, img_w, "IMG_X")
    tq, tk, tv = build_stream_pre("T", txt_x, txt_w, "TXT_X")
    stream_data("I", img_w, "IMG_X", img_x)
    stream_data("T", txt_w, "TXT_X", txt_x)

    # ---- joint: s_concat (txt first, img second) -> flash -> s_split ----
    add_t("QJ", hd1j, "activation"); add_t("KJ", hd1j, "activation"); add_t("VJ", hd1j, "activation")
    add_t("ATTNJ", bshdj, "activation")
    add_t("T_ATTN", hd1, "activation"); add_t("I_ATTN", hd1, "activation")
    add_n("concat_q", SC, sc_kw, {"A_hbm":tq, "B_hbm":iq}, {"Y_hbm":"QJ"})
    add_n("concat_k", SC, sc_kw, {"A_hbm":tk, "B_hbm":ik}, {"Y_hbm":"KJ"})
    add_n("concat_v", SC, sc_kw, {"A_hbm":tv, "B_hbm":iv}, {"Y_hbm":"VJ"})
    add_n("flash", FA, fa_kw, {"Q_hbm":"QJ","K_hbm":"KJ","V_hbm":"VJ"}, {"O_hbm":"ATTNJ"})
    add_n("split_attn", SS, sc_kw, {"X_hbm":"ATTNJ"}, {"A_hbm":"T_ATTN","B_hbm":"I_ATTN"})

    # ---- post-attention for both streams ----
    def build_stream_post(p, w, xtensor, attn_tensor, out_io):
        add_t(f"{p}_WPROJ", wKN, "weight"); add_t(f"{p}_BPROJ", biasMN, "weight")
        add_t(f"{p}_GATE1", bshd, "weight"); add_t(f"{p}_GATE2", bshd, "weight")
        add_t(f"{p}_MOD2S", bshd, "weight"); add_t(f"{p}_MOD2H", bshd, "weight")
        add_t(f"{p}_WMI", wKN, "weight"); add_t(f"{p}_BMI", biasMN, "weight")
        add_t(f"{p}_WMO", wKN, "weight"); add_t(f"{p}_BMO", biasMN, "weight")
        add_t(f"{p}_LNS2", hd1, "weight"); add_t(f"{p}_LNB2", hd1, "weight")
        # activations
        add_t(f"{p}_PROJ", bshd, "activation"); add_t(f"{p}_X1", bshd, "activation")
        add_t(f"{p}_LN2", bshd, "activation"); add_t(f"{p}_MOD2", hd1, "activation")
        add_t(f"{p}_MH", bshd, "activation"); add_t(f"{p}_MHG", hd1, "activation")
        add_t(f"{p}_MLP", bshd, "activation")
        # proj residual: x1 = x + gate1 * proj(attn)
        add_n(f"{p}_proj", LIN, lin_kw, {"A_hbm":attn_tensor,"B_hbm":f"{p}_WPROJ","BIAS_hbm":f"{p}_BPROJ"}, {"C_hbm":f"{p}_PROJ"})
        add_n(f"{p}_res1", RG, norm_geo, {"X_hbm":xtensor,"GATE_hbm":f"{p}_GATE1","Y_hbm":f"{p}_PROJ"}, {"OUT_hbm":f"{p}_X1"})
        # mlp residual: x2 = x1 + gate2 * mlp(modulate(norm2(x1)))
        add_n(f"{p}_norm2", LN, {"rows":MLEN,"hidden_size":HD,"num_s_blocks":NSB,"batch":1,"eps":EPS},
              {"X_hbm":f"{p}_X1","SCALE_hbm":f"{p}_LNS2","BIAS_hbm":f"{p}_LNB2"}, {"Y_hbm":f"{p}_LN2"})
        add_n(f"{p}_mod2", MOD, norm_geo, {"X_hbm":f"{p}_LN2","SCALE1P_hbm":f"{p}_MOD2S","SHIFT_hbm":f"{p}_MOD2H"}, {"Y_hbm":f"{p}_MOD2"})
        add_n(f"{p}_mlpin", LIN, lin_kw, {"A_hbm":f"{p}_MOD2","B_hbm":f"{p}_WMI","BIAS_hbm":f"{p}_BMI"}, {"C_hbm":f"{p}_MH"})
        add_n(f"{p}_gelu", GELU, norm_geo, {"X_hbm":f"{p}_MH"}, {"Y_hbm":f"{p}_MHG"})
        add_n(f"{p}_mlpout", LIN, lin_kw, {"A_hbm":f"{p}_MHG","B_hbm":f"{p}_WMO","BIAS_hbm":f"{p}_BMO"}, {"C_hbm":f"{p}_MLP"})
        add_n(f"{p}_res2", RG, norm_geo, {"X_hbm":f"{p}_X1","GATE_hbm":f"{p}_GATE2","Y_hbm":f"{p}_MLP"}, {"OUT_hbm":out_io})
        # data
        data[f"{p}_WPROJ"] = w.Wproj.view(1,HD,1,HD).contiguous(); data[f"{p}_BPROJ"] = b4(w.bproj)
        data[f"{p}_GATE1"] = w.gate1; data[f"{p}_GATE2"] = w.gate2
        data[f"{p}_MOD2S"] = (1.0+w.mod2_sc); data[f"{p}_MOD2H"] = w.mod2_sh
        data[f"{p}_WMI"] = w.Wmlp_in.view(1,HD,1,HD).contiguous(); data[f"{p}_BMI"] = b4(w.bmlp_in)
        data[f"{p}_WMO"] = w.Wmlp_out.view(1,HD,1,HD).contiguous(); data[f"{p}_BMO"] = b4(w.bmlp_out)
        data[f"{p}_LNS2"] = torch.ones(1,S,1,HD); data[f"{p}_LNB2"] = torch.zeros(1,S,1,HD)

    add_t("IMG_OUT", bshd, "io"); add_t("TXT_OUT", bshd, "io")
    build_stream_post("I", img_w, "IMG_X", "I_ATTN", "IMG_OUT")
    build_stream_post("T", txt_w, "TXT_X", "T_ATTN", "TXT_OUT")

    # ---------------- compares (key checkpoints) ----------------
    compare["QJ"] = Qj.reshape(-1).numpy()
    compare["ATTNJ"] = ATTN.reshape(-1).numpy()
    compare["IMG_OUT"] = img_out.reshape(-1).numpy()
    compare["TXT_OUT"] = txt_out.reshape(-1).numpy()

    graph = {"tensors": tensors, "nodes": nodes}

    mgr = Manager(settings=s)
    out = mgr.run_graph(graph, data=data, compare=compare)

    print()
    fails = 0
    for cmp in out["compares"]:
        ok = cmp.ok(cos_thresh=0.8)
        fails += 0 if ok else 1
        print(f"  {'OK ' if ok else 'FAIL'} {cmp.name}: cosine={cmp.cosine:.6f} nrmse={cmp.nrmse:.6f}")
    print(f"\n{'ALL PASS' if fails==0 else f'{fails} FAILURE(S)'}")
    _end = _dt.datetime.now()
    print(f"wall: {(_end-_start).total_seconds():.1f}s")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
