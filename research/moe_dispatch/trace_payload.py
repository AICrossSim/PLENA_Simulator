"""Replay the Rust event schedule with small, seeded BF16 payloads.

This is a verification observer, not a second performance model. Arithmetic
reads the finite W/X slots filled by the recorded addresses and events. The
large captured-route performance runs still contain timing metadata only.
"""
from collections import Counter
import hashlib
import numpy as np
from numerical import bf16, bf16_bits, from_bf16, dot_tree_512, activation, reference_expert


def replay(workload, report, seed=930):
    rng = np.random.default_rng(seed)
    experts = workload["experts"]
    x = bf16(rng.normal(0, .15, (workload["batch"], workload["hidden"])))
    weights, hbm, reference = {}, {}, {}
    id_to_task = {e["id"]: t for t, e in enumerate(experts)}
    for t, e in enumerate(experts):
        ws = []
        for p, name, n, k in ((0, "gate", e["F"], e["H"]),
                               (1, "up", e["F"], e["H"]), (4, "down", e["H"], e["F"])):
            w = bf16(rng.normal(0, .1, (n, k)))
            ws.append(w)
            weights[t, p] = w
            meta = e["weights"][name]
            raw = np.zeros((n, meta["row_stride_bytes"]), dtype=np.uint8)
            raw[:, :k*2] = bf16_bits(w).astype('<u2').view(np.uint8).reshape(n, k*2)
            hbm[t, p] = (meta["hbm_base"], raw.reshape(-1))
        reference[t] = reference_expert(x[e["token_indices"]], *ws)
    wslots, xslots, pending, acc, done, z = {}, {}, {}, {}, {}, {}
    inflight, owners, next_task, current, retired = {}, {}, {}, {}, set()
    k_order = Counter()
    peak_credit = 0
    for event in report["trace"]:
        kind, now = event["event"], event["cycle"]
        c = event.get("core")
        if kind == "commit_owner":
            t = event["task"]
            assert t not in owners and c not in next_task
            owners[t] = c
            next_task[c] = t
        elif kind == "promote":
            assert c not in current
            assert next_task.pop(c) == event["task"]
            current[c] = event["task"]
        elif kind == "dma_fire":
            r = event["request"]
            assert r["serial"] not in inflight
            assert owners[r["task"]] == r["core"]
            assert r["task"] in (next_task.get(r["core"]), current.get(r["core"]))
            inflight[r["serial"]] = r
            peak_credit = max(peak_credit, len(inflight))
            assert peak_credit <= report["config"]["credits"]
        elif kind == "dma_landed":
            r = event["request"]
            assert inflight.pop(r["serial"]) == r
            c, t, p, slot = r["core"], r["task"], r["phase"], r["slot"]
            tag = (t, p, r["tile"])
            if (c, slot) not in wslots:
                wslots[c, slot] = (tag, np.zeros((4, 512), np.uint16), np.zeros((4, 512), bool))
            old, data, valid = wslots[c, slot]
            assert old == tag, "a live W slot was overwritten"
            k_full = weights[t, p].shape[1]
            base, raw = hbm[t, p]
            offset = r["address"] - base
            assert 0 <= offset <= len(raw)-32
            stride = (k_full*2+31)//32*32
            global_k = (r["address"]-base) % stride // 2
            tile_k = global_k//512*512
            row_stride = (min(512, k_full-tile_k)*2+31)//32*32
            row, col = r["offset"]//row_stride, r["offset"]%row_stride//2
            assert not valid[row, col:col+16].any()
            data[row, col:col+16] = raw[offset:offset+32].copy().view('<u2')
            valid[row, col:col+16] = True
        elif kind == "x_stage":
            t, p = event["task"], event["phase"]
            source = x[experts[t]["token_indices"]] if p < 2 else z[t]
            m, k, mv, kv = (event[a] for a in ("m_start", "k_start", "m_valid", "k_valid"))
            xslots[c, event["slot"]] = ((t, p, m, k), source[m:m+mv, k:k+kv].copy(), event["ready_cycle"])
        elif kind == "mac_issue":
            t, p = event["task"], event["phase"]
            assert current[c] == t
            m, n, k, mv, nv, kv = (event[a] for a in ("m_start", "n_start", "k_start", "m_valid", "n_valid", "k_valid"))
            tag, data, valid = wslots[c, event["slot"]]
            assert tag == (t, p, event["tile"]) and valid[:nv, :kv].all()
            xtag, xx, ready = xslots[c, event["x_slot"]]
            assert xtag == (t, p, m, k) and ready <= now
            products = np.multiply(xx[:, None, :], from_bf16(data[:nv, :kv])[None, :, :], dtype=np.float32)
            key = (c, t, p, event["issue"])
            assert key not in pending
            pending[key] = (event, dot_tree_512(products))
        elif kind == "weight_slot_free":
            assert wslots.pop((c, event["slot"]), None) is not None
        elif kind == "k_commit":
            t, p = event["task"], event["phase"]
            issue, values = pending.pop((c, t, p, event["issue"]))
            m, n, k, mv, nv = (issue[a] for a in ("m_start", "n_start", "k_start", "m_valid", "n_valid"))
            key = (t, p, m, n)
            assert k_order[key] == k//512
            k_order[key] += 1
            out = acc.setdefault((t, p), np.zeros((experts[t]["Me"], weights[t, p].shape[0]), np.float32))
            out[m:m+mv, n:n+nv] = np.add(out[m:m+mv, n:n+nv], values, dtype=np.float32)
        elif kind == "projection_done":
            t, p = id_to_task[event["expert"]], event["phase"]
            done[t, p] = bf16(acc.pop((t, p)))
        elif kind == "activation_done":
            t = event["task"]
            z[t] = bf16(activation(done[t, 0], done[t, 1]))
        elif kind == "expert_drained":
            t = id_to_task[event["expert"]]
            assert current.pop(c) == t and t not in retired
            retired.add(t)
            np.testing.assert_array_equal(bf16_bits(done[t, 4]), bf16_bits(reference[t]))
    assert not pending and not acc and not inflight and not wslots
    assert not next_task and not current and len(retired) == len(experts)
    assert len(owners) == len(experts)
    # Ordered route reduction is an existing contract, independent of completion order.
    def combine(outputs):
        y = np.zeros_like(x)
        contributions = sorted((slot, token, t, row, score)
            for t, e in enumerate(experts) if not e.get("is_shared")
            for row, (slot, token, score) in enumerate(zip(e["route_slots"], e["token_indices"], e["route_scores"])))
        for _, token, t, row, score in contributions:
            y[token] = np.add(y[token], np.multiply(outputs[t][row], np.float32(score), dtype=np.float32), dtype=np.float32)
        y = bf16(y)
        for t, e in enumerate(experts):
            if e.get("is_shared"):
                for row, token in enumerate(e["token_indices"]):
                    y[token] = bf16(np.add(y[token], outputs[t][row], dtype=np.float32))
        return y
    output = combine({t: done[t, 4] for t in retired})
    np.testing.assert_array_equal(bf16_bits(output), bf16_bits(combine(reference)))
    return {"all_bit_exact": True, "experts": len(experts), "credit_peak": peak_credit,
            "sha256_bf16": hashlib.sha256(bf16_bits(output).tobytes()).hexdigest(),
            "scope": "seeded payload replay of actual Rust DMA/issue/commit trace"}
