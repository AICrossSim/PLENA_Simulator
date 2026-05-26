from transactional_emulator.testbench.siglip.utils.math import MXFP_REAL_DATA_RATIO


def pad_to(x, m):
    return ((x + m - 1) // m) * m


def compute_vram_layout(mlen, blen, q_len, hq, hkv, d, vector_sram_base=0):
    q_size = q_len * hq * d
    q_base = vector_sram_base
    s_base = q_base + q_size
    s_tile_count = blen if (hq // hkv == blen) else 1
    pv_base = s_base + mlen * mlen * s_tile_count
    o_old_base = pv_base + mlen * mlen * (hq // hkv)
    return {
        "q_base": q_base,
        "s_base": s_base,
        "pv_base": pv_base,
        "o_old_base": o_old_base,
        "q_size": q_size,
        "s_tile_count": s_tile_count,
    }


def compute_hbm_offsets(sizes_in_elements, real_data_ratio=MXFP_REAL_DATA_RATIO, align_elems=64):
    """
    Compute HBM element offsets for a sequence of tensors.
    sizes_in_elements: list of raw element counts (number of elements in tensor)
    real_data_ratio: heuristic expansion factor used by quantizer (matches other tests)
    align_elems: align each block to this many elements (default 64)
    Returns (offsets, total_elems)
    """
    offsets = []
    cur = 0
    for elems in sizes_in_elements:
        size = int(elems * real_data_ratio)
        size = pad_to(size, align_elems)
        offsets.append(cur)
        cur += size
    return offsets, cur
