def compute_hbm_bandwidth(
    mlen: int,
    hbm_rate: int, # in GB/s
    parallelism: int,
    mx_fp_element_width: int, # in bits
    mx_fp_block_size: int, # in bits
    mx_fp_scale_width: int, # in bits
    target_frequency: int # in MHz
) -> int:
    """
    Compute the HBM bandwidth in GB/s.

    :param hbm_channels: Number of HBM channels.
    :param hbm_data_rate: Data rate of HBM in MT/s.
    :param hbm_width: Width of HBM in bits.
    :return: HBM bandwidth in GB/s.
    """
    hbm_bandwidth_in_cycle = hbm_rate * 1024 // target_frequency
    min_scale_bandwidth = parallelism *  (mlen // mx_fp_block_size) * (mx_fp_scale_width) // 8
    min_block_bandwidth = parallelism * mlen * mx_fp_element_width // 8

    return min_scale_bandwidth, min_block_bandwidth, hbm_bandwidth_in_cycle
