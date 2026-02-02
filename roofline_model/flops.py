def estimate_flops(mlen, vlen, operate_freq, batch_size=1):
    """
    Estimate the peak FLOPs, assuming 

    Args:
        mlen (int): The length of the model.
        vlen (int): The length of the vector.
        operate_freq (int): The operating frequency in MHz.

    Returns:
        int: Estimated number of TFLOPs.
    """
    matrix_machine_flops = mlen * mlen * 2 + mlen 
    vector_machine_flops = vlen * 2 
    scalar_machine_flops = 2 * mlen 
    total_flops = (matrix_machine_flops + vector_machine_flops + scalar_machine_flops) * operate_freq * pow(10, 6) * batch_size

    return total_flops / pow(10, 12)  # Convert to TFLOPs


def estimate_hbm_bandwidth(mlen, parallel_rd, mx_fp_element_width, blocksize, mx_fp_scale_width, operate_freq):
    """
    Estimate the HBM bandwidth.

    Args:
        mlen (int): The length of the model.
        mx_fp_element_width (int): The width of the floating-point element in bits.
        mx_fp_scale_width (int): The width of the scale in bits.
        operate_freq (int): The operating frequency in MHz.

    Returns:
        int: Estimated HBM bandwidth in GB/s.
    """
    hbm_bandwidth = parallel_rd * (mlen * mx_fp_element_width + (mlen // blocksize) * mx_fp_scale_width) * operate_freq * pow(10,6) / 8
    return hbm_bandwidth / pow(10, 9)  # Convert to GB/s


if __name__ == "__main__":
    mlen = 1024
    vlen = 64
    operate_freq = 225  # MHz
    batch_size = 32
    # flops = estimate_flops(mlen, vlen, operate_freq, batch_size)
    # print(f"Estimated FLOPs: {flops:.2f} TFLOPs")
    element_width = 8  # bits
    scale_width = 16  # bits
    blocksize = 4
    parallel_rd = 1
    hbm_bandwidth = estimate_hbm_bandwidth(mlen, parallel_rd, element_width, blocksize, scale_width, operate_freq)
    print(f"Estimated HBM Bandwidth: {hbm_bandwidth:.2f} GB/s")