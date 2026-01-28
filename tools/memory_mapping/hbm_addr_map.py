def align_addr_to_hbm_bandwidth(addr, hbm_mem_bandwidth):
    """
    Align the address to the next multiple of hbm_mem_bandwidth.

    Args:
        addr (int): The input address.
        hbm_mem_bandwidth (int): The HBM memory bandwidth (in bytes).

    Returns:
        int: The smallest address >= addr that is a multiple of hbm_mem_bandwidth.
    """
    if hbm_mem_bandwidth <= 0:
        raise ValueError("hbm_mem_bandwidth must be a positive integer")
    remainder = addr % hbm_mem_bandwidth
    if remainder == 0:
        return addr
    else:
        return addr + (hbm_mem_bandwidth - remainder)
