"""Shared MXFP quantization helper for the direct_emit harnesses.

Every script in this directory needs the same "what does hardware actually see
after an HBM->VRAM load" transform. It used to be copy-pasted into each one.

Deliberately local to direct_emit/: the other `quantize_to_mxfp` definitions in
the testbench (sliced_layer_test_builder.py, aten/compare/linear_codegen_compare.py,
models/ffn_test.py) are NOT the same function -- they reshape to 2-D first, pass
a different block_size, or take extra arguments -- so they must not be folded in
here.
"""

from plena_quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware


def quantize_to_mxfp(tensor):
    """
    Quantize tensor to MXFP format matching hardware (E4M3 with 8-bit scale per block of 8).
    Returns the dequantized tensor (what hardware sees after HBM->VRAM load).
    """
    orig_shape = tensor.shape
    bm_x, _, _, _ = _mx_fp_quantize_hardware(tensor, width=8, exponent_width=4, exponent_bias_width=8, block_size=[8])
    return bm_x.reshape(orig_shape)
