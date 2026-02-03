import torch
from bitstring import BitArray
from typing import Union

from quant.quantizer.hardware_quantizer import _minifloat_ieee_quantize_hardware

def pack_fp_to_bin(signed_exponent, signed_mantissa, exp_width, man_width):
    exp_shape = signed_exponent.shape
    man_shape = signed_mantissa.shape
    signed_exponent = signed_exponent.reshape(-1)
    signed_mantissa = signed_mantissa.reshape(-1)

    sign = signed_mantissa.sign()
    sign_bit = torch.where(sign < 0, torch.tensor(1), torch.tensor(0))

    exponent_bias = (2**(exp_width - 1)) - 1
    exponent_bit = signed_exponent + exponent_bias

    for item in exponent_bit:
        assert item >= 0 and item <= (2**exp_width - 1), "Exponent out of range!"

    mantissa = torch.where(signed_mantissa < 0, -signed_mantissa, signed_mantissa)
    mantissa_bit = torch.where(exponent_bit == 0, mantissa, mantissa - 1)

    mantissa_bit = mantissa_bit * 2**(man_width)

    result = ((sign_bit * 2**(exp_width + man_width)) + 
            exponent_bit * 2**(man_width) + 
            mantissa_bit).int()
    
    result = result.reshape(exp_shape)

    return result

def split_bin(bits: Union[int, BitArray], exp_width: int, mant_width: int):
    """
    take the int as input, return int with output
    """
    bin = BitArray(uint=bits, length=exp_width + mant_width + 1)
    sign = bin[0]
    exponent_bits = bin[1:exp_width + 1]
    mantissa_bits = bin[exp_width + 1:]

    exponent = int(exponent_bits.bin, 2)
    mantissa = int(mantissa_bits.bin, 2)
    # Bias and reconstruct
    bias = (1 << (exp_width - 1)) - 1
    exponent_val = exponent - bias
    exponent_min = 0 - bias

    if exponent_val == exponent_min:
        mantissa_val = mantissa / (1 << mant_width)
    else:
        mantissa_val = 1.0 + (mantissa / (1 << mant_width))
    
    return exponent_val, -mantissa_val if sign else mantissa_val

def bin_2_fp(bits: Union[int, BitArray, torch.Tensor, list], exp_width: int, mant_width: int):
    if isinstance(bits, torch.Tensor) or isinstance(bits, list):
        # Handle tensor input - process each element
        results = []
        if isinstance(bits, torch.Tensor):
            _tensor = True
            tensor_shape = bits.shape
            bits = bits.reshape(-1)
            bits_list = bits.tolist()
            for bit_val in bits_list:
                exp_val, mant_val = split_bin(bit_val, exp_width, mant_width)
                results.append(mant_val * 2**exp_val)
        else:
            results = []
            

        if isinstance(bits_list, list):
            # Handle multi-element tensor
            results = []
            for bit_val in bits_list:
                exp_val, mant_val = split_bin(bit_val, exp_width, mant_width)
                results.append(mant_val * 2**exp_val)
            return torch.tensor(results)
        else:
            # Handle single-element tensor
            exp_val, mant_val = split_bin(bits_list, exp_width, mant_width)
            return torch.tensor(mant_val)
    elif isinstance(bits, list):
        # Handle list input
        results = []
        for bit_val in bits:
            exp_val, mant_val = split_bin(bit_val, exp_width, mant_width)
            results.append(mant_val * 2**exp_val)
        return results
    else:
        # Handle single int or BitArray
        exp_val, mant_val = split_bin(bits, exp_width, mant_width)
        return mant_val * 2**exp_val

def fp_2_bin(fp: Union[torch.Tensor, float, list], exp_width: int, mant_width: int):
    fp = torch.tensor(fp)

    q_fp, exp, mant = _minifloat_ieee_quantize_hardware(fp, exp_width + mant_width + 1, exp_width)
    bin = pack_fp_to_bin(exp, mant, exp_width, mant_width)
    return q_fp, bin

def test_fp_bin_conversion():
    # fp = torch.tensor([1.0, 2.0, 3.0, 4.0])
    # fp = torch.randn(100)
    fp = torch.randn(100) * 100 - 50
    exp_width = 4
    mant_width = 3
    q_fp, bin = fp_2_bin(fp, exp_width, mant_width)
    fp_re = bin_2_fp(bin, exp_width, mant_width)
    # Convert fp_re to tensor if it's not already
    if not isinstance(fp_re, torch.Tensor):
        fp_re = torch.tensor(fp_re)
    assert torch.allclose(q_fp, fp_re), f"q_fp: {q_fp}, fp_re: {fp_re}"

if __name__ == "__main__":
    test_fp_bin_conversion()