"""Random tensor generator for MXINT quantization."""

import torch
import os

from .quantizer import _mx_int_quantize_hardware
from ..common.utils import block


class Random_MXINT_Tensor_Generator:
    def __init__(self, shape, quant_config, directory=None, filename=None):
        """
        Initialize the random tensor generator with a given shape in MXINT.
        If directory and filename are provided, the tensor will be saved to a file.
        """
        self.shape = shape
        self.directory = directory
        self.filename = filename
        self.quant_config = quant_config

    def tensor_gen(self):
        tensor = torch.randn(self.shape)
        if self.directory and self.filename:
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            file_path = os.path.join(self.directory, self.filename)
            torch.save(tensor, file_path)

    def tensor_load(self):
        if self.directory and self.filename:
            file_path = os.path.join(self.directory, self.filename)
            if os.path.exists(file_path):
                tensor = torch.load(file_path)
                return tensor
            else:
                return None
        else:
            return None

    def quantize_tensor(self, tensor):
        """
        Quantize tensor to MXINT format.
        Returns (block_list, scaling_list).

        MXINT format: Each element is a signed integer with a shared exponent per block.
        Format: sign_bit | magnitude_bits (total = man_width bits)
        """
        # Handle tensor dimensions - flatten to 2D for quantization
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim > 2:
            # Flatten all but the last dimension
            tensor = tensor.reshape(-1, tensor.shape[-1])

        # Get block size
        block_size = self.quant_config["block_size"]
        if isinstance(block_size, int):
            block_size = [block_size]

        # Get the blocked tensor to extract signs
        blocked_x, _per_block_max, _padded_x_shape, _block_shape = block(
            tensor,
            block_shape=block_size,
            skip_first_dim=self.quant_config["skip_first_dim"],
        )
        # blocked_x shape: [block_elements, num_blocks]
        per_block_sign = torch.sign(blocked_x + 1e-9)

        # Get quantized mantissa and scaling
        _bm_x, per_block_mantissa, per_block_scaling = _mx_int_quantize_hardware(
            tensor,
            width=self.quant_config["man_width"],
            exponent_width=self.quant_config["exp_width"],
            block_size=block_size,
            skip_first_dim=self.quant_config["skip_first_dim"],
        )

        # Transpose to [num_blocks, block_elements] for iteration
        # per_block_mantissa: [block_elements, num_blocks] -> [num_blocks, block_elements]
        per_block_mantissa = per_block_mantissa.T
        per_block_sign = per_block_sign.T
        # per_block_scaling: [1, num_blocks] -> [num_blocks]
        per_block_scaling = per_block_scaling.squeeze(0)

        block_list = []
        scaling_list = []

        # Get exponent bias for converting to unsigned representation
        exponent_width = self.quant_config["exp_width"]
        exponent_bias = 2 ** (exponent_width - 1) - 1

        # man_width is total width including sign bit
        man_width = self.quant_config["man_width"]
        magnitude_bits = man_width - 1

        for i in range(per_block_mantissa.shape[0]):
            # Get unsigned mantissa scaled to integer
            # mantissa is in range [0, 1), scale by 2^magnitude_bits
            mantissa_int = (per_block_mantissa[i] * 2**magnitude_bits).int()

            # Get sign bits (1 for negative, 0 for positive)
            sign_bits = torch.where(
                per_block_sign[i] < 0, torch.tensor(1, dtype=torch.int), torch.tensor(0, dtype=torch.int)
            )

            # Pack: sign_bit << magnitude_bits | mantissa
            packed = (sign_bits << magnitude_bits) | mantissa_int
            block_list.append(packed.tolist())

            # Convert scaling (exponent) to biased unsigned representation
            biased_scale = int(per_block_scaling[i].item()) + exponent_bias
            scaling_list.append(biased_scale)

        return block_list, scaling_list
