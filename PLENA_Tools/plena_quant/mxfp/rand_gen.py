"""Random tensor generator for MXFP quantization."""

import torch
import os

from .quantizer import _mx_fp_quantize_hardware
from .utils import pack_fp_to_bin


class Random_MXFP_Tensor_Generator:
    def __init__(self, shape, quant_config, config_settings=None, directory=None, filename=None):
        """
        Initialize the random tensor generator with a given shape in MXFP.
        If directory and filename are provided, the tensor will be saved to a file.
        """
        self.shape = shape
        self.directory = directory
        self.filename = filename
        self.quant_config = quant_config
        self.config_settings = config_settings or {}

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
        Quantize tensor to MXFP format.
        Returns (block_list, scaling_list).
        """
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        _bm_x, per_block_exponent, per_block_mantissa, per_block_scaling = _mx_fp_quantize_hardware(
            tensor,
            width=self.quant_config["exp_width"] + self.quant_config["man_width"] + 1,
            exponent_width=self.quant_config["exp_width"],
            exponent_bias_width=self.quant_config["exp_bias_width"],
            block_size=self.quant_config["block_size"],
            skip_first_dim=self.quant_config["skip_first_dim"],
        )

        block_list = []
        scaling_list = []

        for i in range(per_block_mantissa.shape[0]):
            bin_block = pack_fp_to_bin(
                per_block_exponent[i],
                per_block_mantissa[i],
                self.quant_config["exp_width"],
                self.quant_config["man_width"],
            )
            block_list.append(bin_block.tolist())
            scaling_list.append(int(per_block_scaling[i]))

        return block_list, scaling_list
