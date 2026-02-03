import torch
import os
import collections
import numpy as np
from utils.torch_fp_conversion import pack_fp_to_bin
from utils.debugger import set_excepthook
from utils.logger import set_logging_verbosity, get_logger
from quant.quantizer.hardware_quantizer import _mx_fp_quantize_hardware, _minifloat_ieee_quantize_hardware
from quant.quantizer.hardware_quantizer.mxint import _mx_int_quantize_hardware

logger = get_logger("test_bin_mxfp")
# set_logging_verbosity("debug")
set_logging_verbosity("warning")
set_excepthook()

class Random_MXINT_Tensor_Generator:
    def __init__(self, shape, quant_config, directory=None, filename=None):
        """
        Initialize the random tensor generator with a given shape in MXFP.
        If directory and filename are provided, the tensor will be saved to a file.
        """
        self.shape          = shape
        self.directory      = directory
        self.filename       = filename
        self.quant_config   = quant_config

    def tensor_gen(self):
        tensor = torch.randn(self.shape)
        if self.directory and self.filename:
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            file_path = os.path.join(self.directory, self.filename)
            torch.save(tensor, file_path)
            logger.debug(f"Tensor saved to {file_path}")
    
    def tensor_load(self):
        if self.directory and self.filename:
            file_path = os.path.join(self.directory, self.filename)
            if os.path.exists(file_path):
                tensor = torch.load(file_path)
                logger.debug(f"Tensor loaded from {file_path}")
                return tensor
            else:
                logger.error(f"File {file_path} does not exist.")
                return None
        else:
            logger.error("Directory and filename must be specified to load the tensor.")
            return None

    def quantize_tensor(self, tensor):
        '''
        note in 2d case the tensor of the original shape [shape_1, shape_2]
        the output bm_x will keep the orignal shape
        but the per_block * will be packed as showns
        [shape_1 * shaped_2 // (block_size[0] * block_size[1]), block_size[0] * block_size[1]]
        '''
        bm_x, per_block_exponent, per_block_mantissa, per_block_scaling = _mx_int_quantize_hardware(
            tensor,
            width               = self.quant_config["man_width"],
            exponent_width      = self.quant_config["exp_width"],
            block_size          = self.quant_config["block_size"],
            skip_first_dim      = self.quant_config["skip_first_dim"],
        )

        logger.debug(f"per_block_mantissa: {per_block_mantissa.shape}")
        logger.debug(f"per_block_exponent: {per_block_exponent.shape}")
        logger.debug(f"per_block_quant_bias: {per_block_scaling.shape}")

        block_list  = []
        scaling_list   = []

        for i in range(per_block_mantissa.shape[0]):
            block_list.append(per_block_mantissa[i] * 2**(self.quant_config["man_width"] - 1).tolist())
            scaling_list.append(int(per_block_scaling[i]))
            # note here the block_mantissa was represented as unsigned integer
            # the exponent was represented as signed integer
        logger.debug(f"block_list: {block_list}")
        logger.debug(f"scaling_list: {scaling_list}")

        return block_list, scaling_list


class Random_MXFP_Tensor_Generator:
    def __init__(self, shape, quant_config, config_settings, directory=None, filename=None):
        """
        Initialize the random tensor generator with a given shape.
        If directory and filename are provided, the tensor will be saved to a file.
        """
        self.shape          = shape
        self.directory      = directory
        self.filename       = filename
        self.quant_config   = quant_config
        self.config_settings = config_settings

    def tensor_gen(self):
        tensor = torch.randn(self.shape)
        if self.directory and self.filename:
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            file_path = os.path.join(self.directory, self.filename)
            torch.save(tensor, file_path)
            logger.debug(f"Tensor saved to {file_path}")

            # Change extension to .mem
            base_name = os.path.splitext(self.filename)[0]
            rd_file_path = os.path.join(self.directory, f"{base_name}.mem")
            
            # Save as human-readable hex format
            with open(rd_file_path, 'w') as f:
                flat_tensor = tensor.flatten().numpy().astype(np.float32)
                floats_per_line = self.config_settings["HBM_WIDTH"] // 4
                for i, value in enumerate(flat_tensor):
                    if i % floats_per_line == 0 and i != 0:
                        f.write('\n')
                    f.write(f"{value:.8f}")
                    if (i + 1) % floats_per_line != 0 and i < len(flat_tensor) - 1:
                        f.write(', ')
    def tensor_load(self):
        if self.directory and self.filename:
            file_path = os.path.join(self.directory, self.filename)
            # print("loading file", file_path)
            if os.path.exists(file_path):
                tensor = torch.load(file_path)
                logger.debug(f"Tensor loaded from {file_path}")
                return tensor
            else:
                logger.error(f"File {file_path} does not exist.")
                return None
        else:
            logger.error("Directory and filename must be specified to load the tensor.")
            return None
            
    def quantize_tensor(self, tensor):
        '''
        note in 2d case the tensor of the original shape [shape_1, shape_2]
        the output bm_x will keep the orignal shape
        but the per_block * will be packed as showns
        [shape_1 * shaped_2 // (block_size[0] * block_size[1]), block_size[0] * block_size[1]]
        '''
        # INSERT_YOUR_CODE
        # Accept tensor as either a torch.Tensor or an OrderedDict of tensors
        # If it's an OrderedDict, quantize each value (assuming each is a tensor), otherwise just quantize the single tensor
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
            print("reshaped to", tensor.shape)

        bm_x, per_block_exponent, per_block_mantissa, per_block_scaling = _mx_fp_quantize_hardware(
            tensor,
            width               = self.quant_config["exp_width"] + self.quant_config["man_width"] + 1,
            exponent_width      = self.quant_config["exp_width"],
            exponent_bias_width = self.quant_config["exp_bias_width"],
            block_size          = self.quant_config["block_size"],
            skip_first_dim      = self.quant_config["skip_first_dim"],
        )

        logger.debug(f"per_block_mantissa: {per_block_mantissa.shape}")
        logger.debug(f"per_block_exponent: {per_block_exponent.shape}")
        logger.debug(f"per_block_quant_bias: {per_block_scaling.shape}")

        inner_block_list  = []
        inner_scaling_list   = []

        for i in range(per_block_mantissa.shape[0]):
            bin_block = pack_fp_to_bin(
                per_block_exponent[i],
                per_block_mantissa[i],
                self.quant_config["exp_width"],
                self.quant_config["man_width"],
            )
            inner_block_list.append(bin_block.tolist())
            inner_scaling_list.append(int(per_block_scaling[i]))
            # note here the block_mantissa was represented as unsigned integer
            # the exponent was represented as signed integer

        # block_list.append(inner_block_list)
        # scaling_list.append(inner_scaling_list)
        return inner_block_list, inner_scaling_list


if __name__ == "__main__":
    pass