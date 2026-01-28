# TODO: Write Function that automatically map the FIXED and FP memory.
from utils import load_svh_settings, load_json
import os
from pathlib import Path

class fix_sram_pre_loader:
    """This class is used to prepare data in FIXED SRAM before running the program."""
    def __init__(self, architecture_feature, ml_feature, precision_feature,directory):
        self.architecture_feature = architecture_feature
        self.directory = os.path.join(directory, "fixed.mem")
        self.ml_feature = ml_feature
        self.precision_feature = precision_feature

    def load(self):
        low_precision_stride_length = (self.ml_feature["hidden_size"] * self.architecture_feature["MLEN"] * 
                         (self.precision_feature["WT_MX_MANT_WIDTH"] + self.precision_feature["WT_MX_MANT_WIDTH"])) // 8
        high_precision_stride_length = (self.ml_feature["hidden_size"] * self.architecture_feature["MLEN"] * 
                         (self.precision_feature["ACT_MXFP_MANT_WIDTH"] + self.precision_feature["ACT_MXFP_MANT_WIDTH"])) // 8
        
        MLEN = (self.architecture_feature["MLEN"])
        Q_Size = (self.ml_feature["batchsize"] * self.ml_feature["hidden_size"] * self.ml_feature["max_position_embeddings"] * (self.precision_feature["ACT_MXFP_MANT_WIDTH"] + self.precision_feature["ACT_MXFP_MANT_WIDTH"])) // 8
        KV_Size = (self.ml_feature["batchsize"] * self.ml_feature["hidden_size"] * self.ml_feature["max_position_embeddings"] * (self.precision_feature["KV_MX_MANT_WIDTH"] + self.precision_feature["KV_MX_MANT_WIDTH"])) // 8
        Weight_Size = (self.ml_feature["hidden_size"] * self.ml_feature["hidden_size"] * (self.precision_feature["WT_MX_MANT_WIDTH"] + self.precision_feature["WT_MX_MANT_WIDTH"])) // 8
        Batch_Size = (self.ml_feature["batchsize"])
        Head_Dim = (self.ml_feature["hidden_size"] // self.ml_feature["num_attention_heads"])
        
        # FlashAtten
        with open(self.directory, "w") as f:
            f.write(f"0x{high_precision_stride_length:08x}\n")
            f.write(f"0x{low_precision_stride_length:08x}\n")
            f.write(f"0x{MLEN:08x}\n")
            f.write(f"0x{2*MLEN:08x}\n")
            f.write(f"0x{Q_Size:08x}\n")
            f.write(f"0x{KV_Size:08x}\n")
            f.write(f"0x{Weight_Size:08x}\n")
            f.write(f"0x{Head_Dim:08x}\n")
            f.write(f"0x{Batch_Size:08x}\n")



class MemPrepare:
    '''
    This class is used to prepare data to the target shape before running the program.
    '''
    def __init__(self,):
        self.architecture_feature = architecture_feature
        self.ml_feature = ml_feature
        self.precision_feature = precision_feature
        self.directory = directory

    def quantize(self):
        pass

    def reshape(self):
        pass

    def reshape(self):
        pass
        
        

class fp_sram_pre_loader:
    pass


class fake_hbm_pre_loader:
    """This class is used to prepare data in fake HBM before running the program."""
    def __init__(self, hbm_size):
        self.hbm_size = hbm_size

    def load(self):
        # This function should implement the logic to pre-load the fake HBM
        # For now, we will just print the size
        print(f"Pre-loading fake HBM with size: {self.hbm_size}")


if __name__ == "__main__":
    project_path = Path(__file__).resolve().parents[2]
    hardware_feature_path = os.path.join(project_path, "src/definitions/configuration.svh")
    print(f"Loading hardware feature from {hardware_feature_path}")
    architecture_feature = load_svh_settings(hardware_feature_path)
    ml_feature_path = os.path.join(project_path, "doc/Model_Lib/llama-3.1-8b.json")
    ml_feature = load_json(ml_feature_path)
    precision_feature_path = os.path.join(project_path, "src/definitions/precision.svh")
    precision_feature = load_svh_settings(precision_feature_path)
    directory = os.path.join(project_path, "test/load_mem")

    fixed_sram_loader = fix_sram_pre_loader(architecture_feature, ml_feature, precision_feature, directory)
    fixed_sram_loader.load()