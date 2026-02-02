import json
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import re

class sys_latency_config:
    def __init__(self, model_param_path, hardware_config, sys_config, batch_size = 1, seq_len = 2048, output_token = 128, device_num = 1):
        
        model_param = json.load(open(model_param_path))
        # Extract model size from the file name, e.g., ".../llama-3.1-8b-w8a8.json" -> 8
        model_filename = str(model_param_path).split('/')[-1]
        match = re.search(r'(\d+)[bB]', model_filename)
        if match:
            self.weight_size = int(match.group(1)) 
        else:
            self.weight_size = None  # or raise an error if model size is required
        
        self.hidden_size = model_param["hidden_size"]
        self.num_attention_heads = model_param["num_attention_heads"]
        self.num_hidden_layers = model_param["num_hidden_layers"]
        self.intermediate_size = model_param["intermediate_size"]
        self.num_key_value_heads = model_param["num_key_value_heads"]
        self.vocab_size = model_param["vocab_size"]
        self.input_seq_len = seq_len
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_head_groups = self.num_attention_heads // self.num_key_value_heads
        
        self.vocab_size = model_param["vocab_size"]      
        self.kv_datatype = 1
        self.act_datatype = 1
        self.wt_datatype = 1
        # Convert weight_size from billion to bytes, then to GB, where wt_datatype is the element size in bytes
        # Assume weight_size is in billions (e.g., 8 for 8B)
        self.weight_size = self.weight_size * 1e9 * self.wt_datatype / (1024**3)  # size in GB
        # print("model weight size: ", self.weight_size)
        self.theoratical_frequency = 10**9                  # 1 GHz
        self.hardware_config = hardware_config
        self.output_token = output_token
        self.batch_size = batch_size
        self.kv_size = 0
        self.device_num = device_num
        self.sram_latency   = sys_config["SRAM_Latency"] # nanoseconds
        self.sram_bandwidth = sys_config["SRAM_Bandwidth"] # GB/s
        self.sram_capacity  = sys_config["SRAM_Capacity"] # GB
        self.hbm_bandwidth  = sys_config["HBM_Bandwidth"] # GB/s
        self.hbm_latency    = sys_config["HBM_Latency"] # nanoseconds

        self.weight_bandwidth = None # GB/s
        self.weight_latency = None # nanoseconds
        self.kv_bandwidth = None # GB/s
        self.kv_latency = None # nanoseconds


    def sys_mem_dist_update(self):
        # Assuming the activations are stored on-chip SRAM, so excluded from the 3D Stacked Modelling.
        # KV are primarily stored on HBM, 
        kv_size = 2 * self.batch_size * self.num_key_value_heads * self.head_dim * (self.kv_size) * self.num_hidden_layers *self.kv_datatype
        kv_size = kv_size / 1024 / 1024 / 1024 # GB
        if kv_size > self.sram_capacity:
            self.kv_latency = ((kv_size - self.sram_capacity) * self.hbm_latency + self.sram_capacity * self.sram_latency) / kv_size
            self.kv_bandwidth = ((kv_size - self.sram_capacity) * self.hbm_bandwidth + self.sram_capacity * self.sram_bandwidth) / kv_size / self.kv_datatype
            self.weight_latency = self.hbm_latency
            self.weight_bandwidth = self.hbm_bandwidth
            # print("KV Cannot fit in 3D Stacked SRAM, using HBM")
        else:
            self.kv_latency = self.sram_latency
            self.kv_bandwidth = self.sram_bandwidth
            self.weight_latency = ((self.weight_size - (self.sram_capacity - kv_size)) * self.hbm_latency + (self.sram_capacity - kv_size) * self.sram_latency) / self.weight_size
            self.weight_bandwidth = ((self.weight_size - (self.sram_capacity - kv_size)) * self.hbm_bandwidth + (self.sram_capacity - kv_size) * self.sram_bandwidth) / self.weight_size / self.wt_datatype


    def rms_layer(self, mode = "prefill"):
        if mode == "prefill":
            setting_inst_num = 10
            loop_inst_num = 8
            loop_num = self.hidden_size // self.hardware_config["VLEN"]["value"]
            instruction_num = 0
            instruction_num += setting_inst_num
            instruction_num += loop_num * loop_inst_num * self.input_seq_len
        elif mode == "decode":
            setting_inst_num = 10
            loop_inst_num = 8
            loop_num = self.hidden_size // self.hardware_config["VLEN"]["value"]
            instruction_num = 0
            instruction_num += setting_inst_num
            instruction_num += loop_num * loop_inst_num            
        return instruction_num * self.batch_size

    def projection(self, mode = "prefill"):

        if self.weight_bandwidth > 2 * self.hardware_config["MLEN"]["value"]:
            mem_access_delay_ratio = 1
        else:
            mem_access_delay_ratio = 2 * self.hardware_config["MLEN"]["value"] / self.weight_bandwidth
            # print("Memory Bandwidth Bounded by Projection Write & Read Bandwidth")
        
        if mode == "prefill":
            overall_exe_cycle = 0
            # Q, K Projection + RoPE
            overall_exe_cycle += mem_access_delay_ratio * self.batch_size * (math.ceil(self.hidden_size / self.hardware_config["BLEN"]["value"]) * math.ceil(max(self.hidden_size, self.input_seq_len) / self.hardware_config["MLEN"]["value"]) * (math.ceil(min(self.hidden_size, self.input_seq_len) / self.hardware_config["BLEN"]["value"])  + self.weight_latency))
            overall_exe_cycle += mem_access_delay_ratio * self.batch_size * (self.num_attention_heads * (self.input_seq_len // self.hardware_config["VLEN"]["value"])) * 3

            overall_exe_cycle += mem_access_delay_ratio * self.batch_size * (math.ceil(self.hidden_size / self.hardware_config["BLEN"]["value"]) * math.ceil(max(self.hidden_size, self.input_seq_len) / self.hardware_config["MLEN"]["value"]) * (math.ceil(min(self.hidden_size, self.input_seq_len) / self.hardware_config["BLEN"]["value"])  + self.weight_latency))
            overall_exe_cycle += mem_access_delay_ratio * self.batch_size * (self.num_key_value_heads * (self.input_seq_len // self.hardware_config["VLEN"]["value"])) * 3

            # V
            overall_exe_cycle += mem_access_delay_ratio * self.batch_size * (math.ceil((self.num_key_value_heads * self.head_dim) / self.hardware_config["BLEN"]["value"]) * (math.ceil(self.hidden_size / self.hardware_config["MLEN"]["value"]) * (math.ceil(self.input_seq_len / self.hardware_config["BLEN"]["value"]) * 2 + self.weight_latency)))
        
        elif mode == "decode":
            overall_exe_cycle = 0
            # Q, K Projection + RoPE
            overall_exe_cycle +=  mem_access_delay_ratio * (math.ceil(self.hidden_size / self.hardware_config["BLEN"]["value"]) * (math.ceil(self.hidden_size / self.hardware_config["MLEN"]["value"]) + self.weight_latency))
            overall_exe_cycle += mem_access_delay_ratio * self.batch_size * (self.num_attention_heads) * 4

            overall_exe_cycle +=  mem_access_delay_ratio * (math.ceil((self.num_key_value_heads * self.head_dim) / self.hardware_config["BLEN"]["value"]) * (math.ceil(self.hidden_size / self.hardware_config["MLEN"]["value"]) + self.weight_latency))
            overall_exe_cycle += mem_access_delay_ratio * self.batch_size * (self.num_key_value_heads) * 4

            # V
            overall_exe_cycle += mem_access_delay_ratio * (math.ceil((self.num_key_value_heads * self.head_dim) / self.hardware_config["BLEN"]["value"]) * (math.ceil(self.hidden_size / self.hardware_config["MLEN"]["value"]) + self.weight_latency))
        
        return overall_exe_cycle
    
    def flash_attention(self, mode = "prefill"):
        overall_exe_cycle = 0
        mlen = self.hardware_config["MLEN"]["value"]
        blen = self.hardware_config["BLEN"]["value"]
        tile_in_atten = min(self.head_dim, mlen)
        assert self.num_key_value_heads <= (self.hardware_config["MLEN"]["value"] // self.hardware_config["BLEN"]["value"]) , "num_key_value_heads must be less than or equal to (MLEN // BLEN)"
        if self.kv_bandwidth > self.hardware_config["MLEN"]["value"]:
            mem_access_delay_ratio = 1
        else:
            mem_access_delay_ratio = self.hardware_config["MLEN"]["value"] / self.kv_bandwidth

        if mode == "prefill":
            # Outer loop
            for i in range(self.input_seq_len // self.hardware_config["MLEN"]["value"]):
                for j in range((self.input_seq_len // self.hardware_config["MLEN"]["value"])):
                    overall_exe_cycle += mem_access_delay_ratio * (4 + (mlen // blen) * 8 + 3) # MLEN * MLEN
                    overall_exe_cycle += 2 + mlen * 5  # Softmax
                    overall_exe_cycle += (9 + 4 + (math.ceil(tile_in_atten / blen)) * 8 + 3) #PV
                    overall_exe_cycle += (1+ tile_in_atten * 3) #Compute O
                    overall_exe_cycle += 8
                overall_exe_cycle += self.kv_latency
        elif mode == "decode":
            for j in range(math.ceil(self.kv_size / self.hardware_config["MLEN"]["value"]) ):
                overall_exe_cycle += mem_access_delay_ratio * (4 + (tile_in_atten // blen) * 4 + 3)
                overall_exe_cycle += 2 + mlen * 5  # Softmax
                overall_exe_cycle += math.ceil(self.head_dim / mlen) * (9 + 4 + math.ceil((tile_in_atten / blen)) * 8 + 3) #PV
                overall_exe_cycle += math.ceil(self.head_dim / mlen) * (1+ tile_in_atten * 3) #Compute O
                overall_exe_cycle += 8
                overall_exe_cycle += self.kv_latency
            self.kv_size = self.kv_size + 1
        return overall_exe_cycle * self.batch_size


    def residual (self, mode = "prefill"):
        overall_exe_cycle = 0
        # -- Residual
        if mode == "prefill":
            iteration = self.hidden_size // self.hardware_config["VLEN"]["value"]
            overall_exe_cycle = (5 * iteration + 3) * self.input_seq_len
        elif mode == "decode":
            iteration = self.hidden_size // self.hardware_config["VLEN"]["value"]
            overall_exe_cycle = 5 * iteration + 3
        return overall_exe_cycle

    def feed_forward(self, mode = "prefill"):
        mlen = self.hardware_config["MLEN"]["value"]
        vlen = self.hardware_config["VLEN"]["value"]
        blen = self.hardware_config["BLEN"]["value"]
        if self.weight_bandwidth > 2 * self.hardware_config["MLEN"]["value"]:
            mem_access_delay_ratio = 1
        else:
            mem_access_delay_ratio = 2 * self.hardware_config["MLEN"]["value"] / self.weight_bandwidth

        overall_exe_cycle = 0
        # -- MLP
        if mode == "prefill":
            overall_exe_cycle += 2 * math.ceil(self.intermediate_size / blen) * math.ceil(self.hidden_size / mlen) * 4 * (self.input_seq_len // blen)
            overall_exe_cycle += math.ceil(self.intermediate_size / vlen) * 5
            overall_exe_cycle += math.ceil(self.intermediate_size / blen) * math.ceil(self.hidden_size / mlen) * 4 * (self.input_seq_len // blen)
            overall_exe_cycle = mem_access_delay_ratio * overall_exe_cycle
        elif mode == "decode":
            overall_exe_cycle += 2 * math.ceil(self.intermediate_size / blen) * math.ceil(self.hidden_size / mlen) * 4
            overall_exe_cycle += math.ceil(self.intermediate_size / vlen) * 5
            overall_exe_cycle += math.ceil(self.intermediate_size / blen) * math.ceil(self.hidden_size / mlen) * 4
            overall_exe_cycle = mem_access_delay_ratio * overall_exe_cycle
        return overall_exe_cycle

    def embeddings(self, mode = "prefill"):
        mlen = self.hardware_config["MLEN"]["value"]
        vlen = self.hardware_config["VLEN"]["value"]
        blen = self.hardware_config["BLEN"]["value"]
        overall_exe_cycle = 3
        if mode == "prefill":
            overall_exe_cycle += self.input_seq_len * math.ceil(self.hidden_size / blen) * math.ceil(self.hidden_size / mlen) * (blen * 2 + 1) + 4
        elif mode == "decode":
            overall_exe_cycle += math.ceil(self.hidden_size / blen) * math.ceil(self.hidden_size / mlen) * (blen * 2 + 1) + 4
        return overall_exe_cycle
    
    def lm_head(self):
        mlen = self.hardware_config["MLEN"]["value"]
        vlen = self.hardware_config["VLEN"]["value"]
        blen = self.hardware_config["BLEN"]["value"]
        overall_exe_cycle = 3
        overall_exe_cycle += (math.ceil(self.hidden_size / blen) * math.ceil(self.vocab_size / mlen) * (blen * 2 + 1) + 4)
        return overall_exe_cycle

    def compute_prefill_time(self):
        mode = "prefill"
        self.kv_size = 0
        self.sys_mem_dist_update()
        overall_exe_cycle = 0
        overall_exe_cycle += self.embeddings(mode)
        for i in range(self.num_hidden_layers):
            overall_exe_cycle += self.rms_layer(mode)
            overall_exe_cycle += self.projection(mode)
            overall_exe_cycle += self.flash_attention(mode)
            overall_exe_cycle += self.residual(mode)
            overall_exe_cycle += self.rms_layer(mode)
            overall_exe_cycle += self.feed_forward(mode)

        # overall_exe_cycle += self.rms_layer()
        overall_exe_cycle += self.lm_head()
        # print("Overall instruction number: ", overall_exe_cycle)
        overall_exe_cycle = overall_exe_cycle * 1
        theoratical_execution_time = overall_exe_cycle / self.theoratical_frequency
        # print("Theoratical execution time: ", theoratical_execution_time)
        return theoratical_execution_time

    def compute_decode_time(self, output_token_size):
        mode = "decode"
        overall_exe_cycle = 0
        self.kv_size = self.input_seq_len
        for j in range (output_token_size):
            self.sys_mem_dist_update()
            for i in range (self.num_hidden_layers):
                overall_exe_cycle += self.rms_layer(mode)
                overall_exe_cycle += self.projection(mode)
                overall_exe_cycle += self.flash_attention(mode)
                overall_exe_cycle += self.residual(mode)
                overall_exe_cycle += self.rms_layer(mode)
                overall_exe_cycle += self.feed_forward(mode)
            self.kv_size = self.kv_size + 1
        # print("Overall instruction number: ", overall_exe_cycle)
        overall_exe_cycle = overall_exe_cycle # avg 3 execution cycles
        theoratical_execution_time = overall_exe_cycle / self.theoratical_frequency
        # print("Theoratical execution time: ", theoratical_execution_time)
        return theoratical_execution_time


    def compute_overall_perf(self):
        kv_size = 2 * self.batch_size * self.num_key_value_heads * self.head_dim * (self.input_seq_len + self.output_token) * self.num_hidden_layers *self.kv_datatype
        kv_size = kv_size / 1024 / 1024 / 1024 # GB
        print(f"KV size: {kv_size}GB, Input Sequence Length: {self.input_seq_len}, Output Token: {self.output_token}")
        if kv_size > self.sram_capacity:
            print("KV Cannot fit in 3D Stacked SRAM, using HBM")
        else:
            print("KV fits in 3D Stacked SRAM")
        
        ttft = (self.compute_prefill_time() + self.compute_decode_time(1)) / self.device_num
        tps = (self.device_num * (self.batch_size * self.output_token)) / self.compute_decode_time(self.output_token // self.device_num)
        return ttft, tps






        

