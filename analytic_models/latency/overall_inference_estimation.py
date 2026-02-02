import json
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


class model_config:
    def __init__(self, model_param_path, hardware_config, batch_size = 1, seq_len = 2048, output_token = 128, device_num = 1):
        model_param = json.load(open(model_param_path))
        self.hidden_size = model_param["hidden_size"]
        self.num_attention_heads = model_param["num_attention_heads"]
        self.num_hidden_layers = model_param["num_hidden_layers"]
        self.intermediate_size = model_param["intermediate_size"]
        self.num_key_value_heads = model_param["num_key_value_heads"]
        self.vocab_size = model_param["vocab_size"]
        self.input_token = seq_len
        self.output_token = output_token
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_head_groups = self.num_attention_heads // self.num_key_value_heads
        self.vocab_size = model_param["vocab_size"]
        self.DataTypeSize = 2
        self.theoratical_frequency = 10**9          # 1 GHz
        self.hardware_config = hardware_config
        self.batch_size = batch_size
        self.device_batch_size = batch_size // device_num
        self.kv_size = seq_len
        self.device_num = device_num
        print("=" * 15, "Model Settings","=" * 15)
        print("hardware config: \n", self.hardware_config)
        print("batch size: ", self.batch_size)
        print("input token: ", self.input_token)
        print("output token: ", self.output_token)
        print("hidden size: ", self.hidden_size)
        print("head dim: ", self.head_dim)
        print("num key value heads: ", self.num_key_value_heads)
        print("num attention heads: ", self.num_attention_heads)
        print("num hidden layers: ", self.num_hidden_layers)
        print("intermediate size: ", self.intermediate_size)
        print("vocab size: ", self.vocab_size)
        print("=" * 25)

    def rms_layer(self, mode = "prefill"):
        if mode == "prefill":
            setting_inst_num = 10
            loop_inst_num = 8
            loop_num = self.hidden_size // self.hardware_config["VLEN"]
            instruction_num = 0
            instruction_num += setting_inst_num
            instruction_num += loop_num * loop_inst_num * self.input_token
        elif mode == "decode":
            setting_inst_num = 10
            loop_inst_num = 8
            loop_num = self.hidden_size // self.hardware_config["VLEN"]
            instruction_num = 0
            instruction_num += setting_inst_num
            instruction_num += loop_num * loop_inst_num            
        return instruction_num * self.device_batch_size

    def projection(self, mode = "prefill"):
        # Compute Q, K, V Projection + RoPE
        if mode == "prefill":
            overall_inst_num = 0
            # Q, K Projection + RoPE
            overall_inst_num += self.device_batch_size * (math.ceil(self.hidden_size / self.hardware_config["BLEN"]) * (math.ceil(self.hidden_size / self.hardware_config["MLEN"]) * (math.ceil(self.input_token / self.hardware_config["BLEN"]) * 2 + 10)))
            overall_inst_num += self.device_batch_size * (self.num_attention_heads * math.ceil(self.input_token / self.hardware_config["VLEN"])) * 3

            overall_inst_num += self.device_batch_size * (math.ceil((self.num_key_value_heads * self.head_dim) / self.hardware_config["BLEN"]) * (math.ceil(self.hidden_size / self.hardware_config["MLEN"]) * (math.ceil(self.input_token / self.hardware_config["BLEN"]) * 2 + 10)))
            overall_inst_num += self.device_batch_size * (self.num_key_value_heads * math.ceil(self.input_token / self.hardware_config["VLEN"])) * 3

            # V
            overall_inst_num += self.device_batch_size * (math.ceil((self.num_key_value_heads * self.head_dim) / self.hardware_config["BLEN"]) * (math.ceil(self.hidden_size / self.hardware_config["MLEN"]) * (math.ceil(self.input_token / self.hardware_config["BLEN"]) * 2 + 10)))
        
        elif mode == "decode":
            overall_inst_num = 0
            # Q, K Projection + RoPE
            overall_inst_num +=  (math.ceil(self.hidden_size / self.hardware_config["BLEN"]) * (math.ceil(self.hidden_size / self.hardware_config["MLEN"]) * 2 + 10))
            overall_inst_num += self.device_batch_size * (self.num_attention_heads) * 4

            overall_inst_num +=  (math.ceil((self.num_key_value_heads * self.head_dim) / self.hardware_config["BLEN"]) * (math.ceil(self.hidden_size / self.hardware_config["MLEN"]) * 2 + 10))
            overall_inst_num += self.device_batch_size * (self.num_key_value_heads) * 4

            # V
            overall_inst_num += (math.ceil((self.num_key_value_heads * self.head_dim) / self.hardware_config["BLEN"]) * (math.ceil(self.hidden_size / self.hardware_config["MLEN"]) * 2 + 10))
        
        return overall_inst_num
    
    def flash_attention(self, mode = "prefill", partitoned_optimised = True):
        overall_inst_num = 0
        mlen = self.hardware_config["MLEN"]
        blen = self.hardware_config["BLEN"]
        prefill_len = min(self.input_token, mlen)
        decode_len = min(self.kv_size, mlen)
        max_head_per_mlen = math.ceil(mlen / self.head_dim)

        per_head_iter = math.ceil(self.num_attention_heads / max_head_per_mlen)

        if mode == "prefill":
        # Outer loop
            if partitoned_optimised:
                for i in range(math.ceil(self.input_token / mlen)):
                    overall_inst_num += mlen * max_head_per_mlen # Reset                    
                    for j in range(math.ceil(self.input_token / mlen)):
                        # overall_inst_num += math.ceil(prefill_len / blen) * blen * math.ceil(prefill_len / blen) * 5 # QKT
                        overall_inst_num += (2 + prefill_len * 5) * max_head_per_mlen  # Softmax
                        overall_inst_num += math.ceil(mlen / blen) * blen * math.ceil(self.head_dim / blen) * max_head_per_mlen
                        overall_inst_num += (prefill_len * 5 + 4) * max_head_per_mlen# Compute O
                        overall_inst_num += 8 * max_head_per_mlen
                self.kv_size = self.input_token
                return overall_inst_num * per_head_iter * self.device_batch_size
            else:
                for i in range(math.ceil(self.input_token / mlen)):
                    # overall_inst_num += mlen # Reset                    
                    for j in range(math.ceil(self.input_token / mlen)):
                        # (mlen, head_dim) @ (head_dim, mlen)
                        overall_inst_num += math.ceil(prefill_len / blen) * (math.ceil(self.head_dim / mlen)) * blen * math.ceil(prefill_len / blen)
                        overall_inst_num += 2 + prefill_len * 5  # Softmax
                        overall_inst_num += math.ceil(mlen / blen) * blen * math.ceil(self.head_dim / blen)
                        overall_inst_num += prefill_len * 5 + 4 # Compute O
                        overall_inst_num += 8
                return overall_inst_num * self.device_batch_size * self.num_attention_heads
        elif mode == "decode":
            if partitoned_optimised:
                overall_inst_num = decode_len
                overall_inst_num += (2 + decode_len * 3) * max_head_per_mlen # Softmax
                # PV (1, decode_len) @ (decode_len, head_dim)
                overall_inst_num += math.ceil(decode_len / mlen) * blen * math.ceil(self.head_dim / blen) * max_head_per_mlen #PV
                overall_inst_num += (decode_len * 3 + 4) * max_head_per_mlen #Compute O
                overall_inst_num += 8 * max_head_per_mlen
                return overall_inst_num * self.device_batch_size * math.ceil(self.kv_size / self.hardware_config["MLEN"]) * (per_head_iter)
            else:
                overall_inst_num = math.ceil(mlen / blen) * (math.ceil(self.head_dim / mlen) * blen)
                overall_inst_num += (2 + decode_len * 3)
                overall_inst_num += math.ceil(self.head_dim / blen) * (4 + math.ceil(decode_len / blen) * 4)
                overall_inst_num += (decode_len * 3 + 4)
                overall_inst_num += 8
                return overall_inst_num * self.device_batch_size * math.ceil(self.kv_size / self.hardware_config["MLEN"]) * self.num_attention_heads


    def self_attention(self, mode = "prefill"):
        # Note: this it the latency estimation model for self-attention without using the flash attention algorithm
        overall_inst_num = 0
        mlen = self.hardware_config["MLEN"]
        blen = self.hardware_config["BLEN"]
        vlen = self.hardware_config["VLEN"]
        if mode == "prefill":
            for i in range(self.num_attention_heads):
                # QKT (batch, s, h, d) @ (batch, s, h, d)
                overall_inst_num += (math.ceil(self.input_token / blen)) * math.ceil(self.head_dim / mlen) * blen * math.ceil(self.input_token / blen) * 4
            # Store QKT (h, s, s)
            overall_inst_num += self.input_token * 4 * (math.ceil(self.input_token / mlen)) * self.num_attention_heads * 2
                # Softmax (batch, s, h, d) 
            overall_inst_num += self.input_token * (math.ceil(self.input_token / mlen)) * 30 * self.num_attention_heads
            # PV (h, s, s) @ (h, s, d)
            for i in range(self.num_attention_heads):
                overall_inst_num += math.ceil(self.input_token / blen) * math.ceil(self.input_token / mlen) * blen * math.ceil(self.head_dim / blen)
            # Store PV (h, s, d)
            overall_inst_num += self.head_dim * 4 * (math.ceil(self.input_token / mlen)) * self.num_attention_heads* 2
            # Compute O (batch, s, h, d)
            for i in range(self.num_attention_heads):
                overall_inst_num += (math.ceil(self.input_token / vlen)) * 5 + 4
            overall_inst_num += 8
            overall_inst_num = overall_inst_num * self.device_batch_size
        elif mode == "decode":
            for i in range(self.num_attention_heads):
                # QKT (batch, 1, h, d) @ (batch, kv, h, d)
                overall_inst_num += 4 * math.ceil(self.head_dim / mlen) * math.ceil(self.kv_size / blen) * blen
            # Store QKT (h, s, s)
            overall_inst_num += 4 * math.ceil(self.kv_size / mlen) * self.num_attention_heads * 36
            # Softmax (batch, s, h, d) 
            overall_inst_num += (math.ceil(self.kv_size / mlen)) * 30 * self.num_attention_heads
            # PV (h, 1, s) @ (h, s, d)
            for i in range(self.num_attention_heads):
                overall_inst_num += math.ceil(self.kv_size / mlen) * blen * (self.head_dim // blen) * 2
            # Compute O (batch, s, h, d)
            for i in range(self.num_attention_heads):
                overall_inst_num += (math.ceil(self.kv_size / mlen)) * 5 + 4
            overall_inst_num = overall_inst_num * self.device_batch_size
        return overall_inst_num

    def residual (self, mode = "prefill"):
        overall_inst_num = 0
        # -- Residual
        if mode == "prefill":
            iteration = self.hidden_size // self.hardware_config["VLEN"]
            overall_inst_num = (5 * iteration + 3) * self.input_token
        elif mode == "decode":
            iteration = self.hidden_size // self.hardware_config["VLEN"]
            overall_inst_num = 5 * iteration + 3
        return overall_inst_num * self.device_batch_size

    def feed_forward(self, mode = "prefill"):
        mlen = self.hardware_config["MLEN"]
        vlen = self.hardware_config["VLEN"]
        blen = self.hardware_config["BLEN"]

        overall_inst_num = 0
        # -- MLP
        if mode == "prefill":
            # Upprojection, and Gate (seq, hidden) @ (hidden, intermediate)
            overall_inst_num += 2 * math.ceil(self.intermediate_size / blen) * math.ceil(self.input_token / blen) * math.ceil(self.hidden_size / mlen) * blen
            overall_inst_num += math.ceil(self.intermediate_size / vlen) * 3 * self.input_token
            # Downprojection (seq, intermediate) @ (intermediate, hidden)
            overall_inst_num += math.ceil(self.intermediate_size / blen) * math.ceil(self.hidden_size / mlen)  * math.ceil(self.input_token / blen) * blen
            overall_inst_num = (overall_inst_num) * self.device_batch_size
        elif mode == "decode":
            overall_inst_num += 2 * math.ceil(self.intermediate_size / blen) * math.ceil(self.hidden_size / mlen) * (math.ceil(self.device_batch_size / blen) ) * blen * math.ceil(self.device_batch_size / blen)
            overall_inst_num += math.ceil(self.intermediate_size / vlen) * 5 * self.device_batch_size
            overall_inst_num += math.ceil(self.intermediate_size / blen) * math.ceil(self.hidden_size / mlen) * (math.ceil(self.device_batch_size / blen) ) * blen * math.ceil(self.device_batch_size / blen)
        return overall_inst_num

    def embeddings(self, mode = "prefill"):
        mlen = self.hardware_config["MLEN"]
        vlen = self.hardware_config["VLEN"]
        blen = self.hardware_config["BLEN"]
        overall_inst_num = 3
        if mode == "prefill":
            overall_inst_num += self.input_token * math.ceil(self.hidden_size / blen) * math.ceil(self.hidden_size / mlen) * (blen * 2 + 1) + 4
        elif mode == "decode":
            overall_inst_num += math.ceil(self.hidden_size / blen) * math.ceil(self.hidden_size / mlen) * (blen * 2 + 1) + 4
        return overall_inst_num
    
    def lm_head(self):
        mlen = self.hardware_config["MLEN"]
        vlen = self.hardware_config["VLEN"]
        blen = self.hardware_config["BLEN"]
        overall_inst_num = 3
        overall_inst_num += (math.ceil(self.hidden_size / blen) * math.ceil(self.vocab_size / mlen) * (blen * 2 + 1) + 4)
        return overall_inst_num

    def compute_prefill_time(self):
        mode = "prefill"
        overall_inst_num = 0
        overall_inst_num += self.embeddings(mode)
        for i in range(self.num_hidden_layers):
            overall_inst_num += self.rms_layer(mode)
            overall_inst_num += self.projection(mode)
            overall_inst_num += self.flash_attention(mode)
            overall_inst_num += self.residual(mode)
            overall_inst_num += self.rms_layer(mode)
            overall_inst_num += self.feed_forward(mode)
        # overall_inst_num += self.residual(mode)
        # overall_inst_num += self.rms_layer()
        overall_inst_num += self.lm_head()
        overall_exe_cycle = overall_inst_num * 2
        theoratical_execution_time = overall_exe_cycle / self.theoratical_frequency
        print("\n")
        print("=" * 5,"Prefill Theoratical execution distribution: ","=" * 5)
        print(f"RMS Layer: {self.rms_layer(mode) * self.num_hidden_layers / overall_inst_num * 100}%")
        print(f"Projection: {self.projection(mode) * self.num_hidden_layers / overall_inst_num * 100}%")
        print(f"Flash Attention: {self.flash_attention(mode) * self.num_hidden_layers / overall_inst_num * 100}%")
        print(f"Residual: {self.residual(mode) * self.num_hidden_layers / overall_inst_num * 100}%")
        print(f"Feed Forward: {self.feed_forward(mode) * self.num_hidden_layers / overall_inst_num * 100}%")
        print(f"LM Head: {self.lm_head() / overall_inst_num * 100}%")
        return theoratical_execution_time

    def compute_decode_time(self, output_token_size):
        mode = "decode"
        overall_inst_num = 0
        rms_count = 0
        projection_count = 0
        flash_attention_count = 0
        residual_count = 0
        feed_forward_count = 0
        for j in range (output_token_size):
            for i in range (self.num_hidden_layers):
                rms_count += self.rms_layer(mode)
                projection_count += self.projection(mode)
                flash_attention_count += self.flash_attention(mode)
                residual_count += self.residual(mode)
                rms_count += self.rms_layer(mode)
                feed_forward_count += self.feed_forward(mode)
            self.kv_size = self.kv_size + 1
        overall_inst_num = rms_count + projection_count + flash_attention_count + residual_count + feed_forward_count
        overall_exe_cycle = overall_inst_num * 2 # avg 2 execution cycles
        theoratical_execution_time = overall_exe_cycle / self.theoratical_frequency
        print("\n")
        print("=" * 5,"Decode Theoratical execution distribution: ","=" * 5)
        print(f"RMS Layer: {rms_count / overall_inst_num * 100}%")
        print(f"Projection: {projection_count / overall_inst_num * 100}%")
        print(f"Flash Attention: {flash_attention_count / overall_inst_num * 100}%")
        print(f"Residual: {residual_count / overall_inst_num * 100}%")
        print(f"Feed Forward: {feed_forward_count / overall_inst_num * 100}%")
        return theoratical_execution_time

    def compute_overall_perf(self):
        # Per batch performance.
        ttft = (self.compute_prefill_time() + self.compute_decode_time(1)) / self.device_num
        tps = (self.batch_size * self.output_token) / self.compute_decode_time(self.output_token)
        return ttft, tps