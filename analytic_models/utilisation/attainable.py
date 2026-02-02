import os
from typing import Dict, List, Any, Optional
import json
import math

class attn_model_config:
    def __init__(self, MLEN, BLEN, VLEN,partitioned_matrix, model_param_path, batch_size = 1, seq_len = 2048, output_token = 128, device_num = 1):
        print(f"Model param path: {model_param_path}")
        model_param = json.load(open(model_param_path))
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
        self.DataTypeSize = 2
        self.theoratical_frequency = 10**9          # 1 GHz
        self.output_token = output_token
        self.batch_size = batch_size
        print("=" * 25)
        print(f"MLEN: {MLEN}, BLEN: {BLEN}, VLEN: {VLEN}")
        print(f"Batch size: {self.batch_size}")
        print(f"num_attention_heads: {self.num_attention_heads}, num_key_value_heads: {self.num_key_value_heads}")
        print(f"head_dim: {self.head_dim}, num_head_groups: {self.num_head_groups}")
        self.kv_size = seq_len
        self.device_num = device_num
        self.MLEN = MLEN
        self.BLEN = BLEN
        self.VLEN = VLEN
        self.partitioned_matrix = partitioned_matrix
        self.max_head_per_mlen = math.ceil(self.MLEN / self.head_dim)

    def _report_flash_attn_utilization(self, mode = "prefill") -> None:
        """
        Report the utilization of flash attention for a given node.
        """
        batch_size = self.batch_size
        hidden_size = self.hidden_size
        num_attn_heads = self.num_attention_heads
        num_kv_heads = self.num_key_value_heads

        head_dim = self.head_dim
        input_token_size = self.input_seq_len
        theoretical_operation = 0
        attainable_operation = 0
        attainable_gemm_operation_amount = 0
        overall_operation_amount = 0
        bubble_ratio = 1.2
        
        # Decoding
        if mode == "prefill":
            # Projection
            gemm_operation_amount = (self.input_seq_len // self.BLEN) * ( hidden_size / self.MLEN) * self.BLEN * ((head_dim * num_attn_heads)  // self.BLEN) + ((head_dim * num_kv_heads) // self.BLEN) * ( hidden_size / self.MLEN) * self.BLEN * (self.input_seq_len // self.BLEN) * 2
            attainable_gemm_operation_amount = gemm_operation_amount * batch_size
            attention_operation_amount = gemm_operation_amount * bubble_ratio * batch_size

            # Flash Attention
            if self.partitioned_matrix:
                for b in range(batch_size):
                    for i in range(math.ceil(num_attn_heads / self.max_head_per_mlen)):
                        for j in range(math.ceil(self.input_seq_len / self.MLEN)):
                            for k in range(math.ceil(self.input_seq_len / self.MLEN)):
                                # QKT (MLEN, Hq) * (Hq, MLEN),  (num_attn_heads / (self.MLEN // self.head_dim)) in parallel
                                gemm_operation_amount = self.head_dim # need to clarify this, coz the actual implementation is not like this
                                attention_operation_amount += gemm_operation_amount * bubble_ratio # Prefetch Data
                                if num_attn_heads > self.max_head_per_mlen:
                                    attainable_gemm_operation_amount += gemm_operation_amount * 1
                                else:
                                    attainable_gemm_operation_amount += gemm_operation_amount * (head_dim * num_attn_heads / self.MLEN)
                                # Softmax
                                attention_operation_amount += self.MLEN * 10
                                # PV (MLEN, MLEN) @ (MLEN, head_dim),  (num_attn_heads / (self.MLEN // self.head_dim)) in parallel
                                gemm_operation_amount = math.ceil(self.MLEN / self.BLEN) * self.BLEN * math.ceil(head_dim / self.BLEN) * self.num_head_groups
                                attention_operation_amount += gemm_operation_amount * bubble_ratio
                                if head_dim > self.BLEN:
                                    attainable_gemm_operation_amount += gemm_operation_amount * 1
                                else:
                                    attainable_gemm_operation_amount += gemm_operation_amount * (head_dim / self.BLEN)
                                attention_operation_amount += self.MLEN * self.num_head_groups * (head_dim // self.BLEN) * bubble_ratio
                                # Compute O
                                attention_operation_amount += self.MLEN * 5 + 4
            else:
                for b in range(batch_size):
                    for i in range(math.ceil(num_attn_heads)):
                        for j in range(math.ceil(self.input_seq_len / self.MLEN)):
                            for k in range(math.ceil(self.input_seq_len / self.MLEN)):
                                # QKT (MLEN, head_dim) @ (head_dim, MLEN) Full Utilization
                                gemm_operation_amount = (self.MLEN // self.BLEN) * math.ceil(head_dim / self.MLEN) * self.BLEN * (self.MLEN // self.BLEN)
                                attention_operation_amount += gemm_operation_amount * bubble_ratio # Prefetch Data
                                if head_dim > self.MLEN:
                                    attainable_gemm_operation_amount += gemm_operation_amount * 1
                                else:
                                    attainable_gemm_operation_amount += gemm_operation_amount * (head_dim / self.MLEN)
                                # Softmax
                                attention_operation_amount += self.MLEN * 10
                                # PV (MLEN, MLEN) @ (MLEN, head_dim)
                                gemm_operation_amount = (self.MLEN // self.BLEN) * math.ceil(head_dim / self.BLEN) * self.BLEN
                                attention_operation_amount += gemm_operation_amount * bubble_ratio
                                if head_dim > self.BLEN:
                                    attainable_gemm_operation_amount += gemm_operation_amount * 1
                                else:
                                    attainable_gemm_operation_amount += gemm_operation_amount * (head_dim / self.BLEN)
                                # Compute O
                                attention_operation_amount += self.MLEN * 5 + 4
     
            attainable_operation = attainable_gemm_operation_amount
            theoretical_operation = attention_operation_amount

        elif mode == "decode":
            # Projection Q, K, V
            gemm_operation_amount = ((head_dim * num_attn_heads)  // self.BLEN) * ( hidden_size // self.MLEN) * self.BLEN * (math.ceil(self.batch_size / self.BLEN)) + ((head_dim * num_kv_heads) // self.BLEN) * ( hidden_size// self.MLEN) * self.BLEN * (math.ceil(self.batch_size / self.BLEN)) * 2
            attention_operation_amount = gemm_operation_amount * bubble_ratio
            if batch_size > self.BLEN:
                attainable_gemm_operation_amount = gemm_operation_amount * 1
            else:
                attainable_gemm_operation_amount = gemm_operation_amount * (batch_size / self.BLEN)

            # Flash Attention
            if self.partitioned_matrix:
                for b in range(batch_size):
                    for i in range(math.ceil(num_kv_heads / self.max_head_per_mlen)):
                        for j in range(math.ceil(self.kv_size / self.MLEN)):
                            # QKT (self.num_head_groups, 1, Hq) * (Hq, MLEN),  (num_kv_heads / (self.MLEN // self.head_dim)) in parallel
                            gemm_operation_amount = math.ceil(self.num_head_groups / self.BLEN) * self.head_dim * math.ceil(self.MLEN / self.BLEN) 
                            attention_operation_amount += gemm_operation_amount * bubble_ratio
                            if num_kv_heads > (self.MLEN // self.head_dim):
                                if self.num_head_groups > self.BLEN:
                                    attainable_gemm_operation_amount += gemm_operation_amount * 1
                                else:
                                    attainable_gemm_operation_amount += gemm_operation_amount * (self.num_head_groups  / self.BLEN)
                            else:
                                if self.num_head_groups > self.BLEN:
                                    attainable_gemm_operation_amount += gemm_operation_amount * 1 * (num_kv_heads / (self.MLEN // self.head_dim))
                                else:
                                    attainable_gemm_operation_amount += gemm_operation_amount * (self.num_head_groups  / self.BLEN) * (num_kv_heads / (self.MLEN // self.head_dim))
                            # Softmax
                            attention_operation_amount += self.MLEN * 10
                            # PV (self.num_head_groups, 1, MLEN) @ (MLEN, head_dim),  (num_kv_heads / (self.MLEN // self.head_dim)) in parallel
                            gemm_operation_amount = math.ceil(self.num_head_groups / self.BLEN) * math.ceil(head_dim / self.BLEN) * self.num_head_groups
                            attention_operation_amount += gemm_operation_amount * bubble_ratio
                            if self.num_head_groups > self.BLEN:
                                attainable_gemm_operation_amount += gemm_operation_amount * 1
                            else:
                                attainable_gemm_operation_amount += gemm_operation_amount * (self.num_head_groups  / self.BLEN)
                            # Compute O
                            attention_operation_amount += self.MLEN * 5 + 4

            else:
                for b in range(batch_size):
                    for i in range(math.ceil(num_attn_heads)):
                        for j in range(math.ceil(self.kv_size / self.MLEN)):
                            # QKT (1, head_dim) * (head_dim, MLEN),  (num_kv_heads / (self.MLEN // self.head_dim)) 
                            gemm_operation_amount = math.ceil(self.head_dim / self.MLEN) * math.ceil(self.MLEN / self.BLEN) * self.BLEN
                            attention_operation_amount += gemm_operation_amount * bubble_ratio # Prefetch Data
                            attainable_gemm_operation_amount += gemm_operation_amount * 1 * (1 / self.BLEN) * (self.head_dim / self.MLEN)
                            
                            # Softmax
                            attention_operation_amount += self.MLEN * 10
                            
                            # PV (1, MLEN) * (MLEN, head_dim),
                            gemm_operation_amount = (head_dim // self.BLEN) * self.BLEN
                            attention_operation_amount += gemm_operation_amount * bubble_ratio
                            attainable_gemm_operation_amount += gemm_operation_amount * 1 * (1 / self.BLEN)

                            # Compute O
                            attention_operation_amount += self.MLEN * 5 + 4
            attainable_operation = attainable_gemm_operation_amount
            theoretical_operation = attention_operation_amount

        return [attainable_operation, theoretical_operation]

    def _report_self_attention_utilization(self, mode = "prefill") -> None:
        """
        Report the utilization of self attention without using flash attention.
        """
        batch_size = self.batch_size
        hidden_size = self.hidden_size
        num_attn_heads = self.num_attention_heads
        num_kv_heads = self.num_key_value_heads

        head_dim = self.head_dim
        input_token_size = self.input_seq_len
        theoretical_operation = 0
        attainable_operation = 0
        attainable_gemm_operation_amount = 0
        overall_operation_amount = 0
        bubble_ratio = 1.2

        if mode == "prefill":
            # Projection
            gemm_operation_amount = ((head_dim * num_attn_heads)  // self.BLEN) * ( hidden_size // self.MLEN) * self.BLEN * (self.input_seq_len // self.BLEN) + ((head_dim * num_kv_heads) // self.BLEN) * ( hidden_size// self.MLEN) * self.BLEN * (self.input_seq_len // self.BLEN) * 2
            attainable_gemm_operation_amount = gemm_operation_amount * batch_size
            attention_operation_amount = gemm_operation_amount * bubble_ratio * batch_size

            # QKT (batch, input_token_size, num_attn_heads, head_dim) @ (batch, input_token_size, num_attn_heads, head_dim)
            for b in range(batch_size):
                for i in range(num_attn_heads):
                    gemm_operation_amount = (self.input_token_size // self.BLEN) * (head_dim // self.MLEN) * (self.input_token_size // self.BLEN)
                    attention_operation_amount += gemm_operation_amount * bubble_ratio
                    if head_dim > self.MLEN:
                        attainable_gemm_operation_amount += gemm_operation_amount * 1
                    else:
                        attainable_gemm_operation_amount += gemm_operation_amount * (head_dim / self.MLEN)
                    # Storing and fetching QKT (input_token_size, input_token_size)
                    attention_operation_amount += self.input_token_size * 4 * (self.input_token_size / self.VLEN)
                    # Softmax
                    attention_operation_amount += self.input_token_size * (self.input_token_size / self.VLEN) * 30
                    # Storing and fetching O (input_token_size, head_dim)
                    attention_operation_amount += self.input_token_size * (self.input_token_size / self.VLEN) * 5

                    # PV (self.input_token_size, self.input_token_size) @ (self.input_token_size, head_dim)
                    gemm_operation_amount = math.ceil(self.input_token_size / self.BLEN) * math.ceil(head_dim / self.BLEN) * self.num_head_groups
                    attention_operation_amount += gemm_operation_amount * bubble_ratio
                    if head_dim > self.BLEN:
                        attainable_gemm_operation_amount += gemm_operation_amount * 1
                    else:
                        attainable_gemm_operation_amount += gemm_operation_amount * (head_dim / self.BLEN)
                    # Compute O
                    attention_operation_amount += self.input_token_size * (self.input_token_size / self.VLEN) * 5 + 4
            attainable_operation = attainable_gemm_operation_amount
            theoretical_operation = attention_operation_amount

        elif mode == "decode":
            # Projection Q, K, V
            gemm_operation_amount = ((head_dim * num_attn_heads)  // self.BLEN) * ( hidden_size // self.MLEN) * self.BLEN * (math.ceil(self.batch_size / self.BLEN)) + ((head_dim * num_kv_heads) // self.BLEN) * ( hidden_size// self.MLEN) * self.BLEN * (math.ceil(self.batch_size / self.BLEN)) * 2
            attention_operation_amount = gemm_operation_amount * bubble_ratio
            if batch_size > self.BLEN:
                attainable_gemm_operation_amount = gemm_operation_amount * 1
            else:
                attainable_gemm_operation_amount = gemm_operation_amount * (batch_size / self.BLEN)
            # QKT (batch, 1, num_attn_heads, head_dim) @ (batch, kv_size, num_attn_heads, head_dim)
            for b in range(batch_size):
                for i in range(num_attn_heads):
                    gemm_operation_amount = (self.kv_size // self.BLEN) * (head_dim // self.MLEN) * (self.kv_size // self.BLEN)
                    attention_operation_amount += gemm_operation_amount * bubble_ratio
                    if head_dim > self.MLEN:
                        attainable_gemm_operation_amount += gemm_operation_amount * 1
                    else:
                        attainable_gemm_operation_amount += gemm_operation_amount * (head_dim / self.MLEN)
                    # Storing and fetching QKT (1, kv_size)
                    attention_operation_amount +=  4 * (self.kv_size / self.VLEN)
                    # Softmax
                    attention_operation_amount += (self.kv_size / self.VLEN) * 30
                    # Storing and fetching O (input_token_size, head_dim)
                    attention_operation_amount += (self.input_token_size / self.VLEN) * 5

                    # PV (1 self.kv_size) @ (self.kv_size, head_dim)
                    gemm_operation_amount = math.ceil(self.kv_size / self.MLEN) * math.ceil(head_dim / self.BLEN)
                    attention_operation_amount += gemm_operation_amount * bubble_ratio
                    if head_dim > self.BLEN:
                        attainable_gemm_operation_amount += gemm_operation_amount * 1
                    else:
                        attainable_gemm_operation_amount += gemm_operation_amount * (head_dim / self.BLEN)
                    # Compute O
                    attention_operation_amount += (self.kv_size / self.VLEN) * 15 + 4

            attainable_operation = attainable_gemm_operation_amount
            theoretical_operation = attention_operation_amount
        return [attainable_operation, theoretical_operation]

    def _report_embedding_utilization(self, mode = "prefill") -> None:
        """
        Report the utilization of flash attention for a given node.
        """

        batch_size = self.batch_size
        hidden_size = self.hidden_size

        theoretical_operation = 0
        attainable_operation = 0

        if mode == "prefill":
            # Assuming Decoding only
            operation_amount = (hidden_size // self.BLEN) * (hidden_size // self.MLEN) * (self.input_seq_len // self.BLEN)
            attainable_operation += operation_amount * (self.BLEN * self.MLEN * self.BLEN)
            theoretical_operation += operation_amount * (self.BLEN * self.MLEN * self.BLEN)
        elif mode == "decode":
            operation_amount = (hidden_size // self.BLEN) * (hidden_size // self.MLEN)
            attainable_operation += operation_amount * (self.BLEN * self.MLEN * min(self.batch_size, self.BLEN))
            theoretical_operation += operation_amount * (self.BLEN * self.MLEN * self.BLEN)

        return [operation_amount, attainable_operation, theoretical_operation]

    def _report_ffn_utilization(self, mode = "prefill") -> None:
        """
        Report the utilization of flash attention for a given node.
        """
        batch_size = self.batch_size
        hidden_size = self.hidden_size
        intermediate_size = self.intermediate_size
        theoretical_operation = 0
        attainable_operation = 0
        bubble_ratio = 1.2

        if mode == "prefill":
            # Up Projection
            operation_amount = (intermediate_size // self.BLEN) * (hidden_size // self.MLEN) * (self.input_seq_len // self.BLEN)
            attainable_operation += operation_amount
            theoretical_operation += operation_amount * bubble_ratio

            # Gate Projection
            operation_amount = (intermediate_size // self.BLEN) * (hidden_size // self.MLEN) * (self.input_seq_len // self.BLEN)
            attainable_operation += operation_amount 
            theoretical_operation += operation_amount * bubble_ratio

            # SiLU
            theoretical_operation += (intermediate_size // self.VLEN) * 5

            # Down Projection
            operation_amount = (hidden_size // self.BLEN) * (intermediate_size // self.MLEN) * (self.input_seq_len // self.BLEN)
            attainable_operation += operation_amount
            theoretical_operation += operation_amount * bubble_ratio

        elif mode == "decode":
            
            # Up projection
            fill_bubble_time = self.BLEN * (intermediate_size // self.BLEN) * math.ceil(self.batch_size / self.BLEN)
            operation_amount = (intermediate_size // self.BLEN) * self.BLEN * (hidden_size // self.MLEN) * math.ceil(self.batch_size / self.BLEN)
            attainable_operation += operation_amount * (min(self.batch_size, self.BLEN) / self.BLEN)
            theoretical_operation += operation_amount + fill_bubble_time

            # Gate Projection
            fill_bubble_time = self.BLEN * (intermediate_size // self.BLEN) * math.ceil(self.batch_size / self.BLEN)
            operation_amount = (intermediate_size // self.BLEN) * self.BLEN * (hidden_size // self.MLEN) * math.ceil(self.batch_size / self.BLEN)
            attainable_operation += operation_amount * (min(self.batch_size, self.BLEN) / self.BLEN)
            theoretical_operation += operation_amount + fill_bubble_time

            # SiLU
            theoretical_operation += (intermediate_size // self.VLEN) * 5

            # Down Projection
            fill_bubble_time = self.BLEN * (hidden_size // self.BLEN) * math.ceil(self.batch_size / self.BLEN)
            operation_amount = (hidden_size // self.BLEN) * self.BLEN * (intermediate_size // self.MLEN) * math.ceil(self.batch_size / self.BLEN)
            attainable_operation += operation_amount * (min(self.batch_size, self.BLEN) / self.BLEN)
            theoretical_operation += operation_amount+ fill_bubble_time

        return [attainable_operation, theoretical_operation]



    def _report_prefill_utilization(self):
        
        overall_attainable_GEMM = {"attention": 0, "ffn": 0}
        overall_theoretical_GEMM = {"attention": 0, "ffn": 0}
        overall_attainable_GEMM["attention"] = self._report_flash_attn_utilization("prefill")[0]
        overall_theoretical_GEMM["attention"] = self._report_flash_attn_utilization("prefill")[1]
        overall_attainable_GEMM["ffn"] = self._report_ffn_utilization("prefill")[0]
        overall_theoretical_GEMM["ffn"] = self._report_ffn_utilization("prefill")[1]

        return {
            "attainable_FLOPS": overall_attainable_GEMM,
            "theoretical_FLOPS": overall_theoretical_GEMM
        }
    
    def _report_decode_utilization(self):
        overall_attainable_GEMM = {"attention": 0, "ffn": 0}
        overall_theoretical_GEMM = {"attention": 0, "ffn": 0}
        overall_attainable_GEMM["attention"] = self._report_flash_attn_utilization("decode")[0]
        overall_theoretical_GEMM["attention"] = self._report_flash_attn_utilization("decode")[1]
        overall_attainable_GEMM["ffn"] = self._report_ffn_utilization("decode")[0]
        overall_theoretical_GEMM["ffn"] = self._report_ffn_utilization("decode")[1]

        return {
            "attainable_FLOPS": overall_attainable_GEMM,
            "theoretical_FLOPS": overall_theoretical_GEMM
        }
    
    def compute_overall_perf(self):
        prefill_perf = self._report_prefill_utilization()
        print(f"Prefill Performance: {prefill_perf}")
        decode_perf = self._report_decode_utilization()
        print(f"Decode Performance: {decode_perf}")
        utilization = (prefill_perf["attainable_FLOPS"]["ffn"] + prefill_perf["attainable_FLOPS"]["attention"] + decode_perf["attainable_FLOPS"]["ffn"] + decode_perf["attainable_FLOPS"]["attention"]) / (prefill_perf["theoretical_FLOPS"]["ffn"] + prefill_perf["theoretical_FLOPS"]["attention"] + decode_perf["theoretical_FLOPS"]["ffn"] + decode_perf["theoretical_FLOPS"]["attention"])
        return utilization


if __name__ == "__main__":
    import os
    from pathlib import Path
    # Get the absolute path of the current file's directory
    current_dir = Path(__file__).resolve().parents[3]
    model_param_path = os.path.join(current_dir, "doc/Model_Lib/llama-3.1-8b.json")
    model = attn_model_config(MLEN=1024, BLEN=4, VLEN=1024, partitioned_matrix=False, model_param_path=model_param_path, batch_size=1, seq_len=5600, output_token=1000, device_num=1)
    model.kv_size = 80000
    actual, theoretical = model._report_flash_attn_utilization("decode")
    print(f"Actual: {actual}, Theoretical: {theoretical}")
    print(f"Utilization: {actual / theoretical}")