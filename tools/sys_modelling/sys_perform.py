# System Performance Evaluation
# Assumptions:




import numpy as np
from tools.cost_model.latency import instr_latency_model





class PerformanceEvaluator:
    """
    Evaluates transformer inference (prefill stage) performance by analyzing compute, memory bandwidth,
    and HBM capacity constraints under varying input sequence lengths. User provides model config and
    hardware settings.
    """
    def __init__(
        self,
        model_config: dict,
        batch_size: int,
        operate_freq: float,
        M: int,
        K: int,
        N: int,
        Vect_Dim: int,
        data_width: int,  # bytes per element (e.g. 2 for fp16)
        hbm_bandwidth: float,  # in GB/s
        hbm_capacity: float,   # in GB
        sram_bandwidth: float = None,  # in GB/s, optional
        sram_capacity: float = None,   # in GB, optional
        area: float = None,            # in mm^2, optional
        power_budget: float = None     # in W, optional
    ):
        self.model_config = model_config
        self.batch_size = batch_size
        self.operate_freq = operate_freq
        self.M = M
        self.K = K
        self.N = N
        self.data_width         = data_width
        self.hbm_bandwidth      = hbm_bandwidth
        self.hbm_capacity       = hbm_capacity
        self.sram_bandwidth     = sram_bandwidth
        self.sram_capacity      = sram_capacity
        self.area               = area
        self.power_budget       = power_budget

    def get_peak_flops(self):
        # FMA: 2 * M * K * freq
        return self.operate_freq * self.M * self.K * 2

    def hbm_capacity_requirement(self, seq_context_length, batch_size):
        """
        Estimates HBM usage for the full attention prefill of a transformer layer stack.
        """
        m = self.model_config
        hidden_size = m.get("hidden_size", 0)
        intermediate_size = m.get("intermediate_size", 0)
        num_layers = m.get("num_hidden_layers", 0)
        DataWidth = self.data_width

        hbm_storage_per_layer = 0
        # QKV Projection Weights & Biases
        hbm_storage_per_layer += hidden_size * hidden_size * 3 * DataWidth
        hbm_storage_per_layer += hidden_size * 3 * DataWidth
        # KV Cache for all tokens
        hbm_storage_per_layer += seq_context_length * hidden_size * 2 * batch_size * DataWidth
        # Attention Output
        hbm_storage_per_layer += hidden_size * seq_context_length * batch_size * DataWidth
        # MLP Weights
        hbm_storage_per_layer += hidden_size * intermediate_size * 2 * DataWidth
        # MLP activations/biases
        hbm_storage_per_layer += intermediate_size * batch_size * DataWidth
        hbm_storage_per_layer += intermediate_size * 2 * DataWidth
        hbm_storage_per_layer += hidden_size * batch_size * DataWidth

        total = hbm_storage_per_layer * num_layers / 1e9  # bytes to GB
        return total

    def prefill_compute_requirement(self, seq_context_length, batch_size):
        """
        Estimates total FLOPs required for prefill stage for the given input.
        """
        m = self.model_config
        hidden_size = m.get("hidden_size", 0)
        intermediate_size = m.get("intermediate_size", 0)
        num_layers = m.get("num_hidden_layers", 0)

        # Basic transformer full pass (estimate):
        # per token per layer: (Self-attention & MLP)
        # Attention: QK^T + softmax + Attention*V ~ 2 * hidden_size * seq_context_length (for each token), per batch
        # MLP: 2 matmuls, hidden->intermediate->hidden

        # Self-Attention FLOPs: (QK^T, softmax is minor, AV), for all tokens and batches
        attn_flops = 2 * (hidden_size * hidden_size * seq_context_length) * batch_size # each token attends to seq and does h_size^2 per head

        # MLP FLOPs: (forward)
        mlp_flops = 2 * hidden_size * intermediate_size * seq_context_length * batch_size
        # Project back
        mlp_flops += 2 * intermediate_size * hidden_size * seq_context_length * batch_size

        flops_total = num_layers * (attn_flops + mlp_flops)
        return flops_total

    def evaluate(self, seq_lengths, batch_sizes):
        """
        For a sweep of input sequence lengths and batch sizes,
        determine per combination:
         - Is compute-bound, memory-bound, or capacity-bound
         - Maximum achievable throughput (FLOPs/s)
         - What bound ("compute", "memory", "capacity") is active
         - Supporting metrics for further analysis
        Returns: results as list of dicts (or can be numpy arrays)
        """
        peak_flops = self.get_peak_flops()
        results = []

        for seq_len in seq_lengths:
            for batch in batch_sizes:
                hbm_usage = self.hbm_capacity_requirement(seq_len, batch)
                flops = self.prefill_compute_requirement(seq_len, batch)
                # Prefill duration if compute-bound (assuming all streaming, ideal):
                compute_time = flops / peak_flops if peak_flops > 0 else float('inf')
                # Prefill duration if memory-bound:
                # Estimate bytes to read: assume all weights once per layer per batch/seq pass
                # For upper bound: read all weight data + all activation (KV cache notably large)
                m = self.model_config
                hidden_size = m.get("hidden_size", 0)
                num_layers = m.get("num_hidden_layers", 0)
                bytes_read_weights = num_layers * (
                    # QKV weights
                    hidden_size * hidden_size * 3 +
                    hidden_size * 3 +
                    hidden_size * m.get("intermediate_size", 0) * 2 +
                    m.get("intermediate_size", 0) * 2 +
                    m.get("intermediate_size", 0) * batch +
                    hidden_size * batch
                )
                bytes_read_activations = num_layers * (
                    seq_len * hidden_size * 2 * batch +     # KV Cache
                    hidden_size * seq_len * batch           # Attn output
                )
                total_bytes = (bytes_read_weights + bytes_read_activations) * self.data_width
                mem_time = total_bytes / (self.hbm_bandwidth * 1e9) if self.hbm_bandwidth > 0 else float('inf')

                # Theoretical throughput (FLOPs/s)
                achievable_flops_mem = flops / mem_time if mem_time > 0 else 0
                achievable_flops_compute = flops / compute_time if compute_time > 0 else 0

                # Determine bound
                if hbm_usage > self.hbm_capacity:
                    bound = "capacity"
                    achieved_flops = 0
                elif mem_time > compute_time:
                    bound = "memory"
                    achieved_flops = achievable_flops_mem
                else:
                    bound = "compute"
                    achieved_flops = peak_flops

                result = {
                    "seq_len": seq_len,
                    "batch": batch,
                    "hbm_usage_GB": hbm_usage,
                    "FLOPs_total": flops,
                    "peak_flops": peak_flops,
                    "compute_time_s": compute_time,
                    "mem_time_s": mem_time,
                    "active_bound": bound,
                    "achieved_flops": achieved_flops
                }
                results.append(result)
        return results

    def print_report(self, results, limit=10):
        """
        Print basic table of results, up to 'limit' rows.
        """
        print(f"{'SeqLen':>8} {'Batch':>6} {'HBM_Usage(GB)':>14} {'FLOPs':>12} {'Bound':>10} {'Achvd_FLOPs':>14}")
        for r in results[:limit]:
            print(f"{r['seq_len']:8d} {r['batch']:6d} {r['hbm_usage_GB']:14.2f} {r['FLOPs_total']/1e12:12.3f}T {r['active_bound']:>10} {r['achieved_flops']/1e12:14.2f}T")

# Example usage
if __name__ == "__main__":
    # Assume loaded model config (example fields filled)
    model_config = {
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_hidden_layers": 80
    }
    evaluator = PerformanceEvaluator(
        model_config=model_config,
        operate_freq=1e9,
        M=8,
        K=512,
        N=8,
        data_width=2,           # fp16
        hbm_bandwidth=512,      # GB/s
        hbm_capacity=144,       # GB
        sram_bandwidth=32768,   # GB/s (optional)
        sram_capacity=4         # GB (optional)
    )
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
    batch_sizes = [1, 2, 4, 8, 16, 32]
    results = evaluator.evaluate(seq_lengths, batch_sizes)
    evaluator.print_report(results, limit=20)

