import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from utils import load_json
from pathlib import Path
import os


colors = {
    "turq": tuple(i / 255 for i in (40, 161, 151)),
    "darkblue": tuple(i / 255 for i in (18, 67, 109)),
    "dark_pink": tuple(i / 255 for i in (128, 22, 80)),
    "orange": tuple(i / 255 for i in (244, 106, 37)),
    "dark_green": tuple(i / 255 for i in (61, 159, 60)),
    "dark_blue": tuple(i / 255 for i in (54, 125, 176)),
}

# HBM Settings
Operate_Freq = 1e9      # 1 GHz
DataWidth = 2           # 1 byte per element
HBM_Bandwidth = 800e9   # 800 GB/s
HBM_Capacity = 128      # 128 GB
SEQ_LENGTH_NORM =  512
SEQ_LENGTH_REASONING = 2048

B200_Params = {
    "HBM_Capacity": HBM_Capacity,  # HBM 3e
    "HBM_Bandwidth": HBM_Bandwidth,  # 512 GB/s
    "Operate_Freq": 1e9,  # 1 GHz
    "M" : 8,
    "K" : 64,
    "N" : 16,
    "DataWidth": 2  # 1 byte per element
}

TPU_Params = {
    "HBM_Capacity": HBM_Capacity,  # HBM 3e
    "HBM_Bandwidth": HBM_Bandwidth,  # 512 GB/s
    "Operate_Freq": 1e9,  # 1 GHz
    "M" : 64,
    "K" : 64,
    "N" : 64,
    "DataWidth": 2  # 1 byte per element
}

PLENA = {
    "HBM_Capacity": HBM_Capacity,  # HBM 3e
    "HBM_Bandwidth": HBM_Bandwidth,  # 512 GB/s
    "Operate_Freq": 1e9,  # 1 GHz
    "M" : 8,
    "K" : 512,
    "N" : 8,
    "DataWidth": 2  # 1 byte per element
}


def select_powers_of_two_with_last(max_batch):
    powers = [2**i for i in range(max_batch.bit_length()) if 2**i < max_batch]
    if max_batch not in powers:
        powers.append(max_batch)
    return powers


class DeviceModel:
    def __init__(self, operate_freq, M, K, N, data_width, hbm_bandwidth, hbm_capacity):
        self.operate_freq = operate_freq
        self.M = M
        self.K = K
        self.N = N
        self.data_width = data_width
        self.hbm_bandwidth = hbm_bandwidth
        self.hbm_capacity = hbm_capacity

    def get_peak_performance(self):
        return self.operate_freq * self.M * self.K * 2
    
    def get_attainable_performance(self, operation_intensity):
        return np.minimum(
            self.get_peak_performance(),
            operation_intensity * self.hbm_bandwidth
        ) / 1e9  # Convert to GFLOPs/s


def hbm_capacity_requirement(
    roofline_model,
    model_config,
    seq_context_length,
    batch_size: int
):
    hbm_storage_per_layer = 0
    # QKV Projection Weights & Biases
    hbm_storage_per_layer += model_config.get("hidden_size", 0) * model_config.get("hidden_size", 0) * 3 * roofline_model.data_width
    hbm_storage_per_layer += model_config.get("hidden_size", 0) * 3 * roofline_model.data_width
    # KV Cache
    hbm_storage_per_layer += seq_context_length * model_config.get("hidden_size", 0) * 2 * batch_size * roofline_model.data_width
    # Attention Output
    hbm_storage_per_layer += model_config.get("hidden_size", 0) * seq_context_length * batch_size * roofline_model.data_width
    # MLP
    hbm_storage_per_layer += model_config.get("hidden_size", 0) * model_config.get("intermediate_size", 0) * 2 * roofline_model.data_width
    hbm_storage_per_layer += model_config.get("intermediate_size", 0) * batch_size * roofline_model.data_width
    hbm_storage_per_layer += model_config.get("intermediate_size", 0) * 2 * roofline_model.data_width
    hbm_storage_per_layer += model_config.get("hidden_size", 0) * batch_size * roofline_model.data_width

    return hbm_storage_per_layer * model_config.get("num_hidden_layers", 0) / 1e9  # Convert to GB


def device_performance(device_model, seq_context_length, max_batch, model_config):
    batch_bound = 0
    for batch_size in range(1, max_batch):
        hbm_capacity = hbm_capacity_requirement(device_model, model_config, seq_context_length, batch_size)
        if hbm_capacity > device_model.hbm_capacity:
            break
        batch_bound = batch_size 

    roofline_performance = {}
    sampled_batch = select_powers_of_two_with_last(max_batch)
    max_tflops = device_model.get_peak_performance() / 1e9
    for batch in sampled_batch:
        peak_tflops = 2 * batch * device_model.K * device_model.operate_freq / 1e9
        roofline_performance[batch] = min(peak_tflops, max_tflops)
    
    actual_performance = {}
    print("device width:", device_model.data_width)
    max_tilesize = device_model.hbm_bandwidth / (2 * device_model.operate_freq * device_model.data_width)
    print("batch_bound:", batch_bound, "max_tilesize:", max_tilesize)
    sampled_batch = select_powers_of_two_with_last(batch_bound)
    for batch in sampled_batch:
        if batch > device_model.M:
            break
        compute_intensity = 2 * min(device_model.K, max_tilesize) * batch * device_model.operate_freq / 1e9
        actual_performance[batch] = min(compute_intensity, max_tflops)


    return roofline_performance, actual_performance, batch_bound


if __name__ == "__main__":
    matplotlib.rcParams['font.size'] = 6
    config_parent_path  = Path(__file__).resolve().parents[2]
    print(f"Config parent path: {config_parent_path}")
    model_config_path   = os.path.join(config_parent_path, "doc/Model_Lib/llama-3.1-70b.json")
    model_config        = load_json(model_config_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)


    # Plot TPU Performance
    tpu_model   = DeviceModel(operate_freq=TPU_Params["Operate_Freq"], M=TPU_Params["M"], K=TPU_Params["K"], N=TPU_Params["N"], data_width=TPU_Params["DataWidth"], hbm_bandwidth=TPU_Params["HBM_Bandwidth"], hbm_capacity=TPU_Params["HBM_Capacity"])
    plena_model = DeviceModel(operate_freq=PLENA["Operate_Freq"], M=PLENA["M"], K=PLENA["K"], N=TPU_Params["N"], data_width= PLENA["DataWidth"], hbm_bandwidth=PLENA["HBM_Bandwidth"], hbm_capacity=PLENA["HBM_Capacity"])
    soft_optimised_plena_model = DeviceModel(operate_freq=PLENA["Operate_Freq"], M=PLENA["M"], K=PLENA["K"], N=TPU_Params["N"], data_width= PLENA["DataWidth"] / 3, hbm_bandwidth=PLENA["HBM_Bandwidth"], hbm_capacity=PLENA["HBM_Capacity"])

    tpu_roofline_performance, tpu_actual_performance_normal, tpu_normal_batch_bound = device_performance(tpu_model, SEQ_LENGTH_NORM, 256, model_config)
    _, tpu_actual_performance_reasoning, tpu_actual_reasoning_batch_bound = device_performance(tpu_model, SEQ_LENGTH_REASONING, 256, model_config)

    plena_roofline_performance, plena_actual_performance_normal, plena_normal_batch_bound = device_performance(plena_model, SEQ_LENGTH_NORM, 256, model_config)
    _, plena_actual_performance_reasoning, plena_reasoning_batch_bound = device_performance(plena_model, SEQ_LENGTH_REASONING, 256, model_config)

    _, soft_optimised_actual_performance_normal, soft_optimised_normal_batch_bound = device_performance(soft_optimised_plena_model, SEQ_LENGTH_NORM, 256, model_config)
    _, soft_optimised_actual_performance_reasoning, soft_optimised_reasoning_batch_bound = device_performance(soft_optimised_plena_model, SEQ_LENGTH_REASONING, 256, model_config)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Attainable GFLOPs/s')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylim(1e2, 1e5)
    ax1.set_xlim(1, 256)
    ax1.set_title('Normal Inference Performance')
    
    ax1.plot(list(plena_roofline_performance.keys()), list(plena_roofline_performance.values()), label='8 * 512 W/O Memory Wall', color="grey", linewidth=1, linestyle='--')
    ax1.vlines(tpu_normal_batch_bound, 1e2, 1e5, color='grey', linestyle='--', linewidth=0.5)
    ax1.plot(
        list(tpu_roofline_performance.keys()),
        [v for v in tpu_roofline_performance.values()],
        label='128 * 128 W/O Memory Wall',
        linewidth=0.8, linestyle='--',
        color='grey'
    )
    
    ax1.plot(
        list(tpu_actual_performance_normal.keys()),
        [v for v in tpu_actual_performance_normal.values()],
        label='TPU',
        color='grey',
        linewidth=2
    )

    ax1.vlines(soft_optimised_normal_batch_bound, 1e2, 1e5, color = 'grey', linestyle='--', linewidth=0.5)

    ax1.plot(
        list(plena_actual_performance_normal.keys()),
        [v for v in plena_actual_performance_normal.values()],
        label='PLENA W/O Quantisation',
        color=colors["dark_green"],
        linewidth=2
    )

    ax1.hlines(max(plena_actual_performance_normal.values()), max(plena_actual_performance_normal.keys()), plena_normal_batch_bound, color=colors["dark_green"], linewidth=2)

    ax1.plot(
        list(soft_optimised_actual_performance_normal.keys()),
        [v for v in soft_optimised_actual_performance_normal.values()],
        label='PLENA W Quantisation',
        color= colors["dark_blue"],
        linewidth=2
    )

    ax1.hlines(max(soft_optimised_actual_performance_normal.values()), max(soft_optimised_actual_performance_normal.keys()), soft_optimised_normal_batch_bound, color=colors["dark_blue"], linewidth=2)


    # Plot Reasoninng
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylabel('Attainable GFLOPs/s')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylim(1e2, 1e5)
    ax2.set_xlim(1, 256)
    ax2.set_title('Reasoning Inference Performance')
    ax2.vlines(plena_reasoning_batch_bound, 1e2, 1e5, color='grey', linestyle='--', linewidth=0.5)
    ax2.vlines(soft_optimised_reasoning_batch_bound, 1e2, 1e5, color='grey', linestyle='--', linewidth=0.5)

    ax2.plot(list(plena_roofline_performance.keys()), list(plena_roofline_performance.values()), color="grey", linewidth=1, linestyle='--')
    
    ax2.plot(
        list(tpu_roofline_performance.keys()),
        [v for v in tpu_roofline_performance.values()],
        linewidth=0.8, linestyle='--',
        color='grey'
    )
    
    ax2.plot(
        list(tpu_actual_performance_reasoning.keys()),
        [v for v in tpu_actual_performance_reasoning.values()],
        color='grey',
        linewidth=2
    )
    
    ax2.plot(
        list(plena_actual_performance_reasoning.keys()),
        [v for v in plena_actual_performance_reasoning.values()],
        color=colors["dark_green"],
        linewidth=2
    )

    ax2.hlines(max(plena_actual_performance_reasoning.values()), max(plena_actual_performance_reasoning.keys()), plena_reasoning_batch_bound, color=colors["dark_green"], linewidth=2)

    ax2.plot(
        list(soft_optimised_actual_performance_reasoning.keys()),
        [ v for v in soft_optimised_actual_performance_reasoning.values()],
        color=colors["dark_blue"],
        linewidth=2
    )

    ax2.hlines(max(soft_optimised_actual_performance_reasoning.values()), max(soft_optimised_actual_performance_reasoning.keys()), soft_optimised_reasoning_batch_bound, color=colors["dark_blue"], linewidth=2)

    from collections import OrderedDict

    # Get handles and labels from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine all handles and labels
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    # Remove duplicates while preserving order
    legend_dict = OrderedDict()
    for h, l in zip(all_handles, all_labels):
        if l not in legend_dict:
            legend_dict[l] = h


    fig.legend(legend_dict.values(), legend_dict.keys(),
            loc='center left',
            bbox_to_anchor=(1.02, 0.8),  # Push to the right of both axes
            fontsize=5, frameon=False)

    # Adjust space so plots don't overlap with the legend
    fig.subplots_adjust(right=0.75)  # Leave space for legend
    plt.tight_layout()
    plt.savefig('systolic_array_comparison.png', bbox_inches='tight', dpi=300)