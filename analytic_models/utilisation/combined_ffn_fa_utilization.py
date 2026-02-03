from attainable import attn_model_config

import os
from pathlib import Path
# Project root is 2 levels up from analytic_models/utilisation/
current_dir = Path(__file__).resolve().parents[2]
model_param_path = os.path.join(current_dir, "doc/Model_Lib/llama-3.1-8b.json")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Define MLEN and BLEN configurations and their labels
configs = [
    {"MLEN": 64, "BLEN": 64, "label": "MLEN=64, BLEN=64"},
    {"MLEN": 128, "BLEN": 32, "label": "MLEN=128, BLEN=32"},
    {"MLEN": 256, "BLEN": 16, "label": "MLEN=256, BLEN=16"},
    {"MLEN": 512, "BLEN": 8, "label": "MLEN=512, BLEN=8"},
    {"MLEN": 1024, "BLEN": 4, "label": "MLEN=1024, BLEN=4"}
]

batch_sizes = [1, 2, 4, 8, 16]
sequence_lengths = [128, 1024, 5120, 20480, 81920]  # 128, 1k, 5k, 20k, 80k
seq_labels = ["128", "1k", "5k", "20k", "80k"]

bar_width = 0.06  # Bar width
bar_gap = 0.02  # Gap between bars in same group
group_gap = 0  # Gap between groups of bars

# Create x positions with spacing between groups
x_ffn = np.arange(len(batch_sizes)) * (0.5 + group_gap)
x_fa = np.arange(len(sequence_lengths)) * (0.5 + group_gap)

# Create 1x4 grid: FFN util, FFN time, FA util, FA time
fig, (ax1_ffn_util, ax2_ffn_time, ax1_fa_util, ax2_fa_time) = plt.subplots(1, 4, figsize=(20, 4))

colors = [
    "#762a83",  # deep purple
    "#af8dc3",  # light violet
    "#e7d4e8",  # very light purple-pink
    "#d9f0d3",  # pale green
    "#7fbf7b",  # medium green
    # Optional sixth if needed: "#1b7837"  # dark green
]

# Process FFN data
for idx, cfg in enumerate(configs):
    # FFN data
    ffn_utilizations = []
    ffn_execution_times = []  # Execution time in microseconds (cycles / 1GHz)
    
    for bs in batch_sizes:
        # reload model with required settings
        model = attn_model_config(
            MLEN=cfg["MLEN"], BLEN=cfg["BLEN"], VLEN=1024, partitioned_matrix=True,
            model_param_path=model_param_path, batch_size=bs, seq_len=8024, output_token=100, device_num=1
        )
        
        # FFN stage
        ffn_attainable, ffn_theoretical = model._report_ffn_utilization("decode")[0:2]
        ffn_utilization = (ffn_attainable / ffn_theoretical if ffn_theoretical != 0 else 0) * 100  # Convert to percentage
        ffn_utilizations.append(ffn_utilization)
        
        # Execution time: theoretical cycles / 1GHz = time in seconds, convert to milliseconds
        execution_time_ms = (ffn_theoretical / 1e9) * 1e3  # Convert to milliseconds
        ffn_execution_times.append(execution_time_ms)

    # Bar positions for configs - with spacing between groups
    pos_ffn = x_ffn + (idx - len(configs)/2) * (bar_width + bar_gap) + bar_width/2
    
    # Plot FFN utilization
    rects_ffn_util = ax1_ffn_util.bar(pos_ffn, ffn_utilizations, width=bar_width, label=cfg["label"], color=colors[idx], alpha=0.7)
    
    # Plot FFN execution time
    rects_ffn_time = ax2_ffn_time.bar(pos_ffn, ffn_execution_times, width=bar_width, label=cfg["label"], color=colors[idx], alpha=0.7)

# Process Flash Attention data
for idx, cfg in enumerate(configs):
    # Flash Attention data
    fa_utilizations = []
    fa_execution_times = []  # Execution time in microseconds (cycles / 1GHz)
    
    for seq_len in sequence_lengths:
        # reload model with required settings
        model = attn_model_config(
            MLEN=cfg["MLEN"], BLEN=cfg["BLEN"], VLEN=1024, partitioned_matrix=True,
            model_param_path=model_param_path, batch_size=1, seq_len=seq_len, output_token=100, device_num=1
        )
        
        # Flash Attention stage
        fa_attainable, fa_theoretical = model._report_flash_attn_utilization("decode")[0:2]
        fa_utilization = (fa_attainable / fa_theoretical if fa_theoretical != 0 else 0) * 100  # Convert to percentage
        fa_utilizations.append(fa_utilization)
        
        # Execution time: theoretical cycles / 1GHz = time in seconds, convert to milliseconds
        execution_time_ms = (fa_theoretical / 1e9) * 1e3  # Convert to milliseconds
        fa_execution_times.append(execution_time_ms)

    # Bar positions for configs - with spacing between groups
    pos_fa = x_fa + (idx - len(configs)/2) * (bar_width + bar_gap) + bar_width/2
    
    # Plot Flash Attention utilization
    rects_fa_util = ax1_fa_util.bar(pos_fa, fa_utilizations, width=bar_width, label=cfg["label"], color=colors[idx], alpha=0.7)
    
    # Plot Flash Attention execution time
    rects_fa_time = ax2_fa_time.bar(pos_fa, fa_execution_times, width=bar_width, label=cfg["label"], color=colors[idx], alpha=0.7)

# Configure FFN utilization plot
ax1_ffn_util.set_xlabel("Batch Size", fontsize=18)
ax1_ffn_util.set_ylabel("Utilization (%)", fontsize=18)
ax1_ffn_util.tick_params(axis='both', which='major', labelsize=14)
ax1_ffn_util.set_title("FFN Decode SA Utilization", fontsize=20)
ax1_ffn_util.set_xticks(x_ffn)
ax1_ffn_util.set_xticklabels([str(bs) for bs in batch_sizes])
ax1_ffn_util.set_ylim(0, 100)
# Format y-axis to show only integer ticks
ax1_ffn_util.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax1_ffn_util.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}' if x == int(x) else ''))

# Set x-axis limits to show all bars with spacing
total_config_width_ffn = len(configs) * (bar_width + bar_gap) - bar_gap
ax1_ffn_util.set_xlim(x_ffn[0] - total_config_width_ffn/2 - group_gap/2, x_ffn[-1] + total_config_width_ffn/2 + group_gap/2)

# Configure FFN execution time plot
ax2_ffn_time.set_xlabel("Batch Size", fontsize=18)
ax2_ffn_time.set_ylabel("Exec Time (ms)", fontsize=18)
ax2_ffn_time.tick_params(axis='both', which='major', labelsize=14)
ax2_ffn_time.set_title("FFN Decode Exec Time", fontsize=20)
ax2_ffn_time.set_xticks(x_ffn)
ax2_ffn_time.set_xticklabels([str(bs) for bs in batch_sizes])
# Format y-axis to show only integer ticks
ax2_ffn_time.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax2_ffn_time.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}' if x == int(x) else ''))

# Set x-axis limits to show all bars with spacing (same as FFN util)
ax2_ffn_time.set_xlim(x_ffn[0] - total_config_width_ffn/2 - group_gap/2, x_ffn[-1] + total_config_width_ffn/2 + group_gap/2)

# Configure Flash Attention utilization plot
ax1_fa_util.set_xlabel("Number of Tokens", fontsize=18)
ax1_fa_util.set_ylabel("Utilization (%)", fontsize=18)
ax1_fa_util.tick_params(axis='both', which='major', labelsize=14)
ax1_fa_util.set_title("FA Decode SA Utilization", fontsize=20)
ax1_fa_util.set_xticks(x_fa)
ax1_fa_util.set_xticklabels(seq_labels)
ax1_fa_util.set_ylim(0, 100)
# Format y-axis to show only integer ticks
ax1_fa_util.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax1_fa_util.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}' if x == int(x) else ''))

# Set x-axis limits to show all bars with spacing
total_config_width_fa = len(configs) * (bar_width + bar_gap) - bar_gap
ax1_fa_util.set_xlim(x_fa[0] - total_config_width_fa/2 - group_gap/2, x_fa[-1] + total_config_width_fa/2 + group_gap/2)

# Configure Flash Attention execution time plot
ax2_fa_time.set_xlabel("Number of Tokens", fontsize=18)
ax2_fa_time.set_ylabel("Exec Time (ms)", fontsize=18)
ax2_fa_time.tick_params(axis='both', which='major', labelsize=14)
ax2_fa_time.set_title("FA Decode Exec Time", fontsize=20)
ax2_fa_time.set_xticks(x_fa)
ax2_fa_time.set_xticklabels(seq_labels)
# Format y-axis to show only integer ticks
ax2_fa_time.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax2_fa_time.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}' if x == int(x) else ''))

# Set x-axis limits to show all bars with spacing (same as FA util)
ax2_fa_time.set_xlim(x_fa[0] - total_config_width_fa/2 - group_gap/2, x_fa[-1] + total_config_width_fa/2 + group_gap/2)

# Create shared legend at the bottom in two rows (placed lower)
handles1, labels1 = ax1_ffn_util.get_legend_handles_labels()
# Create legend with all config labels - transparent with no frame, arranged in 2 rows
# With 5 configs, use ncol=3 to get 2 rows (3 in first row, 2 in second row)
# Push legend lower using bbox_to_anchor
fig.legend(handles1, labels1, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=5, fontsize=15, frameon=False)

plt.tight_layout(rect=[0, 0.1, 1, 0.98])
plt.savefig("combined_FFN_FA_Utilization.png", dpi=300, bbox_inches='tight')
print("Plot saved as 'combined_FFN_FA_Utilization.png'")

