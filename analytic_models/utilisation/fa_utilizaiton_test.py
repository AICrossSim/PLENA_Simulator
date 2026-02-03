from attainable import attn_model_config

import os
from pathlib import Path
# Project root is 2 levels up from analytic_models/utilisation/
current_dir = Path(__file__).resolve().parents[2]
model_param_path = os.path.join(current_dir, "doc/Model_Lib/llama-3.1-8b.json")

import matplotlib.pyplot as plt
import numpy as np

# Define MLEN and BLEN configurations and their labels
configs = [
    {"MLEN": 64, "BLEN": 64, "label": "MLEN=64, BLEN=64"},
    {"MLEN": 128, "BLEN": 32, "label": "MLEN=128, BLEN=32"},
    {"MLEN": 256, "BLEN": 16, "label": "MLEN=256, BLEN=16"},
    {"MLEN": 512, "BLEN": 8, "label": "MLEN=512, BLEN=8"},
    {"MLEN": 1024, "BLEN": 4, "label": "MLEN=1024, BLEN=4"}
]

sequence_lengths = [128, 1024, 5120, 20480, 81920]  # 128, 1k, 5k, 20k, 80k
seq_labels = ["128", "1k", "5k", "20k", "80k"]

bar_width = 0.1  # Thinner bars
bar_gap = 0.02  # Gap between bars in same group
group_gap = 0  # Gap between groups of bars

# Create x positions with spacing between groups
x = np.arange(len(sequence_lengths)) * (0.7 + group_gap)

# Create single plot for Flash Attention
fig, ax1_fa = plt.subplots(1, 1, figsize=(12, 4))

colors = [
    (240/255, 249/255, 232/255),  # light green
    (186/255, 228/255, 188/255),  # green
    (123/255, 204/255, 196/255),  # teal
    (67/255, 162/255, 202/255),   # blue
    (8/255, 104/255, 172/255)     # dark blue
]

# Process each configuration
for idx, cfg in enumerate(configs):
    # Flash Attention data
    fa_utilizations = []
    
    for seq_len in sequence_lengths:
        # reload model with required settings
        model = attn_model_config(
            MLEN=cfg["MLEN"], BLEN=cfg["BLEN"], VLEN=1024, partitioned_matrix=cfg["partitioned_matrix"],
            model_param_path=model_param_path, batch_size=1, seq_len=seq_len, output_token=100, device_num=1
        )
        
        # Flash Attention stage
        fa_attainable, fa_theoretical = model._report_flash_attn_utilization("decode")[0:2]
        fa_utilization = fa_attainable / fa_theoretical if fa_theoretical != 0 else 0
        fa_utilizations.append(fa_utilization)

    # Bar positions for configs - with spacing between groups
    pos = x + (idx - len(configs)/2) * (bar_width + bar_gap) + bar_width/2
    
    # Plot Flash Attention
    rects_fa = ax1_fa.bar(pos, fa_utilizations, width=bar_width, label=cfg["label"], color=colors[idx], alpha=0.7)

# Configure Flash Attention plot
ax1_fa.set_xlabel("Sequence Length", fontsize=18)
ax1_fa.set_ylabel("Utilization", fontsize=18)
ax1_fa.tick_params(axis='both', which='major', labelsize=14)
ax1_fa.set_title("Flash Attention Decode Systolic Array Utilization vs. Sequence Length", fontsize=20)
ax1_fa.set_xticks(x)
ax1_fa.set_xticklabels(seq_labels)
ax1_fa.set_ylim(0, 1)

# Set x-axis limits to show all bars with spacing
total_config_width = len(configs) * (bar_width + bar_gap) - bar_gap
ax1_fa.set_xlim(x[0] - total_config_width/2 - group_gap/2, x[-1] + total_config_width/2 + group_gap/2)

# Create shared legend at the bottom in two rows
handles1, labels1 = ax1_fa.get_legend_handles_labels()
# Create legend with all config labels - transparent with no frame, arranged in 2 rows
# With 5 configs, use ncol=3 to get 2 rows (3 in first row, 2 in second row)
fig.legend(handles1, labels1, loc='lower center', ncol=3, fontsize=15, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.savefig("decode_FA_GEMM_Utilization.png", dpi=300, bbox_inches='tight')
