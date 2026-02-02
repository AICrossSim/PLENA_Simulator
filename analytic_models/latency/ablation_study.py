import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from overall_inference_estimation import model_config

batch_size = 4

# Experiment configurations
# sample_1: baseline, MLEN = BLEN = 64, VLEN = 1024, using selfattention
# sample_2: MLEN = BLEN = 64, VLEN = 1024, using flash attention
# sample_3: MLEN = 1024, BLEN = 4, VLEN = 1024, using flash attention, not partitioned optimised
# sample_4: MLEN = 1024, BLEN = 4, VLEN = 1024, using flash attention, partitioned optimised


def compute_attention_ffn_distribution(model_param_path, hardware_config, batch_size, input_seq, decode_seq_lengths,
                                       use_flash_attention=True, partitioned_optimized=False, device_num=1):
    """
    Compute attention and FFN computation distribution for prefill and decode stages at different sequence lengths.
    
    Args:
        model_param_path: Path to model parameter JSON file
        hardware_config: Dictionary with MLEN, BLEN, VLEN
        batch_size: Batch size
        input_seq: Input sequence length (for prefill)
        decode_seq_lengths: List of total sequence lengths for decode (e.g., [10k, 40k, 80k])
        use_flash_attention: Whether to use flash attention (True) or self attention (False)
        partitioned_optimized: Whether to use partitioned optimization (for sample 4)
        device_num: Number of devices
    
    Returns:
        dict: Dictionary with 'prefill' and 'decode' keys
              'prefill' contains 'attention' and 'ffn' for input_seq
              'decode' contains list of dicts, each with 'attention' and 'ffn' for each decode_seq_length
    """
    print("<", "="*10, "starting computation", "="*10, ">")
    print("hardware config: ", hardware_config)
    print("use flash attention: ", use_flash_attention)
    print("partitioned optimized: ", partitioned_optimized)

    print("starting computation")
    # Use max decode length for model initialization
    max_decode_len = max(decode_seq_lengths) if decode_seq_lengths else 0
    model = model_config(
        model_param_path=model_param_path,
        hardware_config=hardware_config,
        batch_size=batch_size,
        seq_len=int(input_seq),
        output_token=1000,
        device_num=device_num
    )
    
    results = {}
    
    # Prefill stage
    mode = "prefill"
    if use_flash_attention:
        attention_inst = model.flash_attention(mode, partitioned_optimized)
    else:
        attention_inst = model.self_attention(mode)

    ffn_inst = model.feed_forward(mode)
    
    results['prefill'] = {
        'attention': attention_inst,
        'ffn': ffn_inst,
        'total': attention_inst + ffn_inst
    }
    print("completed prefill computation")
    print("prefill attention: ", attention_inst)
    print("prefill ffn: ", ffn_inst)
    
    # Decode stage - compute for each sequence length
    mode = "decode"
    results['decode'] = []
    
    for total_seq_len in decode_seq_lengths:
        model.kv_size = total_seq_len # Reset KV size to initial value
        
        # Accumulate instruction counts as KV cache grows
        attention_inst_decode = 0
        ffn_inst_decode = 0
        
        for token_idx in range(total_seq_len):
            if use_flash_attention:
                attention_inst_per_token = model.flash_attention(mode, partitioned_optimized)
            else:
                attention_inst_per_token = model.self_attention(mode)
            ffn_inst_per_token = model.feed_forward(mode)
            
            attention_inst_decode += attention_inst_per_token
            ffn_inst_decode += ffn_inst_per_token
            
            # KV cache grows by 1 for each token
            model.kv_size += 1
        
        results['decode'].append({
            'attention': attention_inst_decode,
            'ffn': ffn_inst_decode,
            'total': attention_inst_decode + ffn_inst_decode
        })
        print(f"completed decode computation for seq_len={total_seq_len}")
        print(f"  decode attention: {attention_inst_decode:.2e}, ffn: {ffn_inst_decode:.2e}")
    
    return results


def plot_ablation_study():
    """
    Conduct ablation study experiments and plot bar chart showing
    FFN + Attention computation distribution across sequence lengths.
    Three plots: Left (Prefill 5.6k), Middle (Decode 5k), Right (Decode 8k)
    Y-axis: Execution Time (% relative to S1)
    """
    colors = [
        "#762a83",  # deep purple
        "#af8dc3",  # light violet
        "#e7d4e8",  # very light purple-pink
        "#d9f0d3",  # pale green
        "#7fbf7b",  # medium green
        # Optional sixth if needed: "#1b7837"  # dark green
    ]

    # Get model parameter path
    current_dir = Path(__file__).resolve().parents[3]
    model_param_path = os.path.join(current_dir, "doc/Model_Lib/llama-3.1-8b.json")
    
    # Sequence lengths: prefill at 5.6k, decode at 5k and 8k total
    input_seq = 5.6 * 1000
    decode_seq_lengths = [5 * 1000, 8 * 10000]  # Total sequence lengths after decode (5k and 8k)
    
    # Define experiment samples
    experiments = [
        {
            'name': 'Sample 1',
            'hardware_config': {'MLEN': 64, 'BLEN': 64, 'VLEN': 1024},
            'use_flash_attention': False,
            'partitioned_optimized': False,
            'label': 'Sample 1 (Baseline): MLEN=BLEN=64, VLEN=1024'
        },
        {
            'name': 'Sample 2',
            'hardware_config': {'MLEN': 64, 'BLEN': 64, 'VLEN': 1024},
            'use_flash_attention': True,
            'partitioned_optimized': False,
            'label': 'Sample 2 (Flash Attention): MLEN=BLEN=64, VLEN=1024'
        },
        {
            'name': 'Sample 3',
            'hardware_config': {'MLEN': 1024, 'BLEN': 4, 'VLEN': 1024},
            'use_flash_attention': True,
            'partitioned_optimized': True,
            'label': 'Sample 3 (Flash Attention + Flattened Systolic Array): MLEN=1024, BLEN=4, VLEN=1024'
        }
    ]
    
    # Collect data for all sequence lengths
    # Structure: data[sample_idx][seq_idx] = {'attention': ..., 'ffn': ...}
    all_data = []
    experiment_names = []
    
    for exp in experiments:
        results = compute_attention_ffn_distribution(
            model_param_path=model_param_path,
            hardware_config=exp['hardware_config'],
            batch_size=batch_size,
            input_seq=input_seq,
            decode_seq_lengths=decode_seq_lengths,
            use_flash_attention=exp['use_flash_attention'],
            partitioned_optimized=exp['partitioned_optimized']
        )
        
        experiment_names.append(exp['name'])
        sample_data = []
        
        # Add prefill data (at 5.6k)
        sample_data.append({
            'attention': results['prefill']['attention'],
            'ffn': results['prefill']['ffn']
        })
        
        # Add decode data (at 10k, 40k, 80k)
        for decode_result in results['decode']:
            sample_data.append({
                'attention': decode_result['attention'],
                'ffn': decode_result['ffn']
            })
        
        all_data.append(sample_data)
    
    print("completed computation")
    
    # Prepare data for plotting
    num_samples = len(experiments)
    
    # Extract data for each plot: prefill (index 0), decode 5k (index 1), decode 8k (index 2)
    prefill_attention = np.array([all_data[s][0]['attention'] for s in range(num_samples)])
    prefill_ffn = np.array([all_data[s][0]['ffn'] for s in range(num_samples)])
    
    decode_5k_attention = np.array([all_data[s][1]['attention'] for s in range(num_samples)])
    decode_5k_ffn = np.array([all_data[s][1]['ffn'] for s in range(num_samples)])
    
    decode_8k_attention = np.array([all_data[s][2]['attention'] for s in range(num_samples)])
    decode_8k_ffn = np.array([all_data[s][2]['ffn'] for s in range(num_samples)])
    
    # Normalize all values relative to S1 (S1 = 100%)
    s1_prefill_total = prefill_attention[0] + prefill_ffn[0]
    s1_decode_5k_total = decode_5k_attention[0] + decode_5k_ffn[0]
    s1_decode_8k_total = decode_8k_attention[0] + decode_8k_ffn[0]
    
    # Normalize prefill
    prefill_attention_norm = (prefill_attention / s1_prefill_total) * 100
    prefill_ffn_norm = (prefill_ffn / s1_prefill_total) * 100
    
    # Normalize decode 5k
    decode_5k_attention_norm = (decode_5k_attention / s1_decode_5k_total) * 100
    decode_5k_ffn_norm = (decode_5k_ffn / s1_decode_5k_total) * 100
    
    # Normalize decode 8k
    decode_8k_attention_norm = (decode_8k_attention / s1_decode_8k_total) * 100
    decode_8k_ffn_norm = (decode_8k_ffn / s1_decode_8k_total) * 100
    
    # Create figure with 3 subplots, shared y-axis - reduced height for compactness
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    
    # Colors - using the color palette defined above
    attention_color = colors[0]  # dark blue
    ffn_color = colors[4]  # teal
    
    # Bar positioning - make more compact
    x = np.arange(num_samples)
    bar_width = 0.5  # Slightly wider bars for better visibility
    
    # Find maximum value across all plots for consistent y-axis
    max_total = max(
        (prefill_attention_norm + prefill_ffn_norm).max(),
        (decode_5k_attention_norm + decode_5k_ffn_norm).max(),
        (decode_8k_attention_norm + decode_8k_ffn_norm).max()
    )
    y_max = max_total * 1.05  # Reduced padding to 5% for more compact y-axis
    
    # Data for each plot (normalized)
    plot_data = [
        {
            'ax': ax1,
            'xlabel': 'Prefill (5.6k)',
            'attention': prefill_attention_norm,
            'ffn': prefill_ffn_norm
        },
        {
            'ax': ax2,
            'xlabel': 'Decode (10k)',
            'attention': decode_5k_attention_norm,
            'ffn': decode_5k_ffn_norm
        },
        {
            'ax': ax3,
            'xlabel': 'Decode (80k)',
            'attention': decode_8k_attention_norm,
            'ffn': decode_8k_ffn_norm
        }
    ]
    
    # Plot each subplot
    for plot_info in plot_data:
        ax = plot_info['ax']
        attn_data = plot_info['attention']
        ffn_data = plot_info['ffn']
        
        # Plot stacked bars
        ax.bar(x, attn_data, bar_width, label='Attention', color=attention_color, edgecolor='black', alpha=0.8)
        ax.bar(x, ffn_data, bar_width, bottom=attn_data, label='FFN', color=ffn_color, edgecolor='black', alpha=0.8)
        
        # X-axis - make more compact
        ax.set_xticks(x)
        ax.set_xticklabels([f'S{i+1}' for i in range(num_samples)], fontsize=13)
        # Set title at the top instead of x-axis label
        ax.set_title(plot_info['xlabel'], fontsize=14)
        
        # Set x-axis limits to reduce space between bars
        ax.set_xlim(-0.5, num_samples - 0.5)
        
        # Set shared y-axis limits with minimal padding
        ax.set_ylim(0, y_max)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Reduce tick label sizes for more compact appearance
        ax.tick_params(axis='both', labelsize=12)
        
        # Add legend only to the first plot
        if ax == ax1:
            ax.legend(loc='upper right', fontsize=13, framealpha=0.9)
            # Y-axis label only on the leftmost plot - positioned lower
            ax.set_ylabel('Exec Time (% relative to S1)', fontsize=12)
            # Move y-axis label down by adjusting its position
            ax.yaxis.set_label_coords(-0.15, 0.50)
        else:
            # Remove y-axis labels from middle and right plots
            ax.set_ylabel('')
    
    # Add overall title
    # fig.suptitle('Ablation Study for LLaMA 3.1 8B', fontsize=18, y=1.02)
    
    # Add sample legend at the bottom in three rows (one per sample)
    sample_legend = [exp["label"] for exp in experiments]
    # Arrange in three rows: one sample per row
    legend_text =  '\n'.join(sample_legend)
    
    # More compact vertical spacing
    plt.subplots_adjust(wspace=0.15, left=0.12, right=0.95, top=0.88, bottom=0.25)
    
    # Add sample legend at the bottom in three rows
    # fig.text(0.5, 0.12, legend_text, 
    #         fontsize=12, va='top', ha='center', 
    #         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=4))

    plt.savefig('ablation_study_attention_ffn.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'ablation_study_attention_ffn.png'")
    
    # Print summary
    print("\n" + "="*70)
    print("Ablation Study Results Summary")
    print("="*70)
    for i, exp in enumerate(experiments):
        print(f"\n{exp['label']}:")
        print(f"  Prefill (5.6k) - Attention: {prefill_attention[i]:.2e}, FFN: {prefill_ffn[i]:.2e}")
        print(f"  Decode (5k) - Attention: {decode_5k_attention[i]:.2e}, FFN: {decode_5k_ffn[i]:.2e}")
        print(f"  Decode (8k) - Attention: {decode_8k_attention[i]:.2e}, FFN: {decode_8k_ffn[i]:.2e}")
    print("="*70)


if __name__ == "__main__":
    plot_ablation_study()
