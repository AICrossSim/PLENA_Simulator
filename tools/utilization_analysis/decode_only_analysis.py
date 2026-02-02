import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.size'] = 6

# ---- Data sets (two decode cases) ----
low_context_decode = {'operations': {'embedding': 16384, 'attention': 63979520.0, 'ffn': 13762560.0, 'lm_head': 256512}, 'attainable_FLOPS': {'embedding': 536870912, 'attention': 577807319040.0, 'ffn': 450971566080.0, 'lm_head': 8405385216}, 'theoretical_FLOPS': {'embedding': 4294967296, 'attention': 33973191311360.0, 'ffn': 3607772528640.0, 'lm_head': 67243081728}}

high_context_decode = {'operations': {'embedding': 16384, 'attention': 127959040.0, 'ffn': 13762560.0, 'lm_head': 256512}, 'attainable_FLOPS': {'embedding': 536870912, 'attention': 1101927546880.0, 'ffn': 450971566080.0, 'lm_head': 8405385216}, 'theoretical_FLOPS': {'embedding': 4294967296, 'attention': 67516885893120.0, 'ffn': 3607772528640.0, 'lm_head': 67243081728}}


# Color palette (provided)
colors = {
    "orange": tuple(i / 255 for i in (231, 98, 84)),
    "yellow": tuple(i / 255 for i in (111, 97, 130)),
    "light_blue": tuple(i / 255 for i in (190, 195, 137)),
    "dark_blue": tuple(i / 255 for i in (55, 103, 149))
}
pie_colors = [colors["orange"], colors["yellow"], colors["light_blue"], colors["dark_blue"]]

def make_decode_figure(title, data, outfile):
    ops = data['operations']
    att = data['attainable_FLOPS']
    theo = data['theoretical_FLOPS']

    labels = list(ops.keys())
    sizes = [ops[k] for k in labels]
    ratios = [(att[k] / theo[k]) * 100 if theo[k] != 0 else 0 for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2))

    # Pie (no percentages shown)
    axes[0].pie(sizes, startangle=90, colors=pie_colors)
    # axes[0].set_title(f'{title}\nGEMM Distribution', fontsize=7)
    axes[0].axis('equal')

    # Bar
    axes[1].bar(labels, ratios, color=pie_colors)
    # axes[1].set_title('Attainable / Theoretical Performance', fontsize=7)
    axes[1].set_ylabel('Utilization (%)')
    axes[1].set_ylim(0, 50)

    # Legend shared within this figure
    fig.legend(labels, loc='lower center', ncol=4, fontsize=6)

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return outfile

out1 = make_decode_figure("Decode (2k context length)", low_context_decode, "decode_5k.png")
out2 = make_decode_figure("Decode (10k context length)", high_context_decode, "decode_10k.png")

print(out1)
print(out2)
