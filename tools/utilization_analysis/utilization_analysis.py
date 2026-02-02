import matplotlib.pyplot as plt

llama_gemm_utilization = {'operations': {'embedding': 4096, 'attention': 262144, 'ffn': 1376256, 'lm_head': 128256}, 'attainable_FLOPS': {'embedding': 134217728, 'attention': 8589934592, 'ffn': 45097156608, 'lm_head': 4202692608}, 'theoretical_FLOPS': {'embedding': 1073741824, 'attention': 188978561024, 'ffn': 360777252864, 'lm_head': 33621540864}}

# Data
ops = llama_gemm_utilization['operations']
att = llama_gemm_utilization['attainable_FLOPS']
theo = llama_gemm_utilization['theoretical_FLOPS']

labels = list(ops.keys())
sizes = [ops[k] for k in labels]
ratios = [att[k] / theo[k] if theo[k] != 0 else 0 for k in labels]

# Create side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Pie chart
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.set_title('Operation Distribution by Type')
ax1.axis('equal')

# Right: Bar chart
bars = ax2.bar(range(len(labels)), ratios, tick_label=labels)
ax2.set_ylabel('Attainable / Theoretical')
ax2.set_title('Performance Ratio by Operator')
ax2.axhline(1.0, linestyle='--', color='red', linewidth=1)

# Annotate bars
for i, b in enumerate(bars):
    height = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2, height, f'{height*100:.2f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
