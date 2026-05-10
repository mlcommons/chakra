import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Data for InfiniBand 400 Gb/s ---
data_400 = {
    'Subcategory': ['AllToAll', 'AllGather', 'ReduceScatter', 'AllReduce'],
    'Total Duration (us)': [2334290.06, 766347.27, 435756.7, 15609.41]
}

# --- Data for InfiniBand 100 Gb/s ---
data_100 = {
    'Subcategory': ['AllToAll', 'AllGather', 'ReduceScatter', 'AllReduce'],
    'Total Duration (us)': [9640178.28, 3351602.3, 648624.11, 151577.8]
}


df_400 = pd.DataFrame(data_400)
df_100 = pd.DataFrame(data_100)

df = df_400.merge(df_100, on='Subcategory', suffixes=('_400Gbps', '_100Gbps'))

# Plot
x = np.arange(len(df['Subcategory']))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 4))


bars_400 = ax.bar(x - width/2, df['Total Duration (us)_400Gbps'], width, label='400 Gb/s', color='#1f77b4', alpha=0.85)
bars_100 = ax.bar(x + width/2, df['Total Duration (us)_100Gbps'], width, label='100 Gb/s', color='#ff7f0e', alpha=0.85)

ax.set_xlabel('Collective Communication Type', fontsize=20)
ax.set_ylabel('Total Duration (µs)', fontsize=20)
# ax.set_title('Comparison of Collective Communication Duration\nunder Different InfiniBand Speeds', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(df['Subcategory'], fontsize=16)
ax.legend(fontsize=18, loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.6)

ax.tick_params(axis='y', labelsize=16)  # Increase only the font size for y-axis ticks
ax.tick_params(axis='x', labelsize=16)  # Keep x-axis consistent

# Annotate ratio (optional, easy to compare)
for i, (v400, v100) in enumerate(zip(df['Total Duration (us)_400Gbps'], df['Total Duration (us)_100Gbps'])):
    ratio = v100 / v400
    ax.text(x[i], max(v400, v100) * 0.95, f"{ratio:.1f}× slower", ha='center', fontsize=18, color='darkred')

plt.tight_layout()
plt.savefig("coll_comm_ib_perf.pdf", bbox_inches='tight')
plt.close(fig)
