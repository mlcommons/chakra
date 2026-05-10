import matplotlib.pyplot as plt
import numpy as np

# Example data
topologies = ['Fully-Connected', 'Ring', 'Switch']
bandwidths = ['75GB/s', '150GB/s', '300GB/s', '600GB/s', '900GB/s']

# Suppose this is your 2D data: rows=topologies, cols=bandwidths
data = np.array([
    [7085495469, 4064450176, 2905072696, 1990520371, 1708311078],  # Switch
    [4331750922, 2520517119, 1801545391, 1454825514, 1342633550],  # Ring
    [2514776156, 1838174566, 1513367965, 1338718656, 1280430992]   # Fully-Connected
])

data = data / min(data.flatten())  # Normalize by the minimum value

# Number of topologies and bandwidths
n_topo = len(topologies)
n_bw = len(bandwidths)

# X locations for the groups
x = np.arange(n_topo)

# Width of each bar
width = 0.13

fig, ax = plt.subplots(figsize=(8, 2.5))

# Draw bars for each bandwidth
for i in range(n_bw):
    ax.bar(x + i*width - width*(n_bw-1)/2, data[:, i], width, label=bandwidths[i])

# Labels and legend
ax.set_xticks(x)
ax.set_xticklabels(topologies)
ax.set_ylabel('Normalized Comm Time')
# ax.set_title('Commifferent topologies and bandwidths')
ax.legend(title='Bandwidth')
plt.tight_layout()

# plt.show()
plt.savefig("astra-sim-chakra-bw-analysis.pdf")
