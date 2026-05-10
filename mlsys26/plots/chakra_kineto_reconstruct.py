import matplotlib.pyplot as plt
import numpy as np

# ----------------------
# Input data (in microseconds)
# ----------------------
workloads = ["GPT3 5B", "GPT3 175B", "Llama3 8B", "Llama3 70B",
             "Mixtral8x7B", "Mixtral8x22B", "DeepSeek"]

Kineto_total = np.array([4756993, 122523790, 50121297, 169539857,
                         10617205, 15850134, 32575067], dtype=float)
Chakra_total = np.array([4614357, 118234851, 43391211, 165717189,
                         8495977, 15122716, 27726698], dtype=float)
astra_total  = np.array([4876814, 118233705, 43386681, 165716010, 8495393, 14242611, 27726132], dtype=float)
astra_comm   = np.array([10158, 113617612, 22726909, 151149875, 3315647, 12467803, 25816064], dtype=float)

base = Chakra_total 
astra_comp = astra_total - astra_comm

Kineto_comp = np.minimum(astra_comp, Kineto_total)
Kineto_comm = Kineto_total - Kineto_comp
Chakra_comp = np.minimum(astra_comp, Chakra_total)
Chakra_comm = Chakra_total - Chakra_comp
# ----------------------
# Normalization: divide by Kineto total
# ----------------------
base = np.where(base == 0, 1.0, base)
# base = np.where(Kineto_total == 0, 1.0, Kineto_total)

# Compute and communication assumed identical for both systems
# comp_ratio = 0.8
# comm_ratio = 0.2

# comp_n = np.full_like(Kineto_total, comp_ratio, dtype=float)
# comm_n = np.full_like(Kineto_total, comm_ratio, dtype=float)

comp_n   = Kineto_comp   / base
comm_n   = Kineto_comm   / base
replay_comp_n = Chakra_comp / base
replay_comm_n = Chakra_comm / base
astra_comp_n  = astra_comp  / base
astra_comm_n  = astra_comm  / base

# Idle time = difference between Kineto and Chakra normalized by Kineto
idle_n = (Kineto_total - Chakra_total) / Kineto_total
idle_n = np.clip(idle_n, 0, None)  # no negative idle

# ----------------------
# Layout settings
# ----------------------
bars_per_group = 2
bar_width = 0.25
intra_gap = 0.08
group_gap = 0.15

num_groups = len(workloads)
group_width = (bars_per_group - 1) * (bar_width + intra_gap) + bar_width
group_centers = np.arange(num_groups) * (group_width + group_gap)
offsets = np.array([-0.4 * (bar_width + intra_gap), +0.4 * (bar_width + intra_gap)])

# Color scheme
comp_color = "#d62728"   # red
comm_color = "#1f77b4"   # blue
idle_color = "#aaaaaa"   # gray

axis_label_size = 22
tick_size = 20

fig, ax = plt.subplots(figsize=(14, 6))

x_Kineto = group_centers + offsets[0]  # Left side
x_Chakra = group_centers + offsets[1]  # Right side


# Kineto bars (with idle)
ax.bar(x_Kineto, comp_n, bar_width, color=comp_color, zorder=3)
ax.bar(x_Kineto, comm_n, bar_width, bottom=comp_n, color=comm_color, zorder=3)
for i in range(num_groups):
    ax.bar(x_Kineto[i], idle_n[i], bar_width,
           bottom=comp_n[i] + comm_n[i], color=idle_color, zorder=3)

# Chakra bars (baseline)
ax.bar(x_Chakra, comp_n, bar_width, color=comp_color, zorder=3)
ax.bar(x_Chakra, comm_n, bar_width, bottom=comp_n, color=comm_color, zorder=3)

# ----------------------
# Axes and labels
# ----------------------
ax.set_ylabel("Normalized Execution Time", fontsize=axis_label_size)
ax.set_xticks(group_centers)
ax.set_xticklabels(workloads, fontsize=tick_size)
ax.tick_params(axis="y", labelsize=tick_size)

comp_patch = plt.Rectangle((0, 0), 1, 1, color=comp_color)
comm_patch = plt.Rectangle((0, 0), 1, 1, color=comm_color)
idle_patch = plt.Rectangle((0, 0), 1, 1, color=idle_color)

ax.legend(
    [comp_patch, comm_patch, idle_patch],
    ["Computation", "Exposed Communication", "Idle Time"],
    ncol=3,
    fontsize=20,
    loc="upper left",
    handletextpad=0.4,    # spacing between legend marker and text
    columnspacing=0.8,    # horizontal space between columns
    handlelength=1.2,     # length of the color box
    borderpad=0.3,        # inner padding of the legend box
    labelspacing=0.4      # vertical space between entries (useful if multiple rows)
)

# ----------------------
# Grid and dividers
# ----------------------
ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
boundaries = (group_centers[:-1] + group_centers[1:]) / 2.0
for b in boundaries:
    ax.axvline(b, color="#999999", linestyle=":", linewidth=1, zorder=1)


# sub_labels = ["Chakra", "Kineto"]
# sub_positions = [x_Chakra, x_Kineto]
# y_text = -0.07
# for pos, lab in zip(sub_positions, sub_labels):
#     for xi in pos:
#         ax.text(xi, y_text, lab, ha="center", va="top", fontsize=14,
#                 transform=ax.get_xaxis_transform())

# ----------------------
# Adjust limits and add sub-labels
# ----------------------
all_x = np.concatenate([x_Kineto, x_Chakra])  # include both sides
xmin, xmax = all_x.min() - bar_width * 1.4, all_x.max() + bar_width * 1.2
ax.set_xlim(xmin, xmax)

# plt.subplots_adjust(bottom=0.15)  # make room for labels
# plt.tight_layout()
plt.subplots_adjust(left=0.10, bottom=0.15, right=0.98, top=0.95)
plt.tight_layout()
plt.savefig("runtime_idle_label.pdf", bbox_inches="tight")
# plt.show()
