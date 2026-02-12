import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
from matplotlib.colors import LinearSegmentedColormap

# Global styling
plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.facecolor': '#ffffff',
    'figure.facecolor': '#ffffff',
})

# Create figure and axis
fig, ax = plt.subplots(figsize=(7, 3))
ax.set_facecolor('#ffffff')
ax.axis('off')

# ANN layout
layers = 3
neurons = [3, 4, 2]
x_gap = 2.0
r = 0.25
x_pos = [1.0 + i * x_gap for i in range(layers)]
y_pos = [[(k - (n - 1) / 2) * 2.2 * r for k in range(n)] for n in neurons]

# Create vibrant gradient colors
def create_vibrant_cmap(color1, color2):
    return LinearSegmentedColormap.from_list('vibrant', [color1, color2])

# Layer colors
cmap_in = create_vibrant_cmap('#6a11cb', '#2575fc')
cols_in = [cmap_in(i / max(1, len(y_pos[0]) - 1)) for i in range(len(y_pos[0]))]
cmap_hid = create_vibrant_cmap('#ff6b6b', '#ff9e3d')
cols_hid = [cmap_hid(i / max(1, len(y_pos[1]) - 1)) for i in range(len(y_pos[1]))]
cmap_out = create_vibrant_cmap('#00b09b', '#96c93d')
cols_out = [cmap_out(i / max(1, len(y_pos[2]) - 1)) for i in range(len(y_pos[2]))]
all_cols = [cols_in, cols_hid, cols_out]

# Draw neurons
for x, y_list, cols in zip(x_pos, y_pos, all_cols):
    for y, c in zip(y_list, cols):
        circ = Circle((x, y), r, facecolor=c, edgecolor='white', linewidth=1.5, alpha=0.95)
        ax.add_patch(circ)
        highlight = Circle((x - r*0.15, y + r*0.15), r*0.4, facecolor='white', alpha=0.3, edgecolor='none')
        ax.add_patch(highlight)

# Draw connections
for i in range(layers - 1):
    for idx, y1 in enumerate(y_pos[i]):
        for j, y2 in enumerate(y_pos[i + 1]):
            mid_color = np.array(all_cols[i][idx][:3]) * 0.6 + np.array(all_cols[i+1][j][:3]) * 0.4
            arrow = FancyArrow(
                x_pos[i] + r, y1, x_gap - 2 * r, y2 - y1,
                width=0.015, head_width=0.09, head_length=0.09,
                length_includes_head=True, color=mid_color, alpha=0.7, lw=1.2, zorder=1
            )
            ax.add_patch(arrow)

# Add layer labels
labels = ['INPUT LAYER', 'HIDDEN LAYER', 'OUTPUT LAYER']
for x, label, color in zip(x_pos, labels, ['#6a11cb', '#ff6b6b', '#00b09b']):
    ax.text(x, max(y_pos[0]) + 0.7, label, ha='center', va='center', fontsize=11, weight='bold', color=color)

# Add neuron count labels
for i, (x, n) in enumerate(zip(x_pos, neurons)):
    ax.text(x, min(y_pos[0]) - 0.7, f'{n} neurons', ha='center', va='center', fontsize=10, color='#666666')

# Set plot limits
all_x = [x for x in x_pos for _ in y_pos[x_pos.index(x)]]
all_y = [y for y_list in y_pos for y in y_list]
pad = 0.4
ax.set_xlim(min(all_x) - r - pad, max(all_x) + r + pad)
ax.set_ylim(min(all_y) - r - pad - 0.5, max(all_y) + r + pad + 0.5)

# Add title
ax.text((min(x_pos) + max(x_pos)) / 2, max(y_pos[0]) + 1.2, '',
        ha='center', va='center', fontsize=16, weight='bold', color='#2c3e50')

plt.tight_layout()
plt.show()