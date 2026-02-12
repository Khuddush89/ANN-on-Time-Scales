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
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.set_facecolor('#ffffff')
ax.axis('off')

# ANN layout
layers = 3
neurons = [1, 3, 1]
bias = [False, False, False]
x_gap = 2.0
r = 0.18
x = [1.0 + i * x_gap for i in range(layers)]
y_sets = [[((n - 1) / 2 - k) * 2.2 * r for k in range(n)] for n in neurons]

# Create vibrant gradient colors
def create_vibrant_cmap(color1, color2):
    return LinearSegmentedColormap.from_list('vibrant', [color1, color2])

# Layer colors
cmap_in = create_vibrant_cmap('#6a11cb', '#2575fc')
cols_in = [cmap_in(i / max(1, len(y_sets[0]) - 1)) for i in range(len(y_sets[0]))]
cmap_hid = create_vibrant_cmap('#ff6b6b', '#ff9e3d')
cols_hid = [cmap_hid(i / max(1, len(y_sets[1]) - 1)) for i in range(len(y_sets[1]))]
cmap_out = create_vibrant_cmap('#00b09b', '#96c93d')
cols_out = [cmap_out(i / max(1, len(y_sets[2]) - 1)) for i in range(len(y_sets[2]))]
cols = [cols_in, cols_hid, cols_out]

# Draw neurons
for x_pos, y_list, col_list in zip(x, y_sets, cols):
    for y, c in zip(y_list, col_list):
        circ = Circle((x_pos, y), r, facecolor=c, edgecolor='white', linewidth=1.5, alpha=0.95)
        ax.add_patch(circ)
        highlight = Circle((x_pos - r*0.15, y + r*0.15), r*0.4, facecolor='white', alpha=0.3, edgecolor='none')
        ax.add_patch(highlight)

# Draw connections
for i in range(layers - 1):
    for idx, y1 in enumerate(y_sets[i]):
        for j, y2 in enumerate(y_sets[i + 1]):
            mid_color = np.array(cols[i][idx][:3]) * 0.6 + np.array(cols[i+1][j][:3]) * 0.4
            arrow = FancyArrow(
                x[i] + r, y1, x_gap - 2 * r, y2 - y1,
                width=0.01, head_width=0.08, head_length=0.08,
                length_includes_head=True, color=mid_color, alpha=0.7, lw=0.8, zorder=1
            )
            ax.add_patch(arrow)

# Weight labels
def place_weight(x0, y0, x1, y1, text):
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    nx, ny = -dy, dx
    nrm = np.hypot(nx, ny) or 1
    nx, ny = nx / nrm, ny / nrm
    xt, yt = xm + 0.11 * nx, ym + 0.11 * ny
    ax.text(xt, yt, text, ha='center', va='center', fontsize=9, color='#0d47a1')

# Input → Hidden
for yi_idx, yi in enumerate(y_sets[0]):
    for j, yh in enumerate(y_sets[1]):
        place_weight(x[0], yi, x[1], yh, f'$m_{{{j+1}{yi_idx+1}}}$')

# Hidden → Output
for j, yh in enumerate(y_sets[1]):
    place_weight(x[1], yh, x[2], y_sets[2][0], f'$d_{{1{j+1}}}$')

# Node labels
ax.text(x[0], y_sets[0][0], 'q', ha='center', va='center', fontsize=10, weight='bold', color='black')
for j, yh in enumerate(y_sets[1]):
    ax.text(x[1], yh, f'$q_{{{j+1}}}$', ha='center', va='center', fontsize=10, weight='bold', color='black')
ax.text(x[2], y_sets[2][0], 'Q', ha='center', va='center', fontsize=10, weight='bold', color='black')

# Layer labels
labels = ['Input layer', 'Hidden layer', 'Output layer']
for x_pos, label in zip(x, labels):
    ax.text(x_pos, min(y_sets[labels.index(label)]) - 0.45, label, ha='center', va='center', fontsize=11, style='italic', color='#262626')

# Set plot limits
all_x = [x_pos for x_pos in x for _ in y_sets[x.index(x_pos)]]
all_y = [y for y_list in y_sets for y in y_list]
pad = 0.25
ax.set_xlim(min(all_x) - r - pad, max(all_x) + r + pad)
ax.set_ylim(min(all_y) - r - pad, max(all_y) + r + pad)

plt.tight_layout()
plt.show()