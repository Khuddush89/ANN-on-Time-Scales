import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, FancyBboxPatch

# -------------  white, border-free style -------------
plt.style.use('default')
plt.rcParams.update({
    'figure.dpi'      : 300,
    'savefig.dpi'     : 600,
    'font.family'     : 'DejaVu Sans',
    'font.size'       : 12,
    'axes.facecolor'  : '#ffffff',
    'figure.facecolor': '#ffffff',
})

fig, ax = plt.subplots(figsize=(7, 3.5))

# -------------  geometry -------------
# positions are identical to your original file
x0, y0  = -1.2, -1.2
w, h    = 6.8, 2.6
center_x = x0 + w / 2.0

# -------------  elements -------------
# input circle P
ax.add_patch(Circle((0, 0), 0.4, facecolor='#00e5ff', edgecolor='none'))
ax.text(0, 0, 'P', ha='center', va='center', fontsize=14, weight='bold', color='white')
ax.text(0, -0.75, 'Input', ha='center', va='center', fontsize=11, style='italic', color='#333333')

# function block f = [MP]
func = FancyBboxPatch(
    (1.6, -0.3), 1.4, 0.6,
    boxstyle='round,pad=0.02',
    facecolor='#00c853', edgecolor='none',
    mutation_scale=8
)
ax.add_patch(func)
ax.text(2.3, 0, 'f = [MP]', ha='center', va='center', fontsize=13, weight='bold', color='white')
ax.text(2.3, -0.75, 'Function block', ha='center', va='center', fontsize=11,
        style='italic', color='#333333')

# output circle Q
ax.add_patch(Circle((4.2, 0), 0.4, facecolor='#ff1744', edgecolor='none'))
ax.text(4.2, 0, 'Q', ha='center', va='center', fontsize=14, weight='bold', color='white')
ax.text(4.2, -0.75, 'Output', ha='center', va='center', fontsize=11, style='italic', color='#333333')

# arrows
kw = dict(head_width=0.12, head_length=0.12, width=0.025, color='#333333')
ax.add_patch(FancyArrow(0.4, 0, 0.9, 0, **kw))
ax.add_patch(FancyArrow(3.17, 0, 0.5, 0, **kw))

# -------------  tidy-up -------------
ax.set_xlim(x0 - 0.2, x0 + w + 0.2)
ax.set_ylim(y0 - 0.5, y0 + h + 0.3)
ax.set_aspect('equal')
ax.axis('off')

# plt.savefig('flow_white_plain.png', bbox_inches='tight', facecolor='white')
plt.show()