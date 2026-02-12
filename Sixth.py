import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -------------  CORE FUNCTIONS  -------------
def tanh_alpha_cont(tau, tau0, alpha=1):
    exp_plus = np.exp(alpha * (tau - tau0))
    exp_minus = np.exp(-alpha * (tau - tau0))
    return (exp_plus - exp_minus) / (exp_plus + exp_minus)

def compute_e_value(tau_values, tau0_idx, alpha, direction=1):
    e_values = np.ones(len(tau_values))
    for i in range(len(tau_values)):
        if i == tau0_idx:
            e_values[i] = 1.0
        else:
            product = 1.0
            step_sign = 1 if i > tau0_idx else -1
            start, end = (tau0_idx, i) if i > tau0_idx else (i, tau0_idx)
            for j in range(start, end - 1):  # Adjusted to end before the last step
                if j + 1 >= len(tau_values):
                    break
                mu_j = tau_values[j + 1] - tau_values[j]
                term = 1 + direction * alpha * mu_j
                if abs(term) < 1e-10:  # Avoid near-zero division
                    term = 1e-10
                product *= term
            if step_sign < 0:
                product = 1 / product  # Inverse for backward
            e_values[i] = product
    return e_values

def tanh_alpha_disc(tau_values, tau0, alpha):
    tau0_idx = np.where(tau_values == tau0)[0][0]
    e_plus = compute_e_value(tau_values, tau0_idx, alpha, direction=1)
    e_minus = compute_e_value(tau_values, tau0_idx, -alpha, direction=1)
    # Ensure bounded range by clipping or normalizing
    return np.tanh(np.clip((e_plus - e_minus) / (e_plus + e_minus), -10, 10))

# Continuous case (T = R)
tau_cont = np.linspace(-20, 20, 1000)
tau0_cont_values = [-5, 0, 5]
alpha_cont = 1

# Discrete case (T = {n^2 : n in N_0})
n_values = np.arange(0, 11)
tau_disc = n_values ** 2  # Squares: 0, 1, 4, 9, 16, ...
tau0_disc_values = [0, 1, 4]  # Corresponding to n=0, 1, 2
alpha_disc = 0.5

# -------------  WHITE-THEME STYLE  -------------
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True,
    'axes.grid': True,
    'grid.linewidth': 0.6,
    'grid.alpha': 0.3,
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#666666',
})

# High-contrast color map
glow_cmap = LinearSegmentedColormap.from_list(
    'glow', ['#0066cc', '#00cc66', '#cc0066'])

# -------------  PLOTTING  -------------
fig, axs = plt.subplots(1, 2, figsize=(17, 7))

# Left: Continuous Hyperbolic Tangent on R
colors_cont = ['#0066cc', '#00cc66', '#cc0066']
for tau0, color in zip(tau0_cont_values, colors_cont):
    y = tanh_alpha_cont(tau_cont, tau0, alpha_cont)
    axs[0].plot(tau_cont, y, color=color, linewidth=3, label=f'$tanh_1$(τ, {tau0})', alpha=0.9)
    axs[0].scatter(tau_cont[::50], y[::50], c=y[::50], cmap=glow_cmap, s=30, alpha=0.8)

axs[0].axhline(0, color='#000000', lw=1.2, ls='-.', alpha=0.7)
axs[0].set_title(r'Continuous Hyperbolic Tangent on $\mathbb{R}$', color='#000000', pad=15)
axs[0].set_xlabel('τ', color='#000000')
axs[0].set_ylabel(r'$tanh_α(τ, τ₀)$', color='#000000')
axs[0].legend(loc='lower right', facecolor='#ffffff', edgecolor='#666666')
axs[0].set_xlim(-20, 20)
axs[0].set_ylim(-1.05, 1.05)

# Right: Discrete Hyperbolic Tangent on Squares Time Scale
colors_disc = ['#0066cc', '#00cc66', '#cc0066']
for tau0, color in zip(tau0_disc_values, colors_disc):
    y = tanh_alpha_disc(tau_disc, tau0, alpha_disc)
    axs[1].plot(tau_disc, y, color=color, linewidth=3, marker='o', markersize=8,
                label='$tanh_{0.5}$'f'(τ, {int(tau0)})')
    axs[1].scatter(tau_disc, y, c=y, cmap=glow_cmap, s=100, edgecolors=color, linewidths=1.2)

axs[1].axhline(0, color='#000000', lw=1.2, ls='-.', alpha=0.7)
axs[1].set_title(r'Discrete Hyperbolic Tangent on $\mathbb{T} = \{n^2\}$', color='#000000', pad=15)
axs[1].set_xlabel('τ', color='#000000')
axs[1].set_ylabel(r'$tanh_α(τ, τ₀)$', color='#000000')
axs[1].legend(loc='lower right', facecolor='#ffffff', edgecolor='#666666')
axs[1].set_ylim(-1.05, 1.05)
axs[1].set_xlim(-1, 120)  # Adjusted for visibility

plt.tight_layout(pad=3)
plt.show()