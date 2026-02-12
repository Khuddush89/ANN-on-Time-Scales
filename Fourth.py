import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -------------  CORE FUNCTIONS  -------------
def f_cont(tau, tau0, beta=1):
    return 1 / (1 + np.exp(-beta * (tau - tau0)))

tau_cont = np.linspace(-20, 20, 1000)
tau0_cont_values = [-5, 0, 5]

k_values   = np.arange(0, 11)
tau_disc   = 2 ** k_values
beta_disc  = 2
tau0_disc_values = [1, 2, 4]

# -------------  WHITE-THEME STYLE  -------------
plt.style.use('default')
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'figure.dpi'       : 300,
    'savefig.dpi'      : 600,
    'axes.titlesize'   : 18,
    'axes.labelsize'   : 14,
    'xtick.labelsize'  : 12,
    'ytick.labelsize'  : 12,
    'legend.fontsize'  : 12,
    'legend.frameon'   : True,
    'legend.fancybox'  : True,
    'legend.shadow'    : True,
    'axes.grid'        : True,
    'grid.linewidth'   : 0.6,
    'grid.alpha'       : 0.3,
    'figure.facecolor' : '#ffffff',
    'axes.facecolor'   : '#ffffff',
    'axes.edgecolor'   : '#666666',
})

# high-contrast colour-map on white
glow_cmap = LinearSegmentedColormap.from_list(
    'glow', ['#0066cc', '#00cc66', '#cc0066'])

# -------------  PLOTTING  -------------
fig, axs = plt.subplots(1, 2, figsize=(17, 7))

# Left: Continuous
colors_cont = ['#0066cc', '#00cc66', '#cc0066']
for tau0, color in zip(tau0_cont_values, colors_cont):
    y = f_cont(tau_cont, tau0)
    axs[0].plot(tau_cont, y, color=color, linewidth=3, label=f'f(τ, {tau0})', alpha=0.9)
    axs[0].scatter(tau_cont[::20], y[::20], c=y[::20], cmap=glow_cmap, s=20, alpha=0.8)

axs[0].axhline(0.5, color='#000000', lw=1.2, ls='-.', alpha=0.7)
axs[0].set_title(r'Continuous Sigmoid on $\mathbb{R}$', color='#000000', pad=15)
axs[0].set_xlabel('τ', color='#000000')
axs[0].set_ylabel('f(τ, τ₀)', color='#000000')
axs[0].legend(loc='lower right', facecolor='#ffffff', edgecolor='#666666')
axs[0].set_xlim(-20, 20)
axs[0].set_ylim(-0.05, 1.05)

# Right: Discrete
colors_disc = ['#0066cc', '#00cc66', '#cc0066']
for tau0, color in zip(tau0_disc_values, colors_disc):
    tau0_idx = np.where(tau_disc == tau0)[0][0]
    e_values = np.ones(len(tau_disc))
    for i in range(len(tau_disc)):
        if i == tau0_idx:
            e_values[i] = 1.0
        else:
            product = 1.0
            if i > tau0_idx:
                for j in range(tau0_idx, i):
                    mu_j = tau_disc[j+1] - tau_disc[j]
                    product *= 1 / (1 + beta_disc * mu_j)
            else:
                for j in range(i, tau0_idx):
                    mu_j = tau_disc[j+1] - tau_disc[j]
                    product *= (1 + beta_disc * mu_j)
            e_values[i] = product
    f_values = 1 / (1 + e_values)

    axs[1].plot(tau_disc, f_values, color=color, linewidth=3, marker='o', markersize=8,
                label=f'f(τ, {int(tau0)})')
    axs[1].scatter(tau_disc, f_values, c=f_values, cmap=glow_cmap,
                   s=120, edgecolors=color, linewidths=1.2)

axs[1].axhline(0.5, color='#000000', lw=1.2, ls='-.', alpha=0.7)
axs[1].set_title(r'Discrete Sigmoid on $T = 2^{\mathbb{N}_0}$', color='#000000', pad=15)
axs[1].set_xlabel('τ', color='#000000')
axs[1].set_ylabel('f(τ, τ₀)', color='#000000')
axs[1].set_xscale('log')
axs[1].legend(loc='lower right', facecolor='#ffffff', edgecolor='#666666')
axs[1].set_ylim(-0.05, 1.05)
axs[1].set_xlim(0.8, 1.2e3)

plt.tight_layout(pad=3)
plt.show()