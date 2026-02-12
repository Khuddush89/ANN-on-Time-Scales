import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from math import factorial

# -------------  CORE FUNCTIONS  -------------
def f_bipolar_cont(tau, tau0, beta=1):
    return np.tanh(0.5 * beta * (tau - tau0))

tau_cont  = np.linspace(-20, 20, 1000)
tau0_cont = [-5, 0, 5]

n_values  = np.arange(0, 11)
tau_disc  = n_values * (n_values + 1) // 2
beta_disc = 0.3
tau0_disc = [0, 1, 3]

# -------------  WHITE THEME  -------------
plt.style.use('default')
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'figure.dpi'       : 300,
    'savefig.dpi'      : 600,
    'axes.titlesize'   : 18,
    'axes.labelsize'   : 14,
    'axes.grid'        : True,
    'grid.linewidth'   : 0.8,
    'grid.alpha'       : 0.25,
    'figure.facecolor' : '#ffffff',
    'axes.facecolor'   : '#ffffff',
})

glow_cmap = LinearSegmentedColormap.from_list('glow', ['#0066cc', '#00cc66', '#cc0066'])

# -------------  PLOTTING  -------------
fig, axs = plt.subplots(1, 2, figsize=(18, 7.5))

# ---------- Left: Continuous ----------
colors = ['#0066cc', '#00cc66', '#cc0066']
for tau0, color in zip(tau0_cont, colors):
    y = f_bipolar_cont(tau_cont, tau0)
    axs[0].plot(tau_cont, y, color=color, linewidth=4, label=f'f(τ, {tau0})')
    axs[0].scatter(tau_cont[::30], y[::30], c=y[::30], cmap=glow_cmap,
                   s=40, edgecolors='white', linewidths=2, alpha=0.9)

axs[0].axhline(0, color='black', lw=2, ls='--', alpha=0.7)
axs[0].set_title(r'Continuous Bipolar Sigmoid on $\mathbb{R}$', pad=20)
axs[0].set_xlabel('τ')
axs[0].set_ylabel('f(τ, τ₀)')
axs[0].legend(loc='lower right')
axs[0].set_xlim(-20, 20)
axs[0].set_ylim(-1.25, 1.25)

# ---------- Right: Discrete ----------
colors = ['#0066cc', '#00cc66', '#cc0066']
for tau0, color in zip(tau0_disc, colors):
    idx = np.where(tau_disc == tau0)[0][0]
    e_vals = np.ones(len(tau_disc))
    for i in range(len(tau_disc)):
        if i == idx:
            e_vals[i] = 1.0
        else:
            step = 1 if i > idx else -1
            prod = 1.0
            rng = range(idx, i) if i > idx else range(i, idx)
            for j in rng:
                mu = tau_disc[j+1] - tau_disc[j]
                term = 1 + (-beta_disc) * mu
                if term == 0:
                    term = 1e-10
                prod *= term ** step
            e_vals[i] = prod
    f_vals = np.nan_to_num((1 - e_vals) / (1 + e_vals))
    axs[1].plot(tau_disc, f_vals, color=color, linewidth=4,
                marker='o', markersize=12, label=f'f(τ, {int(tau0)})')
    axs[1].scatter(tau_disc, f_vals, c=f_vals, cmap=glow_cmap,
                   s=200, edgecolors=color, linewidths=2)

axs[1].axhline(0, color='black', lw=2, ls='--', alpha=0.7)
axs[1].set_title(r'Discrete Bipolar Sigmoid on Triangular Numbers', pad=20)
axs[1].set_xlabel('τ')
axs[1].set_ylabel('f(τ, τ₀)')
axs[1].legend(loc='lower right')
axs[1].set_xlim(-1, 65)
axs[1].set_ylim(-1.25, 1.25)

plt.tight_layout(pad=4)
plt.show()