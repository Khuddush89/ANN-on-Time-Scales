import numpy as np

# Time scale T = {1/(n+1) for n in range(0,11)}
T = np.array([1.0 / (n + 1) for n in range(11)])[::-1]  # [1/11, 1/10, ..., 1]
assert np.all(np.diff(T) > 0)

# Params
M = np.array([0.5, 0.3, 0.2], dtype=float)  # Initial weights for hidden layer
D = np.array([0.4, 0.3, 0.2], dtype=float)  # Initial weights for output layer
beta = 0.15  # Slightly increased to speed convergence
E_limit = 0.01
max_iter = 5000  # Increased to ensure convergence
np.random.seed(42)  # For reproducibility

# Time-scale exponential helper
def compute_e_values(tau_values, tau0_idx, alpha):
    """e_alpha(t, t0) over the grid tau_values (ascending), constant alpha."""
    e = np.ones_like(tau_values, dtype=float)
    # Forward from t0
    for i in range(tau0_idx, len(tau_values) - 1):
        mu = tau_values[i + 1] - tau_values[i]
        e[i + 1] = e[i] * max(1.0 + alpha * mu, 1e-12)  # Guard positivity
    # Backward from t0
    for i in range(tau0_idx, 0, -1):
        mu = tau_values[i] - tau_values[i - 1]
        e[i - 1] = e[i] / max(1.0 + alpha * mu, 1e-12)
    return e

# Anchor t0 at t=1 (last index because T is ascending)
tau0_idx = len(T) - 1  # Index of 1.0
e_plus = compute_e_values(T, tau0_idx, alpha=1.0)  # e_1
e_minus = compute_e_values(T, tau0_idx, alpha=-1.0)  # e_{-1}
f_T = (e_plus - e_minus) / (e_plus + e_minus)  # Bipolar sigmoid on the grid

def interp_and_slope(x, X=T, Y=f_T):
    """Linear interpolation y(x) and local slope dy/dx on an ascending grid X."""
    if x <= X[0]:
        i0, i1 = 0, 1
    elif x >= X[-1]:
        i0, i1 = len(X) - 2, len(X) - 1
    else:
        i1 = np.searchsorted(X, x, side='right')
        i0 = i1 - 1 if i1 > 0 else 0
    dx = X[i1] - X[i0]
    if dx == 0:
        return Y[i0], 0.0  # Avoid division by zero at boundaries
    slope = (Y[i1] - Y[i0]) / dx
    y = Y[i0] + slope * (x - X[i0])
    return y, slope

# Training loop with logging
iteration = 0
while iteration < max_iter:
    iteration += 1
    E_total = 0.0
    grad_M = np.zeros_like(M)
    grad_D = np.zeros_like(D)
    for xi in T:
        p = xi
        u = xi  # Target function u(xi) = xi
        # Hidden layer: q_j = f(m_j * p)
        z = M * p
        q = np.empty_like(M)
        df_hidden = np.empty_like(M)
        for j in range(3):
            q[j], df_hidden[j] = interp_and_slope(z[j])  # f and f'(·) at m_j p
            if df_hidden[j] == 0:
                df_hidden[j] = 1e-6  # Small default slope to avoid zero gradient
        # Output: s = f(D·q)
        ws = np.dot(D, q)
        s, df_out = interp_and_slope(ws)  # f and f'(·) at weighted sum
        if df_out == 0:
            df_out = 1e-6  # Avoid zero slope at boundaries
        # Loss and gradients
        E = 0.5 * (u - s)**2
        E_total += E
        g = (s - u) * df_out  # dE/ds * f'(ws)
        # dE/dD = g * q (element-wise, adjusted for slope)
        grad_D += g * q
        # dE/dM_j = g * D_j * f'(m_j p) * p (element-wise)
        grad_M += g * D * df_hidden * p
    # SGD step (average over samples)
    M -= beta * grad_M / len(T)
    D -= beta * grad_D / len(T)
    if iteration % 100 == 0 or E_total <= E_limit:
        print(f"Iteration {iteration}: E_total = {E_total:.4f}, M = {M.round(4)}, D = {D.round(4)}")
    if E_total <= E_limit:
        print(f"Converged at Iteration {iteration}: E_total = {E_total:.4f}, M = {M.round(4)}, D = {D.round(4)}")
        break
if iteration == max_iter:
    print(f"Max iterations {max_iter} reached: E_total = {E_total:.4f}, M = {M.round(4)}, D = {D.round(4)}")

# Post-process to ensure table alignment (optional adjustment if needed)
if E_total > 0.01 and iteration == max_iter:
    print("Warning: Did not converge to E_limit. Consider increasing max_iter or adjusting beta.")