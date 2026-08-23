import numpy as np
import matplotlib.pyplot as plt

from project_paths import script_output

# =====================================
# Ground Truth Parameters
# =====================================

alpha_true = -0.5

beta_true = 0.95

sigma_eta_true = 0.30

T = 4000

# =====================================
# Simulate SV Model
# =====================================

log_sigma2 = np.zeros(T)

log_sigma2[0] = (
    alpha_true /
    (1 - beta_true)
)

for t in range(1, T):

    eta = np.random.normal(
        0,
        sigma_eta_true
    )

    log_sigma2[t] = (
        alpha_true
        + beta_true * log_sigma2[t - 1]
        + eta
    )

sigma = np.exp(
    log_sigma2 / 2
)

eps = np.random.normal(
    0,
    1,
    T
)

returns = sigma * eps

# =====================================
# Save Data
# =====================================

np.save(
    script_output("synthetic_sv_returns.npy"),
    returns
)

# =====================================
# Plot Returns
# =====================================

plt.figure(figsize=(12, 5))

plt.plot(
    returns,
    linewidth=0.8
)

plt.title(
    "Synthetic SV Returns"
)

plt.xlabel(
    "Time"
)

plt.ylabel(
    "Return"
)

plt.tight_layout()

plt.show()

# =====================================
# Plot Latent Volatility
# =====================================

plt.figure(figsize=(12, 5))

plt.plot(
    sigma,
    linewidth=0.8
)

plt.title(
    "True Latent Volatility"
)

plt.xlabel(
    "Time"
)

plt.ylabel(
    "Volatility"
)

plt.tight_layout()

plt.show()

# =====================================
# Print Truth
# =====================================

print("\nGround Truth Parameters")

print("--------------------------")

print(
    "alpha =",
    alpha_true
)

print(
    "beta =",
    beta_true
)

print(
    "sigma_eta =",
    sigma_eta_true
)
