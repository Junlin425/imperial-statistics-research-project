import numpy as np
import matplotlib.pyplot as plt

# =====================================
# Parameters
# =====================================

alpha = -0.5

beta = 0.98

sigma_eta = 0.3

T = 4000

# =====================================
# Arrays
# =====================================

log_sigma2 = np.zeros(T)

sigma = np.zeros(T)

y = np.zeros(T)

# =====================================
# Initial value
# =====================================

log_sigma2[0] = alpha / (1 - beta)

# =====================================
# Simulation
# =====================================

for t in range(1, T):

    eta = np.random.normal(
        0,
        sigma_eta
    )

    log_sigma2[t] = (
        alpha
        + beta * log_sigma2[t-1]
        + eta
    )

# =====================================
# Generate returns
# =====================================

sigma = np.exp(
    log_sigma2 / 2
)

epsilon = np.random.normal(
    0,
    1,
    T
)

y = sigma * epsilon

# =====================================
# Plot returns
# =====================================

plt.figure(figsize=(12,5))

plt.plot(y)

plt.title(
    "Simulated SV Returns"
)

plt.show()

# =====================================
# Plot volatility
# =====================================

plt.figure(figsize=(12,5))

plt.plot(sigma)

plt.title(
    "Latent Volatility"
)

plt.show()