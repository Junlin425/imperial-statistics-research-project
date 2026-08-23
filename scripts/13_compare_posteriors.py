import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

from project_paths import script_output

# =====================================
# Load Results
# =====================================

manual = np.load(
    script_output("abc_manual_sv.npy")
)

auto = np.load(
    script_output("abc_auto_sv.npy")
)

wasserstein = np.load(
    script_output("abc_wasserstein_sv.npy")
)

# =====================================
# Extract Parameters
# =====================================

alpha_manual = manual[:,0]
beta_manual = manual[:,1]
sigma_manual = manual[:,2]

alpha_auto = auto[:,0]
beta_auto = auto[:,1]
sigma_auto = auto[:,2]

alpha_wass = wasserstein[:,0]
beta_wass = wasserstein[:,1]
sigma_wass = wasserstein[:,2]

# =====================================
# Figure 1 : Alpha
# =====================================

plt.figure(figsize=(8,5))

x = np.linspace(
    min(
        alpha_manual.min(),
        alpha_auto.min(),
        alpha_wass.min()
    ),
    max(
        alpha_manual.max(),
        alpha_auto.max(),
        alpha_wass.max()
    ),
    500
)

plt.plot(
    x,
    gaussian_kde(alpha_manual)(x),
    label="Manual ABC",
    linewidth=2
)

plt.plot(
    x,
    gaussian_kde(alpha_auto)(x),
    label="Auto ABC",
    linewidth=2
)

plt.plot(
    x,
    gaussian_kde(alpha_wass)(x),
    label="Wasserstein ABC",
    linewidth=2
)

plt.xlabel(r"$\alpha$")
plt.ylabel("Density")

plt.title(
    "Posterior Comparison: Alpha"
)

plt.legend()

plt.tight_layout()

plt.show()

# =====================================
# Figure 2 : Beta
# =====================================

plt.figure(figsize=(8,5))

x = np.linspace(
    min(
        beta_manual.min(),
        beta_auto.min(),
        beta_wass.min()
    ),
    max(
        beta_manual.max(),
        beta_auto.max(),
        beta_wass.max()
    ),
    500
)

plt.plot(
    x,
    gaussian_kde(beta_manual)(x),
    label="Manual ABC",
    linewidth=2
)

plt.plot(
    x,
    gaussian_kde(beta_auto)(x),
    label="Auto ABC",
    linewidth=2
)

plt.plot(
    x,
    gaussian_kde(beta_wass)(x),
    label="Wasserstein ABC",
    linewidth=2
)

plt.xlabel(r"$\beta$")
plt.ylabel("Density")

plt.title(
    "Posterior Comparison: Beta"
)

plt.legend()

plt.tight_layout()

plt.show()

# =====================================
# Figure 3 : Sigma Eta
# =====================================

plt.figure(figsize=(8,5))

x = np.linspace(
    min(
        sigma_manual.min(),
        sigma_auto.min(),
        sigma_wass.min()
    ),
    max(
        sigma_manual.max(),
        sigma_auto.max(),
        sigma_wass.max()
    ),
    500
)

plt.plot(
    x,
    gaussian_kde(sigma_manual)(x),
    label="Manual ABC",
    linewidth=2
)

plt.plot(
    x,
    gaussian_kde(sigma_auto)(x),
    label="Auto ABC",
    linewidth=2
)

plt.plot(
    x,
    gaussian_kde(sigma_wass)(x),
    label="Wasserstein ABC",
    linewidth=2
)

plt.xlabel(r"$\sigma_\eta$")
plt.ylabel("Density")

plt.title(
    r"Posterior Comparison: $\sigma_\eta$"
)

plt.legend()

plt.tight_layout()

plt.show()
