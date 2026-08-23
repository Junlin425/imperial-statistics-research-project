import numpy as np
import matplotlib.pyplot as plt

from project_paths import script_output

# =====================================
# Load ABC Results
# =====================================

accepted = np.load(
    script_output("abc_manual_sv.npy")
)

alpha_samples = accepted[:, 0]

beta_samples = accepted[:, 1]

sigma_eta_samples = accepted[:, 2]

# =====================================
# Summary Function
# =====================================

def print_summary(name, samples):

    mean = np.mean(samples)

    sd = np.std(samples)

    ci_lower = np.percentile(
        samples,
        2.5
    )

    ci_upper = np.percentile(
        samples,
        97.5
    )

    print(f"\n{name}")

    print("-----------------------")

    print(
        f"Mean = {mean:.4f}"
    )

    print(
        f"SD = {sd:.4f}"
    )

    print(
        f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]"
    )


# =====================================
# Print Posterior Summary
# =====================================

print_summary(
    "Alpha",
    alpha_samples
)

print_summary(
    "Beta",
    beta_samples
)

print_summary(
    "Sigma Eta",
    sigma_eta_samples
)

# =====================================
# Plot
# =====================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15,5)
)

# -------------------------------------
# Alpha
# -------------------------------------

axes[0].hist(
    alpha_samples,
    bins=20,
    density=True,
    alpha=0.7
)

axes[0].axvline(
    np.mean(alpha_samples),
    color="red",
    linestyle="--"
)

axes[0].set_title(
    "Posterior of Alpha"
)

axes[0].set_xlabel(
    r"$\alpha$"
)

axes[0].set_ylabel(
    "Density"
)

# -------------------------------------
# Beta
# -------------------------------------

axes[1].hist(
    beta_samples,
    bins=20,
    density=True,
    alpha=0.7
)

axes[1].axvline(
    np.mean(beta_samples),
    color="red",
    linestyle="--"
)

axes[1].set_title(
    "Posterior of Beta"
)

axes[1].set_xlabel(
    r"$\beta$"
)

# -------------------------------------
# Sigma Eta
# -------------------------------------

axes[2].hist(
    sigma_eta_samples,
    bins=20,
    density=True,
    alpha=0.7
)

axes[2].axvline(
    np.mean(sigma_eta_samples),
    color="red",
    linestyle="--"
)

axes[2].set_title(
    r"Posterior of $\sigma_\eta$"
)

axes[2].set_xlabel(
    r"$\sigma_\eta$"
)

plt.suptitle(
    "ABC Manual Posterior Distributions for SV Model",
    fontsize=14
)

plt.tight_layout()

plt.show()
