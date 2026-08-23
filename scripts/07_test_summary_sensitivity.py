import numpy as np
from scipy.stats import kurtosis
from statsmodels.tsa.stattools import acf

# =====================================
# Summary Function
# =====================================

def summary_statistics(y):

    variance = np.var(y)

    kurt = kurtosis(
        y,
        fisher=False
    )

    sq = y**2

    acf_values = acf(
        sq,
        nlags=5,
        fft=False
    )

    return np.array([
        variance,
        kurt,
        acf_values[1],
        acf_values[5]
    ])


# =====================================
# SV Simulator
# =====================================

def simulate_sv(
        alpha,
        beta,
        sigma_eta,
        T=4000):

    log_sigma2 = np.zeros(T)

    log_sigma2[0] = (
        alpha/(1-beta)
    )

    for t in range(1,T):

        eta = np.random.normal(
            0,
            sigma_eta
        )

        log_sigma2[t] = (
            alpha
            + beta*log_sigma2[t-1]
            + eta
        )

    sigma = np.exp(
        log_sigma2/2
    )

    eps = np.random.normal(
        0,
        1,
        T
    )

    y = sigma * eps

    return y


# =====================================
# Model A
# =====================================

y_A = simulate_sv(
    alpha=-0.5,
    beta=0.90,
    sigma_eta=0.20
)

S_A = summary_statistics(y_A)

# =====================================
# Model B
# =====================================

y_B = simulate_sv(
    alpha=-0.5,
    beta=0.98,
    sigma_eta=0.35
)

S_B = summary_statistics(y_B)

print("\nModel A")
print(S_A)

print("\nModel B")
print(S_B)