import numpy as np

def heston_cf(u, params, S0, r, q, T):
    """
    Heston characteristic function
    """

    v0, theta, rho, kappa, sigma = params

    i = 1j

    d = np.sqrt(
        (rho * sigma * i * u - kappa)**2 +
        sigma**2 * (i*u + u**2)
    )

    g = (
        kappa - rho * sigma * i * u - d
    ) / (
        kappa - rho * sigma * i * u + d
    )

    # Avoid numerical issues
    exp_neg_dT = np.exp(-d * T)

    C = (
        (r - q) * i * u * T
        + (kappa * theta / sigma**2) *
        (
            (kappa - rho * sigma * i * u - d) * T
            - 2 * np.log((1 - g * exp_neg_dT) / (1 - g))
        )
    )

    D = (
        (kappa - rho * sigma * i * u - d) / sigma**2
    ) * (
        (1 - exp_neg_dT) / (1 - g * exp_neg_dT)
    )

    return np.exp(
        C + D * v0 + i * u * np.log(S0)
    )

from scipy.integrate import quad

def heston_call_price_cf(S0, K, T, r, q, params):
    """
    European call price via Heston CF
    """
    def integrand_P1(u):
        i = 1j
        phi_ui = heston_cf(u - i, params, S0, r, q, T)
        phi_minus_i = heston_cf(-i, params, S0, r, q, T)
        numerator = np.exp(-i * u * np.log(K)) * phi_ui
        denominator = i * u * phi_minus_i
        return np.real(numerator / denominator)

    def integrand_P2(u):
        i = 1j
        phi = heston_cf(u, params, S0, r, q, T)
        numerator = np.exp(-i * u * np.log(K)) * phi
        denominator = i * u
        return np.real(numerator / denominator)

    P1 = 0.5 + (1 / np.pi) * quad(integrand_P1, 0, 100)[0]
    P2 = 0.5 + (1 / np.pi) * quad(integrand_P2, 0, 100)[0]

    call_price = (
        S0 * np.exp(-q * T) * P1
        - K * np.exp(-r * T) * P2
    )

    return call_price

