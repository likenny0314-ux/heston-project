import numpy as np
from scipy.stats import norm

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

S0 = 100.0     # spot
K = 100.0      # ATM strike
T = 1.0        # 1 year
r = 0.02       # 2% rate
q = 0.0        # no dividends

params = np.array([
    0.04,   # v0   (20% vol^2)
    0.04,   # theta
    -0.7,   # rho
    1.5,    # kappa
    0.3     # sigma
])

bs_price = bs_call(S0, K, T, r, np.sqrt(params[0]))

print("BS price:", bs_price)