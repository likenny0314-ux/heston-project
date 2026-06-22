import numpy as np

from src.heston import heston_price
from src.heston_cf import heston_call_price_cf
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
price_cf = heston_call_price_cf(S0, K, T, r, q, params)


ql_price = heston_price(
    S0, K, T, r, q, params
)

print("QuantLib:", ql_price)
print("CF price:", price_cf)
print("Abs diff:", abs(ql_price - price_cf))