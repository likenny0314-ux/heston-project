import numpy as np
from scipy.optimize import least_squares

from .heston import heston_price_vector, MarketPrice
def objective(params):

    model_prices = heston_price_vector(params)

    return model_prices - MarketPrice

initial_guess = np.array([
    0.04,   # v0
    0.04,   # theta
    -0.7,   # rho
    1.5,    # kappa
    0.3     # sigma
])

lower_bounds = [
    1e-6,
    1e-6,
    -0.999,
    1e-6,
    1e-6
]

upper_bounds = [
    5.0,
    5.0,
    0.999,
    20.0,
    5.0
]



result = least_squares(
    objective,
    initial_guess,
    bounds=(lower_bounds, upper_bounds),
    verbose=2,
    max_nfev=50
)

