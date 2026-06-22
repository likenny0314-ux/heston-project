# heston-project
This project implements the Heston stochastic volatility model for option pricing and calibration using both:  QuantLib (reference implementation) A custom implementation based on the characteristic function and numerical integration.

The Heston model assumes stochastic variance:

[
dv_t = \kappa(\theta - v_t)dt + \sigma \sqrt{v_t} dW_t^v
]

[
dS_t = S_t \left((r - q)dt + \sqrt{v_t} dW_t^S \right)
]

with correlation:

[
dW_t^S \cdot dW_t^v = \rho dt
]

Parameters
v0 — initial variance
theta — long-run variance
kappa — mean reversion speed
sigma — volatility of variance
rho — correlation

Implementations
1. QuantLib Engine

Uses AnalyticHestonEngine for pricing and calibration.

2. Custom Characteristic Function Engine

A manual implementation based on:

Heston characteristic function
Fourier inversion / numerical integration

Used to compute European call prices:

price_cf = heston_call_price_cf(S0, K, T, r, q, params)

How to Run
Calibration + Smile Plot
python main.py
Validate CF Implementation
python test_cf.py

Calibration Method
Objective: minimize pricing error

[
\text{error} = \text{model price} - \text{market price}
]

Solver: scipy.optimize.least_squares
Constraints: parameter bounds enforced

Volatility Smile Visualization
X-axis: Moneyness (K / S)
Y-axis: Implied Volatility
Comparison:
Market data (circles)
Heston model (crosses)