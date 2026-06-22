import matplotlib.pyplot as plt
import numpy as np

from src.calibration import result
from src.heston import heston_price_vector
from src.Implied_vol import compute_implied_vol

from src.data_loader import K, Price, PC, S, T
from src.data_loader import dateO, r, q, MarketPrice 
from src.data_loader import calendar, today, day_count

ModelPrices = heston_price_vector(result.x)

MarketIV = np.array([
    compute_implied_vol(
        MarketPrice[i],
        S[i],
        K[i],
        T[i],
        r[i],
        q[i]
    )
    for i in range(len(K))
])

ModelIV = np.array([
    compute_implied_vol(
        ModelPrices[i],
        S[i],
        K[i],
        T[i],
        r[i],
        q[i]
    )
    for i in range(len(K))
])

rmse = np.sqrt(
    np.mean(
        (ModelPrices - MarketPrice) ** 2
    )
)

print("\nRMSE =", rmse)

different_T = np.sort(np.unique(T))

nrows = int(np.ceil(len(different_T) / 2))

fig, axes = plt.subplots(
    nrows,
    2,
    figsize=(14, 10)
)

axes = axes.flatten()

for i, maturity in enumerate(different_T):

    idx = (T == maturity)

    ax = axes[i]

    ax.plot(
        K[idx] / S[idx],
        MarketIV[idx],
        'o',
        label='Market'
    )

    ax.plot(
        K[idx] / S[idx],
        ModelIV[idx],
        'rx',
        label='Heston'
    )

    ax.set_title(
        f'{round(maturity * 365)} days'
    )

    ax.set_xlabel('Moneyness K/S')
    ax.set_ylabel('Implied Volatility')

    ax.grid(True)

    ax.legend()

plt.tight_layout()
plt.show()


