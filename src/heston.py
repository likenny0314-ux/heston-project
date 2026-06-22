import numpy as np
import QuantLib as ql

from .data_loader import K, Price, PC, S, T
from .data_loader import dateO, r, q, MarketPrice 
from .data_loader import calendar, today, day_count

from .Implied_vol import compute_implied_vol

def heston_price(
        spot,
        strike,
        maturity,
        rate,
        dividend,
        params):

    v0, theta, rho, kappa, sigma = [
        float(x) for x in params
    ]

    spot = float(spot)
    strike = float(strike)
    maturity = float(maturity)
    rate = float(rate)
    dividend = float(dividend)

    spot_handle = ql.QuoteHandle(
        ql.SimpleQuote(spot)
    )

    risk_free_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(
            today,
            rate,
            day_count
        )
    )

    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(
            today,
            dividend,
            day_count
        )
    )

    heston_process = ql.HestonProcess(
        risk_free_ts,
        dividend_ts,
        spot_handle,
        v0,
        kappa,
        theta,
        sigma,
        rho
    )

    heston_model = ql.HestonModel(
        heston_process
    )

    engine = ql.AnalyticHestonEngine(
        heston_model
    )

    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call,
        strike
    )

    expiry = today + int(maturity * 365)

    exercise = ql.EuropeanExercise(expiry)

    option = ql.VanillaOption(
        payoff,
        exercise
    )

    option.setPricingEngine(engine)

    return option.NPV()



def heston_price_vector(params):

    prices = np.zeros(len(K))

    for i in range(len(K)):

        prices[i] = heston_price(
            S[i],
            K[i],
            T[i],
            r[i],
            q[i],
            params
        )

    return prices

