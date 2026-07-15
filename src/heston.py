import numpy as np
import QuantLib as ql

def heston_price(
        spot,
        strike,
        maturity,
        rate,
        dividend,
        params,
        today,
        day_count,
    ):

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


def heston_price_vector(
    params,
    spot,
    strike,
    maturity,
    rate,
    dividend,
    today,
    day_count,
):
    prices = np.zeros(len(strike))

    for i in range(len(strike)):
        prices[i] = heston_price(
        spot[i],
        strike[i],
        maturity[i],
        rate[i],
        dividend[i],
        params,
        today,
        day_count,
    )
    return prices

