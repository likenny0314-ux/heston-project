import numpy as np
import QuantLib as ql

from .data_loader import K, Price, PC, S, T
from .data_loader import dateO, r, q, MarketPrice 
from .data_loader import calendar, today, day_count

def compute_implied_vol(
        option_price,
        spot,
        strike,
        maturity,
        rate,
        dividend,
        option_type=ql.Option.Call):

    try:

        payoff = ql.PlainVanillaPayoff(
            option_type,
            float(strike)
        )

        expiry = today + int(maturity * 365)

        exercise = ql.EuropeanExercise(expiry)

        option = ql.VanillaOption(
            payoff,
            exercise
        )

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

        vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(
                today,
                calendar,
                0.20,
                day_count
            )
        )

        process = ql.BlackScholesMertonProcess(
            spot_handle,
            dividend_ts,
            risk_free_ts,
            vol_ts
        )

        engine = ql.AnalyticEuropeanEngine(process)

        option.setPricingEngine(engine)

        iv = option.impliedVolatility(
            option_price,
            process
        )

        return iv

    except:
        return np.nan
