from scipy.io import loadmat
import numpy as np

import QuantLib as ql

data = loadmat("Options2011.mat")

K = data["K"].flatten()
PC = data["PC"].flatten()
Price = data["Price"].flatten()
S = data["S"].flatten()
T = data["T"].flatten()
dateO = data["dateO"].flatten()
r = data["r"].flatten()
q = data["q"].flatten()

indx = (dateO == np.min(dateO)) & (PC == 1)

K = K[indx]
S = S[indx]
T = T[indx]
r = r[indx]
q = q[indx]
MarketPrice = Price[indx]

calendar = ql.NullCalendar()

today = ql.Date.todaysDate()

ql.Settings.instance().evaluationDate = today

day_count = ql.Actual365Fixed()