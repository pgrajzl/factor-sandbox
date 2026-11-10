import pandas as pd


def beta(stock_ret, mkt_ret):
    """OLS beta of stock on market: Cov(r_s, r_m) / Var(r_m), over overlapping dates."""
    df = pd.concat([stock_ret, mkt_ret], axis=1, join="inner").dropna()
    s, m = df.iloc[:, 0], df.iloc[:, 1]
    return s.cov(m) / m.var()