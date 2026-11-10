import pandas as pd


def regress(stock_ret, mkt_ret):
    """OLS of stock on market: r_s = alpha + beta * r_m + eps. Uses overlapping dates."""
    df = pd.concat([stock_ret, mkt_ret], axis=1, join="inner").dropna()
    y, x = df.iloc[:, 0], df.iloc[:, 1]
    b = y.cov(x) / x.var()
    a = y.mean() - b * x.mean()
    rho = y.corr(x)
    return dict(alpha=a, beta=b, r2=rho**2, corr=rho, n=len(df))


def beta(stock_ret, mkt_ret):
    return regress(stock_ret, mkt_ret)["beta"]