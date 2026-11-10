import numpy as np
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


def ols(y, X):
    """Multivariate OLS with intercept. X is a DataFrame of factors."""
    df = pd.concat([y, X], axis=1, join="inner").dropna()
    yv = df.iloc[:, 0].values
    Xv = np.column_stack([np.ones(len(df)), df.iloc[:, 1:].values])
    coef, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    resid = yv - Xv @ coef
    return dict(
        alpha=coef[0],
        betas=pd.Series(coef[1:], index=df.columns[1:]),
        r2=1 - resid.var() / yv.var(),
        resid=pd.Series(resid, index=df.index),
        n=len(df),
    )


def sequential_orthogonalize(F):
    """Gram-Schmidt on factor columns, in column order: each factor is replaced
    by its residual after regressing on all factors before it."""
    out = F[[F.columns[0]]].copy()
    for col in F.columns[1:]:
        out[col] = ols(F[col], out)["resid"]
    return out