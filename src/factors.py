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

def momentum(prices, lookback=252, skip=21):
    """12-1 momentum: return from t-252 to t-21, skipping the most recent month."""
    return prices.shift(skip) / prices.shift(lookback) - 1


def standardize(X, clip=3.0):
    """Cross-sectional z-score each day, clip outliers, then re-standardize."""
    z = X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1), axis=0).clip(-clip, clip)
    return z.sub(z.mean(axis=1), axis=0).div(z.std(axis=1), axis=0)


def cs_regression(returns, exposure, min_obs=50):
    """Daily cross-sectional OLS: r_it = a_t + f_t * z_i,t-1 + e_it.
    Returns factor return, intercept, R² and stock count per day."""
    X = exposure.shift(1)                      # exposure known at prior close
    mask = X.notna() & returns.notna()
    X, R = X.where(mask), returns.where(mask)

    xd = X.sub(X.mean(axis=1), axis=0)
    rd = R.sub(R.mean(axis=1), axis=0)
    sxx = (xd**2).sum(axis=1)

    f = (xd * rd).sum(axis=1) / sxx
    out = pd.DataFrame({
        "factor_ret": f,
        "intercept": R.mean(axis=1) - f * X.mean(axis=1),
        "r2": f**2 * sxx / (rd**2).sum(axis=1),
        "n": mask.sum(axis=1),
    })
    return out[out["n"] >= min_obs]

def volatility(returns, window=63):
    """Trailing annualized realized vol of daily returns."""
    return returns.rolling(window, min_periods=int(0.8 * window)).std() * np.sqrt(252)

def cs_regression_multi(returns, exposures, min_obs=50):
    """Daily cross-sectional OLS with several exposures:
    r_it = a_t + sum_k f_kt * z_ik,t-1 + e_it.   exposures: dict {name: panel}."""
    names = list(exposures)
    R = returns.values
    X = np.stack([exposures[k].shift(1).reindex(index=returns.index, columns=returns.columns).values
                  for k in names], axis=-1)                       # (days, stocks, factors)

    rows = []
    for t in range(len(R)):
        y, x = R[t], X[t]
        ok = ~np.isnan(y) & ~np.isnan(x).any(axis=1)
        n = ok.sum()
        if n < min_obs:
            rows.append([np.nan] * (len(names) + 3))
            continue
        A = np.column_stack([np.ones(n), x[ok]])
        coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        e = y[ok] - A @ coef
        rows.append([coef[0], *coef[1:], 1 - e.var() / y[ok].var(), n])

    out = pd.DataFrame(rows, index=returns.index, columns=["intercept", *names, "r2", "n"])
    return out.dropna(subset=["r2"])

def reversal(prices, window=21):
    """Short-term reversal: negative of the trailing 1-month return (long losers, short winners)."""
    return -(prices / prices.shift(window) - 1)

def beta_exposure(returns, mkt_ret, window=252):
    """Rolling beta of each stock to the market: Cov(r_i, r_m) / Var(r_m) over the trailing window."""
    m = mkt_ret.reindex(returns.index)
    mp = int(0.8 * window)
    roll = lambda x: x.rolling(window, min_periods=mp).mean()

    cov = roll(returns.mul(m, axis=0)) - roll(returns).mul(roll(m), axis=0)
    var = roll(m**2) - roll(m)**2
    return cov.div(var, axis=0)

def factor_stats(fr):
    """Summary stats for a DataFrame of daily factor returns (decimals), one column per series."""
    fr = fr.dropna() * 100
    return pd.DataFrame({
        "Ann. return (%)": fr.mean() * 252,
        "Ann. vol (%)": fr.std() * np.sqrt(252),
        "Sharpe": fr.mean() / fr.std() * np.sqrt(252),
        "t-stat": fr.mean() / fr.std() * np.sqrt(len(fr)),
        "Max drawdown (%)": (fr.cumsum() - fr.cumsum().cummax()).min(),
        "Days": len(fr),
    })