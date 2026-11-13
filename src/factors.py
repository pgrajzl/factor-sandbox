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


def standardize(X, clip=3.0, mask=None, weights=None):
    """Cross-sectional z-score each day (equal-weighted, clipped), re-standardized.
    mask: boolean panel; stats computed only over masked-in stocks.
    weights: optional panel (e.g. market cap); after clipping, shifts exposures so the
    cap-weighted mean is zero (no re-clipping), making the intercept ~ the cap-weighted market."""
    X = X.replace([np.inf, -np.inf], np.nan)
    m = None if mask is None else mask.reindex(index=X.index, columns=X.columns, fill_value=False)
    sel = (lambda Z: Z) if m is None else (lambda Z: Z.where(m))

    z = X.sub(sel(X).mean(axis=1), axis=0).div(sel(X).std(axis=1), axis=0).clip(-clip, clip)
    z = z.sub(sel(z).mean(axis=1), axis=0).div(sel(z).std(axis=1), axis=0)

    if weights is not None:
        w = weights.reindex(index=X.index, columns=X.columns).where(sel(z).notna())
        z = z.sub((z * w).sum(axis=1) / w.sum(axis=1), axis=0)
    return z


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

def cs_regression_multi(returns, exposures, min_obs=50, intercept=True, return_resid=False, weights=None):
    """Daily cross-sectional regression: r_it = [a_t] + sum_k f_kt * x_ik,t-1 + e_it.
    exposures: dict {name: panel}. weights: optional panel (e.g. sqrt market cap) for WLS, lagged a day.
    Set intercept=False when sector dummies are included."""
    names = list(exposures)
    align = lambda p: p.shift(1).reindex(index=returns.index, columns=returns.columns).values
    R = returns.values
    X = np.stack([align(exposures[k]) for k in names], axis=-1)
    W = align(weights) if weights is not None else np.ones_like(R)
    E = np.full_like(R, np.nan)

    lead = ["intercept"] if intercept else []
    rows = []
    for t in range(len(R)):
        y, x, w = R[t], X[t], W[t]
        ok = ~np.isnan(y) & ~np.isnan(x).any(axis=1) & ~np.isnan(w) & (w > 0)
        n = ok.sum()
        if n < min_obs:
            rows.append([np.nan] * (len(lead) + len(names) + 2))
            continue
        y, x, w = y[ok], x[ok], w[ok]
        A = np.column_stack([np.ones(n), x]) if intercept else x
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        e = y - A @ coef
        E[t, ok] = e
        ybar = np.average(y, weights=w)
        r2 = 1 - np.sum(w * e**2) / np.sum(w * (y - ybar) ** 2)
        rows.append([*coef, r2, n])

    out = pd.DataFrame(rows, index=returns.index, columns=[*lead, *names, "r2", "n"])
    keep = out["r2"].notna()
    out = out[keep]
    if return_resid:
        return out, pd.DataFrame(E, index=returns.index, columns=returns.columns)[keep]
    return out

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

def dollar_volume(prices, volume, window=63):
    """Log of average daily dollar volume over the trailing window."""
    dv = (prices * volume.reindex_like(prices)).replace(0, np.nan)
    return np.log(dv.rolling(window, min_periods=int(0.8 * window)).mean())

def sector_exposures(universe, index, columns):
    """One 0/1 dummy panel per GICS sector, aligned to the returns panel."""
    sec = universe.set_index("ticker")["sector"].reindex(columns)
    return {
        s: pd.DataFrame(np.tile((sec == s).astype(float).values, (len(index), 1)),
                        index=index, columns=columns)
        for s in sorted(sec.dropna().unique())
    }

def attribute(ticker, returns, exposures, res, resid, groups=None):
    """Daily return attribution for one stock: lagged exposure x factor return per factor,
    plus residual. groups: {new_name: [factor columns]} to collapse, e.g. all sectors -> 'Sector'."""
    idx = res.index
    X = pd.DataFrame({k: v[ticker].shift(1).reindex(idx) for k, v in exposures.items()})
    contrib = X * res[X.columns]

    for name, cols in (groups or {}).items():
        contrib[name] = contrib[cols].sum(axis=1)
        contrib = contrib.drop(columns=cols)

    contrib["Idio"] = resid[ticker].reindex(idx)
    contrib["Total"] = returns[ticker].reindex(idx)
    return contrib

def risk_decomp(contrib, total="Total"):
    """Ex-post risk decomposition of a return into its additive components.
    Share_j = Cov(c_j, r) / Var(r); shares sum to 100%."""
    c = contrib.dropna()
    r, comps = c[total], c.drop(columns=total)
    out = pd.DataFrame({
        "Ann. vol (%)": comps.std() * np.sqrt(252) * 100,
        "Corr with total": comps.corrwith(r),
        "Share of variance (%)": comps.apply(lambda s: s.cov(r)) / r.var() * 100,
    })
    out.loc[total] = [r.std() * np.sqrt(252) * 100, 1.0, 100.0]
    return out

def cs_regression_market(returns, styles, sectors, min_obs=50, return_resid=False):
    """Daily cross-sectional OLS with an explicit market factor:
    r_it = f_mkt + sum_s D_is f_s + sum_k x_ik f_k + e_it,
    subject to sum_s w_s f_s = 0 (w_s = sector's share of stocks that day)."""
    snames, knames = list(sectors), list(styles)
    S, R = len(snames), returns.values
    lag = lambda p: p.shift(1).reindex(index=returns.index, columns=returns.columns).values
    D = np.stack([lag(sectors[s]) for s in snames], axis=-1)
    X = np.stack([lag(styles[k]) for k in knames], axis=-1)
    E = np.full_like(R, np.nan)

    rows = []
    for t in range(len(R)):
        ok = ~np.isnan(R[t]) & ~np.isnan(D[t]).any(axis=1) & ~np.isnan(X[t]).any(axis=1)
        n = ok.sum()
        if n < min_obs:
            rows.append([np.nan] * (1 + S + len(knames) + 2))
            continue
        y, d, x = R[t][ok], D[t][ok], X[t][ok]

        w = d.sum(axis=0) / n                                  # sector weights
        C = np.vstack([np.eye(S - 1), -w[:-1] / w[-1]])        # maps 10 free sectors -> all 11
        A = np.column_stack([np.ones(n), d @ C, x])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        e = y - A @ coef
        E[t, ok] = e

        rows.append([coef[0], *(C @ coef[1:S]), *coef[S:], 1 - e.var() / y.var(), n])

    out = pd.DataFrame(rows, index=returns.index, columns=["Market", *snames, *knames, "r2", "n"])
    keep = out["r2"].notna()
    out = out[keep]
    if return_resid:
        return out, pd.DataFrame(E, index=returns.index, columns=returns.columns)[keep]
    return out

def value(book, prices, shares):
    """Log book-to-price: log(book equity / market cap). Negative or zero book equity excluded."""
    mcap = prices * shares.reindex_like(prices)
    bp = book.reindex_like(prices) / mcap
    return np.log(bp.where(bp > 0))

def size(prices, shares):
    """Log market cap: price x point-in-time shares outstanding."""
    mcap = prices * shares.reindex_like(prices)
    return np.log(mcap.where(mcap > 0))

def turnover(volume, shares, window=63):
    """Log average daily turnover: shares traded / shares outstanding."""
    daily = (volume / shares.reindex_like(volume)).replace(0, np.nan)
    return np.log(daily.rolling(window, min_periods=int(0.8 * window)).mean())