import numpy as np
import pandas as pd

from . import plotting, factors
from pathlib import Path

import textwrap


def simulate_random_returns(n_stocks=1000, years=5, mu_range=(-0.10, 0.20), vol_range=(0.15, 0.60),
                            start_price=100.0, seed=44):
    """Independent geometric Brownian motions: each stock gets its own annual drift and vol.
    Returns (prices, returns, params) with a business-day index ending today."""
    rng = np.random.default_rng(seed)
    n_days = int(years * 252)
    dt = 1 / 252

    tickers = [f"S{i:04d}" for i in range(1, n_stocks + 1)]
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    mu = rng.uniform(*mu_range, n_stocks)
    vol = rng.uniform(*vol_range, n_stocks)

    log_ret = rng.normal((mu - 0.5 * vol**2) * dt, vol * np.sqrt(dt), size=(n_days, n_stocks))
    returns = pd.DataFrame(np.expm1(log_ret), index=dates, columns=tickers)
    prices = start_price * np.exp(pd.DataFrame(log_ret, index=dates, columns=tickers).cumsum())

    params = pd.DataFrame({"mu": mu, "vol": vol}, index=tickers)
    return prices, returns, params

def plot_simulated_stock(prices, ticker, params=None, idio_vol=None, default="1y"):
    """Price path of one simulated stock. Caption shows the true parameters for whichever world it came from:
    params (random-walk world: drift and vol) or idio_vol (structural world)."""
    def subtitle(p):
        r = plotting._slice(prices[ticker], p).pct_change().dropna()
        return (f"Realized over window: return {r.mean() * 252:.1%} per year, "
                f"vol {r.std() * np.sqrt(252):.1%} per year")

    if params is not None:
        caption = (f"Simulated stock (geometric Brownian motion). True parameters: "
                   f"drift μ = {params.loc[ticker, 'mu']:.1%} per year, "
                   f"volatility σ = {params.loc[ticker, 'vol']:.1%} per year.")
    elif idio_vol is not None:
        caption = (f"Simulated stock (structural factor world): returns driven by market, size, value and "
                   f"momentum, plus idio. True idio vol = {idio_vol[ticker]:.1%} per year.")
    else:
        caption = "Simulated stock."

    return plotting._period_figure(
        lambda p: [plotting._line(plotting._slice(prices[ticker], p), ticker)],
        title=f"{ticker} (simulated)", ylabel="Price", default=default, subtitle=subtitle, caption=caption,
    )

def simulate_market_caps(prices, total=100.0, skew=1.75, seed=1):
    """Assign each stock a random initial share of a market worth `total`, right-skewed (lognormal).
    Caps then move with each stock's price (fixed share counts).
    Returns (caps panel, market index level, initial weights)."""
    rng = np.random.default_rng(seed)
    raw = rng.lognormal(mean=0.0, sigma=skew, size=prices.shape[1])
    w0 = pd.Series(raw / raw.sum(), index=prices.columns)

    caps = (total * w0) * prices.div(prices.iloc[0])
    market = caps.sum(axis=1).rename("Market")
    return caps, market, w0


def plot_simulated_market(market, w0, default="5y"):
    """Cap-weighted simulated market index, with concentration stats in the caption."""
    top10 = w0.nlargest(10).sum()
    top1pct = w0.nlargest(max(1, len(w0) // 100)).sum()

    def subtitle(p):
        r = plotting._slice(market, p).pct_change().dropna()
        return f"Realized over window: return {r.mean() * 252:.1%} / yr, vol {r.std() * np.sqrt(252):.1%} / yr"

    return plotting._period_figure(
        lambda p: [plotting._line(plotting._slice(market, p), "Market")],
        title="Simulated market (cap-weighted)", ylabel="Index level", default=default, subtitle=subtitle,
        caption=f"Simulated: {len(w0):,} independent stocks, market starts at 100. Initial concentration: "
                f"top 10 stocks {top10:.0%}, top 1% {top1pct:.0%} of market cap.",
    )

sim_dir = Path(__file__).resolve().parents[1] / "data" / "sim"


def save_sim(name="baseline", **objs):
    """Save simulated DataFrames/Series as parquet under data/sim/<name>/."""
    d = sim_dir / name
    d.mkdir(parents=True, exist_ok=True)
    for key, obj in objs.items():
        (obj.to_frame(name=key) if isinstance(obj, pd.Series) else obj).to_parquet(d / f"{key}.parquet")


def load_sim(name="baseline"):
    """Load everything saved under data/sim/<name>/ into a dict. Single-column files come back as Series."""
    out = {}
    for p in sorted((sim_dir / name).glob("*.parquet")):
        df = pd.read_parquet(p)
        out[p.stem] = df.iloc[:, 0] if df.shape[1] == 1 and df.columns[0] == p.stem else df
    return out

def simulate_book_equity(caps, bp_median=0.4, bp_log_sd=0.8, size_corr=-0.3,
                         growth_mean=0.08, growth_sd=0.06, quarter_noise=0.03,
                         report_every=63, lag_range=(20, 45), seed=2):
    """Fake but realistic point-in-time book equity.
    Starting B/P lognormal around bp_median, correlated with log size via size_corr.
    Book grows quarterly at a stock-specific rate plus noise, and is only *reported*
    after a stock-specific lag, on staggered fiscal calendars. Returns a days x stocks panel
    of reported book, in the same units as caps."""
    rng = np.random.default_rng(seed)
    n_days, n = caps.shape

    # starting B/P, correlated with starting size
    log_cap0 = np.log(caps.iloc[0].values)
    z_cap = (log_cap0 - log_cap0.mean()) / log_cap0.std()
    z_bp = size_corr * z_cap + np.sqrt(1 - size_corr**2) * rng.standard_normal(n)
    book0 = bp_median * np.exp(bp_log_sd * z_bp) * caps.iloc[0].values

    # true book path, one value per fiscal quarter
    n_q = n_days // report_every + 2
    growth = rng.normal(growth_mean, growth_sd, n)
    steps = growth / 4 + quarter_noise * rng.standard_normal((n_q, n))
    steps[0] = 0.0
    book_path = book0 * np.exp(np.cumsum(steps, axis=0))          # (quarters, stocks)

    # reported book: quarter q becomes known on day q*report_every + offset_i + lag_i
    offset = rng.integers(0, report_every, n)
    lag = rng.integers(lag_range[0], lag_range[1] + 1, n)
    t = np.arange(n_days)[:, None]
    q_known = np.clip((t - offset - lag) // report_every, 0, n_q - 1)
    reported = book_path[q_known, np.arange(n)]

    return pd.DataFrame(reported, index=caps.index, columns=caps.columns)


def plot_simulated_value(caps, book, ticker, default="1y"):
    """Book-to-price for one simulated stock: moves daily with price, steps when book is reported."""
    bp = (book[ticker] / caps[ticker]).rename("B/P")

    return plotting._period_figure(
        lambda p: [plotting._line(plotting._slice(bp, p), "B/P")],
        title=f"{ticker} book-to-price (simulated)", ylabel="Book / market cap", default=default,
        subtitle=lambda p: f"Book reported {plotting._slice(book[ticker], p).diff().ne(0).sum()} times in window",
        caption="Simulated: B/P moves daily with price and jumps when new (lagged) book equity is reported.",
    )

def _zscore(x, w, clip=3.0):
    """Same recipe as factors.standardize for one day: equal-weighted z, clip,
    re-standardize, then shift so the cap-weighted mean is zero."""
    ok = np.isfinite(x)
    z = np.full_like(x, np.nan)
    z[ok] = np.clip((x[ok] - x[ok].mean()) / x[ok].std(ddof=1), -clip, clip)
    z[ok] = (z[ok] - z[ok].mean()) / z[ok].std(ddof=1)
    z[ok] -= np.sum(w[ok] * z[ok]) / np.sum(w[ok])
    return z


def simulate_structural_market(n_stocks=1000, years=5, burn_in=253, seed=0,
                               factor_mu=None, factor_vol=None, factor_corr=None,
                               idio_vol_range=(0.15, 0.45), cap_skew=1.75, book_kwargs=None):
    """Factor-driven market where exposures are computed from each stock's own history every day:
    r_it = f_mkt,t + sum_k z_ik,t-1 * f_k,t + e_it, for k in size, value, momentum.
    The first `burn_in` days run market + idio only. Returns a dict of panels and true parameters."""
    rng = np.random.default_rng(seed)
    T, N, dt = int(years * 252), n_stocks, 1 / 252
    styles = ["Size", "Value", "Momentum"]
    names = ["Market", *styles]

    factor_mu = factor_mu or {"Market": 0.07, "Size": 0.00, "Value": 0.02, "Momentum": 0.04}
    factor_vol = factor_vol or {"Market": 0.16, "Size": 0.03, "Value": 0.04, "Momentum": 0.06}
    if factor_corr is None:
        factor_corr = pd.DataFrame(np.eye(4), index=names, columns=names)
        factor_corr.loc["Value", "Momentum"] = factor_corr.loc["Momentum", "Value"] = -0.3

    tickers = [f"S{i:04d}" for i in range(1, N + 1)]
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=T)

    # factor returns and idio, drawn up front (they don't depend on exposures)
    mu_d = np.array([factor_mu[k] for k in names]) * dt
    vol_d = np.array([factor_vol[k] for k in names]) * np.sqrt(dt)
    cov_d = np.outer(vol_d, vol_d) * factor_corr.loc[names, names].values
    F = rng.multivariate_normal(mu_d, cov_d, size=T)
    idio_vol = rng.uniform(*idio_vol_range, N)
    E = rng.standard_normal((T, N)) * idio_vol * np.sqrt(dt)

    # starting caps (market worth 100) and reported book equity
    raw = rng.lognormal(0.0, cap_skew, N)
    cap0 = 100.0 * raw / raw.sum()
    caps_start = pd.DataFrame(np.tile(cap0, (T, 1)), index=dates, columns=tickers)
    book = simulate_book_equity(caps_start, **(book_kwargs or {})).values

    P = np.empty((T, N))
    R = np.zeros((T, N))
    Z = {k: np.full((T, N), np.nan) for k in styles}
    P[0] = 100.0

    for t in range(1, T):
        s = t - 1                                             # yesterday: exposures known at its close
        cap_s = cap0 * P[s] / P[0]
        r = F[t, 0] + E[t]                                    # market + idio
        if s >= 252:
            x = {"Size": np.log(cap_s),
                 "Value": np.log(book[s] / cap_s),
                 "Momentum": P[s - 21] / P[s - 252] - 1}
            for j, k in enumerate(styles, start=1):
                Z[k][s] = _zscore(x[k], cap_s)
                r = r + Z[k][s] * F[t, j]
        R[t] = np.maximum(r, -0.95)                           # a stock can't lose more than everything
        P[t] = P[s] * (1 + R[t])

    prices = pd.DataFrame(P, index=dates, columns=tickers)
    caps = pd.DataFrame(cap0 * P / P[0], index=dates, columns=tickers)
    return dict(
        prices=prices,
        returns=pd.DataFrame(R, index=dates, columns=tickers).iloc[1:],
        caps=caps,
        market=caps.sum(axis=1).rename("Market"),
        book=pd.DataFrame(book, index=dates, columns=tickers),
        exposures={k: pd.DataFrame(Z[k], index=dates, columns=tickers) for k in styles},
        factor_returns=pd.DataFrame(F, index=dates, columns=names),
        idio=pd.DataFrame(E, index=dates, columns=tickers),
        idio_vol=pd.Series(idio_vol, index=tickers),
        factor_cov_daily=pd.DataFrame(cov_d, index=names, columns=names),
    )

def compare_decomposition(sw, res, resid, ticker):
    """Print estimated vs true return and risk decomposition for one stock in a structural world,
    and return the attribution plot."""
    returns, expo, F = sw["returns"], sw["exposures"], sw["factor_returns"]

    ones = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
    est = factors.attribute(ticker, returns, {"intercept": ones, **expo}, res, resid)
    est = est.rename(columns={"intercept": "Market"}).dropna()
    idx = est.index

    Ft = F.loc[idx]
    true_c = pd.DataFrame({
        "Market": Ft["Market"],
        **{k: expo[k][ticker].shift(1).reindex(idx) * Ft[k] for k in expo},
        "Idio": sw["idio"][ticker].reindex(idx),
    })
    true_c["Total"] = returns[ticker].reindex(idx)

    print(f"{ticker}: return decomposition, structural world ({idx.min():%b %Y} to {idx.max():%b %Y})\n")
    err = (est.drop(columns="Total").sum(axis=1) - est["Total"]).abs().max()
    print(f"Exactness check (estimated pieces sum to total): max error {err:.1e}\n")

    print("Latest exposures (z-scores):")
    print(pd.Series({k: expo[k][ticker].dropna().iloc[-1] for k in expo}).round(2).to_string(), "\n")

    print("Cumulative contribution (%), estimated vs true:")
    print(pd.DataFrame({"Estimated": est.sum() * 100, "True": true_c.sum() * 100}).round(1).to_string(), "\n")

    print("Share of variance (%), estimated vs true:")
    print(pd.DataFrame({"Estimated": factors.risk_decomp(est)["Share of variance (%)"],
                        "True": factors.risk_decomp(true_c)["Share of variance (%)"]}).round(1).to_string(), "\n")

    print(f"True idio vol for {ticker}: {sw['idio_vol'][ticker]:.1%} per year   "
          f"Estimated idio vol: {est['Idio'].std() * np.sqrt(252):.1%} per year")

    return plotting.plot_attribution(est, ticker)

def position_stats(sw, res, resid, ticker, return_table=False):
    """Hold one stock as a position: total, factor and idio return, vol, Sharpe and t-stat,
    estimated vs true idio. Renders a clean table; optionally also returns the DataFrame."""
    returns, expo = sw["returns"], sw["exposures"]
    ones = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
    c = factors.attribute(ticker, returns, {"intercept": ones, **expo}, res, resid).dropna()
    c = c.rename(columns={"intercept": "Market"})
    years = len(c) / 252

    def stats(x):
        ann_ret, ann_vol = x.mean() * 252, x.std() * np.sqrt(252)
        sr = ann_ret / ann_vol
        return pd.Series({"Ann. return (%)": ann_ret * 100, "Ann. vol (%)": ann_vol * 100,
                          "Sharpe": sr, "t-stat": sr * np.sqrt(years)})

    table = pd.DataFrame({
        "Total (unhedged)": stats(c["Total"]),
        "Factor PnL": stats(c["Total"] - c["Idio"]),
        "Idio (estimated)": stats(c["Idio"]),
        "Idio (true)": stats(sw["idio"][ticker].reindex(c.index)),
    }).T

    fig = plotting.plot_table(
        table, title=f"{ticker}: position stats",
        subtitle=f"{c.index.min():%b %Y} to {c.index.max():%b %Y} ({years:.1f} years)",
        index_name="Stream",
        caption=f"Idio (estimated) is what the risk model credits; Idio (true) is the planted stock-specific stream. "
                f"Fully factor-hedged Sharpe = idio Sharpe. Sharpe standard error over {years:.1f} years: ~{1 / np.sqrt(years):.2f}.",
    )
    return (fig, table) if return_table else fig