import numpy as np
import pandas as pd

from . import plotting
from pathlib import Path


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

def plot_simulated_stock(prices, params, ticker, default="1y"):
    """Price path of one simulated stock. Caption shows the true parameters; subtitle the realized ones."""
    mu, vol = params.loc[ticker, "mu"], params.loc[ticker, "vol"]

    def subtitle(p):
        r = plotting._slice(prices[ticker], p).pct_change().dropna()
        return (f"Realized over window: return {r.mean() * 252:.1%} / yr, "
                f"vol {r.std() * np.sqrt(252):.1%} / yr")

    return plotting._period_figure(
        lambda p: [plotting._line(plotting._slice(prices[ticker], p), ticker)],
        title=f"{ticker} (simulated)", ylabel="Price", default=default, subtitle=subtitle,
        caption=f"Simulated stock (geometric Brownian motion). True parameters: "
                f"drift μ = {mu:.1%} / yr, volatility σ = {vol:.1%} / yr.",
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