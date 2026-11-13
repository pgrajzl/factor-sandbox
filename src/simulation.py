import numpy as np
import pandas as pd

from . import plotting


def simulate_random_returns(n_stocks=1000, years=5, mu_range=(0.0, 0.15), vol_range=(0.15, 0.60),
                            start_price=100.0, seed=0):
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