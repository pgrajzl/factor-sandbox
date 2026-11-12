from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

data_dir = Path(__file__).resolve().parents[1] / "data"


def get_sp500_constituents(save=True):
    """Scrape current S&P 500 tickers and GICS sectors from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
    df = pd.read_html(StringIO(html))[0]

    df = df[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]]
    df.columns = ["ticker", "name", "sector", "sub_industry"]
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B for yfinance

    if save:
        df.to_csv(data_dir / "constituents.csv", index=False)
    return df


def get_prices(tickers, years=5, save=True):
    """Download daily adjusted close prices."""
    start = pd.Timestamp.today() - pd.DateOffset(years=years)
    prices = yf.download(tickers, start=start, auto_adjust=True, progress=True)["Close"]
    prices = prices.dropna(axis=1, how="all")  # drop tickers that failed entirely

    if save:
        prices.to_parquet(data_dir / "prices" / "prices.parquet")
    return prices


def compute_returns(prices, save=True):
    """Daily simple returns from prices."""
    returns = prices.pct_change(fill_method=None).iloc[1:]

    if save:
        returns.to_parquet(data_dir / "returns" / "returns.parquet")
    return returns


def load_prices():
    return pd.read_parquet(data_dir / "prices" / "prices.parquet")


def load_returns():
    return pd.read_parquet(data_dir / "returns" / "returns.parquet")


def load_constituents():
    return pd.read_csv(data_dir / "constituents.csv")

def get_market(ticker="SPY", years=5, save=True):
    """Download benchmark prices and returns, saved separately from the stock panel."""
    start = pd.Timestamp.today() - pd.DateOffset(years=years)
    px = yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"]
    px = px.squeeze().rename(ticker)  # make sure it's a Series, not a 1-column DataFrame
    ret = px.pct_change(fill_method=None).iloc[1:]

    if save:
        px.to_frame().to_parquet(data_dir / "prices" / f"{ticker.lower()}.parquet")
        ret.to_frame().to_parquet(data_dir / "returns" / f"{ticker.lower()}_returns.parquet")
    return px, ret


def load_market(ticker="SPY"):
    path = data_dir / "returns" / f"{ticker.lower()}_returns.parquet"
    return pd.read_parquet(path).squeeze()

def get_volume(tickers, years=5, save=True):
    """Download daily share volume."""
    start = pd.Timestamp.today() - pd.DateOffset(years=years)
    vol = yf.download(tickers, start=start, auto_adjust=True, progress=True)["Volume"]
    vol = vol.dropna(axis=1, how="all")

    if save:
        vol.to_parquet(data_dir / "prices" / "volume.parquet")
    return vol


def load_volume():
    return pd.read_parquet(data_dir / "prices" / "volume.parquet")