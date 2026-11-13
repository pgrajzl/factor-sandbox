from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

import json
import time

import numpy as np

import io
import zipfile

root_dir = Path(__file__).resolve().parents[1] / "data"
data_dir = root_dir                                         # sp500 by default
fund_dir = data_dir / "fundamentals"
facts_dir = root_dir / "fundamentals" / "companyfacts"      # EDGAR cache, shared by all universes

sec_headers = {"User-Agent": "Paul Grajzl paulgrajzl@gmail.com"}


def set_universe(name):
    """Point every load/save at a universe folder: 'sp500' (default) or e.g. 'r3000'."""
    global data_dir, fund_dir
    data_dir = root_dir if name == "sp500" else root_dir / name
    fund_dir = data_dir / "fundamentals"
    for sub in ["prices", "returns", "fundamentals"]:
        (data_dir / sub).mkdir(parents=True, exist_ok=True)


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


def get_cik_map(save=True):
    """Ticker -> CIK (SEC company ID, zero-padded to 10 digits)."""
    raw = requests.get("https://www.sec.gov/files/company_tickers.json", headers=sec_headers).json()
    df = pd.DataFrame(raw.values())
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df["cik"] = df["cik_str"].astype(int).map(lambda x: f"{x:010d}")
    df = df[["ticker", "cik", "title"]].drop_duplicates("ticker")

    if save:
        fund_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(fund_dir / "cik_map.csv", index=False)
    return df


def get_company_facts(cik, refresh=False):
    """Full XBRL history for one company. Cached locally after the first download."""
    path = facts_dir / f"CIK{cik}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=sec_headers)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(r.text)
    time.sleep(0.12)                     # stay under the SEC's ~10 requests/second
    return r.json()


def download_all_facts(tickers, refresh=False):
    """Download and cache companyfacts for every ticker. Returns tickers that failed."""
    cik = get_cik_map().set_index("ticker")["cik"]
    missing = [t for t in tickers if t not in cik.index]
    failed = []

    todo = [t for t in tickers if t in cik.index]
    for i, t in enumerate(todo, 1):
        try:
            get_company_facts(cik[t], refresh=refresh)
        except Exception as e:
            failed.append((t, str(e)))
        if i % 50 == 0:
            print(f"{i}/{len(todo)}")
    return missing, failed


def facts_to_frame(facts, tag, taxonomy="us-gaap", unit="USD"):
    """One line item from a companyfacts JSON as a tidy DataFrame."""
    entries = facts["facts"][taxonomy][tag]["units"][unit]
    df = pd.DataFrame(entries)
    for c in ["end", "filed", "start"]:
        if c in df:
            df[c] = pd.to_datetime(df[c])
    return df.sort_values(["end", "filed"])

def pit_instant(facts, tags, taxonomy="us-gaap", unit="USD", sum_classes=False):
    """Point-in-time series for a balance-sheet (instant) item.
    First-reported value per period end, indexed by filing date. tags: fallbacks in priority order."""
    frames = []
    for prio, tag in enumerate(tags):
        try:
            df = facts_to_frame(facts, tag, taxonomy, unit)
        except KeyError:
            continue
        frames.append(df.assign(prio=prio))
    if not frames:
        return pd.Series(dtype=float)

    df = pd.concat(frames)
    df = df[df["form"].str.startswith(("10-Q", "10-K"))]
    if sum_classes:
        df = df.groupby(["accn", "end", "filed", "prio"], as_index=False)["val"].sum()

    df = df.sort_values(["end", "prio", "filed"]).drop_duplicates("end", keep="first")  # first-reported
    df = df.sort_values(["filed", "end"])
    df = df[df["end"] >= df["end"].cummax()]                                            # never step back
    return df.groupby("filed")["val"].last()

def build_fundamentals(tickers, items, index, max_age_days=400, save=True):
    """Daily point-in-time panels for several balance-sheet items.
    items: {name: dict(tags=[...], taxonomy=..., unit=..., sum_classes=...)}."""
    cik = pd.read_csv(fund_dir / "cik_map.csv", dtype={"cik": str}).set_index("ticker")["cik"]
    raw = {name: {} for name in items}

    for i, t in enumerate(tickers, 1):
        try:
            facts = get_company_facts(cik[t])
        except Exception:
            facts = None                        # no XBRL data at the SEC (404) or unknown ticker
        for name, spec in items.items():
            raw[name][t] = pit_instant(facts, **spec) if facts else pd.Series(dtype=float)
        if i % 100 == 0:
            print(f"{i}/{len(tickers)}")

    panels = {}
    for name, series in raw.items():
        s = pd.DataFrame(series).reindex(columns=tickers)
        s.index = pd.to_datetime(s.index)
        full = s.index.union(index)

        # filing date of each value, as a proper datetime panel (NaT where no value)
        filed = pd.DataFrame(
            np.where(s.notna().values, s.index.values[:, None], np.datetime64("NaT")),
            index=s.index, columns=s.columns,
        ).astype("datetime64[ns]")

        vals = s.reindex(full).ffill()
        filed = filed.reindex(full).ffill()
        age = full.values[:, None] - filed.values             # days since filing (NaT if none)
        fresh = age <= np.timedelta64(max_age_days, "D")     # NaT compares False -> masked

        panel = vals.where(fresh).reindex(index)
        if save:
            panel.to_parquet(fund_dir / f"{name}.parquet")
        panels[name] = panel
    return panels


def load_fundamental(name):
    return pd.read_parquet(fund_dir / f"{name}.parquet")

def get_ff_factors(save=True):
    """Daily Fama-French 3 factors + momentum from Ken French's Data Library (decimals)."""
    base = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    files = ["F-F_Research_Data_Factors_daily_CSV.zip", "F-F_Momentum_Factor_daily_CSV.zip"]

    frames = []
    for f in files:
        z = zipfile.ZipFile(io.BytesIO(requests.get(base + f, headers={"User-Agent": "Mozilla/5.0"}).content))
        rows = [line.split(",") for line in z.read(z.namelist()[0]).decode("latin-1").splitlines()]
        is_date = lambda r: r[0].strip().isdigit() and len(r[0].strip()) == 8
        start = next(i for i, r in enumerate(rows) if is_date(r))
        header = [c.strip() for c in rows[start - 1]][1:]
        body = []
        for r in rows[start:]:
            if not is_date(r):
                break
            body.append([c.strip() for c in r])
        df = pd.DataFrame(body, columns=["date", *header])
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        frames.append(df.set_index("date").astype(float) / 100)

    ff = pd.concat(frames, axis=1).dropna()
    if save:
        ff.to_parquet(data_dir / "ff_factors.parquet")
    return ff

def get_russell3000(path=None, save=True):
    """Russell 3000 constituents (tickers + GICS sectors) from a downloaded iShares IWV holdings CSV."""
    path = path or data_dir / "IWV_holdings.csv"
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().strip('"').startswith("Ticker"))
    df = pd.read_csv(StringIO("\n".join(lines[start:])), on_bad_lines="skip")

    df = df[df["Asset Class"] == "Equity"][["Ticker", "Name", "Sector"]]
    df.columns = ["ticker", "name", "sector"]

    df["ticker"] = df["ticker"].astype(str).str.strip().str.replace(r"[ .]", "-", regex=True)   # BRK B -> BRK-B
    df = df[df["ticker"].str.fullmatch(r"[A-Z]+(-[A-Z])?") & (df["sector"] != "Other")]           # drop junk rows

    # iShares sometimes writes class shares without a separator (BRKB); match to SEC style (BRK-B)
    sec = set(get_cik_map(save=False)["ticker"])
    fix = lambda t: t if t in sec or len(t) < 2 or (t[:-1] + "-" + t[-1]) not in sec else t[:-1] + "-" + t[-1]
    df["ticker"] = df["ticker"].map(fix)
    df = df[df["ticker"].isin(sec)].drop_duplicates("ticker").reset_index(drop=True)

    if save:
        df.to_csv(data_dir / "constituents.csv", index=False)
    return df

def get_prices_volume(tickers, years=5, chunk=200, save=True):
    """Daily adjusted close and volume, downloaded in batches. Returns prices, volume, failed tickers."""
    start = pd.Timestamp.today() - pd.DateOffset(years=years)
    px, vol = [], []

    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        d = yf.download(batch, start=start, auto_adjust=True, progress=False, threads=True)
        px.append(d["Close"])
        vol.append(d["Volume"])
        print(f"{min(i + chunk, len(tickers))}/{len(tickers)}")
        time.sleep(1)

    prices = pd.concat(px, axis=1).dropna(axis=1, how="all").dropna(axis=0, how="all")
    volume = pd.concat(vol, axis=1).reindex_like(prices)
    failed = sorted(set(tickers) - set(prices.columns))

    if save:
        prices.to_parquet(data_dir / "prices" / "prices.parquet")
        volume.to_parquet(data_dir / "prices" / "volume.parquet")
    return prices, volume, failed