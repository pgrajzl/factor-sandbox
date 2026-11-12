import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import factors
from plotly.subplots import make_subplots

PERIODS = {
    "5y": pd.DateOffset(years=5),
    "3y": pd.DateOffset(years=3),
    "1y": pd.DateOffset(years=1),
    "6mo": pd.DateOffset(months=6),
}
FONT = dict(family="Times New Roman", size=14, color="black")
GRID = dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", griddash="dash")


def _slice(s, period):
    s = s.dropna()
    return s[s.index >= s.index.max() - PERIODS[period]]


def _window(prices, market_px, ticker, period):
    """Align stock and benchmark on common dates, then slice the window."""
    df = pd.concat([prices[ticker], market_px], axis=1, join="inner").dropna()
    df = df[df.index >= df.index.max() - PERIODS[period]]
    return df.iloc[:, 0], df.iloc[:, 1]


def _rebase(s):
    return 100 * s / s.iloc[0]


def _line(s, name, color="black", width=1.6, dash=None):
    return go.Scatter(x=s.index, y=s.values, mode="lines", name=name,
                      line=dict(color=color, width=width, dash=dash))


def _period_figure(make_traces, title, ylabel, default="1y", hline=None,
                   caption=None, subtitle=None, xlabel=None, hovermode="x unified", fig=None):
    """make_traces(p) -> list of traces; subtitle(p) -> str. One figure with a period dropdown."""
    periods = list(PERIODS)

    def full_title(p):
        t = f"{title}  ({p})"
        if subtitle:
            t += f"<br><span style='font-size:12px;color:gray'>{subtitle(p)}</span>"
        return t

    fig, n = (fig if fig is not None else go.Figure()), None
    for p in periods:
        traces = make_traces(p)
        n = len(traces)
        for t in traces:
            t.visible = (p == default)
            fig.add_trace(t)

    buttons = [dict(label=p, method="update",
                    args=[{"visible": [j // n == i for j in range(n * len(periods))]},
                          {"title.text": full_title(p)}])
               for i, p in enumerate(periods)]

    fig.update_layout(
        template="simple_white", font=FONT,
        title=dict(text=full_title(default), x=0.5),
        xaxis_title=xlabel, yaxis_title=ylabel, hovermode=hovermode,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom"),
        margin=dict(t=100, l=60, r=30, b=50), width=900, height=480,
        updatemenus=[dict(buttons=buttons, active=periods.index(default),
                          x=0, xanchor="left", y=1.2, yanchor="top",
                          bgcolor="white", font=FONT)],
    )
    fig.update_yaxes(**GRID)
    if hline is not None:
        fig.add_hline(y=hline, line=dict(color="gray", width=0.8), opacity=0.6)
    if caption:
        fig.add_annotation(text=caption, xref="paper", yref="paper", x=0.5, y=-0.16,
                           xanchor="center", yanchor="top", showarrow=False,
                           font=dict(family="Times New Roman", size=11, color="gray"))
        fig.update_layout(margin_b=80)
    return fig


def plot_stock(prices, ticker, default="1y"):
    """Price history for one stock."""
    return _period_figure(
        lambda p: [_line(_slice(prices[ticker], p), ticker)],
        title=ticker, ylabel="Price ($)", default=default,
    )


def plot_vs_market(prices, market_px, ticker, default="1y", market_name="SPY"):
    """Stock vs market, rebased to 100 at window start, with window beta in the subtitle."""
    def traces(p):
        stock, mkt = _window(prices, market_px, ticker, p)
        return [_line(_rebase(stock), ticker),
                _line(_rebase(mkt), market_name, color="gray", width=1.3, dash="dash")]

    def subtitle(p):
        stock, mkt = _window(prices, market_px, ticker, p)
        b = factors.beta(stock.pct_change(), mkt.pct_change())
        return f"{p} beta to {market_name}: {b:.2f}  (daily returns)"

    return _period_figure(
        traces, title=f"{ticker} vs {market_name}",
        ylabel="Growth of $100", default=default, hline=100, subtitle=subtitle,
        caption="Both series scaled to 100 at the start of the selected window; "
                "values show growth of $100 invested.",
    )


def plot_regression(prices, market_px, ticker, default="1y", market_name="SPY"):
    """Scatter of daily stock vs market returns with OLS fit, equation, R² and correlation."""
    data, fits = {}, {}
    for p in PERIODS:
        stock, mkt = _window(prices, market_px, ticker, p)
        df = pd.concat([stock.pct_change(), mkt.pct_change()], axis=1).dropna() * 100
        df.columns = ["s", "m"]
        data[p], fits[p] = df, factors.regress(df["s"], df["m"])

    def traces(p):
        df, f = data[p], fits[p]
        x = np.linspace(df["m"].min(), df["m"].max(), 50)
        return [
            go.Scatter(x=df["m"], y=df["s"], mode="markers", name="Daily returns",
                       marker=dict(color="gray", size=5, opacity=0.45),
                       text=df.index.strftime("%Y-%m-%d"),
                       hovertemplate=f"%{{text}}<br>{market_name}: %{{x:.2f}}%"
                                     f"<br>{ticker}: %{{y:.2f}}%<extra></extra>"),
            go.Scatter(x=x, y=f["alpha"] + f["beta"] * x, mode="lines", name="OLS fit",
                       line=dict(color="black", width=1.8), hoverinfo="skip"),
        ]

    def subtitle(p):
        f = fits[p]
        return (f"r<sub>{ticker}</sub> = {f['alpha']:.3f}% + {f['beta']:.2f} r<sub>{market_name}</sub>"
                f"   |   R² = {f['r2']:.2f}   ρ = {f['corr']:.2f}   n = {f['n']}")

    fig = _period_figure(
        traces, title=f"{ticker} vs {market_name}: daily returns",
        xlabel=f"{market_name} daily return (%)", ylabel=f"{ticker} daily return (%)",
        default=default, subtitle=subtitle, hovermode="closest",
    )
    fig.add_annotation(
        text="Line is the OLS fit r = α + β·m; vertical distance from each point "
             "to the line is the idiosyncratic return ε.",
        xref="paper", yref="paper", x=0.5, y=-0.24, xanchor="center", yanchor="top",
        showarrow=False, font=dict(family="Times New Roman", size=11, color="gray"),
    )
    fig.update_layout(margin_b=110)
    fig.update_xaxes(**GRID, zeroline=True, zerolinecolor="gray", zerolinewidth=0.8)
    fig.update_yaxes(zeroline=True, zerolinecolor="gray", zerolinewidth=0.8)
    return fig

def plot_two_factor(prices, market_px, sector_px, ticker, default="1y",
                    market_name="SPY", sector_name="XLK"):
    """AAPL-style two-factor fit: market, then sector orthogonalized to market."""
    base = pd.concat([prices[ticker], market_px, sector_px], axis=1, join="inner").dropna()
    base.columns = ["s", "m", "x"]

    fits = {}
    for p in PERIODS:
        w = base[base.index >= base.index.max() - PERIODS[p]]
        r = w.pct_change().dropna() * 100
        F = factors.sequential_orthogonalize(r[["m", "x"]])
        fits[p] = dict(r=r, u=F["x"],
                       f1=factors.ols(r["s"], F[["m"]]),
                       f2=factors.ols(r["s"], F))

    pts = dict(color="gray", size=5, opacity=0.45)
    fit_line = dict(color="black", width=1.8)

    def traces(p):
        d = fits[p]
        r, u, f1, f2 = d["r"], d["u"], d["f1"], d["f2"]
        bm, bs = f2["betas"]["m"], f2["betas"]["x"]
        dates = r.index.strftime("%Y-%m-%d")
        xm = np.linspace(r["m"].min(), r["m"].max(), 50)
        xu = np.linspace(u.min(), u.max(), 50)
        hover = "%{text}<br>x: %{x:.2f}%<br>y: %{y:.2f}%<extra></extra>"
        return [
            go.Scatter(x=r["m"], y=r["s"], mode="markers", marker=pts,
                       text=dates, hovertemplate=hover, showlegend=False),
            go.Scatter(x=xm, y=f1["alpha"] + bm * xm, mode="lines", line=fit_line,
                       hoverinfo="skip", showlegend=False),
            go.Scatter(x=u, y=f1["resid"], mode="markers", marker=pts,
                       text=dates, hovertemplate=hover, showlegend=False,
                       xaxis="x2", yaxis="y2"),
            go.Scatter(x=xu, y=bs * xu, mode="lines", line=fit_line,
                       hoverinfo="skip", showlegend=False, xaxis="x2", yaxis="y2"),
        ]

    def subtitle(p):
        f1, f2 = fits[p]["f1"], fits[p]["f2"]
        bm, bs = f2["betas"]["m"], f2["betas"]["x"]
        return (f"r<sub>{ticker}</sub> = {f2['alpha']:.3f}% + {bm:.2f} r<sub>{market_name}</sub>"
                f" + {bs:.2f} u<sub>{sector_name}</sub>   |   "
                f"R²: {f1['r2']:.2f} → {f2['r2']:.2f}  (ΔR² = {f2['r2'] - f1['r2']:+.2f})"
                f"   n = {f2['n']}")

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.1,
        subplot_titles=(f"{ticker} vs {market_name}",
                        f"{ticker} market residual vs {sector_name} ⊥ {market_name}"),
    )
    fig = _period_figure(traces, title=f"{ticker} on {market_name} + {sector_name} (orthogonalized)",
                         ylabel=None, default=default, subtitle=subtitle,
                         hovermode="closest", fig=fig)

    fig.update_layout(width=1150, height=540, margin=dict(t=130, b=110))
    fig.update_annotations(font=dict(family="Times New Roman", size=13))
    fig.update_xaxes(title_text=f"{market_name} return (%)", row=1, col=1)
    fig.update_yaxes(title_text=f"{ticker} return (%)", row=1, col=1)
    fig.update_xaxes(title_text=f"{sector_name} residual u (%)", row=1, col=2)
    fig.update_yaxes(title_text=f"{ticker} market residual (%)", row=1, col=2)
    fig.update_xaxes(**GRID, zeroline=True, zerolinecolor="gray", zerolinewidth=0.8)
    fig.update_yaxes(zeroline=True, zerolinecolor="gray", zerolinewidth=0.8)

    fig.add_annotation(
        text="Left: market fit (slope β_m). Right: AAPL's market residual vs the part of XLK "
             "unexplained by SPY (slope β_s). Factors orthogonalized within each window.",
        xref="paper", yref="paper", x=0.5, y=-0.22, xanchor="center", yanchor="top",
        showarrow=False, font=dict(family="Times New Roman", size=11, color="gray"),
    )
    return fig

def plot_actual_vs_fitted(prices, factor_px, ticker, default="1y"):
    """Actual vs fitted daily returns for a multi-factor model.
    factor_px: dict {name: price series}, in orthogonalization order."""
    names = list(factor_px)
    base = pd.concat([prices[ticker], *factor_px.values()], axis=1, join="inner").dropna()
    base.columns = ["s", *names]

    fits = {}
    for p in PERIODS:
        w = base[base.index >= base.index.max() - PERIODS[p]]
        r = w.pct_change().dropna() * 100
        f = factors.ols(r["s"], factors.sequential_orthogonalize(r[names]))
        f["actual"], f["fitted"] = r["s"], r["s"] - f["resid"]
        fits[p] = f

    def traces(p):
        f = fits[p]
        lo = min(f["actual"].min(), f["fitted"].min())
        hi = max(f["actual"].max(), f["fitted"].max())
        return [
            go.Scatter(x=f["fitted"], y=f["actual"], mode="markers", name="Daily returns",
                       marker=dict(color="gray", size=5, opacity=0.45),
                       text=f["actual"].index.strftime("%Y-%m-%d"),
                       hovertemplate="%{text}<br>Fitted: %{x:.2f}%<br>Actual: %{y:.2f}%<extra></extra>"),
            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="45° line (perfect fit)",
                       line=dict(color="black", width=1.6, dash="dash"), hoverinfo="skip"),
        ]

    def subtitle(p):
        f = fits[p]
        terms = " + ".join(
            f"{b:.2f} {'r' if i == 0 else 'u'}<sub>{name}</sub>"
            for i, (name, b) in enumerate(f["betas"].items())
        )
        idio_vol = f["resid"].std() * np.sqrt(252)
        return (f"r<sub>{ticker}</sub> = {f['alpha']:.3f}% + {terms}   |   "
                f"R² = {f['r2']:.2f}   idio vol = {idio_vol:.1f}% ann.   n = {f['n']}")

    fig = _period_figure(
        traces, title=f"{ticker}: actual vs fitted ({' + '.join(names)})",
        xlabel="Fitted daily return (%)", ylabel="Actual daily return (%)",
        default=default, subtitle=subtitle, hovermode="closest",
    )
    fig.update_layout(
        width=720, height=660, margin=dict(t=100, b=110),
        title=dict(y=0.97, yanchor="top"),
        legend=dict(orientation="v", x=0.02, xanchor="left", y=0.98, yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)"),
    )
    fig.update_xaxes(**GRID, zeroline=True, zerolinecolor="gray", zerolinewidth=0.8)
    fig.update_yaxes(zeroline=True, zerolinecolor="gray", zerolinewidth=0.8,
                     scaleanchor="x", scaleratio=1)
    fig.add_annotation(
        text="Fitted = α + Σ β·factor. Points on the dashed line are fully explained; "
             "vertical distance to it is the idiosyncratic return ε.",
        xref="paper", yref="paper", x=0.5, y=-0.17, xanchor="center", yanchor="top",
        showarrow=False, font=dict(family="Times New Roman", size=11, color="gray"),
    )
    return fig

def plot_factor_returns(fr, name="Momentum (12-1)", default="5y"):
    """Cumulative factor return over the selected window, with window stats in the subtitle."""
    s = fr * 100

    def subtitle(p):
        w = _slice(s, p)
        ann, vol = w.mean() * 252, w.std() * np.sqrt(252)
        return f"Ann. return {ann:.1f}%   vol {vol:.1f}%   Sharpe {ann / vol:.2f}   days = {len(w)}"

    return _period_figure(
        lambda p: [_line(_slice(s, p).cumsum(), name)],
        title=f"{name} factor: cumulative return", ylabel="Cumulative return (%)",
        default=default, hline=0, subtitle=subtitle,
        caption="Daily return per 1σ of momentum exposure from the cross-sectional regression, "
                "summed from the start of the window.",
    )


def plot_cs_r2(r2, name="Momentum (12-1)", default="1y", window=21):
    """Daily cross-sectional R² with a rolling mean."""
    s = r2 * 100
    roll = s.rolling(window).mean()

    return _period_figure(
        lambda p: [_line(_slice(s, p), "Daily R²", color="lightgray", width=1),
                   _line(_slice(roll, p), f"{window}d average")],
        title=f"Cross-sectional R²: {name}", ylabel="R² (%)", default=default,
        subtitle=lambda p: f"Mean daily R²: {_slice(s, p).mean():.2f}%",
    )

def plot_exposure_corr(z_a, z_b, name_a="Momentum", name_b="Volatility", default="1y", window=21):
    """Daily cross-sectional correlation between two factor exposures, with a rolling mean."""
    rho = z_a.corrwith(z_b, axis=1).dropna()
    roll = rho.rolling(window).mean()

    return _period_figure(
        lambda p: [_line(_slice(rho, p), "Daily correlation", color="lightgray", width=1),
                   _line(_slice(roll, p), f"{window}d average")],
        title=f"Cross-sectional exposure correlation: {name_a} vs {name_b}",
        ylabel="Correlation", default=default, hline=0,
        subtitle=lambda p: f"Mean correlation: {_slice(rho, p).mean():.2f}",
        caption="Correlation across all stocks, each day, between the two standardized exposures. "
                "Far from zero means the isolated factor returns contaminate each other.",
    )

def plot_factor_compare(series, title="Factor returns", default="1y", caption=None):
    """Cumulative returns of several factor series on common dates, Sharpe of each in the subtitle."""
    styles = [dict(color="black"), dict(color="gray", dash="dash"),
              dict(color="#4a6fa5"), dict(color="#b5651d", dash="dot")]
    df = pd.concat(series, axis=1, join="inner").dropna() * 100

    def window(p):
        return df[df.index >= df.index.max() - PERIODS[p]]

    def subtitle(p):
        w = window(p)
        sr = w.mean() / w.std() * np.sqrt(252)
        return "   ".join(f"{c}: Sharpe {sr[c]:.2f}" for c in w)

    fig = _period_figure(
        lambda p: [_line(window(p)[c].cumsum(), c, **styles[i % len(styles)])
                   for i, c in enumerate(df)],
        title=title, ylabel="Cumulative return (%)", default=default,
        hline=0, subtitle=subtitle, caption=caption,
    )
    fig.update_layout(margin_t=120)
    return fig