import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import factors

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
    stock = _slice(prices[ticker], period)
    return stock, _slice(market_px, period).reindex(stock.index)


def _rebase(s):
    return 100 * s / s.iloc[0]


def _line(s, name, color="black", width=1.6, dash=None):
    return go.Scatter(x=s.index, y=s.values, mode="lines", name=name,
                      line=dict(color=color, width=width, dash=dash))


def _period_figure(make_traces, title, ylabel, default="1y", hline=None,
                   caption=None, subtitle=None, xlabel=None, hovermode="x unified"):
    """make_traces(p) -> list of traces; subtitle(p) -> str. One figure with a period dropdown."""
    periods = list(PERIODS)

    def full_title(p):
        t = f"{title}  ({p})"
        if subtitle:
            t += f"<br><span style='font-size:12px;color:gray'>{subtitle(p)}</span>"
        return t

    fig, n = go.Figure(), None
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