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


def _slice(s, period):
    s = s.dropna()
    return s[s.index >= s.index.max() - PERIODS[period]]


def _rebase(s):
    return 100 * s / s.iloc[0]


def _line(s, name, color="black", width=1.6, dash=None):
    return go.Scatter(x=s.index, y=s.values, mode="lines", name=name,
                      line=dict(color=color, width=width, dash=dash))


def _period_figure(make_traces, title, ylabel, default="1y", hline=None,
                   caption=None, subtitle=None):
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
        yaxis_title=ylabel, hovermode="x unified",
        legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom"),
        margin=dict(t=100, l=60, r=30, b=50), width=900, height=480,
        updatemenus=[dict(buttons=buttons, active=periods.index(default),
                          x=0, xanchor="left", y=1.2, yanchor="top",
                          bgcolor="white", font=FONT)],
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", griddash="dash")
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
    def window(p):
        stock = _slice(prices[ticker], p)
        return stock, _slice(market_px, p).reindex(stock.index)

    def traces(p):
        stock, mkt = window(p)
        return [_line(_rebase(stock), ticker),
                _line(_rebase(mkt), market_name, color="gray", width=1.3, dash="dash")]

    def subtitle(p):
        stock, mkt = window(p)
        b = factors.beta(stock.pct_change(), mkt.pct_change())
        return f"{p} beta to {market_name}: {b:.2f}  (daily returns)"

    return _period_figure(
        traces, title=f"{ticker} vs {market_name}",
        ylabel="Growth of $100", default=default, hline=100, subtitle=subtitle,
        caption="Both series scaled to 100 at the start of the selected window; "
                "values show growth of $100 invested.",
    )