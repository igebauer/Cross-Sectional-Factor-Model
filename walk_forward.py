"""
walk_forward.py
---------------
Stage 5: Walk-Forward Portfolio Validation — Quality Decomposition.

Builds L/S portfolios for gross margin, ROA, and quality composite.
Long top 30%, short bottom 30%, equal-weighted. 60-month burn-in.

Usage:
    python walk_forward.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PLOT_DIR  = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

TOP_PCT    = 0.30
BOT_PCT    = 0.30
BURN_IN    = 60
MIN_STOCKS = 10


def load_data():
    returns = pd.read_csv(
        os.path.join(DATA_DIR, "returns_monthly.csv"),
        index_col=0, parse_dates=True
    )
    gm_ranked = pd.read_csv(
        os.path.join(DATA_DIR, "grossmargin_ranked.csv"),
        index_col=0, parse_dates=True
    )
    roa_ranked = pd.read_csv(
        os.path.join(DATA_DIR, "roa_ranked.csv"),
        index_col=0, parse_dates=True
    )
    quality_ranked = pd.read_csv(
        os.path.join(DATA_DIR, "quality_signal_ranked.csv"),
        index_col=0, parse_dates=True
    )
    factors = pd.read_csv(
        os.path.join(DATA_DIR, "factors_aligned.csv"),
        index_col=0, parse_dates=True
    )
    return returns, gm_ranked, roa_ranked, quality_ranked, factors


def build_ls_portfolio(signal, returns):
    dates      = signal.index.tolist()
    records    = []
    prev_long  = set()
    prev_short = set()

    for i in range(BURN_IN, len(dates) - 1):
        date     = dates[i]
        fwd_date = dates[i + 1]

        sig_row = signal.loc[date].dropna()
        if len(sig_row) < MIN_STOCKS * 2:
            continue
        if fwd_date not in returns.index:
            continue

        fwd_ret       = returns.loc[fwd_date]
        n_long        = max(MIN_STOCKS, int(len(sig_row) * TOP_PCT))
        n_short       = max(MIN_STOCKS, int(len(sig_row) * BOT_PCT))
        sorted_sig    = sig_row.sort_values(ascending=False)
        long_tickers  = set(sorted_sig.iloc[:n_long].index)
        short_tickers = set(sorted_sig.iloc[-n_short:].index)

        long_rets  = fwd_ret[list(long_tickers)].dropna()
        short_rets = fwd_ret[list(short_tickers)].dropna()
        if len(long_rets) < 5 or len(short_rets) < 5:
            continue

        if prev_long and prev_short:
            lo = len(long_tickers  & prev_long)  / max(len(long_tickers),  1)
            so = len(short_tickers & prev_short) / max(len(short_tickers), 1)
            turnover = 1 - (lo + so) / 2
        else:
            turnover = 1.0

        prev_long  = long_tickers
        prev_short = short_tickers

        records.append({
            "date":      fwd_date,
            "ret":       long_rets.mean() - short_rets.mean(),
            "long_ret":  long_rets.mean(),
            "short_ret": short_rets.mean(),
            "n_long":    len(long_rets),
            "n_short":   len(short_rets),
            "turnover":  turnover,
        })

    df = pd.DataFrame(records).set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    return df


def compute_metrics(port, rf, label):
    rets       = port["ret"].dropna()
    common_idx = rets.index.intersection(rf.index)
    rets_al    = rets.loc[common_idx]
    rf_al      = rf.loc[common_idx]
    excess     = rets_al - rf_al
    ann_ret    = rets_al.mean() * 12
    ann_vol    = rets_al.std()  * np.sqrt(12)
    ann_sharpe = excess.mean() / excess.std() * np.sqrt(12) if excess.std() > 0 else np.nan
    cum        = (1 + rets_al).cumprod()
    max_dd     = ((cum - cum.cummax()) / cum.cummax()).min()
    return {
        "label":        label,
        "ann_return":   ann_ret,
        "ann_vol":      ann_vol,
        "sharpe":       ann_sharpe,
        "max_drawdown": max_dd,
        "calmar":       ann_ret / abs(max_dd) if max_dd != 0 else np.nan,
        "win_rate":     (rets_al > 0).mean(),
        "avg_turnover": port["turnover"].mean(),
        "n_months":     len(rets_al),
    }


def print_metrics_table(metrics_list):
    print("\n  ┌────────────────────────┬────────────────┬────────────────┬────────────────┬────────────────┐")
    print("  │ Metric                 │  Gross Margin  │      ROA       │   Composite    │    Market      │")
    print("  ├────────────────────────┼────────────────┼────────────────┼────────────────┼────────────────┤")
    rows = [
        ("Ann. Return",     "ann_return",   ".2%"),
        ("Ann. Volatility", "ann_vol",      ".2%"),
        ("Sharpe Ratio",    "sharpe",       ".3f"),
        ("Max Drawdown",    "max_drawdown", ".2%"),
        ("Calmar Ratio",    "calmar",       ".3f"),
        ("Win Rate",        "win_rate",     ".1%"),
        ("Avg Turnover",    "avg_turnover", ".1%"),
        ("N Months",        "n_months",     "d"),
    ]
    for label, key, fmt in rows:
        vals = []
        for m in metrics_list:
            v = m.get(key, np.nan)
            try:
                vals.append(format(v, fmt))
            except Exception:
                vals.append(str(v))
        print(f"  │ {label:<22} │ {vals[0]:>14} │ {vals[1]:>14} │ {vals[2]:>14} │ {vals[3]:>14} │")
    print("  └────────────────────────┴────────────────┴────────────────┴────────────────┴────────────────┘")


def plot_results(port_gm, port_roa, port_q, market_ret):
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    fig.suptitle("Walk-Forward L/S: Gross Margin vs ROA vs Quality Composite", fontsize=13)

    COLOR_GM  = "#2196F3"
    COLOR_ROA = "#E91E63"
    COLOR_Q   = "#9C27B0"
    COLOR_MK  = "#4CAF50"

    common = (port_gm.index
              .intersection(port_roa.index)
              .intersection(port_q.index)
              .intersection(market_ret.index))

    r_gm  = port_gm.loc[common,  "ret"]
    r_roa = port_roa.loc[common, "ret"]
    r_q   = port_q.loc[common,   "ret"]
    r_mkt = market_ret.loc[common]

    # Cumulative returns
    ax = axes[0]
    for rets, color, label, ls in [
        (r_gm,  COLOR_GM,  "Gross Margin L/S", "-"),
        (r_roa, COLOR_ROA, "ROA L/S",          "-"),
        (r_q,   COLOR_Q,   "Composite L/S",    "-"),
        (r_mkt, COLOR_MK,  "Market",           "--"),
    ]:
        ax.plot((1 + rets).cumprod().index, (1 + rets).cumprod().values,
                color=color, linewidth=1.8 if ls == "-" else 1.2,
                linestyle=ls, alpha=0.85, label=label)
    ax.axhline(1, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title("Cumulative Returns (out-of-sample)")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=9)

    # Drawdown
    ax = axes[1]
    for rets, color, label in [(r_gm, COLOR_GM, "Gross Margin"), (r_roa, COLOR_ROA, "ROA"), (r_q, COLOR_Q, "Composite")]:
        cum = (1 + rets).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        ax.fill_between(dd.index, dd.values, 0, alpha=0.25, color=color)
        ax.plot(dd.index, dd.values, color=color, linewidth=1.2, label=label)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=9)

    # Turnover
    ax = axes[2]
    for port, color, label in [(port_gm, COLOR_GM, "Gross Margin"), (port_roa, COLOR_ROA, "ROA"), (port_q, COLOR_Q, "Composite")]:
        turn = port.loc[common, "turnover"].rolling(6).mean()
        ax.plot(turn.index, turn.values * 100, color=color, linewidth=1.5,
                label=f"{label} (6m avg)")
    ax.set_title("Portfolio Turnover (6-month rolling average)")
    ax.set_ylabel("Turnover (%)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "walk_forward.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {out}")


if __name__ == "__main__":
    print("\n=== Stage 5: Walk-Forward Validation (Quality Decomposition) ===\n")

    print("[1/5] Loading data...")
    returns, gm_ranked, roa_ranked, quality_ranked, factors = load_data()
    market_ret = factors["Mkt-RF"] + factors["RF"]
    rf         = factors["RF"]

    print("[2/5] Building portfolios...")
    port_gm  = build_ls_portfolio(gm_ranked,      returns)
    port_roa = build_ls_portfolio(roa_ranked,     returns)
    port_q   = build_ls_portfolio(quality_ranked, returns)
    print(f"  Gross margin months : {len(port_gm)}")
    print(f"  ROA months          : {len(port_roa)}")
    print(f"  Composite months    : {len(port_q)}")

    print("\n[3/5] Computing performance metrics...")
    m_gm  = compute_metrics(port_gm,  rf, "Gross Margin")
    m_roa = compute_metrics(port_roa, rf, "ROA")
    m_q   = compute_metrics(port_q,   rf, "Composite")
    mkt_df = pd.DataFrame({"ret": market_ret, "turnover": 0})
    m_mkt  = compute_metrics(mkt_df, rf, "Market")
    print_metrics_table([m_gm, m_roa, m_q, m_mkt])

    print("\n  --- Interpretation ---")
    sharpes = {"Gross Margin": m_gm["sharpe"], "ROA": m_roa["sharpe"], "Composite": m_q["sharpe"]}
    best    = max(sharpes, key=sharpes.get)
    print(f"  Best Sharpe: {best} ({sharpes[best]:.3f})")

    if m_q["sharpe"] > m_gm["sharpe"] and m_q["sharpe"] > m_roa["sharpe"]:
        print("  ✓ Composite outperforms both components on Sharpe.")
        print("    Signal combination is validated: gross margin and ROA")
        print("    capture distinct dimensions of firm quality.")
    elif m_q["sharpe"] > min(m_gm["sharpe"], m_roa["sharpe"]):
        print("  ~ Composite beats the weaker component — partial benefit.")
    else:
        print(f"  ~ {best} alone outperforms composite in this universe/period.")

    for label, m in [("Gross Margin", m_gm), ("ROA", m_roa), ("Composite", m_q)]:
        print(f"  {label} avg turnover: {m['avg_turnover']:.1%}/month")
    print("  Note: No transaction costs applied.")

    port_all = (port_gm[["ret"]].rename(columns={"ret": "gross_margin"})
                .join(port_roa[["ret"]].rename(columns={"ret": "roa"}),       how="outer")
                .join(port_q[["ret"]].rename(columns={"ret": "composite"}),   how="outer"))
    port_all.to_csv(os.path.join(DATA_DIR, "portfolio_returns.csv"))

    print("\n[4/5] Plotting...")
    plot_results(port_gm, port_roa, port_q, market_ret)

    print("\n=== Stage 5 Complete ===")
    print("All outputs saved to /data and /plots.")
