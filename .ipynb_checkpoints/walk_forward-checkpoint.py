"""
walk_forward.py
---------------
Stage 5: Walk-Forward Portfolio Validation with Purging.

Methodology:
  - Each month, rank stocks by signal (residual momentum).
  - Go long top 30%, short bottom 30% (equal-weighted within each leg).
  - This is a "paper" long-short portfolio — no leverage assumptions.

Walk-forward design:
  - Initial training window: first 60 months (5 years) — not used for
    model fitting here since our signal is purely cross-sectional, but
    the burn-in period is respected by only trading after sufficient history.
  - Purge gap: 1 month between signal computation and portfolio formation
    to prevent lookahead bias (already baked in via t-2 skip in momentum).
  - Out-of-sample period: everything after the initial burn-in.

Performance metrics:
  - Annualized return
  - Annualized Sharpe ratio (using RF from FF data)
  - Maximum drawdown
  - Calmar ratio (annualized return / max drawdown)
  - Turnover (avg monthly portfolio change)
  - Win rate (% of months with positive return)

Comparison:
  - Residual momentum L/S portfolio
  - Raw momentum L/S portfolio
  - Market (Mkt-RF + RF from FF factors)

Outputs:
  - data/portfolio_returns.csv    : monthly L/S returns for both signals
  - plots/walk_forward.png        : cumulative returns + drawdown + turnover

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
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Portfolio construction parameters
TOP_PCT    = 0.30   # long leg: top 30% by signal
BOT_PCT    = 0.30   # short leg: bottom 30% by signal
BURN_IN    = 60     # months before we start trading (ensures enough history)
MIN_STOCKS = 10     # minimum stocks per leg


# ── Load Data ─────────────────────────────────────────────────────────────────

def load_data():
    returns = pd.read_csv(
        os.path.join(DATA_DIR, "returns_monthly.csv"),
        index_col=0, parse_dates=True
    )
    raw_ranked = pd.read_csv(
        os.path.join(DATA_DIR, "raw_mom_ranked.csv"),
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
    return returns, raw_ranked, quality_ranked, factors


# ── Portfolio Construction ────────────────────────────────────────────────────

def build_ls_portfolio(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    label: str = "signal",
) -> pd.DataFrame:
    """
    Build a monthly long-short portfolio from a ranked signal.

    At each month t:
      1. Rank stocks by signal_t
      2. Long top 30%, short bottom 30% (equal-weighted)
      3. Realized return = mean(long returns_{t+1}) - mean(short returns_{t+1})

    Returns a DataFrame with columns:
      ret, long_ret, short_ret, n_long, n_short, turnover
    """
    dates   = signal.index.tolist()
    records = []

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

        fwd_ret = returns.loc[fwd_date]

        # Portfolio formation: top/bottom by signal rank
        n_long  = max(MIN_STOCKS, int(len(sig_row) * TOP_PCT))
        n_short = max(MIN_STOCKS, int(len(sig_row) * BOT_PCT))

        sorted_sig = sig_row.sort_values(ascending=False)
        long_tickers  = set(sorted_sig.iloc[:n_long].index)
        short_tickers = set(sorted_sig.iloc[-n_short:].index)

        # Get forward returns for each leg
        long_rets  = fwd_ret[list(long_tickers)].dropna()
        short_rets = fwd_ret[list(short_tickers)].dropna()

        if len(long_rets) < 5 or len(short_rets) < 5:
            continue

        long_ret  = long_rets.mean()
        short_ret = short_rets.mean()
        ls_ret    = long_ret - short_ret

        # Turnover: fraction of portfolio that changed
        if prev_long and prev_short:
            long_overlap  = len(long_tickers & prev_long)  / max(len(long_tickers),  1)
            short_overlap = len(short_tickers & prev_short) / max(len(short_tickers), 1)
            turnover = 1 - (long_overlap + short_overlap) / 2
        else:
            turnover = 1.0

        prev_long  = long_tickers
        prev_short = short_tickers

        records.append({
            "date":      fwd_date,
            "ret":       ls_ret,
            "long_ret":  long_ret,
            "short_ret": short_ret,
            "n_long":    len(long_rets),
            "n_short":   len(short_rets),
            "turnover":  turnover,
        })

    df = pd.DataFrame(records).set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    return df


# ── Performance Metrics ───────────────────────────────────────────────────────

def compute_metrics(
    port: pd.DataFrame,
    rf: pd.Series,
    label: str,
) -> dict:
    """Compute annualized performance metrics for a L/S portfolio."""
    rets = port["ret"].dropna()
    common_idx = rets.index.intersection(rf.index)
    rets_aligned = rets.loc[common_idx]
    rf_aligned   = rf.loc[common_idx]

    excess = rets_aligned - rf_aligned
    ann_ret   = rets_aligned.mean() * 12
    ann_vol   = rets_aligned.std()  * np.sqrt(12)
    ann_sharpe = excess.mean() / excess.std() * np.sqrt(12) if excess.std() > 0 else np.nan

    # Max drawdown
    cum = (1 + rets_aligned).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    win_rate = (rets_aligned > 0).mean()
    avg_turnover = port["turnover"].mean()

    return {
        "label":        label,
        "ann_return":   ann_ret,
        "ann_vol":      ann_vol,
        "sharpe":       ann_sharpe,
        "max_drawdown": max_dd,
        "calmar":       calmar,
        "win_rate":     win_rate,
        "avg_turnover": avg_turnover,
        "n_months":     len(rets_aligned),
    }


def print_metrics_table(metrics_list: list) -> None:
    print("\n  ┌────────────────────────┬──────────────────┬──────────────────┬──────────────────┐")
    print("  │ Metric                 │  Raw Momentum    │    Quality       │    Market        │")
    print("  ├────────────────────────┼──────────────────┼──────────────────┼──────────────────┤")

    rows = [
        ("Ann. Return",    "ann_return",   ".2%"),
        ("Ann. Volatility","ann_vol",      ".2%"),
        ("Sharpe Ratio",   "sharpe",       ".3f"),
        ("Max Drawdown",   "max_drawdown", ".2%"),
        ("Calmar Ratio",   "calmar",       ".3f"),
        ("Win Rate",       "win_rate",     ".1%"),
        ("Avg Turnover",   "avg_turnover", ".1%"),
        ("N Months",       "n_months",     "d"),
    ]

    for label, key, fmt in rows:
        vals = []
        for m in metrics_list:
            v = m.get(key, np.nan)
            try:
                vals.append(format(v, fmt))
            except Exception:
                vals.append(str(v))
        print(f"  │ {label:<22} │ {vals[0]:>16} │ {vals[1]:>16} │ {vals[2]:>16} │")

    print("  └────────────────────────┴──────────────────┴──────────────────┴──────────────────┘")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(
    port_raw:   pd.DataFrame,
    port_resid: pd.DataFrame,
    market_ret: pd.Series,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    fig.suptitle("Walk-Forward L/S Portfolio: Residual vs Raw Momentum", fontsize=13)

    COLOR_RAW    = "#2196F3"
    COLOR_RESID  = "#E91E63"
    COLOR_MARKET = "#4CAF50"

    # Align all to common dates
    common = port_raw.index.intersection(port_resid.index).intersection(market_ret.index)
    r_raw   = port_raw.loc[common, "ret"]
    r_resid = port_resid.loc[common, "ret"]
    r_mkt   = market_ret.loc[common]

    # ── Cumulative returns ────────────────────────────────────────────────────
    ax = axes[0]
    cum_raw   = (1 + r_raw).cumprod()
    cum_resid = (1 + r_resid).cumprod()
    cum_mkt   = (1 + r_mkt).cumprod()

    ax.plot(cum_raw.index,   cum_raw.values,   color=COLOR_RAW,
            linewidth=1.8, label="Raw Momentum L/S")
    ax.plot(cum_resid.index, cum_resid.values, color=COLOR_RESID,
            linewidth=1.8, label="Residual Momentum L/S")
    ax.plot(cum_mkt.index,   cum_mkt.values,   color=COLOR_MARKET,
            linewidth=1.2, linestyle="--", alpha=0.7, label="Market")
    ax.axhline(1, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title("Cumulative Returns (out-of-sample)")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=9)

    # ── Drawdown ──────────────────────────────────────────────────────────────
    ax = axes[1]
    for rets, color, label in [
        (r_raw,   COLOR_RAW,   "Raw Momentum"),
        (r_resid, COLOR_RESID, "Residual Momentum"),
    ]:
        cum = (1 + rets).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        ax.fill_between(dd.index, dd.values, 0, alpha=0.35, color=color, label=label)
        ax.plot(dd.index, dd.values, color=color, linewidth=0.8, alpha=0.7)

    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=9)

    # ── Monthly turnover ──────────────────────────────────────────────────────
    ax = axes[2]
    turn_raw   = port_raw.loc[common, "turnover"].rolling(6).mean()
    turn_resid = port_resid.loc[common, "turnover"].rolling(6).mean()

    ax.plot(turn_raw.index,   turn_raw.values   * 100, color=COLOR_RAW,
            linewidth=1.5, label="Raw Momentum (6m avg)")
    ax.plot(turn_resid.index, turn_resid.values * 100, color=COLOR_RESID,
            linewidth=1.5, label="Residual Momentum (6m avg)")
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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Stage 5: Walk-Forward Validation ===\n")

    print("[1/5] Loading data...")
    returns, raw_ranked, quality_ranked, factors = load_data()

    # Market return = Mkt-RF + RF
    market_ret = factors["Mkt-RF"] + factors["RF"]

    print("[2/5] Building raw momentum L/S portfolio...")
    port_raw = build_ls_portfolio(raw_ranked, returns, label="Raw Momentum")
    print(f"  Portfolio months: {len(port_raw)}")

    print("[3/5] Building residual momentum L/S portfolio...")
    port_quality = build_ls_portfolio(quality_ranked, returns, label="Quality")
    print(f"  Portfolio months: {len(port_resid)}")

    print("[4/5] Computing performance metrics...")
    rf = factors["RF"]

    m_raw   = compute_metrics(port_raw,   rf, "Raw Momentum")
    m_quality = compute_metrics(port_quality, rf, "Quality")

    # Market benchmark metrics
    mkt_df = pd.DataFrame({"ret": market_ret, "turnover": 0})
    m_mkt  = compute_metrics(mkt_df, rf, "Market")

    print_metrics_table([m_raw, m_quality, m_mkt])

    # Interpretation
    print("\n  --- Interpretation ---")
    s_resid = m_resid["sharpe"]
    s_raw   = m_raw["sharpe"]

    if s_resid > s_raw and s_resid > 0:
        print(f"  ✓ Residual momentum L/S Sharpe ({s_resid:.3f}) > raw ({s_raw:.3f})")
        print("    Factor stripping improves risk-adjusted returns.")
    elif s_resid > 0:
        print(f"  ~ Residual momentum Sharpe positive ({s_resid:.3f}) but below raw ({s_raw:.3f}).")
        print("    L/S portfolio generates positive returns; factor stripping")
        print("    reduces gross return but also risk.")
    else:
        print(f"  ~ Sharpe near zero or negative for both signals in this universe/period.")
        print("    Large-cap momentum is widely arbitraged — consistent with IC results.")
        print("    Turnover and drawdown metrics still validate the pipeline.")
        print("    Consider: quality factor (Stage 3 rebuild) for stronger signal.")

    print(f"\n  Avg monthly turnover (resid): {m_resid['avg_turnover']:.1%}")
    print("  Note: No transaction costs applied. Real implementation would")
    print("  reduce Sharpe by ~0.1-0.3 depending on execution quality.")

    # Save portfolio returns
    port_all = port_raw[["ret"]].rename(columns={"ret": "raw_mom"}).join(
        port_quality[["ret"]].rename(columns={"ret": "quality"}), how="outer"
    )
    port_all.to_csv(os.path.join(DATA_DIR, "portfolio_returns.csv"))

    print("\n[5/5] Plotting...")
    plot_results(port_raw, port_quality, market_ret)

    print("\n=== Stage 5 Complete ===")
    print("\nProject pipeline complete. All outputs saved to /data and /plots.")
    print("Next steps: write up methodology + findings, push to GitHub.")
