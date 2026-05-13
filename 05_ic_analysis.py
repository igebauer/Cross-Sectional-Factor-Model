"""
ic_analysis.py
--------------
Stage 4: IC Analysis — Quality Signal Decomposition.

Compares IC of gross margin, ROA, and quality composite.

Usage:
    python ic_analysis.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PLOT_DIR  = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

MAX_DECAY_LAGS = 6


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
    return returns, gm_ranked, roa_ranked, quality_ranked


def compute_ic_series(signal, returns, lag=1):
    ic_series = {}
    dates = signal.index.tolist()
    for i, date in enumerate(dates):
        if i + lag >= len(dates):
            break
        fwd_date = dates[i + lag]
        if fwd_date not in returns.index:
            continue
        sig_row = signal.loc[date]
        ret_row = returns.loc[fwd_date]
        common  = sig_row.dropna().index.intersection(ret_row.dropna().index)
        if len(common) < 20:
            continue
        ic, _ = stats.spearmanr(sig_row[common], ret_row[common])
        ic_series[date] = ic
    return pd.Series(ic_series, name=f"IC_lag{lag}")


def compute_ic_decay(signal, returns, max_lag=MAX_DECAY_LAGS):
    rows = []
    for lag in range(1, max_lag + 1):
        ic_s    = compute_ic_series(signal, returns, lag=lag)
        ic_mean = ic_s.mean()
        ic_std  = ic_s.std()
        rows.append({
            "lag":          lag,
            "ic_mean":      ic_mean,
            "ic_std":       ic_std,
            "ic_ir":        ic_mean / ic_std if ic_std > 0 else np.nan,
            "pct_positive": (ic_s > 0).mean(),
            "n_months":     len(ic_s),
        })
    return pd.DataFrame(rows)


def compute_ic_stats(ic_series):
    ic_clean = ic_series.dropna()
    t_stat, p_val = stats.ttest_1samp(ic_clean, 0)
    return {
        "ic_mean":      ic_clean.mean(),
        "ic_std":       ic_clean.std(),
        "ic_ir":        ic_clean.mean() / ic_clean.std() if ic_clean.std() > 0 else np.nan,
        "pct_positive": (ic_clean > 0).mean(),
        "t_stat":       t_stat,
        "p_value":      p_val,
        "n_months":     len(ic_clean),
    }


def print_summary_table(stats_gm, stats_roa, stats_q):
    print("\n  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │          IC Summary: Lag = 1 Month (Quality Decomposition)      │")
    print("  ├─────────────────────┬───────────────┬───────────────┬───────────┤")
    print("  │ Metric              │  Gross Margin │      ROA      │ Composite │")
    print("  ├─────────────────────┼───────────────┼───────────────┼───────────┤")
    metrics = [
        ("IC Mean",     "ic_mean",      ".4f"),
        ("IC Std",      "ic_std",       ".4f"),
        ("IC IR",       "ic_ir",        ".4f"),
        ("% Positive",  "pct_positive", ".1%"),
        ("T-Stat",      "t_stat",       ".2f"),
        ("P-Value",     "p_value",      ".3f"),
        ("N Months",    "n_months",     "d"),
    ]
    for label, key, fmt in metrics:
        vals = []
        for s in [stats_gm, stats_roa, stats_q]:
            v = s.get(key, np.nan)
            try:
                vals.append(format(v, fmt))
            except Exception:
                vals.append(str(v))
        print(f"  │ {label:<19} │ {vals[0]:>13} │ {vals[1]:>13} │ {vals[2]:>9} │")
    print("  └─────────────────────┴───────────────┴───────────────┴───────────┘")

    ir_gm  = stats_gm["ic_ir"]
    ir_roa = stats_roa["ic_ir"]
    ir_q   = stats_q["ic_ir"]
    print(f"\n  IC IR — Gross Margin: {ir_gm:.4f} | ROA: {ir_roa:.4f} | Composite: {ir_q:.4f}")

    if ir_q > ir_gm and ir_q > ir_roa:
        print("  ✓ Composite outperforms both components — combination adds value.")
        print("    Gross margin and ROA capture distinct quality dimensions.")
    elif ir_q > min(ir_gm, ir_roa):
        weaker = "Gross Margin" if ir_gm < ir_roa else "ROA"
        print(f"  ~ Composite beats {weaker} but not both — partial benefit.")
    else:
        best = "Gross Margin" if ir_gm > ir_roa else "ROA"
        print(f"  ~ {best} alone outperforms composite — components are correlated.")
        print("    Single-component signal may be preferable.")


def plot_ic_analysis(ic_gm, ic_roa, ic_q, decay_gm, decay_roa, decay_q):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("IC Analysis: Quality Decomposition — Gross Margin vs ROA vs Composite", fontsize=12)

    COLOR_GM  = "#2196F3"
    COLOR_ROA = "#E91E63"
    COLOR_Q   = "#9C27B0"

    signals = [
        (ic_gm,  decay_gm,  COLOR_GM,  "Gross Margin"),
        (ic_roa, decay_roa, COLOR_ROA, "ROA"),
        (ic_q,   decay_q,   COLOR_Q,   "Composite"),
    ]

    # 12m rolling IC
    ax = axes[0, 0]
    for ic_s, _, color, label in signals:
        roll = ic_s.rolling(12).mean()
        ax.plot(roll.index, roll.values, color=color, linewidth=2.0, label=f"{label} (12m avg)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title("Monthly IC — 12m Rolling Mean")
    ax.set_ylabel("IC")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.tick_params(axis="x", rotation=30)

    # IC decay
    ax = axes[0, 1]
    lags = decay_gm["lag"].values
    for _, decay, color, label in signals:
        ax.plot(lags, decay["ic_mean"].values, "o-", color=color,
                linewidth=1.8, markersize=6, label=label)
        ax.fill_between(lags,
                        decay["ic_mean"] - decay["ic_std"],
                        decay["ic_mean"] + decay["ic_std"],
                        alpha=0.08, color=color)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title("IC Decay Curve (lags 1-6)")
    ax.set_xlabel("Forward Lag (months)")
    ax.set_ylabel("IC Mean")
    ax.set_xticks(lags)
    ax.legend(fontsize=8)

    # IC IR bar chart
    ax = axes[1, 0]
    x     = np.arange(len(lags))
    width = 0.25
    for k, (_, decay, color, label) in enumerate(signals):
        ax.bar(x + (k - 1) * width, decay["ic_ir"].values, width,
               color=color, alpha=0.8, label=label)
    ax.axhline(0.3, color="green", linewidth=1.0, linestyle="--", label="IR=0.3 threshold")
    ax.axhline(0,   color="gray",  linewidth=0.8, linestyle=":")
    ax.set_title("IC IR by Lag")
    ax.set_xlabel("Forward Lag (months)")
    ax.set_ylabel("IC IR")
    ax.set_xticks(x)
    ax.set_xticklabels([f"t+{l}" for l in lags])
    ax.legend(fontsize=7)

    # IC distribution
    ax = axes[1, 1]
    for ic_s, _, color, label in signals:
        ax.hist(ic_s.dropna().values, bins=25, color=color, alpha=0.5,
                label=f"{label} (mean={ic_s.mean():.3f})", edgecolor="white")
        ax.axvline(ic_s.mean(), color=color, linewidth=2.0, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title("IC Distribution (lag=1)")
    ax.set_xlabel("IC")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "ic_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {out}")


if __name__ == "__main__":
    print("\n=== Stage 4: IC Analysis (Quality Decomposition) ===\n")

    print("[1/5] Loading data...")
    returns, gm_ranked, roa_ranked, quality_ranked = load_data()

    print("[2/5] Computing IC series (lag=1)...")
    ic_gm      = compute_ic_series(gm_ranked,      returns, lag=1)
    ic_roa     = compute_ic_series(roa_ranked,     returns, lag=1)
    ic_quality = compute_ic_series(quality_ranked, returns, lag=1)
    print(f"  IC gross margin months: {len(ic_gm)}")
    print(f"  IC ROA months         : {len(ic_roa)}")
    print(f"  IC composite months   : {len(ic_quality)}")

    print("\n[3/5] Computing IC decay curves (lags 1-6)...")
    print("  Gross margin decay...")
    decay_gm      = compute_ic_decay(gm_ranked,      returns)
    print("  ROA decay...")
    decay_roa     = compute_ic_decay(roa_ranked,     returns)
    print("  Composite decay...")
    decay_quality = compute_ic_decay(quality_ranked, returns)

    print("\n[4/5] Computing summary statistics...")
    stats_gm  = compute_ic_stats(ic_gm)
    stats_roa = compute_ic_stats(ic_roa)
    stats_q   = compute_ic_stats(ic_quality)
    print_summary_table(stats_gm, stats_roa, stats_q)

    print("\n[5/5] Plotting...")
    plot_ic_analysis(ic_gm, ic_roa, ic_quality,
                     decay_gm, decay_roa, decay_quality)

    ic_df = pd.DataFrame({
        "ic_grossmargin": ic_gm,
        "ic_roa":         ic_roa,
        "ic_quality":     ic_quality,
    })
    ic_df.to_csv(os.path.join(DATA_DIR, "ic_results.csv"))
    decay_gm.to_csv(os.path.join(DATA_DIR,      "decay_grossmargin.csv"), index=False)
    decay_roa.to_csv(os.path.join(DATA_DIR,     "decay_roa.csv"),         index=False)
    decay_quality.to_csv(os.path.join(DATA_DIR, "decay_quality.csv"),     index=False)

    print("\n=== Stage 4 Complete. Ready for Stage 5: Walk-Forward Validation. ===")
