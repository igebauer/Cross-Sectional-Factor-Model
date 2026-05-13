"""
ic_analysis.py
--------------
Stage 4: Information Coefficient (IC) Analysis.

The IC is the rank correlation (Spearman) between a signal at time t and
forward returns at time t+1. It measures whether your signal predicts returns.

Key metrics:
  - IC mean     : average predictive power (higher = better signal)
  - IC std      : consistency of predictive power
  - IC IR       : IC mean / IC std — the "Sharpe ratio of the signal"
                  IC IR > 0.3 is generally considered a useful signal
  - IC decay    : how quickly predictive power fades at t+2, t+3, ...
                  faster decay = more alpha, less factor exposure

We compute these for both raw momentum and residual momentum, then compare.
The core research question: does stripping FF5 factor exposure improve IC IR?

Outputs:
  - data/ic_results.csv         : monthly IC time series for both signals
  - plots/ic_analysis.png       : IC time series, decay curve, distributions

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
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

MAX_DECAY_LAGS = 6   # compute IC decay out to t+6


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
    return returns, raw_ranked, quality_ranked



# ── IC Computation ────────────────────────────────────────────────────────────

def compute_ic_series(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    lag: int = 1,
) -> pd.Series:
    """
    Compute monthly IC at a given forward lag.
    IC_t = Spearman rank correlation between signal_t and return_{t+lag}.

    Uses Spearman (rank) correlation — standard in factor research because
    it's robust to outliers and doesn't assume linearity.
    """
    ic_series = {}
    dates = signal.index.tolist()

    for i, date in enumerate(dates):
        if i + lag >= len(dates):
            break

        fwd_date = dates[i + lag]

        sig_row = signal.loc[date]
        ret_row = returns.loc[fwd_date] if fwd_date in returns.index else None
        if ret_row is None:
            continue

        # Align tickers
        common = sig_row.dropna().index.intersection(ret_row.dropna().index)
        if len(common) < 20:
            continue

        ic, pval = stats.spearmanr(sig_row[common], ret_row[common])
        ic_series[date] = ic

    return pd.Series(ic_series, name=f"IC_lag{lag}")


def compute_ic_decay(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    max_lag: int = MAX_DECAY_LAGS,
) -> pd.DataFrame:
    """
    Compute IC mean and IC IR for lags 1 through max_lag.
    Returns a DataFrame with columns: lag, ic_mean, ic_std, ic_ir, ic_pct_positive.
    """
    rows = []
    for lag in range(1, max_lag + 1):
        ic_s = compute_ic_series(signal, returns, lag=lag)
        ic_mean = ic_s.mean()
        ic_std  = ic_s.std()
        ic_ir   = ic_mean / ic_std if ic_std > 0 else np.nan
        pct_pos = (ic_s > 0).mean()
        rows.append({
            "lag": lag,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "pct_positive": pct_pos,
            "n_months": len(ic_s),
        })
    return pd.DataFrame(rows)


def compute_ic_stats(ic_series: pd.Series) -> dict:
    """Summary stats for a monthly IC series."""
    ic_clean = ic_series.dropna()
    t_stat, p_val = stats.ttest_1samp(ic_clean, 0)
    return {
        "ic_mean":     ic_clean.mean(),
        "ic_std":      ic_clean.std(),
        "ic_ir":       ic_clean.mean() / ic_clean.std() if ic_clean.std() > 0 else np.nan,
        "pct_positive": (ic_clean > 0).mean(),
        "t_stat":      t_stat,
        "p_value":     p_val,
        "n_months":    len(ic_clean),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_ic_analysis(
    ic_raw: pd.Series,
    ic_resid: pd.Series,
    decay_raw: pd.DataFrame,
    decay_resid: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("IC Analysis: Raw Momentum vs Residual Momentum", fontsize=13)

    COLOR_RAW   = "#2196F3"
    COLOR_RESID = "#E91E63"

    # ── Top left: IC time series ──────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(ic_raw.index,   ic_raw.values,   color=COLOR_RAW,
            linewidth=1.0, alpha=0.7, label="Raw Momentum")
    ax.plot(ic_resid.index, ic_resid.values, color=COLOR_RESID,
            linewidth=1.0, alpha=0.7, label="Residual Momentum")

    # Rolling 12m mean
    roll_raw   = ic_raw.rolling(12).mean()
    roll_resid = ic_resid.rolling(12).mean()
    ax.plot(roll_raw.index,   roll_raw.values,   color=COLOR_RAW,
            linewidth=2.0, linestyle="--", label="Raw 12m avg")
    ax.plot(roll_resid.index, roll_resid.values, color=COLOR_RESID,
            linewidth=2.0, linestyle="--", label="Resid 12m avg")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title("Monthly IC (Spearman, lag=1)")
    ax.set_ylabel("IC")
    ax.legend(fontsize=7, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.tick_params(axis="x", rotation=30)

    # ── Top right: IC decay curve ─────────────────────────────────────────────
    ax = axes[0, 1]
    lags = decay_raw["lag"].values

    ax.plot(lags, decay_raw["ic_mean"].values,   "o-", color=COLOR_RAW,
            linewidth=1.8, markersize=6, label="Raw Momentum")
    ax.plot(lags, decay_resid["ic_mean"].values, "o-", color=COLOR_RESID,
            linewidth=1.8, markersize=6, label="Residual Momentum")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.fill_between(lags, decay_raw["ic_mean"] - decay_raw["ic_std"],
                    decay_raw["ic_mean"] + decay_raw["ic_std"],
                    alpha=0.10, color=COLOR_RAW)
    ax.fill_between(lags, decay_resid["ic_mean"] - decay_resid["ic_std"],
                    decay_resid["ic_mean"] + decay_resid["ic_std"],
                    alpha=0.10, color=COLOR_RESID)
    ax.set_title("IC Decay Curve (lags 1–6 months)")
    ax.set_xlabel("Forward Lag (months)")
    ax.set_ylabel("IC Mean")
    ax.set_xticks(lags)
    ax.legend(fontsize=8)

    # ── Bottom left: IC IR bar chart ──────────────────────────────────────────
    ax = axes[1, 0]
    x = np.arange(len(lags))
    width = 0.35
    ax.bar(x - width/2, decay_raw["ic_ir"].values,   width,
           color=COLOR_RAW,   alpha=0.8, label="Raw Momentum")
    ax.bar(x + width/2, decay_resid["ic_ir"].values, width,
           color=COLOR_RESID, alpha=0.8, label="Residual Momentum")
    ax.axhline(0.3,  color="green", linewidth=1.0, linestyle="--",
               label="IC IR = 0.3 (useful signal threshold)")
    ax.axhline(0,    color="gray",  linewidth=0.8, linestyle=":")
    ax.set_title("IC IR by Lag")
    ax.set_xlabel("Forward Lag (months)")
    ax.set_ylabel("IC IR (mean / std)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"t+{l}" for l in lags])
    ax.legend(fontsize=7)

    # ── Bottom right: IC distribution ─────────────────────────────────────────
    ax = axes[1, 1]
    common_idx = ic_raw.dropna().index.intersection(ic_resid.dropna().index)
    ax.hist(ic_raw.loc[common_idx].values,   bins=25, color=COLOR_RAW,
            alpha=0.6, label="Raw Momentum", edgecolor="white")
    ax.hist(ic_resid.loc[common_idx].values, bins=25, color=COLOR_RESID,
            alpha=0.6, label="Residual Momentum", edgecolor="white")
    ax.axvline(ic_raw.mean(),   color=COLOR_RAW,   linewidth=2.0,
               linestyle="--", label=f"Raw mean={ic_raw.mean():.3f}")
    ax.axvline(ic_resid.mean(), color=COLOR_RESID, linewidth=2.0,
               linestyle="--", label=f"Resid mean={ic_resid.mean():.3f}")
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


def print_summary_table(
    stats_raw: dict,
    stats_resid: dict,
) -> None:
    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │           IC Summary: Lag = 1 Month                 │")
    print("  ├─────────────────────┬───────────────┬───────────────┤")
    print("  │ Metric              │ Raw Momentum  │    Quality    │")
    print("  ├─────────────────────┼───────────────┼───────────────┤")
    metrics = [
        ("IC Mean",       "ic_mean",      ".4f"),
        ("IC Std",        "ic_std",       ".4f"),
        ("IC IR",         "ic_ir",        ".4f"),
        ("% Positive",    "pct_positive", ".1%"),
        ("T-Stat",        "t_stat",       ".2f"),
        ("P-Value",       "p_value",      ".3f"),
        ("N Months",      "n_months",     "d"),
    ]
    for label, key, fmt in metrics:
        v_raw   = stats_raw.get(key, np.nan)
        v_resid = stats_resid.get(key, np.nan)
        try:
            r_str = format(v_raw,   fmt)
            s_str = format(v_resid, fmt)
        except Exception:
            r_str = str(v_raw)
            s_str = str(v_resid)
        print(f"  │ {label:<19} │ {r_str:>13} │ {s_str:>13} │")
    print("  └─────────────────────┴───────────────┴───────────────┘")

    # Interpretation
    ir_raw   = stats_raw.get("ic_ir", 0)
    ir_resid = stats_resid.get("ic_ir", 0)
    improvement = (ir_resid - ir_raw) / abs(ir_raw) * 100 if ir_raw != 0 else 0

    print(f"\n  IC IR improvement (resid vs raw): {improvement:+.1f}%")

    if ir_resid > ir_raw:
        print("  ✓ Residual momentum outperforms raw — factor stripping adds value.")
        print("    Interpretation: predictive power comes from firm-specific")
        print("    information, not factor co-movement.")
    else:
        print("  ~ Raw momentum IC IR >= residual in this universe/period.")
        print("    Possible reason: large-cap universe has lower factor contamination")
        print("    than a broad market sample, so stripping adds less value.")
        print("    Still a valid finding — worth discussing in writeup.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Stage 4: IC Analysis ===\n")

    print("[1/5] Loading data...")
    returns, raw_ranked, quality_ranked = load_data()

    print("[2/5] Computing IC series (lag=1)...")
    ic_raw   = compute_ic_series(raw_ranked,   returns, lag=1)
    ic_quality = compute_ic_series(quality_ranked, returns, lag=1)
    print(f"  IC raw   months: {len(ic_raw)}")
    print(f"  IC resid months: {len(ic_resid)}")

    print("\n[3/5] Computing IC decay curves (lags 1–6)...")
    print("  Raw momentum decay...")
    decay_raw   = compute_ic_decay(raw_ranked,   returns)
    print("  Residual momentum decay...")
    decay_quality = compute_ic_decay(quality_ranked, returns)

    print("\n[4/5] Computing summary statistics...")
    stats_raw   = compute_ic_stats(ic_raw)
    stats_quality = compute_ic_stats(ic_quality)
    print_summary_table(stats_raw, stats_quality)

    print("\n[5/5] Plotting...")
    plot_ic_analysis(ic_raw, ic_quality, decay_raw, decay_quality)

    # Save IC series
    ic_df = pd.DataFrame({"ic_raw": ic_raw, "ic_quality": ic_quality})
    ic_df.to_csv(os.path.join(DATA_DIR, "ic_results.csv"))
    decay_raw.to_csv(os.path.join(DATA_DIR, "decay_raw.csv"),   index=False)
    decay_quality.to_csv(os.path.join(DATA_DIR, "decay_quality.csv"), index=False)

    print("\n=== Stage 4 Complete. Ready for Stage 5: Walk-Forward Validation. ===")
