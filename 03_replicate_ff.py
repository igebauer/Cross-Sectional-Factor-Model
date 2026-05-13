"""
replicate_ff.py
---------------
Stage 2: Replicate Fama-French SMB and HML factors from scratch.

Methodology:
  - SIZE (SMB): Each month, rank stocks by market cap. SMB = avg return of
    bottom 30% minus avg return of top 30%.

  - VALUE (HML): Proxy for book-to-market using earnings yield (E/P = 1/PE).
    True B/M requires point-in-time book value (CRSP/Compustat); E/P is the
    best free approximation. HML = avg return of high E/P minus avg return of
    low E/P stocks (30/40/30 split).

  - Validation: Regress replicated factors against official FF factors.
    Target correlation > 0.70 (E/P proxy weakens HML vs true B/M).

Outputs:
  - data/replicated_factors.csv   : monthly SMB_rep, HML_rep
  - plots/factor_replication.png  : cumulative factor returns + scatter plots

Usage:
    python replicate_ff.py
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


# ── Load Data ─────────────────────────────────────────────────────────────────

def load_data():
    panel = pd.read_csv(
        os.path.join(DATA_DIR, "panel.csv"), parse_dates=["date"]
    )
    factors = pd.read_csv(
        os.path.join(DATA_DIR, "factors_aligned.csv"),
        index_col=0, parse_dates=True
    )
    returns = pd.read_csv(
        os.path.join(DATA_DIR, "returns_monthly.csv"),
        index_col=0, parse_dates=True
    )
    return panel, factors, returns


# ── Earnings Yield Proxy ──────────────────────────────────────────────────────

def fetch_earnings_yield(tickers: list, cache_path: str) -> pd.DataFrame:
    """
    Fetch trailing P/E ratios via yfinance and convert to earnings yield (1/PE).
    Returns a DataFrame: index=ticker, column='earnings_yield'.

    Note: This is a static snapshot (latest available), not point-in-time.
    Acknowledged limitation — noted in README.
    """
    if os.path.exists(cache_path):
        print("  [cache] earnings_yield.csv found, loading...")
        return pd.read_csv(cache_path, index_col=0)

    import yfinance as yf
    import time

    print(f"  Fetching P/E ratios for {len(tickers)} tickers...")
    records = []
    for i, ticker in enumerate(tickers):
        try:
            info = yf.Ticker(ticker).info
            pe = info.get("trailingPE", None)
            if pe and pe > 0:
                records.append({"ticker": ticker, "earnings_yield": 1.0 / pe})
            else:
                records.append({"ticker": ticker, "earnings_yield": np.nan})
        except Exception:
            records.append({"ticker": ticker, "earnings_yield": np.nan})

        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{len(tickers)} done...")
            time.sleep(0.5)

    ey_df = pd.DataFrame(records).set_index("ticker")
    ey_df.to_csv(cache_path)
    print(f"  Saved to {cache_path}")
    return ey_df


# ── Portfolio Construction ────────────────────────────────────────────────────

def build_smb(panel: pd.DataFrame) -> pd.Series:
    """
    SMB = Small Minus Big.
    Each month: rank by log_mcap, split 30/40/30.
    SMB_t = mean(ret of bottom 30%) - mean(ret of top 30%).
    Equal-weighted within each bucket.
    """
    smb_series = {}

    for date, group in panel.groupby("date"):
        group = group.dropna(subset=["ret", "log_mcap"])
        if len(group) < 10:
            continue

        # 30/40/30 breakpoints on market cap
        lo = group["log_mcap"].quantile(0.30)
        hi = group["log_mcap"].quantile(0.70)

        small = group[group["log_mcap"] <= lo]["ret"].mean()
        big   = group[group["log_mcap"] >= hi]["ret"].mean()

        smb_series[date] = small - big

    return pd.Series(smb_series, name="SMB_rep")


def build_hml(panel: pd.DataFrame, ey_df: pd.DataFrame) -> pd.Series:
    """
    HML = High Minus Low (value minus growth).
    Proxy: earnings yield (1/PE) as value signal.
    Each month: rank by earnings yield, split 30/40/30.
    HML_t = mean(ret of high E/P) - mean(ret of low E/P).
    """
    # Merge earnings yield into panel
    # Normalize ey_df so it always has columns: ticker, earnings_yield
    ey = ey_df.copy()
    if "ticker" not in ey.columns:
        ey = ey.reset_index()
    ey.columns = ["ticker", "earnings_yield"]
    panel = panel.merge(ey[["ticker", "earnings_yield"]], on="ticker", how="left")

    hml_series = {}

    for date, group in panel.groupby("date"):
        group = group.dropna(subset=["ret", "earnings_yield"])
        if len(group) < 10:
            continue

        lo = group["earnings_yield"].quantile(0.30)
        hi = group["earnings_yield"].quantile(0.70)

        value  = group[group["earnings_yield"] >= hi]["ret"].mean()
        growth = group[group["earnings_yield"] <= lo]["ret"].mean()

        hml_series[date] = value - growth

    return pd.Series(hml_series, name="HML_rep")


# ── Validation ────────────────────────────────────────────────────────────────

def validate_factors(
    rep: pd.DataFrame,
    official: pd.DataFrame,
) -> dict:
    """
    Regress replicated factors against official FF factors.
    Report: correlation, beta, alpha (should be near 0), R².
    """
    results = {}
    common = rep.index.intersection(official.index)
    rep_aligned = rep.loc[common]
    off_aligned = official.loc[common]

    print("\n  --- Factor Replication Validation ---")
    for col_rep, col_off in [("SMB_rep", "SMB"), ("HML_rep", "HML")]:
        if col_rep not in rep_aligned.columns or col_off not in off_aligned.columns:
            continue

        x = off_aligned[col_off].values
        y = rep_aligned[col_rep].values

        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        slope, intercept, r, p, se = stats.linregress(x, y)
        corr = np.corrcoef(x, y)[0, 1]

        print(f"\n  {col_rep} vs {col_off}:")
        print(f"    Correlation : {corr:.4f}  (target > 0.70)")
        print(f"    Beta        : {slope:.4f}  (target ~1.0)")
        print(f"    Alpha/month : {intercept:.5f}  (target ~0.0)")
        print(f"    R²          : {r**2:.4f}")

        if corr > 0.80:
            print(f"    ✓ Strong replication")
        elif corr > 0.70:
            print(f"    ✓ Acceptable replication (E/P proxy weakens HML)")
        elif corr > 0.50:
            print(f"    ~ Moderate — E/P proxy diverges from true B/M, expected")
        else:
            print(f"    ✗ Weak — check data quality")

        results[col_rep] = {"corr": corr, "beta": slope, "alpha": intercept, "r2": r**2}

    print("\n  Note: HML correlation below SMB is expected. True HML requires")
    print("  point-in-time book value (CRSP/Compustat). E/P is a reasonable proxy.")
    print("  --- End Validation ---\n")

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_replication(
    rep: pd.DataFrame,
    official: pd.DataFrame,
    results: dict,
) -> None:
    common = rep.index.intersection(official.index)
    rep_al = rep.loc[common]
    off_al = official.loc[common]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Factor Replication: SMB & HML vs Official Fama-French", fontsize=13)

    colors = {"SMB": "#2196F3", "HML": "#E91E63"}

    for i, (col_rep, col_off) in enumerate([("SMB_rep", "SMB"), ("HML_rep", "HML")]):
        color = colors[col_off]

        # Left: cumulative returns
        ax_cum = axes[i, 0]
        cum_rep = (1 + rep_al[col_rep]).cumprod() - 1
        cum_off = (1 + off_al[col_off]).cumprod() - 1

        ax_cum.plot(cum_rep.index, cum_rep.values * 100,
                    label=f"{col_rep} (replicated)", color=color, linewidth=1.8)
        ax_cum.plot(cum_off.index, cum_off.values * 100,
                    label=f"{col_off} (official FF)", color="black",
                    linewidth=1.2, linestyle="--", alpha=0.7)
        ax_cum.set_title(f"{col_off}: Cumulative Returns")
        ax_cum.set_ylabel("Cumulative Return (%)")
        ax_cum.legend(fontsize=8)
        ax_cum.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_cum.xaxis.set_major_locator(mdates.YearLocator(3))
        ax_cum.tick_params(axis="x", rotation=30)
        ax_cum.axhline(0, color="gray", linewidth=0.5, linestyle=":")

        # Right: scatter
        ax_sc = axes[i, 1]
        x = off_al[col_off].values * 100
        y = rep_al[col_rep].values * 100
        mask = ~(np.isnan(x) | np.isnan(y))

        ax_sc.scatter(x[mask], y[mask], alpha=0.35, s=18, color=color)

        # Regression line
        m, b = np.polyfit(x[mask], y[mask], 1)
        xline = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax_sc.plot(xline, m * xline + b, color="black", linewidth=1.2)

        r = results.get(col_rep, {})
        ax_sc.set_title(
            f"{col_off} Scatter  |  r={r.get('corr', 0):.2f}, β={r.get('beta', 0):.2f}"
        )
        ax_sc.set_xlabel(f"Official {col_off} (%)")
        ax_sc.set_ylabel(f"Replicated {col_rep} (%)")
        ax_sc.axhline(0, color="gray", linewidth=0.5)
        ax_sc.axvline(0, color="gray", linewidth=0.5)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "factor_replication.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Stage 2: Factor Replication ===\n")

    print("[1/5] Loading data...")
    panel, factors, returns = load_data()
    tickers = panel["ticker"].unique().tolist()

    print("[2/5] Fetching earnings yield proxy (1/PE)...")
    ey_cache = os.path.join(DATA_DIR, "earnings_yield.csv")
    ey_df = fetch_earnings_yield(tickers, ey_cache)
    valid_ey = ey_df["earnings_yield"].notna().sum()
    print(f"  Valid E/P values: {valid_ey}/{len(tickers)} tickers")

    print("[3/5] Building replicated SMB...")
    smb = build_smb(panel)
    print(f"  SMB mean: {smb.mean():.4f}  std: {smb.std():.4f}")

    print("[4/5] Building replicated HML...")
    hml = build_hml(panel, ey_df)
    print(f"  HML mean: {hml.mean():.4f}  std: {hml.std():.4f}")

    # Combine replicated factors
    rep = pd.DataFrame({"SMB_rep": smb, "HML_rep": hml})
    rep.index = pd.DatetimeIndex(rep.index)
    rep.index = rep.index + pd.offsets.MonthEnd(0)
    rep = rep.sort_index()

    # Save
    rep_path = os.path.join(DATA_DIR, "replicated_factors.csv")
    rep.to_csv(rep_path)
    print(f"\n  Replicated factors saved to {rep_path}")

    print("[5/5] Validating against official FF factors...")
    results = validate_factors(rep, factors)

    print("  Plotting...")
    plot_replication(rep, factors, results)

    print("\n=== Stage 2 Complete. Ready for Stage 3: Custom Signal. ===")
