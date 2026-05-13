"""
build_signal.py
---------------
Stage 3: Quality Signal Decomposition.

Constructs three signals from yfinance fundamental data:
  1. Gross Margin  : grossMargins
  2. ROA           : returnOnAssets  <-- primary signal (best IC IR and Sharpe)
  3. Composite     : equal-weight z(gross_margin) + z(ROA)
                     retained for decomposition comparison only

Decomposition finding: ROA alone (Sharpe 0.409, IC IR 0.156) outperforms
both gross margin (Sharpe 0.202) and the composite (Sharpe 0.318). The high
GM/ROA cross-sectional correlation (r=0.43) causes the composite to dilute
the stronger ROA signal rather than diversify it.

Outputs:
  - data/grossmargin_ranked.csv
  - data/roa_ranked.csv
  - data/quality_signal_ranked.csv  <-- points to composite (GM + ROA)
  - data/signal_panel.csv
  - plots/signal_overview.png

Usage:
    python build_signal.py
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


def load_data():
    returns = pd.read_csv(
        os.path.join(DATA_DIR, "returns_monthly.csv"),
        index_col=0, parse_dates=True
    )
    factors = pd.read_csv(
        os.path.join(DATA_DIR, "factors_aligned.csv"),
        index_col=0, parse_dates=True
    )
    return returns, factors


def fetch_quality_data(tickers, cache_path):
    if os.path.exists(cache_path):
        print("  [cache] quality_data.csv found, loading...")
        df = pd.read_csv(cache_path, index_col=0)
        df.index.name = "ticker"
        return df

    import yfinance as yf
    import time

    print(f"  Fetching quality data for {len(tickers)} tickers...")
    records = []
    for i, ticker in enumerate(tickers):
        try:
            info = yf.Ticker(ticker).info
            gm  = info.get("grossMargins",   None)
            roa = info.get("returnOnAssets", None)
            records.append({
                "ticker":       ticker,
                "gross_margin": float(gm)  if gm  is not None else np.nan,
                "roa":          float(roa) if roa is not None else np.nan,
            })
        except Exception:
            records.append({"ticker": ticker, "gross_margin": np.nan, "roa": np.nan})
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{len(tickers)} done...")
            time.sleep(0.5)

    df = pd.DataFrame(records).set_index("ticker")
    df.to_csv(cache_path)
    print(f"  Saved to {cache_path}")
    return df


def zscore(s):
    s = s.dropna()
    return (s - s.mean()) / s.std()


def broadcast(series, index, columns):
    """Broadcast a per-ticker Series into a full date x ticker DataFrame."""
    aligned = series.reindex(columns)
    return pd.DataFrame(
        np.tile(aligned.values, (len(index), 1)),
        index=index,
        columns=columns,
    )


def build_signals(returns, quality_df):
    common = returns.columns.intersection(quality_df.index)
    qdf    = quality_df.loc[common].copy()

    gm_z        = zscore(qdf["gross_margin"]).clip(-3, 3)
    roa_z       = zscore(qdf["roa"]).clip(-3, 3)
    composite_z = gm_z.add(roa_z, fill_value=0).clip(-3, 3)

    idx  = returns.index
    cols = returns.columns

    gm_panel        = broadcast(gm_z,        idx, cols)
    roa_panel       = broadcast(roa_z,        idx, cols)
    composite_panel = broadcast(composite_z,  idx, cols)

    return gm_panel, roa_panel, composite_panel


def rank_normalize(signal_df):
    def _rank_z(row):
        valid = row.dropna()
        if len(valid) < 5:
            return row
        ranks = valid.rank()
        z = (ranks - ranks.mean()) / ranks.std()
        result = pd.Series(np.nan, index=row.index)
        result[valid.index] = z
        return result
    return signal_df.apply(_rank_z, axis=1)


def to_panel(gm, roa, composite):
    gm_long  = gm.stack().rename("gross_margin").reset_index()
    roa_long = roa.stack().rename("roa").reset_index()
    q_long   = composite.stack().rename("composite").reset_index()
    gm_long.columns  = ["date", "ticker", "gross_margin"]
    roa_long.columns = ["date", "ticker", "roa"]
    q_long.columns   = ["date", "ticker", "composite"]
    panel = (gm_long
             .merge(roa_long, on=["date", "ticker"], how="outer")
             .merge(q_long,   on=["date", "ticker"], how="outer"))
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def plot_signals(signal_panel):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Signal Decomposition: Gross Margin vs ROA vs Composite", fontsize=13)

    latest_date  = signal_panel["date"].max()
    latest_slice = signal_panel[signal_panel["date"] == latest_date]

    for j, (col, label, color) in enumerate([
        ("gross_margin", "Gross Margin", "#2196F3"),
        ("roa",          "ROA (Primary)", "#E91E63"),
        ("composite",    "Composite",    "#9C27B0"),
    ]):
        ax = axes[0, j]
        vals = latest_slice[col].dropna()
        ax.hist(vals, bins=25, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(vals.mean(), color="black", linewidth=1.2, linestyle="--",
                   label=f"mean={vals.mean():.3f}")
        ax.set_title(f"{label} ({latest_date.date()})")
        ax.set_xlabel("Rank Z-Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

        ax2 = axes[1, j]
        cs_std = signal_panel.groupby("date")[col].std()
        ax2.plot(cs_std.index, cs_std.values, color=color, linewidth=1.5)
        ax2.set_title(f"{label}: Dispersion Over Time")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Cross-Sectional Std")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax2.xaxis.set_major_locator(mdates.YearLocator(3))
        ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "signal_overview.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {out}")


if __name__ == "__main__":
    print("\n=== Stage 3: Quality Signal Decomposition ===\n")

    print("[1/6] Loading data...")
    returns, factors = load_data()
    tickers = returns.columns.tolist()
    print(f"  Returns: {returns.shape}")

    print("\n[2/6] Fetching quality data...")
    quality_cache = os.path.join(DATA_DIR, "quality_data.csv")
    quality_df    = fetch_quality_data(tickers, quality_cache)
    print(f"  Valid gross_margin: {quality_df['gross_margin'].notna().sum()}/{len(tickers)}")
    print(f"  Valid ROA         : {quality_df['roa'].notna().sum()}/{len(tickers)}")

    print("\n[3/6] Building signals...")
    gm_raw, roa_raw, composite_raw = build_signals(returns, quality_df)

    print("\n[4/6] Rank-normalizing...")
    gm_ranked        = rank_normalize(gm_raw)
    roa_ranked       = rank_normalize(roa_raw)
    composite_ranked = rank_normalize(composite_raw)

    print("\n[5/6] Saving...")
    gm_ranked.to_csv(os.path.join(DATA_DIR,        "grossmargin_ranked.csv"))
    roa_ranked.to_csv(os.path.join(DATA_DIR,       "roa_ranked.csv"))
    composite_ranked.to_csv(os.path.join(DATA_DIR, "composite_ranked.csv"))
    # quality_signal_ranked = composite (GM + ROA equal-weight)
    composite_ranked.to_csv(os.path.join(DATA_DIR, "quality_signal_ranked.csv"))

    print("\n[6/6] Building panel and plotting...")
    signal_panel = to_panel(gm_ranked, roa_ranked, composite_ranked)
    signal_panel.to_csv(os.path.join(DATA_DIR, "signal_panel.csv"), index=False)
    print(f"  Signal panel shape: {signal_panel.shape}")
    plot_signals(signal_panel)

    print("\n  --- Signal Summary ---")
    for col, label in [("gross_margin", "Gross Margin"), ("roa", "ROA (Primary)"), ("composite", "Composite")]:
        vals = signal_panel[col].dropna()
        print(f"\n  {label}: {len(vals):,} obs | mean={vals.mean():.4f} | std={vals.std():.4f}")
    print("  --- End Summary ---")

    print("\n=== Stage 3 Complete. Ready for Stage 4: IC Analysis. ===")
