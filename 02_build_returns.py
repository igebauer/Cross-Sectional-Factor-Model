"""
build_returns.py
----------------
Constructs the clean, aligned monthly returns panel used by all downstream stages.

Outputs (saved to /data):
  - returns_monthly.csv   : simple monthly returns, index=date, columns=tickers
  - panel.csv             : long-format panel: date, ticker, ret, mcap, log_mcap

Run after fetch_data.py.

Usage:
    python build_returns.py
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three cached CSVs from fetch_data.py."""
    prices = pd.read_csv(
        os.path.join(DATA_DIR, "prices_monthly.csv"),
        index_col=0, parse_dates=True
    )
    mcaps = pd.read_csv(
        os.path.join(DATA_DIR, "market_caps_monthly.csv"),
        index_col=0, parse_dates=True
    )
    factors = pd.read_csv(
        os.path.join(DATA_DIR, "ff_factors_monthly.csv"),
        index_col=0, parse_dates=True
    )
    return prices, mcaps, factors


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple monthly returns from monthly adjusted close prices.
    ret_t = (price_t / price_{t-1}) - 1

    Drops the first row (NaN) and any tickers with >10% remaining NaNs.
    """
    rets = prices.pct_change()
    rets = rets.iloc[1:]  # drop first NaN row

    # Winsorize at 1%/99% to remove data errors (e.g. stock splits not adjusted)
    lower = rets.stack().quantile(0.01)
    upper = rets.stack().quantile(0.99)
    rets = rets.clip(lower=lower, upper=upper)

    # Drop tickers still missing >10% of months
    max_missing = 0.10 * len(rets)
    rets = rets.dropna(axis=1, thresh=int(len(rets) - max_missing))

    return rets


def align_data(
    rets: pd.DataFrame,
    mcaps: pd.DataFrame,
    factors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align all three DataFrames to the same date index and ticker universe.
    Factor dates drive the alignment since they're the most reliable.
    """
    common_dates = rets.index.intersection(factors.index)
    common_tickers = rets.columns.intersection(mcaps.columns)

    rets    = rets.loc[common_dates, common_tickers]
    mcaps   = mcaps.loc[common_dates, common_tickers]
    factors = factors.loc[common_dates]

    print(f"  Aligned date range : {common_dates[0].date()} → {common_dates[-1].date()}")
    print(f"  Months             : {len(common_dates)}")
    print(f"  Tickers            : {len(common_tickers)}")

    return rets, mcaps, factors


def build_panel(rets: pd.DataFrame, mcaps: pd.DataFrame) -> pd.DataFrame:
    """
    Melt returns and market caps into a long-format panel.
    Columns: date, ticker, ret, mcap, log_mcap

    Long format is convenient for cross-sectional operations later.
    """
    ret_long  = rets.stack().rename("ret").reset_index()
    ret_long.columns = ["date", "ticker", "ret"]

    mcap_long = mcaps.stack().rename("mcap").reset_index()
    mcap_long.columns = ["date", "ticker", "mcap"]

    panel = ret_long.merge(mcap_long, on=["date", "ticker"], how="left")
    panel["log_mcap"] = np.log(panel["mcap"].clip(lower=1))  # log for size factor
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    return panel


def validate(rets: pd.DataFrame, factors: pd.DataFrame) -> None:
    """Basic sanity checks — print warnings if something looks off."""
    print("\n  --- Validation ---")

    # Check return magnitudes
    mean_ret = rets.stack().mean()
    std_ret  = rets.stack().std()
    print(f"  Mean monthly return : {mean_ret:.4f}  (expect ~0.008–0.015)")
    print(f"  Std monthly return  : {std_ret:.4f}   (expect ~0.05–0.10)")

    if abs(mean_ret) > 0.05:
        print("  WARNING: mean return looks too large — check for data errors")
    if std_ret > 0.20:
        print("  WARNING: std too high — winsorization may need adjustment")

    # Check factor alignment
    mkt_mean = factors["Mkt-RF"].mean()
    print(f"  Mean monthly Mkt-RF : {mkt_mean:.4f}  (expect ~0.006–0.010)")

    # Check for any all-NaN columns
    nan_cols = rets.columns[rets.isna().all()].tolist()
    if nan_cols:
        print(f"  WARNING: {len(nan_cols)} all-NaN columns: {nan_cols[:5]}")
    else:
        print("  No all-NaN columns — good.")

    print("  --- End Validation ---\n")


if __name__ == "__main__":
    print("\n=== Stage 1b: Building Returns Panel ===\n")

    print("[1/4] Loading raw data...")
    prices, mcaps, factors = load_raw()

    print("[2/4] Computing returns...")
    rets = compute_returns(prices)
    print(f"  Returns shape: {rets.shape}")

    print("[3/4] Aligning data...")
    rets, mcaps, factors = align_data(rets, mcaps, factors)

    validate(rets, factors)

    print("[4/4] Building long-format panel...")
    panel = build_panel(rets, mcaps)
    print(f"  Panel shape: {panel.shape}")
    print(f"  Columns: {list(panel.columns)}")
    print(f"\n  Sample (first 5 rows):\n{panel.head().to_string()}")

    # Save outputs
    rets_path   = os.path.join(DATA_DIR, "returns_monthly.csv")
    panel_path  = os.path.join(DATA_DIR, "panel.csv")
    factor_path = os.path.join(DATA_DIR, "factors_aligned.csv")

    rets.to_csv(rets_path)
    panel.to_csv(panel_path, index=False)
    factors.to_csv(factor_path)

    print(f"\n  Saved: returns_monthly.csv")
    print(f"  Saved: panel.csv")
    print(f"  Saved: factors_aligned.csv")

    print("\n=== Stage 1 Complete. Ready for Stage 2: Factor Replication. ===")
