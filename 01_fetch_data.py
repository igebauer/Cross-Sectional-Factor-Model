"""
fetch_data.py
-------------
Pulls all raw data needed for the factor model and saves to /data.

Data sources:
  1. Monthly adjusted close prices  -- yfinance
  2. Market cap (monthly proxy)     -- yfinance (shares * price, quarterly fundamentals)
  3. Fama-French 5 factors          -- Ken French data library (manual CSV download)

Run once; outputs are cached as CSVs so you don't re-download every session.

Usage:
    python fetch_data.py
"""

import os
import time
import warnings
import pandas as pd
import yfinance as yf

from universe import TICKERS, START_DATE, END_DATE

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ── 1. Monthly Adjusted Close Prices ─────────────────────────────────────────

def fetch_prices() -> pd.DataFrame:
    """
    Download monthly adjusted close prices for all tickers.
    Returns a DataFrame: index=date (month-end), columns=tickers.
    """
    cache_path = os.path.join(DATA_DIR, "prices_monthly.csv")
    if os.path.exists(cache_path):
        print("  [cache] prices_monthly.csv found, loading...")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print(f"  Downloading daily prices for {len(TICKERS)} tickers via yfinance...")
    print("  (This may take 2-4 minutes on first run)")

    # Download in batches to avoid rate limits
    batch_size = 50
    all_prices = []

    for i in range(0, len(TICKERS), batch_size):
        batch = TICKERS[i:i + batch_size]
        print(f"  Batch {i // batch_size + 1}: {batch[0]} ... {batch[-1]}")
        try:
            raw = yf.download(
                batch,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,
                progress=False,
                threads=True,
            )["Close"]
            all_prices.append(raw)
        except Exception as e:
            print(f"  Warning: batch failed ({e}), retrying individually...")
            for ticker in batch:
                try:
                    t = yf.download(ticker, start=START_DATE, end=END_DATE,
                                    auto_adjust=True, progress=False)["Close"]
                    t.name = ticker
                    all_prices.append(t.to_frame())
                except Exception:
                    print(f"    Skipping {ticker}")
        time.sleep(0.5)

    prices_daily = pd.concat(all_prices, axis=1)
    prices_daily = prices_daily.loc[:, ~prices_daily.columns.duplicated()]

    # Resample to month-end
    prices_monthly = prices_daily.resample("ME").last()

    # Drop tickers with >20% missing months
    threshold = 0.8 * len(prices_monthly)
    prices_monthly = prices_monthly.dropna(axis=1, thresh=int(threshold))

    print(f"  Retained {prices_monthly.shape[1]} tickers after missing-data filter.")
    prices_monthly.to_csv(cache_path)
    print(f"  Saved to {cache_path}")
    return prices_monthly


# ── 2. Market Cap Proxy ───────────────────────────────────────────────────────

def fetch_market_caps(prices_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Approximates monthly market cap using shares outstanding from yfinance info.
    Shares outstanding is point-in-time approximate (latest filing), so this is
    a proxy -- sufficient for a research project, noted as limitation.

    Returns a DataFrame: index=date (month-end), columns=tickers.
    """
    cache_path = os.path.join(DATA_DIR, "market_caps_monthly.csv")
    if os.path.exists(cache_path):
        print("  [cache] market_caps_monthly.csv found, loading...")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print("  Fetching shares outstanding for market cap proxy...")
    shares = {}
    tickers_in_universe = prices_monthly.columns.tolist()

    for i, ticker in enumerate(tickers_in_universe):
        try:
            info = yf.Ticker(ticker).info
            so = info.get("sharesOutstanding", None)
            if so:
                shares[ticker] = so
        except Exception:
            pass
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(tickers_in_universe)} tickers processed...")
            time.sleep(0.5)

    # Market cap = price * shares (static shares, dynamic price)
    shares_series = pd.Series(shares)
    common = prices_monthly.columns.intersection(shares_series.index)
    mcap = prices_monthly[common].multiply(shares_series[common], axis=1)

    mcap.to_csv(cache_path)
    print(f"  Saved to {cache_path}")
    return mcap


# ── 3. Fama-French Factors ────────────────────────────────────────────────────

FF5_RAW_PATH = os.path.join(DATA_DIR, "raw_ff5.csv")

FF5_DOWNLOAD_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)

MANUAL_DOWNLOAD_MSG = """
  ┌─────────────────────────────────────────────────────────────────┐
  │  ONE-TIME MANUAL STEP: Download FF factor data (takes ~30 sec)  │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │  Go to:                                                         │
  │     {ff5_url}                                                   │
  │     → Save the CSV inside the ZIP as:  data/raw_ff5.csv        │
  │                                                                 │
  │  Then re-run: python fetch_data.py                              │
  └─────────────────────────────────────────────────────────────────┘
""".format(ff5_url=FF5_DOWNLOAD_URL)


def _parse_french_csv(path: str, value_col_name: str = None) -> pd.DataFrame:
    """
    Parse a Ken French data library CSV.
    These files have a text header, then monthly data (YYYYMM), then annual data.
    We extract only the monthly rows.
    """
    with open(path, "r", encoding="latin-1") as f:
        raw = f.read()

    lines = raw.splitlines()

    # Find first line where the date field looks like YYYYMM (6 digits)
    data_start = None
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.split(",")]
        if parts[0].isdigit() and len(parts[0]) == 6:
            data_start = i
            break

    if data_start is None:
        raise ValueError(f"Could not find monthly data block in {path}")

    # Collect monthly rows (YYYYMM = 6-digit dates); stop at annual block (4-digit)
    data_lines = []
    for line in lines[data_start:]:
        parts = [p.strip() for p in line.split(",")]
        if not parts[0]:
            continue
        if parts[0].isdigit() and len(parts[0]) == 6:
            data_lines.append(line)
        elif parts[0].isdigit() and len(parts[0]) == 4:
            break  # hit annual summary section

    from io import StringIO
    # Use the header line just before data_start if it looks like a header
    header_line = lines[data_start - 1] if data_start > 0 else None
    if header_line and not header_line.strip()[0].isdigit():
        csv_text = header_line + "\n" + "\n".join(data_lines)
        df = pd.read_csv(StringIO(csv_text))
        df.columns = [c.strip() for c in df.columns]
        date_col = df.columns[0]
    else:
        df = pd.read_csv(StringIO("\n".join(data_lines)), header=None)
        date_col = 0

    df[date_col] = pd.to_datetime(df[date_col].astype(str).str.strip(), format="%Y%m")
    df = df.set_index(date_col)
    df.index.name = "date"
    df.index = df.index + pd.offsets.MonthEnd(0)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def fetch_ff_factors() -> pd.DataFrame:
    """
    Load Fama-French 5 factors from manually downloaded CSV.
    Returns monthly factor returns (decimal, not percent) indexed by month-end date.
    Columns: Mkt-RF, SMB, HML, RMW, CMA, RF
    """
    cache_path = os.path.join(DATA_DIR, "ff_factors_monthly.csv")
    if os.path.exists(cache_path):
        print("  [cache] ff_factors_monthly.csv found, loading...")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    missing = []
    if not os.path.exists(FF5_RAW_PATH):
        missing.append("data/raw_ff5.csv")

    if missing:
        print(f"\n  Missing raw factor files: {missing}")
        print(MANUAL_DOWNLOAD_MSG)
        raise FileNotFoundError(
            "Please download the FF5 CSV manually (see instructions above)."
        )

    print("  Parsing FF5 factors...")
    df5 = _parse_french_csv(FF5_RAW_PATH)
    df5.columns = [c.strip() for c in df5.columns]

    factors = df5

    # Convert from percent to decimal
    factors = factors / 100.0

    # Trim to our date range
    factors = factors.loc[START_DATE:END_DATE].dropna()

    factors.to_csv(cache_path)
    print(f"  Saved to {cache_path}")
    return factors


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Stage 1: Fetching Data ===\n")

    print("[1/3] Prices...")
    prices = fetch_prices()
    print(f"  Shape: {prices.shape}  (months x tickers)\n")

    print("[2/3] Market caps...")
    mcaps = fetch_market_caps(prices)
    print(f"  Shape: {mcaps.shape}\n")

    print("[3/3] Fama-French factors...")
    factors = fetch_ff_factors()
    print(f"  Shape: {factors.shape}")
    print(f"  Columns: {list(factors.columns)}\n")

    print("=== All data fetched successfully ===")
    print(f"Data saved to: {DATA_DIR}")
