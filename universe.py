"""
universe.py
-----------
Defines the stock universe for the factor model.
~150 liquid Russell 1000 names, diversified across sectors.
Chosen to minimize survivorship bias concern while keeping data pull manageable.

Due to lack of CRSP point-in-time constituents, I use a static liquid large-cap 
universe as an approximation. Survivorship bias note: these are current constituents. 
"""

TICKERS = [
    # Technology (25)
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL",
    "CSCO", "ACN", "IBM", "TXN", "QCOM",
    "AMAT", "ADI", "MU", "KLAC", "LRCX",
    "MSI", "CDNS", "SNPS", "ANSS", "FTNT",
    "CTSH", "HPQ", "GLW", "JNPR", "NTAP",

    # Financials (25)
    "JPM", "BAC", "WFC", "GS", "MS",
    "BLK", "SCHW", "AXP", "USB", "PNC",
    "TFC", "COF", "MTB", "RF", "HBAN",
    "CFG", "FITB", "KEY", "ZION", "CMA",
    "STT", "BK", "NTRS", "AFL", "MET",

    # Healthcare (20)
    "JNJ", "UNH", "LLY", "ABT", "TMO",
    "DHR", "MDT", "BMY", "AMGN", "GILD",
    "CVS", "CI", "HUM", "ELV", "CNC",
    "ISRG", "BSX", "SYK", "ZBH", "BAX",

    # Consumer Staples (15)
    "PG", "KO", "PEP", "COST", "WMT",
    "MO", "PM", "CL", "KMB", "GIS",
    "K", "CPB", "HRL", "SJM", "CAG",

    # Consumer Discretionary (15)
    "AMZN", "HD", "MCD", "NKE", "SBUX",
    "TJX", "LOW", "TGT", "ROST", "DHI",
    "LEN", "PHM", "NVR", "TOL", "MDC",

    # Industrials (15)
    "HON", "UPS", "CAT", "DE", "GE",
    "MMM", "RTX", "LMT", "NOC", "GD",
    "EMR", "ETN", "PH", "ROK", "IR",

    # Energy (10)
    "XOM", "CVX", "COP", "EOG", "SLB",
    "MPC", "VLO", "PSX", "PXD", "HAL",

    # Utilities (10)
    "NEE", "DUK", "SO", "D", "AEP",
    "EXC", "SRE", "XEL", "ES", "WEC",

    # Materials (10)
    "LIN", "APD", "ECL", "SHW", "FCX",
    "NEM", "NUE", "VMC", "MLM", "ALB",

    # Real Estate (5)
    "PLD", "AMT", "CCI", "EQIX", "SPG",
]

START_DATE = "2010-01-01"
END_DATE   = "2024-12-31"

if __name__ == "__main__":
    print(f"Universe: {len(TICKERS)} tickers")
    print(f"Date range: {START_DATE} to {END_DATE}")
    from collections import Counter
    print("\nSector counts defined inline above.")
