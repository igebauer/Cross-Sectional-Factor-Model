# Cross-Sectional Equity Factor Model

A from-scratch cross-sectional factor model built on 141 Russell 1000 stocks (2010–2024). The project replicates Fama-French SMB and HML factors, constructs and decomposes a quality alpha signal, and validates out-of-sample using walk-forward backtesting with a 60-month burn-in.

**Primary finding:** ROA (return on assets) as a standalone quality proxy produced the strongest signal in this universe, achieving a Sharpe ratio of **0.409**, **0.8% monthly turnover**, and positive out-of-sample IC (p=0.038).

---

## Pipeline Overview

```
01_fetch_data.py       → prices, market caps, FF5 factors
02_build_returns.py    → monthly returns panel + validation
03_replicate_ff.py     → SMB/HML replication + correlation check
04_build_signal.py     → gross margin, ROA, composite signals
05_ic_analysis.py      → IC, IC IR, decay curves (lags 1–6)
06_walk_forward.py     → L/S portfolios, Sharpe, drawdown, turnover
```

Run the full pipeline with one command:
```bash
python run_pipeline.py
```

Or run each stage individually in numbered order.

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download Fama-French factor data (one-time, ~30 seconds)**

Ken French's data library blocks programmatic downloads, so this step is manual:

- Go to: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip
- Download the ZIP, open it, and save the CSV inside as `data/raw_ff5.csv`

**3. Run the pipeline**
```bash
python run_pipeline.py
```

First run takes ~10–15 minutes (data downloads and caches). Subsequent runs are fast.

---

## Methodology

### Universe
150 Russell 1000 tickers across 10 sectors, 2010–2024. 6 tickers dropped due to delisting (CMA, JNPR, ANSS, K, MDC, PXD), and 3 additional tickers dropped during data alignment (missing market cap data or >10% missing monthly returns), leaving a final universe of 141 stocks.

### Stage 1 — Data
Monthly adjusted close prices via `yfinance`, resampled to month-end. Market cap approximated as shares outstanding × price. Returns winsorized at 1%/99% to remove split-adjustment errors.

### Stage 2 — Factor Replication
SMB (Small Minus Big) constructed via 30/40/30 market cap sorts each month. HML (High Minus Low) constructed using earnings yield (1/PE) as a book-to-market proxy — the best approximation available without CRSP. SMB correlation with official FF: **r=0.633**. HML: **r=0.679**. Lower correlations are expected given the large-cap universe and E/P proxy.

### Stage 3 — Quality Signal Decomposition
Three signals constructed from `yfinance` fundamentals:

| Signal | Construction |
|---|---|
| Gross Margin | `grossMargins` z-scored cross-sectionally |
| ROA | `returnOnAssets` z-scored cross-sectionally |
| Composite | Equal-weight sum of gross margin + ROA z-scores |

All signals are static snapshots (current trailing values), rank-normalized each month to cross-sectional z-scores.

### Stage 4 — IC Analysis
Spearman rank correlation between signal at time *t* and forward returns at *t+1*. IC decay computed at lags 1–6 months.

| Signal | IC Mean | IC IR | % Positive | p-value |
|---|---|---|---|---|
| Gross Margin | 0.0151 | 0.080 | 55.6% | 0.285 |
| ROA | 0.0300 | 0.156 | 60.1% | **0.038** |
| Composite | 0.0270 | 0.138 | 59.0% | 0.068 |

ROA is the only statistically significant signal (p<0.05). The composite underperforms ROA alone due to high cross-sectional correlation between components (r=0.43), causing dilution rather than diversification.

### Stage 5 — Walk-Forward Validation
Long top 30% / short bottom 30% by signal, equal-weighted within each leg. 60-month burn-in before trading begins. Turnover computed as fraction of portfolio that changes each month.

| Signal | Ann. Return | Sharpe | Max Drawdown | Win Rate | Avg Turnover |
|---|---|---|---|---|---|
| Gross Margin | 3.75% | 0.202 | -18.26% | 60.2% | 0.8% |
| ROA | **5.90%** | **0.409** | **-13.95%** | **61.9%** | **0.8%** |
| Composite | 5.27% | 0.318 | -20.75% | 58.5% | 0.8% |
| Market | 14.47% | 0.882 | -24.83% | 68.2% | — |

All results are out-of-sample. No transaction costs applied — at 0.8% monthly turnover, real-world drag would be minimal.

---

## Key Findings

**ROA drives the quality premium in this universe.** Decomposition showed that ROA alone outperforms both gross margin and the composite on Sharpe ratio (0.409 vs 0.202 and 0.318). The high cross-sectional correlation between components (r=0.43) means adding gross margin to ROA dilutes rather than diversifies the signal.

**The signal is economically positive but modest relative to the strong long-only equity environment of 2010–2024.** The ROA L/S portfolio achieves a positive Sharpe with low drawdown and low turnover, but the market itself (Sharpe 0.882 in this period) significantly outperforms, consistent with the 2010–2024 period being an unusually strong bull market for large-cap US equities.

---

## Limitations

**Survivorship bias** — Universe uses current Russell 1000 constituents, not point-in-time. Delisted and acquired firms are excluded. A production implementation would use CRSP constituent history.

**Static fundamental data** — Gross margin and ROA are pulled as current trailing values from `yfinance`, not historical point-in-time. A real factor model would use quarterly Compustat data with a reporting lag.

**Market cap proxy** — Shares outstanding from the latest filing multiplied by current price. Not historically accurate.

**SMB/HML replication** — Constrained to large/mid-cap universe; the size effect is concentrated in micro-caps. True HML requires point-in-time book value from CRSP/Compustat.

**No transaction costs** — No transaction costs modeled. Given low turnover (0.8% monthly), implementation costs are likely modest but nonzero.

---

## References

- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.
- Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1–22.
- Novy-Marx, R. (2013). The other side of value: The gross profitability premium. *Journal of Financial Economics*, 108(1), 1–28.
- Ken French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

---

## Tech Stack

Python · pandas · NumPy · statsmodels · scipy · matplotlib · yfinance
