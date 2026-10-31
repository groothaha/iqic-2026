# IQIC 2026: low-volatility stocks under CAPM-based scoring

Code and records behind my strategy declaration report for the International Quantitative Investing Challenge 2026 (STEMise; StockTrak session "IQIC 2026"; trading 2–20 November 2026). The report is [`report/report.pdf`](report/report.pdf).

The design: using data to the 30 October 2026 close, take the S&P 1500 stocks that filed an earnings release (8-K Item 2.02) between 21 September and 29 October, rank them by 252-day volatility, and buy the 30 least volatile in equal dollar amounts, with at most six per sector. Hold them to the end, with two exit rules: sell a stock that closes 15% below its purchase price, and sell half of every position if the account closes 8% below $100,000 (once). On 30 October the selected stocks had a one-year beta of −0.01, below every backtest window, so I also buy Micro E-mini S&P 500 futures to bring portfolio beta to about 0.75 (`DECISION_beta_overlay.md`). No short sales and no option selling.

## Contents

| Path | What it holds |
|---|---|
| `report/` | Report source (`report.html`), PDF, and the script for Figure 1 |
| `PREREG*.md` | Pre-registrations, each written before its test was run (in Korean) |
| `RESULTS*.md` | Results and the decision taken (in Korean) |
| `DECISION_beta_overlay.md` | The 31 October decision to add futures, applying the beta test to the final selection (in Korean) |
| `scripts/` | All code |
| `out/` | Derived outputs behind the numbers in the report |

## Tests, in the order they were run

| Test | Pre-registration | Results | Code (`scripts/`) | Output (`out/`) | Report |
|---|---|---|---|---|---|
| Opportunistic insider purchases | `PREREG.md` | `RESULTS_insider.md` | `fetch_insider.py`, `fetch_splits.py`, `insider_test.py` | `monthly.parquet`, `summary.csv`, `alphas.csv`, `splits.parquet` | §4 |
| Contest simulator, 13 candidate portfolios (exploratory, not pre-registered) | – | – | `contest_sim.py`, `crowding_sim.py`, `floor_check.py` | `contest_sim.parquet`, `crowding_sim.parquet` | §2.2, §2.5 |
| Earnings filter and post-earnings drift | `PREREG_earnings.md` | `RESULTS_earnings.md` | `fetch_8k.py`, `earnings_test.py` | `ea_dates.parquet`, `earnings_test.parquet`, `earnings_mech.parquet` | §1.4, §4 |
| Buying out-of-the-money SPY calls | `PREREG_options.md` | `RESULTS_options.md` | `options_test.py` | `options_test.parquet` | §4 |
| Holding SPY/QQQ only overnight | `PREREG_overnight.md` | `RESULTS_overnight.md` | `overnight_test.py` | `overnight_test.parquet` | §4 |
| Sector cap and exit rules | `PREREG_risk.md` | `RESULTS_risk.md` | `fetch_sic.py`, `risk_test.py` | `sic.parquet`, `risk_test.parquet` | §2.3, §3 |
| Index futures and call selling | `PREREG_derivatives.md` | `RESULTS_derivatives.md` | `deriv_test.py` | `deriv_test.parquet`, `deriv_test.log` | §4 |
| Score against beta | `PREREG_beta.md` | `RESULTS_beta.md` | `beta_sweep.py`, `../report/make_fig1.py` | `beta_sweep.parquet`, `beta_sweep.log` | §1.5, Figure 1 |
| Two readings of the scoring formulas | `PREREG_formula.md` | `RESULTS_formula.md` | `formula_test.py` | `formula_test.parquet`, `formula_test.log` | §2.4 |
| Futures for the final selection (applies the beta test; not a new test) | – | `DECISION_beta_overlay.md` | `beta_overlay.py` | `beta_overlay_2026-10-30.csv` | §2.3, §4, §5 |

In the outputs, the two simulated fields have Korean names: `학생형` is Field A (student-like) and `숙련형` is Field B (experienced).

## When each record was written

File modification times on my machine (KST, 2026), copied before the first commit.

| Test | Pre-registration | Results |
|---|---|---|
| Insider purchases | 26 Oct 01:45 | 26 Oct 01:53 |
| Earnings filter | 26 Oct 06:24 | 26 Oct 06:30 |
| Buying calls | 26 Oct 07:09 | 26 Oct 07:10 |
| Overnight | 26 Oct 07:16 | 26 Oct 07:18 |
| Risk rules | 26 Oct 11:04 | 26 Oct 11:08 |
| Futures and call selling | 26 Oct 18:58 | 28 Oct 09:31 (edited, see below) |
| Beta curve | 26 Oct 22:04 | 26 Oct 22:05 |
| Scoring formulas | 28 Oct 09:26 | 28 Oct 09:28 |
| Futures decision | – | 31 Oct 20:53 (after the final selection at 06:36) |

These times are not independent proof of order: the first commit of this repository came after all the tests. `RESULTS_derivatives.md` was edited on 28 October to record the decision to exclude call selling. In this public copy, one parenthetical in `PREREG_overnight.md` that referred to an unrelated private project was removed. No other text was changed.

## Final selection

`scripts/live_select.py` produces the 30 stocks bought on 2 November: `out/live_e0_2026-10-30.csv` (prices as of the 30 October close; added after that close). `scripts/beta_overlay.py` then sets the futures position: `out/beta_overlay_2026-10-30.csv` (2 contracts). `out/live_e0_2026-10-23.csv` is the 23 October preview cited in §3. During trading I will add the order log, with the rule behind each order.

## Running the code

- Python 3.14 with pandas 2.2, numpy 2.4, scipy 1.17, statsmodels 0.14, pyarrow, requests, lxml, yfinance 0.2.66 and matplotlib 3.10.
- Requests to SEC EDGAR need a contact string: `export SEC_USER_AGENT="Your Name you@example.com"`.
- Scripts read local data from the paths at the top of each file. Raw price and option data are not included, because the Yahoo Finance and optionsDX terms do not allow redistribution:

| Data | Source | Expected location |
|---|---|---|
| Daily closes and volumes for current S&P 1500 constituents, 2005–2024; 13-week T-bill and other market series | Yahoo Finance via yfinance, built in a separate project | `~/us_quant2/data/raw/{closes,volumes,macro}.parquet` |
| SPY and QQQ daily prices | Yahoo Finance | `~/AAAAAAbacktest/tier_a/` |
| Insider transactions, 2009–2026 | SEC Insider Transactions Data Sets (`fetch_insider.py`) | `~/data/us/sec_insider/ps/` |
| Monthly prices and market capitalization inputs for the insider test | built in a separate project | `~/fundamental_ml/data/` |
| SPY option chains, end of day, 2010–2023 | optionsDX (licensed) | `~/options_data/data/optionsdx/parquet/` |
| 8-K Item 2.02 dates and SIC codes | SEC EDGAR submissions API (`fetch_8k.py`, `fetch_sic.py`) | `out/` (included) |

Every table in the report can be checked from `out/` without the raw data. For example, Figure 1:

```python
import pandas as pd
d = pd.read_parquet("out/beta_sweep.parquet")
print(d.groupby(["field", "arm"]).score.mean().unstack(0))
```

## Limitations

See §6 of the report. The backtest universe is today's S&P 1500, so it has survivorship bias, and both simulated fields of participants are assumptions.
