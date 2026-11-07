# HRT Algo Arena

Winning entry for the Hudson River Trading Algo Arena challenge. Each session is a made-up stock that trades for 100 bars while made-up news headlines come out. Using bars 0–49 and their headlines, predict the return from the close of bar 49 to the close of bar 59. Entries are scored by the Sharpe ratio of a trader that takes a position equal to the prediction.

**Final submission:** `predictions_improved_all.csv` (model `improved_all`) scored a Sharpe of about **3.6** on the competition's out-of-sample leaderboard, the best result of the project.

## How the model works

1. **Headline drift (base model).** Each type of headline pushes the price at a fixed rate for 10 bars, then stops. The rate depends only on the kind of event. It doesn't depend on the company, sector, region or the numbers in the headline. The forecast is the push still owed by headlines from bars 40–49, plus two corrections: whether recent headlines moved the price more or less than usual, and how much of the first-half move the headlines don't explain.
2. **News count (s2).** Sessions with a lot of news early tend to have less news later.
3. **Refined drift (s3).** Extra inputs for repeated news about the same company, good and bad news at the same time, and the number of headlines still pushing the price.
4. **Industries (s11).** The 80 companies fall into 5 industries, found from which companies appear in the same sessions. Each session is built around one industry, and headlines about companies outside it barely move the price, so the model only uses headlines from the session's main industry.
5. **Bar-based sizing.** A small model predicts how big each session's return will be from the bars 0–49 open/high/low/close. Positions are divided by the square root of that predicted size.
6. **`improved_all`.** Adds three changes: a separate push weight depending on how many same-industry companies are in the session; push rates estimated from in-sample sessions plus out-of-sample first halves (prices only, no answers); and headline counts in the size model, with the forecast fit weighting each session by 1 / its predicted size.

## Benchmark results

Average Sharpe over 5-fold cross-validation with 3 seeds, all on in-sample data (from `benchmark_results.csv`):

| Model | Score |
|---|---|
| `improved_oosrates` | 3.50 |
| `barsized_vol` | 3.50 |
| `barsized_var` | 3.48 |
| `improved_peer` | 3.48 |
| `combined_industry` | 3.48 |
| `s11_industry` | 3.37 |
| `combined_s2_s3_s7` | 3.18 |
| `s2_newscount` | 3.11 |
| `s3_refined` | 3.11 |
| `s7_volscaled` | 3.08 |
| `drift_clip2.0` (base model) | 3.04 |
| `keyword_notebook` (starter idea) | 0.57 |
| `constant_long` | 0.00 |

Strategies that didn't beat the base model: gradient boosting (s5), past price moves (s6), headline words (s8), which companies appear (s9), similar price paths (s10), and other clip levels (s4).

`improved_all`, `improved_newsrisk`, the `windows_*` models and the `_prop` models have no benchmark scores on record.

## Files

### Provided by the challenge

| File | Purpose |
|---|---|
| `algoarena.ipynb` | Starter notebook: task, scoring, data exploration and how to write a submission. |
| `sessions.py` | Organisers' helpers for loading data, running in-sample backtests and writing submissions. |
| `ins_bars.parquet` | In-sample prices, all 100 bars of 10,000 sessions. |
| `ins_headlines.parquet` | In-sample headlines, all 100 bars. |
| `ins_returns.parquet` | In-sample answers: the bar 49→59 return. |
| `oos_bars.parquet`, `oos_headlines.parquet` | Out-of-sample sessions (11000 and up), bars 0–49 only. |

### Code

| File | Purpose |
|---|---|
| `headline_drift_model.py` | Base drift model and the shared helpers (`load_headlines`, `fit_rates`, `features`). Running it writes `predictions.csv`. |
| `backtest_bench.py` | Fixed test setup used to compare models: cross-validation, train on one half and test on the other, less training data, dropped headlines, added noise, subsets of sessions, resampling. Appends results to `benchmark_results.csv`. |
| `strategies.py` | Every candidate strategy (s2–s11, combined, bar-sized, `improved_*`) and `write_predictions` for making upload files. |
| `strategy_windows.py` | Experiment: train on the same problem repeated at later bars of each in-sample session. Never benchmarked. |
| `research/strat2.py` | Looks at whether early news predicts later news. Led to s2. |
| `research/industry.py` | Groups companies into industries and measures how much outside-industry news counts. Led to s11. Writes `industry_map.csv` to the folder it's run from. |
| `research/barsize.py` | Tests which price-range measures predict the size of returns. Led to bar-based sizing. |

### Outputs

| File | Model |
|---|---|
| `benchmark_results.csv` | Log of every benchmark run, one row per model and test. |
| `predictions.csv` | Base drift model |
| `predictions_barsized.csv` | Bar-sized (combined + industry) |
| `predictions_improved_all.csv` | `improved_all`, the final submission |

All upload files have the columns `headline_id,pred`, one row per out-of-sample session.

## Usage

Requires Python 3 with numpy, pandas and scikit-learn (the notebook also needs attrs, rich, tqdm and matplotlib). Run from this folder.

```
# score models (results appended to benchmark_results.csv)
python strategies.py --models drift combined_industry barsized_vol --label "note"
python strategies.py --models drift --quick        # faster: one seed, fewer checks

# write an upload file: fit on all in-sample sessions, predict out-of-sample
python strategies.py --write improved_all predictions_improved_all.csv

# base model only
python headline_drift_model.py
```

Model names for `--models` and `--write` are the keys of `MODELS` in `backtest_bench.py`, `strategies.py` and `strategy_windows.py`.
