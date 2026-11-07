"""HRT Algo Arena: fixed benchmark for comparing models across scenarios.

Every scenario uses fixed seeds and only in-sample data, so a model scores the same way on
every run and different models are compared on identical splits. Predictions only ever see
bars 0-49 and headlines up to bar 49; training gets full in-sample sessions.

Scenarios
  cv5_seed{0,1,2}      5-fold cross-validation, three different fold assignments
  ordered_a->b         train on one half of the session numbers, test on the other (both ways)
  train_frac_{x}       5-fold CV training on only a fraction of each training fold
  headline_dropout_20  5-fold CV with 20% of test headlines randomly removed
  pred_noise_50        cv5_seed0 predictions plus noise at 50% of their spread
  slice:*              cv5_seed0 predictions scored on subsets (recent news, volatility,
                       first-half move)
  bootstrap_p*         percentiles of the cv5_seed0 Sharpe over 1000 resamples of sessions
The headline number, BENCHMARK, is the mean Sharpe of cv5_seed0..2.

Usage (from this folder)
  python backtest_bench.py                       # all registered models
  python backtest_bench.py --models drift keyword --label "try clip 1.5"
  python backtest_bench.py --quick               # one CV seed, no learning curve, fewer resamples
Results are printed and appended to benchmark_results.csv (one row per model x scenario).

Adding a model: write a class with a `name` attribute, `fit(train: Split)` and
`predict(test: Split) -> np.ndarray` (one value per row of test.sessions), then add it to MODELS.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from headline_drift_model import CLIP_SD, D, features, fit_rates, load_headlines

RESULTS = D / 'benchmark_results.csv'
HALF = 49
K = 5
CV_SEEDS = (0, 1, 2)
TRAIN_FRACS = (0.1, 0.25, 0.5)
DROPOUT = 0.2
NOISE = 0.5
N_BOOT = 1000


@dataclass
class Split:
    sessions: np.ndarray
    closes: np.ndarray  # n x 100 for training, n x 50 (bars 0-49) for predicting
    headlines: pd.DataFrame  # includes `ev` (event type) and `sidx` (row in closes)
    returns: np.ndarray | None  # None when predicting


class Data:
    def __init__(self):
        closes = pd.read_parquet(D / 'ins_bars.parquet').pivot(index='session', columns='bar_ix', values='close')
        self.sessions = closes.index.values
        self.C = closes.values
        self.n = len(self.sessions)
        self.ret = (pd.read_parquet(D / 'ins_returns.parquet').set_index('session').returns
                    .reindex(self.sessions).values)
        h = load_headlines()
        self.h = h[h.session.isin(self.sessions)].reset_index(drop=True)

    def view(self, idx, train, drop_frac=0.0, rng=None):
        idx = np.sort(np.asarray(idx))
        pos = pd.Series(np.arange(len(idx)), index=self.sessions[idx])
        h = self.h[self.h.session.isin(pos.index)]
        if not train:
            h = h[h.bar_ix <= HALF]
            if drop_frac:
                h = h[rng.random(len(h)) >= drop_frac]
        h = h.assign(sidx=h.session.map(pos).values).reset_index(drop=True)
        C = self.C[idx] if train else self.C[idx, :HALF + 1]
        return Split(self.sessions[idx], C, h, self.ret[idx] if train else None)


# ---------------------------------------------------------------- models

class ConstantLong:
    name = 'constant_long'

    def fit(self, train):
        pass

    def predict(self, test):
        return np.ones(len(test.sessions))


class KeywordNotebook:
    """The starter idea from the notebook: +1 per 'secures', -1 per 'decline'."""
    name = 'keyword_notebook'

    def fit(self, train):
        pass

    def predict(self, test):
        hl = test.headlines
        score = hl.headline.str.contains('secures').astype(float) - hl.headline.str.contains('decline').astype(float)
        return score.groupby(hl.sidx).sum().reindex(range(len(test.sessions)), fill_value=0).values


class HeadlineDrift:
    """Same model as headline_drift_model.py, refit on each training split."""

    def __init__(self, clip_sd=CLIP_SD):
        self.clip_sd = clip_sd
        self.name = f'drift_clip{clip_sd}' if clip_sd else 'drift_noclip'

    def fit(self, train):
        L = np.log(train.closes)
        logret = np.diff(np.concatenate([np.zeros((len(L), 1)), L], 1), axis=1)
        self.rates = fit_rates(train.headlines, logret)
        X = features(train.headlines, self.rates, train.closes)
        self.centre = X.mean(0)
        self.coef = np.linalg.lstsq(X - self.centre, train.returns, rcond=None)[0]
        self.cap = self.clip_sd * ((X - self.centre) @ self.coef).std() if self.clip_sd else np.inf

    def predict(self, test):
        p = (features(test.headlines, self.rates, test.closes) - self.centre) @ self.coef
        return np.clip(p, -self.cap, self.cap)


MODELS = {
    'constant': ConstantLong,
    'keyword': KeywordNotebook,
    'drift': HeadlineDrift,
    'drift_noclip': lambda: HeadlineDrift(None),
}


# ---------------------------------------------------------------- scoring

def metrics(p, r):
    pnl = p * r
    sd = pnl.std()
    return dict(
        sharpe=np.sqrt(252) * pnl.mean() / sd if sd > 0 else 0.0,
        corr=np.corrcoef(p, r)[0, 1] if p.std() > 0 else 0.0,
        hit=np.mean(np.sign(p) == np.sign(r)),
        n=len(r),
    )


def run_folds(data, make, folds, train_frac=1.0, drop_frac=0.0, seed=0):
    """Out-of-fold predictions for a fold assignment; returns (preds, per-fold Sharpes)."""
    rng = np.random.default_rng(seed)
    p = np.zeros(data.n)
    per_fold = []
    for f in np.unique(folds):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        if train_frac < 1:
            tr = rng.choice(tr, int(len(tr) * train_frac), replace=False)
        model = make()
        model.fit(data.view(tr, train=True))
        p[te] = model.predict(data.view(te, train=False, drop_frac=drop_frac, rng=rng))
        per_fold.append(metrics(p[te], data.ret[te])['sharpe'])
    return p, per_fold


def cv_folds(n, seed):
    return np.random.default_rng(seed).permutation(n) % K


def slices(data):
    """Boolean masks for the slice scenarios, computed from first-half data only."""
    h = data.h[data.h.bar_ix <= HALF]
    recent = (h[h.bar_ix >= 40].groupby('session').size()
              .reindex(data.sessions, fill_value=0).values)
    L = np.log(data.C[:, :HALF + 1])
    vol = np.diff(L, axis=1).std(1)
    move = L[:, HALF]
    out = {
        'slice:recent_news_0': recent == 0,
        'slice:recent_news_1': recent == 1,
        'slice:recent_news_2+': recent >= 2,
    }
    for name, x in [('vol', vol), ('first_half_move', move)]:
        lo, hi = np.quantile(x, [1 / 3, 2 / 3])
        out[f'slice:{name}_low'] = x <= lo
        out[f'slice:{name}_mid'] = (x > lo) & (x <= hi)
        out[f'slice:{name}_high'] = x > hi
    return out


def benchmark(data, key, quick=False):
    make = MODELS[key]
    rows = []

    def add(scenario, p, r, fold_sd=np.nan):
        rows.append(dict(scenario=scenario, **metrics(p, r), fold_sd=fold_sd))

    seeds = CV_SEEDS[:1] if quick else CV_SEEDS
    base = None
    for s in seeds:
        p, pf = run_folds(data, make, cv_folds(data.n, s), seed=s)
        add(f'cv5_seed{s}', p, data.ret, np.std(pf))
        if base is None:
            base = p

    # Session-number order: does what's learned on one block carry to the other?
    first = (np.arange(data.n) < data.n // 2).astype(int)
    for train_block, test_block, label in [(0, 1, 'ordered_a->b'), (1, 0, 'ordered_b->a')]:
        tr, te = np.where(first == 1 - train_block)[0], np.where(first == 1 - test_block)[0]
        model = make()
        model.fit(data.view(tr, train=True))
        add(label, model.predict(data.view(te, train=False)), data.ret[te])

    if not quick:
        for frac in TRAIN_FRACS:
            p, pf = run_folds(data, make, cv_folds(data.n, 0), train_frac=frac, seed=0)
            add(f'train_frac_{frac}', p, data.ret, np.std(pf))

    p, pf = run_folds(data, make, cv_folds(data.n, 0), drop_frac=DROPOUT, seed=0)
    add(f'headline_dropout_{int(DROPOUT * 100)}', p, data.ret, np.std(pf))

    rng = np.random.default_rng(7)
    add(f'pred_noise_{int(NOISE * 100)}', base + rng.normal(0, NOISE * base.std(), data.n), data.ret)

    for name, mask in slices(data).items():
        add(name, base[mask], data.ret[mask])

    rng = np.random.default_rng(123)
    pnl = base * data.ret
    boot = []
    for _ in range(200 if quick else N_BOOT):
        x = pnl[rng.integers(0, data.n, data.n)]
        boot.append(np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0.0)
    for q in (2.5, 50, 97.5):
        rows.append(dict(scenario=f'bootstrap_p{q}', sharpe=np.percentile(boot, q), n=data.n))

    df = pd.DataFrame(rows)
    df.insert(0, 'model', make().name)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS))
    ap.add_argument('--label', default='', help='free-text note stored with this run')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--no-save', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    data = Data()
    res = pd.concat([benchmark(data, m, args.quick) for m in args.models], ignore_index=True)
    res.insert(0, 'label', args.label)
    res.insert(0, 'run_at', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    res.insert(2, 'quick', args.quick)

    pd.set_option('display.width', 200)
    table = res.pivot(index='scenario', columns='model', values='sharpe').reindex(res.scenario.unique())
    print('Sharpe by scenario\n')
    print(table.round(2).to_string())
    cv = res[res.scenario.str.startswith('cv5_')].groupby('model').sharpe.mean()
    print('\nBENCHMARK (mean cv5 Sharpe):')
    print(cv.round(3).sort_values(ascending=False).to_string())

    if not args.no_save:
        res.to_csv(RESULTS, mode='a', header=not RESULTS.exists(), index=False)
        print(f'\nappended {len(res)} rows to {RESULTS}')
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
