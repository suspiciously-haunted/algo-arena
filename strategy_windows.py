"""Window-trained strategy, scored with backtest_bench.

The task is one example per session: bars 0-49 in, return over bars 49-59 out. In-sample we
see all 100 bars, so every training session also gives the same problem at later anchor bars:
bars T-49..T in, return T -> T+10 out, for T = 49..89. Fitting the linear correction on all of
these stacked windows gives ~40x more examples for the small coefficients (momentum,
volatility, range, price location) that are too noisy to learn from 10k sessions alone.

Features at anchor T copy the current best model (bar-sized combined + industry) with every
bar index shifted by T - 49: main-industry headlines only, drift still owed, size surprise,
unexplained move, news count, repeat / live / conflict news. Targets are market-adjusted like
the real one: each window's return minus the mean over training sessions at the same T.
Sizing is unchanged: clip, then scale by bar-predicted risk at bar 49.

Run from this folder:
  python strategy_windows.py --models windows_core windows_all windows_all_t49 --label "..."
  python strategy_windows.py --write windows_all predictions_windows.csv
"""

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import backtest_bench
from backtest_bench import HALF, MODELS
from headline_drift_model import CLIP_SD, KER, fit_rates
from strategies import bar_risk, company, industry_map, main_industry_only, write_predictions

WIN = HALF + 1  # bars visible to a prediction
HORIZON = 10
KER_CUM = np.concatenate([[0.0], np.cumsum(KER)])  # KER_CUM[k] = KER[:k].sum()

CORE = ['mu', 'surprise', 'unexplained', 'n_news', 'repeat_mu', 'n_live', 'conflict_mu']
PRICE = ['mom3', 'mom10', 'early', 'vol', 'range', 'location']


def window_features(closes, headlines, rates, anchors):
    """One row per (session, anchor T): features from bars T-49..T and headlines in them.

    closes: n x bars log-free close prices; headlines need `sidx`, `bar_ix`, `ev`, `headline`.
    Returns a DataFrame with `sidx`, `T` and the CORE + PRICE columns.
    """
    n = len(closes)
    L = np.log(closes)
    Lpad = np.concatenate([np.zeros((n, 1)), L], 1)  # Lpad[:, b + 1] = log close of bar b; Lpad[:, 0] = open 1
    anchors = np.asarray(anchors)
    A = len(anchors)

    # ---- headlines: one row per (headline, anchor) with the headline inside the window
    hs, hb = headlines.sidx.values, headlines.bar_ix.values
    hr = headlines.ev.map(rates).fillna(0).values
    hco = company(headlines).map(industry_map())
    hco_name = company(headlines).values
    rows = []
    for T in anchors:
        m = (hb <= T) & (hb >= T - HALF)
        rows.append(pd.DataFrame({'sidx': hs[m], 'T': T, 'x': hb[m], 'r': hr[m],
                                  'ind': hco.values[m], 'co': hco_name[m]}))
    h = pd.concat(rows, ignore_index=True)

    # main industry of the window: most companies seen, ties keep all tied industries
    k = h.drop_duplicates(['sidx', 'T', 'co']).groupby(['sidx', 'T', 'ind']).size()
    top = k[k == k.groupby(level=[0, 1]).transform('max')].index
    h = h[pd.MultiIndex.from_frame(h[['sidx', 'T', 'ind']]).isin(top)].reset_index(drop=True)

    s, T, x, r = h.sidx.values, h['T'].values, h.x.values, h.r.values
    lag = T - x
    rem = np.where(lag <= 9, KER_CUM[-1] - KER_CUM[np.minimum(lag + 1, len(KER))], 0.0)
    done = KER_CUM[np.minimum(lag, 10) + 1]
    obs = Lpad[s, T + 1] - Lpad[s, x]
    recent = (lag >= 1) & (lag <= 9) & (np.abs(r) > 1e-4)

    prev = h.sort_values('x', kind='stable').groupby(['sidx', 'T', 'co']).x.shift().reindex(h.index)
    repeat = (x - prev.fillna(-99).values) <= 10

    cell = s * A + np.searchsorted(anchors, T)

    def agg(v):
        return np.bincount(cell, weights=v, minlength=n * A)

    pos, neg = agg(np.where(r > 0, r * rem, 0)), agg(np.where(r < 0, r * rem, 0))
    mu = pos + neg
    base = Lpad[np.arange(n)[:, None], anchors[None, :] - HALF].ravel()  # log close before window
    LT = Lpad[np.arange(n)[:, None], anchors[None, :] + 1].ravel()
    f = pd.DataFrame({
        'sidx': np.repeat(np.arange(n), A), 'T': np.tile(anchors, n),
        'mu': mu,
        'surprise': agg(np.where(recent, rem * (obs - done * r) / done, 0)),
        'unexplained': LT - base - agg(r * done),
        'n_news': agg(np.ones(len(h))),
        'repeat_mu': agg(np.where(repeat, r * rem, 0)),
        'n_live': agg((rem > 0).astype(float)),
        'conflict_mu': mu * ((pos > 0) & (neg < 0)),
    })

    # ---- price shape over the window
    idx = anchors[:, None] + 1 + np.arange(-HALF, 1)[None, :]  # A x 50 columns of Lpad
    W = Lpad[:, idx]  # n x A x 50
    d = np.diff(np.concatenate([base.reshape(n, A, 1), W], 2), axis=2)
    lo, hi = W.min(2), W.max(2)
    f['mom3'] = (W[:, :, -1] - W[:, :, -4]).ravel()
    f['mom10'] = (W[:, :, -1] - W[:, :, -11]).ravel()
    f['early'] = (W[:, :, 29] - base.reshape(n, A)).ravel()  # log(C29/C0) at T = 49
    f['vol'] = d.std(2).ravel()
    f['range'] = (hi - lo).ravel()
    f['location'] = ((W[:, :, -1] - lo) / np.maximum(hi - lo, 1e-9)).ravel()
    return f


class WindowTrained:
    """Linear model fit on stacked windows, sized like barsized_vol."""
    clip_sd = CLIP_SD
    power = 0.5

    def __init__(self, name, cols, anchors=range(HALF, 100 - HORIZON)):
        self.name, self.cols, self.anchors = name, cols, np.array(list(anchors))

    def fit(self, train):
        main = main_industry_only(train)
        L = np.log(main.closes)
        self.rates = fit_rates(main.headlines, np.diff(np.concatenate([np.zeros((len(L), 1)), L], 1), axis=1))

        f = window_features(train.closes, train.headlines, self.rates, self.anchors)
        Ltr = np.log(train.closes)
        fwd = Ltr[:, self.anchors + HORIZON] - Ltr[:, self.anchors]
        y = (fwd - fwd.mean(0)).ravel()
        at49 = (f['T'] == HALF).values
        y[at49] = train.returns[f.sidx.values[at49]]  # the real target where we have it

        Z = f[self.cols].values
        self.centre = Z.mean(0)
        self.scale = Z.std(0) + 1e-12
        Zs = (Z - self.centre) / self.scale
        self.coef = np.linalg.lstsq(Zs, y - y.mean(), rcond=None)[0]

        p49 = Zs[at49] @ self.coef
        self.cap = self.clip_sd * p49.std()
        X = bar_risk(train)
        self.risk_model = LinearRegression().fit(X, train.returns ** 2)
        v = self.risk_model.predict(X)
        self.v_floor, self.v_mid = np.quantile(v, 0.05), np.median(v)

    def predict(self, test):
        f = window_features(test.closes, test.headlines, self.rates, [HALF])
        p = ((f[self.cols].values - self.centre) / self.scale) @ self.coef
        v = np.maximum(self.risk_model.predict(bar_risk(test)), self.v_floor)
        return np.clip(p, -self.cap, self.cap) * (self.v_mid / v) ** self.power


MODELS.update({
    # same features as the current best, coefficients learned from all windows
    'windows_core': lambda: WindowTrained('windows_core', CORE),
    # plus momentum, volatility, range and price location, learned from all windows
    'windows_all': lambda: WindowTrained('windows_all', CORE + PRICE),
    # control: the extended features fit on bar 49 only, to show what the windows add
    'windows_all_t49': lambda: WindowTrained('windows_all_t49', CORE + PRICE, anchors=[HALF]),
})


if __name__ == '__main__':
    if sys.argv[1:2] == ['--write']:
        write_predictions(sys.argv[2], backtest_bench.D / sys.argv[3])
    else:
        backtest_bench.main()
