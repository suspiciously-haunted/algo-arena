"""HRT Algo Arena: headline-drift model.

Each headline pushes the price at a constant rate for 10 bars, then stops (half weight on
the first and last bar). The rate depends only on the event type, not the company, sector,
region or the numbers in the headline. So the forecast for bars 49 -> 59 is the drift still
owed by headlines published in bars 40-49, plus two small corrections.

Run from this folder:  python headline_drift_model.py   (writes predictions.csv)
"""

from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent
KER = np.array([0.5] + [1.0] * 9 + [0.5])  # drift weight at lags 0..10 after a headline
CLIP_SD = 2.0

SECT = ['supply chain optimization', 'wireless connectivity', 'cloud infrastructure', 'automated logistics',
        'precision manufacturing', 'process automation', 'digital payments', 'enterprise software',
        'renewable storage', 'data analytics']
REG = ['North America', 'Latin America', 'Southeast Asia', 'Central Europe', 'Asia Pacific', 'Middle East',
       'Scandinavia', 'Europe']
PART = ['a leading cloud platform', 'a top-tier research institute', 'a major logistics provider',
        'a global retailer', 'an international consortium', 'a multinational manufacturer',
        'a national infrastructure agency']
CONF = ['Global Innovation Summit', 'Emerging Markets Forum', 'Annual Industry Forum', 'Capital Markets Day',
        'Corporate Strategy Symposium', 'Tech Leadership Conference']
OFF = ['Chief Strategy Officer', 'Chief Operating Officer', 'CTO', 'CFO', 'CEO']


def event_type(rest):
    t = pd.Series(rest).str.replace(r'\$[\d.]+[MB]', '$X', regex=True).str.replace(r'\d+%', 'N%', regex=True)
    for lst, tag in [(SECT, 'SECT'), (REG, 'REG'), (PART, 'PART'), (CONF, 'CONF')]:
        for s in lst:
            t = t.str.replace(s, tag, regex=False)
    for s in OFF:
        t = t.str.replace('^' + s + ' ', 'OFF ', regex=True)
    return t.values


def load_headlines():
    h = pd.concat([pd.read_parquet(D / 'ins_headlines.parquet'), pd.read_parquet(D / 'oos_headlines.parquet')],
                  ignore_index=True)
    w = h.headline.str.split()
    p2, t3 = w.str[:2].str.join(' '), w.str[2]
    # Company names are 2 words, or 3 when the 2-word prefix is always followed by the same word.
    n_third = pd.DataFrame({'p2': p2, 't3': t3}).groupby('p2').t3.nunique()
    three = set(n_third[n_third == 1].index)
    co = np.where(p2.isin(three), w.str[:3].str.join(' '), p2)
    h['ev'] = event_type([s[len(c) + 1:] for s, c in zip(h.headline, co)])
    return h


def closes(name):
    return pd.read_parquet(D / name).pivot(index='session', columns='bar_ix', values='close')


def fit_rates(h, logret):
    """Mean per-bar log drift over lags 1..9 for each event type."""
    s, t = h.sidx.values, h.bar_ix.values
    acc, n = np.zeros(len(h)), np.zeros(len(h))
    for k in range(1, 10):
        m = t + k <= logret.shape[1] - 1
        acc[m] += logret[s[m], t[m] + k]
        n[m] += 1
    return pd.Series(acc / np.maximum(n, 1), index=h.index).groupby(h.ev).mean()


def features(h, rates, C):
    """Per session: remaining drift, size surprise on recent headlines, unexplained first-half move."""
    n, L = len(C), np.log(C)
    h = h[h.bar_ix <= 49]
    s, t, r = h.sidx.values, h.bar_ix.values, h.ev.map(rates).fillna(0).values
    rem = np.array([KER[max(50 - x, 0):].sum() if x >= 40 else 0.0 for x in t])
    done = np.array([KER[:min(49 - x, 10) + 1].sum() for x in t])
    obs = L[s, 49] - np.where(t > 0, L[s, np.maximum(t - 1, 0)], 0)

    def per_session(v):
        return pd.Series(v).groupby(s).sum().reindex(range(n), fill_value=0).values

    mu = per_session(r * rem)
    recent = (t >= 40) & (t <= 48) & (np.abs(r) > 1e-4)
    surprise = per_session(np.where(recent, rem * (obs - done * r) / done, 0))
    unexplained = L[:, 49] - per_session(r * done)
    return np.column_stack([mu, surprise, unexplained])


def main():
    h = load_headlines()
    C_ins, C_oos = closes('ins_bars.parquet'), closes('oos_bars.parquet')
    ret = pd.read_parquet(D / 'ins_returns.parquet').set_index('session').returns.reindex(C_ins.index).values

    h_ins = h[h.session.isin(C_ins.index)].copy()
    h_ins['sidx'] = h_ins.session.map({s: i for i, s in enumerate(C_ins.index)})
    h_oos = h[h.session.isin(C_oos.index)].copy()
    h_oos['sidx'] = h_oos.session.map({s: i for i, s in enumerate(C_oos.index)})

    L = np.log(C_ins.values)
    logret = np.diff(np.concatenate([np.zeros((len(L), 1)), L], 1), axis=1)
    rates = fit_rates(h_ins, logret)

    X_ins, X_oos = features(h_ins, rates, C_ins.values), features(h_oos, rates, C_oos.values)
    centre = X_ins.mean(0)
    coef = np.linalg.lstsq(X_ins - centre, ret, rcond=None)[0]
    p_ins, p_oos = (X_ins - centre) @ coef, (X_oos - centre) @ coef
    cap = CLIP_SD * p_ins.std()
    p_ins, p_oos = np.clip(p_ins, -cap, cap), np.clip(p_oos, -cap, cap)

    pnl = p_ins * ret
    print(f'coefficients {np.round(coef, 4)}')
    print(f'in-sample Sharpe (fit on same data, optimistic): {np.sqrt(252) * pnl.mean() / pnl.std():.3f}')

    out = pd.DataFrame({'headline_id': C_oos.index.astype(np.int64), 'pred': p_oos})
    out.to_csv(D / 'predictions.csv', index=False, lineterminator='\n')
    print(f'wrote {len(out)} rows to {D / "predictions.csv"}')


if __name__ == '__main__':
    main()
