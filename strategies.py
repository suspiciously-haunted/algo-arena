"""Candidate strategies, each scored with backtest_bench.

Every model is the headline-drift model plus one strategy's addition, so its benchmark score
compares directly with `drift_clip2.0`. Run from this folder:

  python strategies.py --models drift s2_newscount s3_refined ...   (same flags as backtest_bench)
  python strategies.py --write combined predictions_combined.csv    (fit on all in-sample, write OOS)
"""

import sys
from dataclasses import replace
from functools import cache

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import NearestNeighbors

import backtest_bench
from backtest_bench import HALF, MODELS, HeadlineDrift, Split
from headline_drift_model import CLIP_SD, KER, D, features, fit_rates, load_headlines


def headline_info(split, rates):
    """First-half headlines with their drift rate, weight left in bars 50-59 and weight done."""
    h = split.headlines[split.headlines.bar_ix <= HALF]
    t = h.bar_ix.values
    r = h.ev.map(rates).fillna(0).values
    rem = np.array([KER[max(50 - x, 0):].sum() if x >= 40 else 0.0 for x in t])
    return h, t, r, rem


def per_session(split, h, v):
    return pd.Series(v).groupby(h.sidx.values).sum().reindex(range(len(split.sessions)), fill_value=0).values


def first_half_logs(split):
    return np.log(split.closes[:, :HALF + 1])


class DriftPlus:
    """Drift model with extra linear features from `extra`, an optional residual model, and sizing."""
    name = 'drift_plus'
    clip_sd = CLIP_SD

    def extra(self, split):
        return np.empty((len(split.sessions), 0))

    def design(self, split):
        return np.column_stack([features(split.headlines, self.rates, split.closes), self.extra(split)])

    def fit_residual(self, train, resid):
        pass

    def predict_residual(self, test):
        return 0.0

    def size(self, p, split):
        return np.clip(p, -self.cap, self.cap)

    def fit(self, train):
        L = np.log(train.closes)
        logret = np.diff(np.concatenate([np.zeros((len(L), 1)), L], 1), axis=1)
        self.rates = fit_rates(train.headlines, logret)
        Z = self.design(train)
        self.centre = Z.mean(0)
        self.coef = np.linalg.lstsq(Z - self.centre, train.returns, rcond=None)[0]
        base = (Z - self.centre) @ self.coef
        self.fit_residual(train, train.returns - base)
        p = base + self.predict_residual(train)
        self.cap = self.clip_sd * p.std()

    def predict(self, test):
        p = (self.design(test) - self.centre) @ self.coef + self.predict_residual(test)
        return self.size(p, test)


# 2. Unseen news: busy first halves are followed by quieter second halves.
class S2NewsCount(DriftPlus):
    name = 's2_newscount'

    def extra(self, split):
        h, *_ = headline_info(split, self.rates)
        return per_session(split, h, np.ones(len(h)))[:, None]


# 3. Finer drift: repeated company news, conflicting news, number of live headlines.
class S3Refined(DriftPlus):
    name = 's3_refined'

    def extra(self, split):
        h, t, r, rem = headline_info(split, self.rates)
        co = h.headline.str.split().str[:2].str.join(' ').values
        key = pd.Series(h.sidx.values.astype(str)) + '|' + co
        prev_bar = pd.Series(t).groupby(key.values).shift()
        repeat = (t - prev_bar.fillna(-99).values) <= 10
        live = rem > 0
        pos = per_session(split, h, np.where(r > 0, r * rem, 0))
        neg = per_session(split, h, np.where(r < 0, r * rem, 0))
        mu = pos + neg
        conflict = (pos > 0) & (neg < 0)
        return np.column_stack([
            per_session(split, h, np.where(repeat, r * rem, 0)),
            per_session(split, h, live.astype(float)),
            mu * conflict,
        ])


# 4. Position sizing: cap level.
class S4Clip(HeadlineDrift):
    def __init__(self, clip_sd):
        super().__init__(clip_sd)
        self.name = f's4_clip{clip_sd}'


# 5. Gradient boosting over the drift features plus a few extras.
class S5Boosting(DriftPlus):
    name = 's5_boosting'

    def gb_features(self, split):
        h, t, r, rem = headline_info(split, self.rates)
        L = first_half_logs(split)
        return np.column_stack([
            features(split.headlines, self.rates, split.closes),
            per_session(split, h, np.ones(len(h))),
            per_session(split, h, np.where(r > 0, r * rem, 0)),
            per_session(split, h, np.where(r < 0, r * rem, 0)),
            np.diff(L, axis=1).std(1),
            L[:, HALF] - L[:, HALF - 5],
            L[:, HALF] - L[:, HALF - 20],
        ])

    def fit(self, train):
        L = np.log(train.closes)
        logret = np.diff(np.concatenate([np.zeros((len(L), 1)), L], 1), axis=1)
        self.rates = fit_rates(train.headlines, logret)
        self.gb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                                min_samples_leaf=200, l2_regularization=1.0, random_state=0)
        X = self.gb_features(train)
        self.gb.fit(X, train.returns)
        p = self.gb.predict(X)
        self.offset = p.mean()
        self.cap = self.clip_sd * p.std()

    def predict(self, test):
        return np.clip(self.gb.predict(self.gb_features(test)) - self.offset, -self.cap, self.cap)


# 6. Historical price action: recent returns over several lookbacks.
class S6PriceAction(DriftPlus):
    name = 's6_priceaction'

    def extra(self, split):
        L = first_half_logs(split)
        return np.column_stack([L[:, HALF] - L[:, HALF - k] for k in (1, 5, 10, 20)])


# 7. Volatility: scale positions by 1 / first-half volatility.
class S7VolScaled(DriftPlus):
    name = 's7_volscaled'

    def size(self, p, split):
        vol = np.diff(first_half_logs(split), axis=1).std(1)
        return np.clip(p, -self.cap, self.cap) * np.median(vol) / vol


# 8. Text: ridge on words of live headlines (weighted by drift still to come), fit to residuals.
class S8Text(DriftPlus):
    name = 's8_text'

    def text_matrix(self, split):
        h, t, r, rem = headline_info(split, self.rates)
        live = rem > 0
        W = self.vec.transform(h.headline.values[live]).multiply(rem[live][:, None]).tocsr()
        S = np.zeros((len(split.sessions), W.shape[1]))
        np.add.at(S, h.sidx.values[live], W.toarray())
        return S

    def fit_residual(self, train, resid):
        self.vec = CountVectorizer(ngram_range=(1, 2), min_df=20, token_pattern=r'[A-Za-z$%]+')
        self.vec.fit(train.headlines.headline.values)
        self.ridge = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(self.text_matrix(train), resid)

    def predict_residual(self, split):
        return self.ridge.predict(self.text_matrix(split))


# 9. Basket / company exposure: ridge on which companies appear, fit to residuals.
class S9Company(DriftPlus):
    name = 's9_company'

    def company_matrix(self, split):
        h = split.headlines[split.headlines.bar_ix <= HALF]
        co = h.headline.str.split().str[:2].str.join(' ')
        M = np.zeros((len(split.sessions), len(self.companies)))
        idx = co.map(self.companies)
        ok = idx.notna().values
        M[h.sidx.values[ok], idx.values[ok].astype(int)] = 1.0
        return M

    def fit_residual(self, train, resid):
        co = train.headlines.headline.str.split().str[:2].str.join(' ').unique()
        self.companies = pd.Series(np.arange(len(co)), index=np.sort(co))
        self.ridge = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(self.company_matrix(train), resid)

    def predict_residual(self, split):
        return self.ridge.predict(self.company_matrix(split))


# 10. Pairs / cointegration analogue: match each price path to similar training paths.
class S10PathMatch(DriftPlus):
    name = 's10_pathmatch'
    k = 200

    def paths(self, split):
        L = first_half_logs(split)
        return L[:, HALF - 19:] - L[:, [HALF - 20]]

    def fit_residual(self, train, resid):
        P = self.paths(train)
        self.nn = NearestNeighbors(n_neighbors=self.k + 1).fit(P)
        self.train_resid = resid
        _, idx = self.nn.kneighbors(P)
        own = self.train_resid[idx[:, 1:]].mean(1)  # leave self out
        self.beta = np.dot(own - own.mean(), resid) / max(np.dot(own - own.mean(), own - own.mean()), 1e-18)
        self.fitting = True

    def predict_residual(self, split):
        if getattr(self, 'fitting', False):
            self.fitting = False
            return 0.0
        _, idx = self.nn.kneighbors(self.paths(split), n_neighbors=self.k)
        return self.beta * self.train_resid[idx].mean(1)


MODELS.update({
    's2_newscount': S2NewsCount,
    's3_refined': S3Refined,
    's4_clip1.0': lambda: S4Clip(1.0),
    's4_clip1.5': lambda: S4Clip(1.5),
    's4_clip2.5': lambda: S4Clip(2.5),
    's4_clip3.0': lambda: S4Clip(3.0),
    's5_boosting': S5Boosting,
    's6_priceaction': S6PriceAction,
    's7_volscaled': S7VolScaled,
    's8_text': S8Text,
    's9_company': S9Company,
    's10_pathmatch': S10PathMatch,
})



# Combination of the strategies that beat the baseline: 2 + 3 features, 7 sizing.
class Combined(S7VolScaled):
    name = 'combined_s2_s3_s7'

    def extra(self, split):
        return np.column_stack([S2NewsCount.extra(self, split), S3Refined.extra(self, split)])


MODELS['combined'] = Combined


# 11. Industries: each session is built around one industry, and headlines about companies
# outside it barely move the price. Industries come from which companies share sessions
# (headlines only, no returns), so out-of-sample sessions can be tagged the same way.
def company(h):
    return h.headline.str.split().str[:2].str.join(' ')


@cache
def industry_map():
    """Company -> industry (0-4), by spectral clustering of how often companies share a session."""
    h = load_headlines()
    pres = pd.crosstab(h.session, company(h)).gt(0).astype(float)
    occ = pres.values.T @ pres.values
    np.fill_diagonal(occ, 0)
    p = pres.values.mean(0)
    lift = occ / (np.outer(p, p) * len(pres))
    labels = SpectralClustering(5, affinity='precomputed', random_state=0).fit_predict(lift)
    return pd.Series(labels, index=pres.columns)


def main_industry_only(split):
    """Keep headlines from the session's main industry: the one with most companies seen by
    bar 49 (ties keep every tied industry)."""
    h = split.headlines
    co = company(h)
    ind = co.map(industry_map())
    seen = pd.DataFrame({'sidx': h.sidx.values, 'co': co.values, 'ind': ind.values})[h.bar_ix.values <= HALF]
    counts = seen.drop_duplicates(['sidx', 'co']).groupby(['sidx', 'ind']).size()
    top = counts[counts == counts.groupby(level=0).transform('max')].index
    keep = pd.MultiIndex.from_arrays([h.sidx.values, ind.values]).isin(top)
    return replace(split, headlines=h[keep].reset_index(drop=True))


class MainIndustryOnly:
    """Mixin: fit and predict using main-industry headlines only."""

    def fit(self, train):
        super().fit(main_industry_only(train))

    def predict(self, test):
        return super().predict(main_industry_only(test))


class S11Industry(MainIndustryOnly, DriftPlus):
    name = 's11_industry'


class CombinedIndustry(MainIndustryOnly, Combined):
    name = 'combined_industry'


MODELS['s11_industry'] = S11Industry
MODELS['combined_industry'] = CombinedIndustry


# 12. Bar-based sizing: predict each session's return variance from bars 0-49 (open/high/low/
# close) and divide positions by predicted risk. Replaces strategy 7's close-only vol scaling.
@cache
def ohlc():
    bars = pd.concat([pd.read_parquet(D / 'ins_bars.parquet'), pd.read_parquet(D / 'oos_bars.parquet')])
    bars = bars[bars.bar_ix <= HALF]
    return {c: bars.pivot(index='session', columns='bar_ix', values=c) for c in ('open', 'high', 'low', 'close')}


def bar_risk(split):
    """Per session: Garman-Klass variance over 50 and 10 bars, squared opening gaps, close-close variance."""
    O, H, Lo, C = (np.log(ohlc()[c].loc[split.sessions].values) for c in ('open', 'high', 'low', 'close'))
    prev = np.concatenate([np.zeros((len(C), 1)), C[:, :-1]], 1)
    gk = 0.5 * (H - Lo) ** 2 - (2 * np.log(2) - 1) * (C - O) ** 2
    return np.column_stack([gk.mean(1), gk[:, -10:].mean(1), ((O - prev) ** 2).mean(1), ((C - prev) ** 2).mean(1)])


class BarSized(CombinedIndustry):
    power = 1.0  # 1 = divide by predicted variance, 0.5 = by predicted volatility

    def fit(self, train):
        super().fit(train)
        X = bar_risk(train)
        self.risk_model = LinearRegression().fit(X, train.returns ** 2)
        v = self.risk_model.predict(X)
        self.v_floor, self.v_mid = np.quantile(v, 0.05), np.median(v)

    def size(self, p, split):
        v = np.maximum(self.risk_model.predict(bar_risk(split)), self.v_floor)
        return np.clip(p, -self.cap, self.cap) * (self.v_mid / v) ** self.power


class BarSizedVol(BarSized):
    name = 'barsized_vol'
    power = 0.5


class BarSizedVar(BarSized):
    name = 'barsized_var'
    power = 1.0


MODELS['barsized_vol'] = BarSizedVol
MODELS['barsized_var'] = BarSizedVar


class BarSizedOutsider(BarSizedVol):
    """Bar-sized combined + industry, plus remaining drift from odd-one-out headlines as its own
    input, so the fit decides how much outsider news counts instead of dropping it."""
    name = 'barsized_vol_outsider'

    def fit(self, train):
        self.full = train
        super().fit(train)

    def predict(self, test):
        self.full = test
        return super().predict(test)

    def extra(self, split):
        full = self.full.headlines
        key = ['session', 'bar_ix', 'headline']
        outsider = full[~pd.MultiIndex.from_frame(full[key]).isin(pd.MultiIndex.from_frame(split.headlines[key]))]
        h, t, r, rem = headline_info(replace(self.full, headlines=outsider.reset_index(drop=True)), self.rates)
        return np.column_stack([super().extra(split), per_session(self.full, h, r * rem)])


MODELS['barsized_vol_outsider'] = BarSizedOutsider


def proportional(make):
    """Same model with position exactly proportional to the forecast: no cap, no risk scaling."""
    def build():
        m = make()
        m.name = m.name + '_prop'
        fit = m.fit

        def fit_prop(train):
            fit(train)
            m.cap = np.inf

        m.fit = fit_prop
        m.size = lambda p, split: p
        return m
    return build


# Sizing strategies (s4 caps, s7 vol scaling, bar sizing) collapse into these once proportional.
for key in ['drift', 's2_newscount', 's3_refined', 's5_boosting', 's6_priceaction', 's8_text',
            's9_company', 's10_pathmatch', 'combined', 's11_industry', 'combined_industry']:
    MODELS[key + '_prop'] = proportional(MODELS[key])


@cache
def oos_split():
    """Out-of-sample sessions as a prediction Split: bars 0-49 and their headlines, no returns."""
    C = pd.read_parquet(D / 'oos_bars.parquet').pivot(index='session', columns='bar_ix', values='close')
    h = load_headlines()
    h = h[h.session.isin(C.index) & (h.bar_ix <= HALF)]
    pos = pd.Series(np.arange(len(C)), index=C.index)
    return Split(C.index.values, C.values[:, :HALF + 1],
                 h.assign(sidx=h.session.map(pos).values).reset_index(drop=True), None)


# Improvements on bar-sized combined + industry (each switchable, so they can be scored alone).
PEER_CATS = ['4_1', '2_1', '1_2']  # (same-industry companies, industries tied for most); 3_1 is the reference


def drift_observations(split):
    """Per headline: mean per-bar log drift over lags 1..9 that fall inside the split's bars."""
    L = np.log(split.closes)
    logret = np.diff(np.concatenate([np.zeros((len(L), 1)), L], 1), axis=1)
    h = split.headlines
    s, t = h.sidx.values, h.bar_ix.values
    acc, n = np.zeros(len(h)), np.zeros(len(h))
    for k in range(1, 10):
        m = t + k <= logret.shape[1] - 1
        acc[m] += logret[s[m], t[m] + k]
        n[m] += 1
    return pd.DataFrame({'ev': h.ev.values, 'obs': acc / np.maximum(n, 1)})[n > 0]


class Improved(BarSizedVol):
    peer = oos_rates = news_risk = False

    def extra(self, split):
        cols = [super().extra(split)]
        if self.peer:
            h, t, r, rem = headline_info(split, self.rates)
            seen = pd.DataFrame({'sidx': h.sidx.values, 'co': company(h).values})
            seen['ind'] = seen.co.map(industry_map())
            first = seen.drop_duplicates(['sidx', 'co'])
            k = first.groupby(['sidx', 'ind']).size()
            n_ind = first.groupby('sidx').ind.nunique()
            cat = (pd.Series(k.reindex(pd.MultiIndex.from_frame(seen[['sidx', 'ind']])).values).astype(str) + '_'
                   + seen.sidx.map(n_ind).astype(str).values)
            cols += [per_session(split, h, np.where(cat.values == c, r * rem, 0)) for c in PEER_CATS]
        return np.column_stack(cols)

    def risk_inputs(self, split):
        X = bar_risk(split)
        if not self.news_risk:
            return X
        main = main_industry_only(split)
        h = main.headlines[main.headlines.bar_ix <= HALF]
        return np.column_stack([
            X,
            per_session(main, h, np.ones(len(h))),
            per_session(main, h, (h.bar_ix.values >= 40).astype(float)),
            h.assign(co=company(h)).groupby('sidx').co.nunique().reindex(range(len(main.sessions)), fill_value=0).values,
        ])

    def fit(self, train):
        main = main_industry_only(train)
        if self.oos_rates:
            obs = pd.concat([drift_observations(main), drift_observations(main_industry_only(oos_split()))])
            self.rates = obs.groupby('ev').obs.mean()
        else:
            L = np.log(main.closes)
            self.rates = fit_rates(main.headlines, np.diff(np.concatenate([np.zeros((len(L), 1)), L], 1), axis=1))
        R = self.risk_inputs(train)
        self.risk_model = LinearRegression().fit(R, train.returns ** 2)
        v = self.risk_model.predict(R)
        self.v_floor, self.v_mid = np.quantile(v, 0.05), np.median(v)
        w = 1 / np.maximum(v, self.v_floor) if self.news_risk else np.ones(len(v))
        Z = self.design(main)
        self.centre = np.average(Z, axis=0, weights=w)
        sw = np.sqrt(w)[:, None]
        self.coef = np.linalg.lstsq((Z - self.centre) * sw, train.returns * sw[:, 0], rcond=None)[0]
        self.cap = self.clip_sd * ((Z - self.centre) @ self.coef).std()

    def size(self, p, split):
        v = np.maximum(self.risk_model.predict(self.risk_inputs(split)), self.v_floor)
        return np.clip(p, -self.cap, self.cap) * (self.v_mid / v) ** self.power


class ImprovedPeer(Improved):
    name, peer = 'improved_peer', True


class ImprovedOOSRates(Improved):
    name, oos_rates = 'improved_oosrates', True


class ImprovedNewsRisk(Improved):
    name, news_risk = 'improved_newsrisk', True


class ImprovedAll(Improved):
    name, peer, oos_rates, news_risk = 'improved_all', True, True, True


MODELS.update({'improved_peer': ImprovedPeer, 'improved_oosrates': ImprovedOOSRates,
               'improved_newsrisk': ImprovedNewsRisk, 'improved_all': ImprovedAll})


def write_predictions(key, path):
    """Fit a model on all in-sample sessions and write out-of-sample predictions for upload."""
    data = backtest_bench.Data()
    model = MODELS[key]()
    model.fit(data.view(np.arange(data.n), train=True))
    C = pd.read_parquet(D / 'oos_bars.parquet').pivot(index='session', columns='bar_ix', values='close')
    test = oos_split()
    out = pd.DataFrame({'headline_id': C.index.astype(np.int64), 'pred': model.predict(test)})
    out.to_csv(path, index=False, lineterminator='\n')
    print(f'{model.name}: wrote {len(out)} rows to {path}')


if __name__ == '__main__':
    if sys.argv[1:2] == ['--write']:
        write_predictions(sys.argv[2], D / sys.argv[3])
    else:
        backtest_bench.main()
