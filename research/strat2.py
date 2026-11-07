import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from headline_drift_model import D, load_headlines, fit_rates, features, KER
h=load_headlines(); C=pd.read_parquet(D/'ins_bars.parquet').pivot(index='session',columns='bar_ix',values='close')
ses=C.index.values; C=C.values; n=len(ses)
ret=pd.read_parquet(D/'ins_returns.parquet').set_index('session').returns.reindex(ses).values
hi=h[h.session.isin(ses)].copy(); hi['sidx']=hi.session.map({s:i for i,s in enumerate(ses)})
w=h.headline.str.split(); p2=w.str[:2].str.join(' ')
hi['co']=hi.headline.str.split().str[:2].str.join(' ')  # 2-word prefix is unique per company
L=np.log(C); dl=np.diff(np.concatenate([np.zeros((n,1)),L],1),axis=1)
rates=fit_rates(hi,dl); hi['r']=hi.ev.map(rates); hi['sg']=np.sign(hi.r.where(hi.r.abs()>1e-4,0))
first=hi[hi.bar_ix<=49]; win=hi[(hi.bar_ix>=50)&(hi.bar_ix<=59)]; second=hi[hi.bar_ix>=50]
g=lambda d,v: d.assign(v=v).groupby('sidx').v.sum().reindex(range(n),fill_value=0).values
# unseen-news contribution to the 49->59 window
unseen=g(win, win.r.values*np.array([KER[:59-t+1].sum() for t in win.bar_ix]))
print('sd of unseen-news contribution (bp):', round(unseen.std()*1e4,1), ' corr with ret:', round(np.corrcoef(unseen,ret)[0,1],3))
X=features(hi,rates,C); print('sd of current mu (bp):', round(X[:,0].std()*1e4,1))
# 1. session level: first-half tone vs window tone
feats={'first_net_rate':g(first,first.r.values),'first_net_sign':g(first,first.sg.values),
       'first_frac_pos':g(first,(first.sg>0).astype(float).values)/np.maximum(g(first,np.ones(len(first))),1),
       'last20_net_sign':g(first[first.bar_ix>=30],first[first.bar_ix>=30].sg.values),
       'n_first':g(first,np.ones(len(first)))}
for k,v in feats.items(): print(f'{k:16s} corr unseen {np.corrcoef(v,unseen)[0,1]:+.4f}   corr ret-resid {np.corrcoef(v,ret-X[:,0]*0.916)[0,1]:+.4f}')
# 2. company level: first-half tone vs that company's later tone
a=first.groupby(['session','co']).sg.agg(['sum','size']); b=second.groupby(['session','co']).sg.agg(['sum','size'])
m=a.join(b,lsuffix='_1',rsuffix='_2',how='outer').fillna(0)
print('company: corr(first-half net sign, second-half net sign)', round(np.corrcoef(m.sum_1,m.sum_2)[0,1],4), 'pairs', len(m))
print('company: corr(first-half count, second-half count)', round(np.corrcoef(m.size_1,m.size_2)[0,1],4))
# sign persistence: P(next headline same sign | previous same company)
s=hi[hi.sg!=0].sort_values(['session','co','bar_ix']); s['prev']=s.groupby(['session','co']).sg.shift()
s=s.dropna(subset=['prev']); print('P(same sign as company prev):', round((s.sg==s.prev).mean(),4), ' base:', round(((s.sg>0).mean())**2+((s.sg<0).mean())**2,4))
# event-type sequencing: does the previous event type predict next sign beyond base?
t=s.groupby('ev').apply(lambda d:(d.prev>0).mean()) ; 
# 3. timing: is the count of news in bars 50-59 predictable?
cnt=g(win,np.ones(len(win))); print('window count mean', cnt.mean().round(2), 'corr with n_first', round(np.corrcoef(cnt,feats['n_first'])[0,1],4))
# 4. incremental OOF Sharpe
def sh(p,r): x=p*r; return np.sqrt(252)*x.mean()/x.std()
fold=np.random.default_rng(0).permutation(n)%5
for extra in [None,'first_net_rate','first_net_sign','last20_net_sign']:
    p=np.zeros(n)
    for f in range(5):
        tr=fold!=f; Z=X if extra is None else np.column_stack([X,feats[extra]])
        Zd=Z-Z[tr].mean(0); c=np.linalg.lstsq(Zd[tr],ret[tr],rcond=None)[0]; p[~tr]=Zd[~tr]@c
    cap=2*p.std(); print(f'model + {extra}: OOF Sharpe {sh(np.clip(p,-cap,cap),ret):.3f}')
print('mean rate of a headline (bp/bar):', round(hi.r.mean()*1e4,2))
tot=g(hi,np.ones(len(hi))); print('total headlines per session sd', tot.std().round(2), 'corr(first,second count)', round(np.corrcoef(feats['n_first'],tot-feats['n_first'])[0,1],3))
for extra in ['n_first']:
    for seed in [0,1,2]:
        fold=np.random.default_rng(seed).permutation(n)%5; out=[]
        for Z in [X,np.column_stack([X,feats[extra]])]:
            p=np.zeros(n)
            for f in range(5):
                tr=fold!=f; Zd=Z-Z[tr].mean(0); c=np.linalg.lstsq(Zd[tr],ret[tr],rcond=None)[0]; p[~tr]=Zd[~tr]@c
            cap=2*p.std(); out.append(sh(np.clip(p,-cap,cap),ret))
        print(f'seed {seed}: base {out[0]:.3f}  + n_first {out[1]:.3f}')
