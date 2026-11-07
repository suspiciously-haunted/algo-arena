import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from sklearn.cluster import SpectralClustering
from headline_drift_model import D, load_headlines, fit_rates, KER
h=load_headlines(); h['co']=h.headline.str.split().str[:2].str.join(' ')
cos=np.sort(h.co.unique()); ci={c:i for i,c in enumerate(cos)}
pres=h.groupby(['session','co']).size().unstack(fill_value=0).reindex(columns=cos).gt(0).astype(float).values
co_occ=pres.T@pres; np.fill_diagonal(co_occ,0)
p=pres.mean(0); exp=np.outer(p,p)*len(pres); lift=co_occ/exp
lab=SpectralClustering(5,affinity='precomputed',random_state=0).fit_predict(lift)
print('cluster sizes', np.bincount(lab))
full=pd.Series(h.headline.str.split().str[:3].str.join(' ').values,index=h.co.values).groupby(level=0).first()
for k in range(5): print(k, sorted(full[cos[lab==k]].values))
same=lab[:,None]==lab[None,:]; off=~np.eye(80,dtype=bool)
print('lift same industry', round(co_occ[same&off].sum()/exp[same&off].sum(),2), ' different', round(co_occ[~same].sum()/exp[~same].sum(),2))
# stability: ins-only vs oos-only clustering
for nm,m in [('ins',h.session<11000),('oos',h.session>=11000)]:
    pr=h[m].groupby(['session','co']).size().unstack(fill_value=0).reindex(columns=cos,fill_value=0).gt(0).astype(float).values
    c=pr.T@pr; np.fill_diagonal(c,0); pp=pr.mean(0); l=c/(np.outer(pp,pp)*len(pr))
    lb=SpectralClustering(5,affinity='precomputed',random_state=0).fit_predict(l)
    from sklearn.metrics import adjusted_rand_score; print(nm,'agreement with full clustering (ARI)', round(adjusted_rand_score(lab,lb),3))
# sessions: industry mix
h['ind']=h.co.map(dict(zip(cos,lab)))
mix=h.groupby('session').apply(lambda d: d.drop_duplicates('co').ind.value_counts().max())
print('size of largest industry group among session companies:', mix.value_counts().sort_index().to_dict())
# odd-one-out effect on realised drift, in-sample
ins=h[h.session<11000].copy(); ses=np.sort(ins.session.unique())
C=pd.read_parquet(D/'ins_bars.parquet').pivot(index='session',columns='bar_ix',values='close').reindex(ses).values
ins['sidx']=ins.session.map({s:i for i,s in enumerate(ses)})
L=np.log(C); dl=np.diff(np.concatenate([np.zeros((len(L),1)),L],1),axis=1)
rates=fit_rates(ins,dl); ins['r']=ins.ev.map(rates)
cs=ins.drop_duplicates(['session','co']); peers=cs.groupby(['session','ind']).size()
ins['peers']=[peers[(s,i)]-1 for s,i in zip(ins.session,ins.ind)]
s,t=ins.sidx.values,ins.bar_ix.values; acc=np.zeros(len(ins)); n=np.zeros(len(ins))
for k in range(1,10):
    m=t+k<=99; acc[m]+=dl[s[m],t[m]+k]; n[m]+=1
ins['real']=acc/np.maximum(n,1)
ins=ins[ins.r.abs()>1e-4]
for nm,m in [('first half',ins.bar_ix<=49),('second half',ins.bar_ix>49)]:
    d=ins[m]; out=d.groupby('peers').apply(lambda x: pd.Series({'realised/expected':(x.real*x.r).sum()/(x.r**2).sum(),'n':len(x)}))
    print(nm); print(out.round(2).to_string())
pd.Series(dict(zip(cos,lab))).to_csv('industry_map.csv')
