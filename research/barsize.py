import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from headline_drift_model import D
b=pd.read_parquet(D/'ins_bars.parquet'); P={c:b.pivot(index='session',columns='bar_ix',values=c).values[:,:50] for c in ['open','high','low','close']}
ret=pd.read_parquet(D/'ins_returns.parquet').set_index('session').returns.values
O,H,Lo,C=[np.log(P[c]) for c in ['open','high','low','close']]
prevC=np.concatenate([np.zeros((len(C),1)),C[:,:-1]],1)
cc=(C-prevC)**2; park=(H-Lo)**2/(4*np.log(2)); gk=0.5*(H-Lo)**2-(2*np.log(2)-1)*(C-O)**2; gap=(O-prevC)**2
F={}
for nm,x in [('cc',cc),('park',park),('gk',gk)]:
    for w in (10,20,50): F[f'{nm}_{w}']=x[:,-w:].mean(1)
F['gap_50']=gap.mean(1); F['range_trend']=park[:,-10:].mean(1)/park.mean(1)
y=ret**2
print('corr with squared return (higher = better at predicting risk):')
for k,v in F.items(): print(f'  {k:12s} {np.corrcoef(v,y)[0,1]:+.4f}  | corr with |ret| {np.corrcoef(np.sqrt(v),abs(ret))[0,1]:+.4f}')
print('dispersion of park_50 vol (p5,p50,p95):', np.quantile(np.sqrt(F['park_50']),[.05,.5,.95]).round(5))
# realised 10-bar vol ratio by vol decile
q=pd.qcut(F['park_50'],5,labels=False); print('ret sd by park_50 quintile:', pd.Series(ret).groupby(q).std().round(5).to_dict())
