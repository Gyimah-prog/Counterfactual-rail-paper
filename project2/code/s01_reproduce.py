import numpy as np,pandas as pd,time,json
from sklearn.metrics import pairwise_distances
from cf2lib import *
df,model,lib=load(); madv=mad(df); B=bounds(df)
X0=lib[[f'x0_{f}' for f in F]].values; D0=lib[[f'd_{f}' for f in F]].values
out={}
for y in (2014,2015,2016):
    s=df[(df.year==y)&(df.defect==1)]; Xt=s[F].values
    nn=pairwise_distances(Xt/madv,X0/madv).argmin(1)
    Z=apply_action(Xt,D0[nn]); vf,vp,vb,_=validate(model,Xt,Z,B)
    elig=model.predict(Xt)==1
    out[y]=dict(n_proj=len(s),n_elig=int(elig.sum()),flipped=int(vf.sum()),pct=round(100*vf.sum()/elig.sum(),2))
print(json.dumps(out,indent=1))
print('mean|dW|=%.4f signed mean dW=%.4f'%(np.abs(D0[:,W]).mean(),D0[:,W].mean()))
# timing fresh generation
cm=Counter(model); s=df[(df.year==2016)&(df.defect==1)]; Xt=s[F].values[:30]; t=time.time()
res=[find_cf(cm,x,B,madv) for x in Xt]
print('fresh: %.2f s/case, %.0f queries/case, success %d/30'%((time.time()-t)/30,cm.n/30,sum(r is not None for r in res)))
