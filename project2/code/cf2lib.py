"""Shared definitions for Counterfactual Project 2. Frozen reference objects come from the
canonical cache (model_base_nb.pkl, cf_2014_base_nb.pkl, data.pkl) that reproduces the locked paper numbers."""
import pickle, time, warnings, numpy as np, pandas as pd
from scipy.optimize import brentq, minimize
warnings.filterwarnings('ignore')
C='/mnt/user-data/uploads/counterfactual_work/vscode/scripts/.cache/'
F=['GAGE','ALI_L_62','ALI_R_62','WARP_62','XLEVEL']; W=3; X=4; SEED=42; TAU=0.5
PROJ_LO=np.array([56.3,-0.20,-0.20,-0.30,-0.80]); PROJ_HI=np.array([57.2,0.20,0.20,0.30,0.80])
FRA_LO=np.array([56.0,-0.5,-0.5,-1.0,-1.0]); FRA_HI=np.array([57.5,0.5,0.5,1.0,1.0])
def load():
    df=pickle.load(open(C+'data.pkl','rb')).reset_index().rename(columns={'index':'rid'})
    model=pickle.load(open(C+'model_base_nb.pkl','rb'))
    lib=pickle.load(open(C+'cf_2014_base_nb.pkl','rb'))
    return df,model,lib
def bounds(df):
    lo=np.maximum(FRA_LO,df[F].min().values); hi=np.minimum(FRA_HI,df[F].max().values)
    return np.c_[lo,hi]
def mad(df):
    tr=df[df.year.isin([2013,2014])][F]
    m=(tr-tr.median()).abs().median().values; return np.where(m==0,1e-6,m)
def G(Z):  # project defect indicator
    return ((Z<PROJ_LO)|(Z>PROJ_HI)).any(axis=1)
def excess(Z):  # per feature project excess
    return np.maximum(np.maximum(PROJ_LO-Z, Z-PROJ_HI),0)
def group(Z):
    v=(Z<PROJ_LO)|(Z>PROJ_HI); w=v[:,W]; x=v[:,X]; oth=v[:,[0,1,2]].any(1)
    g=np.full(len(Z),'Other',dtype=object)
    g[w&~x&~oth]='WARP only'; g[x&~w&~oth]='XLEVEL only'; g[w&x&~oth]='WARP and XLEVEL'
    return g
class Counter:
    def __init__(s,m): s.m=m; s.n=0
    def p(s,Z): Z=np.atleast_2d(Z); s.n+=len(Z); return s.m.predict_proba(Z)[:,1]
def warp_dir(Xt,D):
    """Paper replication rule: WARP action opposes target WARP sign (x>0 -> -|d|, else +|d|)."""
    D=D.copy(); D[:,W]=np.where(Xt[:,W]>0,-np.abs(D[:,W]),np.abs(D[:,W])); return D
def validate(model,Xt,Z,B):
    p0=model.predict_proba(Xt)[:,1]; p1=model.predict_proba(Z)[:,1]
    vf=(p0>=TAU)&(p1<TAU); vp=~G(Z); vb=((Z>=B[:,0]-1e-12)&(Z<=B[:,1]+1e-12)).all(1)
    return vf,vp,vb,p1
def apply_action(Xt,D):
    """Shared postprocessing: direction rule, then component clipping to FRA outer bounds (paper replication)."""
    Dd=warp_dir(Xt,D); Z=np.clip(Xt+Dd,FRA_LO,FRA_HI); return Z
# fresh generation: notebook two-stage optimizer, verbatim logic, with query counting
def find_cf(cm,x0,Barr,madv,l2=0.5,seed=SEED):
    rng=np.random.default_rng(seed); x0=np.asarray(x0,float); cands=[]
    def obj(x):
        x=np.clip(x,Barr[:,0],Barr[:,1]); p=np.clip(cm.p(x)[0],1e-9,1-1e-9)
        return -np.log(1-p)+l2*np.sum(np.abs(x-x0)/madv)
    def pp(v):
        x=x0.copy(); x[W]=v; return cm.p(x)[0]-0.5
    flip=None
    try:
        wv=x0[W]
        if wv<0:
            lo=max(Barr[W,0],wv); hi=0.0
            if pp(lo)*pp(hi)<0: flip=brentq(pp,lo,hi,xtol=1e-4)
            elif pp(0.0)*pp(Barr[W,1])<0: flip=brentq(pp,0.0,Barr[W,1],xtol=1e-4)
        else:
            lo=0.0; hi=min(Barr[W,1],wv)
            if pp(lo)*pp(hi)<0: flip=brentq(pp,lo,hi,xtol=1e-4)
            elif pp(Barr[W,0])*pp(0.0)<0: flip=brentq(pp,Barr[W,0],0.0,xtol=1e-4)
    except ValueError: pass
    if flip is not None:
        xi=x0.copy(); xi[W]=flip-np.sign(x0[W])*0.05*abs(flip-x0[W]); xi=np.clip(xi,Barr[:,0],Barr[:,1])
        if cm.p(xi)[0]<0.5: cands.append((xi.copy(),obj(xi)))
        for ns in (0.0,0.005,0.01):
            xs=np.clip(xi+rng.normal(0,ns,5)*(Barr[:,1]-Barr[:,0]),Barr[:,0],Barr[:,1])
            r=minimize(obj,xs,method='Nelder-Mead',options={'maxiter':100,'xatol':1e-5,'fatol':1e-5})
            xr=np.clip(r.x,Barr[:,0],Barr[:,1])
            if cm.p(xr)[0]<0.5: cands.append((xr.copy(),r.fun))
    if not cands:
        rg=Barr[:,1]-Barr[:,0]
        for _ in range(5):
            xr=np.clip(x0+rng.normal(0,0.02,5)*rg,Barr[:,0],Barr[:,1])
            r=minimize(obj,xr,method='Nelder-Mead',options={'maxiter':200,'xatol':1e-5,'fatol':1e-5})
            x2=np.clip(r.x,Barr[:,0],Barr[:,1])
            if cm.p(x2)[0]<0.5: cands.append((x2.copy(),r.fun))
    if not cands: return None
    return min(cands,key=lambda c:c[1])[0]
