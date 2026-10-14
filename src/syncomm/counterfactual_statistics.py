"""Pre-evaluation numerical convention for the locked family bootstrap."""
import numpy as np


def stable_bootstrap(values,seeds=(170917,270917,370917),n=20000):
    x=np.asarray(values,dtype=float)
    assert x.ndim==1 and len(x)>1 and np.isfinite(x).all()
    point=float(x.mean())
    # Preserve the specified inclusive >= comparison at exact rational
    # boundaries (accuracy increments are fractions of four responses).
    tol=128*np.finfo(float).eps*max(1.,abs(point),float(np.abs(x).max()))
    runs=[]
    for seed in seeds:
        rng=np.random.default_rng(seed)
        sampled=x[rng.integers(0,len(x),(n,len(x)))].mean(1)
        lo,hi=np.quantile(sampled,[.025,.975])
        p=(1+np.count_nonzero(sampled-point >= point-tol))/(n+1)
        runs.append(dict(seed=seed,lo=float(lo),hi=float(hi),p=float(p)))
    return dict(mean=point,lo=min(r['lo'] for r in runs),hi=max(r['hi'] for r in runs),
                p=max(r['p'] for r in runs),families=len(x),degenerate=bool(np.ptp(x)==0),
                p_boundary_tolerance=tol,monte_carlo=runs)
