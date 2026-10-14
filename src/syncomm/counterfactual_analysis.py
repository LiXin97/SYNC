"""Pure-CPU analysis; inference units are material families, not questions."""
from collections import defaultdict
import numpy as np


def condition(row):
    return f"{row['arm']}:{row['gamma']}:{row['writer']}"


def collapse(rows, scenes, budget=64):
    grouped = defaultdict(dict)
    seen = set()
    for r in rows:
        assert r['key'] not in seen, r['key']
        seen.add(r['key'])
        grouped[r['scene'], condition(r)][r['world'], r['question']] = r
    result = []
    for (sid, cond), cells in sorted(grouped.items()):
        assert set(cells) == {(0,0),(0,1),(1,0),(1,1)}, (sid,cond)
        s = scenes[sid]
        answers = {k:r[f'score{budget}']['answer'] for k,r in cells.items()}
        ok = {k: answers[k] == s['worlds'][k[0]]['answers'][k[1]] for k in cells}
        g = {k:r['gap'] for k,r in cells.items()}
        sign = 1 if s['worlds'][0]['answers'][0] == s['aliases'][0] else -1
        result.append(dict(scene=sid, family=s['family'], domain=s['domain'],
            condition=cond, accuracy=sum(ok.values())/4,
            coverage=sum(r[f'score{budget}']['answered'] for r in cells.values())/4,
            relevant_both=float(ok[0,0] and ok[1,0]),
            unrelated_both=float(ok[0,1] and ok[1,1]),
            all_four=float(all(ok.values())),
            unrelated_consistent=float(answers[0,1] is not None and answers[0,1]==answers[1,1]),
            related_gap=sign*(g[0,0]-g[1,0]), unrelated_abs_gap=abs(g[0,1]-g[1,1])))
    return result


METRICS = ('accuracy','coverage','relevant_both','unrelated_both','all_four',
           'unrelated_consistent','related_gap','unrelated_abs_gap')


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups[r['condition']].append(r)
    return {k:dict(n=len(v), families=len({r['family'] for r in v}),
                  **{metric:float(np.mean([r[metric] for r in v])) for metric in METRICS})
            for k,v in sorted(groups.items())}


def select_strength(summary, writer):
    candidates = [(0,summary['none:0:band'])]
    candidates += [(g,summary[f'write:{g}:{writer}']) for g in (1,2,4,8,16)]
    admitted = [(g,s) for g,s in candidates if s['coverage'] >= .90]
    if not admitted:
        return dict(gamma=0, eligible=False, reason='No candidate meets 90% format coverage')
    gamma,s = min(admitted,key=lambda x:(-x[1]['accuracy'],x[0]))
    return dict(gamma=gamma,eligible=True,dev_accuracy=s['accuracy'],dev_coverage=s['coverage'])


def family_values(records, metric):
    groups = defaultdict(list)
    for r in records:
        groups[r['family']].append(r[metric])
    return np.array([np.mean(groups[k]) for k in sorted(groups)],dtype=float)


def bootstrap(values, seeds=(170917,270917,370917), n=20000):
    """Union percentile CI and largest centered-null one-sided Monte Carlo p."""
    x=np.asarray(values,dtype=float)
    assert x.ndim==1 and len(x)>1 and np.isfinite(x).all()
    point=float(x.mean()); runs=[]
    for seed in seeds:
        rng=np.random.default_rng(seed)
        samples=x[rng.integers(0,len(x),(n,len(x)))].mean(1)
        lo,hi=np.quantile(samples,[.025,.975])
        p=(1+np.count_nonzero(samples-point >= point))/(n+1)
        runs.append(dict(seed=seed,lo=float(lo),hi=float(hi),p=float(p)))
    return dict(mean=point,lo=min(v['lo'] for v in runs),hi=max(v['hi'] for v in runs),
                p=max(v['p'] for v in runs),families=len(x),degenerate=bool(np.ptp(x)==0),
                monte_carlo=runs)


def paired(records, treatment, reference, metric):
    a={r['scene']:r for r in records if r['condition']==treatment}
    b={r['scene']:r for r in records if r['condition']==reference}
    assert set(a)==set(b) and a
    return [dict(scene=s,family=a[s]['family'],value=a[s][metric]-b[s][metric]) for s in sorted(a)]


def holm(ps):
    p=np.asarray(ps,dtype=float);assert np.all((p>=0)&(p<=1))
    order=np.argsort(p,kind='stable');out=np.empty(len(p));last=0.
    for rank,i in enumerate(order):
        last=max(last,min(1.,p[i]*(len(p)-rank)));out[i]=last
    return out.tolist()
