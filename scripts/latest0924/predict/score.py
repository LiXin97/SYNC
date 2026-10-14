"""Experiment F scoring (locked rule in gates/latest0924/expF_predict.lock.json).

Observed J per held-out receiver x depth from the depth-scan raw rows (fitted, gamma 1; family-macro over the development
families; 64-token budget; engineering rows re-checked), compared with artifacts/latest0924/predict/predictions.json.
Cluster bootstrap over the 9 held-out receivers, 20,000 draws x seeds 20260924-6, enclosing interval.
Usage: python3 scripts/latest0924/predict/score.py  -> artifacts/latest0924/predict/score.{json,md}
"""
import collections
import glob
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]; P = ROOT / 'artifacts/latest0924/predict'
LOCK = json.loads((ROOT / 'gates/latest0924/expF_predict.lock.json').read_text())
SEEDS = (20260924, 20260925, 20260926); DRAWS = 20000


def correct(x):
    return x['correct'] if isinstance(x, dict) else bool(x)


def observed(k):
    root = P / 'depth' / k / 'raw'
    eng = [json.loads(l) for f in glob.glob(str(root / 'engineering_*.jsonl')) for l in open(f)]
    assert eng and all(r['engineering_checks'] is None or r['engineering_checks']['pass'] for r in eng), (k, 'engineering')
    m = json.loads((P / 'depth' / k / 'manifest.json').read_text())
    rows = [json.loads(l) for f in glob.glob(str(root / 'development_*.jsonl')) for l in open(f)]
    shards = sorted(glob.glob(str(root / 'development_*.done.json')))
    assert shards, (k, 'no development receipts')
    by = collections.defaultdict(list); fam = {}
    for r in rows:
        by[(r['arm'], r['depth_label'], r['gamma'], r['scene'])].append(r); fam[r['scene']] = r['family']
    out = {}
    for arm in ('fitted', 'native'):
        for d in LOCK['depths']:
            dl = f'd{int(round(d * 100))}'; g = 1 if arm == 'fitted' else 0
            fj = collections.defaultdict(list)
            for (a, l, gg, s), rs in by.items():
                if a == arm and l == dl and gg == g:
                    assert len(rs) == 4, (k, a, l, s, len(rs)); fj[fam[s]].append(all(correct(x['score64']) for x in rs))
            n = sum(len(v) for v in fj.values())
            assert n == 92, (k, arm, dl, n)
            out[(arm, str(d))] = 100 * sum(sum(v) / len(v) for v in fj.values()) / len(fj)
    return out


def ba(pred, obs):
    pos = [p for p, o in zip(pred, obs) if o]; neg = [p for p, o in zip(pred, obs) if not o]
    if not pos or not neg:
        return float('nan')
    return 0.5 * (sum(pos) / len(pos) + sum(1 - p for p in neg) / len(neg))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def boot(stat, recs, cells):
    lo, hi = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed); vals = []
        for _ in range(DRAWS):
            pick = rng.integers(0, len(recs), len(recs))
            sub = [c for i in pick for c in cells[recs[i]]]
            v = stat(sub)
            if v == v:
                vals.append(v)
        lo.append(np.percentile(vals, 2.5)); hi.append(np.percentile(vals, 97.5))
    return float(min(lo)), float(max(hi))


def main():
    pred = json.loads((P / 'predictions.json').read_text())
    recs = LOCK['heldout_receivers']
    cells = {}; table = {}
    for k in recs:
        o = observed(k); table[k] = o; cells[k] = []
        for d in LOCK['depths']:
            p = pred['predictions'][k][str(d)]
            cells[k].append(dict(receiver=k, depth=d, metric=p['metric'], pred=bool(p['predicted_usable']),
                                 depth_only=bool(p['depth_only_usable']), J=o[('fitted', str(d))], usable=o[('fitted', str(d))] >= 50.0,
                                 native=o[('native', str(d))]))
    allc = [c for k in recs for c in cells[k]]
    f_ba = lambda cs: ba([c['pred'] for c in cs], [c['usable'] for c in cs])
    f_acc = lambda cs: sum(c['pred'] == c['usable'] for c in cs) / len(cs)
    f_rho = lambda cs: spearman([c['metric'] for c in cs], [c['J'] for c in cs])
    f_base = lambda cs: ba([c['depth_only'] for c in cs], [c['usable'] for c in cs])
    f_diff = lambda cs: f_ba(cs) - f_base(cs)
    res = dict(balanced_accuracy=f_ba(allc), ba_ci=boot(f_ba, recs, cells), accuracy=f_acc(allc), acc_ci=boot(f_acc, recs, cells),
               spearman=f_rho(allc), spearman_ci=boot(f_rho, recs, cells), depth_only_ba=f_base(allc), depth_only_ci=boot(f_base, recs, cells),
               ba_minus_depth_only=f_diff(allc), diff_ci=boot(f_diff, recs, cells))
    res['success'] = bool(res['balanced_accuracy'] >= 0.80 and res['ba_ci'][0] > 0.50)
    deepest = {}
    for k in recs:
        du = [c['depth'] for c in cells[k] if c['usable']]; dp = [c['depth'] for c in cells[k] if c['pred']]
        deepest[k] = dict(observed=max(du) if du else None, predicted=max(dp) if dp else None)
    # non-blind check vs Experiment A middle band (already observed)
    A = json.loads((ROOT / 'artifacts/latest0924/expA/analysis/expA_results.json').read_text())['cells']
    nb = {}
    for k in recs:
        c = A[f'gemma-4-26b-a4b-it -> {k}']
        mid = 100 * c['estimates']['gemma-4-26b-a4b-it/original']['J']['family']['mean']
        nb[k] = dict(expA_middle_J=mid, expA_middle_usable=mid >= 50.0, predicted_usable_at_0_5=bool(pred['predictions'][k]['0.5']['predicted_usable']))
    nb_acc = sum(v['expA_middle_usable'] == v['predicted_usable_at_0_5'] for v in nb.values()) / len(nb)
    out = dict(lock_sha256=hashlib.sha256((ROOT / 'gates/latest0924/expF_predict.lock.json').read_bytes()).hexdigest(),
               predictions_sha256=hashlib.sha256((P / 'predictions.json').read_bytes()).hexdigest(), tau=pred['tau'], d_star=pred['d_star'],
               results=res, cells=allc, deepest_usable=deepest, non_blind_expA_check=dict(per_receiver=nb, agreement=nb_acc))
    (P / 'score.json').write_text(json.dumps(out, indent=1) + '\n')
    md = ['# Experiment F: blind held-out validation', '',
          f"tau = {pred['tau']:.4f} (primary metric {pred['metric']}); depth-only comparator d* = {pred['d_star']}", '',
          f"Balanced accuracy {res['balanced_accuracy']:.3f} [{res['ba_ci'][0]:.3f}, {res['ba_ci'][1]:.3f}]; accuracy {res['accuracy']:.3f}; "
          f"Spearman {res['spearman']:.3f} [{res['spearman_ci'][0]:.3f}, {res['spearman_ci'][1]:.3f}]; depth-only BA {res['depth_only_ba']:.3f} "
          f"[{res['depth_only_ci'][0]:.3f}, {res['depth_only_ci'][1]:.3f}]; difference {res['ba_minus_depth_only']:+.3f} [{res['diff_ci'][0]:+.3f}, {res['diff_ci'][1]:+.3f}]. "
          f"Locked success criterion met: {res['success']}.", '',
          '| receiver | ' + ' | '.join(f'd={d}' for d in LOCK['depths']) + ' |', '|---' * (len(LOCK['depths']) + 1) + '|']
    for k in recs:
        md.append(f'| {k} | ' + ' | '.join(f"{c['J']:.1f} ({'U' if c['pred'] else '.'}, {c['metric']:.3f})" for c in cells[k]) + ' |')
    md += ['', 'Cells: observed J (predicted usable U / not ., calibration metric).', '',
           f"Non-blind check vs Experiment A middle band (already observed): agreement {nb_acc:.2f}."]
    (P / 'score.md').write_text('\n'.join(md) + '\n'); print('\n'.join(md))


if __name__ == '__main__':
    main()
