"""Experiment E (latest0924) analysis, exactly as locked in gates/latest0924/expE_layout.lock.json.

Endpoint: four-answer success J per scene (both worlds x both questions correct at 64 tokens; invalid = wrong),
family-macro. Primary per receiver x condition c in {L1, L2, L3}: Delta_c = J_internal(c) - J_internal(L0), paired by
scene (the three-object scene is derived from the same two-object scene). Cluster bootstrap over the 36 families and,
separately, over the 12 domain x template clusters, 20,000 draws for each seed 20260924-6; the reported interval
encloses all seeds and both levels. Transfer rule (non-inferiority, margin 5 points): H0 Delta_c <= -5; one-sided
p = max over levels and seeds of the bootstrap fraction of Delta* <= -5; Holm across all receiver x condition cells
(alpha 0.05). Verdict: "transfers" if Holm rejects; "fails to transfer" if the enclosing upper limit < -5;
otherwise "inconclusive". Secondary (descriptive): internal - text and internal - sham0 per condition, text(c) - text(L0),
J ratio internal(c)/internal(L0). Checks: receipts and exact row sets, engineering rows all pass, identity write inert,
L0 rows identical (tokens and logits) to Experiment A complete-fact rows (R3 for qwen3-32b).

Usage: python3 scripts/latest0924/layout/analyze_layout.py
"""
import glob
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/latest0924/layout'
LOCK = ROOT / 'gates/latest0924/expE_layout.lock.json'
RECEIVERS = ['qwen3-32b', 'qwen3.5-27b', 'qwen3.6-27b', 'mistral-small-3.2-24b', 'qwen3.5-9b', 'granite-4.1-8b']
CONDS = ['L0', 'L1', 'L2', 'L3']
ARMS = ['none', 'internal', 'text', 'sham0']
SEEDS = (20260924, 20260925, 20260926)
DRAWS = 20000
MARGIN = 0.05
REF = {'qwen3-32b': ROOT / 'artifacts/receiver_recovery/r3/qwen3-32b'}
for k in RECEIVERS[1:]:
    REF[k] = ROOT / f'artifacts/latest0924/expA/gemma-4-26b-a4b-it/{k}'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(rec):
    m = json.loads((OUT / rec / 'manifest.json').read_text())
    bank = json.loads((OUT / 'bank.json').read_text())
    rows, eng = {}, []
    files = sorted(glob.glob(str(OUT / rec / 'raw' / 'development_*.jsonl')))
    receipts = [Path(f).with_suffix('.done.json') for f in files]
    status = 'COMPLETE' if files and all(r.exists() for r in receipts) else 'INCOMPLETE'
    for f in files:
        for line in open(f):
            r = json.loads(line); rows[r['key']] = r
    for f in sorted(glob.glob(str(OUT / rec / 'raw' / 'engineering_*.jsonl'))):
        eng += [json.loads(l) for l in open(f)]
    want = {f"{s['scene']}|{c}|{w}|{q}|{a}" for s in bank['scenes2'] for c in CONDS for w in (0, 1) for q in (0, 1) for a in ARMS}
    if set(rows) != want:
        status = 'INCOMPLETE' if set(rows) < want else 'ROW_SET_MISMATCH'
    return m, bank, rows, eng, status, [str(f) for f in files]


def eng_audit(eng):
    fail = [r['key'] for r in eng if r.get('engineering_checks') and not r['engineering_checks']['pass']]
    ident = {(r['scene'], r['condition'], r['world'], r['question']): r for r in eng if r['arm'] == 'identity_write'}
    none = {(r['scene'], r['condition'], r['world'], r['question']): r for r in eng if r['arm'] == 'none'}
    not_inert = [k for k, r in ident.items() if (r['logits_sha256'], r['tokens']) != (none[k]['logits_sha256'], none[k]['tokens'])]
    ml = max((r['engineering_checks']['max_logit_difference'] for r in eng if r.get('engineering_checks')), default=None)
    return dict(rows=len(eng), failing=len(fail), identity_not_inert=len(not_inert), max_logit_difference=ml,
                verdict='PASS' if eng and not fail and not not_inert else 'FAIL')


def l0_reference(rec, rows):
    ref = {}
    for f in sorted(glob.glob(str(REF[rec] / 'raw' / 'development_*.jsonl'))):
        for line in open(f):
            x = json.loads(line)
            if x['source'] == 'literal' and x['arm'] in ('selected', 'packet_text'):
                ref[x['scene'], x['world'], x['question'], 'internal' if x['arm'] == 'selected' else 'text'] = x
    n = same = 0
    for r in rows.values():
        if r['condition'] != 'L0' or r['arm'] not in ('internal', 'text'):
            continue
        x = ref.get((r['scene'], r['world'], r['question'], r['arm']))
        if x is None:
            continue
        n += 1
        same += (x['tokens'] == r['tokens'] and x['logits_sha256'] == r['logits_sha256'])
    return dict(compared=n, identical=same)


def scene_J(rows, bank):
    J = defaultdict(dict); cov = defaultdict(dict)
    for s in bank['scenes2']:
        for c in CONDS:
            for a in ARMS:
                rs = [rows[f"{s['scene']}|{c}|{w}|{q}|{a}"] for w in (0, 1) for q in (0, 1)]
                J[c, a][s['scene']] = float(all(r['score64']['correct'] for r in rs))
                cov[c, a][s['scene']] = float(np.mean([r['score64']['answered'] for r in rs]))
    return J, cov


def clusters(bank):
    fam = {s['scene']: s['family'] for s in bank['scenes2']}
    tpl = {s['scene']: f"{s['domain']}|{s['template']}" for s in bank['scenes2']}
    return fam, tpl


def boot(diff_by_scene, cl):
    """Cluster-macro mean of per-scene values; percentile intervals per seed; one-sided NI p per seed."""
    keys = sorted(set(cl.values())); idx = {k: i for i, k in enumerate(keys)}
    sums = np.zeros(len(keys)); cnt = np.zeros(len(keys))
    for s, v in diff_by_scene.items():
        sums[idx[cl[s]]] += v; cnt[idx[cl[s]]] += 1
    means = sums / cnt
    est = float(means.mean())
    out = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        b = means[rng.integers(0, len(keys), size=(DRAWS, len(keys)))].mean(axis=1)
        out.append(dict(lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)),
                        p_ni=float(np.mean(b <= -MARGIN)), p_two=float(min(1.0, 2 * min(np.mean(b <= 0), np.mean(b >= 0))))))
    return est, out


def contrast(a, b, fam, tpl):
    d = {s: a[s] - b[s] for s in a}
    ef, of = boot(d, fam); et, ot = boot(d, tpl)
    return dict(estimate=ef, lo=min(x['lo'] for x in of + ot), hi=max(x['hi'] for x in of + ot),
                p_ni=max(x['p_ni'] for x in of + ot), p_two=max(x['p_two'] for x in of + ot),
                family=dict(estimate=ef, per_seed=of), template=dict(estimate=et, per_seed=ot))


def holm(pvals, alpha=0.05):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i]); m = len(pvals); rej = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            rej[i] = True
        else:
            break
    return rej


def main():
    lock = json.loads(LOCK.read_text())
    res = dict(lock=str(LOCK.relative_to(ROOT)), lock_sha256=sha(LOCK), script_sha256=sha(Path(__file__)), receivers={})
    cells = []
    for rec in RECEIVERS:
        m, bank, rows, eng, status, files = load(rec)
        R = dict(status=status, files=files, engineering=eng_audit(eng))
        if status != 'COMPLETE':
            res['receivers'][rec] = R; continue
        R['l0_reproduction'] = l0_reference(rec, rows)
        J, cov = scene_J(rows, bank); fam, tpl = clusters(bank)
        est = {}
        for c in CONDS:
            for a in ARMS:
                e, of = boot(J[c, a], fam)
                est[f'{c}/{a}'] = dict(J=e, lo=min(x['lo'] for x in of), hi=max(x['hi'] for x in of),
                                      coverage=float(np.mean(list(cov[c, a].values()))))
        R['estimates'] = est
        R['contrasts'] = {}
        for c in CONDS:
            R['contrasts'][f'{c}: internal - text'] = contrast(J[c, 'internal'], J[c, 'text'], fam, tpl)
            R['contrasts'][f'{c}: internal - sham0'] = contrast(J[c, 'internal'], J[c, 'sham0'], fam, tpl)
            if c != 'L0':
                R['contrasts'][f'{c} - L0: internal'] = contrast(J[c, 'internal'], J['L0', 'internal'], fam, tpl)
                R['contrasts'][f'{c} - L0: text'] = contrast(J[c, 'text'], J['L0', 'text'], fam, tpl)
                cells.append((rec, c))
        res['receivers'][rec] = R
    ps = [res['receivers'][r]['contrasts'][f'{c} - L0: internal']['p_ni'] for r, c in cells]
    rej = holm(ps)
    res['holm'] = dict(m=len(cells), alpha=0.05, margin_points=100 * MARGIN)
    for (r, c), p, rj in zip(cells, ps, rej):
        x = res['receivers'][r]['contrasts'][f'{c} - L0: internal']
        eng_ok = res['receivers'][r]['engineering']['verdict'] == 'PASS'
        x['holm_reject'] = rj
        x['verdict'] = ('not interpretable (engineering)' if not eng_ok else 'transfers' if rj else
                        'fails to transfer' if x['hi'] < -MARGIN else 'inconclusive')
        x['ratio_internal'] = (res['receivers'][r]['estimates'][f'{c}/internal']['J'] /
                               max(res['receivers'][r]['estimates']['L0/internal']['J'], 1e-9))
    (OUT / 'analysis').mkdir(exist_ok=True)
    (OUT / 'analysis' / 'layout_results.json').write_text(json.dumps(res, indent=1) + '\n')
    md = ['# Experiment E: frozen-setting transfer to new layouts', '',
          f"Lock `{res['lock']}` sha `{res['lock_sha256'][:12]}`; Holm m = {len(cells)}, non-inferiority margin 5 points.", '',
          '| receiver | status | eng | L0 repro | cond | J internal | J text | J sham0 | int(c)-int(L0) [CI] | verdict | int-text [CI] |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in RECEIVERS:
        R = res['receivers'][r]
        if R['status'] != 'COMPLETE':
            md.append(f"| {r} | {R['status']} | {R['engineering']['verdict']} | | | | | | | | |"); continue
        for c in CONDS:
            e = R['estimates']; x = R['contrasts'].get(f'{c} - L0: internal'); t = R['contrasts'][f'{c}: internal - text']
            d = f"{100 * x['estimate']:+.1f} [{100 * x['lo']:+.1f}, {100 * x['hi']:+.1f}]" if x else ''
            md.append(f"| {r} | {R['status']} | {R['engineering']['verdict']} | {R['l0_reproduction']['identical']}/{R['l0_reproduction']['compared']} | {c} | "
                      f"{100 * e[c + '/internal']['J']:.1f} | {100 * e[c + '/text']['J']:.1f} | {100 * e[c + '/sham0']['J']:.1f} | {d} | "
                      f"{x['verdict'] if x else ''} | {100 * t['estimate']:+.1f} [{100 * t['lo']:+.1f}, {100 * t['hi']:+.1f}] |")
    (OUT / 'analysis' / 'layout_results.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
