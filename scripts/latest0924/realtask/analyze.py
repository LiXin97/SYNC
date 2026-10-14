"""Experiment D analysis (rules frozen in gates/latest0924/expD_realtask.lock.json).

Rows: frozen evaluate_confirm rows under artifacts/latest0924/realtask/eval/<receiver>/raw/. A receiver counts only if its
engineering rows all pass the gate (recomputed from the rows: max logit difference <= 0.05, alias-gap difference <= 0.1,
equal parses; identity writes reproduce no-message tokens) and every development shard has a receipt with the full row set.
Endpoint J: four-answer success per scene (both worlds x both questions correct at 64 tokens), averaged within family
(country pair) then across families. Secondary: single-answer accuracy, coverage (answered), J at 256.
Intervals: cluster bootstrap over families and, separately, over templates (continent pairs), 20,000 draws x seeds
20260924-6, percentile 95%, enclosing interval over seeds and both levels; p = larger two-sided bootstrap p.
Positive (per receiver): Holm (alpha 0.05, across evaluated receivers) rejects BOTH selected - none and
selected - mean(reassignment), and both enclosing lower limits > 0. Secondary contrasts: selected - packet_text,
selected - full_text, selected - original. Usage: python3 scripts/latest0924/realtask/analyze.py
"""
import glob, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / 'artifacts/latest0924/realtask'
SENDER = 'gemma-4-26b-a4b-it'
SEEDS, DRAWS = (20260924, 20260925, 20260926), 20000
RECEIVERS = ['qwen3.5-27b', 'qwen3.6-27b', 'mistral-small-3.2-24b', 'qwen3-32b']
ARMS = ['none', 'original', 'selected', 'packet_text', 'full_text', 'sham0', 'sham1', 'sham2', 'other_world']


def load(rec):
    root = A / 'eval' / rec
    m = json.loads((root / 'manifest.json').read_text())
    eng = [json.loads(l) for f in sorted(glob.glob(str(root / 'raw/engineering_*.jsonl'))) for l in open(f)]
    dev = [json.loads(l) for f in sorted(glob.glob(str(root / 'raw/development_*.jsonl'))) for l in open(f)]
    receipts = sorted(glob.glob(str(root / 'raw/development_*.done.json')))
    return m, eng, dev, receipts


def engineering_ok(eng):
    bad = [r for r in eng if r.get('engineering_checks') and not (r['engineering_checks']['max_logit_difference'] <= .05 and
           r['engineering_checks']['alias_gap_difference'] <= .1 and r['engineering_checks']['parsed_equal'])]
    none = {(r['scene'], r['world'], r['question']): (r['tokens'], r['logits_sha256']) for r in eng if r['arm'] == 'none'}
    ident = [r for r in eng if r['arm'] == 'identity_write']
    inert = all(none[(r['scene'], r['world'], r['question'])] == (r['tokens'], r['logits_sha256']) for r in ident)
    return (len(eng) > 0 and not bad and inert and len(ident) > 0), dict(rows=len(eng), failing=len(bad), identity_rows=len(ident), identity_inert=inert)


def scene_table(dev, arm, metric='J'):
    by = defaultdict(list); meta = {}
    for r in dev:
        if r['source'] == SENDER and r['arm'] == arm:
            by[r['scene']].append(r); meta[r['scene']] = (r['family'], r['template'])
    out = {}
    for s, rs in by.items():
        assert len(rs) == 4, (arm, s, len(rs))
        if metric == 'J':
            v = float(all(x['score64']['correct'] for x in rs))
        elif metric == 'J256':
            v = float(all(x['score256']['correct'] for x in rs))
        elif metric == 'acc':
            v = float(np.mean([x['score64']['correct'] for x in rs]))
        elif metric == 'cov':
            v = float(np.mean([x['score64']['answered'] for x in rs]))
        elif metric == 'supplied':
            v = float(np.mean([x['score64_supplied']['correct'] for x in rs]))
        out[s] = (meta[s][0], meta[s][1], v)
    return out


def macro(vals, keys):
    fam = defaultdict(list)
    for s in keys:
        fam[vals[s][0]].append(vals[s][2])
    return float(np.mean([np.mean(v) for v in fam.values()]))


def boot(diff_by_scene, meta):
    """diff_by_scene: scene -> value (a difference or a level). Returns family- and template-level enclosing CI + p."""
    scenes = sorted(diff_by_scene)
    res = {}
    for level, idx in (('family', 0), ('template', 1)):
        groups = defaultdict(list)
        for s in scenes:
            groups[meta[s][idx]].append(s)
        G = sorted(groups)
        if level == 'family':
            gv = np.array([np.mean([diff_by_scene[s] for s in groups[g]]) for g in G])
            stat = lambda sel: gv[sel].mean(axis=1)
        else:
            sums = np.array([np.sum([diff_by_scene[s] for s in groups[g]]) for g in G]); cnt = np.array([len(groups[g]) for g in G], float)
            stat = lambda sel: sums[sel].sum(axis=1) / cnt[sel].sum(axis=1)
        lo, hi, ps = [], [], []
        for seed in SEEDS:
            sel = np.random.default_rng(seed).integers(0, len(G), (DRAWS, len(G)))
            b = stat(sel)
            lo.append(float(np.percentile(b, 2.5))); hi.append(float(np.percentile(b, 97.5)))
            ps.append(float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))))
        res[level] = dict(clusters=len(G), lo=100 * min(lo), hi=100 * max(hi), p=max(ps))
    return dict(lo=min(res['family']['lo'], res['template']['lo']), hi=max(res['family']['hi'], res['template']['hi']),
                p=max(res['family']['p'], res['template']['p']), levels=res)


def holm(ps):
    order = sorted(ps, key=lambda k: ps[k]); m = len(ps); out = {}; stop = False
    for i, k in enumerate(order):
        thr = 0.05 / (m - i)
        rej = (not stop) and ps[k] <= thr
        if not rej:
            stop = True
        out[k] = dict(p=float(ps[k]), threshold=float(thr), reject=bool(rej))
    return out


def main():
    result = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), receivers={})
    for rec in RECEIVERS:
        try:
            m, eng, dev, receipts = load(rec)
        except FileNotFoundError:
            result['receivers'][rec] = dict(status='NOT RUN'); continue
        ok, eng_info = engineering_ok(eng)
        n_eval = m['bank_size'] - len(m['engineering_scene_ids'])
        want = n_eval * 4 * len(m['arms'][SENDER])  # the frozen evaluator also writes identity_write rows in development
        complete = ok and len(dev) == want and len(receipts) > 0
        R = dict(status='COMPLETE' if complete else ('ENGINEERING FAILURE' if not ok else 'INCOMPLETE'),
                 engineering=eng_info, rows=len(dev), expected_rows=want)
        if complete:
            tabs = {arm: scene_table(dev, arm) for arm in ARMS}
            keys = sorted(tabs['selected']); meta = {s: tabs['selected'][s][:2] for s in keys}
            R['J'] = {arm: 100 * macro(tabs[arm], keys) for arm in ARMS}
            R['accuracy'] = {arm: 100 * macro(scene_table(dev, arm, 'acc'), keys) for arm in ARMS}
            R['coverage'] = {arm: 100 * macro(scene_table(dev, arm, 'cov'), keys) for arm in ARMS}
            R['J256_selected'] = 100 * macro(scene_table(dev, 'selected', 'J256'), keys)
            R['other_world_supplied_accuracy'] = 100 * macro(scene_table(dev, 'other_world', 'supplied'), keys)
            sham = {s: np.mean([tabs[f'sham{i}'][s][2] for i in range(3)]) for s in keys}
            con = {}
            for name, ref in [('selected_minus_none', {s: tabs['none'][s][2] for s in keys}), ('selected_minus_mean_sham', sham),
                              ('selected_minus_text', {s: tabs['packet_text'][s][2] for s in keys}),
                              ('selected_minus_full_text', {s: tabs['full_text'][s][2] for s in keys}),
                              ('selected_minus_original', {s: tabs['original'][s][2] for s in keys})]:
                d = {s: tabs['selected'][s][2] - ref[s] for s in keys}
                con[name] = dict(estimate=100 * macro({s: (meta[s][0], meta[s][1], d[s]) for s in keys}, keys), **boot(d, meta))
            R['J_selected_ci'] = boot({s: tabs['selected'][s][2] for s in keys}, meta)
            R['contrasts'] = con; R['scenes'] = len(keys)
        result['receivers'][rec] = R
    done = [r for r, v in result['receivers'].items() if v.get('status') == 'COMPLETE']
    for c in ('selected_minus_none', 'selected_minus_mean_sham'):
        h = holm({r: result['receivers'][r]['contrasts'][c]['p'] for r in done}) if done else {}
        for r in done:
            result['receivers'][r]['contrasts'][c]['holm'] = h[r]
    for r in done:
        c = result['receivers'][r]['contrasts']
        pos = all(c[k]['holm']['reject'] and c[k]['lo'] > 0 for k in ('selected_minus_none', 'selected_minus_mean_sham'))
        result['receivers'][r]['verdict'] = 'POSITIVE' if pos else 'NOT POSITIVE'
    out = A / 'analysis'; out.mkdir(exist_ok=True)
    (out / 'expD_results.json').write_text(json.dumps(result, indent=1) + '\n')
    lines = ['# Experiment D (Countries, final interface) results', '',
             '| receiver | status | J none | J internal [CI] | J text | J full evidence | J middle | J reassigned (mean) | cov internal | internal - text [CI] | internal - reassigned [CI] | verdict |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in RECEIVERS:
        v = result['receivers'][r]
        if v.get('status') != 'COMPLETE':
            lines.append(f"| {r} | {v.get('status')} |" + ' |' * 10); continue
        J = v['J']; c = v['contrasts']; ci = v['J_selected_ci']
        lines.append(f"| {r} | COMPLETE | {J['none']:.1f} | {J['selected']:.1f} [{ci['lo']:.1f}, {ci['hi']:.1f}] | {J['packet_text']:.1f} | {J['full_text']:.1f} | "
                     f"{J['original']:.1f} | {np.mean([J[f'sham{i}'] for i in range(3)]):.1f} | {v['coverage']['selected']:.1f} | "
                     f"{c['selected_minus_text']['estimate']:+.1f} [{c['selected_minus_text']['lo']:+.1f}, {c['selected_minus_text']['hi']:+.1f}] | "
                     f"{c['selected_minus_mean_sham']['estimate']:+.1f} [{c['selected_minus_mean_sham']['lo']:+.1f}, {c['selected_minus_mean_sham']['hi']:+.1f}] | {v['verdict']} |")
    (out / 'expD_results.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    main()
