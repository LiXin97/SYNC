"""Analysis v2 of the Experiment B text-format follow-up (gates/latest0924/expB_textfmt.lock.json incl. amendments[0]).

Supersedes analyze_textfmt.py (kept unchanged): identical rules for F1-F3, plus amendment 1 arms (F4 sorted, F5 hedged,
embedding route) and a Holm family of 16 (8 per receiver: pct, verbal, instr, instr_internal, sorted, hedged,
internal_minus_emb, emb_minus_text).

Statistics are Experiment B's, imported unchanged from scripts/latest0924/analyze_graded.py (answer parser, cluster
bootstrap `env`/`clustered`/`contrast`, `holm`): four-answer success J per scene at 64 tokens, family-macro over 36
families and template-macro over 12 domain x template clusters, 20,000 draws x seeds 20260924-6, enclosing interval;
p = larger of the two clustering levels' two-sided bootstrap p.

Primary contrasts (Holm family of 8, alpha 0.05), per receiver in {qwen3-32b, qwen3.5-27b}:
  pct      mean over L in {64,128,191} of J_internal(L) [Experiment B internal_L* rows] - J_textpct(L)
  verbal   ... - J_textverbal(L)
  instr    ... - J_textinstr(L)                         (original internal vs instructed text)
  instr_internal  mean over L of J_internal_instr(L) - J_textinstr(L)   (both instructed)
Labels: "internal still better" (Holm reject and enclosing lower > 0); "text better" (Holm reject and enclosing
upper < 0); "similar" (enclosing interval inside [-3, +3] points); otherwise "unresolved". A contrast whose text arm
(or internal_instr arm) fails the L=0 positive control (J >= 0.90 and coverage >= 0.90) or the L=255 structural zero is
labelled NOT INTERPRETABLE (numbers still reported). Results are reported whichever way they come out.
Usage: python3 scripts/latest0924/textfmt/analyze_textfmt.py
"""
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
_spec = importlib.util.spec_from_file_location('analyze_graded', ROOT / 'scripts/latest0924/analyze_graded.py')
ag = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ag)

OUT = ROOT / 'artifacts/latest0924/graded_textfmt'
EXPB = ROOT / 'artifacts/latest0924/graded'
LOCK = ROOT / 'gates/latest0924/expB_textfmt.lock.json'
RECEIVERS = ('qwen3-32b', 'qwen3.5-27b')
LEVELS = (0, 64, 128, 191, 255)
POOLED = (64, 128, 191)

N_SCENES = 144


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(root, phase, rec, arms_expected=None):
    rows, files = {}, {}
    for f in sorted(Path(root).glob(f'raw/{phase}_{rec}_s*.jsonl')):
        receipt = f.with_suffix('.done.json')
        assert receipt.exists() and json.loads(receipt.read_text())['status'] == 'PASS', f'missing/failed receipt {receipt}'
        files[str(f)] = sha(f)
        for line in open(f):
            r = json.loads(line)
            k = (r['scene'], r['world'], r['question'], r['arm'])
            assert k not in rows, f'duplicate row {k}'
            rows[k] = r
    return rows, files


def per_scene_J(rows, scenes, arm):
    return ag.scene_metrics(rows, scenes, arm)


def main():
    lock = json.loads(LOCK.read_text())
    report = dict(lock=str(LOCK.relative_to(ROOT)), lock_sha256=sha(LOCK), script_sha256=sha(Path(__file__)),
                  analyze_graded_sha256=sha(ROOT / 'scripts/latest0924/analyze_graded.py'), receivers={}, inputs={})
    pvals, contrasts = {}, {}
    for rec in RECEIVERS:
        tf, f1 = load(OUT / rec, 'development', rec)
        eb, f2 = load(EXPB / rec, 'development', rec)
        te, f3 = load(OUT / rec, 'engineering', rec)
        ta, f5 = load(OUT / rec / 'a1', 'development', rec)
        tae, f6 = load(OUT / rec / 'a1', 'engineering', rec)
        report['inputs'].update(f5); report['inputs'].update(f6)
        assert sorted({k[0] for k in ta}) == sorted({k[0] for k in tf}), 'amendment scene set differs'
        arms_a1 = sorted({k[3] for k in ta})
        assert len(ta) == N_SCENES * 4 * len(arms_a1), (len(ta), len(arms_a1))
        ee, f4 = load(EXPB / rec, 'engineering', rec)
        report['inputs'].update(f1); report['inputs'].update(f2); report['inputs'].update(f3); report['inputs'].update(f4)
        scenes = sorted({k[0] for k in tf})
        assert len(scenes) == N_SCENES and scenes == sorted({k[0] for k in eb}), 'scene sets differ'
        arms = sorted({k[3] for k in tf})
        assert len(tf) == N_SCENES * 4 * len(arms), (len(tf), len(arms))
        meta = {s: tf[(s, 0, 0, 'none')] for s in scenes}
        levels = {'family': {s: meta[s]['family'] for s in scenes},
                  'template': {s: f"{meta[s]['domain']}/T{meta[s]['template']}" for s in scenes}}
        assert len(set(levels['family'].values())) == 36 and len(set(levels['template'].values())) == 12
        # engineering: checks recomputed from rows (receipts not trusted), identity inertness, continuity with Exp B
        te_all = dict(te); te_all.update({('a1',) + k: r for k, r in tae.items()})
        eng = dict(rows=len(te_all), failing=[k for k, r in te_all.items() if r['engineering_checks'] and not r['engineering_checks']['pass']],
                   max_logit_difference=max((r['engineering_checks']['max_logit_difference'] for r in te_all.values() if r['engineering_checks']), default=None))
        cont = defaultdict(int)
        for k, r in list(te.items()) + list(tae.items()):
            if k[3] in ('internal_L064', 'text_L064'):
                ref = ee.get(k)
                same = ref is not None and ref['logits_sha256'] == r['logits_sha256'] and ref['tokens'] == r['tokens']
                cont[f'{k[3]}:{"identical" if same else "different"}'] += 1
        for k, r in ta.items():
            if k[3] == 'emb_L000':
                ref = eb.get(k[:3] + ('text_L000',))
                same = ref is not None and ref['logits_sha256'] == r['logits_sha256'] and ref['tokens'] == r['tokens']
                cont[f'emb_L000_vs_expB_text_L000:{"identical" if same else "different"}'] += 1
        eng['continuity_with_expB'] = dict(cont)
        # estimates
        est = {}
        def curve(rows, arm):
            per = per_scene_J(rows, scenes, arm)
            return {m: ag.clustered(per[m], levels)['family'] for m in ('J', 'coverage', 'J256')}, per
        per_all = {}
        for L in LEVELS:
            for name, arm, rows in [('internal', f'internal_L{L:03d}', eb), ('text_expB', f'text_L{L:03d}', eb),
                                    ('textpct', f'textpct_L{L:03d}', tf), ('textverbal', f'textverbal_L{L:03d}', tf),
                                    ('textinstr', f'textinstr_L{L:03d}', tf), ('internal_instr', f'internal_instr_L{L:03d}', tf),
                                    ('textsorted', f'textsorted_L{L:03d}', ta), ('texthedged', f'texthedged_L{L:03d}', ta),
                                    ('emb', f'emb_L{L:03d}', ta)]:
                e, per = curve(rows, arm)
                est.setdefault(name, {})[str(L)] = e
                per_all[name, L] = per
        for name, arm, rows in [('none', 'none', tf), ('none_instr', 'none_instr', tf), ('none_a1', 'none', ta)]:
            e, per = curve(rows, arm); est[name] = e; per_all[name, None] = per
        # controls
        ctrl = {}
        for name in ('textpct', 'textverbal', 'textinstr', 'internal_instr', 'internal', 'text_expB', 'textsorted', 'texthedged', 'emb'):
            j0, c0 = est[name]['0']['J']['mean'], est[name]['0']['coverage']['mean']
            j255 = sum(per_all[name, 255]['J'].values())
            ctrl[f'positive_L0_{name}'] = dict(pass_=j0 >= ag.POSITIVE_J and c0 >= ag.POSITIVE_COVERAGE, J=j0, coverage=c0)
            ctrl[f'negative_L255_{name}'] = dict(pass_=j255 == 0, J_sum=j255)
        for name in ('none', 'none_instr', 'none_a1'):
            ctrl[f'no_message_{name}'] = dict(pass_=sum(per_all[name, None]['J'].values()) == 0)
        l255 = [r for r in list(tf.values()) + list(ta.values()) if r['level'] == 255 and r['world'] == 1]
        ctrl['l255_outputs_identical_across_worlds'] = dict(pass_=all(r['l255_world_identical'] for r in l255), pairs=len(l255))
        # contrasts
        def pooled_diff(a_name, b_name):
            return {s: float(np.mean([per_all[a_name, L]['J'][s] - per_all[b_name, L]['J'][s] for L in POOLED])) for s in scenes}
        cdefs = {'pct': ('internal', 'textpct'), 'verbal': ('internal', 'textverbal'), 'instr': ('internal', 'textinstr'),
                 'instr_internal': ('internal_instr', 'textinstr'), 'sorted': ('internal', 'textsorted'),
                 'hedged': ('internal', 'texthedged'), 'internal_minus_emb': ('internal', 'emb'),
                 'emb_minus_text': ('emb', 'text_expB')}
        rc = {}
        for cname, (a_name, b_name) in cdefs.items():
            c = ag.contrast(pooled_diff(a_name, b_name), levels)
            key = f'{rec}:{cname}'
            rc[cname] = dict(c, minuend=a_name, subtrahend=b_name,
                             controls_needed=[f'positive_L0_{a_name}', f'positive_L0_{b_name}',
                                              f'negative_L255_{a_name}', f'negative_L255_{b_name}'])
            pvals[key] = c['enclosing']['p']; contrasts[key] = rc[cname]
        # descriptive: how much each text format gains over Experiment B's text
        desc = {f'{n}_minus_text_expB': ag.contrast(pooled_diff(n, 'text_expB'), levels)['enclosing']
                for n in ('textpct', 'textverbal', 'textinstr', 'textsorted', 'texthedged', 'emb')}
        report['receivers'][rec] = dict(scenes=len(scenes), arms=arms + arms_a1, engineering=eng, estimates=est, controls=ctrl,
                                        contrasts=rc, descriptive=desc)
    H = ag.holm(pvals)
    verdicts = {}
    for key, c in contrasts.items():
        rec = key.split(':')[0]; e = c['enclosing']; h = H[key]
        a_name, b_name = c['minuend'], c['subtrahend']
        if h['reject'] and e['lo'] > 0:
            call = 'internal still better' if a_name.startswith('internal') and b_name.startswith('text') else f'{a_name} better'
        elif h['reject'] and e['hi'] < 0:
            call = 'text better' if a_name.startswith('internal') and b_name.startswith('text') else f'{b_name} better'
        elif e['lo'] >= -ag.MARGIN and e['hi'] <= ag.MARGIN:
            call = 'similar'
        else:
            call = 'unresolved'
        ctrl = report['receivers'][rec]['controls']
        failed = [k for k in c['controls_needed'] if not ctrl[k]['pass_']]
        verdicts[key] = dict(verdict=call if not failed else f'NOT INTERPRETABLE ({call})', failed_controls=failed,
                             estimate=e['mean'], lo=e['lo'], hi=e['hi'], p=e['p'], holm=h)
    report['holm'] = H; report['verdicts'] = verdicts
    assert len(pvals) == 16, len(pvals)
    (OUT / 'analysis_v2.json').write_text(json.dumps(report, indent=1) + '\n')
    pc = lambda x: f'{100 * x:.1f}'
    md = ['# Experiment B text-format follow-up — analysis v2 (F1-F5 + embedding)', '',
          f"Lock `{report['lock']}` sha `{report['lock_sha256'][:12]}`; Holm family of {len(pvals)}; intervals enclose "
          f"family (36) and template (12) cluster bootstraps, 20,000 draws x 3 seeds.", '',
          '| contrast | estimate [enclosing 95%] | p | Holm thr | verdict |', '|---|---|---|---|---|']
    for key, v in verdicts.items():
        md.append(f"| {key} | {pc(v['estimate'])} [{pc(v['lo'])}, {pc(v['hi'])}] | {v['p']:.3g} | {v['holm']['threshold']:.4g} | {v['verdict']} |")
    for rec in RECEIVERS:
        R = report['receivers'][rec]
        md += ['', f'## {rec}', '', '| route | ' + ' | '.join(f'L={L}' for L in LEVELS) + ' |', '|---|' + '---|' * len(LEVELS)]
        for name in ('internal', 'text_expB', 'textpct', 'textverbal', 'textinstr', 'internal_instr', 'textsorted', 'texthedged', 'emb'):
            md.append(f'| {name} J | ' + ' | '.join(pc(R['estimates'][name][str(L)]['J']['mean']) for L in LEVELS) + ' |')
        md += ['', 'Controls: ' + '; '.join(f"{k}: {'PASS' if v['pass_'] else 'FAIL'}" for k, v in R['controls'].items()),
               '', f"Engineering: {R['engineering']}", '',
               'Text-format gain over Experiment B text (pooled L=64/128/191): ' +
               '; '.join(f"{k} {pc(v['mean'])} [{pc(v['lo'])}, {pc(v['hi'])}]" for k, v in R['descriptive'].items())]
    (OUT / 'analysis_v2.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
