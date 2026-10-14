"""Analyze the D1/D2 diagnostics exactly as frozen in gates/latest0924/diag_protocol.lock.json.

Endpoint: four-answer (joint) success per scene (both worlds x both questions correct at the 64-token budget),
averaged within development family and then across families (family-macro, as in R1); coverage = fraction of the
four answers that parse. Intervals: paired cluster bootstrap over families, 20,000 draws for each of the seeds
20260924..20260926, reported as the enclosing interval (min lower, max upper over seeds).

D1 (all four receivers, sources fitted and native, cut L in d35/d50/d65):
  A = write at d20; B(L) = write at L; C(L) = write at d20, then prefix keys/values of layers <= L replaced by neutral.
  processing_suffices_at_L:       C-B >= 20 and A-C <= 10 and lower(C-B) > 0
  lower_layer_reading_needed_at_L: A-C >= 20 and lower(A-C) > 0
  otherwise: unresolved/partial.
D2 (glm-4-9b primary, qwen3-14b-base secondary; depths d20/d35/d50):
  G_layout = J(matched) - J(fixed); G_joint = J(native) - J(matched); G_delivery = J(text) - J(native).
Built-in checks (lock): C at L=d20 byte-identical prefix cache to A; C at the last layer reproduces no-message outputs;
identity write inert; the engineering receipts (cached vs complete forward) exist.

Usage: python3 scripts/latest0924/analyze_diag.py   (writes artifacts/latest0924/diag/analysis/diag_results.{json,md})
"""
import glob
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
LOCK = ROOT / 'gates/latest0924/diag_protocol.lock.json'
DIAG = ROOT / 'artifacts/latest0924/diag'
OUT = DIAG / 'analysis'
SEEDS = (20260924, 20260925, 20260926)
DRAWS = 20000
RECEIVERS = ['qwen3-14b-base', 'glm-4-9b', 'gemma-3-27b-it', 'qwen3-32b']
D2_RECEIVERS = ['glm-4-9b', 'qwen3-14b-base']


def load(rec):
    rows = []
    for f in sorted(glob.glob(str(DIAG / rec / 'raw' / 'development_*.jsonl'))):
        rows += [json.loads(l) for l in open(f)]
    keys = [r['key'] for r in rows]
    assert len(keys) == len(set(keys)), f'{rec}: duplicate row keys'
    return rows


def cond_key(r):
    return (r['diagnostic'], r['arm'], r['source'], r['write_layer'], r['cut_layer'])


def per_scene(rows):
    """{condition: {scene: (family, joint, coverage)}} with all four answers required per scene."""
    acc = defaultdict(lambda: defaultdict(list))
    fam = {}
    for r in rows:
        acc[cond_key(r)][r['scene']].append(r)
        fam[r['scene']] = r['family']
    out = {}
    for c, scenes in acc.items():
        d = {}
        for s, rs in scenes.items():
            assert len(rs) == 4 and {(x['world'], x['question']) for x in rs} == {(0, 0), (0, 1), (1, 0), (1, 1)}, (c, s)
            d[s] = (fam[s], float(all(x['score64']['correct'] for x in rs)),
                    float(np.mean([x['score64']['answered'] for x in rs])))
        out[c] = d
    return out


class Boot:
    """Family-macro estimates and paired family-bootstrap differences over a fixed scene set."""

    def __init__(self, scenes_by_cond):
        any_c = next(iter(scenes_by_cond.values()))
        self.scenes = sorted(any_c)
        for c, d in scenes_by_cond.items():
            assert sorted(d) == self.scenes, f'{c}: scene set differs'
        self.families = sorted({any_c[s][0] for s in self.scenes})
        fidx = {f: i for i, f in enumerate(self.families)}
        self.sf = np.array([fidx[any_c[s][0]] for s in self.scenes])
        self.counts = np.bincount(self.sf, minlength=len(self.families)).astype(float)
        self.data = scenes_by_cond
        self.samples = [np.random.default_rng(seed).integers(0, len(self.families), (DRAWS, len(self.families)))
                        for seed in SEEDS]

    def fam_means(self, c, what=1):
        v = np.array([self.data[c][s][what] for s in self.scenes])
        return np.bincount(self.sf, weights=v, minlength=len(self.families)) / self.counts

    def est(self, c, what=1):
        return 100 * float(self.fam_means(c, what).mean())

    def diff(self, a, b):
        """100 * (macro(a) - macro(b)) with the enclosing 95% interval over seeds."""
        fd = self.fam_means(a) - self.fam_means(b)
        lo, hi = [], []
        for idx in self.samples:
            boots = fd[idx].mean(axis=1)
            lo.append(np.percentile(boots, 2.5)); hi.append(np.percentile(boots, 97.5))
        return dict(diff=100 * float(fd.mean()), lo=100 * float(min(lo)), hi=100 * float(max(hi)))


def checks(rec, rows, layers, n_layers_last):
    by = defaultdict(dict)
    for r in rows:
        by[cond_key(r)][(r['scene'], r['world'], r['question'])] = r
    none = by[('reference', 'none', None, None, None)]
    res = {}
    for src in ('fitted', 'native'):
        a = by[('D1_A', 'write', src, layers['d20'], None)]
        cw = [k for k in by if k[0] == 'check_cut_at_write_layer' and k[2] == src]
        cl = [k for k in by if k[0] == 'check_cut_at_last_layer' and k[2] == src]
        assert len(cw) == 1 and len(cl) == 1, (rec, src, cw, cl)
        cw, cl = by[cw[0]], by[cl[0]]
        res[f'{src}_cut_at_write_layer_cache_identical_to_A'] = all(
            cw[k]['prefix_cache_sha256'] == a[k]['prefix_cache_sha256'] for k in a)
        res[f'{src}_cut_at_write_layer_tokens_equal_A'] = all(cw[k]['tokens'] == a[k]['tokens'] for k in a)
        res[f'{src}_cut_at_last_layer_tokens_equal_no_message'] = all(cl[k]['tokens'] == none[k]['tokens'] for k in none)
    ident = by[('check', 'identity_write', None, None, None)]
    res['identity_write_tokens_equal_no_message'] = all(ident[k]['tokens'] == none[k]['tokens'] for k in none)
    res['identity_write_logits_equal_no_message'] = all(ident[k]['logits_sha256'] == none[k]['logits_sha256'] for k in none)
    eng = DIAG / rec / 'raw' / f'engineering_{rec}_s0.done.json'
    res['engineering_receipt'] = eng.exists() and eng.stat().st_size > 0
    return res


def d1_verdict(A_C, C_B):
    if C_B['diff'] >= 20 and A_C['diff'] <= 10 and C_B['lo'] > 0:
        return 'processing_suffices_at_L'
    if A_C['diff'] >= 20 and A_C['lo'] > 0:
        return 'lower_layer_reading_needed_at_L'
    return 'unresolved/partial'


def main():
    lock = json.loads(LOCK.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(lock=str(LOCK.relative_to(ROOT)), lock_sha256=hashlib.sha256(LOCK.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), seeds=SEEDS, draws=DRAWS,
                  receivers={})
    md = ['# D1/D2 diagnostics (development material; diagnostic, not confirmation)', '',
          f'Lock `{result["lock"]}` sha `{result["lock_sha256"][:12]}`. Four-answer success, family-macro (%), '
          f'95% paired family-bootstrap enclosing interval ({DRAWS} draws x seeds {SEEDS}).', '']
    for rec in RECEIVERS:
        L = lock['layers'][rec]
        rows = load(rec)
        sc = per_scene(rows)
        B = Boot(sc)
        chk = checks(rec, rows, L, None)
        R = dict(n_rows=len(rows), n_scenes=len(B.scenes), n_families=len(B.families), checks=chk,
                 reference=dict(none=B.est(('reference', 'none', None, None, None)),
                                text=B.est(('reference', 'text', None, None, None)),
                                text_coverage=B.est(('reference', 'text', None, None, None), 2)),
                 D1={}, D2={})
        md += [f'## {rec}  ({len(B.scenes)} scenes, {len(B.families)} families, {len(rows)} rows)', '',
               'Checks: ' + ', '.join(f'{k}={v}' for k, v in chk.items()), '',
               f'No message {R["reference"]["none"]:.1f}; complete-fact text {R["reference"]["text"]:.1f}.', '',
               '| source | L | A (d20) | B (write at L) | C (d20, cut L) | C-B [95% CI] | A-C [95% CI] | verdict |',
               '|---|---|---|---|---|---|---|---|']
        for src in ('fitted', 'native'):
            a = ('D1_A', 'write', src, L['d20'], None)
            for dname in ('d35', 'd50', 'd65'):
                b = ('D1_B', 'write', src, L[dname], None)
                c = ('D1_C', 'early_cut', src, L['d20'], L[dname])
                A_C, C_B = B.diff(a, c), B.diff(c, b)
                v = d1_verdict(A_C, C_B)
                R['D1'][f'{src}_{dname}'] = dict(A=B.est(a), B=B.est(b), C=B.est(c), A_cov=B.est(a, 2), B_cov=B.est(b, 2),
                                                 C_cov=B.est(c, 2), C_minus_B=C_B, A_minus_C=A_C, verdict=v)
                md.append(f'| {src} | {dname} | {B.est(a):.1f} | {B.est(b):.1f} | {B.est(c):.1f} | '
                          f'{C_B["diff"]:+.1f} [{C_B["lo"]:+.1f}, {C_B["hi"]:+.1f}] | '
                          f'{A_C["diff"]:+.1f} [{A_C["lo"]:+.1f}, {A_C["hi"]:+.1f}] | {v} |')
        md.append('')
        if rec in D2_RECEIVERS:
            text = ('reference', 'text', None, None, None)
            md += ['| depth | fixed | matched | native | text | G_layout [CI] | G_joint [CI] | G_delivery [CI] |',
                   '|---|---|---|---|---|---|---|---|']
            for dname in ('d20', 'd35', 'd50'):
                diag_fixed = 'D1_A' if dname == 'd20' else 'D1_B'
                fixed = (diag_fixed, 'write', 'fitted', L[dname], None)
                native = (diag_fixed, 'write', 'native', L[dname], None)
                matched = ('D2_matched', 'write', 'matched', L[dname], None)
                g = dict(G_layout=B.diff(matched, fixed), G_joint=B.diff(native, matched), G_delivery=B.diff(text, native))
                R['D2'][dname] = dict(fixed=B.est(fixed), matched=B.est(matched), native=B.est(native), text=B.est(text),
                                      matched_cov=B.est(matched, 2), **g)
                md.append(f'| {dname} | {B.est(fixed):.1f} | {B.est(matched):.1f} | {B.est(native):.1f} | {B.est(text):.1f} | '
                          + ' | '.join(f'{x["diff"]:+.1f} [{x["lo"]:+.1f}, {x["hi"]:+.1f}]' for x in g.values()) + ' |')
            md.append('')
        result['receivers'][rec] = R
    (OUT / 'diag_results.json').write_text(json.dumps(result, indent=1) + '\n')
    (OUT / 'diag_results.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
