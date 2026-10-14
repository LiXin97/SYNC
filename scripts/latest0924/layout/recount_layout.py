"""Experiment E independent recount (Hard Rule 2): recompute every J point estimate from the raw rows with code that
shares nothing with analyze_layout.py, and compare with artifacts/latest0924/layout/analysis/layout_results.json.

Correctness is re-derived as parsed answer == gold (and checked against the stored flag); J per scene = all four
(world, question) rows correct; family-macro = mean over families of the per-family mean.

Usage: python3 scripts/latest0924/layout/recount_layout.py
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/latest0924/layout'


def main():
    res = json.loads((OUT / 'analysis/layout_results.json').read_text())
    worst = 0.0; n = 0; flag_mismatch = 0
    for rec, R in res['receivers'].items():
        if R['status'] != 'COMPLETE':
            print(rec, R['status']); continue
        per = defaultdict(list); fam = {}
        for f in glob.glob(str(OUT / rec / 'raw' / 'development_*.jsonl')):
            for line in open(f):
                r = json.loads(line)
                ok = r['score64']['answer'] == r['gold']
                flag_mismatch += ok != r['score64']['correct']
                per[r['scene'], r['condition'], r['arm']].append(ok)
                fam[r['scene']] = r['family']
        assert all(len(v) == 4 for v in per.values()), rec
        by = defaultdict(lambda: defaultdict(list))
        for (s, c, a), v in per.items():
            by[c, a][fam[s]].append(float(all(v)))
        line = []
        for (c, a), fams in sorted(by.items()):
            J = sum(sum(x) / len(x) for x in fams.values()) / len(fams)
            d = abs(J - R['estimates'][f'{c}/{a}']['J']); worst = max(worst, d); n += 1
            if a in ('internal', 'text'):
                line.append(f'{c}/{a}={100 * J:.1f}')
        print(rec, ' '.join(line))
    if n == 0:
        raise SystemExit('No per-row records found under artifacts/latest0924/layout/*/raw/; '
                         'download the SYNC-data dataset first (see README).')
    print(f'compared {n} J cells; max |recount - analysis| = {worst:.2e}; stored-flag mismatches = {flag_mismatch}')
    assert worst < 1e-12 and flag_mismatch == 0
    print('RECOUNT PASS')


if __name__ == '__main__':
    main()
