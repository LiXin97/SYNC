"""Experiment E (latest0924) bank: the 144 prespecified two-object scenes of Experiment B (tiers 1+2) plus its four
engineering scenes, and a derived three-object version of each. Deterministic; refuses to overwrite different content.

Three-object derivation (decided before any outcome): object C is added with, for each field, the first value in
sorted order of that domain's value pool that neither A nor B uses in the scene, so C never holds a queried value and
never matches either gold; C's facts are identical in both worlds. C's alias is drawn deterministically from the pool
of aliases used anywhere in the R3 confirmation bank (sha256 of the scene id indexes the sorted pool), excluding the
scene's own aliases. Questions, queried values and golds are unchanged, so the four-answer counterfactual logic holds.

Usage: python3 scripts/latest0924/layout/build_layout_bank.py [--check]
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from syncomm.latest.layout_render import RENDER, CONDITIONS, fact_words, condition_slots, render_standard  # noqa: E402

GRADED = ROOT / 'artifacts/latest0924/graded'
R3_BANK = ROOT / 'artifacts/receiver_recovery/r3_bank/bank.json'
OUT = ROOT / 'artifacts/latest0924/layout'
STUDY = 'latest0924_layout'
_spec = importlib.util.spec_from_file_location('memory_layout', ROOT / 'artifacts/mechanism0920/word_memory_fit_v1/snapshot/memory_layout.py')
memory_layout = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(memory_layout)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
    gm = json.loads((GRADED / 'qwen3-32b/manifest.json').read_text())
    words_file = Path(gm['words_file'])
    words = json.loads(words_file.read_text())
    gb = json.loads((GRADED / 'bank.json').read_text())
    r3 = json.loads(R3_BANK.read_text())
    tiers = {1, 2}
    scenes2 = [s for s in gb['scenes'] if gb['tier_of'][s['scene']] in tiers]
    eng2 = list(gb['engineering_scenes'])
    assert len(scenes2) == 144 and len(eng2) == 4 and [s['scene'] for s in eng2] == gb['engineering_scene_ids']
    pools = {}
    for s in r3:
        for f, pair in s['family_pairs'].items():
            pools.setdefault((s['domain'], f), set()).update(pair)
    alias_pool = sorted({al for s in r3 for al in s['aliases']})

    def three(scene):
        t = copy.deepcopy(scene)
        used = {f: set(scene['family_pairs'][f]) for f in scene['fields']}
        c_facts = {f: sorted(pools[(scene['domain'], f)] - used[f])[0] for f in scene['fields']}
        idx = int(hashlib.sha256(scene['scene'].encode()).hexdigest(), 16)
        cands = [x for x in alias_pool if x not in scene['aliases']]
        c_alias = cands[idx % len(cands)]
        t['aliases'] = scene['aliases'] + [c_alias]
        for w in t['worlds']:
            w['facts'] = w['facts'] + [dict(c_facts)]
        t['third_object'] = dict(alias=c_alias, facts=c_facts)
        t['scene_source'] = scene['scene']
        return t

    scenes3 = [three(s) for s in scenes2]
    eng3 = [three(s) for s in eng2]
    # construction checks
    checks = dict(l0_equals_frozen_render=0, l1_neutral_equals_l0=0, spans_per_condition={}, c_distinct=0, worlds=0)
    for scene in scenes2 + eng2:
        n_al, _ = render_standard(scene['aliases'])
        f_al, _ = memory_layout.render(scene['aliases'], None, width=4)
        assert n_al == f_al
        for w in scene['worlds']:
            wbo = fact_words(w, words)
            ow, entries = condition_slots('L0', wbo)
            t0, s0 = render_standard(scene['aliases'], ow)
            tf, sf = memory_layout.render(scene['aliases'], ow, width=4)
            assert t0 == tf and s0 == sf
            checks['l0_equals_frozen_render'] += 1
            for c in ('L0', 'L1', 'L2'):
                ow, entries = condition_slots(c, wbo)
                text, spans = RENDER[c](scene['aliases'], ow)
                assert len(spans) == 8 and {(e[0], e[1]) for e in entries} <= {(s['object'], s['index']) for s in spans}
                for o, i, wd in entries:
                    assert [s for s in spans if s['object'] == o and s['index'] == i][0]['word'] == wd
                checks['spans_per_condition'][c] = checks['spans_per_condition'].get(c, 0) + 1
    for scene in scenes3 + eng3:
        assert len(scene['aliases']) == 3
        c = scene['third_object']['facts']
        for w in scene['worlds']:
            assert w['facts'][2] == c
            for f in scene['fields']:
                assert c[f] not in {w['facts'][0][f], w['facts'][1][f]}
            assert scene['queried_values'][0] != c[scene['fields'][0]] and all(v not in c.values() for v in scene['queried_values'])
            wbo = fact_words(w, words)
            ow, entries = condition_slots('L3', wbo)
            text, spans = RENDER['L3'](scene['aliases'], ow)
            assert len(spans) == 12
            checks['worlds'] += 1
        checks['c_distinct'] += 1
    for scene in scenes2 + eng2:
        assert RENDER['L1'](scene['aliases'])[0] == RENDER['L0'](scene['aliases'])[0]
        checks['l1_neutral_equals_l0'] += 1
    bank = dict(study=STUDY, conditions=list(CONDITIONS), words_file=str(words_file), words_sha256=sha(words_file),
                source_graded_bank=str(GRADED / 'bank.json'), source_graded_bank_sha256=sha(GRADED / 'bank.json'),
                source_r3_bank=str(R3_BANK), source_r3_bank_sha256=sha(R3_BANK), tiers=sorted(tiers),
                engineering_scene_ids=[s['scene'] for s in eng2],
                scenes2=scenes2, eng2=eng2, scenes3=scenes3, eng3=eng3, construction_checks=checks,
                renderer=str(ROOT / 'src/syncomm/latest/layout_render.py'), renderer_sha256=sha(ROOT / 'src/syncomm/latest/layout_render.py'),
                builder_sha256=sha(Path(__file__)))
    text = json.dumps(bank, indent=1, sort_keys=True) + '\n'
    out = OUT / 'bank.json'
    if a.check:
        assert out.read_text() == text, 'stale bank'; print('PASS bank'); return
    OUT.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.read_text() != text:
        raise SystemExit(f'REFUSE: {out} exists with different content')
    out.write_text(text)
    print('WROTE', out, json.dumps(checks))


if __name__ == '__main__':
    main()
