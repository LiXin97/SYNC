"""P0-B conjunction banks (the study protocol §5) — deterministic, no model.

Every scene: four painted blocks carrying ALL FOUR combinations of two colors x two materials, so every single attribute
value is held by exactly two objects and only a color+material conjunction picks out one object. World 1 exchanges one
attribute between the two objects that share the other attribute's value (the example in the plan swaps the colors of the
two glass blocks); the four combinations stay present, so both worlds remain component-ambiguous. Two conjunction queries:
  q0 CHANGED   — the world-0 combination of one swapped object (its answer moves to the other swapped object)
  q1 UNCHANGED — the combination of one non-swapped object (same answer in both worlds)
Each query is answered from its own cache branch, so q0/q1 order is bookkeeping, not a causal variable.

Design per (color pair, material pair) group: changed attribute (2) x which shared-value pair is swapped (2) x changed-query
target (2) x unchanged-query target (2) = 16 scenes; 36 groups -> 576. Object positions come from a per-scene permutation
(balanced answer positions); twelve evidence templates are CROSSED with the groups (each template 48 scenes); two query
wordings alternate (color-first / material-first) and are parsed back symbolically. Gold, uniqueness, component ambiguity
and unchanged-question truth are verified by assertion before anything is written.

Outputs: the evaluation bank (576 + 4 separate engineering scenes) and a 96-scene development bank (+4 engineering) built
from a disjoint namespace and alias pool.
"""
import argparse
import hashlib
import itertools
import json
import re
from collections import Counter
from pathlib import Path

COLORS = ('green', 'orange', 'red', 'yellow')
MATERIALS = ('plastic', 'glass', 'rubber', 'copper')
FRAMES = ('Block {a} is painted {color} and is made of {material}.',
          'Block {a}: {material} body, {color} surface paint.',
          'The block called {a} has {color} paint over a {material} body.',
          'Made of {material}, block {a} carries a {color} coat of paint.',
          'Catalogue line for {a}: surface paint {color}; material {material}.',
          'Block {a} was cast in {material} and then painted {color}.',
          'On the shelf sits {a}, a {material} block with {color} paint.',
          'Paint on block {a}: {color}. Material of block {a}: {material}.',
          '{a} is a block; its paint is {color} and its material is {material}.',
          'Block {a}, {color}-painted, has a body of {material}.',
          'Inventory card {a}: a block of {material}, painted {color}.',
          'The {color} paint on block {a} covers a body made of {material}.')
QUERIES = ('Which block has {color} surface paint and is made of {material}?',
           'Which block is made of {material} and has {color} surface paint?')
PERMS = list(itertools.permutations(range(4)))
PERM_A, PERM_B = 1, 6   # chosen by a design-only search (no model output): all 24 permutations used, answer positions and position-by-combination counts exactly balanced (144 each)


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def alias_pool(n, taken, words, seed):
    """Deterministic 4-letter uppercase names (C V C C), disjoint from `taken` and the inventory."""
    cons = 'BDFGHKLMNPRSTVZ'; vows = 'AEIOU'; out = []
    k = seed
    while len(out) < n:
        h = hashlib.sha256(f'{seed}:{k}'.encode()).digest(); k += 1
        a = cons[h[0] % 15] + vows[h[1] % 5] + cons[h[2] % 15] + cons[h[3] % 15]
        if a in taken or a.lower() in words or a in out or a[2] == a[3] or any(x[:2] == a[:2] or x[2:] == a[2:] for x in out):
            continue
        out.append(a)
    return out


def parse_query(q):
    for pat in QUERIES:
        rx = '^' + re.escape(pat).replace(r'\{color\}', '(?P<color>[a-z]+)').replace(r'\{material\}', '(?P<material>[a-z]+)') + '$'
        m = re.match(rx, q)
        if m:
            return m.group('color'), m.group('material')
    raise AssertionError(q)


def scene(ns, group_index, cp, mp, changed, pair_sel, qc, qu, template, perm, wording, names):
    (c1, c2), (m1, m2) = cp, mp
    combos = [(c1, m1), (c1, m2), (c2, m1), (c2, m2)]
    objs = [combos[perm[p]] for p in range(4)]                    # position p holds combos[perm[p]]
    facts0 = [dict(color=c, material=m) for c, m in objs]
    if changed == 'color':
        shared = (m1, m2)[pair_sel]; swap = [p for p in range(4) if facts0[p]['material'] == shared]
    else:
        shared = (c1, c2)[pair_sel]; swap = [p for p in range(4) if facts0[p]['color'] == shared]
    assert len(swap) == 2
    facts1 = [dict(f) for f in facts0]
    facts1[swap[0]][changed], facts1[swap[1]][changed] = facts0[swap[1]][changed], facts0[swap[0]][changed]
    keep = [p for p in range(4) if p not in swap]
    q_changed = (facts0[swap[qc]]['color'], facts0[swap[qc]]['material'])
    q_unchanged = (facts0[keep[qu]]['color'], facts0[keep[qu]]['material'])
    questions = [QUERIES[wording].format(color=q_changed[0], material=q_changed[1]),
                 QUERIES[wording].format(color=q_unchanged[0], material=q_unchanged[1])]
    worlds = []
    for w, facts in enumerate((facts0, facts1)):
        answers = []
        for q in questions:
            c, m = parse_query(q)                                   # symbolic check of the wording
            hits = [names[p] for p in range(4) if facts[p]['color'] == c and facts[p]['material'] == m]
            assert len(hits) == 1, 'conjunction not unique'
            assert sum(f['color'] == c for f in facts) == 2 and sum(f['material'] == m for f in facts) == 2, 'a single attribute determines the answer'
            answers.append(hits[0])
        assert sorted((f['color'], f['material']) for f in facts) == sorted(combos), 'a combination went missing'
        evidence = [FRAMES[template].format(a=names[p], **facts[p]) for p in range(4)]
        worlds.append(dict(world=w, facts=facts, evidence=evidence, answers=answers))
    assert worlds[0]['answers'][0] != worlds[1]['answers'][0], 'changed query did not change'
    assert worlds[0]['answers'][1] == worlds[1]['answers'][1], 'unchanged query changed'
    fam = digest([ns, 'group', cp, mp])[:16]
    sid = digest([ns, cp, mp, changed, pair_sel, qc, qu, template, list(perm), wording])[:20]
    return dict(scene=sid, family=fam, split=ns, domain='block4conj', template=template, aliases=names, fields=['color', 'material'],
                changed_field=changed, swapped_positions=swap, conjunctions=[list(q_changed), list(q_unchanged)],
                queried_values=[f'{q_changed[0]}+{q_changed[1]}', f'{q_unchanged[0]}+{q_unchanged[1]}'], questions=questions, worlds=worlds,
                design_bits=dict(group=group_index, changed=changed, pair_sel=pair_sel, qc=qc, qu=qu, perm=list(perm), wording=wording))


def build(ns, names_pool, groups, per_group_bits, template_offset):
    bank = []; k = 0
    for gi, (cp, mp) in enumerate(groups):
        for bi, (changed, pair_sel, qc, qu) in enumerate(per_group_bits):
            template = (gi + bi + template_offset) % 12
            perm = PERMS[(PERM_A * gi + PERM_B * bi) % 24]
            names = [names_pool[(k + 6 * o) % len(names_pool)] for o in range(4)]
            bank.append(scene(ns, gi, cp, mp, changed, pair_sel, qc, qu, template, perm, (gi + bi) % 2, names)); k += 1
    return bank


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, required=True); ap.add_argument('--words', type=Path, required=True); ap.add_argument('--exclusions', type=Path, required=True)
    a = ap.parse_args(); out = a.out_dir; out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'eval_bank.json').exists(), 'refusing to overwrite'
    words = set(json.loads(a.words.read_text())); prior = json.loads(a.exclusions.read_text())['union']
    assert set(COLORS) <= words and set(MATERIALS) <= words
    taken = set(prior['aliases'])
    pool_eval = alias_pool(24, taken, words, 11); taken |= set(pool_eval)
    pool_dev = alias_pool(24, taken, words, 5003); taken |= set(pool_dev)
    pool_eng = alias_pool(8, taken, words, 9001)
    groups = [(cp, mp) for cp in itertools.combinations(COLORS, 2) for mp in itertools.combinations(MATERIALS, 2)]
    bits = list(itertools.product(('color', 'material'), (0, 1), (0, 1), (0, 1)))
    ev = build('p0b_eval', pool_eval, groups, bits, 0)
    assert len(ev) == 576 and len({s['scene'] for s in ev}) == 576 and len({s['family'] for s in ev}) == 36
    eng = build('p0b_eval_engineering', pool_eng, groups[:1], bits[:4], 3)
    dev_groups = [groups[6 * i + i] for i in range(6)]            # one group per color pair, material pair cycling
    dv = build('p0b_dev', pool_dev, dev_groups, bits, 5)
    assert len(dv) == 96
    dv_eng = build('p0b_dev_engineering', pool_eng, groups[1:2], bits[:4], 8)
    for name, bank, engs in (('eval_bank.json', ev, eng), ('dev_bank.json', dv, dv_eng)):
        full = engs + bank
        for s in full: s['split'] = 'p0b'
        (out / name).write_text(json.dumps(dict(scenes=full, engineering_scene_ids=[s['scene'] for s in engs]), indent=1, sort_keys=True) + '\n')
    rep = {}
    for name, bank in (('eval', ev), ('dev', dv)):
        rep[name] = dict(scenes=len(bank), groups=len({s['family'] for s in bank}), templates=dict(Counter(s['template'] for s in bank)),
                         changed_field=dict(Counter(s['changed_field'] for s in bank)), wording=dict(Counter(s['design_bits']['wording'] for s in bank)),
                         changed_answer_position_w0=dict(Counter(s['aliases'].index(s['worlds'][0]['answers'][0]) for s in bank)),
                         unchanged_answer_position=dict(Counter(s['aliases'].index(s['worlds'][0]['answers'][1]) for s in bank)),
                         group_x_template_counts=dict(Counter(Counter((s['family'], s['template']) for s in bank).values())))
    sk = set()
    for s in ev + dv:
        for w in s['worlds']:
            for e in w['evidence']:
                x = e
                for al in s['aliases']: x = x.replace(al, '<A>')
                for f in w['facts']:
                    for v in f.values(): x = re.sub(r'\b' + re.escape(v) + r'\b', '<V>', x)
                sk.add(x)
    rep.update(alias_pools=dict(eval=pool_eval, dev=pool_dev, engineering=pool_eng), aliases_shared_with_prior=sorted((set(pool_eval) | set(pool_dev) | set(pool_eng)) & set(prior['aliases'])),
               skeletons=sorted(sk), skeletons_shared_with_prior=sorted(sk & set(prior['skeletons'])),
               verification='asserted per scene and world: conjunction answer unique; each queried color and each queried material held by exactly two objects; all four combinations present in both worlds; changed answer moves, unchanged answer stays; query wording parsed back symbolically',
               reused_by_design='the four colors and the four approved materials (inventory words), the painted-block object class and the "surface paint"/"made of" question vocabulary; not an open-vocabulary test',
               eval_sha256=hashlib.sha256((out / 'eval_bank.json').read_bytes()).hexdigest(), dev_sha256=hashlib.sha256((out / 'dev_bank.json').read_bytes()).hexdigest())
    (out / 'report.json').write_text(json.dumps(rep, indent=1) + '\n')
    for name in ('eval', 'dev'):
        print(name, {k: v for k, v in rep[name].items()})
    print('aliases shared with prior:', rep['aliases_shared_with_prior'], '| skeletons shared:', rep['skeletons_shared_with_prior'])


if __name__ == '__main__':
    main()
