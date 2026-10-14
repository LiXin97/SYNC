"""Experiment B (latest0924): graded-weight packets on the R3 confirmation bank.

Pure data preparation: deterministic, CPU-only, no model, no receiver output read. Refuses to overwrite an existing
output whose bytes differ (an identical rebuild is a no-op), so a locked bank cannot drift silently.

What it builds (artifacts/latest0924/graded/):
  bank.json                 the 572 evaluated scenes of artifacts/receiver_recovery/r3_bank/bank.json (verbatim), each
                            assigned to one of four nested, pre-declared TIERS by its design bits only, plus the 4
                            engineering scenes kept separately. The default evaluation set is tiers 1+2 (4 scenes per
                            family, 144 scenes); see DESIGN.md for why and for the fallback/extension rule.
  packets_L{level}.json     for level L in {0, 64, 128, 191, 255}: one 8-bit packet per scene|world (engineering scenes
                            included), serialized exactly as base.serialize, with its public text rendering.
  construction_report.json  counts, every construction check, the tier balance tables and the sha256 of every output.

Graded packet for one world (X = the scene's changed attribute, Y = the kept one; complete facts, no sender):
  object o carries  its true X value at weight 255,
                    the OTHER object's X value at weight L   (omitted when L = 0: the standard single-value packet,
                                                              byte-identical to the complete-fact packet of the paper),
                    its Y value at weight 255.
  Lane word order is ascending inventory index (the frozen packet format), so order cannot reveal which X value is
  strong: in every (scene, world) exactly one of the two lanes lists its strong X value first (checked).
  L = 255 makes the two worlds' packets byte-identical (X is ambiguous by construction; checked).

Text rendering: memory_layout.render hard-codes " [255/255]" after every slot, so it cannot carry graded weights.
render_weighted() below reproduces its exact format with the actual weight in each filled slot (neutral slots keep
"unknown [255/255]"), and is verified byte-identical -- text AND character spans -- to render() whenever all weights
are 255 (every scene, every world, the neutral layout, and 4,000 seeded random word lists).

Usage (CPU, dev box system python is enough):
  python3 scripts/latest0924/build_graded_bank.py            # build (or verify an identical existing build)
  python3 scripts/latest0924/build_graded_bank.py --check    # rebuild in memory and compare with the files on disk
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
FIT = ROOT / 'artifacts/mechanism0920/word_memory_fit_v1/snapshot'
for _p in (str(ROOT / 'src'), str(FIT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from memory_layout import render, packet_entries, WIDTH, NEUTRAL_WORD   # noqa: E402  frozen public layout

STUDY = 'LATEST0924_GRADED_B_V1'
OUT = ROOT / 'artifacts/latest0924/graded'
SOURCE_BANK = ROOT / 'artifacts/receiver_recovery/r3_bank/bank.json'
WORDS_FILE = ROOT / 'artifacts/receiver_recovery/words_221.json'
R3_MANIFEST = ROOT / 'artifacts/receiver_recovery/r3/qwen3-32b/manifest.json'
LAYOUT_FILE = FIT / 'memory_layout.py'
PACKET_SRC = ROOT / 'src/syncomm/packet.py'
PACKET_FROZEN = ROOT / 'artifacts/reviewbinding0917/snapshot/src/syncomm/packet.py'
PINS = {   # inputs, re-hashed on every build
    SOURCE_BANK: 'c88ff3197717a338eb84cbaec5c0a3c9fa9b5504f6ab077c1cb4ca78c5b15927',
    WORDS_FILE: 'b7c686ee1c47392ee2499f31c4784a4d3427e1d981a624e8ed77af49aca5767e',
    R3_MANIFEST: '01d8530f47af474e2affc43738032ab5ae307b973fd3e4c3f2adae68e7f783a1',
    LAYOUT_FILE: '774bbf465a772e4f3a7db3aa9de05304f8286e91d93edd73d5cab7bace1ee818',
}
HIGH = 255
LEVELS = (0, 64, 128, 191, 255)
DEFAULT_TIERS = (1, 2)
COLUMNS = ('c', 'fa', 'fb', 'q', 'fX', 'fY', 'sX', 'sY')
HALF_PRIORITY = ('c', 'sY', 'q', 'fX', 'fY', 'sX', 'fa', 'fb')


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path) -> str:
    return sha_bytes(Path(path).read_bytes())


def dumps(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True) + '\n').encode()


# ------------------------------------------------------------------------------------------------ public text renderer

def render_weighted(aliases, object_words=None, object_weights=None, width=WIDTH):
    """memory_layout.render with the ACTUAL 8-bit weight of every filled slot.

    Identical construction to render(): same header, same field separators, `unknown [255/255]` in every unfilled slot,
    same assertions, same character spans. `object_weights[o][i]` is the weight of `object_words[o][i]`; None means
    255 everywhere, in which case the output is byte-identical to render() (asserted by the builder).
    """
    assert width in (2, 4)
    assert len(aliases) == 2 and len(set(aliases)) == 2
    if object_words is None:
        object_words = [[], []]
    if object_weights is None:
        object_weights = [[HIGH] * len(w) for w in object_words]
    assert len(object_words) == 2 and len(object_weights) == 2
    text, spans = 'Selected words and exact relative weights (integers divided by 255): ', []
    for obj, (alias, words, weights) in enumerate(zip(aliases, object_words, object_weights)):
        assert len(words) <= width and len(weights) == len(words)
        assert isinstance(alias, str) and alias and '\n' not in alias
        if obj:
            text += '; '
        text += alias + ': '
        for index in range(width):
            if index:
                text += ', '
            word = words[index] if index < len(words) else NEUTRAL_WORD
            weight = weights[index] if index < len(words) else HIGH
            assert isinstance(word, str) and word and '\n' not in word
            assert isinstance(weight, int) and 0 <= weight <= 255
            start = len(text)
            text += word
            spans.append(dict(object=obj, index=index, word=word, start=start, end=len(text)))
            text += f' [{weight}/255]'
    text += '.'
    return text, spans


# ------------------------------------------------------------------------------------------------ design bits / tiers

def design_point(scene):
    b = scene['design_bits']
    return (b['changed'], b['flip_a'], b['flip_b'], b['qc'])


def derived(point):
    """(changed, flip_a, flip_b, qc) -> the eight design columns.

    fX / fY: which value of the (alphabetical = ascending-index) pair object 0 holds for X / Y in world 0;
    sX / sY: the world-0 answer object for the X / Y question (re-derived from the bank for every scene)."""
    c, fa, fb, q = point
    fX = fa if c == 0 else fb
    fY = fb if c == 0 else fa
    return dict(c=c, fa=fa, fb=fb, q=q, fX=fX, fY=fY, sX=q ^ fX, sY=q ^ fY)


def balanced(points, col):
    return 2 * sum(derived(p)[col] for p in points) == len(points)


def crossed(points, a, b):
    cnt = Counter((derived(p)[a], derived(p)[b]) for p in points)
    return len(cnt) == 4 and len(set(cnt.values())) == 1


def n_crossed(points):
    return sum(crossed(points, a, b) for a, b in itertools.combinations(COLUMNS, 2))


def select_tiers(avoid):
    """Outcome-blind tier rule, a function of the 2^4 design and the engineering scenes' design points only.

    P1 = the 4-point pattern (applied to every family) balanced on all eight columns, containing no design point that
         an engineering scene occupies, with the most fully crossed column pairs (ties: lexicographically smallest).
    P2 = the disjoint 4-point pattern with the same properties maximizing fully crossed pairs of P1+P2 (ties: lexico).
    Tier 1 / tier 2 = the split of P1 into two 2-point halves balanced first on c, then sY, q, fX, fY, sX, fa, fb
         (lexicographic comparison of the balance vector; ties: lexicographically smallest half first).
    Tier 3 = P2. Tier 4 = the remaining eight points (minus the engineering scenes).
    """
    allp = sorted(itertools.product((0, 1), repeat=4))
    cands = [S for S in itertools.combinations(allp, 4)
             if all(balanced(S, c) for c in COLUMNS) and not set(S) & avoid]
    assert cands, 'no fully balanced 4-point pattern avoids the engineering design points'
    p1 = sorted(cands, key=lambda S: (-n_crossed(S), S))[0]
    rest = [S for S in cands if not set(S) & set(p1)]
    assert rest
    p2 = sorted(rest, key=lambda S: (-n_crossed(tuple(sorted(p1 + S))), S))[0]
    halves = [H for H in itertools.combinations(p1, 2)]
    halves.sort(key=lambda H: (tuple(-int(balanced(H, c)) for c in HALF_PRIORITY), H))
    t1 = halves[0]
    t2 = tuple(p for p in p1 if p not in t1)
    t4 = tuple(p for p in allp if p not in p1 and p not in p2)
    tiers = {1: t1, 2: t2, 3: p2, 4: t4}
    assert sorted(sum(tiers.values(), ())) == allp
    return tiers, dict(p1=p1, p2=p2, candidates=len(cands), p1_crossed_pairs=n_crossed(p1),
                       p1p2_crossed_pairs=n_crossed(tuple(sorted(p1 + p2))))


# ------------------------------------------------------------------------------------------------ packets

def fields_of(scene):
    x = scene['changed_field']
    y = next(f for f in scene['fields'] if f != x)
    return x, y


def graded_packet(P, scene, wi, level, words, index):
    """One world's graded packet (see module docstring). Returns the frozen Packet object."""
    x, y = fields_of(scene)
    facts = scene['worlds'][wi]['facts']
    lanes = []
    for o in (0, 1):
        weights = {index[facts[o][x]]: HIGH, index[facts[o][y]]: HIGH}
        if level > 0:
            other = index[facts[1 - o][x]]
            assert other not in weights
            weights[other] = level
        ids = tuple(sorted(weights))
        lanes.append(P.Lane(o, ids, tuple(weights[i] for i in ids)))
    packet = P.Packet(P.inventory_digest(words), 8, tuple(lanes))
    assert P.serialize(P.deserialize(P.serialize(packet))) == P.serialize(packet)
    assert P.deserialize(P.serialize(packet)).lanes == packet.lanes
    return packet


def generic_literal(P, world, words, index):
    """evaluate_confirm.literal_packet_generic's construction, restated here without the model imports."""
    lanes = []
    for i, obj in enumerate(world['facts']):
        values = list(obj.values())
        lanes.append(P.Lane(i, tuple(sorted(index[v] for v in values)), (HIGH, HIGH)))
    return P.Packet(P.inventory_digest(words), 8, tuple(lanes))


def by_object(entries, key):
    out = [[], []]
    for o in (0, 1):
        sel = sorted((e for e in entries if e['object'] == o), key=lambda e: e['index'])
        assert [e['index'] for e in sel] == list(range(len(sel)))
        out[o] = [e[key] for e in sel]
    return out


def packet_record(P, packet, scene, wi, level, words):
    x, y = fields_of(scene)
    entries = packet_entries(packet, words)
    assert all(e['weight'] > 0 for e in entries)
    words_obj, weights_obj = by_object(entries, 'word'), by_object(entries, 'weight')
    text, _ = render_weighted(scene['aliases'], words_obj, weights_obj, width=WIDTH)
    raw = P.serialize(packet)
    weight_of = [{w: q for w, q in zip(words_obj[o], weights_obj[o])} for o in (0, 1)]
    strong = {}
    for v in sorted({f[x] for f in scene['worlds'][wi]['facts']}):
        ws = [weight_of[o].get(v, 0) for o in (0, 1)]
        strong[v] = (0 if ws[0] > ws[1] else 1) if ws[0] != ws[1] else None
    return dict(hex=raw.hex(), sha256=sha_bytes(raw), bytes=len(raw), level=level, x_field=x, y_field=y,
                entries=entries, words_by_object=words_obj, weights_by_object=weights_obj,
                strong_object_by_x_value=strong, text=text, text_sha256=sha_bytes(text.encode()))


# ------------------------------------------------------------------------------------------------ build

def build():
    for path, want in PINS.items():
        got = sha_file(path)
        assert got == want, f'pinned input changed: {path} {got} != {want}'
    assert sha_file(PACKET_SRC) == sha_file(PACKET_FROZEN), 'src/syncomm/packet.py differs from the frozen snapshot'
    import syncomm.packet as P   # the frozen wire format (identical file to the snapshot, asserted above)

    source = json.loads(SOURCE_BANK.read_text())
    words = json.loads(WORDS_FILE.read_text())
    r3 = json.loads(R3_MANIFEST.read_text())
    index = {w: i for i, w in enumerate(words)}
    assert len(words) == 221 and len(index) == 221 and NEUTRAL_WORD not in index
    eng_ids = list(r3['engineering_scene_ids'])
    by_id = {s['scene']: s for s in source}
    assert len(source) == 576 == len(by_id) and all(s['split'] == 'confirmation' for s in source)
    assert len(eng_ids) == 4 and all(e in by_id for e in eng_ids)
    engineering = [by_id[e] for e in eng_ids]
    evaluated = [s for s in source if s['scene'] not in set(eng_ids)]
    assert len(evaluated) == 572

    checks = defaultdict(int)
    # ---- design bits re-derived from the facts for every scene (no outcome is read anywhere)
    for s in source:
        d = derived(design_point(s))
        x, y = fields_of(s)
        w0 = s['worlds'][0]
        assert s['aliases'].index(w0['answers'][0]) == d['sX'] and s['aliases'].index(w0['answers'][1]) == d['sY']
        assert (index[w0['facts'][0][x]] < index[w0['facts'][1][x]]) == (d['fX'] == 0)
        assert (index[w0['facts'][0][y]] < index[w0['facts'][1][y]]) == (d['fY'] == 0)
        assert s['fields'][s['design_bits']['changed']] == x
        checks['design_bits_rederived'] += 1
    families = defaultdict(list)
    for s in evaluated:
        families[s['family']].append(s)
    assert len(families) == 36
    templates = {f"{s['domain']}/T{s['template']}" for s in evaluated}
    assert len(templates) == 12

    # ---- tiers
    avoid = {design_point(s) for s in engineering}
    tiers, tier_meta = select_tiers(avoid)
    tier_of_point = {p: t for t, pts in tiers.items() for p in pts}
    tier_of = {s['scene']: tier_of_point[design_point(s)] for s in evaluated}
    for s in engineering:
        assert tier_of_point[design_point(s)] == 4, 'an engineering design point landed in a planned tier'
    tier_counts = Counter(tier_of.values())
    per_family = {t: Counter(s['family'] for s in evaluated if tier_of[s['scene']] == t) for t in tiers}
    for t in (1, 2, 3):
        assert set(per_family[t].values()) == {len(tiers[t])} and len(per_family[t]) == 36
    balance = {}
    for name, ts in (('tier1', (1,)), ('tier2', (2,)), ('tiers1+2', (1, 2)), ('tier3', (3,)), ('tiers1-3', (1, 2, 3)), ('all', (1, 2, 3, 4))):
        sel = [s for s in evaluated if tier_of[s['scene']] in ts]
        dd = [derived(design_point(s)) for s in sel]
        balance[name] = dict(scenes=len(sel),
                             column_ones={c: sum(d[c] for d in dd) for c in COLUMNS},
                             changed_field={f'{d}/{f}': n for (d, f), n in sorted(Counter((s['domain'], s['changed_field']) for s in sel).items())},
                             per_domain=dict(Counter(s['domain'] for s in sel)),
                             per_template=dict(sorted(Counter(f"{s['domain']}/T{s['template']}" for s in sel).items())),
                             x_answer_first_alias_world0=sum(1 - d['sX'] for d in dd), y_answer_first_alias=sum(1 - d['sY'] for d in dd))

    # ---- renderer byte-identity with the frozen render() at all-255 weights
    ident = 0
    for s in source:
        for arg in (None,):
            assert render_weighted(s['aliases'], arg, None) == render(s['aliases'], arg)
            ident += 1
        for w in s['worlds']:
            pk = generic_literal(P, w, words, index)
            ent = packet_entries(pk, words)
            wb = by_object(ent, 'word')
            assert render_weighted(s['aliases'], wb, [[HIGH] * len(x) for x in wb]) == render(s['aliases'], wb)
            assert render_weighted(s['aliases'], wb, None) == render(s['aliases'], wb)
            ident += 2
    rng = random.Random(20260924)
    for _ in range(4000):
        width = rng.choice((2, 4))
        aliases = rng.sample(['XOPA', 'YUME', 'BANQ', 'CIRV', 'DOMP', 'FELK'], 2)
        wb = [rng.sample(words, rng.randint(0, width)) for _ in (0, 1)]
        assert render_weighted(aliases, wb, [[HIGH] * len(x) for x in wb], width=width) == render(aliases, wb, width=width)
        ident += 1
    checks['renderer_byte_identical_cases_at_255'] = ident
    # and with graded weights the ONLY difference from render() is the annotation digits of the non-255 slots
    graded_diff_ok = 0

    # ---- packets
    packets = {L: {} for L in LEVELS}
    strong_first = Counter()
    for s in engineering + evaluated:
        x, y = fields_of(s)
        for L in LEVELS:
            recs = []
            for wi, world in enumerate(s['worlds']):
                pk = graded_packet(P, s, wi, L, words, index)
                rec = packet_record(P, pk, s, wi, L, words)
                facts = world['facts']
                # lane contents and weights
                for o, lane in enumerate(pk.lanes):
                    assert list(lane.concept_ids) == sorted(lane.concept_ids)
                    want = sorted([HIGH, HIGH] + ([L] if L else []))
                    assert sorted(lane.q) == want
                    wmap = dict(zip(lane.concept_ids, lane.q))
                    assert wmap[index[facts[o][x]]] == HIGH and wmap[index[facts[o][y]]] == HIGH
                    if L:
                        assert wmap[index[facts[1 - o][x]]] == L
                    else:
                        assert index[facts[1 - o][x]] not in wmap
                    assert index[facts[1 - o][y]] not in wmap
                    checks['lanes_checked'] += 1
                # which object is strong for each X value; the X question's weight-derived gold
                for o in (0, 1):
                    v = facts[o][x]
                    assert rec['strong_object_by_x_value'][v] == (None if L == HIGH else o)
                qx = s['queried_values'][0]; qy = s['queried_values'][1]
                gold_x = s['aliases'].index(world['answers'][0]); gold_y = s['aliases'].index(world['answers'][1])
                assert rec['strong_object_by_x_value'][qx] == (None if L == HIGH else gold_x)
                assert sum(qy in rec['words_by_object'][o] for o in (0, 1)) == 1 and qy in rec['words_by_object'][gold_y]
                # L = 0 is byte-identical to the paper's complete-fact packet
                if L == 0:
                    assert P.serialize(pk) == P.serialize(generic_literal(P, world, words, index))
                    checks['L0_equals_complete_fact_packet'] += 1
                # order cannot reveal strength
                if 0 < L < HIGH:
                    firsts = []
                    for o in (0, 1):
                        ws = rec['words_by_object'][o]
                        firsts.append(ws.index(facts[o][x]) < ws.index(facts[1 - o][x]))
                    assert sorted(firsts) == [False, True], 'both lanes list the strong X value in the same order'
                    strong_first[(L, 'object0_strong_first')] += int(firsts[0])
                    checks['order_uninformative_worlds'] += 1
                    # graded text differs from the all-255 rendering only in the weak slots' digits
                    t255, _ = render(s['aliases'], rec['words_by_object'])
                    assert rec['text'].replace(f' [{L}/255]', ' [255/255]') == t255 and rec['text'].count(f' [{L}/255]') == 2
                    graded_diff_ok += 1
                assert render_weighted(s['aliases'], rec['words_by_object'], rec['weights_by_object'])[0] == rec['text']
                recs.append(rec)
                packets[L][f"{s['scene']}|{wi}"] = rec
            # across worlds
            if L == HIGH:
                assert recs[0]['hex'] == recs[1]['hex'] and recs[0]['text'] == recs[1]['text']
                checks['L255_worlds_byte_identical'] += 1
            elif L > 0:
                assert recs[0]['words_by_object'] == recs[1]['words_by_object']
                assert recs[0]['weights_by_object'] != recs[1]['weights_by_object']
                checks['graded_worlds_same_words_different_weights'] += 1
            else:
                assert recs[0]['hex'] != recs[1]['hex']
    checks['graded_text_differs_only_in_weak_digits'] = graded_diff_ok

    # ---- the reassignment control's permutation (R3 manifest, seed 202609201): no scene value maps onto a scene value
    perm = r3['permutations'][0]
    assert sorted(perm) == list(range(221)) and all(i != v for i, v in enumerate(perm))
    scene_values = {v for s in source for w in s['worlds'] for f in w['facts'] for v in f.values()}
    assert all(words[perm[index[v]]] not in scene_values for v in scene_values)
    checks['sham0_images_outside_scene_values'] = len(scene_values)

    # ---- assemble outputs
    bank = dict(study=STUDY, source_bank=str(SOURCE_BANK.relative_to(ROOT)), source_bank_sha256=PINS[SOURCE_BANK],
                builder=str(Path(__file__).resolve().relative_to(ROOT)), builder_sha256=sha_file(Path(__file__).resolve()),
                engineering_scene_ids=eng_ids, engineering_scene_ids_source=str(R3_MANIFEST.relative_to(ROOT)),
                engineering_scenes=engineering, scenes=evaluated, tier_of=tier_of, default_tiers=list(DEFAULT_TIERS),
                tiers={str(t): dict(design_points=[list(p) for p in pts], design_points_derived=[derived(p) for p in pts],
                                    scenes_per_family=len(pts) if t < 4 else None, scenes=tier_counts[t])
                       for t, pts in tiers.items()},
                design_point_order='(changed, flip_a, flip_b, qc) = scene.design_bits',
                selection_rule=select_tiers.__doc__.strip(),
                note=('Scenes are verbatim copies of the source bank records. The evaluation set of a run is the union of the '
                      'tiers named in its manifest; the default is tiers 1+2 (4 scenes per family, 144 scenes).'))
    files = {'bank.json': dumps(bank)}
    for L in LEVELS:
        body = dict(study=STUDY, level=L, high=HIGH, qbits=8, inventory_digest=P.inventory_digest(words),
                    words_file=str(WORDS_FILE.relative_to(ROOT)), words_sha256=PINS[WORDS_FILE],
                    source_bank=str(SOURCE_BANK.relative_to(ROOT)), source_bank_sha256=PINS[SOURCE_BANK],
                    builder_sha256=sha_file(Path(__file__).resolve()),
                    construction=('per object: true X value at 255, the other object\'s X value at the level (omitted at level 0), '
                                  'own Y value at 255; ascending inventory ids; serialized with syncomm.packet.serialize (qbits 8); '
                                  'text = render_weighted(aliases, words, weights) in the four-slot public layout'),
                    packets=packets[L])
        files[f'packets_L{L}.json'] = dumps(body)
    report = dict(study=STUDY, status='PASS_CONSTRUCTION',
                  inputs={str(p.relative_to(ROOT)): h for p, h in PINS.items()},
                  packet_module_sha256=sha_file(PACKET_SRC), builder_sha256=sha_file(Path(__file__).resolve()),
                  counts=dict(source_scenes=len(source), evaluated_scenes=len(evaluated), engineering_scenes=len(engineering),
                              families=len(families), templates=len(templates),
                              packets_per_level=len(packets[0]), levels=list(LEVELS),
                              tier_scenes={str(t): tier_counts[t] for t in sorted(tier_counts)},
                              default_tiers=list(DEFAULT_TIERS),
                              default_scenes=sum(tier_counts[t] for t in DEFAULT_TIERS)),
                  tier_selection=dict(meta={k: (list(map(list, v)) if isinstance(v, tuple) else v) for k, v in tier_meta.items()},
                                      avoided_engineering_design_points=sorted(map(list, avoid)),
                                      tiers={str(t): [list(p) for p in pts] for t, pts in tiers.items()},
                                      balance=balance),
                  checks=dict(checks),
                  order_check={f'L{L}': dict(worlds=2 * (len(engineering) + len(evaluated)),
                                             object0_lists_strong_x_first=strong_first[(L, 'object0_strong_first')],
                                             rule='in every world exactly one of the two lanes lists its strong X value first')
                               for L in LEVELS if 0 < L < HIGH},
                  wire_bytes={f'L{L}': dict(Counter(r['bytes'] for r in packets[L].values())) for L in LEVELS},
                  analytic_floor_L255=('packets and texts are byte-identical across the two worlds of every scene, so a deterministic '
                                       'receiver gives identical answers in both worlds: the changed-question answer can be right in at '
                                       'most one world, hence four-answer success = 0 and R = 0 exactly, changed-question accuracy <= 1/2'),
                  outputs={name: sha_bytes(data) for name, data in files.items()})
    files['construction_report.json'] = dumps(report)
    return files, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='compare an in-memory rebuild with the files on disk; write nothing')
    a = ap.parse_args()
    files, report = build()
    status = {}
    for name, data in files.items():
        path = OUT / name
        if path.exists():
            if path.read_bytes() != data:
                print(f'REFUSE: {path} exists with different content (sha {sha_file(path)} vs rebuilt {sha_bytes(data)})')
                sys.exit(2)
            status[name] = 'identical'
        elif a.check:
            status[name] = 'MISSING'
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + '.tmp')
            tmp.write_bytes(data)
            tmp.replace(path)
            status[name] = 'written'
    for name, st in status.items():
        print(f'{st:9s} {name} sha256={sha_bytes(files[name])}')
    c = report['counts']
    print('scenes', c['evaluated_scenes'], 'engineering', c['engineering_scenes'], 'tiers', c['tier_scenes'],
          'default', c['default_tiers'], '=', c['default_scenes'], 'scenes')
    print('checks', json.dumps(report['checks'], sort_keys=True))
    if a.check and 'MISSING' in status.values():
        sys.exit(1)


if __name__ == '__main__':
    main()
