"""Experiment D: build the two-person Countries bank and the task inventory (deterministic; no model outputs used).

Source task: data/bench/ac-countries_dev.jsonl (Countries, adapted from Ramesh & Li 2025; 400 dev items, each a random
person-landmark pairing). The test split is NOT used (it was touched once by the earlier utility study).
Inventory: every landmark and every country in the builder's fixed word lists (src/syncomm/bench/build_ac.py LANDMARKS),
lowercased as the frozen parser requires -> 275 symbols. Derived from the task definition only.

Scene format = the frozen two-object counterfactual format (R3 bank schema), so the frozen sender prompt, receiver
prompt, parser, scorer and evaluator (src/syncomm/recovery/evaluate_confirm.py) apply unchanged:
  two objects with synthetic uppercase aliases (reused from the R3 bank's alias pairs, which every receiver tokenizes),
  world 0: A is at landmark L1, B is at landmark L2 (different countries C1 != C2);
  world 1: the landmarks are swapped;
  questions: "Which object is in C1?", "Which object is in C2?"  -> answers w0 [A, B], w1 [B, A].
The receiver must map each landmark to its country with its own knowledge. Without a message both worlds give the
same input, so four-answer success is zero by construction.
Split: gate scenes (sender message-quality check), 4 engineering scenes, evaluation scenes.
Usage: python3 scripts/latest0924/realtask/build_countries_bank.py [--check]
"""
import argparse, hashlib, importlib.util, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'artifacts/latest0924/realtask'
SRC = ROOT / 'data/bench/ac-countries_dev.jsonl'
R3 = ROOT / 'artifacts/receiver_recovery/r3_bank/bank.json'
SEED = 20260925
N_GATE, N_ENG = 24, 4
THE = {'United States', 'United Kingdom', 'United Arab Emirates', 'Netherlands', 'Philippines', 'Czech Republic'}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def load_builder():
    spec = importlib.util.spec_from_file_location('build_ac', ROOT / 'src/syncomm/bench/build_ac.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def country_phrase(c):
    return ('the ' if c in THE else '') + c


def build():
    ac = load_builder()
    landmarks = [(n, c, t) for n, c, t in ac.LANDMARKS]
    lm_country = {n: c for n, c, _ in landmarks}
    lm_the = {n: t for n, _, t in landmarks}
    countries = sorted({c for _, c, _ in landmarks})
    inventory = sorted({n.lower() for n, _, _ in landmarks}) + sorted({c.lower() for c in countries})
    assert len(inventory) == len(set(inventory)) == 275, len(inventory)
    continent = dict(ac.CONTINENTS)
    assert set(countries) <= set(continent), set(countries) - set(continent)
    items = [json.loads(l) for l in open(SRC)]
    assert len(items) == 400 and all(i['split'] == 'dev' for i in items)
    rng = random.Random(SEED)
    order = list(range(len(items))); rng.shuffle(order)
    used, pairs = set(), []
    for i in order:
        if i in used:
            continue
        for j in order:
            if j == i or j in used:
                continue
            a, b = items[i]['ac_meta'], items[j]['ac_meta']
            if a['country'] != b['country'] and a['landmark'] != b['landmark']:
                pairs.append((i, j)); used |= {i, j}; break
    r3 = json.loads(R3.read_text())
    alias_pairs = [s['aliases'] for s in r3]
    inv_upper = {w.upper() for w in inventory}
    alias_pairs = [p for p in alias_pairs if not (set(p) & inv_upper)]
    scenes = []
    for k, (i, j) in enumerate(pairs):
        a, b = items[i]['ac_meta'], items[j]['ac_meta']
        L1, L2 = a['landmark'], b['landmark']; C1, C2 = lm_country[L1], lm_country[L2]
        assert C1 == a['country'] and C2 == b['country'] and C1 != C2
        A, B = alias_pairs[k]
        ev = lambda who, L: f"{who} is at {'the ' if lm_the[L] else ''}{L}."
        worlds = [dict(world=0, answers=[A, B], evidence=[ev(A, L1), ev(B, L2)],
                       facts=[{'landmark': L1.lower()}, {'landmark': L2.lower()}], countries=[C1, C2]),
                  dict(world=1, answers=[B, A], evidence=[ev(A, L2), ev(B, L1)],
                       facts=[{'landmark': L2.lower()}, {'landmark': L1.lower()}], countries=[C2, C1])]
        fam = '|'.join(sorted([C1, C2]))
        tmpl = '|'.join(sorted([continent.get(C1, 'other'), continent.get(C2, 'other')]))
        body = json.dumps(dict(aliases=[A, B], l=[L1, L2]), sort_keys=True)
        scenes.append(dict(scene=sha(body.encode())[:20], split='countries_pairs', domain='countries', family=fam,
                           template=tmpl, aliases=[A, B], fields=['location'], changed_field='location',
                           questions=[f'Which object is in {country_phrase(C1)}?', f'Which object is in {country_phrase(C2)}?'],
                           queried_values=[C1, C2], source_items=[items[i]['item_id'], items[j]['item_id']],
                           landmarks=[L1, L2], worlds=worlds))
    ids = [s['scene'] for s in scenes]; assert len(ids) == len(set(ids))
    gate, eng, evals = scenes[:N_GATE], scenes[N_GATE:N_GATE + N_ENG], scenes[N_GATE + N_ENG:]
    return inventory, gate, eng + evals, [s['scene'] for s in eng]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
    inventory, gate, bank, eng_ids = build()
    files = {'inventory_275.json': json.dumps(inventory, indent=1) + '\n',
             'gate_bank.json': json.dumps(gate, indent=1) + '\n',
             'bank.json': json.dumps(bank, indent=1) + '\n',
             'engineering_scene_ids.json': json.dumps(eng_ids) + '\n'}
    OUT.mkdir(parents=True, exist_ok=True)
    for n, body in files.items():
        p = OUT / n
        if a.check or p.exists():
            assert p.read_text() == body, f'{p} differs from a rebuild (refusing to overwrite)'
        else:
            p.write_text(body)
    print('inventory', len(inventory), 'gate scenes', len(gate), 'bank scenes', len(bank), '(engineering', len(eng_ids), ')')
    print({n: sha(b.encode())[:12] for n, b in files.items()})


if __name__ == '__main__':
    main()
