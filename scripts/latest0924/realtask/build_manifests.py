"""Experiment D: write one frozen-evaluator manifest per receiver (src/syncomm/recovery/evaluate_confirm.py, unchanged).

Inputs must exist and are pinned: the Countries bank + inventory, the receiver's stacked directions for the inventory,
the gate-passing sender messages. Existing manifests are never overwritten.
Usage: python3 scripts/latest0924/realtask/build_manifests.py [--receiver KEY ...]
"""
import argparse, hashlib, json, random
from pathlib import Path

R = Path(__file__).resolve().parents[3]
A = R / 'artifacts/latest0924/realtask'
LOCK = R / 'gates/latest0924/expD_realtask.lock.json'
SENDER = 'gemma-4-26b-a4b-it'
ARMS = ['none', 'identity_write', 'original', 'selected', 'packet_text', 'full_text', 'sham0', 'sham1', 'sham2', 'other_world']
PERM_SEEDS = [202609251, 202609252, 202609253]
RECEIVERS = {
    'qwen3.5-27b': dict(model='/path/to/models/Qwen3.5-27B', n=64, selected=[13], original=[26, 32, 38]),
    'qwen3.6-27b': dict(model='/path/to/models/Qwen3.6-27B', n=64, selected=[13], original=[26, 32, 38]),
    'mistral-small-3.2-24b': dict(model='/path/to/models/Mistral-Small-3.2-24B-Instruct-2506', n=40, selected=[8], original=[16, 20, 24]),
    'qwen3-32b': dict(model='/path/to/models/Qwen3-32B', n=64, selected=[13], original=[26, 32, 38]),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def derangement(n, seed):
    rng = random.Random(seed)
    while True:
        p = list(range(n)); rng.shuffle(p)
        if all(i != v for i, v in enumerate(p)):
            return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--receiver', nargs='*'); a = ap.parse_args()
    words = json.loads((A / 'inventory_275.json').read_text()); bank = json.loads((A / 'bank.json').read_text())
    eng = json.loads((A / 'engineering_scene_ids.json').read_text())
    gate = A / 'senders' / f'{SENDER}_gate.json'; packets = A / 'senders' / f'{SENDER}_countries.json'
    assert json.loads(gate.read_text())['status'] == 'PASS', 'sender gate did not pass'
    perms = [derangement(len(words), s) for s in PERM_SEEDS]
    for rec, spec in RECEIVERS.items():
        if a.receiver and rec not in a.receiver:
            continue
        d = A / 'fits' / f'directions_{rec}.pt'
        if not d.exists():
            print('SKIP', rec, '(directions missing)'); continue
        root = A / 'eval' / rec; path = root / 'manifest.json'
        if path.exists():
            print('EXISTS', path); continue
        files = [A / 'bank.json', A / 'inventory_275.json', A / 'engineering_scene_ids.json', d, packets, gate, LOCK,
                 R / 'src/syncomm/recovery/evaluate_confirm.py', R / 'src/syncomm/recovery/evaluate_depth.py']
        m = dict(study=f'LATEST0924_EXPD_COUNTRIES_{rec}'.upper(), lock=str(LOCK.relative_to(R)),
                 purpose=f'Experiment D (Countries, two-person counterfactual format): sender {SENDER} to receiver {rec}, final interface '
                         f'(selected layer {spec["selected"]} = depth 0.2, gamma 1, raw fitted 275-symbol directions; original band {spec["original"]}).',
                 receiver=rec, sender=SENDER, root=str(root), models={rec: spec['model']}, base_models=[], n_blocks=spec['n'],
                 directions=str(d), words_file=str(A / 'inventory_275.json'), bank=str(A / 'bank.json'), bank_size=len(bank),
                 bank_split='countries_pairs', engineering_scene_ids=eng, row_split='latest0924_expD_countries', gamma=1,
                 selected_depth=0.2, selected_layers=spec['selected'], original_layers=spec['original'],
                 permutation_seeds=PERM_SEEDS, permutations=perms,
                 numerics=dict(budgets=[64, 256], dtype='FP32 eager, TF32 off, batch 1', parser='strict first-line alias'),
                 scale_rule='raw', sources={SENDER: str(packets)}, arms={SENDER: list(ARMS)}, pins={str(f): sha(f) for f in files})
        root.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(m, indent=1) + '\n'); print('WROTE', path)


if __name__ == '__main__':
    main()
