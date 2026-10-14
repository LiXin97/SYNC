"""Build locked experiment-A manifests (one per receiver x sender) for src/syncomm/recovery/evaluate_confirm.py.

Everything scientific is copied from gates/latest0924/expA_protocol.lock.json and from the R3 confirmation manifest
(bank, engineering scene ids, word permutations, gamma, numerics), so the new cells differ from the paper's R3 cells
only in the receiver, its locally fitted directions and the sender's messages. A manifest is written only when all of
its inputs exist (sender messages that passed the gate, stacked receiver directions); existing manifests are never
overwritten, so a manifest is fixed before its first outcome row.

Usage: python3 scripts/latest0924/build_expA_manifests.py [--sender KEY ...] [--receiver KEY ...]
"""
import argparse
import hashlib
import json
from pathlib import Path

R = Path('.')
LOCK = R / 'gates/latest0924/expA_protocol.lock.json'
R3 = R / 'artifacts/receiver_recovery/r3/qwen3-32b/manifest.json'
BASE = R / 'artifacts/latest0924'
SENDER_ARMS = ['none', 'identity_write', 'original', 'selected', 'packet_text', 'sham0', 'sham1', 'sham2', 'other_world']
LITERAL_ARMS = ['selected', 'packet_text', 'full_text', 'native']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sender', nargs='*')
    ap.add_argument('--receiver', nargs='*')
    a = ap.parse_args()
    lock = json.loads(LOCK.read_text())
    r3 = json.loads(R3.read_text())
    senders = sorted(lock['senders'], key=lambda s: lock['senders'][s]['priority'])
    first = senders[0]
    for sender in (a.sender or senders):
        gate = BASE / 'senders' / f'{sender}_gate.json'
        packets = BASE / 'senders' / f'{sender}_r3.json'
        if not (gate.exists() and packets.exists()):
            print('SKIP sender', sender, '(gate or confirmation messages missing)'); continue
        report = json.loads(gate.read_text())
        status = report.get('status') or report.get('verdict', {}).get('status')  # sender_gate.py nests it under verdict
        if status != 'PASS':
            print('SKIP sender', sender, 'gate', status); continue
        for receiver, spec in lock['receivers'].items():
            if a.receiver and receiver not in a.receiver:
                continue
            directions = BASE / 'fits' / f'directions_{receiver}.pt'
            if not directions.exists():
                print('SKIP', sender, receiver, '(directions missing)'); continue
            root = BASE / 'expA' / sender / receiver
            path = root / 'manifest.json'
            if path.exists():
                print('EXISTS', path); continue
            sources = {sender: str(packets)}
            arms = {sender: list(SENDER_ARMS)}
            if sender == first:
                sources['literal'] = None
                arms['literal'] = list(LITERAL_ARMS)
            same_model = f'{sender} -> {receiver}' in lock['same_model_pairs']
            files = [r3['bank'], r3['words_file'], str(directions), str(packets), str(LOCK),
                     str(R / 'src/syncomm/recovery/evaluate_confirm.py'), str(R / 'src/syncomm/recovery/evaluate_depth.py')]
            m = dict(study=f'LATEST0924_EXPA_{sender}_{receiver}'.upper(), lock=str(LOCK.relative_to(R)),
                     purpose=f'Experiment A cell: sender {sender} (priority {lock["senders"][sender]["priority"]}) to receiver {receiver} '
                             f'({spec["family"]}, {spec["size"]}) on the R3 bank with the fixed interface (selected layer '
                             f'{spec["selected_layer"]} = relative depth 0.2, gamma 1, raw scale; original band {spec["original_band"]}).'
                             + (' SAME-MODEL PAIR: excluded from cross-model claims.' if same_model else ''),
                     same_model_pair=same_model, receiver=receiver, sender=sender, root=str(root),
                     models={receiver: spec['model_dir']}, base_models=[], n_blocks=spec['n_layers'],
                     directions=str(directions), words_file=r3['words_file'], bank=r3['bank'], bank_size=r3['bank_size'],
                     bank_split=r3['bank_split'], engineering_scene_ids=r3['engineering_scene_ids'], row_split='latest0924_expA',
                     gamma=r3['gamma'], selected_depth=0.2, selected_layers=[spec['selected_layer']],
                     original_layers=spec['original_band'], permutation_seeds=r3['permutation_seeds'],
                     permutations=r3['permutations'], numerics=r3['numerics'], scale_rule='raw',
                     sources=sources, arms=arms, pins={f: sha(f) for f in files})
            root.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(m, indent=1) + '\n')
            print('WROTE', path)


if __name__ == '__main__':
    main()
