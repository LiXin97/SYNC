"""Experiment D: descriptive message quality on the evaluation bank (not a gate). Run where transformers is importable."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / 'src'))
from syncomm.recovery.evaluate_depth import base, packet_entries  # noqa: E402
A = ROOT / 'artifacts/latest0924/realtask'
P = json.loads((A / 'senders/gemma-4-26b-a4b-it_countries.json').read_text()); bank = json.loads((A / 'bank.json').read_text()); words = P['words']
n = rec = wrong = tot = 0
for s in bank:
    for wi, w in enumerate(s['worlds']):
        pk = base.deserialize(bytes.fromhex(P['packets'][f"{s['scene']}|{wi}"]['hex'])); ent = packet_entries(pk, words)
        for o in (0, 1):
            ws = [e['word'] for e in ent if e['object'] == o]; n += 1; tot += len(ws)
            rec += w['facts'][o]['landmark'] in ws; wrong += w['facts'][1 - o]['landmark'] in ws
out = dict(lanes=n, landmark_recall=rec / n, wrong_object_lane_rate=wrong / n, symbols_per_lane=tot / n)
(A / 'senders/eval_bank_quality.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out))
