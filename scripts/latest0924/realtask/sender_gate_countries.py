"""Experiment D sender gate on the 24 gate scenes (thresholds from gates/latest0924/expD_realtask.lock.json).

Metrics (48 worlds, 96 object lanes): fully parsed worlds; landmark recall = the world's landmark symbol appears in the
lane of the object that is at that landmark; wrong-object lanes = a lane containing the OTHER object's landmark;
country symbols selected (reported, not gated). Every packet is decoded with the frozen serializer and checked against
the logged selection. Usage: python3 .../sender_gate_countries.py --packets P --bank B --lock L --out O
"""
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from syncomm.recovery.evaluate_depth import base, packet_entries  # noqa: E402  frozen helpers


def main():
    ap = argparse.ArgumentParser()
    for k in ('packets', 'bank', 'lock', 'out'):
        ap.add_argument('--' + k, type=Path, required=(k != 'lock'))
    a = ap.parse_args()
    P = json.loads(a.packets.read_text()); bank = json.loads(a.bank.read_text()); words = P['words']
    countries = {w for w in words if w in {s.lower() for sc in bank for w0 in sc['worlds'] for s in w0['countries']}}
    n = parsed = lanes = recall = wrong = with_country = 0
    for sc in bank:
        for wi, w in enumerate(sc['worlds']):
            key = f"{sc['scene']}|{wi}"; n += 1
            rec = P['packets'][key]; log = P['log'][key]
            pk = base.deserialize(bytes.fromhex(rec['hex'])); assert base.sha(base.serialize(pk)) == rec['sha256']
            ent = packet_entries(pk, words)
            by_obj = [[e['word'] for e in ent if e['object'] == o] for o in (0, 1)]
            sel = [sorted(log['selection'].get(al, []), key=words.index) for al in sc['aliases']]
            assert [sorted(x, key=words.index) for x in by_obj] == sel, key
            parsed += log['parse_status'] == 'ok'
            for o in (0, 1):
                lanes += 1
                recall += w['facts'][o]['landmark'] in by_obj[o]
                wrong += w['facts'][1 - o]['landmark'] in by_obj[o]
                with_country += any(x in countries for x in by_obj[o])
    metrics = dict(worlds=n, parse_fully_rate=parsed / n, landmark_recall=recall / lanes, wrong_object_lane_rate=wrong / lanes,
                   lanes_with_country_symbol=with_country / lanes)
    verdict = dict(status='NO_LOCK')
    if a.lock:
        th = json.loads(a.lock.read_text())['sender_gate']['thresholds']
        ok = (metrics['parse_fully_rate'] >= th['parse_fully_rate_min'] and metrics['landmark_recall'] >= th['landmark_recall_min']
              and metrics['wrong_object_lane_rate'] <= th['wrong_object_lane_rate_max'])
        verdict = dict(status='PASS' if ok else 'FAIL', thresholds=th)
    out = dict(metrics=metrics, packets=str(a.packets), packets_sha256=hashlib.sha256(a.packets.read_bytes()).hexdigest(),
               bank=str(a.bank), **verdict)
    a.out.write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out))


if __name__ == '__main__':
    main()
