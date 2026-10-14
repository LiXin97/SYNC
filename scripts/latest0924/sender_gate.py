"""Sender gate: how many required facts a sender's packets carry, checked against the bank; PASS/FAIL from a lock file.

Reads one packets file (candidate A's structure: `packets[f'{scene}|{world}']` with `hex`/`sha256`, plus `log`) and the
bank it was generated from, decodes every packet from its bytes, and counts per object lane:

  required facts      the object's two gold attribute values (`world['facts'][object].values()`)
  required_fact_recall  gold values present in that object's lane / all gold values        (Table 17 "Fact recall")
  complete_object_rate  lanes containing every gold value / lanes                           (Table 17 "Complete objects")
  complete_packet_rate  packets in which every object lane is complete / packets
  wrong_object          a lane containing another object's attribute value that is not its own (a subset of
                        unsupported_attribute; scripts/quality_audit_packets_20260922.py's definition, generalised
                        from "the other object" to "any other object" so it also covers four-object banks)
  unsupported_attribute a lane containing any bank attribute value that is not its own
  extra_non_attribute   a lane containing inventory words that are not attribute values (nouns etc.; allowed)
  words per object      lane size after the frozen parser (0..4)
  parse status          the frozen parser's status from the log: fully = 'ok' (every object got >= 1 inventory word),
                        partially = 'partial', failed = 'empty' + 'malformed'
Recall and complete-object counts are exactly scripts/build_current_appendix_views.py::support_quality, the generator of
Table 17 ("Facts retained by the automatic sender"): extra selected words are allowed; engineering scenes are INCLUDED
in population `all` (as in Table 17). Population `evaluated` additionally drops the scene ids listed under
`engineering_scene_ids` of --exclude-scenes-from, when given. Precision and exact-set rate follow
scripts/adjudicate_b6_candidate_a_20260922.py::message_quality and are reported as secondary numbers.

Integrity (verdict INVALID if any fails and the lock requires integrity, which is the default): every bank world has a
packet and no other packet exists; packet bytes match the recorded sha256/bytes and deserialize; every weight is 255;
lane i belongs to object i and holds ascending unique ids; the log's `selection` equals the decoded lane; re-running the
frozen parse_selection on the parsed text (`answer_text` if present, else candidate A's `raw_output`) reproduces the
recorded status and selection; the header's prompt hash is the frozen candidate A/A4 prompt; the inventory equals the
221-word reference; the header's bank hash equals the bank file.

Intervals: 95% percentile bootstrap over scenes (both worlds of a scene resampled together), 2000 draws, seed 0.

Thresholds are never in this file. --lock is a JSON file:
  {"name": "...", "population": "all" | "evaluated", "require_integrity": true, "bank_sha256": "<optional>",
   "thresholds": {"<metric>": {"op": ">=" | ">" | "<=" | "<" | "==", "value": <number>}, ...}}
where <metric> is any numeric key of the population's metrics, including `<name>_ci_lo` / `<name>_ci_hi`.
Without --lock the numbers are reported with verdict NO_LOCK.

Usage (CPU):
  python3 scripts/latest0924/sender_gate.py --packets <packets.json> [--bank <bank.json>] [--lock <lock.json>] \\
      [--exclude-scenes-from <manifest with engineering_scene_ids>] --out <report.json> [--md <report.md>]
The default bank is the 96-scene exposed development bank of the receiver-recovery study.
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
sys.path.insert(0, str(ROOT / 'src'))
from syncomm.packet import deserialize, serialize  # noqa: E402
from syncomm.routeb.sender_candidate_a import parse_selection  # noqa: E402  the frozen parser

DEV_BANK = ROOT / 'artifacts/mechanism0919/discovery_v1/exposed_development.json'
REFERENCE_WORDS = ROOT / 'artifacts/receiver_recovery/words_221.json'
FROZEN_PROMPT_SHA256 = {'328c8b5df9d196724cdabd6cc7a291c0cfc7ca13ab524bd3cb70239de7183e0b',   # two objects (candidate A)
                        'bc796522ac1d8739be671c27f7f847810dc5163855999553fa380dd583fe81bf'}   # four objects (candidate A4)
N_BOOT, SEED = 2000, 0
OPS = {'>=': lambda x, v: x >= v, '>': lambda x, v: x > v, '<=': lambda x, v: x <= v, '<': lambda x, v: x < v,
       '==': lambda x, v: x == v}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def world_record(scene, world, rec, log, words, index, attr_values, integ, key):
    """Per-world counts, and integrity notes appended to `integ`."""
    aliases = scene['aliases']
    data = bytes.fromhex(rec['hex'])
    if hashlib.sha256(data).hexdigest() != rec.get('sha256'):
        integ['sha256_mismatch'].append(key)
    if rec.get('bytes') is not None and rec['bytes'] != len(data):
        integ['bytes_mismatch'].append(key)
    try:
        pk = deserialize(data)
        assert serialize(pk) == data
    except Exception as e:
        integ['decode_error'].append(f'{key}: {e}')
        pk = None
    lanes = {}
    if pk is not None:
        for ln in pk.lanes:
            if list(ln.concept_ids) != sorted(set(ln.concept_ids)):
                integ['lane_ids_not_ascending_unique'].append(key)
            if any(q != 255 for q in ln.q):
                integ['weight_not_255'].append(key)
            lanes[ln.slot_id] = [words[i] for i in ln.concept_ids]
        if sorted(lanes) != list(range(len(aliases))):
            integ['lane_slots_not_objects'].append(key)
    status = log.get('parse_status') if log else None
    if log is None:
        integ['log_missing'].append(key)
    else:
        sel_log = log.get('selection') or {}
        if any(sorted(index[w] for w in sel_log.get(a, [])) != sorted(index[w] for w in lanes.get(i, []))
               for i, a in enumerate(aliases)):
            integ['selection_vs_packet_mismatch'].append(key)
        text = log['answer_text'] if 'answer_text' in log else log.get('raw_output', '')
        sel2, status2, _ = parse_selection(text, aliases, index)
        if status2 != status or any(sel2[a] != sel_log.get(a) for a in aliases):
            integ['reparse_mismatch'].append(key)
    objs = []
    for o, alias in enumerate(aliases):
        sel = lanes.get(o, [])
        gold = set(world['facts'][o].values())
        other = set().union(*[set(world['facts'][j].values()) for j in range(len(aliases)) if j != o]) if len(aliases) > 1 else set()
        s = set(sel)
        uns = {x for x in s if x in attr_values and x not in gold}
        objs.append(dict(n_words=len(sel), gold=len(gold), hit=len(gold & s), complete=gold <= s, empty=not sel,
                         wrong_words=len({x for x in uns if x in other}), unsupported=bool(uns),
                         extra=bool({x for x in s if x not in attr_values}), fp=len(s - gold), exact=(s == gold),
                         field_hits={f: int(v in s) for f, v in world['facts'][o].items()}))
    return dict(scene=scene['scene'], status=status, bytes=len(data), objs=objs, log=log or {})


def metrics(rows):
    n = len(rows)
    objs = [o for r in rows for o in r['objs']]
    L = max(1, len(objs))
    st = Counter(r['status'] for r in rows)
    hit, gold = sum(o['hit'] for o in objs), sum(o['gold'] for o in objs)
    tp, fp = hit, sum(o['fp'] for o in objs)
    fields = {}
    for r in rows:
        for o in r['objs']:
            for f, h in o['field_hits'].items():
                c = fields.setdefault(f, [0, 0]); c[0] += h; c[1] += 1
    logs = [r['log'] for r in rows]

    def logsum(fn):
        vals = [fn(g) for g in logs]
        vals = [v for v in vals if v is not None]
        return sum(vals) if vals else None

    gen = [g.get('generated_tokens') for g in logs if g.get('generated_tokens') is not None]
    secs = [g.get('decode_seconds') for g in logs if g.get('decode_seconds') is not None]
    m = dict(
        n_packets=n, n_objects=len(objs),
        parse_ok=st.get('ok', 0), parse_partial=st.get('partial', 0), parse_empty=st.get('empty', 0),
        parse_malformed=st.get('malformed', 0), parse_unknown=sum(v for k, v in st.items() if k not in ('ok', 'partial', 'empty', 'malformed')),
        parse_fully_rate=st.get('ok', 0) / max(1, n), parse_partial_rate=st.get('partial', 0) / max(1, n),
        parse_failed_rate=(st.get('empty', 0) + st.get('malformed', 0)) / max(1, n),
        required_fact_hits=hit, required_fact_total=gold, required_fact_recall=hit / max(1, gold),
        complete_objects=sum(o['complete'] for o in objs), complete_object_rate=sum(o['complete'] for o in objs) / L,
        complete_packets=sum(all(o['complete'] for o in r['objs']) for r in rows),
        complete_packet_rate=sum(all(o['complete'] for o in r['objs']) for r in rows) / max(1, n),
        wrong_object_lanes=sum(o['wrong_words'] > 0 for o in objs), wrong_object_words=sum(o['wrong_words'] for o in objs),
        wrong_object_lane_rate=sum(o['wrong_words'] > 0 for o in objs) / L,
        unsupported_attribute_lanes=sum(o['unsupported'] for o in objs), unsupported_attribute_lane_rate=sum(o['unsupported'] for o in objs) / L,
        extra_non_attribute_lanes=sum(o['extra'] for o in objs), extra_non_attribute_lane_rate=sum(o['extra'] for o in objs) / L,
        empty_lanes=sum(o['empty'] for o in objs), empty_lane_rate=sum(o['empty'] for o in objs) / L,
        mean_words_per_object=sum(o['n_words'] for o in objs) / L,
        words_per_object_hist={str(k): v for k, v in sorted(Counter(o['n_words'] for o in objs).items())},
        precision=tp / max(1, tp + fp), exact_set_rate=sum(o['exact'] for o in objs) / L,
        mean_bytes=sum(r['bytes'] for r in rows) / max(1, n),
        recall_by_field={f: dict(hits=h, total=t, recall=h / max(1, t)) for f, (h, t) in sorted(fields.items())},
        mean_generated_tokens=(sum(gen) / len(gen)) if gen else None, max_generated_tokens=max(gen) if gen else None,
        mean_decode_seconds=(sum(secs) / len(secs)) if secs else None,
        hit_max_new_tokens=logsum(lambda g: int(g['stop_reason'] == 'max_new_tokens') if 'stop_reason' in g else None),
        final_channel_missing=logsum(lambda g: int(g['answer_info']['final_channel_found'] is False)
                                     if 'final_channel_found' in (g.get('answer_info') or {}) else None),
        reasoning_marker_in_output=logsum(lambda g: int(bool(g['answer_info']['reasoning_marker_in_output']))
                                          if 'reasoning_marker_in_output' in (g.get('answer_info') or {}) else None),
        greedy_argmax_mismatches=logsum(lambda g: g.get('greedy_argmax_mismatches')))
    return m


def bootstrap(rows, names):
    """Scene-clustered percentile intervals for the rate metrics in `names`."""
    by_scene = {}
    for r in rows:
        by_scene.setdefault(r['scene'], []).append(r)
    scenes = sorted(by_scene)
    rng = random.Random(SEED)
    draws = {k: [] for k in names}
    for _ in range(N_BOOT):
        sample = [r for s in (rng.choice(scenes) for _ in scenes) for r in by_scene[s]]
        objs = [o for r in sample for o in r['objs']]
        vals = dict(required_fact_recall=sum(o['hit'] for o in objs) / max(1, sum(o['gold'] for o in objs)),
                    complete_object_rate=sum(o['complete'] for o in objs) / max(1, len(objs)),
                    complete_packet_rate=sum(all(o['complete'] for o in r['objs']) for r in sample) / max(1, len(sample)),
                    parse_fully_rate=sum(r['status'] == 'ok' for r in sample) / max(1, len(sample)),
                    wrong_object_lane_rate=sum(o['wrong_words'] > 0 for o in objs) / max(1, len(objs)))
        for k in names:
            draws[k].append(vals[k])
    out = {}
    for k, v in draws.items():
        v.sort()
        out[f'{k}_ci_lo'] = v[int(0.025 * N_BOOT)]
        out[f'{k}_ci_hi'] = v[int(0.975 * N_BOOT) - 1]
    return out


def evaluate(packets_path, bank_path, exclude_from=None, reference_words=REFERENCE_WORDS):
    P = json.loads(Path(packets_path).read_text())
    bank_bytes = Path(bank_path).read_bytes()
    bank = json.loads(bank_bytes)
    words = P['words']
    index = {w: i for i, w in enumerate(words)}
    attr_values = {v for s in bank for w in s['worlds'] for f in w['facts'] for v in f.values()}
    ref_words = json.loads(Path(reference_words).read_text())
    ref_words = ref_words['words'] if isinstance(ref_words, dict) else ref_words
    integ = {k: [] for k in ('missing_packets', 'extra_packets', 'sha256_mismatch', 'bytes_mismatch', 'decode_error',
                             'lane_ids_not_ascending_unique', 'weight_not_255', 'lane_slots_not_objects', 'log_missing',
                             'selection_vs_packet_mismatch', 'reparse_mismatch')}
    expected = [f"{s['scene']}|{w['world']}" for s in bank for w in s['worlds']]
    integ['missing_packets'] = [k for k in expected if k not in P['packets']]
    integ['extra_packets'] = sorted(set(P['packets']) - set(expected))
    header = dict(bank_sha256_matches=(P.get('bank_sha256') == hashlib.sha256(bank_bytes).hexdigest()),
                  prompt_sha256_frozen=(P.get('prompt_sha256') in FROZEN_PROMPT_SHA256),
                  inventory_matches_reference=(words == ref_words), inventory_size=len(words))
    excluded = set()
    if exclude_from:
        excluded = set(json.loads(Path(exclude_from).read_text())['engineering_scene_ids'])
    rows = []
    for s in bank:
        for w in s['worlds']:
            k = f"{s['scene']}|{w['world']}"
            if k not in P['packets']:
                continue
            r = world_record(s, w, P['packets'][k], P.get('log', {}).get(k), words, index, attr_values, integ, k)
            r['engineering'] = s['scene'] in excluded
            rows.append(r)
    pops = {'all': rows}
    if exclude_from:
        pops['evaluated'] = [r for r in rows if not r['engineering']]
    ci_names = ('required_fact_recall', 'complete_object_rate', 'complete_packet_rate', 'parse_fully_rate', 'wrong_object_lane_rate')
    populations = {}
    for name, rs in pops.items():
        m = metrics(rs)
        m.update(bootstrap(rs, ci_names))
        m['n_scenes'] = len({r['scene'] for r in rs})
        populations[name] = m
    summary_vs_log = None
    if isinstance(P.get('summary'), dict):
        s = P['summary']; a = populations['all']
        summary_vs_log = all(s.get(k) == a[f'parse_{k}'] for k in ('ok', 'partial', 'empty', 'malformed')) if not integ['missing_packets'] else None
    counts = {k: len(v) for k, v in integ.items()}
    ok = all(v == 0 for v in counts.values()) and all(header.values()) and summary_vs_log is not False
    integrity = dict(ok=ok, header=header, counts=counts, file_summary_matches_recount=summary_vs_log,
                     examples={k: v[:5] for k, v in integ.items() if v})
    meta = dict(packets=str(packets_path), packets_sha256=sha_file(packets_path), bank=str(bank_path),
                bank_sha256=hashlib.sha256(bank_bytes).hexdigest(), sender=P.get('sender'), sender_checkpoint=P.get('sender_checkpoint'),
                split=P.get('split'), prompt_sha256=P.get('prompt_sha256'), chat_template_kwargs=P.get('chat_template_kwargs'),
                exclude_scenes_from=str(exclude_from) if exclude_from else None, excluded_scene_ids=sorted(excluded),
                n_bank_scenes=len(bank), n_bank_worlds=len(expected))
    return meta, integrity, populations


def verdict(lock, integrity, populations):
    if lock is None:
        return dict(status='NO_LOCK', checks=[])
    pop = lock.get('population', 'all')
    if pop not in populations:
        raise SystemExit(f'lock asks for population {pop!r}, have {sorted(populations)} (pass --exclude-scenes-from for "evaluated")')
    m = populations[pop]
    checks = []
    for name, rule in lock['thresholds'].items():
        if name not in m or not isinstance(m[name], (int, float)) or isinstance(m[name], bool):
            raise SystemExit(f'lock metric {name!r} is not a numeric metric of this report')
        if rule['op'] not in OPS:
            raise SystemExit(f'unknown op {rule["op"]!r}')
        checks.append(dict(metric=name, op=rule['op'], threshold=rule['value'], observed=m[name], pass_=OPS[rule['op']](m[name], rule['value'])))
    status = 'PASS' if all(c['pass_'] for c in checks) else 'FAIL'
    if lock.get('require_integrity', True) and not integrity['ok']:
        status = 'INVALID'
    return dict(status=status, population=pop, checks=checks, require_integrity=lock.get('require_integrity', True),
                bank_sha256_checked=bool(lock.get('bank_sha256')))


def fmt(x, pct=False):
    if x is None:
        return 'n/a'
    if isinstance(x, float):
        return f'{100 * x:.1f}' if pct else f'{x:.3f}'
    return str(x)


def markdown(report):
    m, integ, v = report['meta'], report['integrity'], report['verdict']
    lines = [f"# Sender gate — {m['sender']} on {Path(m['bank']).name} ({m['split']})", '',
             f"**Verdict: {v['status']}**" + (f" (population `{v.get('population')}`, lock `{report['lock']['path']}` sha256 `{report['lock']['sha256'][:12]}…`)" if report.get('lock') else ''),
             '', f"Packets `{m['packets']}` (sha256 `{m['packets_sha256'][:12]}…`); bank `{m['bank']}` (sha256 `{m['bank_sha256'][:12]}…`, "
             f"{m['n_bank_scenes']} scenes / {m['n_bank_worlds']} worlds). Generated {report['utc']} by `scripts/latest0924/sender_gate.py` "
             f"(sha256 `{report['script_sha256'][:12]}…`).", '',
             f"Integrity: **{'OK' if integ['ok'] else 'FAILED'}** — header {integ['header']}; nonzero counts "
             f"{ {k: c for k, c in integ['counts'].items() if c} or 'none'}; file summary matches recount: {integ['file_summary_matches_recount']}.", '']
    head = '| metric | ' + ' | '.join(report['populations']) + ' |'
    lines += [head, '|' + '---|' * (1 + len(report['populations']))]
    P = report['populations']

    def row(label, f):
        lines.append(f'| {label} | ' + ' | '.join(f(P[p]) for p in P) + ' |')
    row('messages (scenes)', lambda x: f"{x['n_packets']} ({x['n_scenes']})")
    row('parse fully / partially / failed', lambda x: f"{x['parse_ok']} / {x['parse_partial']} / {x['parse_empty'] + x['parse_malformed']}")
    row('parse fully (%) [95% CI]', lambda x: f"{fmt(x['parse_fully_rate'], 1)} [{fmt(x['parse_fully_rate_ci_lo'], 1)}, {fmt(x['parse_fully_rate_ci_hi'], 1)}]")
    row('fact recall (Table 17)', lambda x: f"{x['required_fact_hits']}/{x['required_fact_total']} ({fmt(x['required_fact_recall'], 1)}) [{fmt(x['required_fact_recall_ci_lo'], 1)}, {fmt(x['required_fact_recall_ci_hi'], 1)}]")
    row('complete objects (Table 17)', lambda x: f"{x['complete_objects']}/{x['n_objects']} ({fmt(x['complete_object_rate'], 1)}) [{fmt(x['complete_object_rate_ci_lo'], 1)}, {fmt(x['complete_object_rate_ci_hi'], 1)}]")
    row('complete packets', lambda x: f"{x['complete_packets']}/{x['n_packets']} ({fmt(x['complete_packet_rate'], 1)}) [{fmt(x['complete_packet_rate_ci_lo'], 1)}, {fmt(x['complete_packet_rate_ci_hi'], 1)}]")
    row('wrong-object lanes (words)', lambda x: f"{x['wrong_object_lanes']} ({x['wrong_object_words']})")
    row('unsupported-attribute lanes', lambda x: str(x['unsupported_attribute_lanes']))
    row('extra non-attribute lanes', lambda x: str(x['extra_non_attribute_lanes']))
    row('empty lanes', lambda x: str(x['empty_lanes']))
    row('mean words per object', lambda x: f"{x['mean_words_per_object']:.2f} {x['words_per_object_hist']}")
    row('precision / exact set (B6 defs)', lambda x: f"{fmt(x['precision'], 1)} / {fmt(x['exact_set_rate'], 1)}")
    row('mean bytes (Table 17)', lambda x: f"{x['mean_bytes']:.1f}")
    row('generated tokens mean / max', lambda x: f"{fmt(x['mean_generated_tokens'])} / {fmt(x['max_generated_tokens'])}")
    row('hit max_new_tokens / no final channel', lambda x: f"{fmt(x['hit_max_new_tokens'])} / {fmt(x['final_channel_missing'])}")
    row('greedy argmax mismatches', lambda x: fmt(x['greedy_argmax_mismatches']))
    if v['checks']:
        lines += ['', '| threshold | op | value | observed | pass |', '|---|---|---|---|---|']
        for c in v['checks']:
            lines.append(f"| {c['metric']} | {c['op']} | {c['threshold']} | {fmt(c['observed']) if isinstance(c['observed'], float) else c['observed']} | {'PASS' if c['pass_'] else 'FAIL'} |")
    lines += ['', 'Recall counts required attribute words assigned to the correct object; complete objects contain both required facts; '
              'extra selected words are allowed (Table 17 definitions, engineering scenes included in `all`). '
              'Intervals: scene-clustered percentile bootstrap, 2000 draws, seed 0. Message content only — not a receiver outcome.', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--packets', type=Path, required=True)
    ap.add_argument('--bank', type=Path, default=DEV_BANK)
    ap.add_argument('--lock', type=Path, default=None)
    ap.add_argument('--exclude-scenes-from', type=Path, default=None, help='JSON with engineering_scene_ids (population "evaluated")')
    ap.add_argument('--reference-words', type=Path, default=REFERENCE_WORDS)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--md', type=Path, default=None)
    a = ap.parse_args()
    meta, integrity, populations = evaluate(a.packets, a.bank, a.exclude_scenes_from, a.reference_words)
    lock = json.loads(a.lock.read_text()) if a.lock else None
    if lock and lock.get('bank_sha256') and lock['bank_sha256'] != meta['bank_sha256']:
        raise SystemExit(f"lock is for bank {lock['bank_sha256']}, got {meta['bank_sha256']}")
    v = verdict(lock, integrity, populations)
    report = dict(utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), script_sha256=sha_file(__file__),
                  meta=meta, lock=(dict(path=str(a.lock), sha256=sha_file(a.lock), content=lock) if lock else None),
                  integrity=integrity, populations=populations, verdict=v,
                  bootstrap=dict(unit='scene', draws=N_BOOT, seed=SEED, interval='95% percentile'))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1) + '\n')
    md = a.md or a.out.with_suffix('.md')
    md.write_text(markdown(report))
    p = populations['all']
    print(f"[sender_gate] {meta['sender']} {Path(meta['bank']).name}: verdict {v['status']}; integrity {'OK' if integrity['ok'] else 'FAILED'}; "
          f"parse ok {p['parse_ok']}/{p['n_packets']}; recall {p['required_fact_hits']}/{p['required_fact_total']} "
          f"({100 * p['required_fact_recall']:.1f}%); complete objects {p['complete_objects']}/{p['n_objects']} "
          f"({100 * p['complete_object_rate']:.1f}%); complete packets {p['complete_packets']}/{p['n_packets']}; "
          f"wrong-object lanes {p['wrong_object_lanes']}; mean words/object {p['mean_words_per_object']:.2f}; mean bytes {p['mean_bytes']:.1f}"
          f" -> {a.out}", flush=True)


if __name__ == '__main__':
    main()
