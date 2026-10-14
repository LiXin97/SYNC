"""Experiment B (latest0924) analysis: graded weights delivered internally vs as text. CPU-only.

Imports nothing from the runner (numpy + stdlib; the bank and packets are read as data). Per receiver root:
  integrity   pinned inputs re-hashed; receipts validated (status, raw hash, row count, manifest digest); every row's
              gold re-derived from the bank facts and its weight-gold from the packets file; every level arm's packet
              hex / text / active entries / write positions checked against the packets file; stored scores re-parsed
              with an independent strict first-line parser (64 and 256 tokens); engineering checks required to pass;
              the no-message row identical across worlds; the identity write exactly inert; every reassignment
              norm-matched per layer to the internal write of the same scene, world and level; L=255 outputs compared
              across worlds (identical packets must give identical outputs).
  estimates   per route x level: four-answer success J (family-macro over 36 families; template-macro over 12 parent
              templates), coverage, accuracy, changed/unchanged components R/U, X/Y accuracy, J at 256 tokens;
              cluster-bootstrap intervals (20,000 draws x 3 seeds, the interval enclosing all three seeds).
  contrasts   PRIMARY  internal - text, pooled over L in {64, 128, 191}, scene-paired
              per-level internal - text; difference-in-differences vs L=0; internal - reassignment at L=128;
              graded cost per route (mean over the pooled levels of J(L) - J(0)).
              Each at both cluster levels; the reported interval encloses both levels, p = the larger two-sided p.
  multiplicity  Holm across the declared four receivers on the primary contrast (a receiver without rows enters
              with p = 1 and is reported NOT RUN).
  continuity  optional --r3-root KEY=ROOT: the L=0 rows vs the paper's R3 complete-fact rows (same scenes, same prompts).

Decision rules are those of artifacts/latest0924/graded/DESIGN.md; nothing here is tuned or moved.

Usage:
  python3 scripts/latest0924/analyze_graded.py --roots artifacts/latest0924/graded/{qwen3-32b,qwen3.5-27b,...} \
      --output artifacts/latest0924/graded/analysis.json --markdown artifacts/latest0924/graded/analysis.md \
      [--r3-root qwen3-32b=artifacts/receiver_recovery/r3/qwen3-32b] [--allow-partial]
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEEDS = (20260924, 20260925, 20260926)      # the latest0924 study seeds (gates/latest0924/expA_protocol.lock.json)
DRAWS = 20000
LEVELS = (0, 64, 128, 191, 255)
POOLED = (64, 128, 191)
HOLM_FAMILY = ('qwen3.5-27b', 'qwen3.5-122b-a10b', 'gpt-oss-20b', 'qwen3-32b')
ALPHA = 0.05
MARGIN = 0.03                                # "similar": the enclosing interval lies inside [-3, +3] points
POSITIVE_J = 0.90                            # L=0 positive control, per route
POSITIVE_COVERAGE = 0.90
METRICS = ('J', 'accuracy', 'coverage', 'R', 'U', 'x_accuracy', 'y_accuracy', 'J256')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def answer(text, aliases):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    v = (lines[0] if lines else '').strip(' \t\r.,:;!?"\'`').upper()
    return v if v in aliases else None


def env(values, with_p=False):
    v = np.asarray(values, dtype=float)
    lo, hi, pooled = [], [], []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        d = v[rng.integers(len(v), size=(DRAWS, len(v)))].mean(axis=1)
        lo.append(float(np.quantile(d, .025))); hi.append(float(np.quantile(d, .975))); pooled.append(d)
    r = dict(mean=float(v.mean()), lo=min(lo), hi=max(hi), clusters=int(len(v)))
    if with_p:
        p = np.concatenate(pooled)
        r['p_two_sided'] = float(min(1.0, max(2 * min((p <= 0).mean(), (p >= 0).mean()), 1.0 / len(p))))
    return r


def holm(pvals, alpha=ALPHA):
    order = sorted(pvals, key=lambda k: (pvals[k], k)); m = len(order); out = {}; blocked = False
    for i, k in enumerate(order):
        thr = alpha / (m - i); rej = (pvals[k] <= thr) and not blocked
        if not rej:
            blocked = True
        out[k] = dict(p=pvals[k], threshold=thr, reject=bool(rej))
    return out


class AuditFailure(Exception):
    pass


def fail(msg):
    raise AuditFailure(msg)


def parse_arm(arm):
    if arm in ('none', 'identity_write'):
        return arm, None
    head, level = arm.split('_L')
    return ('sham' if head.startswith('sham') else head), int(level)


# ------------------------------------------------------------------------------------------------ one receiver

def load_rows(root, m, allow_partial):
    current = digest(root / 'manifest.json')
    raws = sorted(root.glob('raw/*.jsonl'))
    if not raws:
        return current, {}, {}
    newest = max(os.path.getmtime(p) for p, w in m['pins'].items() if w != 'PENDING')
    earliest = min(os.path.getmtime(r) for r in raws)
    if newest >= earliest:
        fail(f'{root.name}: a pinned input was modified after the first row')
    receipts, rows = {}, defaultdict(dict)
    for raw in raws:
        rp = raw.with_suffix('.done.json')
        if not rp.exists():
            if allow_partial:
                continue
            fail(f'{root.name}: {raw.name} has rows but no receipt')
        r = json.loads(rp.read_text()); n = sum(1 for _ in open(raw))
        if r.get('status') != 'PASS' or r.get('raw_sha256') != digest(raw) or r.get('rows') != n or r.get('manifest_sha256') != current:
            fail(f'{root.name}: receipt {raw.name} invalid')
        receipts[raw.name[:-6]] = r
        phase = r['phase']
        for line in open(raw):
            row = json.loads(line)
            k = (row['scene'], row['world'], row['question'], row['arm'])
            if row['key'] != '|'.join(map(str, k)) or row['phase'] != phase:
                fail(f'{root.name}: inconsistent key {k}')
            if k in rows[phase]:
                fail(f'{root.name}: duplicate row {k}')
            rows[phase][k] = row
    return current, receipts, rows


def check_rows(root, m, bank, packets, phase, rec):
    name = root.name
    eng_ids = set(m['engineering_scene_ids'])
    tiers = set(m['tiers'])
    by_id = {s['scene']: s for s in bank['engineering_scenes'] + bank['scenes']}
    for k, r in rec.items():
        sid, w, q, arm = k
        scene = by_id[sid]
        if phase == 'engineering' and sid not in eng_ids:
            fail(f'{name}: non-engineering scene in the engineering phase {k}')
        if phase == 'development' and (sid in eng_ids or bank['tier_of'][sid] not in tiers):
            fail(f'{name}: scene outside the manifest tiers in development {k}')
        field = scene['changed_field'] if q == 0 else next(f for f in scene['fields'] if f != scene['changed_field'])
        owners = [i for i, o in enumerate(scene['worlds'][w]['facts']) if o[field] == scene['queried_values'][q]]
        if not (len(owners) == 1 and r['gold'] == scene['aliases'][owners[0]] == scene['worlds'][w]['answers'][q]):
            fail(f'{name}: gold does not re-derive for {k}')
        route, level = parse_arm(arm)
        if (r['route'], r['level']) != (route, level):
            fail(f'{name}: route/level fields disagree with the arm for {k}')
        if level is not None:
            p = packets[level][f'{sid}|{w}']
            if r['condition_packet_hex'] != p['hex'] or r['words_by_object'] != p['words_by_object'] or r['weights_by_object'] != p['weights_by_object']:
                fail(f'{name}: packet differs from the packets file for {k}')
            if r['entries'] != p['entries']:
                fail(f'{name}: active entries differ from the packets file for {k}')
            ws = [dict(zip(p['words_by_object'][o], p['weights_by_object'][o])).get(scene['queried_values'][q], 0) for o in (0, 1)]
            wg = None if ws[0] == ws[1] else scene['aliases'][0 if ws[0] > ws[1] else 1]
            if r['weight_gold'] != wg or (wg is not None and wg != r['gold']):
                fail(f'{name}: weight gold does not re-derive for {k}')
            if route == 'text' and r['extra_text'] != p['text']:
                fail(f'{name}: text-route message differs from the packets file for {k}')
            if route in ('internal', 'sham'):
                slot = {(s['object'], s['index']): s['last_position'] for s in r['slots']}
                want = sorted(slot[e['object'], e['index']] for e in p['entries'])
                if r['patch_positions'] != want or set(r['patch_report']) != {str(l) for l in m['selected_layers']}:
                    fail(f'{name}: write positions/layers wrong for {k}')
                if r['gamma'] != m['gamma']:
                    fail(f'{name}: gamma wrong for {k}')
        if r['prefix_cache_sha256'] != r['branch_initial_cache_sha256']:
            fail(f'{name}: cache fingerprint mismatch for {k}')
        for b in (64, 256):
            parsed = answer(r[f'text{b}'], r['aliases'])
            if r[f'score{b}'] != dict(answer=parsed, answered=parsed is not None, correct=parsed == r['gold']):
                fail(f'{name}: stored score{b} disagrees with the re-parse for {k}')
        if phase == 'engineering' and not (r['engineering_checks'] and r['engineering_checks']['pass']):
            fail(f'{name}: engineering checks failed for {k}')
    # cross-row integrity
    l255 = dict(pairs=0, identical=0)
    for (sid, w, q, arm), r in rec.items():
        route, level = parse_arm(arm)
        if route == 'none':
            o = rec.get((sid, 1 - w, q, 'none'))
            if o and (r['logits_sha256'], r['tokens']) != (o['logits_sha256'], o['tokens']):
                fail(f'{name}: no-message row depends on the world for {sid}')
        if route == 'identity_write':
            ref = rec.get((sid, w, q, 'none'))
            if ref is None or (r['logits_sha256'], r['tokens']) != (ref['logits_sha256'], ref['tokens']):
                fail(f'{name}: identity write is not inert for {sid}')
            if not r['patch_report'] or any(v['norm'] > 1e-6 for v in r['patch_report'].values()):
                fail(f'{name}: identity write reported a nonzero edit for {sid}')
        if route == 'sham':
            ref = rec.get((sid, w, q, f'internal_L{level:03d}'))
            if ref is None:
                fail(f'{name}: reassignment without its internal reference for {sid}')
            if r['patch_positions'] != ref['patch_positions'] or set(r['patch_report']) != set(ref['patch_report']):
                fail(f'{name}: reassignment positions/layers differ from the internal write for {sid}')
            for li, v in r['patch_report'].items():
                want = ref['patch_report'][li]['norm']
                if abs(v['norm'] - want) > max(1e-4, 1e-4 * want):
                    fail(f'{name}: reassignment norm not matched at layer {li} for {sid}')
                if v['target_sha256'] == ref['patch_report'][li]['target_sha256']:
                    fail(f'{name}: reassignment target identical to the internal write for {sid}')
        if level == 255 and w == 0:
            o = rec.get((sid, 1, q, arm))
            if o is not None:
                l255['pairs'] += 1
                l255['identical'] += int((r['logits_sha256'], r['tokens']) == (o['logits_sha256'], o['tokens']))
    if phase == 'engineering' and l255['pairs'] != l255['identical']:
        fail(f'{name}: identical L=255 packets gave different outputs across worlds in the engineering phase')
    return l255


def scene_metrics(rec, scenes, arm, budget=64):
    per = defaultdict(dict)
    for s in scenes:
        rr = [rec[(s, w, q, arm)] for w in (0, 1) for q in (0, 1)]
        aa = [answer(r[f'text{budget}'], r['aliases']) for r in rr]
        ok = [a == r['gold'] for a, r in zip(aa, rr)]
        ok256 = [answer(r['text256'], r['aliases']) == r['gold'] for r in rr]
        per['J'][s] = float(all(ok)); per['accuracy'][s] = sum(ok) / 4
        per['coverage'][s] = sum(a is not None for a in aa) / 4
        per['R'][s] = float(ok[0] and ok[2]); per['U'][s] = float(ok[1] and ok[3])
        per['x_accuracy'][s] = (ok[0] + ok[2]) / 2; per['y_accuracy'][s] = (ok[1] + ok[3]) / 2
        per['J256'][s] = float(all(ok256))
    return per


def clustered(per_scene, levels, with_p=False):
    out = {}
    for name, mp in levels.items():
        g = defaultdict(list)
        for s, v in per_scene.items():
            g[mp[s]].append(v)
        out[name] = env([float(np.mean(v)) for _, v in sorted(g.items())], with_p)
    return out


def contrast(per_scene, levels):
    c = clustered(per_scene, levels, with_p=True)
    c['enclosing'] = dict(mean=c['family']['mean'], lo=min(c['family']['lo'], c['template']['lo']),
                          hi=max(c['family']['hi'], c['template']['hi']),
                          p=max(c['family']['p_two_sided'], c['template']['p_two_sided']))
    return c


def audit_root(root, allow_partial, r3_root=None):
    m = json.loads((root / 'manifest.json').read_text())
    pend = [p for p, w in m['pins'].items() if w == 'PENDING']
    if pend:
        fail(f'{root.name}: manifest still has PENDING pins {pend}')
    bad = [p for p, w in m['pins'].items() if not Path(p).exists() or digest(p) != w]
    if bad:
        fail(f'{root.name}: pinned inputs changed: {bad[:2]}')
    current, receipts, rows = load_rows(root, m, allow_partial)
    out = dict(root=str(root), receiver=m['receiver'], manifest_sha256=current, tiers=m['tiers'],
               selected_layers=m['selected_layers'], gamma=m['gamma'], receipts=len(receipts))
    if not rows:
        out['status'] = 'NO_ROWS'
        return out
    bank = json.loads(Path(m['bank']).read_text())
    packets = {int(L): json.loads(Path(p).read_text())['packets'] for L, p in m['packets'].items()}
    out['throughput'] = {k: dict(rows=v['rows'], elapsed_seconds=v['elapsed_seconds'],
                                 seconds_per_computed_row=v.get('seconds_per_computed_row'), model_dtype=v.get('model_dtype'),
                                 adapter_installed=(v.get('adapter') or {}).get('installed'), cuda_devices=v.get('cuda_devices'))
                         for k, v in receipts.items()}
    out['phases'] = {}
    for phase, rec in rows.items():
        l255 = check_rows(root, m, bank, packets, phase, rec)
        arms = sorted({k[3] for k in rec})
        scenes = sorted({k[0] for k in rec})
        complete = [s for s in scenes if all((s, w, q, a) in rec for w in (0, 1) for q in (0, 1) for a in arms)]
        if len(complete) != len(scenes) and not allow_partial:
            fail(f'{root.name}: {phase} has incomplete scenes')
        scenes = complete
        levels = {'family': {s: rec[(s, 0, 0, 'none')]['family'] for s in scenes},
                  'template': {s: f"{rec[(s, 0, 0, 'none')]['domain']}/T{rec[(s, 0, 0, 'none')]['template']}" for s in scenes}}
        ph = dict(scenes=len(scenes), rows=len(rec), arms=arms, l255_world_pairs=l255,
                  cluster_counts={k: len(set(v.values())) for k, v in levels.items()})
        out['phases'][phase] = ph
        if phase != 'development' or not scenes:
            continue
        J, cells = {}, {}
        for arm in arms:
            per = scene_metrics(rec, scenes, arm)
            J[arm] = per['J']
            cells[arm] = {name: clustered(per[name], levels) for name in METRICS}
            cells[arm]['J_sum'] = int(sum(per['J'].values()))
            cells[arm]['R_sum'] = int(sum(per['R'].values()))
            cells[arm]['x_accuracy_max_scene'] = max(per['x_accuracy'].values())
        ph['cells'] = cells
        have = lambda *a: all(x in J for x in a)
        C = {}
        if have(*[f'{r}_L{L:03d}' for r in ('internal', 'text') for L in POOLED]):
            C['primary_internal_minus_text_pooled'] = contrast(
                {s: float(np.mean([J[f'internal_L{L:03d}'][s] - J[f'text_L{L:03d}'][s] for L in POOLED])) for s in scenes}, levels)
        for L in LEVELS:
            if have(f'internal_L{L:03d}', f'text_L{L:03d}'):
                C[f'internal_minus_text_L{L:03d}'] = contrast({s: J[f'internal_L{L:03d}'][s] - J[f'text_L{L:03d}'][s] for s in scenes}, levels)
        if have(*[f'{r}_L{L:03d}' for r in ('internal', 'text') for L in POOLED + (0,)]):
            C['did_internal_minus_text_vs_L0'] = contrast(
                {s: float(np.mean([(J[f'internal_L{L:03d}'][s] - J['internal_L000'][s]) - (J[f'text_L{L:03d}'][s] - J['text_L000'][s])
                                   for L in POOLED])) for s in scenes}, levels)
            for r in ('internal', 'text'):
                C[f'graded_cost_{r}'] = contrast({s: float(np.mean([J[f'{r}_L{L:03d}'][s] - J[f'{r}_L000'][s] for L in POOLED])) for s in scenes}, levels)
        for arm in arms:
            route, level = parse_arm(arm)
            if route == 'sham' and have(f'internal_L{level:03d}'):
                C[f'internal_minus_{arm}'] = contrast({s: J[f'internal_L{level:03d}'][s] - J[arm][s] for s in scenes}, levels)
        ph['contrasts'] = C
        # controls
        ctrl = {}
        for r in ('internal', 'text'):
            a0 = f'{r}_L000'
            if a0 in cells:
                ctrl[f'positive_L0_{r}'] = dict(J=cells[a0]['J']['family']['mean'], coverage=cells[a0]['coverage']['family']['mean'],
                                                pass_=bool(cells[a0]['J']['family']['mean'] >= POSITIVE_J and
                                                           cells[a0]['coverage']['family']['mean'] >= POSITIVE_COVERAGE))
            a255 = f'{r}_L255'
            if a255 in cells:
                ctrl[f'negative_L255_{r}'] = dict(J_sum=cells[a255]['J_sum'], R_sum=cells[a255]['R_sum'],
                                                  x_accuracy_max_scene=cells[a255]['x_accuracy_max_scene'],
                                                  pass_=bool(cells[a255]['J_sum'] == 0 and cells[a255]['R_sum'] == 0))
        if 'none' in cells:
            ctrl['no_message'] = dict(J_sum=cells['none']['J_sum'], pass_=cells['none']['J_sum'] == 0)
        ctrl['l255_outputs_identical_across_worlds'] = dict(**l255, pass_=l255['pairs'] == l255['identical'])
        ph['controls'] = ctrl
        # continuity with the paper's R3 rows (qwen3-32b): same prompts, same packets, same code path at L=0
        if r3_root is not None:
            ref = {}
            for raw in sorted(Path(r3_root).glob('raw/development_*.jsonl')):
                for line in open(raw):
                    r = json.loads(line)
                    if r['source'] == 'literal' and r['arm'] in ('selected', 'packet_text'):
                        ref[(r['scene'], r['world'], r['question'], r['arm'])] = r
            cont = {}
            for mine, theirs in (('internal_L000', 'selected'), ('text_L000', 'packet_text')):
                n = same_tokens = same_logits = same_prompt = 0
                for s in scenes:
                    for w in (0, 1):
                        for q in (0, 1):
                            t = ref.get((s, w, q, theirs))
                            if t is None:
                                continue
                            r = rec[(s, w, q, mine)]
                            n += 1
                            same_tokens += int(r['tokens'] == t['tokens'])
                            same_logits += int(r['logits_sha256'] == t['logits_sha256'])
                            same_prompt += int(r['prefix_ids'] == t['prefix_ids'] and r['suffix_ids'] == t['suffix_ids'])
                cont[f'{mine}_vs_R3_literal_{theirs}'] = dict(answers=n, identical_prompt=same_prompt, identical_tokens=same_tokens,
                                                              identical_logits_sha256=same_logits)
            ph['continuity_r3'] = cont
    out['status'] = 'PASS_INDEPENDENT_GRADED_ROOT_AUDIT'
    return out


# ------------------------------------------------------------------------------------------------ report

def verdict(v, H):
    dev = v.get('phases', {}).get('development')
    rcv = v['receiver']
    if not dev or 'contrasts' not in dev or 'primary_internal_minus_text_pooled' not in dev['contrasts']:
        return dict(verdict='NOT RUN', interpretable=False)
    c = dev['contrasts']['primary_internal_minus_text_pooled']['enclosing']
    h = H.get(rcv, {})
    if h.get('reject') and c['lo'] > 0:
        call = 'internal better than text'
    elif h.get('reject') and c['hi'] < 0:
        call = 'text better than internal'
    elif c['lo'] >= -MARGIN and c['hi'] <= MARGIN:
        call = 'similar (within +/-3 points)'
    else:
        call = 'unresolved'
    ctrl = dev['controls']
    reasons = [k for k in ('positive_L0_internal', 'positive_L0_text', 'negative_L255_internal', 'negative_L255_text', 'no_message',
                           'l255_outputs_identical_across_worlds') if not ctrl.get(k, {}).get('pass_', False)]
    return dict(verdict=call, interpretable=not reasons, failed_controls=reasons, primary=c, holm=h)


def pct(x):
    return f'{100 * x:.1f}'


def ci(d):
    return f"{pct(d['mean'])} [{pct(d['lo'])}, {pct(d['hi'])}]"


def markdown(report):
    L = [f"# Experiment B (graded weights): internal vs text — analysis", '',
         f"Generated {report['utc']} by `scripts/latest0924/analyze_graded.py` (sha256 `{report['script_sha256'][:16]}`). "
         f"Status: **{report['status']}**. Intervals: family-level (36 families) 95% cluster bootstrap, {DRAWS:,} draws x "
         f"{len(SEEDS)} seeds, enclosing interval; contrasts also report the template level (12) and the interval enclosing both.", '']
    L += ['## Verdicts (primary: internal - text, pooled over L = 64, 128, 191; Holm across the four declared receivers)', '',
          '| receiver | verdict | interpretable | estimate [enclosing 95%] (points) | p (max of levels) | Holm threshold | failed controls |',
          '|---|---|---|---|---|---|---|']
    for rcv in HOLM_FAMILY:
        vd = report['verdicts'].get(rcv, dict(verdict='NOT RUN'))
        if vd['verdict'] == 'NOT RUN':
            L.append(f'| {rcv} | NOT RUN | - | - | - | - | - |'); continue
        c = vd['primary']; h = vd['holm']
        L.append(f"| {rcv} | {vd['verdict']} | {vd['interpretable']} | {pct(c['mean'])} [{pct(c['lo'])}, {pct(c['hi'])}] | "
                 f"{c['p']:.4g} | {h.get('threshold', float('nan')):.4g} | {', '.join(vd['failed_controls']) or '-'} |")
    L.append('')
    for name, v in report['receivers'].items():
        dev = v.get('phases', {}).get('development')
        L += [f"## {v['receiver']} ({v.get('status')})", '']
        if not dev or 'cells' not in dev:
            L += ['No development rows.', '']; continue
        L += [f"Scenes {dev['scenes']}, rows {dev['rows']}, clusters {dev['cluster_counts']}, tiers {v['tiers']}, layer {v['selected_layers']}, gamma {v['gamma']}.", '']
        L += ['| route | metric | ' + ' | '.join(f'L={l}' for l in LEVELS) + ' |', '|---|---|' + '---|' * len(LEVELS)]
        for route in ('internal', 'text'):
            for metric in ('J', 'coverage', 'R', 'U', 'x_accuracy', 'J256'):
                cells = [dev['cells'].get(f'{route}_L{l:03d}') for l in LEVELS]
                L.append(f'| {route} | {metric} | ' + ' | '.join(ci(c[metric]['family']) if c else '-' for c in cells) + ' |')
        for arm, c in dev['cells'].items():
            if arm.startswith('sham') or arm == 'none':
                L.append(f"| {arm} | J / coverage | {ci(c['J']['family'])} / {ci(c['coverage']['family'])} |" + ' |' * (len(LEVELS) - 1))
        L += ['', '| contrast | family | template | enclosing | p (max) |', '|---|---|---|---|---|']
        for k, c in dev['contrasts'].items():
            e = c['enclosing']
            L.append(f"| {k} | {ci(c['family'])} | {ci(c['template'])} | [{pct(e['lo'])}, {pct(e['hi'])}] | {e['p']:.4g} |")
        L += ['', 'Controls: ' + '; '.join(f"{k}: {'PASS' if c.get('pass_') else 'FAIL'} "
                                           f"({', '.join(f'{a}={b}' for a, b in c.items() if a != 'pass_')})" for k, c in dev['controls'].items()), '']
        if 'continuity_r3' in dev:
            L += ['Continuity with the paper\'s R3 complete-fact rows: ' + '; '.join(f'{k}: {c}' for k, c in dev['continuity_r3'].items()), '']
        thr = [f"{k}: {t['seconds_per_computed_row']:.3f} s/row ({t['model_dtype']}, adapter {t['adapter_installed']}, GPUs {t['cuda_devices']})"
               for k, t in v.get('throughput', {}).items() if t.get('seconds_per_computed_row')]
        if thr:
            L += ['Throughput: ' + '; '.join(thr), '']
    L += ['## Limits', '', report['limits'], '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--roots', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--markdown', type=Path, default=None)
    ap.add_argument('--r3-root', action='append', default=[], help='KEY=ROOT, e.g. qwen3-32b=artifacts/receiver_recovery/r3/qwen3-32b')
    ap.add_argument('--allow-partial', action='store_true')
    a = ap.parse_args()
    r3 = dict(x.split('=', 1) for x in a.r3_root)
    receivers, errors = {}, {}
    for root in a.roots:
        if not (root / 'manifest.json').exists():
            continue
        key = json.loads((root / 'manifest.json').read_text())['receiver']
        try:
            receivers[root.name] = audit_root(root, a.allow_partial, r3.get(key))
        except AuditFailure as e:
            errors[root.name] = str(e)
            receivers[root.name] = dict(root=str(root), receiver=key, status='AUDIT_FAIL', error=str(e))
    pv = {}
    for rcv in HOLM_FAMILY:
        v = next((x for x in receivers.values() if x['receiver'] == rcv), None)
        dev = (v or {}).get('phases', {}).get('development', {})
        c = dev.get('contrasts', {}).get('primary_internal_minus_text_pooled')
        pv[rcv] = c['enclosing']['p'] if c else 1.0          # a receiver without rows enters with p = 1
    H = holm(pv)
    verdicts = {v['receiver']: verdict(v, H) for v in receivers.values() if v.get('status') == 'PASS_INDEPENDENT_GRADED_ROOT_AUDIT'}
    for rcv in HOLM_FAMILY:
        verdicts.setdefault(rcv, dict(verdict='NOT RUN', interpretable=False))
    status = ('AUDIT_FAIL' if errors else
              'PASS_INDEPENDENT_GRADED_AUDIT' if all(v.get('status') == 'PASS_INDEPENDENT_GRADED_ROOT_AUDIT' for v in receivers.values()) else
              'PARTIAL_GRADED_AUDIT')
    report = dict(status=status, utc=datetime.now(timezone.utc).isoformat(), script_sha256=digest(Path(__file__)),
                  seeds=list(SEEDS), draws=DRAWS, pooled_levels=list(POOLED), holm_family=list(HOLM_FAMILY), holm=H,
                  margin=MARGIN, positive_control=dict(J=POSITIVE_J, coverage=POSITIVE_COVERAGE),
                  verdicts=verdicts, receivers=receivers, errors=errors,
                  limits=('Scene-paired repeated measures: routes and levels live on the same scenes and add no independent N. '
                          'Intervals are cluster-bootstrap envelopes over three seeds at the family and template levels; seeds are not '
                          'model replications. The L=255 and no-message rows are structural zeros of four-answer success for a '
                          'deterministic receiver, not evidence about communication. Complete facts only (no sender). The bank is finite '
                          'and constructed; the default evaluation set is a pre-declared balanced subset of it.'))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    if a.markdown:
        a.markdown.write_text(markdown(report))
    print(status)
    for rcv, vd in verdicts.items():
        if vd['verdict'] == 'NOT RUN':
            print(f'  {rcv:20s} NOT RUN'); continue
        c = vd['primary']
        print(f"  {rcv:20s} {vd['verdict']:32s} interpretable={vd['interpretable']} primary {pct(c['mean'])} "
              f"[{pct(c['lo'])}, {pct(c['hi'])}] p={c['p']:.4g} holm_reject={vd['holm'].get('reject')} failed={vd['failed_controls']}")
    for k, e in errors.items():
        print('  FAIL', k, e)
    if errors:
        sys.exit(2)


if __name__ == '__main__':
    main()
