#!/usr/bin/env python3
"""Experiment A (latest0924) frozen analysis. CPU only: numpy + stdlib; nothing is imported from the runner.

Implements gates/latest0924/expA_protocol.lock.json (endpoints, H1-H4, ci, same_model_rule, same_model_pairs, not_claims)
on the raw rows under artifacts/latest0924/expA/<sender>/<receiver>/. Every threshold below is copied from the lock and
the script refuses to run if the lock text no longer contains the copied values. Nothing is tuned or moved here.

Per cell (sender x receiver, both from the lock):
  engineering   the engineering rows are re-checked row by row under the gate that applies -- the lock's gate as the
                pinned evaluator encodes it (max logit difference <= 0.05, alias-gap difference <= 0.1, equal parsed
                answers) or, for qwen3.5-122b-a10b only, ADR-latest0924-moe-gate (equal parsed answers and alias-gap
                difference <= 0.1); identity_write rows must reproduce the none rows' tokens and logits_sha256 with a zero
                edit; the none rows must not depend on the world. A `.done.json` receipt is NOT taken as evidence that the
                gate passed. A receiver that fails in any cell is an engineering failure for every sender (no outcome rows).
  completeness  every development shard must have a valid receipt (status PASS, raw sha256, row count, manifest sha256,
                shard index; all shards 0..S-1 present) and the unique keys must equal dev scenes x 2 worlds x 2 questions x
                arms. Otherwise the cell is INCOMPLETE (= NOT RUN for inference; rows present are reported, never
                extrapolated).
  integrity     pinned inputs re-hashed; gold and supplied gold re-derived from the bank; sender packets checked against the
                frozen packets file; complete-fact entries checked against the bank facts; full_text checked against the
                evidence; stored scores re-parsed from the generated text (both budgets, both golds); no-message rows
                identical across worlds; identity write inert; reassignments norm-matched to the selected write with a
                different target. Any failure -> AUDIT_FAIL (excluded, reasons listed).
  estimates     per (source, arm): four-answer success J (primary; 64 tokens; invalid = wrong), single-answer accuracy,
                coverage (fraction answered), changed (R) / unchanged (U) components, J / accuracy / coverage at 256 tokens;
                other_world also against the supplied gold. Family-macro (36 families) and template-macro (12 parent
                templates = domain x template) means, cluster bootstrap with scenes kept intact.
  contrasts     scene-paired differences, cluster bootstrap at both levels, two-sided p per level, p = the larger.
Across cells: H1 and H2 with Holm across each sender's evaluated cross-model receivers (m reported); H3 primary is
evaluated only if both qwen3.5-2b and qwen3.5-122b-a10b have rows, otherwise NOT EVALUABLE (no substitute); H3 secondary
Spearman and gap(27B) - gap(2B) are descriptive; H4 descriptive; same-model pairs reported separately and excluded from
every cross-model count; complete-fact references per receiver.

Usage:
  python3 scripts/latest0924/analyze_expA.py
      -> artifacts/latest0924/expA/analysis/expA_results.json and expA_results.md
  python3 scripts/latest0924/analyze_expA.py --lock L --expa-root D --adr-glob G --out-dir O     (tests / synthetic data)
Every field of the outputs is a deterministic function of the inputs except `generated_utc`.
"""
import argparse
import glob
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
DEFAULT_LOCK = ROOT / 'gates/latest0924/expA_protocol.lock.json'
DEFAULT_EXPA = ROOT / 'artifacts/latest0924/expA'
DEFAULT_ADR_GLOB = str(ROOT / 'artifacts/latest0924/decisions/ADR-latest0924-moe-gate-*.md')

# ---- constants copied from the lock (verify_lock() checks that the lock text still contains each of them)
SEEDS = (20260924, 20260925, 20260926)
DRAWS = 20000
ALPHA = 0.05
MARGIN = 0.03                 # H2 "similar": both intervals within [-3, +3] points
H1_J = 0.50
H1_COVERAGE = 0.90
BUDGETS = (64, 256)
GATE_MAX_LOGIT = 0.05
GATE_ALIAS_GAP = 0.1          # the pinned evaluator's third clause (src/syncomm/recovery/evaluate_confirm.py)
ADR_NAME = 'ADR-latest0924-moe-gate'
ADR_RECEIVERS = ('qwen3.5-122b-a10b',)
SENDER_ARMS = ['none', 'identity_write', 'original', 'selected', 'packet_text', 'sham0', 'sham1', 'sham2', 'other_world']
LITERAL_ARMS = ['selected', 'packet_text', 'full_text', 'native']
SHAMS = ('sham0', 'sham1', 'sham2')
# H3 ladder: total parameter counts (the lock's size strings; MoE entries use total parameters)
LADDER = {'qwen3.5-2b': 2e9, 'qwen3.5-4b': 4e9, 'qwen3.5-9b': 9e9, 'qwen3.5-27b': 27e9, 'qwen3.5-122b-a10b': 122e9,
          'qwen3.6-27b': 27e9, 'qwen3.6-35b-a3b': 35e9}
H3_PRIMARY = ('qwen3.5-122b-a10b', 'qwen3.5-2b')           # gap(122B) - gap(2B), preregistered
H3_DESCRIPTIVE = ('qwen3.5-27b', 'qwen3.5-2b')             # gap(27B) - gap(2B), descriptive only
METRICS = ('J', 'accuracy', 'coverage', 'R', 'U', 'J256', 'accuracy256', 'coverage256')
SUPPLIED_METRICS = ('J_supplied', 'accuracy_supplied', 'J256_supplied')
MAX_EXAMPLES = 25

RESOLUTIONS = [
    'Parent templates: the bank field `template` takes only 4 values (0-3); the lock says 12 parent templates. The 12 '
    'clusters are domain x template (`{domain}/T{template}`), the convention of every earlier audit of this bank '
    '(scripts/audit_r3_20260923.py) and of Experiment B (scripts/latest0924/analyze_graded.py). Families nest in them.',
    'Engineering gate: the lock text names max logit difference <= 0.05 with equal parsed answers; the pinned evaluator '
    '(the manifest pins src/syncomm/recovery/evaluate_confirm.py) also requires alias-gap difference <= 0.1. The script '
    'applies the evaluator\'s three-clause gate (stricter) and also reports the lock-literal two-clause count. '
    'qwen3.5-122b-a10b uses ADR-latest0924-moe-gate (parsed answers equal and alias gap <= 0.1).',
    'Engineering failure is receiver-level: a failure in any sender\'s cell makes the receiver an engineering failure for '
    'every sender (the lock: "A receiver failing the gate is reported as an engineering failure with no outcome rows").',
    '"Lower limit > 0 after Holm" (H1, H2): Holm (alpha 0.05) is applied to p = max(two-sided bootstrap p at the family '
    'level, at the template level), each p pooled over the 3 seeds x 20,000 draws; the decision additionally requires '
    'the unadjusted 95% seed-enclosing interval to exclude 0 in the stated direction at BOTH levels. This is the rule '
    'already used for Experiment B (scripts/latest0924/analyze_graded.py). Holm-adjusted p values are also reported.',
    'Holm family size m: the receivers EVALUATED for that sender (complete, engineering-passed, audit-passed, cross-model). '
    'As a sensitivity, the verdict is also computed with m = every planned cross-model receiver of that sender '
    '(unevaluated receivers enter with p = 1); disagreements are listed.',
    'H1 point criteria use family-macro point estimates: J(selected) >= 0.50 and coverage(selected) >= 0.90 at 64 tokens.',
    'H2 order: "internal better" / "text better" are checked before "similar" (the lock lists them first); a pair that '
    'is both significant and within +/-3 points is labelled by its significant direction and flagged within_margin=true. '
    '"Similar" uses the unadjusted 95% seed-enclosing intervals at both levels, inclusive bounds [-0.03, +0.03].',
    'Same-model pairs (lock same_model_pairs) are reported in their own table with unadjusted intervals and p (no Holm), '
    'are excluded from every cross-model count and Holm family, and are excluded from the H3 secondary ladder.',
    'H3 secondary: gap = family-macro J(selected) - J(packet_text) at 64 tokens; Spearman with average ranks over the '
    'available ladder points (qwen3.5-27b and qwen3.6-27b tie at 27e9); computed only when >= 3 points exist. Missing '
    'points (engineering failures, incomplete cells) are listed, never imputed.',
    '"Reassigned success/valid" in the main table = the range of J over sham0-sham2 / the mean coverage over sham0-sham2.',
    'Cells whose development shards are not all receipted are INCOMPLETE and are reported as NOT RUN for inference '
    '(the lock\'s execution_order), with the rows present.',
]


class LockError(Exception):
    pass


def verify_lock(lock):
    """Refuse to run if the lock no longer contains the constants copied into this script."""
    h = lock['hypotheses_and_decision_rules']
    need = [(lock['ci'], '20,000'), (lock['ci'], '36 families'), (lock['ci'], '12 parent templates'),
            (lock['ci'], 'enclosing all three'),
            (h['H1_fixed_rule_generalizes'], 'J(selected) >= 0.50'), (h['H1_fixed_rule_generalizes'], 'coverage >= 0.90'),
            (h['H1_fixed_rule_generalizes'], 'BOTH family and template'), (h['H1_fixed_rule_generalizes'], 'Holm'),
            (h['H2_internal_vs_text'], '[-3, +3]'), (h['H2_internal_vs_text'], 'Holm'),
            (h['H3_scale'], 'gap(122B) minus gap(2B)'), (h['H3_scale'], 'Spearman'), (h['H4_depth'], 'selected minus original'),
            (lock['engineering_gate'], '<= 0.05'), (lock['endpoints']['primary'], 'family-macro over 36'),
            (lock['numerics'], 'answer budgets 64 (primary) and 256')]
    need += [(lock['ci'], str(s)) for s in SEEDS]
    missing = [text for where, text in need if text not in where]
    if missing:
        raise LockError(f'lock text no longer contains {missing}; constants are not verbatim')
    for r in list(LADDER) + list(ADR_RECEIVERS):
        if r not in lock['receivers']:
            raise LockError(f'ladder/ADR receiver {r} not in the lock')
    pairs = []
    for p in lock['same_model_pairs']:
        a, b = [x.strip() for x in p.split('->')]
        pairs.append((a, b))
    return pairs


# ------------------------------------------------------------------------------------------------ inputs registry

def rel(p):
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


class Inputs:
    def __init__(self):
        self.files, self._cache = {}, {}

    def add(self, path, data, role):
        sha = hashlib.sha256(data).hexdigest()
        key = rel(path)
        prev = self.files.get(key)
        roles = sorted(set((prev or {}).get('roles', [])) | {role})
        self.files[key] = dict(sha256=sha, bytes=len(data), roles=roles)
        if prev is not None and prev['sha256'] != sha:
            self.files[key]['changed_between_reads'] = [prev['sha256'], sha]
        return sha

    def read(self, path, role):
        data = Path(path).read_bytes()
        return data, self.add(path, data, role)

    def sha_cached(self, path, role):
        key = str(Path(path).resolve())
        if key not in self._cache:
            self._cache[key] = self.read(path, role)[1]
        return self._cache[key]


def split_lines(data):
    """Complete (newline-terminated, non-blank) lines and whether a partial trailing line exists."""
    parts = data.split(b'\n')
    partial = parts[-1] != b''
    return [ln for ln in parts[:-1] if ln.strip()], partial


def parse_answer(text, aliases):
    """Independent re-implementation of the frozen strict first-line alias parser."""
    line = next((x.strip() for x in text.splitlines() if x.strip()), '')
    clean = line.strip(' \t\r.,:;!?"\'`').upper()
    return clean if clean in aliases else None


# ------------------------------------------------------------------------------------------------ bootstrap

_IDX = {}


def _draws(seed, n):
    key = (seed, n)
    if key not in _IDX:
        _IDX[key] = np.random.default_rng(seed).integers(n, size=(DRAWS, n))
    return _IDX[key]


def boot(cluster_means, with_p=False):
    """Percentile 95% cluster bootstrap per seed; the reported interval encloses all three seeds."""
    v = np.asarray(cluster_means, dtype=float)
    lo, hi, pooled, per_seed = [], [], [], []
    for s in SEEDS:
        d = v[_draws(s, len(v))].mean(axis=1)
        a, b = float(np.quantile(d, .025)), float(np.quantile(d, .975))
        lo.append(a); hi.append(b); pooled.append(d); per_seed.append([a, b])
    r = dict(mean=float(v.mean()), lo=min(lo), hi=max(hi), clusters=int(len(v)), per_seed=per_seed)
    if with_p:
        p = np.concatenate(pooled)
        r['p_two_sided'] = float(min(1.0, max(2 * min((p <= 0).mean(), (p >= 0).mean()), 1.0 / len(p))))
    return r


class Clusters:
    """Scene -> cluster maps at the two pre-declared levels; macro means are means of cluster means."""

    def __init__(self, scenes, bank):
        self.scenes = list(scenes)
        self.levels = {}
        for name, keyf in (('family', lambda s: s['family']), ('template', lambda s: f"{s['domain']}/T{s['template']}")):
            keys = sorted({keyf(bank[s]) for s in self.scenes})
            pos = {k: i for i, k in enumerate(keys)}
            gid = np.array([pos[keyf(bank[s])] for s in self.scenes])
            self.levels[name] = (gid, np.bincount(gid, minlength=len(keys)).astype(float), keys)

    def counts(self):
        return {k: len(v[2]) for k, v in self.levels.items()}

    def cmeans(self, v, level):
        gid, cnt, _ = self.levels[level]
        return np.bincount(gid, weights=np.asarray(v, dtype=float), minlength=len(cnt)) / cnt

    def summary(self, v):
        return {lv: boot(self.cmeans(v, lv)) for lv in self.levels}

    def contrast(self, diff):
        c = {lv: boot(self.cmeans(diff, lv), with_p=True) for lv in self.levels}
        c['estimate'] = c['family']['mean']
        c['lo'] = min(c['family']['lo'], c['template']['lo'])
        c['hi'] = max(c['family']['hi'], c['template']['hi'])
        c['p_max'] = max(c['family']['p_two_sided'], c['template']['p_two_sided'])
        c['scenes'] = len(self.scenes)
        return c


def holm(pvals, alpha=ALPHA):
    order = sorted(pvals, key=lambda k: (pvals[k], k))
    m, out, blocked, running = len(order), {}, False, 0.0
    for i, k in enumerate(order):
        thr = alpha / (m - i)
        rej = (pvals[k] <= thr) and not blocked
        if not rej:
            blocked = True
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = dict(p=pvals[k], rank=i + 1, m=m, threshold=thr, reject=bool(rej), p_holm_adjusted=running)
    return out


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r, i = [0.0] * len(x), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = np.array(ranks(x)), np.array(ranks(y))
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------------------------------------------ cell audit

class Errors:
    def __init__(self):
        self.n, self.examples = 0, []

    def __call__(self, msg):
        self.n += 1
        if len(self.examples) < MAX_EXAMPLES:
            self.examples.append(msg)

    def __bool__(self):
        return self.n > 0

    def dump(self):
        return dict(count=self.n, examples=self.examples)


def read_receipt(raw_path, inputs):
    """Read the receipt BEFORE its raw file: the runner writes the receipt after the last row, so a receipt seen first
    guarantees the raw bytes read next are final (a shard finishing mid-analysis then reads as INCOMPLETE, not invalid)."""
    path = raw_path.with_suffix('.done.json')
    if not path.exists():
        return None
    return json.loads(inputs.read(path, 'receipt')[0])


def check_receipt(r, raw_sha, n_lines, manifest_sha, phase, receiver, shard):
    if r is None:
        return None, ['receipt missing']
    bad = []
    for field, want in (('status', 'PASS'), ('phase', phase), ('receiver', receiver), ('shard', shard),
                        ('raw_sha256', raw_sha), ('rows', n_lines), ('manifest_sha256', manifest_sha)):
        if r.get(field) != want:
            bad.append(f'receipt {field}={r.get(field)!r} expected {want!r}')
    return r, bad


def gate_of(receiver):
    return 'adr' if receiver in ADR_RECEIVERS else 'lock'


def engineering_audit(cell, m, inputs):
    """Row-by-row recomputation of the engineering gate (receipts are not trusted)."""
    rec = m['receiver']
    path = cell['path'] / 'raw' / f'engineering_{rec}_s0.jsonl'
    arms = m['arms']
    per_scene = 4 * sum(len(v) for v in arms.values())
    expected_rows = len(m['engineering_scene_ids']) * per_scene
    gate = gate_of(rec)
    out = dict(file=rel(path), gate=('ADR-latest0924-moe-gate (parsed equal and alias gap <= 0.1)' if gate == 'adr'
                                     else 'lock/evaluator (max logit <= 0.05, alias gap <= 0.1, parsed equal)'),
               expected_rows=expected_rows)
    if not path.exists():
        out.update(verdict='NOT_STARTED', rows=0)
        return out
    receipt0 = read_receipt(path, inputs)
    data, sha = inputs.read(path, 'raw engineering rows')
    lines, partial = split_lines(data)
    out.update(rows=len(lines), partial_trailing_line=partial, raw_sha256=sha)
    receipt, rbad = check_receipt(receipt0, sha, len(lines), cell['manifest_sha256'], 'engineering', rec, 0)
    out['receipt'] = 'missing' if receipt is None else ('valid' if not rbad else 'INVALID: ' + '; '.join(rbad))
    eng_ids = set(m['engineering_scene_ids'])
    rows, err = {}, Errors()
    fail_lock, fail_literal, fail_adr, stored_disagree = [], [], [], []
    ml_max, ag_max, parsed_unequal = (-1.0, None), (-1.0, None), 0
    for ln in lines:
        r = json.loads(ln)
        k = (r['scene'], r['world'], r['question'], r['source'], r['arm'])
        if r.get('key') != '|'.join(map(str, k)) or r.get('phase') != 'engineering':
            err(f'inconsistent key/phase {k}')
        if r['scene'] not in eng_ids:
            err(f'non-engineering scene in the engineering phase {k}')
        if k in rows:
            err(f'duplicate row {k}')
        rows[k] = dict(tokens=r['tokens'], logits=r['logits_sha256'], report=r.get('patch_report') or {})
        ec = r.get('engineering_checks')
        if not ec:
            err(f'missing engineering_checks {k}'); continue
        ml, ag, pe = float(ec['max_logit_difference']), float(ec['alias_gap_difference']), ec['parsed_equal'] is True
        tag = f'{k[0][:8]}|w{k[1]}|q{k[2]}|{k[3]}|{k[4]}'
        if ml > ml_max[0]:
            ml_max = (ml, tag)
        if ag > ag_max[0]:
            ag_max = (ag, tag)
        parsed_unequal += int(not pe)
        p_lock = ml <= GATE_MAX_LOGIT and ag <= GATE_ALIAS_GAP and pe
        p_literal = ml <= GATE_MAX_LOGIT and pe
        p_adr = pe and ag <= GATE_ALIAS_GAP
        detail = dict(key=k, max_logit_difference=ml, alias_gap_difference=ag, parsed_equal=pe, stored_pass=ec.get('pass'))
        if not p_lock:
            fail_lock.append(detail)
        if not p_literal:
            fail_literal.append(detail)
        if not p_adr:
            fail_adr.append(detail)
        applicable = p_adr if gate == 'adr' else p_lock
        if bool(ec.get('pass')) != applicable:
            stored_disagree.append(detail)
    # identity inertness and no-message invariance
    ident_bad, none_bad, ident_n = [], [], 0
    for (sid, w, q, src, arm), r in rows.items():
        if arm == 'identity_write':
            ident_n += 1
            ref = rows.get((sid, w, q, src, 'none'))
            if ref is None or (r['tokens'], r['logits']) != (ref['tokens'], ref['logits']):
                ident_bad.append((sid, w, q, src))
            elif not r['report'] or any(v['norm'] > 1e-6 for v in r['report'].values()):
                ident_bad.append((sid, w, q, src, 'nonzero edit'))
        if arm == 'none':
            o = rows.get((sid, 1 - w, q, src, 'none'))
            if o is not None and (r['tokens'], r['logits']) != (o['tokens'], o['logits']):
                none_bad.append((sid, w, q, src))
    expected_keys = {(s, w, q, src, a) for s in eng_ids for w in (0, 1) for q in (0, 1) for src, al in arms.items() for a in al}
    missing = len(expected_keys - set(rows))
    extra = len(set(rows) - expected_keys)
    failing = fail_adr if gate == 'adr' else fail_lock
    out.update(max_logit_difference=dict(value=ml_max[0], at=ml_max[1]), max_alias_gap_difference=dict(value=ag_max[0], at=ag_max[1]),
               parsed_unequal_rows=parsed_unequal, rows_failing_applicable_gate=len(failing),
               rows_failing_lock_evaluator_gate=len(fail_lock), rows_failing_lock_literal_gate=len(fail_literal),
               rows_failing_adr_gate=len(fail_adr), failing_rows=[dict(d, key='|'.join(map(str, d['key']))) for d in failing[:MAX_EXAMPLES]],
               stored_pass_disagrees_with_recomputation=len(stored_disagree),
               identity_rows=ident_n, identity_not_inert=len(ident_bad), identity_examples=[list(x) for x in ident_bad[:5]],
               none_depends_on_world=len(none_bad), missing_keys=missing, extra_keys=extra, row_errors=err.dump())
    if failing or ident_bad or none_bad:
        out['verdict'] = 'FAIL'
    elif err or extra:
        out['verdict'] = 'AUDIT_FAIL'
    elif receipt is None or missing or partial:
        out['verdict'] = 'INCOMPLETE'
    elif rbad:
        out['verdict'] = 'RECEIPT_INVALID'
    else:
        out['verdict'] = 'PASS'
    return out


def scan_dev_files(cell, m, inputs):
    """Hash and count every development shard; validate receipts; decide completeness (no parsing)."""
    rec = m['receiver']
    files = sorted(cell['path'].glob(f'raw/development_{rec}_s*.jsonl'),
                   key=lambda p: int(re.search(r'_s(\d+)\.jsonl$', p.name).group(1)))
    per_scene = 4 * sum(len(v) for v in m['arms'].values())
    expected_rows = cell['n_dev_scenes'] * per_scene
    shards, data_by_file, problems = [], {}, []
    declared = set()
    for f in files:
        idx = int(re.search(r'_s(\d+)\.jsonl$', f.name).group(1))
        receipt0 = read_receipt(f, inputs)
        data, sha = inputs.read(f, 'raw development rows')
        lines, partial = split_lines(data)
        receipt, rbad = check_receipt(receipt0, sha, len(lines), cell['manifest_sha256'], 'development', rec, idx)
        if receipt is not None:
            declared.add(receipt.get('shards'))
        shards.append(dict(shard=idx, file=rel(f), rows=len(lines), partial_trailing_line=partial, raw_sha256=sha,
                           receipt=('missing' if receipt is None else 'valid' if not rbad else 'INVALID: ' + '; '.join(rbad)),
                           receipt_scenes=(receipt or {}).get('scenes'), receipt_shards=(receipt or {}).get('shards')))
        data_by_file[f] = lines
        if receipt is not None and rbad:
            problems.append(f'{f.name}: ' + '; '.join(rbad))
    rows_present = sum(s['rows'] for s in shards)
    info = dict(shards=shards, rows_present=rows_present, expected_rows=expected_rows, per_scene_rows=per_scene,
                declared_shards=sorted(x for x in declared if x is not None))
    if not files:
        info['state'] = 'NOT_STARTED'
    elif problems:
        info['state'] = 'RECEIPT_INVALID'; info['problems'] = problems
    elif any(s['receipt'] == 'missing' for s in shards):
        info['state'] = 'INCOMPLETE'
    elif len(declared) != 1 or not isinstance(next(iter(declared)), int) or \
            sorted(s['shard'] for s in shards) != list(range(next(iter(declared)))):
        info['state'] = 'INCOMPLETE'; info['problems'] = ['receipted shards do not cover 0..S-1']
    elif sum(s['receipt_scenes'] or 0 for s in shards) != cell['n_dev_scenes'] or rows_present != expected_rows:
        info['state'] = 'INCOMPLETE'; info['problems'] = [f'rows {rows_present} != expected {expected_rows}']
    else:
        info['state'] = 'RECEIPTED'
    return info, data_by_file


def parse_dev(cell, m, data_by_file, bank, dev_ids, packets):
    """Full integrity audit of a receipted cell; returns per-(source, arm) parsed answers or errors."""
    err = Errors()
    arms = m['arms']
    ans = defaultdict(dict)                     # (src, arm) -> (scene, w, q) -> (a64, a256, gold, gold_supplied)
    none_ref, ident, patch, outputs = {}, [], {}, {}
    seen = set()
    sel_layers, orig_layers = m['selected_layers'], m['original_layers']
    for f, lines in data_by_file.items():
        for ln in lines:
            r = json.loads(ln)
            sid, w, q, src, arm = r['scene'], r['world'], r['question'], r['source'], r['arm']
            k = (sid, w, q, src, arm)
            if r.get('key') != '|'.join(map(str, k)) or r.get('phase') != 'development':
                err(f'inconsistent key/phase {k}')
            if k in seen:
                err(f'duplicate row {k}'); continue
            seen.add(k)
            if sid not in dev_ids:
                err(f'scene outside the development set {k}'); continue
            if src not in arms or arm not in arms[src]:
                err(f'source/arm not in the manifest {k}'); continue
            if r.get('receiver') != m['receiver']:
                err(f'receiver field {r.get("receiver")} {k}')
            sc = bank[sid]
            if (r['family'], r['domain'], r['template']) != (sc['family'], sc['domain'], sc['template']):
                err(f'family/domain/template differ from the bank {k}')
            if r['aliases'] != sc['aliases']:
                err(f'aliases differ from the bank {k}')
            field = sc['changed_field'] if q == 0 else next(x for x in sc['fields'] if x != sc['changed_field'])
            owners = [i for i, o in enumerate(sc['worlds'][w]['facts']) if o[field] == sc['queried_values'][q]]
            if not (len(owners) == 1 and r['gold'] == sc['aliases'][owners[0]] == sc['worlds'][w]['answers'][q]):
                err(f'gold does not re-derive {k}')
            sw = (1 - w) if arm == 'other_world' else w
            if r['patch_source_world'] != sw or r['gold_supplied'] != sc['worlds'][sw]['answers'][q]:
                err(f'supplied world/gold wrong {k}')
            if packets[src] is None:
                for o in (0, 1):
                    if {e['word'] for e in r['entries'] if e['object'] == o} != set(sc['worlds'][sw]['facts'][o].values()):
                        err(f'complete-fact entries are not the bank facts {k}')
            else:
                own, cond = packets[src].get(f'{sid}|{w}'), packets[src].get(f'{sid}|{sw}')
                if own is None or cond is None or r['source_packet_sha256'] != own['sha256'] or r['condition_packet_hex'] != cond['hex']:
                    err(f'packet differs from the frozen packets file {k}')
            if arm == 'full_text' and r.get('extra_text') != ' '.join(sc['worlds'][w]['evidence']):
                err(f'full_text is not the world evidence {k}')
            if arm == 'packet_text' and not r.get('extra_text'):
                err(f'packet_text row has no text {k}')
            if r['layers'] != (orig_layers if arm == 'original' else sel_layers):
                err(f'written layers wrong {k}')
            if r['prefix_cache_sha256'] != r['branch_initial_cache_sha256']:
                err(f'cache fingerprint mismatch {k}')
            parsed = {}
            for b in BUDGETS:
                a = parse_answer(r[f'text{b}'], r['aliases'])
                parsed[b] = a
                for suffix, gold in (('', r['gold']), ('_supplied', r['gold_supplied'])):
                    want = dict(answered=a is not None, correct=a is not None and a == gold, answer=a)
                    if r.get(f'score{b}{suffix}') != want:
                        err(f'stored score{b}{suffix} disagrees with the re-parse {k}')
            ans[(src, arm)][(sid, w, q)] = (parsed[64], parsed[256], r['gold'], r['gold_supplied'])
            if arm in ('selected', 'other_world') and packets[src] is not None:
                outputs[(sid, w, q, arm)] = (r['tokens'], r['logits_sha256'])
            if arm == 'none':
                none_ref[(sid, w, q, src)] = (r['tokens'], r['logits_sha256'])
            elif arm == 'identity_write':
                ident.append((sid, w, q, src, r['tokens'], r['logits_sha256'], r.get('patch_report') or {}))
            if arm == 'selected' or arm in SHAMS:
                patch[k] = (r['patch_positions'], {li: (v['norm'], v.get('target_sha256')) for li, v in (r.get('patch_report') or {}).items()},
                            any(e.get('weight') for e in r['entries']))
    expected = {(s, w, q, src, a) for s in dev_ids for w in (0, 1) for q in (0, 1) for src, al in arms.items() for a in al}
    if seen != expected:
        err(f'keys differ from the expected set: missing {len(expected - seen)}, extra {len(seen - expected)}')
    checks = dict(identity_rows=len(ident), identity_not_inert=0, none_depends_on_world=0, reassignment_norm_mismatch=0,
                  reassignment_target_identical=0)
    # consistency (reported, not an exclusion rule): other_world in world w delivers the packet that `selected` delivers
    # in world 1-w through the same neutral prompt, so the two outputs should be identical
    pairs = [(v, outputs.get((sid, 1 - w, q, 'selected'))) for (sid, w, q, arm), v in outputs.items() if arm == 'other_world']
    checks['other_world_rows'] = len(pairs)
    checks['other_world_identical_to_selected_of_other_world'] = sum(a == b for a, b in pairs)
    for sid, w, q, src, toks, lg, rep in ident:
        ref = none_ref.get((sid, w, q, src))
        if ref is None or (toks, lg) != ref or not rep or any(v['norm'] > 1e-6 for v in rep.values()):
            checks['identity_not_inert'] += 1
            err(f'identity write not inert {(sid, w, q, src)}')
    for (sid, w, q, src), v in none_ref.items():
        o = none_ref.get((sid, 1 - w, q, src))
        if o is not None and o != v:
            checks['none_depends_on_world'] += 1
            err(f'no-message row depends on the world {(sid, w, q, src)}')
    for (sid, w, q, src, arm), (pos, rep, weighted) in patch.items():
        if arm not in SHAMS:
            continue
        ref = patch.get((sid, w, q, src, 'selected'))
        if ref is None or pos != ref[0] or set(rep) != set(ref[1]):
            checks['reassignment_norm_mismatch'] += 1
            err(f'reassignment positions/layers differ from the selected write {(sid, w, q, src, arm)}'); continue
        if any(abs(rep[li][0] - ref[1][li][0]) > max(1e-4, 1e-4 * ref[1][li][0]) for li in rep):
            checks['reassignment_norm_mismatch'] += 1
            err(f'reassignment norm not matched {(sid, w, q, src, arm)}')
        if weighted and not any(rep[li][1] != ref[1][li][1] for li in rep):
            checks['reassignment_target_identical'] += 1
            err(f'reassignment target identical to the selected write {(sid, w, q, src, arm)}')
    return ans, checks, err


def scene_metrics(rows, scenes):
    """Per-scene metrics; order of the four answers: (w0,q0), (w0,q1), (w1,q0), (w1,q1); q0 = changed field."""
    out = {k: np.zeros(len(scenes)) for k in METRICS + SUPPLIED_METRICS}
    for i, s in enumerate(scenes):
        rr = [rows[(s, w, q)] for w in (0, 1) for q in (0, 1)]
        for b, sfx in ((64, ''), (256, '256')):
            a = [x[0] if b == 64 else x[1] for x in rr]
            ok = [ai is not None and ai == x[2] for ai, x in zip(a, rr)]
            oks = [ai is not None and ai == x[3] for ai, x in zip(a, rr)]
            out['J' + sfx][i] = float(all(ok))
            out['accuracy' + sfx][i] = sum(ok) / 4
            out['coverage' + sfx][i] = sum(ai is not None for ai in a) / 4
            out['J' + sfx + '_supplied'][i] = float(all(oks))
            if b == 64:
                out['R'][i] = float(ok[0] and ok[2])
                out['U'][i] = float(ok[1] and ok[3])
                out['accuracy_supplied'][i] = sum(oks) / 4
    return out


def estimate_cell(ans, scenes, cl, sender, has_literal):
    per = {f'{src}/{arm}': scene_metrics(rows, scenes) for (src, arm), rows in sorted(ans.items())}
    est = {}
    for key, mets in per.items():
        names = METRICS + (SUPPLIED_METRICS if key.endswith('/other_world') else ())
        est[key] = {n: cl.summary(mets[n]) for n in names}
        est[key]['J_sum'] = int(mets['J'].sum())
        est[key]['scenes'] = len(scenes)
    J = {k: v['J'] for k, v in per.items()}
    S = sender
    sham_mean = (J[f'{S}/sham0'] + J[f'{S}/sham1'] + J[f'{S}/sham2']) / 3
    C = {'selected_minus_mean_reassignment': cl.contrast(J[f'{S}/selected'] - sham_mean),
         'selected_minus_packet_text': cl.contrast(J[f'{S}/selected'] - J[f'{S}/packet_text']),
         'selected_minus_original': cl.contrast(J[f'{S}/selected'] - J[f'{S}/original']),
         'selected_minus_none': cl.contrast(J[f'{S}/selected'] - J[f'{S}/none']),
         'selected_minus_other_world': cl.contrast(J[f'{S}/selected'] - J[f'{S}/other_world']),
         'selected_minus_other_world_supplied_gold': cl.contrast(J[f'{S}/selected'] - per[f'{S}/other_world']['J_supplied'])}
    for d in SHAMS:
        C[f'selected_minus_{d}'] = cl.contrast(J[f'{S}/selected'] - J[f'{S}/{d}'])
    if has_literal:
        C['literal_selected_minus_packet_text'] = cl.contrast(J['literal/selected'] - J['literal/packet_text'])
        C['literal_selected_minus_native'] = cl.contrast(J['literal/selected'] - J['literal/native'])
        C['literal_selected_minus_full_text'] = cl.contrast(J['literal/selected'] - J['literal/full_text'])
        C['sender_selected_minus_literal_selected'] = cl.contrast(J[f'{S}/selected'] - J['literal/selected'])
    gap = J[f'{S}/selected'] - J[f'{S}/packet_text']
    return est, C, gap


# ------------------------------------------------------------------------------------------------ driver

def load_cell_manifest(sender, rec, cdir, same_pairs, lock_path, lock_sha, inputs):
    cell = dict(sender=sender, receiver=rec, path=cdir, dir=rel(cdir), same_model_pair=(sender, rec) in same_pairs,
                warnings=[], errors=[])
    mp = cdir / 'manifest.json'
    if not mp.exists():
        cell['status'] = 'NOT_BUILT'
        return cell, None
    data, sha = inputs.read(mp, 'manifest')
    m = json.loads(data)
    cell['manifest_sha256'] = sha
    if m.get('sender') != sender or m.get('receiver') != rec:
        cell['errors'].append(f"manifest sender/receiver {m.get('sender')}/{m.get('receiver')} do not match the directory")
    if bool(m.get('same_model_pair')) != cell['same_model_pair']:
        cell['errors'].append(f"manifest same_model_pair={m.get('same_model_pair')} disagrees with the lock's same_model_pairs")
    if m.get('arms', {}).get(sender) != SENDER_ARMS:
        cell['errors'].append(f"sender arms {m.get('arms', {}).get(sender)} differ from the lock's per-sender conditions")
    extra_src = set(m.get('arms', {})) - {sender, 'literal'}
    if extra_src:
        cell['errors'].append(f'unexpected sources {sorted(extra_src)}')
    if 'literal' in m.get('arms', {}) and m['arms']['literal'] != LITERAL_ARMS:
        cell['errors'].append(f"literal arms {m['arms']['literal']} differ from the lock's complete-fact conditions")
    lock_pins = [w for p, w in m.get('pins', {}).items() if Path(p).resolve() == Path(lock_path).resolve()]
    if lock_pins != [lock_sha]:
        cell['errors'].append(f'manifest does not pin the current lock sha256 ({lock_pins})')
    for p, want in m.get('pins', {}).items():
        if not Path(p).exists():
            cell['errors'].append(f'pinned input missing: {p}')
        elif inputs.sha_cached(p, 'pinned input') != want:
            cell['errors'].append(f'pinned input changed: {rel(p)}')
    return cell, m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    ap.add_argument('--expa-root', type=Path, default=DEFAULT_EXPA)
    ap.add_argument('--adr-glob', default=DEFAULT_ADR_GLOB)
    ap.add_argument('--out-dir', type=Path, default=None, help='default: <expa-root>/analysis')
    a = ap.parse_args()
    out_dir = a.out_dir or (a.expa_root / 'analysis')
    inputs = Inputs()
    lock_bytes, lock_sha = inputs.read(a.lock, 'lock')
    lock = json.loads(lock_bytes)
    same_pairs = set(verify_lock(lock))
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    adr_files = sorted(glob.glob(a.adr_glob))
    for f in adr_files:
        inputs.read(f, 'ADR')
    senders = sorted(lock['senders'], key=lambda s: lock['senders'][s]['priority'])
    receivers = list(lock['receivers'])

    # ---- pass A: manifests, sender gates, engineering
    cells, manifests, sender_gate = {}, {}, {}
    for s in senders:
        for r in receivers:
            cell, m = load_cell_manifest(s, r, a.expa_root / s / r, same_pairs, a.lock, lock_sha, inputs)
            cells[(s, r)] = cell
            manifests[(s, r)] = m
            if m is None:
                continue
            src_path = m['sources'].get(s)
            if s not in sender_gate:
                gp = Path(src_path).parent / f'{s}_gate.json' if src_path else None
                if gp is None or not gp.exists():
                    sender_gate[s] = dict(status='MISSING', file=rel(gp) if gp else None)
                else:
                    g = json.loads(inputs.read(gp, 'sender gate')[0])
                    sender_gate[s] = dict(status=g.get('verdict', {}).get('status'), file=rel(gp),
                                          checks=g.get('verdict', {}).get('checks'))
            marker = cell['path'] / 'raw' / 'ENGINEERING_FAILED'
            note = cell['path'] / 'ENGINEERING_FAILURE_NOTE.txt'
            cell['engineering_failed_marker'] = marker.exists()
            if marker.exists():
                inputs.read(marker, 'engineering failure marker')
            if note.exists():
                cell['engineering_failure_note'] = inputs.read(note, 'engineering failure note')[0].decode().strip()
            bank_path = Path(m['bank'])
            cell['n_dev_scenes'] = m['bank_size'] - len(m['engineering_scene_ids'])
            cell['bank'] = rel(bank_path)
            cell['engineering'] = engineering_audit(cell, m, inputs)
    ignored = sorted(rel(p) for p in a.expa_root.glob('*/*') if p.is_dir() and
                     (p.parent.name not in senders or p.name not in receivers))
    ignored += sorted(rel(p) for p in a.expa_root.iterdir() if p.is_dir() and p.name not in senders)

    # receiver-level engineering failures (any sender's cell)
    eng_fail = defaultdict(list)
    for (s, r), cell in cells.items():
        e = cell.get('engineering', {})
        if e.get('verdict') == 'FAIL':
            eng_fail[r].append(dict(cell=f'{s} -> {r}', reason='recomputed gate/identity failure on engineering rows'))
        elif cell.get('engineering_failed_marker') and cell.get('engineering_failure_note'):
            eng_fail[r].append(dict(cell=f'{s} -> {r}', reason='ENGINEERING_FAILED marker with failure note'))

    # ---- pass B: development
    banks, packets_cache, gaps, dev_scene_order = {}, {}, {}, None
    for (s, r), cell in cells.items():
        m = manifests[(s, r)]
        if m is None:
            continue
        info, data_by_file = scan_dev_files(cell, m, inputs)
        cell['development'] = info
        if r in eng_fail:
            cell['status'] = 'ENGINEERING_FAILURE'
            if info['rows_present']:
                cell['warnings'].append(f"{info['rows_present']} development rows exist and are ignored by rule")
        elif cell['errors']:
            cell['status'] = 'AUDIT_FAIL'
        elif sender_gate.get(s, {}).get('status') != 'PASS':
            cell['status'] = 'SENDER_GATE_NOT_PASS'
        elif cell.get('engineering_failed_marker'):
            cell['status'] = 'ENGINEERING_MARKER_UNEXPLAINED'
        elif cell['engineering']['verdict'] != 'PASS':
            cell['status'] = {'NOT_STARTED': 'NOT_STARTED', 'INCOMPLETE': 'ENGINEERING_INCOMPLETE'}.get(
                cell['engineering']['verdict'], 'AUDIT_FAIL')
            if cell['status'] == 'AUDIT_FAIL':
                cell['errors'].append(f"engineering verdict {cell['engineering']['verdict']}")
        elif info['state'] == 'NOT_STARTED':
            cell['status'] = 'NOT_STARTED'
        elif info['state'] == 'INCOMPLETE':
            cell['status'] = 'INCOMPLETE'
        elif info['state'] == 'RECEIPT_INVALID':
            cell['status'] = 'AUDIT_FAIL'; cell['errors'] += info['problems']
        else:
            cell['status'] = 'PARSING'
        if cell['status'] != 'PARSING':
            continue
        bpath = str(Path(m['bank']).resolve())
        if bpath not in banks:
            bl = json.loads(inputs.read(m['bank'], 'bank')[0])
            banks[bpath] = {sc['scene']: sc for sc in bl}
            if len(bl) != m['bank_size']:
                cell['errors'].append('bank size differs from the manifest')
        bank = banks[bpath]
        eng_ids = set(m['engineering_scene_ids'])
        dev_ids = sorted(x for x in bank if x not in eng_ids)
        if dev_scene_order is None:
            dev_scene_order = dev_ids
        packets = {}
        for src, pth in m['sources'].items():
            if pth is None:
                packets[src] = None
                continue
            key = str(Path(pth).resolve())
            if key not in packets_cache:
                packets_cache[key] = json.loads(inputs.read(pth, 'sender packets')[0])['packets']
            packets[src] = packets_cache[key]
        ans, checks, err = parse_dev(cell, m, data_by_file, bank, set(dev_ids), packets)
        cell['development']['integrity'] = dict(checks, row_errors=err.dump())
        if err or cell['errors']:
            cell['status'] = 'AUDIT_FAIL'
            continue
        cl = Clusters(dev_ids, bank)
        counts = cl.counts()
        if counts != {'family': 36, 'template': 12}:
            cell['warnings'].append(f'cluster counts {counts} differ from the lock (36 families, 12 parent templates)')
        est, C, gap = estimate_cell(ans, dev_ids, cl, s, 'literal' in m['arms'])
        cell.update(status='COMPLETE', estimates=est, contrasts=C, cluster_counts=counts, scenes=len(dev_ids))
        if est[f'{s}/none']['J_sum'] != 0:
            cell['warnings'].append(f"no-message J_sum = {est[f'{s}/none']['J_sum']} (structurally 0 expected)")
        if checks['other_world_identical_to_selected_of_other_world'] != checks['other_world_rows']:
            cell['warnings'].append(f"other_world consistency: {checks['other_world_identical_to_selected_of_other_world']}/"
                                    f"{checks['other_world_rows']} rows identical to selected of the other world")
        gaps[(s, r)] = (dev_ids, gap, cl)

    # ---- hypotheses
    def cross(s):
        return [r for r in receivers if (s, r) not in same_pairs]

    def evaluated(s):
        return [r for r in cross(s) if cells[(s, r)].get('status') == 'COMPLETE']

    H1, H2, H4, SAME = {}, {}, {}, {}
    for s in senders:
        ev, planned = evaluated(s), cross(s)
        p1 = {r: cells[(s, r)]['contrasts']['selected_minus_mean_reassignment']['p_max'] for r in ev}
        p2 = {r: cells[(s, r)]['contrasts']['selected_minus_packet_text']['p_max'] for r in ev}
        h1, h2 = holm(p1), holm(p2)
        h1_all = holm({r: p1.get(r, 1.0) for r in planned})
        h2_all = holm({r: p2.get(r, 1.0) for r in planned})
        rows1, rows2, rows4 = {}, {}, {}
        for r in planned:
            c = cells[(s, r)]
            if c.get('status') != 'COMPLETE':
                rows1[r] = rows2[r] = dict(verdict=f"NOT EVALUABLE ({describe_status(c)})")
                continue
            e = c['estimates'][f'{s}/selected']
            con = c['contrasts']['selected_minus_mean_reassignment']
            crit = dict(J_selected_ge_0_50=e['J']['family']['mean'] >= H1_J,
                        coverage_selected_ge_0_90=e['coverage']['family']['mean'] >= H1_COVERAGE,
                        holm_reject=h1[r]['reject'], lo_family_gt_0=con['family']['lo'] > 0, lo_template_gt_0=con['template']['lo'] > 0)
            ok = all(crit.values())
            ok_all = ok and h1_all[r]['reject']
            rows1[r] = dict(verdict='RECOVERY' if ok else 'NO RECOVERY', criteria=crit, J_selected=e['J']['family'],
                            coverage_selected=e['coverage']['family'], contrast=con, holm=h1[r],
                            failed=[k for k, v in crit.items() if not v],
                            sensitivity_m_all_planned=dict(m=len(planned), holm=h1_all[r],
                                                           verdict='RECOVERY' if ok_all else 'NO RECOVERY'))
            d = c['contrasts']['selected_minus_packet_text']
            rows2[r] = dict(h2_verdict(d, h2[r]), contrast=d, holm=h2[r],
                            sensitivity_m_all_planned=dict(m=len(planned), holm=h2_all[r], verdict=h2_verdict(d, h2_all[r])['verdict']))
        for r in receivers:
            c = cells[(s, r)]
            if c.get('status') == 'COMPLETE':
                rows4[r] = dict(same_model_pair=(s, r) in same_pairs, contrast=c['contrasts']['selected_minus_original'],
                                J_selected=c['estimates'][f'{s}/selected']['J']['family'],
                                J_original=c['estimates'][f'{s}/original']['J']['family'])
            else:
                rows4[r] = dict(same_model_pair=(s, r) in same_pairs, verdict=f"NOT EVALUABLE ({describe_status(c)})")
        passing = [r for r in ev if rows1[r]['verdict'] == 'RECOVERY']
        H1[s] = dict(m_evaluated=len(ev), planned_cross_model=len(planned), receivers_passing=passing, count_passing=len(passing),
                     evaluated=ev, not_evaluable={r: describe_status(cells[(s, r)]) for r in planned if r not in ev},
                     sensitivity_disagreements=[r for r in ev if rows1[r]['sensitivity_m_all_planned']['verdict'] != rows1[r]['verdict']],
                     per_receiver=rows1)
        tally = defaultdict(list)
        for r in ev:
            tally[rows2[r]['verdict']].append(r)
        H2[s] = dict(m_evaluated=len(ev), tally=dict(tally), per_receiver=rows2,
                     sensitivity_disagreements=[r for r in ev if rows2[r]['sensitivity_m_all_planned']['verdict'] != rows2[r]['verdict']])
        H4[s] = rows4
    for (s, r) in sorted(same_pairs):
        if (s, r) not in cells:
            SAME[f'{s} -> {r}'] = dict(status='NOT IN LOCK GRID')
            continue
        c = cells[(s, r)]
        if c.get('status') != 'COMPLETE':
            SAME[f'{s} -> {r}'] = dict(status=describe_status(c))
            continue
        e = c['estimates']
        con1 = c['contrasts']['selected_minus_mean_reassignment']
        d = c['contrasts']['selected_minus_packet_text']
        unadj = dict(p=d['p_max'], m=1, threshold=ALPHA, reject=d['p_max'] <= ALPHA)
        crit = dict(J_selected_ge_0_50=e[f'{s}/selected']['J']['family']['mean'] >= H1_J,
                    coverage_selected_ge_0_90=e[f'{s}/selected']['coverage']['family']['mean'] >= H1_COVERAGE,
                    unadjusted_p_le_0_05=con1['p_max'] <= ALPHA, lo_family_gt_0=con1['family']['lo'] > 0,
                    lo_template_gt_0=con1['template']['lo'] > 0)
        SAME[f'{s} -> {r}'] = dict(status='COMPLETE', note='same-model pair: reported separately, excluded from cross-model claims and counts; no Holm',
                                   h1_criteria_unadjusted=crit, h1_like='RECOVERY (same-model)' if all(crit.values()) else 'NO RECOVERY (same-model)',
                                   h2_like_unadjusted=h2_verdict(d, unadj)['verdict'], selected_minus_mean_reassignment=con1,
                                   selected_minus_packet_text=d)
    # H3
    H3 = dict(primary={}, secondary_descriptive={}, gap27_minus_gap2_descriptive={})
    for s in senders:
        big, small = H3_PRIMARY
        if (s, big) in gaps and (s, small) in gaps and (s, big) not in same_pairs:
            H3['primary'][s] = ladder_contrast(gaps, s, big, small)
            c = H3['primary'][s]
            H3['primary'][s]['verdict'] = ('the gap grows with scale' if c['family']['lo'] > 0 and c['template']['lo'] > 0
                                           else 'not shown')
        else:
            why = describe_status(cells[(s, big)]) if cells[(s, big)].get('status') != 'COMPLETE' else describe_status(cells[(s, small)])
            reason = (f'{big} engineering failure, {ADR_NAME}' if cells[(s, big)].get('status') == 'ENGINEERING_FAILURE' else why)
            H3['primary'][s] = dict(verdict=f'NOT EVALUABLE ({reason})', substitute='none (no other model is substituted)')
        pts, missing = [], {}
        for r, n in LADDER.items():
            if (s, r) in same_pairs:
                missing[r] = 'same-model pair (excluded)'
            elif cells[(s, r)].get('status') == 'COMPLETE':
                g = cells[(s, r)]['contrasts']['selected_minus_packet_text']
                pts.append(dict(receiver=r, params=n, log10_params=math.log10(n), gap=g['estimate'], gap_lo=g['lo'], gap_hi=g['hi']))
            else:
                missing[r] = describe_status(cells[(s, r)])
        rho = spearman([p['log10_params'] for p in pts], [p['gap'] for p in pts]) if len(pts) >= 3 else None
        H3['secondary_descriptive'][s] = dict(label='DESCRIPTIVE ONLY - no significance claim', n_points=len(pts), points=pts,
                                              missing=missing, spearman_rho=rho,
                                              note=None if len(pts) >= 3 else 'fewer than 3 ladder points: not computed')
        big, small = H3_DESCRIPTIVE
        if (s, big) in gaps and (s, small) in gaps:
            H3['gap27_minus_gap2_descriptive'][s] = dict(ladder_contrast(gaps, s, big, small),
                                                         label='DESCRIPTIVE - NOT the preregistered primary (primary = gap(122B) - gap(2B))')
        else:
            H3['gap27_minus_gap2_descriptive'][s] = dict(verdict=f'NOT EVALUABLE ({describe_status(cells[(s, big)])}; '
                                                                 f'{describe_status(cells[(s, small)])})')
    # complete facts (literal) per receiver
    CF = {}
    for (s, r), c in cells.items():
        m = manifests[(s, r)]
        if m is None or 'literal' not in m['arms']:
            continue
        if c.get('status') != 'COMPLETE':
            CF[r] = dict(cell=f'{s} -> {r}', status=describe_status(c))
            continue
        e = c['estimates']
        CF[r] = dict(cell=f'{s} -> {r}', status='COMPLETE', label='DESCRIPTIVE',
                     J={arm: e[f'literal/{arm}']['J'] for arm in LITERAL_ARMS},
                     coverage={arm: e[f'literal/{arm}']['coverage']['family'] for arm in LITERAL_ARMS},
                     contrasts={k: c['contrasts'][k] for k in ('literal_selected_minus_packet_text', 'literal_selected_minus_native',
                                                                 'literal_selected_minus_full_text', 'sender_selected_minus_literal_selected')})
    report = dict(
        analysis='Experiment A (latest0924) frozen analysis',
        generated_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        lock=dict(path=rel(a.lock), sha256=lock_sha, name=lock.get('lock'), status=lock.get('status')),
        script=dict(path=rel(Path(__file__)), sha256=script_sha),
        adr=[rel(f) for f in adr_files],
        constants=dict(seeds=list(SEEDS), draws_per_seed=DRAWS, alpha=ALPHA, similar_margin=MARGIN, h1_J=H1_J,
                       h1_coverage=H1_COVERAGE, budgets=list(BUDGETS), engineering_gate=dict(max_logit=GATE_MAX_LOGIT, alias_gap=GATE_ALIAS_GAP),
                       adr_receivers=list(ADR_RECEIVERS), ladder_params=LADDER),
        resolutions=RESOLUTIONS, not_claims=lock.get('not_claims'), same_model_rule=lock.get('same_model_rule'),
        sender_gates=sender_gate, receiver_engineering_failures=dict(eng_fail),
        cells={f'{s} -> {r}': strip_cell(c) for (s, r), c in cells.items()},
        H1=H1, H2=H2, H3=H3, H4=H4, same_model_pairs=SAME, complete_facts=CF,
        ignored_paths=ignored, inputs=inputs.files)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'expA_results.json').write_text(json.dumps(report, indent=1, sort_keys=True, default=_json) + '\n')
    (out_dir / 'expA_results.md').write_text(markdown(report, senders, receivers, cells, same_pairs))
    print_summary(report, senders, receivers, cells, same_pairs)


def _json(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return rel(x)
    if isinstance(x, tuple):
        return list(x)
    raise TypeError(type(x))


def strip_cell(c):
    return {k: v for k, v in c.items() if k != 'path'}


def ladder_contrast(gaps, s, big, small):
    ids_b, gb, cl = gaps[(s, big)]
    ids_s, gs, _ = gaps[(s, small)]
    assert ids_b == ids_s, 'ladder cells evaluated on different scenes'
    return cl.contrast(gb - gs)


def h2_verdict(d, h):
    within = d['family']['lo'] >= -MARGIN and d['family']['hi'] <= MARGIN and d['template']['lo'] >= -MARGIN and d['template']['hi'] <= MARGIN
    if h['reject'] and d['family']['lo'] > 0 and d['template']['lo'] > 0:
        v = 'internal better'
    elif h['reject'] and d['family']['hi'] < 0 and d['template']['hi'] < 0:
        v = 'text better'
    elif within:
        v = 'similar'
    else:
        v = 'unresolved'
    return dict(verdict=v, within_margin=within)


def describe_status(c):
    st = c.get('status', 'NOT_BUILT')
    if st == 'ENGINEERING_FAILURE':
        return 'ENGINEERING FAILURE' + (f' ({ADR_NAME})' if c['receiver'] in ADR_RECEIVERS else '')
    if st == 'INCOMPLETE':
        d = c['development']
        return f"INCOMPLETE - NOT RUN per lock ({d['rows_present']}/{d['expected_rows']} rows)"
    if st == 'NOT_STARTED':
        return 'NOT RUN (no development rows)'
    if st == 'ENGINEERING_INCOMPLETE':
        return 'NOT RUN (engineering phase incomplete)'
    return st


# ------------------------------------------------------------------------------------------------ report

def pct(x):
    return f'{100 * x:.1f}'


def ci(d):
    return f"{pct(d['mean'])} [{pct(d['lo'])}, {pct(d['hi'])}]"


def cdiff(c):
    return f"{100 * c['estimate']:+.1f} [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}]"


def fp(p):
    return f'{p:.2g}'


def markdown(rep, senders, receivers, cells, same_pairs):
    L = ['# Experiment A (latest0924): frozen analysis', '',
         f"Generated {rep['generated_utc']} by `{rep['script']['path']}` (sha256 `{rep['script']['sha256'][:16]}`) under "
         f"`{rep['lock']['path']}` (sha256 `{rep['lock']['sha256'][:16]}`). All numbers are recomputed from the raw rows listed in "
         f"`expA_results.json` → `inputs` (sha256 of every file read).", '',
         f"Point estimates are family-macro means (36 families) at 64 tokens; intervals are 95% percentile cluster-bootstrap "
         f"intervals enclosing all three seeds ({', '.join(map(str, SEEDS))}; {DRAWS:,} draws each). Contrast intervals "
         f"enclose both clustering levels (36 families, 12 parent templates = domain x template); p = the larger two-sided "
         f"bootstrap p of the two levels. All values in percentage points. Cells that are not complete are never estimated.", '']
    # (a) main tables
    for s in senders:
        h1, h2 = rep['H1'][s], rep['H2'][s]
        L += [f'## (a) Sender {s} → cross-model receivers', '',
              f"Sender gate: {rep['sender_gates'].get(s, {}).get('status', 'n/a')}. H1: **{h1['count_passing']} of {h1['m_evaluated']} evaluated** "
              f"cross-model receivers show recovery (Holm m = {h1['m_evaluated']}; {h1['planned_cross_model']} planned). "
              f"H2 tally: " + (', '.join(f'{k}: {len(v)}' for k, v in sorted(h2['tally'].items())) or 'none evaluated') + '.', '',
              '| receiver | status | J(selected) [95%] | J(text) | J(original) | reassigned J range / coverage | coverage(selected) '
              '| selected − mean reassigned [both levels] p, Holm thr | H1 | selected − text [both levels] p | H2 |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
        for r in receivers:
            if (s, r) in same_pairs:
                continue
            c = cells[(s, r)]
            if c.get('status') != 'COMPLETE':
                L.append(f'| {r} | {describe_status(c)} | - | - | - | - | - | - | NOT EVALUABLE | - | NOT EVALUABLE |')
                continue
            e = c['estimates']
            shJ = [e[f'{s}/{d}']['J']['family']['mean'] for d in SHAMS]
            shC = np.mean([e[f'{s}/{d}']['coverage']['family']['mean'] for d in SHAMS])
            r1, r2 = h1['per_receiver'][r], h2['per_receiver'][r]
            c1, c2 = r1['contrast'], r2['contrast']
            L.append(f"| {r} | COMPLETE | {ci(e[f'{s}/selected']['J']['family'])} | {pct(e[f'{s}/packet_text']['J']['family']['mean'])} "
                     f"| {pct(e[f'{s}/original']['J']['family']['mean'])} | {pct(min(shJ))}–{pct(max(shJ))} / {pct(shC)} "
                     f"| {pct(e[f'{s}/selected']['coverage']['family']['mean'])} | {cdiff(c1)} p={fp(c1['p_max'])}, thr {fp(r1['holm']['threshold'])} "
                     f"| {r1['verdict']}{'' if r1['verdict'] == 'RECOVERY' else ' (' + ', '.join(r1['failed']) + ')'} "
                     f"| {cdiff(c2)} p={fp(c2['p_max'])} | {r2['verdict']} |")
        if h1['sensitivity_disagreements'] or h2['sensitivity_disagreements']:
            L += ['', f"Sensitivity (Holm m = all {h1['planned_cross_model']} planned cross-model receivers, unevaluated at p = 1) changes: "
                  f"H1 {h1['sensitivity_disagreements']}, H2 {h2['sensitivity_disagreements']}."]
        else:
            L += ['', f"Sensitivity (Holm m = all {h1['planned_cross_model']} planned cross-model receivers, unevaluated at p = 1): no verdict changes."]
        L.append('')
    # same-model
    L += ['## (a′) Same-model pairs (reported separately; excluded from every cross-model count, no Holm)', '',
          '| pair | status | J(selected) [95%] | J(text) | J(original) | coverage(selected) | selected − mean reassigned | H1 criteria (unadjusted) | selected − text | H2-like (unadjusted) |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for pair, v in rep['same_model_pairs'].items():
        if v.get('status') != 'COMPLETE':
            L.append(f"| {pair} | {v.get('status')} | - | - | - | - | - | - | - | - |")
            continue
        s, r = [x.strip() for x in pair.split('->')]
        e = cells[(s, r)]['estimates']
        L.append(f"| {pair} | COMPLETE | {ci(e[f'{s}/selected']['J']['family'])} | {pct(e[f'{s}/packet_text']['J']['family']['mean'])} "
                 f"| {pct(e[f'{s}/original']['J']['family']['mean'])} | {pct(e[f'{s}/selected']['coverage']['family']['mean'])} "
                 f"| {cdiff(v['selected_minus_mean_reassignment'])} p={fp(v['selected_minus_mean_reassignment']['p_max'])} | {v['h1_like']} "
                 f"| {cdiff(v['selected_minus_packet_text'])} p={fp(v['selected_minus_packet_text']['p_max'])} | {v['h2_like_unadjusted']} |")
    L.append('')
    # (b) engineering
    L += ['## (b) Engineering audit per cell (recomputed from the engineering rows; receipts not trusted)', '',
          '| sender → receiver | gate | rows / expected | receipt | max logit diff (at) | max alias gap | parsed unequal | rows failing applicable gate | stored pass ≠ recomputed | identity not inert | verdict | cell status |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for s in senders:
        for r in receivers:
            c = cells[(s, r)]
            e = c.get('engineering')
            if not e:
                L.append(f'| {s} → {r} | - | - | - | - | - | - | - | - | - | - | {describe_status(c)} |'); continue
            if e['verdict'] == 'NOT_STARTED':
                L.append(f"| {s} → {r} | {'ADR' if gate_of(r) == 'adr' else 'lock'} | 0 / {e['expected_rows']} | - | - | - | - | - | - | - | NOT STARTED | {describe_status(c)} |")
                continue
            L.append(f"| {s} → {r} | {'ADR' if gate_of(r) == 'adr' else 'lock'} | {e['rows']} / {e['expected_rows']} | {e['receipt']} "
                     f"| {e['max_logit_difference']['value']:.4g} ({e['max_logit_difference']['at']}) | {e['max_alias_gap_difference']['value']:.4g} "
                     f"| {e['parsed_unequal_rows']} | {e['rows_failing_applicable_gate']} | {e['stored_pass_disagrees_with_recomputation']} "
                     f"| {e['identity_not_inert']}/{e['identity_rows']} | {e['verdict']} | {describe_status(c)} |")
    L += ['']
    for r, why in rep['receiver_engineering_failures'].items():
        notes = sorted({cells[(s, r)].get('engineering_failure_note') for s in senders if cells[(s, r)].get('engineering_failure_note')})
        L += [f"**{r}: ENGINEERING FAILURE** ({'; '.join(w['cell'] + ': ' + w['reason'] for w in why)}). No outcome rows under any sender.", '']
        for n in notes:
            L += [f'> Note: {n}', '']
        for s in senders:
            e = cells[(s, r)].get('engineering') or {}
            if e.get('rows'):
                L += [f"Recomputed from {e['file']}: {e['rows']} rows; max logit difference {e['max_logit_difference']['value']:.4f} at {e['max_logit_difference']['at']}; "
                      f"max alias-gap difference {e['max_alias_gap_difference']['value']:.4f}; parsed-unequal rows {e['parsed_unequal_rows']}; "
                      f"rows failing: lock/evaluator gate {e['rows_failing_lock_evaluator_gate']}, lock-literal gate (logit + parsed) "
                      f"{e['rows_failing_lock_literal_gate']}, ADR gate {e['rows_failing_adr_gate']}; applicable = {e['gate']}.", '']
    # (c) ladder
    L += ['## (c) Scale ladder (H3)', '']
    for s in senders:
        p = rep['H3']['primary'][s]
        L += [f'### Sender {s}', '', f"**H3 primary (gap(122B) − gap(2B)): {p['verdict']}**" +
              (f" — {cdiff(p)} p={fp(p['p_max'])}" if 'estimate' in p else ''), '']
        sec = rep['H3']['secondary_descriptive'][s]
        L += ['| ladder point | params | status | J(selected) | J(text) | gap = selected − text [both levels] |', '|---|---|---|---|---|---|']
        for r, n in LADDER.items():
            c = cells[(s, r)]
            if c.get('status') == 'COMPLETE' and (s, r) not in same_pairs:
                e = c['estimates']
                L.append(f"| {r} | {n:.0e} | COMPLETE | {pct(e[f'{s}/selected']['J']['family']['mean'])} | {pct(e[f'{s}/packet_text']['J']['family']['mean'])} "
                         f"| {cdiff(c['contrasts']['selected_minus_packet_text'])} |")
            else:
                L.append(f"| {r} | {n:.0e} | {sec['missing'].get(r, describe_status(c))} | - | - | - |")
        rho = sec['spearman_rho']
        L += ['', f"Secondary (DESCRIPTIVE ONLY, no significance claim): Spearman(log params, gap) over {sec['n_points']} available points = "
              + (f'{rho:+.3f}' if rho is not None else f"not computed ({sec['note'] or 'zero variance'})") + '.']
        g = rep['H3']['gap27_minus_gap2_descriptive'][s]
        L += [f"Descriptive gap(qwen3.5-27b) − gap(qwen3.5-2b), NOT the preregistered primary: "
              + (f"{cdiff(g)} (family {ci(g['family'])}; template {ci(g['template'])})" if 'estimate' in g else g['verdict']) + '.', '']
    # (d) completeness
    L += ['## (d) Completeness per cell', '',
          '| sender → receiver | cell status | engineering | dev shards (receipted / present, declared S) | dev rows present / expected | notes |',
          '|---|---|---|---|---|---|']
    for s in senders:
        for r in receivers:
            c = cells[(s, r)]
            d = c.get('development') or {}
            sh = d.get('shards', [])
            notes = '; '.join(c.get('errors', []) + c.get('warnings', []) + d.get('problems', []))
            integ = (d.get('integrity') or {}).get('row_errors', {})
            if integ.get('count'):
                notes += f"; {integ['count']} row errors, e.g. {integ['examples'][:2]}"
            L.append(f"| {s} → {r}{' (same-model)' if (s, r) in same_pairs else ''} | {describe_status(c)} | {(c.get('engineering') or {}).get('verdict', '-')} "
                     f"| {sum(x['receipt'] == 'valid' for x in sh)} / {len(sh)}, S={d.get('declared_shards') or '?'} "
                     f"| {d.get('rows_present', 0)} / {d.get('expected_rows', '-')} | {notes or '-'} |")
    L.append('')
    # (e) complete facts
    L += ['## (e) Complete facts (source `literal`, first-sender cells) — descriptive', '',
          '| receiver | status | J selected [95%] | J native | J packet_text | J full_text | coverage selected | selected − text | selected − native | selected − full_text |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for r in receivers:
        v = rep['complete_facts'].get(r)
        if v is None:
            L.append(f'| {r} | no literal cell | - | - | - | - | - | - | - | - |'); continue
        if v['status'] != 'COMPLETE':
            L.append(f"| {r} | {v['status']} | - | - | - | - | - | - | - | - |"); continue
        J, C = v['J'], v['contrasts']
        L.append(f"| {r} | COMPLETE | {ci(J['selected']['family'])} | {ci(J['native']['family'])} | {ci(J['packet_text']['family'])} "
                 f"| {ci(J['full_text']['family'])} | {pct(v['coverage']['selected']['mean'])} | {cdiff(C['literal_selected_minus_packet_text'])} "
                 f"| {cdiff(C['literal_selected_minus_native'])} | {cdiff(C['literal_selected_minus_full_text'])} |")
    L.append('')
    # (f) H4
    L += ['## (f) H4 depth (descriptive): selected − original', '',
          '| sender → receiver | J(selected) | J(original) | selected − original [both levels] | family [95%] | template [95%] |', '|---|---|---|---|---|---|']
    for s in senders:
        for r in receivers:
            v = rep['H4'][s][r]
            tag = ' (same-model)' if v['same_model_pair'] else ''
            if 'contrast' not in v:
                L.append(f"| {s} → {r}{tag} | {v['verdict']} | - | - | - | - |"); continue
            c = v['contrast']
            L.append(f"| {s} → {r}{tag} | {pct(v['J_selected']['mean'])} | {pct(v['J_original']['mean'])} | {cdiff(c)} | {ci(c['family'])} | {ci(c['template'])} |")
    L.append('')
    # (g) other endpoints
    L += ['## (g) Other endpoints (complete cells, sender source)', '',
          '| sender → receiver | arm | J | accuracy | coverage | changed R | unchanged U | J@256 | coverage@256 |', '|---|---|---|---|---|---|---|---|---|']
    for s in senders:
        for r in receivers:
            c = cells[(s, r)]
            if c.get('status') != 'COMPLETE':
                continue
            for arm in ('none', 'original', 'selected', 'packet_text', 'sham0', 'sham1', 'sham2', 'other_world'):
                e = c['estimates'][f'{s}/{arm}']
                L.append(f"| {s} → {r} | {arm} | {pct(e['J']['family']['mean'])} | {pct(e['accuracy']['family']['mean'])} | {pct(e['coverage']['family']['mean'])} "
                         f"| {pct(e['R']['family']['mean'])} | {pct(e['U']['family']['mean'])} | {pct(e['J256']['family']['mean'])} | {pct(e['coverage256']['family']['mean'])} |")
            ow = c['estimates'][f'{s}/other_world']
            L.append(f"| {s} → {r} | other_world vs SUPPLIED gold (consistency) | {pct(ow['J_supplied']['family']['mean'])} | {pct(ow['accuracy_supplied']['family']['mean'])} | - | - | - | {pct(ow['J256_supplied']['family']['mean'])} | - |")
    L += ['', '## Resolutions of lock ambiguities (stated, not silently chosen)', '']
    L += [f'{i + 1}. {t}' for i, t in enumerate(rep['resolutions'])]
    L += ['', '## Not claims (from the lock)', ''] + [f'- {t}' for t in rep['not_claims'] or []]
    L += ['', f"Same-model rule: {rep['same_model_rule']}", '',
          'Limits: worlds, questions, arms and senders are repeated measures on the same scenes and add no independent N; '
          'bootstrap seeds are not model replications; the no-message J is structurally 0; the bank is finite and constructed '
          'and was previously evaluated with four older receivers and the Llama-3.1-8B-Instruct sender (lock `bank.disclosure`).', '']
    if rep['ignored_paths']:
        L += ['Ignored paths (not lock cells): ' + ', '.join(f'`{p}`' for p in rep['ignored_paths']), '']
    return '\n'.join(L)


def print_summary(rep, senders, receivers, cells, same_pairs):
    for s in senders:
        h1 = rep['H1'][s]
        print(f"[{s}] H1 {h1['count_passing']}/{h1['m_evaluated']} evaluated (planned {h1['planned_cross_model']}); "
              f"H2 {dict((k, len(v)) for k, v in rep['H2'][s]['tally'].items())}")
        for r in receivers:
            c = cells[(s, r)]
            tag = ' [same-model]' if (s, r) in same_pairs else ''
            if c.get('status') == 'COMPLETE':
                e = c['estimates']
                print(f"   {r:24s}{tag} J(sel) {pct(e[f'{s}/selected']['J']['family']['mean']):>5s}  J(text) {pct(e[f'{s}/packet_text']['J']['family']['mean']):>5s}  "
                      f"cov {pct(e[f'{s}/selected']['coverage']['family']['mean']):>5s}  "
                      + ('' if (s, r) in same_pairs else f"H1 {rep['H1'][s]['per_receiver'][r]['verdict']}  H2 {rep['H2'][s]['per_receiver'][r]['verdict']}"))
            else:
                print(f'   {r:24s}{tag} {describe_status(c)}')


if __name__ == '__main__':
    try:
        main()
    except LockError as e:
        print('LOCK MISMATCH:', e)
        sys.exit(3)
