"""Experiment E (latest0924) runner: does frozen delivery survive new slot positions, templates and object counts?

Everything that defines the interface is frozen and imported, not copied: the receiver loader, the write/capture
hooks, the single write at the receiver's depth-0.2 layer with gamma = 1 and raw twelve-context directions
(fitted_states), the norm-matched reassignment (permuted_states), prefix-cache branching per question, greedy decoding,
the strict parser, scores at 64/256 tokens, and the engineering checks (identity write exactly inert; cached vs
complete forward max logit <= 0.05, alias gap <= 0.1, equal parses). Adapter installation, pins and receipts follow
src/syncomm/latest/evaluate_graded.py (imported). What is new is only the message layout (src/syncomm/latest/layout_render.py):

  L0 standard two-object layout (must reproduce the Experiment A complete-fact rows exactly)
  L1 facts in slots 3 and 2 (reversed), slots 0-1 neutral; same template and neutral prefix as L0
  L2 new header wording, separators, weight annotation and reversed object order
  L3 three objects in the standard template (object C is a distractor with distinct values)

Material: complete facts (every weight 255), no model sender. Arms per condition and world: none, internal, text,
sham0 (the R3 reassignment 0, per-layer edit norm matched to internal); engineering adds identity_write.

Subcommands: make-manifest | run (--phase engineering|development, --shard/--shards; resumable) | dry-run (tokenizer only)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import torch  # noqa: E402

from syncomm.recovery.evaluate_depth import (  # noqa: E402
    build_receiver, write_hooks, capture_hooks, forward, fitted_states, prior, base, task, token_spans)
from syncomm.recovery.evaluate_confirm import permuted_states  # noqa: E402
from syncomm.latest import evaluate_graded as eg  # noqa: E402
from syncomm.latest.layout_render import RENDER, CONDITIONS, WIDTH, fact_words, condition_slots  # noqa: E402

STUDY = 'latest0924_layout'
OUT = ROOT / 'artifacts/latest0924/layout'
BANK = OUT / 'bank.json'
RENDERER = ROOT / 'src/syncomm/latest/layout_render.py'
BUILDER = ROOT / 'scripts/latest0924/layout/build_layout_bank.py'
EXPA_LOCK = ROOT / 'gates/latest0924/expA_protocol.lock.json'
R3_MANIFEST = ROOT / 'artifacts/receiver_recovery/r3/qwen3-32b/manifest.json'
ARMS = ['none', 'internal', 'text', 'sham0']
ENGINEERING_ARMS = ['identity_write']
NEW_RECEIVERS = ['qwen3.5-27b', 'qwen3.6-27b', 'mistral-small-3.2-24b', 'qwen3.5-9b', 'granite-4.1-8b']
RECEIVERS = ['qwen3-32b'] + NEW_RECEIVERS


def receiver_spec(key):
    if key == 'qwen3-32b':
        s = eg.RECEIVERS['qwen3-32b']
        return dict(model_dir=str(s['model_dir']), n_blocks=s['n_blocks'], selected_layers=s['selected_layers'],
                    directions=str(s['directions']), adapter='optional', provenance=str(R3_MANIFEST))
    lock = json.loads(EXPA_LOCK.read_text())['receivers'][key]
    return dict(model_dir=lock['model_dir'], n_blocks=lock['n_layers'], selected_layers=[lock['selected_layer']],
                directions=str(ROOT / f'artifacts/latest0924/fits/directions_{key}.pt'), adapter='required',
                provenance=str(EXPA_LOCK))


def make_manifest(key, out):
    spec = receiver_spec(key)
    bank = json.loads(BANK.read_text())
    r3 = json.loads(R3_MANIFEST.read_text())
    files = [BANK, Path(bank['words_file']), RENDERER, BUILDER, Path(__file__).resolve(), ROOT / 'src/syncomm/latest/evaluate_graded.py',
             ROOT / 'src/syncomm/recovery/evaluate_confirm.py', ROOT / 'src/syncomm/recovery/evaluate_depth.py',
             Path(spec['directions']), eg.ADAPTER]
    m = dict(study=STUDY, receiver=key, root=str(out.resolve().parent), models={key: spec['model_dir']}, base_models=[],
             n_blocks=spec['n_blocks'], selected_depth=0.2, selected_layers=spec['selected_layers'], gamma=1,
             scale_rule='raw fitted word-minus-unknown directions, no rescaling', directions=spec['directions'],
             directions_provenance=spec['provenance'], words_file=bank['words_file'], bank=str(BANK),
             conditions=list(CONDITIONS), arms=list(ARMS), engineering_arms=list(ENGINEERING_ARMS),
             engineering_scene_ids=bank['engineering_scene_ids'], adapter=spec['adapter'], row_split='latest0924_layout',
             permutations={'sham0': r3['permutations'][0]},
             permutation_provenance=dict(file=str(R3_MANIFEST), index=0, seed=r3['permutation_seeds'][0]),
             engineering_thresholds=dict(max_logit_difference=0.05, alias_gap_difference=0.1, parsed_equal=True,
                                         identity_write='exact (logits sha and tokens)'),
             pins={str(p): eg.sha_file(p) for p in files})
    text = json.dumps(m, indent=1, sort_keys=True) + '\n'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.read_text() != text:
        raise SystemExit(f'REFUSE: {out} exists with different content')
    out.write_text(text)
    print('MANIFEST', out)


def load_manifest(path, strict=True):
    m = json.loads(Path(path).read_text())
    assert m['study'] == STUDY
    bad = [p for p, want in m['pins'].items() if eg.sha_file(p) != want]
    if strict:
        assert not bad, f'pinned inputs changed: {bad}'
    return m, bad


def prompt_inputs_layout(receiver, scene, condition, object_words):
    message, chars = RENDER[condition](scene['aliases'], object_words)
    texts, seqs, sites, n = receiver.tokenized(scene, message)
    enc = receiver.tok(texts[0], return_offsets_mapping=True, add_special_tokens=not receiver.chat)
    assert enc['input_ids'] == seqs[0]
    spans = token_spans(enc['offset_mapping'], chars, texts[0].index(message))
    assert len(spans) == WIDTH * len(scene['aliases']) and all(s['last_position'] < n for s in spans)
    return dict(message=message, prefix=seqs[0][:n], suffixes=[s[n:] for s in seqs], slots=spans, n=n)


def active_entries(condition, world, words, mapping):
    ow, ent = condition_slots(condition, fact_words(world, words))
    active = [dict(object=o, index=i, address=words.index(w), word=w, weight=255, relative_weight=1.0) for o, i, w in ent]
    positions = sorted(mapping[o, i]['last_position'] for o, i, _ in ent)
    assert len(set(positions)) == len(positions)
    return ow, active, positions


def phase_arms(m, phase):
    arms = list(m['arms'])
    return arms[:1] + list(m['engineering_arms']) + arms[1:] if phase == 'engineering' else arms


@torch.inference_mode()
def scene_rows(receiver, pair, m, phase, dirs, arms):
    layers = m['selected_layers']
    records = []
    for condition in m['conditions']:
        scene = pair[1] if condition == 'L3' else pair[0]
        ni = prompt_inputs_layout(receiver, scene, condition, None)
        neutral_states = {}
        with capture_hooks(receiver.model, layers, neutral_states):
            neutral_out = forward(receiver, ni['prefix'])
        mapping = {(s['object'], s['index']): s for s in ni['slots']}
        all_slots = sorted(s['last_position'] for s in ni['slots'])
        zero_ref = {}
        for wi, world in enumerate(scene['worlds']):
            ow, active, positions = active_entries(condition, world, receiver.words, mapping)
            internal_norms = None
            for arm in arms:
                report, states, nt = {}, None, None
                ids, suffixes, extra, write_positions = ni['prefix'], ni['suffixes'], None, positions
                neutral = {l: neutral_states[l] for l in layers}
                if arm == 'none':
                    out = neutral_out
                elif arm == 'text':
                    extra, _ = RENDER[condition](scene['aliases'], ow)
                    _, seqs, _, n = receiver.tokenized(scene, extra)
                    ids, suffixes = seqs[0][:n], [s[n:] for s in seqs]
                    out = forward(receiver, ids)
                else:
                    if arm == 'identity_write':
                        states, write_positions = neutral, all_slots
                    elif arm == 'internal':
                        states = fitted_states(neutral, layers, dirs, [1.0] * len(layers), active, mapping, m['gamma'])
                    elif arm == 'sham0':
                        assert internal_norms is not None, 'internal must precede sham0'
                        nt = internal_norms
                        states = permuted_states(neutral, layers, dirs, active, mapping, m['gamma'], m['permutations']['sham0'])
                    else:
                        raise AssertionError(arm)
                    with write_hooks(receiver.model, layers, states, write_positions, report, nt):
                        out = forward(receiver, ids)
                if arm == 'internal':
                    internal_norms = {k: v['norm'] for k, v in report.items()}
                cache = out.past_key_values
                fp = prior.cache_fingerprint(cache)
                for qi, suffix in enumerate(suffixes):
                    branch = prior.clone_cache(cache)
                    assert prior.cache_fingerprint(branch) == fp
                    end = receiver.model(torch.tensor([suffix], device='cuda:0'), past_key_values=branch, use_cache=True,
                                         attention_mask=torch.ones((1, len(ids) + len(suffix)), device='cuda:0', dtype=torch.long))
                    logits = end.logits[0, -1].float()
                    value = receiver.decode(logits, end.past_key_values, scene, 256)
                    gold = world['answers'][qi]
                    checks = None
                    if phase == 'engineering':
                        ctx = (write_hooks(receiver.model, layers, states, write_positions, {}, nt) if states is not None
                               else contextlib.nullcontext())
                        with ctx:
                            full = forward(receiver, ids + suffix)
                        direct = receiver.decode(full.logits[0, -1].float(), full.past_key_values, scene, 256)
                        checks = dict(max_logit_difference=float((full.logits[0, -1].float() - logits).abs().max()),
                                      alias_gap_difference=abs(direct['gap'] - value['gap']),
                                      parsed_equal=all(task.score_answer(direct[f'text{b}'], scene['aliases'], gold) ==
                                                       task.score_answer(value[f'text{b}'], scene['aliases'], gold) for b in (64, 256)))
                        checks['pass'] = checks['max_logit_difference'] <= .05 and checks['alias_gap_difference'] <= .1 and checks['parsed_equal']
                    row = dict(scene=pair[0]['scene'], family=scene['family'], domain=scene['domain'], template=scene['template'],
                               condition=condition, world=wi, question=qi, aliases=scene['aliases'], gold=gold,
                               split=m['row_split'], receiver=m['receiver'], arm=arm, layers=layers,
                               gamma=(m['gamma'] if arm in ('internal', 'sham0') else 0), entries=active, extra_text=extra,
                               prefix_ids=ids, suffix_ids=suffix, slots=ni['slots'],
                               patch_positions=write_positions if states is not None else [], patch_report=report,
                               prefix_cache_sha256=fp, engineering_checks=checks, **value)
                    row.update({f'score{b}': task.score_answer(value[f'text{b}'], scene['aliases'], gold) for b in (64, 256)})
                    if arm == 'none':
                        z = (fp, value['logits_sha256'], value['tokens'])
                        assert zero_ref.setdefault(qi, z) == z, 'the no-message prefix depends on the world'
                    if arm == 'identity_write':
                        assert (fp, value['logits_sha256'], value['tokens']) == zero_ref[qi], 'identity write is not inert'
                        assert report and all(v['norm'] <= 1e-6 for v in report.values())
                    if arm == 'sham0':
                        assert all(abs(report[k]['norm'] - internal_norms[k]) <= max(1e-4, 1e-4 * internal_norms[k]) for k in report)
                    records.append(row)
                assert prior.cache_fingerprint(cache) == fp
    return records


def select_pairs(m, bank, phase, shard, shards):
    if phase == 'engineering':
        pairs = list(zip(bank['eng2'], bank['eng3']))
        assert [p[0]['scene'] for p in pairs] == m['engineering_scene_ids']
    else:
        pairs = list(zip(bank['scenes2'], bank['scenes3']))[shard::shards]
    for a, b in pairs:
        assert b['scene_source'] == a['scene']
    return pairs


def run(a):
    m, _ = load_manifest(a.manifest)
    root = Path(m['root'])
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    adapter = eg.install_adapter(m['adapter'])
    bank = json.loads(Path(m['bank']).read_text()); words = json.loads(Path(m['words_file']).read_text())
    receiver = build_receiver(m, m['receiver'], words)
    assert len(base.decoder_layers(receiver.model)) == m['n_blocks']
    dtype = str(next(receiver.model.parameters()).dtype)
    print('RECEIVER', m['receiver'], dtype, 'adapter', json.dumps(adapter), 'phase', a.phase, flush=True)
    blob = torch.load(Path(m['directions']), map_location='cpu', weights_only=False)
    assert blob['words'] == words and all(l in blob['layers'] for l in m['selected_layers'])
    order = {l: i for i, l in enumerate(blob['layers'])}
    dirs = blob['directions'].float()[[order[l] for l in m['selected_layers']]].to('cuda:0')
    arms = phase_arms(m, a.phase)
    pairs = select_pairs(m, bank, a.phase, a.shard, a.shards)
    raw = root / 'raw' / f'{a.phase}_{m["receiver"]}_s{a.shard}.jsonl'
    raw.parent.mkdir(parents=True, exist_ok=True)
    seen = {r['key']: r for r in map(json.loads, raw.read_text().splitlines())} if raw.exists() else {}
    assert all(not r['engineering_checks'] or r['engineering_checks']['pass'] for r in seen.values())
    keyf = lambda r: '|'.join(str(r[f]) for f in ('scene', 'condition', 'world', 'question', 'arm'))
    started = time.time(); computed = 0
    for i, pair in enumerate(pairs):
        sid = pair[0]['scene']
        expected = {f'{sid}|{c}|{w}|{q}|{arm}' for c in m['conditions'] for w in (0, 1) for q in (0, 1) for arm in arms}
        if expected <= seen.keys():
            continue
        rows = scene_rows(receiver, pair, m, a.phase, dirs, arms)
        for row in rows:
            row['key'] = keyf(row); row['phase'] = a.phase
            if row['key'] in seen:
                assert all(row[f] == seen[row['key']][f] for f in ('logits_sha256', 'tokens')), 'resume drift'
            else:
                with raw.open('a') as h:
                    h.write(json.dumps(row) + '\n')
                seen[row['key']] = row
        computed += len(rows)
        assert all(not r['engineering_checks'] or r['engineering_checks']['pass'] for r in rows), 'engineering failure'
        el = time.time() - started
        print('SCENE', i + 1, len(pairs), 'rows', len(seen), f'{el / max(computed, 1):.3f} s/row', flush=True)
    want = len(pairs) * len(m['conditions']) * 4 * len(arms)
    assert len(seen) == want, (len(seen), want)
    elapsed = time.time() - started
    prior.atomic_json(raw.with_suffix('.done.json'),
                      dict(status='PASS', study=STUDY, phase=a.phase, receiver=m['receiver'], shard=a.shard, shards=a.shards,
                           rows=len(seen), scenes=len(pairs), conditions=m['conditions'], arms=arms, rows_computed_this_process=computed,
                           manifest_sha256=eg.sha_file(a.manifest), raw_sha256=eg.sha_file(raw), elapsed_seconds=elapsed,
                           seconds_per_computed_row=(elapsed / computed if computed else None), adapter=adapter, model_dtype=dtype,
                           transformers=__import__('transformers').__version__, python=sys.executable))
    print('COMPLETE', m['receiver'], a.phase, a.shard, flush=True)


def dry_run(a):
    """Tokenizer only: render every condition, map every write position, and compare L0 prompts with Experiment A
    complete-fact rows (or R3 rows for qwen3-32b)."""
    m, bad = load_manifest(a.manifest, strict=False)
    adapter = eg.install_adapter(m['adapter'])
    bank = json.loads(Path(m['bank']).read_text()); words = json.loads(Path(m['words_file']).read_text())
    r = eg.tokenizer_receiver(m, m['receiver'], words, adapter['installed'])
    ref = {}
    if a.compare:
        for raw in sorted(Path(a.compare).glob('raw/development_*.jsonl')) + sorted(Path(a.compare).glob('raw/engineering_*.jsonl')):
            for line in open(raw):
                x = json.loads(line)
                if x['source'] == 'literal' and x['arm'] in ('selected', 'packet_text'):
                    ref[x['scene'], x['world'], x['question'], x['arm']] = x
    stats = dict(worlds=0, conditions={}, l0_compared=0, l0_prompt_mismatch=0, l0_positions_mismatch=0, pin_mismatches=bad)
    for pair in list(zip(bank['scenes2'], bank['scenes3'])) + list(zip(bank['eng2'], bank['eng3'])):
        for c in m['conditions']:
            scene = pair[1] if c == 'L3' else pair[0]
            ni = prompt_inputs_layout(r, scene, c, None)
            mapping = {(s['object'], s['index']): s for s in ni['slots']}
            for wi, world in enumerate(scene['worlds']):
                ow, active, positions = active_entries(c, world, words, mapping)
                for e in active:
                    sl = mapping[e['object'], e['index']]
                    assert r.tok.decode([ni['prefix'][p] for p in sl['positions']]).strip() == 'unknown', (c, e)
                text, _ = RENDER[c](scene['aliases'], ow)
                _, seqs, _, n = r.tokenized(scene, text)
                stats['conditions'][c] = stats['conditions'].get(c, 0) + 1
                if c == 'L0' and ref:
                    for qi in (0, 1):
                        x_int = ref.get((pair[0]['scene'], wi, qi, 'selected')); x_txt = ref.get((pair[0]['scene'], wi, qi, 'packet_text'))
                        if x_int is None or x_txt is None:
                            continue
                        stats['l0_compared'] += 1
                        if x_int['prefix_ids'] != ni['prefix'] or x_int['suffix_ids'] != ni['suffixes'][qi] or \
                           x_txt['prefix_ids'] != seqs[0][:n] or x_txt['suffix_ids'] != seqs[qi][n:]:
                            stats['l0_prompt_mismatch'] += 1
                        if sorted(x_int['patch_positions']) != positions:
                            stats['l0_positions_mismatch'] += 1
                stats['worlds'] += 1
    stats['status'] = 'PASS' if not stats['l0_prompt_mismatch'] and not stats['l0_positions_mismatch'] else 'FAIL'
    out = Path(m['root']) / 'dryrun.json'
    out.write_text(json.dumps(stats, indent=1) + '\n')
    print(json.dumps(stats))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    mk = sub.add_parser('make-manifest'); mk.add_argument('--receiver', required=True, choices=RECEIVERS)
    rn = sub.add_parser('run'); rn.add_argument('--manifest', type=Path, required=True)
    rn.add_argument('--phase', choices=('engineering', 'development'), required=True)
    rn.add_argument('--shard', type=int, default=0); rn.add_argument('--shards', type=int, default=1)
    dr = sub.add_parser('dry-run'); dr.add_argument('--manifest', type=Path, required=True)
    dr.add_argument('--compare', type=Path, default=None, help='an Experiment A cell root (or R3 root) with literal rows')
    a = ap.parse_args()
    if a.cmd == 'make-manifest':
        make_manifest(a.receiver, OUT / a.receiver / 'manifest.json')
    elif a.cmd == 'run':
        run(a)
    else:
        dry_run(a)


if __name__ == '__main__':
    main()
