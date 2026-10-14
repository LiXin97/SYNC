"""Experiment B follow-up (latest0924, text-format baseline): do stronger TEXT renderings of graded weights close the gap
to the internal (write-strength) route?

Built on src/syncomm/latest/evaluate_graded.py (Experiment B runner, imported unchanged) and the frozen primitives it
imports. Same bank, same 144 scenes (tiers 1+2), same graded packets packets_L{0,64,128,191,255}.json, same four
engineering scenes, questions, parser, budgets (64/256), FP32 eager, greedy, batch 1, same selected layer, gamma 1 and
raw fitted directions. The internal comparator for formats F1/F2 is Experiment B's own internal_L* rows (reused, not
rerun). What is new:

  textpct_L{LLL}     F1: each filled slot prints its weight as a percentage of 255, e.g. "red [25%]", header
                     "Selected words and exact relative weights (percent): " (unfilled slots "unknown [100%]")
  textverbal_L{LLL}  F2: ordinal labels 255 strong / 191 fairly strong / 128 moderate / 64 weak, e.g. "red [weak]",
                     header "Selected words and relative weights (strong, fairly strong, moderate, weak): "
  textinstr_L{LLL}   F3 text: Experiment B's exact "[w/255]" message, plus INSTRUCTION inserted in the receiver prompt
  internal_instr_L{LLL}  F3 internal: Experiment B's internal write, with the same INSTRUCTION in the (neutral) prompt
  none, none_instr   no message (plain / instructed prompt)
  identity_write, identity_write_instr   engineering only; must be exactly inert against none / none_instr
  internal_L064, text_L064               engineering only; continuity with Experiment B's engineering rows

INSTRUCTION is one line inserted immediately before "Message:\n" (after the frozen few-shot block); everything else in
the prompt is byte-identical. Subcommands: make-manifest | run | dry-run (tokenizer only).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import torch  # noqa: E402

from syncomm.latest import evaluate_graded as eg  # noqa: E402  Experiment B runner, imported unchanged
from syncomm.recovery.evaluate_depth import (  # noqa: E402  frozen primitives
    WIDTH, build_receiver, write_hooks, capture_hooks, forward, prompt_inputs, entry_positions, fitted_states,
    message_block, parent, prior, base, task, render, token_spans)

STUDY = 'LATEST0924_GRADED_B_TEXTFMT_V1'
OUT = ROOT / 'artifacts/latest0924/graded_textfmt'
EXPB = ROOT / 'artifacts/latest0924/graded'
LEVELS = eg.LEVELS
HIGH = 255
NEUTRAL_WORD = 'unknown'
INSTRUCTION = 'If an object lists two values for the same attribute, the value with the higher weight is the true one.\n'
VERBAL = {255: 'strong', 191: 'fairly strong', 128: 'moderate', 64: 'weak'}
HEADERS = {
    'pct': 'Selected words and exact relative weights (percent): ',
    'verbal': 'Selected words and relative weights (strong, fairly strong, moderate, weak): ',
}
DEV_ARMS = ['none', 'none_instr'] + [f'{h}_L{L:03d}' for L in LEVELS
                                     for h in ('textpct', 'textverbal', 'textinstr', 'internal_instr')]
ENG_EXTRA = ['identity_write', 'identity_write_instr', 'internal_L064', 'text_L064']
RECEIVERS = ('qwen3-32b', 'qwen3.5-27b')


def pct_label(w):
    return f'{round(100 * w / 255)}%'


def render_fmt(fmt, aliases, object_words, object_weights, width=WIDTH):
    """Same construction as build_graded_bank.render_weighted (fields, separators, spans), different weight notation."""
    assert fmt in HEADERS and width in (2, 4) and len(aliases) == 2
    text, spans = HEADERS[fmt], []
    for obj, (alias, words, weights) in enumerate(zip(aliases, object_words, object_weights)):
        assert len(words) <= width and len(weights) == len(words)
        if obj:
            text += '; '
        text += alias + ': '
        for index in range(width):
            if index:
                text += ', '
            word = words[index] if index < len(words) else NEUTRAL_WORD
            weight = weights[index] if index < len(words) else HIGH
            start = len(text)
            text += word
            spans.append(dict(object=obj, index=index, word=word, start=start, end=len(text)))
            text += f' [{pct_label(weight) if fmt == "pct" else VERBAL[weight]}]'
    text += '.'
    return text, spans


class InstrScaffold:
    """parent.Scaffold.receiver_body with INSTRUCTION inserted immediately before 'Message:\\n'."""
    @staticmethod
    def receiver_body(scene, qi, extra=None):
        body, carriers = parent.Scaffold.receiver_body(scene, qi, extra)
        head = task.FEWSHOT + 'Message:\n'
        assert body.startswith(head)
        return task.FEWSHOT + INSTRUCTION + body[len(task.FEWSHOT):], carriers


@contextlib.contextmanager
def instructed(receiver):
    old = receiver.task
    receiver.task = InstrScaffold
    try:
        yield
    finally:
        receiver.task = old


def parse_arm(arm):
    """-> (route, level, variant). route: none | identity_write | internal | text; variant: plain | instr | pct | verbal."""
    if arm in ('none', 'identity_write'):
        return arm, None, 'plain'
    if arm in ('none_instr', 'identity_write_instr'):
        return arm[:-6], None, 'instr'
    head, level = arm.split('_L'); level = int(level)
    assert level in LEVELS and arm == f'{head}_L{level:03d}', arm
    return {'textpct': ('text', level, 'pct'), 'textverbal': ('text', level, 'verbal'), 'textinstr': ('text', level, 'instr'),
            'internal_instr': ('internal', level, 'instr'), 'internal': ('internal', level, 'plain'),
            'text': ('text', level, 'plain')}[head]


def phase_arms(phase):
    return DEV_ARMS[:2] + ENG_EXTRA + DEV_ARMS[2:] if phase == 'engineering' else list(DEV_ARMS)


# ------------------------------------------------------------------------------------------------ manifest

def make_manifest(key, out):
    expb = EXPB / key / 'manifest.json'
    mb = json.loads(expb.read_text())
    pins = dict(mb['pins'])
    for p in (Path(__file__).resolve(), ROOT / 'src/syncomm/latest/evaluate_graded.py', expb):
        pins[str(p)] = eg.sha_file(p)
    m = dict(mb, study=STUDY, status='READY_FOR_LOCK', expb_manifest=str(expb), expb_manifest_sha256=eg.sha_file(expb),
             purpose='Experiment B text-format follow-up: F1 percent, F2 verbal, F3 instructed text and instructed internal.',
             design=str(OUT / 'DESIGN.md'), lock='gates/latest0924/expB_textfmt.lock.json',
             root=str(out.resolve().parent), arms=list(DEV_ARMS), engineering_arms=list(ENG_EXTRA),
             instruction=INSTRUCTION, verbal_labels={str(k): v for k, v in VERBAL.items()}, headers=HEADERS,
             row_split='latest0924_graded_textfmt', pins=pins)
    text = json.dumps(m, indent=1, sort_keys=True) + '\n'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.read_text() != text:
        raise SystemExit(f'REFUSE: {out} exists with different content')
    out.write_text(text)
    print('MANIFEST', out)


def load_manifest(path):
    m = json.loads(Path(path).read_text())
    assert m['study'] == STUDY
    for p, want in m['pins'].items():
        assert eg.sha_file(p) == want, f'pinned input changed: {p}'
    return m


# ------------------------------------------------------------------------------------------------ one scene

def text_for(variant, scene, words_obj, weights_obj, rec):
    if variant in ('plain', 'instr'):
        extra = eg.render_weighted(scene['aliases'], words_obj, weights_obj, width=WIDTH)[0]
        assert extra == rec['text']
        return extra
    return render_fmt(variant, scene['aliases'], words_obj, weights_obj)[0]


@torch.inference_mode()
def scene_rows(receiver, scene, m, phase, packets, dirs, arms, fp32):
    layers = m['selected_layers']
    prompts = {}
    for variant in ('plain', 'instr'):
        ctx = instructed(receiver) if variant == 'instr' else contextlib.nullcontext()
        with ctx:
            ni = prompt_inputs(receiver, scene, None)
        states = {}
        with capture_hooks(receiver.model, layers, states):
            out = forward(receiver, ni['prefix'])
        prompts[variant] = dict(ni=ni, states=states, out=out,
                                mapping={(s['object'], s['index']): s for s in ni['slots']},
                                all_slots=sorted(s['last_position'] for s in ni['slots']))
    records, zero_ref, l255_ref = [], {}, {}
    for wi, world in enumerate(scene['worlds']):
        for arm in arms:
            route, level, variant = parse_arm(arm)
            P = prompts['instr' if variant == 'instr' else 'plain']
            ni, mapping = P['ni'], P['mapping']
            report, states, slot_edits = {}, None, None
            ids, suffixes, extra, rec, active, write_positions = ni['prefix'], ni['suffixes'], None, None, [], []
            if level is not None:
                rec, packet, entries, words_obj, weights_obj = eg.packet_view(packets, level, scene, wi, receiver.words)
                active, write_positions, _ = entry_positions(entries, ni['slots'])
            if route == 'none':
                out = P['out']
            elif route == 'text':
                extra = text_for(variant, scene, words_obj, weights_obj, rec)
                ctx = instructed(receiver) if variant == 'instr' else contextlib.nullcontext()
                with ctx:
                    _, seqs, _, n = receiver.tokenized(scene, extra)
                ids, suffixes = seqs[0][:n], [s[n:] for s in seqs]
                out = forward(receiver, ids)
            else:
                neutral = {l: P['states'][l] for l in layers}
                if route == 'identity_write':
                    states, write_positions = neutral, P['all_slots']
                else:
                    states = fitted_states(neutral, layers, dirs, [1.0] * len(layers), active, mapping, m['gamma'])
                    slot_edits = eg.slot_edit_report(states, neutral, layers, active, mapping, dirs, m['gamma'])
                    if phase == 'engineering' and fp32:
                        for s in slot_edits:
                            assert abs(s['edit_norm'] - s['expected_edit_norm']) <= 1e-2 * s['expected_edit_norm'] + 1e-3, s
                with write_hooks(receiver.model, layers, states, write_positions, report, None):
                    out = forward(receiver, ids)
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
                wgold = eg.weight_gold(scene, rec, qi) if rec is not None else None
                if rec is not None:
                    assert wgold in (gold, None) and (wgold is not None or (qi == 0 and level == 255))
                checks = None
                if phase == 'engineering':
                    ctx = (write_hooks(receiver.model, layers, states, write_positions, {}, None) if states is not None
                           else contextlib.nullcontext())
                    with ctx:
                        full = forward(receiver, ids + suffix)
                    direct = receiver.decode(full.logits[0, -1].float(), full.past_key_values, scene, 256)
                    checks = dict(max_logit_difference=float((full.logits[0, -1].float() - logits).abs().max()),
                                  alias_gap_difference=abs(direct['gap'] - value['gap']),
                                  parsed_equal=all(task.score_answer(direct[f'text{b}'], scene['aliases'], gold) ==
                                                   task.score_answer(value[f'text{b}'], scene['aliases'], gold) for b in (64, 256)))
                    checks['pass'] = checks['max_logit_difference'] <= .05 and checks['alias_gap_difference'] <= .1 and checks['parsed_equal']
                l255_identical = None
                if level == 255:
                    z = (value['logits_sha256'], value['tokens'])
                    if wi == 0:
                        l255_ref[arm, qi] = z
                    else:
                        l255_identical = l255_ref[arm, qi] == z
                        if phase == 'engineering':
                            assert l255_identical, f'{arm}: identical packets gave different outputs across worlds'
                row = dict(scene=scene['scene'], family=scene['family'], domain=scene['domain'], template=scene['template'],
                           world=wi, question=qi, aliases=scene['aliases'], gold=gold, gold_supplied=gold, weight_gold=wgold,
                           split=m['row_split'], receiver=m['receiver'], source='graded_textfmt', arm=arm, route=route,
                           variant=variant, level=level, gamma=(m['gamma'] if route == 'internal' else 0), layers=layers,
                           x_field=(rec['x_field'] if rec else scene['changed_field']), entries=active,
                           words_by_object=(rec['words_by_object'] if rec else None),
                           weights_by_object=(rec['weights_by_object'] if rec else None),
                           condition_packet_sha256=(rec['sha256'] if rec else None),
                           wire_bytes=(len(extra.encode()) if extra is not None else
                                       0 if route in ('none', 'identity_write') else rec['bytes']),
                           extra_text=extra, prefix_ids=ids, suffix_ids=suffix, slots=ni['slots'],
                           patch_positions=write_positions if states is not None else [], patch_report=report,
                           slot_edits=slot_edits, prefix_cache_sha256=fp, l255_world_identical=l255_identical,
                           engineering_checks=checks, **value)
                row.update({f'score{b}': task.score_answer(value[f'text{b}'], scene['aliases'], gold) for b in (64, 256)})
                if route == 'none':
                    z = (fp, value['logits_sha256'], value['tokens'])
                    assert zero_ref.setdefault((variant, qi), z) == z, 'the no-message prefix depends on the world'
                if route == 'identity_write':
                    assert (fp, value['logits_sha256'], value['tokens']) == zero_ref[variant, qi], f'{arm} is not inert'
                    assert report and all(v['norm'] <= 1e-6 for v in report.values())
                records.append(row)
            assert prior.cache_fingerprint(cache) == fp
    return records


# ------------------------------------------------------------------------------------------------ run

def run(a):
    m = load_manifest(a.manifest)
    root = Path(m['root'])
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    adapter = eg.install_adapter(m['adapter'])
    words, bank, packets = eg.load_inputs(m)
    receiver = build_receiver(m, m['receiver'], words)
    assert len(base.decoder_layers(receiver.model)) == m['n_blocks']
    dtype = str(next(receiver.model.parameters()).dtype)
    attn = getattr(receiver.model.config, '_attn_implementation', None)
    print('RECEIVER', m['receiver'], dtype, attn, 'adapter', json.dumps(adapter), 'phase', a.phase,
          'transformers', __import__('transformers').__version__, sys.executable, flush=True)
    blob = torch.load(Path(m['directions']), map_location='cpu', weights_only=False)
    assert blob['words'] == words and all(l in blob['layers'] for l in m['selected_layers'])
    order = {l: i for i, l in enumerate(blob['layers'])}
    dirs = blob['directions'].float()[[order[l] for l in m['selected_layers']]].to('cuda:0')
    arms = phase_arms(a.phase)
    scenes = eg.select_scenes(m, bank, a.phase, a.shard, a.shards)
    raw = root / 'raw' / f'{a.phase}_{m["receiver"]}_s{a.shard}.jsonl'
    raw.parent.mkdir(parents=True, exist_ok=True)
    seen = {r['key']: r for r in map(json.loads, raw.read_text().splitlines())} if raw.exists() else {}
    assert all(not r['engineering_checks'] or r['engineering_checks']['pass'] for r in seen.values())
    keyf = lambda r: '|'.join(str(r[f]) for f in ('scene', 'world', 'question', 'arm'))
    started = time.time(); computed = 0
    for i, scene in enumerate(scenes):
        expected = {f'{scene["scene"]}|{w}|{q}|{arm}' for w in (0, 1) for q in (0, 1) for arm in arms}
        if expected <= seen.keys():
            continue
        rows = scene_rows(receiver, scene, m, a.phase, packets, dirs, arms, fp32=(dtype == 'torch.float32'))
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
        print('SCENE', i + 1, len(scenes), 'rows', len(seen), f'{el / max(computed, 1):.3f} s/row', flush=True)
    want = len(scenes) * 4 * len(arms)
    assert len(seen) == want, (len(seen), want)
    elapsed = time.time() - started
    prior.atomic_json(raw.with_suffix('.done.json'),
                      dict(status='PASS', study=STUDY, phase=a.phase, receiver=m['receiver'], shard=a.shard, shards=a.shards,
                           rows=len(seen), scenes=len(scenes), arms=arms, rows_computed_this_process=computed,
                           manifest_sha256=eg.sha_file(a.manifest), raw_sha256=eg.sha_file(raw), elapsed_seconds=elapsed,
                           seconds_per_computed_row=(elapsed / computed if computed else None), adapter=adapter,
                           model_dtype=dtype, attn_implementation=attn, transformers=__import__('transformers').__version__,
                           python=sys.executable,
                           l255_world_mismatches=sum(1 for r in seen.values() if r.get('l255_world_identical') is False)))
    print('COMPLETE', m['receiver'], a.phase, a.shard, flush=True)


# ------------------------------------------------------------------------------------------------ dry run (CPU)

def dry_run(a):
    m = load_manifest(a.manifest)
    adapter = eg.install_adapter(m['adapter'])
    words, bank, packets = eg.load_inputs(m)
    receiver = eg.tokenizer_receiver(m, m['receiver'], words, adapter.get('installed', False))
    stats, problems, samples = Counter(), [], {}
    for phase in ('engineering', 'development'):
        arms = phase_arms(phase)
        for scene in eg.select_scenes(m, bank, phase, 0, 1):
            ni = prompt_inputs(receiver, scene, None)
            with instructed(receiver):
                nj = prompt_inputs(receiver, scene, None)
            tp, _, _, _ = receiver.tokenized(scene, ni['message'])
            with instructed(receiver):
                tj, _, _, _ = receiver.tokenized(scene, ni['message'])
            for qi in (0, 1):
                assert tj[qi].replace(INSTRUCTION, '', 1) == tp[qi] and tj[qi].count(INSTRUCTION) == 1
            stats['instructed_prompt_differs_only_by_instruction'] += 2
            assert [s['word'] for s in nj['slots']] == [s['word'] for s in ni['slots']]
            for wi in (0, 1):
                for arm in arms:
                    route, level, variant = parse_arm(arm)
                    if level is None:
                        continue
                    rec, packet, entries, words_obj, weights_obj = eg.packet_view(packets, level, scene, wi, words)
                    if route != 'text':
                        continue
                    extra = text_for(variant, scene, words_obj, weights_obj, rec)
                    ctx = instructed(receiver) if variant == 'instr' else contextlib.nullcontext()
                    with ctx:
                        texts, seqs, _, n = receiver.tokenized(scene, extra)
                    base_ni = nj if variant == 'instr' else ni
                    enc = receiver.tok(texts[0], return_offsets_mapping=True, add_special_tokens=not receiver.chat)
                    assert enc['input_ids'] == seqs[0]
                    lo, hi = message_block(enc, texts[0], extra)
                    if seqs[0][:lo] != base_ni['prefix'][:base_ni['lo']] or seqs[0][hi:n] != base_ni['prefix'][base_ni['hi']:]:
                        problems.append(f'{arm} {scene["scene"]}|{wi}: tokens outside the message block differ')
                    assert [s[n:] for s in seqs] == base_ni['suffixes']
                    if variant in ('pct', 'verbal'):
                        _, chars = render_fmt(variant, scene['aliases'], words_obj, weights_obj)
                        spans = token_spans(enc['offset_mapping'], chars, texts[0].index(extra))
                        for sp in spans:
                            piece = receiver.tok.decode([seqs[0][p] for p in sp['positions']])
                            assert sp['word'] in piece and '[' not in piece, (sp, piece)
                        if level == 0:
                            assert '[25%]' not in extra and 'weak' not in extra.split(': ', 1)[1]
                    samples.setdefault(f'{arm}', extra)
                    stats[f'text_prompts_checked:{variant}'] += 1
            stats[f'scenes_{phase}'] += 1
    report = dict(status='PASS_DRY_RUN' if not problems else 'DRY_RUN_PROBLEMS', study=STUDY, receiver=m['receiver'],
                  manifest_sha256=eg.sha_file(a.manifest), stats=dict(stats), problems=problems[:50], samples=samples,
                  sample_instructed_prompt=tj[0], transformers=__import__('transformers').__version__)
    text = json.dumps(report, indent=1, sort_keys=True) + '\n'
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True); Path(a.report).write_text(text)
    print(json.dumps(dict(status=report['status'], stats=report['stats'], problems=report['problems'][:5]), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    mk = sub.add_parser('make-manifest'); mk.add_argument('--receiver', required=True, choices=RECEIVERS)
    rn = sub.add_parser('run'); rn.add_argument('--manifest', type=Path, required=True)
    rn.add_argument('--phase', choices=('engineering', 'development'), required=True)
    rn.add_argument('--shard', type=int, default=0); rn.add_argument('--shards', type=int, default=1)
    dr = sub.add_parser('dry-run'); dr.add_argument('--manifest', type=Path, required=True); dr.add_argument('--report', type=Path)
    a = ap.parse_args()
    if a.cmd == 'make-manifest':
        make_manifest(a.receiver, OUT / a.receiver / 'manifest.json')
    elif a.cmd == 'run':
        run(a)
    else:
        dry_run(a)


if __name__ == '__main__':
    main()
