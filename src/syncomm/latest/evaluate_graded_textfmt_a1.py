"""Amendment 1 to the Experiment B text-format follow-up (gates/latest0924/expB_textfmt.lock.json, amendments[0]).

Adds, for the same receivers, scenes, packets, levels, questions, parser, budgets and numerics as F1-F3:
  textsorted_L{LLL}  F4: Experiment B's exact "[w/255]" message, but within each object the words are listed in
                     DESCENDING weight order (stable; ties keep ascending inventory order); every other byte identical
  texthedged_L{LLL}  F5: header "Selected words, each preceded by a qualifier expressing confidence (definitely,
                     probably, possibly, unlikely): ", filled slot "<qualifier> <word>" with 255 definitely,
                     191 probably, 128 possibly, 64 unlikely; unfilled slot "unknown" (no qualifier)
  emb_L{LLL}         embedding route: the internal route's neutral visible layout, but each active slot's INPUT
                     embedding is replaced by e_unknown + (q_w/q_max) * (e_symbol - e_unknown), e_symbol being the
                     receiver's own input embedding of the symbol in its field context (mean over its tokens; every
                     symbol here is a single token), as in src/syncomm/wordif/evaluate_localrep.py (emb1 at gamma 1)
  none; engineering only: identity_emb (neutral embeddings written back at all eight slots, exactly inert),
                     text_L064 and internal_L064 (continuity with Experiment B engineering rows, reported)
The locked F1-F3 runner (evaluate_graded_textfmt.py) is imported unchanged. Rows go to <receiver>/a1/raw/.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import torch  # noqa: E402

from syncomm.latest import evaluate_graded as eg  # noqa: E402
from syncomm.latest import evaluate_graded_textfmt as tf  # noqa: E402  locked F1-F3 runner, unchanged
from syncomm.wordif.evaluate_localrep import embed_hook, emb_rows  # noqa: E402  the paper's emb1 primitives
from syncomm.recovery.evaluate_depth import (  # noqa: E402
    WIDTH, build_receiver, write_hooks, capture_hooks, forward, prompt_inputs, entry_positions, fitted_states,
    message_block, prior, base, task, token_spans)

STUDY = 'LATEST0924_GRADED_B_TEXTFMT_A1'
LEVELS = eg.LEVELS
HEDGE = {255: 'definitely', 191: 'probably', 128: 'possibly', 64: 'unlikely'}
HEDGE_HEADER = 'Selected words, each preceded by a qualifier expressing confidence (definitely, probably, possibly, unlikely): '
DEV_ARMS = ['none'] + [f'{h}_L{L:03d}' for L in LEVELS for h in ('textsorted', 'texthedged', 'emb')]
ENG_EXTRA = ['identity_emb', 'text_L064', 'internal_L064']


def sorted_view(words_obj, weights_obj):
    out_w, out_q = [], []
    for ws, qs in zip(words_obj, weights_obj):
        order = sorted(range(len(ws)), key=lambda i: -qs[i])        # stable: ties keep ascending inventory order
        out_w.append([ws[i] for i in order]); out_q.append([qs[i] for i in order])
    return out_w, out_q


def render_hedged(aliases, object_words, object_weights, width=WIDTH):
    assert width in (2, 4) and len(aliases) == 2
    text, spans = HEDGE_HEADER, []
    for obj, (alias, words, weights) in enumerate(zip(aliases, object_words, object_weights)):
        assert len(words) <= width and len(weights) == len(words)
        if obj:
            text += '; '
        text += alias + ': '
        for index in range(width):
            if index:
                text += ', '
            if index < len(words):
                text += HEDGE[weights[index]] + ' '
                word = words[index]
            else:
                word = tf.NEUTRAL_WORD
            start = len(text)
            text += word
            spans.append(dict(object=obj, index=index, word=word, start=start, end=len(text)))
    text += '.'
    return text, spans


def parse_arm(arm):
    if arm in ('none', 'identity_emb'):
        return arm, None
    head, level = arm.split('_L'); level = int(level)
    assert level in LEVELS and arm == f'{head}_L{level:03d}', arm
    return head, level


def phase_arms(phase):
    return ['none'] + ENG_EXTRA + DEV_ARMS[1:] if phase == 'engineering' else list(DEV_ARMS)


def make_manifest(key):
    base_m = tf.OUT / key / 'manifest.json'
    m = json.loads(base_m.read_text())
    out = tf.OUT / key / 'a1' / 'manifest.json'
    pins = dict(m['pins'])
    for p in (Path(__file__).resolve(), ROOT / 'src/syncomm/latest/evaluate_graded_textfmt.py',
              ROOT / 'src/syncomm/wordif/evaluate_localrep.py', base_m):
        pins[str(p)] = eg.sha_file(p)
    m.update(study=STUDY, root=str(out.resolve().parent), arms=list(DEV_ARMS), engineering_arms=list(ENG_EXTRA),
             base_manifest=str(base_m), hedge_labels={str(k): v for k, v in HEDGE.items()}, hedge_header=HEDGE_HEADER,
             lock='gates/latest0924/expB_textfmt.lock.json (amendments[0])', row_split='latest0924_graded_textfmt_a1',
             pins=pins)
    text = json.dumps(m, indent=1, sort_keys=True) + '\n'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.read_text() != text:
        raise SystemExit(f'REFUSE: {out} exists with different content')
    out.write_text(text); print('MANIFEST', out)


def load_manifest(path):
    m = json.loads(Path(path).read_text())
    assert m['study'] == STUDY
    for p, want in m['pins'].items():
        assert eg.sha_file(p) == want, f'pinned input changed: {p}'
    return m


def text_for(head, scene, words_obj, weights_obj, rec):
    if head == 'text':
        extra = eg.render_weighted(scene['aliases'], words_obj, weights_obj, width=WIDTH)[0]
        assert extra == rec['text']; return extra
    if head == 'textsorted':
        w, q = sorted_view(words_obj, weights_obj)
        return eg.render_weighted(scene['aliases'], w, q, width=WIDTH)[0]
    return render_hedged(scene['aliases'], words_obj, weights_obj)[0]


@torch.inference_mode()
def scene_rows(receiver, scene, m, phase, packets, dirs, arms, fp32):
    layers = m['selected_layers']
    ni = prompt_inputs(receiver, scene, None)
    neutral_states = {}
    with capture_hooks(receiver.model, layers, neutral_states):
        neutral_out = forward(receiver, ni['prefix'])
    e_neutral = emb_rows(receiver, ni['prefix'])
    mapping = {(s['object'], s['index']): s for s in ni['slots']}
    all_slots = sorted(s['last_position'] for s in ni['slots'])
    records, zero_ref, l255_ref = [], {}, {}
    for wi, world in enumerate(scene['worlds']):
        for arm in arms:
            head, level = parse_arm(arm)
            report, states, etargets, slot_edits = {}, None, None, None
            ids, suffixes, extra, rec, active, write_positions = ni['prefix'], ni['suffixes'], None, None, [], []
            single_token = None
            if level is not None:
                rec, packet, entries, words_obj, weights_obj = eg.packet_view(packets, level, scene, wi, receiver.words)
                active, write_positions, _ = entry_positions(entries, ni['slots'])
            if head == 'none':
                out = neutral_out
            elif head in ('text', 'textsorted', 'texthedged'):
                extra = text_for(head, scene, words_obj, weights_obj, rec)
                _, seqs, _, n = receiver.tokenized(scene, extra)
                ids, suffixes = seqs[0][:n], [s[n:] for s in seqs]
                out = forward(receiver, ids)
            elif head == 'internal':
                neutral = {l: neutral_states[l] for l in layers}
                states = fitted_states(neutral, layers, dirs, [1.0] * len(layers), active, mapping, m['gamma'])
                with write_hooks(receiver.model, layers, states, write_positions, report, None):
                    out = forward(receiver, ids)
            elif head == 'identity_emb':
                etargets = {p: e_neutral[p].clone() for p in all_slots}
                write_positions = all_slots
                with embed_hook(receiver.model, etargets, report):
                    out = forward(receiver, ids)
            elif head == 'emb':
                wbo = [[e['word'] for e in sorted(active, key=lambda x: x['index']) if e['object'] == o] for o in (0, 1)]
                si = prompt_inputs(receiver, scene, wbo)             # the same slots with the real words: field context
                assert len(si['prefix']) == len(ni['prefix'])
                src = {(s['object'], s['index']): s for s in si['slots']}
                e_src = emb_rows(receiver, si['prefix'])
                counts = {(e['object'], e['index']): len(src[e['object'], e['index']]['positions']) for e in active}
                single_token = all(v == 1 for v in counts.values())
                etargets = {}
                for e in active:
                    pos = mapping[e['object'], e['index']]['last_position']
                    word_vec = e_src[src[e['object'], e['index']]['positions']].mean(0)
                    etargets[pos] = e_neutral[pos] + e['relative_weight'] * (word_vec - e_neutral[pos])
                with embed_hook(receiver.model, etargets, report):
                    out = forward(receiver, ids)
            else:
                raise AssertionError(arm)
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
                    if states is not None:
                        ctx = write_hooks(receiver.model, layers, states, write_positions, {}, None)
                    elif etargets is not None:
                        ctx = embed_hook(receiver.model, etargets, {})
                    else:
                        ctx = contextlib.nullcontext()
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
                           split=m['row_split'], receiver=m['receiver'], source='graded_textfmt_a1', arm=arm, route=head,
                           level=level, layers=layers, x_field=(rec['x_field'] if rec else scene['changed_field']),
                           entries=active, words_by_object=(rec['words_by_object'] if rec else None),
                           weights_by_object=(rec['weights_by_object'] if rec else None),
                           condition_packet_sha256=(rec['sha256'] if rec else None),
                           wire_bytes=(len(extra.encode()) if extra is not None else 0 if head in ('none', 'identity_emb') else rec['bytes']),
                           extra_text=extra, prefix_ids=ids, suffix_ids=suffix, slots=ni['slots'],
                           patch_positions=(write_positions if (states is not None or etargets is not None) else []),
                           patch_report=report, single_token_facts=single_token, prefix_cache_sha256=fp,
                           l255_world_identical=l255_identical, engineering_checks=checks, **value)
                row.update({f'score{b}': task.score_answer(value[f'text{b}'], scene['aliases'], gold) for b in (64, 256)})
                if head == 'none':
                    z = (fp, value['logits_sha256'], value['tokens'])
                    assert zero_ref.setdefault(qi, z) == z, 'the no-message prefix depends on the world'
                if head == 'identity_emb':
                    assert (fp, value['logits_sha256'], value['tokens']) == zero_ref[qi], 'identity_emb is not inert'
                    assert report and all(v['edit_norm'] <= 1e-6 for v in report.values())
                records.append(row)
            assert prior.cache_fingerprint(cache) == fp
    return records


def run(a):
    m = load_manifest(a.manifest)
    root = Path(m['root'])
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    adapter = eg.install_adapter(m['adapter'])
    words, bank, packets = eg.load_inputs(m)
    receiver = build_receiver(m, m['receiver'], words)
    assert len(base.decoder_layers(receiver.model)) == m['n_blocks']
    dtype = str(next(receiver.model.parameters()).dtype)
    print('RECEIVER', m['receiver'], dtype, 'adapter', json.dumps(adapter), 'phase', a.phase, sys.executable, flush=True)
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
        print('SCENE', i + 1, len(scenes), 'rows', len(seen), f'{(time.time() - started) / max(computed, 1):.3f} s/row', flush=True)
    want = len(scenes) * 4 * len(arms)
    assert len(seen) == want, (len(seen), want)
    elapsed = time.time() - started
    prior.atomic_json(raw.with_suffix('.done.json'),
                      dict(status='PASS', study=STUDY, phase=a.phase, receiver=m['receiver'], shard=a.shard, shards=a.shards,
                           rows=len(seen), scenes=len(scenes), arms=arms, manifest_sha256=eg.sha_file(a.manifest),
                           raw_sha256=eg.sha_file(raw), elapsed_seconds=elapsed, adapter=adapter, model_dtype=dtype,
                           python=sys.executable, transformers=__import__('transformers').__version__,
                           l255_world_mismatches=sum(1 for r in seen.values() if r.get('l255_world_identical') is False)))
    print('COMPLETE', m['receiver'], a.phase, a.shard, flush=True)


def dry_run(a):
    m = load_manifest(a.manifest)
    adapter = eg.install_adapter(m['adapter'])
    words, bank, packets = eg.load_inputs(m)
    receiver = eg.tokenizer_receiver(m, m['receiver'], words, adapter.get('installed', False))
    stats, problems, samples = Counter(), [], {}
    for phase in ('engineering', 'development'):
        for scene in eg.select_scenes(m, bank, phase, 0, 1):
            ni = prompt_inputs(receiver, scene, None)
            for wi in (0, 1):
                for arm in phase_arms(phase):
                    head, level = parse_arm(arm)
                    if level is None:
                        continue
                    rec, packet, entries, words_obj, weights_obj = eg.packet_view(packets, level, scene, wi, words)
                    active, positions, _ = entry_positions(entries, ni['slots'])
                    if head == 'emb':
                        wbo = [[e['word'] for e in sorted(active, key=lambda x: x['index']) if e['object'] == o] for o in (0, 1)]
                        si = prompt_inputs(receiver, scene, wbo)
                        if len(si['prefix']) != len(ni['prefix']):
                            problems.append(f'emb field-context length differs {scene["scene"]}|{wi}|{arm}')
                        src = {(s['object'], s['index']): s for s in si['slots']}
                        n1 = sum(len(src[e['object'], e['index']]['positions']) == 1 for e in active)
                        stats['emb_single_token_facts'] += n1; stats['emb_facts'] += len(active)
                        continue
                    if head not in ('text', 'textsorted', 'texthedged'):
                        continue
                    extra = text_for(head, scene, words_obj, weights_obj, rec)
                    texts, seqs, _, n = receiver.tokenized(scene, extra)
                    enc = receiver.tok(texts[0], return_offsets_mapping=True, add_special_tokens=not receiver.chat)
                    assert enc['input_ids'] == seqs[0]
                    lo, hi = message_block(enc, texts[0], extra)
                    if seqs[0][:lo] != ni['prefix'][:ni['lo']] or seqs[0][hi:n] != ni['prefix'][ni['hi']:]:
                        problems.append(f'{arm} {scene["scene"]}|{wi}: tokens outside the message block differ')
                    assert [s[n:] for s in seqs] == ni['suffixes']
                    if head == 'textsorted':
                        w, q = sorted_view(words_obj, weights_obj)
                        chars = eg.render_weighted(scene['aliases'], w, q, width=WIDTH)[1]
                        assert all(qs == sorted(qs, reverse=True) for qs in q)
                        if level in (0, 255):
                            assert extra == rec['text']; stats['sorted_equals_expB_text_at_L0_L255'] += 1
                    elif head == 'texthedged':
                        chars = render_hedged(scene['aliases'], words_obj, weights_obj)[1]
                    else:
                        chars = eg.render_weighted(scene['aliases'], words_obj, weights_obj, width=WIDTH)[1]
                    spans = token_spans(enc['offset_mapping'], chars, texts[0].index(extra))
                    for sp in spans:
                        piece = receiver.tok.decode([seqs[0][p] for p in sp['positions']])
                        assert sp['word'] in piece and '[' not in piece, (sp, piece)
                    samples.setdefault(arm, extra)
                    stats[f'text_prompts_checked:{head}'] += 1
            stats[f'scenes_{phase}'] += 1
    report = dict(status='PASS_DRY_RUN' if not problems else 'DRY_RUN_PROBLEMS', study=STUDY, receiver=m['receiver'],
                  manifest_sha256=eg.sha_file(a.manifest), stats=dict(stats), problems=problems[:50], samples=samples)
    text = json.dumps(report, indent=1, sort_keys=True) + '\n'
    if a.report:
        Path(a.report).write_text(text)
    print(json.dumps(dict(status=report['status'], stats=report['stats'], problems=report['problems'][:5]), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    mk = sub.add_parser('make-manifest'); mk.add_argument('--receiver', required=True, choices=tf.RECEIVERS)
    rn = sub.add_parser('run'); rn.add_argument('--manifest', type=Path, required=True)
    rn.add_argument('--phase', choices=('engineering', 'development'), required=True)
    rn.add_argument('--shard', type=int, default=0); rn.add_argument('--shards', type=int, default=1)
    dr = sub.add_parser('dry-run'); dr.add_argument('--manifest', type=Path, required=True); dr.add_argument('--report', type=Path)
    a = ap.parse_args()
    if a.cmd == 'make-manifest':
        make_manifest(a.receiver)
    elif a.cmd == 'run':
        run(a)
    else:
        dry_run(a)


if __name__ == '__main__':
    main()
