"""R1/R2 runner: native fact-state transplantation versus fitted word-direction writing over a frozen set of write
depths, into ONE common target prefix (the study protocol §5-§7).

Every condition for a scene/world/question shares the same neutral target prefix (the public message with four slots
per object, every slot holding the neutral word), the same target positions (each active slot's word-final token) and
the same question suffixes. Only what is written there differs:

  none                    nothing — the shared no-message row (identical across worlds by construction)
  identity_write          the captured neutral state written back at the same sites — an active-hook inertness gate
  packet_text_padded      no write; the prefix carries the real words in those same eight slots
  packet_text_compact     no write; the packet-as-text convention of the earlier packages (only carried words listed)
  full_text               no write; the world's evidence sentences verbatim
  native                  the fact words' own word-final states, read from a source prompt built by the SAME renderer
                          with the same slots, transplanted by (object, slot) identity at their natural scale
  fitted,g                h_neutral + g * relative_weight * d[layer, address]            (raw word-contrast direction)
  local,g                 h_neutral + g * c[layer] * relative_weight * d[layer, address] (R2 generic local scale)
  fitted_matchnative      raw fitted g=1 with each realized per-layer edit norm rescaled to the native edit's norm
  native_matchfitted      native transplantation with each realized per-layer edit norm rescaled to raw fitted g=1

Declared alignment: a real fact word is usually several tokens while the neutral filler ` unknown` is one, so the
source prompt is LONGER than the target prefix. Both prompts come from the same renderer with the same slot structure,
are token-identical outside the message block (asserted per scene), and the transplant is by (object, slot index)
identity onto each slot's word-final token — never a guessed index. Both token lengths, the per-slot token counts and
every realized edit norm, residual norm and delta/state cosine are recorded per row. No claim of byte-identical
source contexts is made.

Messages here are complete-fact (literal) packets only; automatic packets enter at R3.
"""
import argparse
import contextlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
FIT = ROOT / 'artifacts/mechanism0920/word_memory_fit_v1/snapshot'
sys.path.insert(0, str(FIT))
_spec = importlib.util.spec_from_file_location('frozen_framing', ROOT / 'artifacts/mechanism0920/framing_reference_v1/snapshot/framing_reference.py')
parent = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(parent)
prior, base = parent.prior, parent.base
task = prior.task
from memory_layout import render, token_spans, packet_entries   # noqa: E402
from memory_targets import targets as frozen_targets            # noqa: E402

WIDTH = 4


def build_receiver(m, key, words):
    """The frozen Receiver without the old sparse writer's probe dictionary/magnitude, which this campaign never uses."""
    r = prior.Receiver.__new__(prior.Receiver)
    r.model, r.tok = base.load_model(m['models'][key])
    r.chat = key not in m['base_models']
    r.task = parent.Scaffold
    r.words = list(words)
    r.cache = {}
    r.key = key
    r.band = sorted(set(base.tap_layer(r.model, x) for x in (.4, .5, .6)))
    return r


@contextlib.contextmanager
def write_hooks(model, layers, states, positions, report, norm_targets=None):
    handles = []
    for layer in layers:
        def hook(module, args, output, layer=layer):
            h = output[0] if isinstance(output, tuple) else output
            updated = h.clone()
            want = states[layer].to(h.dtype).to(h.device)
            delta = want[:, positions] - h[:, positions]
            raw = float(delta.float().norm())
            req = None if norm_targets is None else norm_targets.get(str(layer))
            if req is None:
                updated[:, positions] = want[:, positions]
            else:
                assert raw > 0 or req == 0
                updated[:, positions] = h[:, positions] + (req / raw if raw > 0 else 0.) * delta
            realized = float((updated[:, positions] - h[:, positions]).float().norm())
            if req is not None:
                assert abs(realized - req) <= max(1e-4, 1e-4 * req), (realized, req)
            report[str(layer)] = dict(positions=len(positions), norm=realized, raw_norm=raw, requested_norm=req,
                                      residual_norm=float(h[:, positions].float().norm()),
                                      delta_state_cosine=float(torch.nn.functional.cosine_similarity(
                                          delta.float().flatten(), want[:, positions].float().flatten(), dim=0)),
                                      target_sha256=prior.digest_tensor(want[:, positions]))
            return (updated,) + output[1:] if isinstance(output, tuple) else updated
        handles.append(base.decoder_layers(model)[layer].register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def capture_hooks(model, layers, store):
    handles = []
    for layer in layers:
        def hook(module, args, output, layer=layer):
            h = output[0] if isinstance(output, tuple) else output
            store[layer] = h.detach().clone()
            return output
        handles.append(base.decoder_layers(model)[layer].register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def forward(receiver, ids, use_cache=True):
    t = torch.tensor([ids], device='cuda:0')
    return receiver.model(t, attention_mask=torch.ones_like(t), use_cache=use_cache)


def message_block(enc, text, message):
    start = text.index(message)
    end = start + len(message)
    pos = [i for i, (a, b) in enumerate(enc['offset_mapping']) if a < end and b > start]
    assert pos == list(range(pos[0], pos[-1] + 1))
    return pos[0], pos[-1] + 1


def prompt_inputs(receiver, scene, object_words):
    """The prompt built by the frozen renderer with `object_words` in the slots (None = every slot neutral)."""
    message, chars = render(scene['aliases'], object_words, width=WIDTH)
    texts, seqs, sites, n = receiver.tokenized(scene, message)
    enc = receiver.tok(texts[0], return_offsets_mapping=True, add_special_tokens=not receiver.chat)
    assert enc['input_ids'] == seqs[0]
    spans = token_spans(enc['offset_mapping'], chars, texts[0].index(message))
    assert len(spans) == 2 * WIDTH and all(s['last_position'] < n for s in spans)
    lo, hi = message_block(enc, texts[0], message)
    return dict(message=message, prefix=seqs[0][:n], suffixes=[s[n:] for s in seqs], slots=spans,
                sites=sites, n=n, lo=lo, hi=hi)


def entry_positions(entries, slots):
    mapping = {(s['object'], s['index']): s for s in slots}
    active = [e for e in entries if e['weight'] > 0]
    positions = sorted(mapping[e['object'], e['index']]['last_position'] for e in active)
    assert len(set(positions)) == len(positions)
    return active, positions, mapping


def fitted_states(neutral, layers, dirs, scale, active, mapping, gamma):
    out = {layer: neutral[layer].clone() for layer in layers}
    for li, layer in enumerate(layers):
        for e in active:
            pos = mapping[e['object'], e['index']]['last_position']
            out[layer][0, pos] += (gamma * scale[li] * e['relative_weight']) * dirs[li, e['address']].to(out[layer])
    return out


def configurations(m, phase):
    grid = m['gammas']
    shared = dict(label='shared', depths=[], layers=[],
                  arms=[('none', 0), ('packet_text_padded', 0), ('packet_text_compact', 0), ('full_text', 0)])
    cfgs = [shared]
    for label, depths, layers in m['configs']:
        arms = [('native', 0)] + [('fitted', g) for g in grid]
        if label in (m.get('local_scale_labels') or []):
            arms += [('local', g) for g in (m.get('local_gammas') or grid)]
        if label in (m.get('match_labels') or []):
            arms += [('fitted_matchnative', 1), ('native_matchfitted', 0)]
        if phase == 'engineering':
            arms = [('identity_write', 0)] + arms
        cfgs.append(dict(label=label, depths=depths, layers=list(layers), arms=arms))
    return cfgs


@torch.inference_mode()
def scene_rows(receiver, scene, m, phase, cfgs, dirs_by_label, scale_by_layer):
    ni = prompt_inputs(receiver, scene, None)
    all_layers = sorted({l for c in cfgs for l in c['layers']})
    neutral_states = {}
    with capture_hooks(receiver.model, all_layers, neutral_states):
        neutral_out = forward(receiver, ni['prefix'])
    records, zero_ref = [], {}
    for wi, world in enumerate(scene['worlds']):
        packet = prior.literal_packet(world, receiver.words)
        entries = packet_entries(packet, receiver.words)
        active, positions, mapping = entry_positions(entries, ni['slots'])
        words_by_obj = [[e['word'] for e in sorted(active, key=lambda x: x['index']) if e['object'] == o] for o in (0, 1)]
        si = prompt_inputs(receiver, scene, words_by_obj)
        src = {(s['object'], s['index']): s for s in si['slots']}
        src_states = {}
        with capture_hooks(receiver.model, all_layers, src_states):
            forward(receiver, si['prefix'], use_cache=False)
        nat = {}
        for layer in all_layers:
            t = neutral_states[layer].clone()
            for e in active:
                t[0, mapping[e['object'], e['index']]['last_position']] = src_states[layer][0, src[e['object'], e['index']]['last_position']]
            nat[layer] = t
        outside_ok = (ni['prefix'][:ni['lo']] == si['prefix'][:si['lo']] and ni['prefix'][ni['hi']:] == si['prefix'][si['hi']:])
        assert outside_ok, 'source and target prompts differ outside the message block'
        alignment = dict(target_tokens=len(ni['prefix']), source_tokens=len(si['prefix']),
                         target_message_tokens=ni['hi'] - ni['lo'], source_message_tokens=si['hi'] - si['lo'],
                         identical_outside_message=True,
                         source_slot_token_counts={f"{e['object']}.{e['index']}": len(src[e['object'], e['index']]['positions']) for e in active},
                         target_slot_token_counts={f"{e['object']}.{e['index']}": len(mapping[e['object'], e['index']]['positions']) for e in active})
        native_norms, fitted_norms = {}, {}
        for cfg in cfgs:
            layers = cfg['layers']
            dirs = dirs_by_label.get(cfg['label'])
            for arm, gamma in cfg['arms']:
                report, states, nt = {}, None, None
                ids, suffixes, extra = ni['prefix'], ni['suffixes'], None
                if arm == 'none':
                    out = neutral_out
                elif arm in ('packet_text_padded', 'packet_text_compact', 'full_text'):
                    extra = (' '.join(world['evidence']) if arm == 'full_text' else
                             render(scene['aliases'], words_by_obj, width=WIDTH)[0] if arm == 'packet_text_padded' else
                             base.packet_text(packet, receiver.words, scene['aliases'], weights=True))
                    _, seqs, _, n = receiver.tokenized(scene, extra)
                    ids, suffixes = seqs[0][:n], [s[n:] for s in seqs]
                    out = forward(receiver, ids)
                else:
                    if arm == 'identity_write':
                        states = {l: neutral_states[l] for l in layers}
                    elif arm == 'native':
                        states = {l: nat[l] for l in layers}
                    elif arm == 'native_matchfitted':
                        states, nt = {l: nat[l] for l in layers}, fitted_norms.get(cfg['label'])
                        assert nt is not None, 'fitted gamma=1 must precede native_matchfitted'
                    else:
                        sc = [1.0] * len(layers) if arm != 'local' else [scale_by_layer[str(l)] for l in layers]
                        states = fitted_states({l: neutral_states[l] for l in layers}, layers, dirs, sc, active, mapping, gamma)
                        if arm == 'fitted_matchnative':
                            nt = native_norms.get(cfg['label'])
                            assert nt is not None, 'native must precede fitted_matchnative'
                    with write_hooks(receiver.model, layers, states, positions, report, nt):
                        out = forward(receiver, ids)
                if arm == 'native':
                    native_norms[cfg['label']] = {k: v['norm'] for k, v in report.items()}
                if arm == 'fitted' and gamma == 1:
                    fitted_norms[cfg['label']] = {k: v['norm'] for k, v in report.items()}
                cache = out.past_key_values
                fp = prior.cache_fingerprint(cache)
                for qi, suffix in enumerate(suffixes):
                    branch = prior.clone_cache(cache)
                    assert prior.cache_fingerprint(branch) == fp
                    tensor = torch.tensor([suffix], device='cuda:0')
                    end = receiver.model(tensor, past_key_values=branch, use_cache=True,
                                         attention_mask=torch.ones((1, len(ids) + len(suffix)), device='cuda:0', dtype=torch.long))
                    logits = end.logits[0, -1].float()
                    value = receiver.decode(logits, end.past_key_values, scene, 256)
                    gold = world['answers'][qi]
                    checks = None
                    if phase == 'engineering':
                        ctx = write_hooks(receiver.model, layers, states, positions, {}, nt) if states is not None else contextlib.nullcontext()
                        with ctx:
                            full = forward(receiver, ids + suffix)
                        direct = receiver.decode(full.logits[0, -1].float(), full.past_key_values, scene, 256)
                        checks = dict(max_logit_difference=float((full.logits[0, -1].float() - logits).abs().max()),
                                      alias_gap_difference=abs(direct['gap'] - value['gap']),
                                      parsed_equal=all(task.score_answer(direct[f'text{b}'], scene['aliases'], gold) ==
                                                       task.score_answer(value[f'text{b}'], scene['aliases'], gold) for b in (64, 256)))
                        checks['pass'] = checks['max_logit_difference'] <= .05 and checks['alias_gap_difference'] <= .1 and checks['parsed_equal']
                    row = dict(scene=scene['scene'], family=scene['family'], domain=scene['domain'], template=scene['template'],
                               world=wi, question=qi, aliases=scene['aliases'], gold=gold, split=m['row_split'],
                               receiver=m['receiver'], source='literal', arm=arm, depth_label=cfg['label'],
                               depths=cfg['depths'], layers=layers, gamma=gamma, width=WIDTH, layout='original',
                               question_style='original', entries=active, packet_sha256=base.sha(base.serialize(packet)),
                               wire_bytes=len(extra.encode()) if extra is not None else (0 if arm in ('none', 'identity_write') else len(base.serialize(packet))),
                               extra_text=extra, prefix_ids=ids, suffix_ids=suffix, slots=ni['slots'],
                               patch_positions=positions if states is not None else [], patch_report=report,
                               alignment=alignment, prefix_cache_sha256=fp, branch_initial_cache_sha256=fp,
                               engineering_checks=checks, **value)
                    row.update({f'score{b}': task.score_answer(value[f'text{b}'], scene['aliases'], gold) for b in (64, 256)})
                    if arm == 'none':
                        z = (fp, value['logits_sha256'], value['tokens'])
                        assert zero_ref.setdefault(qi, z) == z, 'no-message prefix depends on the world'
                    if arm == 'identity_write':
                        assert (fp, value['logits_sha256'], value['tokens']) == zero_ref[qi], 'identity write is not inert'
                        assert report and all(v['norm'] <= 1e-6 for v in report.values())
                    records.append(row)
                assert prior.cache_fingerprint(cache) == fp
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--phase', choices=('engineering', 'development'), required=True)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--shards', type=int, default=1)
    a = ap.parse_args()
    m = json.loads(a.manifest.read_text()); root = Path(m['root'])
    for path, want in m['pins'].items():
        assert base.sha(Path(path).read_bytes()) == want, path
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    words = json.loads(Path(m['words_file']).read_text())
    receiver = build_receiver(m, m['receiver'], words)
    assert len(base.decoder_layers(receiver.model)) == m['n_blocks']
    blob = torch.load(Path(m['directions']), map_location='cpu', weights_only=False)
    assert blob['words'] == words and blob['layers'] == m['all_layers']
    D = blob['directions'].float().to('cuda:0')
    order = {l: i for i, l in enumerate(blob['layers'])}
    cfgs = configurations(m, a.phase)
    dirs_by_label = {c['label']: (D[[order[l] for l in c['layers']]] if c['layers'] else None) for c in cfgs}
    bank = json.loads(Path(m['bank']).read_text())
    assert len(bank) == m['bank_size'] and all(s['split'] == m['bank_split'] for s in bank)
    scenes = ([s for s in bank if s['scene'] in m['engineering_scene_ids']] if a.phase == 'engineering'
              else [s for s in bank if s['scene'] not in m['engineering_scene_ids']][a.shard::a.shards])
    raw = root / 'raw' / f'{a.phase}_{m["receiver"]}_s{a.shard}.jsonl'; raw.parent.mkdir(parents=True, exist_ok=True)
    seen = {r['key']: r for r in map(json.loads, raw.read_text().splitlines())} if raw.exists() else {}
    assert all(not r['engineering_checks'] or r['engineering_checks']['pass'] for r in seen.values())
    keyf = lambda r: '|'.join(str(r[f]) for f in ('scene', 'world', 'question', 'arm', 'depth_label', 'gamma'))
    started = time.time()
    for index, scene in enumerate(scenes):
        expected = {f'{scene["scene"]}|{w}|{q}|{arm}|{c["label"]}|{g}' for w in (0, 1) for q in (0, 1) for c in cfgs for arm, g in c['arms']}
        if expected <= seen.keys():
            continue
        rows = scene_rows(receiver, scene, m, a.phase, cfgs, dirs_by_label, m['scale'])
        for row in rows:
            row['key'] = keyf(row); row['phase'] = a.phase
            if row['key'] in seen:
                assert all(row[f] == seen[row['key']][f] for f in ('logits_sha256', 'tokens'))
            else:
                with raw.open('a') as h:
                    h.write(json.dumps(row) + '\n')
                seen[row['key']] = row
        assert all(not r['engineering_checks'] or r['engineering_checks']['pass'] for r in rows), 'engineering failure retained'
        print('SCENE', index + 1, len(scenes), 'rows', len(seen), flush=True)
    expected_rows = len(scenes) * 4 * sum(len(c['arms']) for c in cfgs)
    assert len(seen) == expected_rows, (len(seen), expected_rows)
    prior.atomic_json(raw.with_suffix('.done.json'),
                      dict(status='PASS', phase=a.phase, receiver=m['receiver'], shard=a.shard, shards=a.shards,
                           rows=len(seen), scenes=len(scenes), conditions=sum(len(c['arms']) for c in cfgs),
                           manifest_sha256=base.sha(a.manifest.read_bytes()), raw_sha256=base.sha(raw.read_bytes()),
                           elapsed_seconds=time.time() - started))
    print('COMPLETE', m['receiver'], a.phase, a.shard, flush=True)


if __name__ == '__main__':
    main()
