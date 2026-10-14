"""Diagnostics D1 (cache-read restriction) and D2 (matched-layout direction); gates/latest0924/diag_protocol.lock.json.

Complete-fact packets on the exposed development bank, the frozen R1 prompt/renderer/hooks/decoder from
src/syncomm/recovery/evaluate_depth.py (imported, not copied). What this runner adds:

  D1  <src>_early               write at the d20 layer, prefix cache kept as computed                     (= arm A)
      <src>_write@<L>           write at layer L                                                          (= arm B)
      <src>_early_cut@<L>       A's prefill, then the prefix keys/values of every layer <= L replaced by the
                                neutral prefix's keys/values; question tokens run normally on that cache   (= arm C)
  D2  matched_write@<L>         per scene and fact, a direction computed in the scene's OWN prompt: the neutral
                                prompt with only that fact word in its slot, minus the same token sequence with the
                                word's tokens replaced one-for-one by the neutral filler token; read at the word-final
                                token of layer L and written like the fixed direction (gamma 1)
      fixed_write@<L>, native_write@<L> and text are the other representations of the D2 ladder.

Built-in checks (asserted on every scene): C at the d20 layer is byte-identical to A (a write at the output of layer
d20 cannot change keys/values at layers <= d20); C at the last layer reproduces the no-message answer exactly; the
identity write is inert; in the engineering phase the write arms also pass the frozen cached-vs-complete-forward check.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
sys.path.insert(0, str(ROOT / 'src'))
from syncomm.recovery.evaluate_depth import (  # noqa: E402
    WIDTH, build_receiver, write_hooks, capture_hooks, forward, prompt_inputs, entry_positions,
    prior, base, task, render, packet_entries)


def restricted_cache(early, neutral, cut):
    """A copy of `early` whose layers 0..cut hold the neutral prefix's keys/values (deep copies)."""
    out = prior.clone_cache(early)
    assert len(out.layers) == len(neutral.layers)
    for i in range(cut + 1):
        for name in ('keys', 'values'):
            src = getattr(neutral.layers[i], name)
            assert getattr(out.layers[i], name).shape == src.shape
            setattr(out.layers[i], name, src.clone())
    return out


def matched_directions(receiver, scene, active, layers, unknown_id):
    """{(object, index): {layer: vector}} computed in the scene's own public prompt, one fact word at a time."""
    result = {}
    for e in active:
        words = [[], []]
        words[e['object']] = ['unknown'] * e['index'] + [e['word']]
        si = prompt_inputs(receiver, scene, words)
        slot = next(s for s in si['slots'] if s['object'] == e['object'] and s['index'] == e['index'])
        assert slot['word'] == e['word'] if 'word' in slot else True
        ids_word = list(si['prefix'])
        ids_neutral = list(ids_word)
        for p in slot['positions']:
            ids_neutral[p] = unknown_id
        got_w, got_n = {}, {}
        with capture_hooks(receiver.model, layers, got_w):
            forward(receiver, ids_word, use_cache=False)
        with capture_hooks(receiver.model, layers, got_n):
            forward(receiver, ids_neutral, use_cache=False)
        pos = slot['last_position']
        result[e['object'], e['index']] = {l: (got_w[l][0, pos] - got_n[l][0, pos]).float() for l in layers}
    return result


def states_from(neutral, layer, active, mapping, vectors, gamma=1.0, replace=None):
    """neutral + gamma * relative_weight * vector at each active position; with `replace`, the position's state is
    set to replace(e) exactly (the frozen native transplant), never reconstructed as neutral + difference."""
    t = neutral[layer].clone()
    for e in active:
        pos = mapping[e['object'], e['index']]['last_position']
        if replace is not None:
            t[0, pos] = replace(e).to(t)
        else:
            t[0, pos] += (gamma * e['relative_weight']) * vectors(e).to(t)
    return {layer: t}


@torch.inference_mode()
def scene_rows(receiver, scene, m, phase, D, order):
    L = m['layers']
    d20, n_blocks = L['d20'], m['n_blocks']
    ni = prompt_inputs(receiver, scene, None)
    unknown_id = ni['prefix'][ni['slots'][0]['last_position']]
    assert len(ni['slots'][0]['positions']) == 1, 'the neutral filler must be one token'
    need = sorted({L[k] for k in ('d20', 'd35', 'd50', 'd65')})
    neutral_states = {}
    with capture_hooks(receiver.model, need, neutral_states):
        neutral_out = forward(receiver, ni['prefix'])
    ncache = neutral_out.past_key_values
    nfp = prior.cache_fingerprint(ncache)
    records, zero_ref = [], {}
    for wi, world in enumerate(scene['worlds']):
        packet = prior.literal_packet(world, receiver.words)
        entries = packet_entries(packet, receiver.words)
        active, positions, mapping = entry_positions(entries, ni['slots'])
        words_by_obj = [[e['word'] for e in sorted(active, key=lambda x: x['index']) if e['object'] == o] for o in (0, 1)]
        si = prompt_inputs(receiver, scene, words_by_obj)
        src = {(s['object'], s['index']): s for s in si['slots']}
        src_states = {}
        with capture_hooks(receiver.model, need, src_states):
            forward(receiver, si['prefix'], use_cache=False)
        assert ni['prefix'][:ni['lo']] == si['prefix'][:si['lo']] and ni['prefix'][ni['hi']:] == si['prefix'][si['hi']:]
        native_src = lambda layer: (lambda e: src_states[layer][0, src[e['object'], e['index']]['last_position']])
        fixed_vec = lambda layer: (lambda e: D[order[layer], e['address']])
        matched = matched_directions(receiver, scene, active, m['d2_layers'], unknown_id) if m['d2_layers'] else {}
        matched_vec = lambda layer: (lambda e: matched[e['object'], e['index']][layer])
        vec = {'fitted': fixed_vec, 'fixed': fixed_vec, 'native': None, 'matched': matched_vec}
        early_caches = {}

        def run(arm, layer=None, cut=None, source=None, diagnostic='reference'):
            report, states, extra, ids, suffixes = {}, None, None, ni['prefix'], ni['suffixes']
            if arm == 'none':
                cache, fp = ncache, nfp
            elif arm == 'text':
                extra = render(scene['aliases'], words_by_obj, width=WIDTH)[0]
                _, seqs, _, n = receiver.tokenized(scene, extra)
                ids, suffixes = seqs[0][:n], [s[n:] for s in seqs]
                out = forward(receiver, ids); cache = out.past_key_values; fp = prior.cache_fingerprint(cache)
            elif arm == 'identity_write':
                states = {d20: neutral_states[d20]}
                with write_hooks(receiver.model, [d20], states, positions, report):
                    out = forward(receiver, ids)
                cache = out.past_key_values; fp = prior.cache_fingerprint(cache)
                assert fp == nfp, 'identity write is not inert'
            else:
                if source not in early_caches or cut is None:
                    wl = d20 if cut is not None else layer
                    states = (states_from(neutral_states, wl, active, mapping, None, replace=native_src(wl)) if source == 'native'
                              else states_from(neutral_states, wl, active, mapping, vec[source](wl)))
                    with write_hooks(receiver.model, [wl], states, positions, report):
                        out = forward(receiver, ids)
                    cache = out.past_key_values; fp = prior.cache_fingerprint(cache)
                    if wl == d20:
                        early_caches[source] = (cache, fp, report, states)
                if cut is not None:
                    early, efp, report, states = early_caches[source]
                    cache = restricted_cache(early, ncache, cut); fp = prior.cache_fingerprint(cache)
                    if cut == d20:
                        assert fp == efp, 'cut at the write layer must reproduce the unrestricted cache'
                    if cut == n_blocks - 1:
                        assert fp == nfp, 'cut at the last layer must reproduce the no-message cache'
                    layer = d20
            rows = []
            for qi, suffix in enumerate(suffixes):
                branch = prior.clone_cache(cache)
                assert prior.cache_fingerprint(branch) == fp
                end = receiver.model(torch.tensor([suffix], device='cuda:0'), past_key_values=branch, use_cache=True,
                                     attention_mask=torch.ones((1, len(ids) + len(suffix)), device='cuda:0', dtype=torch.long))
                value = receiver.decode(end.logits[0, -1].float(), end.past_key_values, scene, 256)
                gold = world['answers'][qi]
                checks = None
                if phase == 'engineering' and states is not None and cut is None:
                    with write_hooks(receiver.model, list(states), states, positions, {}):
                        full = forward(receiver, ids + suffix)
                    direct = receiver.decode(full.logits[0, -1].float(), full.past_key_values, scene, 256)
                    checks = dict(max_logit_difference=float((full.logits[0, -1].float() - end.logits[0, -1].float()).abs().max()),
                                  parsed_equal=all(task.score_answer(direct[f'text{b}'], scene['aliases'], gold) ==
                                                   task.score_answer(value[f'text{b}'], scene['aliases'], gold) for b in (64, 256)))
                    checks['pass'] = checks['max_logit_difference'] <= .05 and checks['parsed_equal']
                if arm == 'none':
                    zero_ref.setdefault(qi, (value['logits_sha256'], value['tokens']))
                if cut is not None and cut == n_blocks - 1:
                    assert (value['logits_sha256'], value['tokens']) == zero_ref[qi], 'last-layer cut differs from no message'
                row = dict(scene=scene['scene'], family=scene['family'], domain=scene['domain'], template=scene['template'],
                           world=wi, question=qi, aliases=scene['aliases'], gold=gold, split=m['row_split'],
                           receiver=m['receiver'], diagnostic=diagnostic, arm=arm,
                           source=source, write_layer=layer, cut_layer=cut, gamma=1 if source else 0,
                           entries=active, extra_text=extra, prefix_cache_sha256=fp, patch_report=report,
                           engineering_checks=checks, **value)
                row.update({f'score{b}': task.score_answer(value[f'text{b}'], scene['aliases'], gold) for b in (64, 256)})
                rows.append(row)
            assert prior.cache_fingerprint(cache) == fp
            return rows

        for spec in m['arms']:
            records += run(**spec)
    assert prior.cache_fingerprint(ncache) == nfp
    return records


def arm_key(spec):
    return '{arm}|{source}|{layer}|{cut}'.format(arm=spec['arm'], source=spec.get('source'), layer=spec.get('layer'), cut=spec.get('cut'))


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
    assert blob['words'] == words
    order = {l: i for i, l in enumerate(blob['layers'])}
    D = blob['directions'].float().to('cuda:0')
    bank = json.loads(Path(m['bank']).read_text())
    assert len(bank) == m['bank_size']
    scenes = ([s for s in bank if s['scene'] in m['engineering_scene_ids']] if a.phase == 'engineering'
              else [s for s in bank if s['scene'] not in m['engineering_scene_ids']][a.shard::a.shards])
    raw = root / 'raw' / f'{a.phase}_{m["receiver"]}_s{a.shard}.jsonl'; raw.parent.mkdir(parents=True, exist_ok=True)
    seen = {r['key']: r for r in map(json.loads, raw.read_text().splitlines())} if raw.exists() else {}
    keyf = lambda r: '|'.join(str(r[f]) for f in ('scene', 'world', 'question', 'arm', 'source', 'write_layer', 'cut_layer'))
    started = time.time()
    for index, scene in enumerate(scenes):
        rows = None
        expected = {f'{scene["scene"]}|{w}|{q}|{s["arm"]}|{s.get("source")}|{s.get("layer") if s.get("cut") is None else m["layers"]["d20"]}|{s.get("cut")}'
                    for w in (0, 1) for q in (0, 1) for s in m['arms']}
        if expected <= seen.keys():
            continue
        rows = scene_rows(receiver, scene, m, a.phase, D, order)
        for row in rows:
            row['key'] = keyf(row); row['phase'] = a.phase
            if row['key'] in seen:
                assert all(row[f] == seen[row['key']][f] for f in ('logits_sha256', 'tokens'))
            else:
                with raw.open('a') as h:
                    h.write(json.dumps(row) + '\n')
                seen[row['key']] = row
        assert all(not r['engineering_checks'] or r['engineering_checks']['pass'] for r in rows), 'engineering failure'
        print('SCENE', index + 1, len(scenes), 'rows', len(seen), round(time.time() - started), 's', flush=True)
    want = len(scenes) * 4 * len(m['arms'])
    assert len(seen) == want, (len(seen), want)
    prior.atomic_json(raw.with_suffix('.done.json'),
                      dict(status='PASS', phase=a.phase, receiver=m['receiver'], shard=a.shard, shards=a.shards, rows=len(seen),
                           scenes=len(scenes), manifest_sha256=base.sha(a.manifest.read_bytes()), raw_sha256=base.sha(raw.read_bytes()),
                           elapsed_seconds=time.time() - started))
    print('COMPLETE', m['receiver'], a.phase, a.shard, flush=True)


if __name__ == '__main__':
    main()
