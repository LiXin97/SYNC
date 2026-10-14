"""R2-rescreen / R3-confirmation runner: the frozen automatic sender's packets and complete-fact packets delivered
through the ORIGINAL and the SELECTED write configuration, with text, reassignment, other-world and native references
(the study protocol §7 rescreen and §8).

Derived from src/syncomm/recovery/evaluate_depth.py: the same common target prefix (public message, four slots per
object, every slot neutral), the same target positions (each active slot's word-final token), the same question
suffixes, the same write hook and reporting, the same native transplantation with its declared alignment. What this
runner adds is (a) several message SOURCES read from frozen packets files, (b) the two named write configurations
`original` (the band this project used until now) and `selected` (R2's frozen choice), (c) three fixed word-to-
direction reassignments of the selected write, norm-matched per layer to it, and (d) the other world's independently
generated packet delivered through the selected configuration, scored against both the current and the supplied gold.

Arms (per source, from the manifest):
  none, identity_write          no-message row and the active-hook inertness gate
  original, selected            raw fitted write at gamma=1 at the original band / at the selected depth
  packet_text                   the identical packet rendered as weighted public-layout text (padded, same slots)
  full_text                     the world's evidence sentences verbatim
  sham0..2                      the selected write with the manifest's frozen word permutations, per-layer norm-matched
  other_world                   the OTHER world's packet of the same source through the selected configuration
  native                        the fact words' own states transplanted at the selected depth (complete facts only)
Nothing is tuned here; every configuration comes from the manifest, which is locked before the first row.
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
sys.path.insert(0, str(ROOT / 'src'))
from syncomm.recovery.evaluate_depth import (  # noqa: E402  the same primitives, imported not copied
    WIDTH, build_receiver, write_hooks, capture_hooks, forward, prompt_inputs, entry_positions,
    fitted_states, parent, prior, base, task, render, packet_entries)


def literal_packet_generic(world, words):
    """The complete-fact packet without the frozen helper's field-name assertion, so the fibre/metal bank works too.

    Construction is otherwise identical to prior.literal_packet (each object's two gold values, ascending concept ids,
    equal 8-bit weights), and where the frozen assertion WOULD pass the two constructions are asserted byte-identical.
    This is the same generic construction the earlier B6/confirmation packages used for that bank.
    """
    index = {w: i for i, w in enumerate(words)}
    lanes = []
    for obj in world['facts']:
        values = list(obj.values())
        assert len(values) == 2 and len(set(values)) == 2 and all(v in index for v in values)
        lanes.append((tuple(sorted(index[v] for v in values)), (255, 255)))
    packet = base.Packet(base.inventory_digest(words), 8, tuple(base.Lane(i, ids, q) for i, (ids, q) in enumerate(lanes)))
    assert base.serialize(base.deserialize(base.serialize(packet))) == base.serialize(packet)
    try:
        reference = prior.literal_packet(world, words)
    except AssertionError:
        reference = None
    if reference is not None:
        assert base.serialize(reference) == base.serialize(packet), 'generic literal packet differs from the frozen one'
    return packet


def permuted_states(neutral, layers, dirs, active, mapping, gamma, permutation):
    """The selected write with each address replaced by permutation[address]; the per-layer norm is matched afterwards
    by the write hook's norm_targets, so the reassignment differs only in WHICH direction is written."""
    out = {layer: neutral[layer].clone() for layer in layers}
    for li, layer in enumerate(layers):
        for e in active:
            pos = mapping[e['object'], e['index']]['last_position']
            out[layer][0, pos] += (gamma * e['relative_weight']) * dirs[li, int(permutation[e['address']])].to(out[layer])
    return out


@torch.inference_mode()
def scene_rows(receiver, scene, m, phase, packets_by_source, dirs, arms_by_source):
    ni = prompt_inputs(receiver, scene, None)
    cfg = {'original': m['original_layers'], 'selected': m['selected_layers']}
    all_layers = sorted(set(cfg['original']) | set(cfg['selected']))
    neutral_states = {}
    with capture_hooks(receiver.model, all_layers, neutral_states):
        neutral_out = forward(receiver, ni['prefix'])
    records, zero_ref = [], {}

    def packet_of(source, wi):
        if packets_by_source[source] is None:
            return literal_packet_generic(scene['worlds'][wi], receiver.words)
        rec = packets_by_source[source]['packets'][f'{scene["scene"]}|{wi}']
        pk = base.deserialize(bytes.fromhex(rec['hex']))
        assert base.sha(base.serialize(pk)) == rec['sha256']
        return pk

    for source, arms in arms_by_source.items():
        for wi, world in enumerate(scene['worlds']):
            packet = packet_of(source, wi); supplied = packet_of(source, 1 - wi)
            entries = packet_entries(packet, receiver.words)
            sup_entries = packet_entries(supplied, receiver.words)
            active, positions, mapping = entry_positions(entries, ni['slots'])
            sup_active, sup_positions, _ = entry_positions(sup_entries, ni['slots'])
            words_by_obj = [[e['word'] for e in sorted(active, key=lambda x: x['index']) if e['object'] == o] for o in (0, 1)]
            weights_by_obj = [[e['weight'] for e in sorted(active, key=lambda x: x['index']) if e['object'] == o] for o in (0, 1)]
            nat, alignment = None, None
            if 'native' in arms:
                si = prompt_inputs(receiver, scene, words_by_obj)
                src = {(s['object'], s['index']): s for s in si['slots']}
                src_states = {}
                with capture_hooks(receiver.model, cfg['selected'], src_states):
                    forward(receiver, si['prefix'], use_cache=False)
                nat = {}
                for layer in cfg['selected']:
                    t = neutral_states[layer].clone()
                    for e in active:
                        t[0, mapping[e['object'], e['index']]['last_position']] = src_states[layer][0, src[e['object'], e['index']]['last_position']]
                    nat[layer] = t
                assert ni['prefix'][:ni['lo']] == si['prefix'][:si['lo']] and ni['prefix'][ni['hi']:] == si['prefix'][si['hi']:]
                alignment = dict(target_tokens=len(ni['prefix']), source_tokens=len(si['prefix']), identical_outside_message=True,
                                 source_slot_token_counts={f"{e['object']}.{e['index']}": len(src[e['object'], e['index']]['positions']) for e in active})
            selected_norms = {}
            for arm in arms:
                report, states, nt, layers = {}, None, None, cfg['selected']
                ids, suffixes, extra = ni['prefix'], ni['suffixes'], None
                cond_packet, cond_entries, cond_positions = packet, active, positions
                if arm == 'none':
                    out = neutral_out
                elif arm in ('packet_text', 'full_text'):
                    extra = (' '.join(world['evidence']) if arm == 'full_text'
                             else render(scene['aliases'], words_by_obj, width=WIDTH)[0])
                    _, seqs, _, n = receiver.tokenized(scene, extra)
                    ids, suffixes = seqs[0][:n], [s[n:] for s in seqs]
                    out = forward(receiver, ids)
                else:
                    if arm == 'identity_write':
                        states = {l: neutral_states[l] for l in layers}
                    elif arm == 'original':
                        layers = cfg['original']
                        states = fitted_states({l: neutral_states[l] for l in layers}, layers,
                                               dirs['original'], [1.0] * len(layers), active, mapping, m['gamma'])
                    elif arm == 'selected':
                        states = fitted_states({l: neutral_states[l] for l in layers}, layers,
                                               dirs['selected'], [1.0] * len(layers), active, mapping, m['gamma'])
                    elif arm.startswith('sham'):
                        states = permuted_states({l: neutral_states[l] for l in layers}, layers, dirs['selected'],
                                                 active, mapping, m['gamma'], m['permutations'][int(arm[-1])])
                        nt = selected_norms
                        assert nt, 'the selected arm must precede its reassignments'
                    elif arm == 'other_world':
                        cond_packet, cond_entries, cond_positions = supplied, sup_active, sup_positions
                        states = fitted_states({l: neutral_states[l] for l in layers}, layers,
                                               dirs['selected'], [1.0] * len(layers), sup_active, mapping, m['gamma'])
                    elif arm == 'native':
                        states = {l: nat[l] for l in layers}
                    else:
                        raise AssertionError(arm)
                    with write_hooks(receiver.model, layers, states, cond_positions, report, nt):
                        out = forward(receiver, ids)
                if arm == 'selected':
                    selected_norms = {k: v['norm'] for k, v in report.items()}
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
                    gold_supplied = scene['worlds'][1 - wi]['answers'][qi] if arm == 'other_world' else gold
                    checks = None
                    if phase == 'engineering':
                        ctx = write_hooks(receiver.model, layers, states, cond_positions, {}, nt) if states is not None else contextlib.nullcontext()
                        with ctx:
                            full = forward(receiver, ids + suffix)
                        direct = receiver.decode(full.logits[0, -1].float(), full.past_key_values, scene, 256)
                        checks = dict(max_logit_difference=float((full.logits[0, -1].float() - logits).abs().max()),
                                      alias_gap_difference=abs(direct['gap'] - value['gap']),
                                      parsed_equal=all(task.score_answer(direct[f'text{b}'], scene['aliases'], gold) ==
                                                       task.score_answer(value[f'text{b}'], scene['aliases'], gold) for b in (64, 256)))
                        checks['pass'] = checks['max_logit_difference'] <= .05 and checks['alias_gap_difference'] <= .1 and checks['parsed_equal']
                    row = dict(scene=scene['scene'], family=scene['family'], domain=scene['domain'], template=scene['template'],
                               world=wi, question=qi, aliases=scene['aliases'], gold=gold, gold_supplied=gold_supplied,
                               split=m['row_split'], receiver=m['receiver'], source=source, arm=arm, gamma=(m['gamma'] if states is not None and arm not in ('identity_write',) else 0),
                               layers=layers, depth_label=('original' if arm == 'original' else 'selected' if arm not in ('none', 'packet_text', 'full_text') else 'shared'),
                               width=WIDTH, layout='original', question_style='original',
                               entries=cond_entries, patch_source_world=(1 - wi) if arm == 'other_world' else wi,
                               source_packet_sha256=base.sha(base.serialize(packet)), condition_packet_sha256=base.sha(base.serialize(cond_packet)),
                               condition_packet_hex=base.serialize(cond_packet).hex(),
                               wire_bytes=len(extra.encode()) if extra is not None else (0 if arm in ('none', 'identity_write') else len(base.serialize(cond_packet))),
                               extra_text=extra, prefix_ids=ids, suffix_ids=suffix, slots=ni['slots'],
                               patch_positions=cond_positions if states is not None else [], patch_report=report,
                               alignment=alignment, prefix_cache_sha256=fp, branch_initial_cache_sha256=fp,
                               engineering_checks=checks, **value)
                    row.update({f'score{b}': task.score_answer(value[f'text{b}'], scene['aliases'], gold) for b in (64, 256)})
                    row.update({f'score{b}_supplied': task.score_answer(value[f'text{b}'], scene['aliases'], gold_supplied) for b in (64, 256)})
                    if arm == 'none':
                        z = (fp, value['logits_sha256'], value['tokens'])
                        assert zero_ref.setdefault(qi, z) == z, 'the no-message prefix depends on the world or the source'
                    if arm == 'identity_write':
                        assert (fp, value['logits_sha256'], value['tokens']) == zero_ref[qi], 'identity write is not inert'
                        assert report and all(v['norm'] <= 1e-6 for v in report.values())
                    if arm.startswith('sham'):
                        ref = selected_norms
                        assert all(abs(report[k]['norm'] - ref[k]) <= max(1e-4, 1e-4 * ref[k]) for k in report), 'reassignment norm not matched'
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
    assert blob['words'] == words
    order = {l: i for i, l in enumerate(blob['layers'])}
    D = blob['directions'].float().to('cuda:0')
    dirs = {k: D[[order[l] for l in m[f'{k}_layers']]] for k in ('original', 'selected')}
    packets_by_source = {s: (json.loads(Path(p).read_text()) if p else None) for s, p in m['sources'].items()}
    arms_by_source = {s: list(v) for s, v in m['arms'].items()}
    bank = json.loads(Path(m['bank']).read_text())
    assert len(bank) == m['bank_size'] and all(s['split'] == m['bank_split'] for s in bank)
    scenes = ([s for s in bank if s['scene'] in m['engineering_scene_ids']] if a.phase == 'engineering'
              else [s for s in bank if s['scene'] not in m['engineering_scene_ids']][a.shard::a.shards])
    raw = root / 'raw' / f'{a.phase}_{m["receiver"]}_s{a.shard}.jsonl'; raw.parent.mkdir(parents=True, exist_ok=True)
    seen = {r['key']: r for r in map(json.loads, raw.read_text().splitlines())} if raw.exists() else {}
    keyf = lambda r: '|'.join(str(r[f]) for f in ('scene', 'world', 'question', 'source', 'arm'))
    started = time.time()
    for index, scene in enumerate(scenes):
        expected = {f'{scene["scene"]}|{w}|{q}|{s}|{arm}' for w in (0, 1) for q in (0, 1)
                    for s, arms in arms_by_source.items() for arm in arms}
        if expected <= seen.keys():
            continue
        rows = scene_rows(receiver, scene, m, a.phase, packets_by_source, dirs, arms_by_source)
        for row in rows:
            row['key'] = keyf(row); row['phase'] = a.phase
            if row['key'] in seen:
                assert all(row[f] == seen[row['key']][f] for f in ('logits_sha256', 'tokens'))
            else:
                with raw.open('a') as h:
                    h.write(json.dumps(row) + '\n')
                seen[row['key']] = row
        assert all(not r['engineering_checks'] or r['engineering_checks']['pass'] for r in rows)
        print('SCENE', index + 1, len(scenes), 'rows', len(seen), flush=True)
    want = len(scenes) * 4 * sum(len(v) for v in arms_by_source.values())
    assert len(seen) == want, (len(seen), want)
    prior.atomic_json(raw.with_suffix('.done.json'),
                      dict(status='PASS', phase=a.phase, receiver=m['receiver'], shard=a.shard, shards=a.shards,
                           rows=len(seen), scenes=len(scenes), manifest_sha256=base.sha(a.manifest.read_bytes()),
                           raw_sha256=base.sha(raw.read_bytes()), elapsed_seconds=time.time() - started))
    print('COMPLETE', m['receiver'], a.phase, a.shard, flush=True)


if __name__ == '__main__':
    main()
