"""P0-A runner: local-representation baselines on the same packets (the study protocol §4).

Same common target prefix, positions, question suffixes, decoding and scoring as the recovery runners
(src/syncomm/recovery/evaluate_depth.py / evaluate_confirm.py, whose primitives are imported, not copied). What varies is
only what is written at the active slots:

  none / identity_write     no message; the active identity write (engineering inertness gate)
  packet_text               the identical packet rendered as padded text in the same eight slots (the text route)
  full_text                 the world's evidence sentences (source-independent reference; literal source only)
  w:<construction>@g        h_neutral + g * relative_weight * d_c[address] at the frozen depth-0.2 layer, where
                            c in {c12, single_t0, single_t6, single_t11, single_t*_nm}
  emb1@g                    the receiver's OWN input-embedding module output at each active slot replaced by
                            e(neutral) + g * (pool(e(word tokens in the field context)) - e(neutral)), pool = mean over the
                            word's tokens as the tokenizer splits it in its actual field (a POOLING ADAPTATION for
                            multi-token words; for a single-token word at g = 1 it is exactly the word's own embedding)

Full token-span replacement is checked in engineering as an identity: the packet-text prompt run through
`inputs_embeds = embedding_module(ids)` must reproduce the ids-path logits; the text rows then stand for that route.
Each row records whether every active word is a single token in its field context (the pre-declared stratum).
"""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
sys.path.insert(0, str(ROOT / 'src'))
from syncomm.recovery.evaluate_depth import (  # noqa: E402
    WIDTH, build_receiver, write_hooks, capture_hooks, forward, prompt_inputs, entry_positions, fitted_states,
    prior, base, task, render, packet_entries)
from syncomm.recovery.evaluate_confirm import literal_packet_generic  # noqa: E402


@contextlib.contextmanager
def embed_hook(model, targets, report):
    """Replace the input-embedding module's output rows at prefix positions (prefill only)."""
    emb = model.get_input_embeddings()

    def hook(module, args, output):
        if output.shape[1] == 1:
            return output
        out = output.clone()
        for pos, vec in targets.items():
            before = out[0, pos].float().clone()   # .float() is a view in FP32; without the copy the report read 0
            out[0, pos] = vec.to(out.dtype)
            report[str(pos)] = dict(edit_norm=float((out[0, pos].float() - before).norm()), neutral_norm=float(before.norm()))
        return out
    h = emb.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def emb_rows(receiver, ids):
    with torch.inference_mode():
        return receiver.model.get_input_embeddings()(torch.tensor([ids], device='cuda:0'))[0].float()


@torch.inference_mode()
def scene_rows(receiver, scene, m, phase, packets_by_source, dirs):
    ni = prompt_inputs(receiver, scene, None)
    layer = m['layer']
    neutral_states = {}
    with capture_hooks(receiver.model, [layer], neutral_states):
        neutral_out = forward(receiver, ni['prefix'])
    e_neutral = emb_rows(receiver, ni['prefix'])
    records, zero_ref = [], {}

    def packet_of(source, wi):
        if packets_by_source[source] is None:
            return literal_packet_generic(scene['worlds'][wi], receiver.words)
        rec = packets_by_source[source]['packets'][f'{scene["scene"]}|{wi}']
        pk = base.deserialize(bytes.fromhex(rec['hex'])); assert base.sha(base.serialize(pk)) == rec['sha256']
        return pk

    for source, arms in m['arms'].items():
        arms = [tuple(x) for x in arms] + ([('identity_write', 0)] if phase == 'engineering' and source == 'literal' else [])
        for wi, world in enumerate(scene['worlds']):
            packet = packet_of(source, wi)
            entries = packet_entries(packet, receiver.words)
            active, positions, mapping = entry_positions(entries, ni['slots'])
            wbo = [[e['word'] for e in sorted(active, key=lambda x: x['index']) if e['object'] == o] for o in (0, 1)]
            si = prompt_inputs(receiver, scene, wbo)   # the same slots with the real words: the field context
            src = {(s['object'], s['index']): s for s in si['slots']}
            e_src = emb_rows(receiver, si['prefix'])
            token_counts = {f"{e['object']}.{e['index']}": len(src[e['object'], e['index']]['positions']) for e in active}
            single_token = all(v == 1 for v in token_counts.values())
            for arm, gamma in arms:
                report, states, ids, suffixes, extra, etargets = {}, None, ni['prefix'], ni['suffixes'], None, None
                construction = None
                if arm == 'none':
                    out = neutral_out
                elif arm in ('packet_text', 'full_text'):
                    extra = ' '.join(world['evidence']) if arm == 'full_text' else render(scene['aliases'], wbo, width=WIDTH)[0]
                    _, seqs, _, n = receiver.tokenized(scene, extra)
                    ids, suffixes = seqs[0][:n], [s[n:] for s in seqs]
                    out = forward(receiver, ids)
                elif arm == 'identity_write':
                    states = {layer: neutral_states[layer]}
                    with write_hooks(receiver.model, [layer], states, positions, report):
                        out = forward(receiver, ids)
                elif arm.startswith('w:'):
                    construction = arm[2:]
                    states = fitted_states({layer: neutral_states[layer]}, [layer], dirs[construction][None], [1.0], active, mapping, gamma)
                    with write_hooks(receiver.model, [layer], states, positions, report):
                        out = forward(receiver, ids)
                elif arm == 'emb1':
                    etargets = {}
                    for e in active:
                        pos = mapping[e['object'], e['index']]['last_position']
                        word_vec = e_src[src[e['object'], e['index']]['positions']].mean(0)     # pooling adaptation (mean)
                        etargets[pos] = e_neutral[pos] + gamma * e['relative_weight'] * (word_vec - e_neutral[pos])
                    with embed_hook(receiver.model, etargets, report):
                        out = forward(receiver, ids)
                else:
                    raise AssertionError(arm)
                cache = out.past_key_values; fp = prior.cache_fingerprint(cache)
                for qi, suffix in enumerate(suffixes):
                    branch = prior.clone_cache(cache); assert prior.cache_fingerprint(branch) == fp
                    end = receiver.model(torch.tensor([suffix], device='cuda:0'), past_key_values=branch, use_cache=True,
                                         attention_mask=torch.ones((1, len(ids) + len(suffix)), device='cuda:0', dtype=torch.long))
                    logits = end.logits[0, -1].float()
                    value = receiver.decode(logits, end.past_key_values, scene, 256)
                    gold = world['answers'][qi]
                    checks = None
                    if phase == 'engineering':
                        if states is not None:
                            ctx = write_hooks(receiver.model, [layer], states, positions, {})
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
                        if arm == 'packet_text':   # full token-span embedding replacement == text (identity check)
                            E = receiver.model.get_input_embeddings()(torch.tensor([ids + suffix], device='cuda:0'))
                            fe = receiver.model(inputs_embeds=E, attention_mask=torch.ones((1, len(ids) + len(suffix)), device='cuda:0', dtype=torch.long), use_cache=False)
                            checks['fullspan_embedding_max_logit_difference'] = float((fe.logits[0, -1].float() - full.logits[0, -1].float()).abs().max())
                            checks['fullspan_identity'] = checks['fullspan_embedding_max_logit_difference'] <= 1e-4
                        checks['pass'] = (checks['max_logit_difference'] <= .05 and checks['alias_gap_difference'] <= .1 and checks['parsed_equal']
                                          and checks.get('fullspan_identity', True))
                    row = dict(scene=scene['scene'], family=scene['family'], domain=scene['domain'], template=scene['template'],
                               world=wi, question=qi, aliases=scene['aliases'], gold=gold, gold_supplied=gold, patch_source_world=wi,
                               split=m['row_split'], receiver=m['receiver'], source=source, arm=arm, gamma=gamma, construction=construction,
                               layer=layer, width=WIDTH, entries=active, single_token_facts=single_token, token_counts=token_counts,
                               source_packet_sha256=base.sha(base.serialize(packet)), condition_packet_hex=base.serialize(packet).hex(),
                               wire_bytes=len(extra.encode()) if extra is not None else (0 if arm in ('none', 'identity_write') else len(base.serialize(packet))),
                               extra_text=extra, prefix_ids=ids, suffix_ids=suffix, patch_positions=positions if (states is not None or etargets is not None) else [],
                               patch_report=report, prefix_cache_sha256=fp, branch_initial_cache_sha256=fp, engineering_checks=checks, **value)
                    row.update({f'score{b}': task.score_answer(value[f'text{b}'], scene['aliases'], gold) for b in (64, 256)})
                    row.update({f'score{b}_supplied': row[f'score{b}'] for b in (64, 256)})
                    if arm == 'none':
                        z = (fp, value['logits_sha256'], value['tokens'])
                        assert zero_ref.setdefault(qi, z) == z, 'no-message prefix depends on the world'
                    if arm == 'identity_write':
                        assert (fp, value['logits_sha256'], value['tokens']) == zero_ref[qi], 'identity write is not inert'
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
    assert blob['words'] == words and blob['layer'] == m['layer']
    dirs = {k: v.float().to('cuda:0') for k, v in blob['constructions'].items()}
    packets_by_source = {s: (json.loads(Path(p).read_text()) if p else None) for s, p in m['sources'].items()}
    bank = json.loads(Path(m['bank']).read_text())
    assert len(bank) == m['bank_size'] and all(s['split'] == m['bank_split'] for s in bank)
    scenes = ([s for s in bank if s['scene'] in m['engineering_scene_ids']] if a.phase == 'engineering'
              else [s for s in bank if s['scene'] not in m['engineering_scene_ids']][a.shard::a.shards])
    raw = root / 'raw' / f'{a.phase}_{m["receiver"]}_s{a.shard}.jsonl'; raw.parent.mkdir(parents=True, exist_ok=True)
    seen = {r['key']: r for r in map(json.loads, raw.read_text().splitlines())} if raw.exists() else {}
    keyf = lambda r: '|'.join(str(r[f]) for f in ('scene', 'world', 'question', 'source', 'arm', 'gamma'))
    n_arms = sum(len(v) for v in m['arms'].values()) + (1 if a.phase == 'engineering' else 0)
    started = time.time()
    for index, scene in enumerate(scenes):
        if sum(1 for k in seen if k.startswith(scene['scene'] + '|')) == 4 * n_arms:
            continue
        rows = scene_rows(receiver, scene, m, a.phase, packets_by_source, dirs)
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
    assert len(seen) == len(scenes) * 4 * n_arms, (len(seen), len(scenes) * 4 * n_arms)
    prior.atomic_json(raw.with_suffix('.done.json'), dict(status='PASS', phase=a.phase, receiver=m['receiver'], shard=a.shard, shards=a.shards,
                      rows=len(seen), scenes=len(scenes), manifest_sha256=base.sha(a.manifest.read_bytes()), raw_sha256=base.sha(raw.read_bytes()),
                      elapsed_seconds=time.time() - started))
    print('COMPLETE', m['receiver'], a.phase, a.shard, flush=True)


if __name__ == '__main__':
    main()
