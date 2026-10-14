"""Native lexical reference retaining public message framing.

Discovery only: no dictionary decomposition or new communication method claim.
Source facts precede carrier sites. Neutral/source prefixes have identical token
lengths and identical tokens outside the message block, across both worlds.
"""
import argparse
import contextlib
import importlib.util
import json
import re
from pathlib import Path
import sys
import time

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
FROZEN = ROOT / 'artifacts/mechanism0919/discovery_v3/snapshot'
sys.path.insert(0, str(FROZEN))
spec = importlib.util.spec_from_file_location('frozen_discovery', FROZEN / 'run.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
base, task = prior.base, prior.task

ARMS = ('neutral', 'identity_band', 'full_source', 'native_band', 'message_band',
        'fact_tokens_band', 'fact_next_band', 'fact_tokens_norm_band', 'nonfact_k_band', 'other_world_fact_tokens_band')


class Scaffold:
    @staticmethod
    def receiver_body(scene, qi, extra=None):
        carriers = [f'{alias} is an object.' for alias in scene['aliases']]
        assert extra
        body = task.FEWSHOT + 'Message:\n' + extra + '\nObjects:\n'
        body += ' '.join(carriers) + '\nQ: ' + scene['questions'][qi] + '\nA:'
        return body, carriers


def build_inputs(receiver, scene):
    worlds = []
    for world in scene['worlds']:
        packet = prior.literal_packet(world, receiver.words)
        message = base.packet_text(packet, receiver.words, scene['aliases'], weights=True)
        texts, sequences, sites, n = receiver.tokenized(scene, message)
        enc = receiver.tok(texts[0], return_offsets_mapping=True, add_special_tokens=not receiver.chat)
        start = texts[0].index(message)
        memory = [i for i, (a, b) in enumerate(enc['offset_mapping'])
                  if a < start + len(message) and b > start]
        assert memory == list(range(memory[0], memory[-1] + 1))
        lo, hi = memory[0], memory[-1] + 1
        assert hi <= min(p for span in sites for p in span) < n
        worlds.append(dict(packet=packet, message=message, prefix=sequences[0][:n],
                           suffixes=[seq[n:] for seq in sequences], sites=sites, lo=lo, hi=hi))
    # A fixed message block length per scene removes world-dependent length
    # from the neutral receiver. This task is a scoped patch instrument.
    length = max(w['hi'] - w['lo'] for w in worlds)
    newline = receiver.tok('\n', add_special_tokens=False)['input_ids']
    filler = receiver.tok(' unknown', add_special_tokens=False)['input_ids']
    assert len(newline) == len(filler) == 1
    for w in worlds:
        lo, hi, old = w['lo'], w['hi'], w['prefix']
        pad = length - (hi - lo)
        w['source'] = old[:hi] + newline * pad + old[hi:]
        w['neutral'] = old[:lo] + filler * length + old[hi:]
        w['sites'] = [[p + pad for p in span] for span in w['sites']]
        w['message_positions'] = list(range(lo, lo + length))
        outside = set(range(len(w['source']))) - set(w['message_positions'])
        assert all(w['source'][p] == w['neutral'][p] for p in outside)
        assert len(w['source']) == len(w['neutral'])
    assert worlds[0]['neutral'] == worlds[1]['neutral']
    assert worlds[0]['sites'] == worlds[1]['sites']
    assert worlds[0]['suffixes'] == worlds[1]['suffixes']
    return worlds


def lexical_positions(receiver, scene, inputs):
    """Only literal packet words determine sites; questions/gold never inspected.

    Use offsets from the exact source prompt, not isolated word tokenization.
    Source padding is after the message so these offsets need no adjustment.
    The other-world arm uses the other world's sites as well as its states.
    """
    for w in inputs:
        texts, sequences, _, n = receiver.tokenized(scene, w['message'])
        enc = receiver.tok(texts[0], return_offsets_mapping=True,
                           add_special_tokens=not receiver.chat)
        assert enc['input_ids'] == sequences[0]
        start = texts[0].index(w['message'])
        matches = list(re.finditer(r'([a-z]+) \[255/255\]', w['message']))
        expected = [receiver.words[i] for ln in w['packet'].lanes for i in ln.concept_ids]
        assert [match.group(1) for match in matches] == expected and len(matches) == 4
        spans = []
        for match in matches:
            lo, hi = start + match.start(1), start + match.end(1)
            sites = [i for i, (a, b) in enumerate(enc['offset_mapping']) if a < hi and b > lo]
            assert sites and all(p in w['message_positions'] for p in sites)
            assert sites == list(range(sites[0], sites[-1]+1))
            spans.append(dict(word=match.group(1), char_span=[lo, hi], positions=sites))
        positions = sorted(p for span in spans for p in span['positions'])
        assert len(set(positions)) == len(positions)
        nonfact = sorted(set(w['message_positions']) - set(positions))
        k = len(positions)
        assert len(nonfact) >= k > 1
        selected = [nonfact[i*(len(nonfact)-1)//(k-1)] for i in range(k)]
        assert len(set(selected)) == k and not set(selected)&set(positions)
        assert len({span['positions'][-1]+1 for span in spans}) == 4
        assert all(span['positions'][-1]+1 in nonfact for span in spans)
        w.update(fact_spans=spans, fact_positions=positions,
                 fact_next_positions=[span['positions'][-1]+1 for span in spans],
                 nonfact_positions=selected)
    return inputs


def preserve_public_frame(receiver, inputs):
    """Hide only literal facts; aliases/weights/framing are fixed public data.

    This is restricted to the equal-weight literal diagnostic. It must not
    retain world-dependent sender coefficients in a supposed no-message input.
    """
    filler = receiver.tok(' unknown', add_special_tokens=False)['input_ids']
    assert len(filler) == 1
    for w in inputs:
        assert all(q == 255 for lane in w['packet'].lanes for q in lane.q)
        assert len(w['fact_positions']) == 4
        assert all(len(span['positions']) == 1 for span in w['fact_spans'])
        neutral = list(w['source'])
        for p in w['fact_positions']:
            neutral[p] = filler[0]
        assert {i for i,(a,b) in enumerate(zip(w['source'],neutral)) if a!=b} == set(w['fact_positions'])
        w['neutral'] = neutral
    assert inputs[0]['fact_positions'] == inputs[1]['fact_positions']
    assert inputs[0]['neutral'] == inputs[1]['neutral'], 'Private content survived masking'
    assert inputs[0]['sites'] == inputs[1]['sites']
    assert inputs[0]['suffixes'] == inputs[1]['suffixes']
    return inputs


@contextlib.contextmanager
def activation_hooks(receiver, layers, captured=None, targets=None, positions=None, report=None, norm_targets=None):
    handles = []
    for layer in layers:
        def hook(module, args, output, layer=layer):
            h = output[0] if isinstance(output, tuple) else output
            if captured is not None:
                captured[layer] = h.detach().clone()
                return output
            updated = h.clone()
            delta = targets[layer][:, positions] - h[:, positions]
            raw_norm = float(delta.float().norm())
            requested = None if norm_targets is None else norm_targets[str(layer)]
            if requested is None:
                updated[:, positions] = targets[layer][:, positions]
            else:
                assert raw_norm > 0 or requested == 0
                scale = requested / raw_norm if raw_norm > 0 else 0.
                updated[:, positions] = h[:, positions] + scale * delta
            realized = float((updated[:, positions] - h[:, positions]).float().norm())
            if requested is not None:
                assert abs(realized - requested) <= max(1e-5, 1e-5 * requested), (realized, requested)
            report[str(layer)] = dict(positions=len(positions), norm=realized, raw_norm=raw_norm,
                                     requested_norm=requested,
                                     target_sha256=prior.digest_tensor(targets[layer][:, positions]))
            return (updated,) + output[1:] if isinstance(output, tuple) else updated
        handles.append(base.decoder_layers(receiver.model)[layer].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def forward(receiver, ids):
    t = torch.tensor([ids], device='cuda:0')
    return receiver.model(t, attention_mask=torch.ones_like(t), use_cache=True)


@torch.inference_mode()
def evaluate_scene(receiver, scene, engineering, m):
    inputs = preserve_public_frame(receiver, lexical_positions(receiver, scene, build_inputs(receiver, scene)))
    source_states, native_outputs = [], []
    for w in inputs:
        states = {}
        with activation_hooks(receiver, receiver.band, captured=states):
            out = forward(receiver, w['source'])
        source_states.append(states)
        native_outputs.append(out)
    neutral_states = {}
    with activation_hooks(receiver, receiver.band, captured=neutral_states):
        neutral_output = forward(receiver, inputs[0]['neutral'])
    zero_reference = {}
    records = []
    for wi, w in enumerate(inputs):
        carrier_positions = sorted(p for span in w['sites'] for p in span)
        message_positions = w['message_positions']
        carrier_norms = None
        for arm in ARMS:
            report, hooks, patch = {}, [], None
            ids = w['source'] if arm == 'full_source' else w['neutral']
            if arm == 'full_source':
                out = native_outputs[wi]
            elif arm == 'neutral':
                out = neutral_output
            elif arm == 'literal_write':
                with receiver.hooks(w['sites'], w['packet'], m['literal_gamma'][str(m['active_pair'])],
                                    'band', False, report) as hooks:
                    out = forward(receiver, ids)
                assert all(h.handle is None for h in hooks)
            else:
                source_wi = 1 - wi if arm == 'other_world_fact_tokens_band' else wi
                source_w = inputs[source_wi]
                targets = neutral_states if arm == 'identity_band' else source_states[source_wi]
                layers = receiver.band
                if arm == 'message_band':
                    positions = message_positions
                elif arm in ('fact_tokens_band', 'fact_tokens_norm_band', 'other_world_fact_tokens_band'):
                    positions = source_w['fact_positions']
                elif arm == 'fact_next_band':
                    positions = source_w['fact_next_positions']
                elif arm == 'nonfact_k_band':
                    positions = source_w['nonfact_positions']
                else:
                    positions = carrier_positions
                if arm == 'fact_tokens_norm_band':
                    assert carrier_norms is not None
                patch = dict(layers=layers, targets=targets, positions=positions,
                             norm_targets=carrier_norms if arm == 'fact_tokens_norm_band' else None)
                with activation_hooks(receiver, **patch, report=report):
                    out = forward(receiver, ids)
            if arm == 'native_band':
                carrier_norms = {li: rec['norm'] for li, rec in report.items()}
            prefix_cache = out.past_key_values
            prefix_hash = prior.cache_fingerprint(prefix_cache)
            for qi, suffix in enumerate(w['suffixes']):
                branch = prior.clone_cache(prefix_cache)
                branch_hash = prior.cache_fingerprint(branch)
                assert branch_hash == prefix_hash
                suffix_ids = torch.tensor([suffix], device='cuda:0')
                end = receiver.model(suffix_ids, past_key_values=branch, use_cache=True,
                    attention_mask=torch.ones((1, len(ids) + len(suffix)), device='cuda:0', dtype=torch.long))
                logits = end.logits[0, -1].float()
                value = receiver.decode(logits, end.past_key_values, scene, 256)
                checks = None
                if engineering:
                    if patch is not None:
                        # Apply the same prefix-only sites in a complete forward.
                        with activation_hooks(receiver, **patch, report={}):
                            direct = forward(receiver, ids + suffix)
                    elif arm == 'literal_write':
                        with receiver.hooks(w['sites'], w['packet'], m['literal_gamma'][str(m['active_pair'])], 'band', False, {}):
                            direct = forward(receiver, ids + suffix)
                    else:
                        direct = forward(receiver, ids + suffix)
                    direct_value = receiver.decode(direct.logits[0, -1].float(), direct.past_key_values, scene, 256)
                    checks = dict(max_logit_difference=float((direct.logits[0, -1].float() - logits).abs().max()),
                                  alias_gap_difference=abs(direct_value['gap'] - value['gap']),
                                  parsed_equal=all(task.score_answer(direct_value[f'text{b}'], scene['aliases'], scene['worlds'][wi]['answers'][qi]) ==
                                                   task.score_answer(value[f'text{b}'], scene['aliases'], scene['worlds'][wi]['answers'][qi]) for b in (64, 256)))
                    checks['pass'] = (checks['max_logit_difference'] <= .05 and checks['alias_gap_difference'] <= .1 and checks['parsed_equal'])
                gold = scene['worlds'][wi]['answers'][qi]
                row = dict(scene=scene['scene'], family=scene['family'], world=wi, question=qi,
                    arm=arm, gold=gold, aliases=scene['aliases'], source_message=w['message'],
                    patch_source_world=(1 - wi if arm == 'other_world_fact_tokens_band' else wi)
                                       if arm not in ('neutral', 'identity_band', 'full_source') else None,
                    packet_sha256=base.sha(base.serialize(w['packet'])), patch_report=report,
                    prefix_ids=ids, suffix_ids=suffix, carrier_positions=w['sites'],
                    message_positions=message_positions, patch_positions=patch['positions'] if patch else [],
                    lexical_worlds=[{k: sw[k] for k in ('fact_spans', 'fact_positions', 'fact_next_positions', 'nonfact_positions')} for sw in inputs],
                    prefix_cache_sha256=prefix_hash, branch_initial_cache_sha256=branch_hash,
                    engineering_checks=checks, **value)
                row.update({f'score{b}': task.score_answer(value[f'text{b}'], scene['aliases'], gold) for b in (64, 256)})
                if arm == 'neutral':
                    key = (qi,)
                    zero = (prefix_hash, value['logits_sha256'], value['tokens'])
                    assert zero_reference.setdefault(key, zero) == zero
                if arm == 'identity_band':
                    assert (prefix_hash, value['logits_sha256'], value['tokens']) == zero_reference[(qi,)]
                    assert all(x['norm'] == 0 for x in report.values())
                records.append(row)
            assert prior.cache_fingerprint(prefix_cache) == prefix_hash
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--phase', choices=('engineering', 'development'), required=True)
    ap.add_argument('--pair', type=int, required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    a = ap.parse_args(); m = json.loads(a.manifest.read_text()); root = Path(m['root'])
    for path, expected in m['pins'].items():
        assert base.sha(Path(path).read_bytes()) == expected, path
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m['active_pair'] = a.pair
    bank = json.loads(Path(m['bank']).read_text())
    scenes = [s for s in bank if s['scene'] in m['engineering_scene_ids']] if a.phase == 'engineering' else bank[a.shard::a.shards]
    root.joinpath('raw').mkdir(exist_ok=True)
    raw = root / 'raw' / f'{a.phase}_p{a.pair}_s{a.shard}.jsonl'
    previous = [json.loads(line) for line in raw.open()] if raw.exists() else []
    seen = {r['key']: r for r in previous}
    assert len(seen) == len(previous)
    assert all(not r['engineering_checks'] or r['engineering_checks']['pass'] for r in previous)
    receiver = prior.Receiver(m, m['pairs'][a.pair]['receiver'], Scaffold)
    start = time.time()
    for index, scene in enumerate(scenes):
        keys = {f'{scene["scene"]}|{w}|{q}|{arm}' for w in (0, 1) for q in (0, 1) for arm in ARMS}
        if keys <= set(seen):
            continue
        rows = evaluate_scene(receiver, scene, a.phase == 'engineering', m)
        for row in rows:
            row.update(pair=a.pair, phase=a.phase)
            key = f'{row["scene"]}|{row["world"]}|{row["question"]}|{row["arm"]}'
            row['key'] = key
            if key in seen:
                assert all(seen[key][field] == row[field] for field in ('prefix_cache_sha256', 'logits_sha256', 'tokens'))
            else:
                with raw.open('a') as handle:
                    handle.write(json.dumps(row) + '\n')
                seen[key] = row
        assert all(not row['engineering_checks'] or row['engineering_checks']['pass'] for row in rows), 'Replay failure retained'
        if a.phase == 'engineering':
            parent_raw = Path(m['parent_root']) / 'raw' / f'engineering_p{a.pair}_s0.jsonl'
            parent_rows = {r['key']: r for r in map(json.loads, parent_raw.read_text().splitlines())}
            for row in rows:
                if row['arm'] in m['parent_replay_arms'] and row['key'] in parent_rows:
                    assert all(row[f] == parent_rows[row['key']][f] for f in ('prefix_ids', 'prefix_cache_sha256', 'logits_sha256', 'tokens')), 'Shared-condition replay differs from parent'

        print('SCENE', index + 1, len(scenes), 'rows', len(seen), flush=True)
    assert len(seen) == len(scenes) * 4 * len(ARMS)
    if a.phase == 'engineering':
        # A failed scaffold reference cannot justify a negative local-patch claim.
        joint = sum(all(seen[f'{s["scene"]}|{w}|{q}|full_source']['score64']['correct'] for w in (0, 1) for q in (0, 1)) for s in scenes) / len(scenes)
        assert joint >= .9, f'Full-source instrument failed: {joint}'
    receipt = dict(status='PASS', stage=a.phase, pair=a.pair, shard=a.shard,
                   rows=len(seen), scenes=len(scenes), raw_sha256=base.sha(raw.read_bytes()),
                   manifest_sha256=base.sha(a.manifest.read_bytes()), elapsed_seconds=time.time() - start)
    prior.atomic_json(raw.with_suffix('.done.json'), receipt)
    print('COMPLETE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
