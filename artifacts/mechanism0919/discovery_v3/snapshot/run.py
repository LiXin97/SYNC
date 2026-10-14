"""Question-independent prefix engineering and exposed-material discovery.

Reuses the immutable original writer. This is not a confirmation runner.
The literal-fact condition is an extraction oracle, not a new sender method.
"""
from pathlib import Path
import argparse
import contextlib
import copy
import hashlib
import json
import os
import sys
import time

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
sys.path.insert(0, str(ROOT / 'artifacts/counterfactual0917/v2/snapshot'))
import counterfactual_run as base
import content_counterfactual as task
import syncomm.writer as writer_module
sys.path.insert(0, str(ROOT / 'infra/pi0917'))
from exact_solver_cache import ExactSolverCache
sys.path.insert(0, str(Path(__file__).parent))
from design import conditions, literal_codes, prefix_length, row_key


def digest_tensor(tensor):
    tensor = tensor.detach().contiguous().cpu()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def cache_fingerprint(cache):
    """This stage supports standard dense Qwen DynamicCache only."""
    records = []
    for i, layer in enumerate(cache.layers):
        for name in ('keys', 'values'):
            t = getattr(layer, name)
            assert isinstance(t, torch.Tensor) and t.numel() > 0
            records.append((i, name, list(t.shape), str(t.dtype), digest_tensor(t)))
    assert records
    raw = json.dumps([int(cache.get_seq_length()), records], sort_keys=True).encode()
    return base.sha(raw)


def clone_cache(cache):
    cloned = copy.deepcopy(cache)
    assert cloned.get_seq_length() == cache.get_seq_length()
    for left, right in zip(cache.layers, cloned.layers):
        for name in ('keys', 'values'):
            assert getattr(left, name).data_ptr() != getattr(right, name).data_ptr()
    return cloned


class Receiver(base.Receiver):
    def tokenized(self, scene, extra):
        texts, ids, offsets, positions = [], [], [], []
        for qi in (0, 1):
            text, sites = self.prompt(scene, qi, extra)
            enc = self.tok(text, return_offsets_mapping=True,
                           add_special_tokens=not self.chat)
            texts.append(text)
            ids.append(enc['input_ids'])
            offsets.append(enc['offset_mapping'])
            positions.append(sites)
        n = prefix_length(texts, ids, offsets, positions)
        return texts, ids, positions[0], n

    def decode(self, logits, cache, scene, budget):
        aliases = [self.tok(' ' + a, add_special_tokens=False)['input_ids']
                   for a in scene['aliases']]
        logprobs = torch.log_softmax(logits, -1)
        pieces = []
        for tokens in aliases:
            values = [float(logprobs[tokens[0]])]
            branch = clone_cache(cache)
            for prev, target in zip(tokens[:-1], tokens[1:]):
                step = self.model(torch.tensor([[prev]], device='cuda:0'),
                                  past_key_values=branch, use_cache=True)
                branch = step.past_key_values
                values.append(float(torch.log_softmax(step.logits[0, -1].float(), -1)[target]))
            pieces.append(values)
        branch = clone_cache(cache)
        current = logits
        generated = []
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [self.tok.eos_token_id])
        stop = 'budget'
        for _ in range(budget):
            token = int(current.argmax())
            generated.append(token)
            text = self.tok.decode(generated, skip_special_tokens=True)
            if token in eos:
                stop = 'eos'
                break
            if '\n' in text.lstrip():
                stop = 'line'
                break
            step = self.model(torch.tensor([[token]], device='cuda:0'),
                              past_key_values=branch, use_cache=True)
            branch, current = step.past_key_values, step.logits[0, -1].float()
        return dict(gap=sum(pieces[0]) - sum(pieces[1]),
                    alias_token_ids=aliases, alias_logprob_pieces=pieces,
                    text64=self.tok.decode(generated[:64], skip_special_tokens=True),
                    text256=self.tok.decode(generated, skip_special_tokens=True),
                    tokens=generated, stop=stop,
                    logits_sha256=digest_tensor(logits),
                    truncated64=len(generated) > 64 or (len(generated) == 64 and stop == 'budget'))

    @torch.inference_mode()
    def evaluate_world(self, scene, packet, gamma, extra, zero, verify, limits):
        texts, sequences, positions, n = self.tokenized(scene, extra)
        ids = torch.tensor([sequences[0][:n]], device='cuda:0')
        report = {}
        with self.hooks(positions, packet, gamma, 'band', zero, report) as hooks:
            output = self.model(ids, attention_mask=torch.ones_like(ids), use_cache=True)
        assert all(h.handle is None for h in hooks), 'Hooks remain active after prefix'
        writes = sum(h.n_writes for h in hooks)
        if zero:
            assert writes > 0, 'Zero integrity hook did not fire'
        cache = output.past_key_values
        assert cache.get_seq_length() == n
        prefix_hash = cache_fingerprint(cache)
        prefix_ids_hash = base.sha(json.dumps(sequences[0][:n]).encode())
        results = []
        for qi, full_ids in enumerate(sequences):
            branch = clone_cache(cache)
            # The same saved state is used for both questions, with no hooks.
            branch_hash = cache_fingerprint(branch)
            assert branch_hash == prefix_hash
            suffix = torch.tensor([full_ids[n:]], device='cuda:0')
            step = self.model(suffix, past_key_values=branch,
                              attention_mask=torch.ones((1, len(full_ids)), device='cuda:0', dtype=torch.long),
                              use_cache=True)
            logits = step.logits[0, -1].float()
            result = self.decode(logits, step.past_key_values, scene, limits['max_new_tokens'])
            replay = None
            if verify:
                enc = torch.tensor([full_ids], device='cuda:0')
                full_report = {}
                with self.hooks(positions, packet, gamma, 'band', zero, full_report):
                    full = self.model(enc, attention_mask=torch.ones_like(enc), use_cache=True)
                full_logits = full.logits[0, -1].float()
                direct = self.decode(full_logits, full.past_key_values, scene, limits['max_new_tokens'])
                replay = dict(max_logit_difference=float((full_logits - logits).abs().max()),
                              alias_gap_difference=abs(direct['gap'] - result['gap']),
                              tokens_equal=direct['tokens'] == result['tokens'],
                              parsed64_equal=task.score_answer(direct['text64'], scene['aliases'], scene['aliases'][0]) ==
                                             task.score_answer(result['text64'], scene['aliases'], scene['aliases'][0]),
                              parsed256_equal=task.score_answer(direct['text256'], scene['aliases'], scene['aliases'][0]) ==
                                              task.score_answer(result['text256'], scene['aliases'], scene['aliases'][0]),
                              full_text64=direct['text64'], full_text256=direct['text256'])
                # Persist diagnostics even when a predeclared engineering rule fails.
                replay['pass'] = (replay['max_logit_difference'] <= limits['max_logit_difference'] and
                                  replay['alias_gap_difference'] <= limits['max_alias_gap_difference'] and
                                  replay['parsed64_equal'] and replay['parsed256_equal'])
            result.update(prompt_sha256=base.sha(texts[qi].encode()),
                          prefix_token_sha256=prefix_ids_hash, prefix_cache_sha256=prefix_hash,
                          branch_initial_cache_sha256=branch_hash,
                          prefix_tokens=n, prompt_tokens=len(full_ids),
                          hook_writes=writes, sites=report, cached_full_replay=replay)
            results.append(result)
        assert cache_fingerprint(cache) == prefix_hash, 'Question branch mutated original state'
        assert sum(h.n_writes for h in hooks) == writes, 'Write occurred during query/decoding'
        return results


def literal_packet(world, words):
    lanes = literal_codes(world['facts'], words)
    packet = base.Packet(base.inventory_digest(words), 8,
                         tuple(base.Lane(i, ids, q) for i, (ids, q) in enumerate(lanes)))
    assert base.serialize(base.deserialize(base.serialize(packet))) == base.serialize(packet)
    return packet


def atomic_json(path, record):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def run(m, args):
    root = Path(m['root'])
    root.joinpath('raw').mkdir(exist_ok=True)
    pair = m['pairs'][args.pair]
    pdata = json.loads(Path(m['packets']).read_text())
    bank = json.loads(Path(m['bank']).read_text())
    assert len(bank) == 96 and all(s['split'] == 'dev' for s in bank)
    if args.phase == 'engineering':
        ids = set(m['engineering_scene_ids'])
        selected = [s for s in bank if s['scene'] in ids]
        configs = conditions(m['engineering_gammas'])
        assert args.shards == 1
    else:
        receipt = root / f'engineering_p{args.pair}.json'
        assert json.loads(receipt.read_text())['status'] == 'PASS', 'Engineering stage has not passed'
        selected = bank[args.shard::args.shards]
        configs = conditions(m['development_gammas'])
    raw = root / 'raw' / f'{args.phase}_p{args.pair}_s{args.shard}.jsonl'
    marker = raw.with_suffix('.done.json')
    expected = len(selected) * 4 * len(configs)
    done = {}
    if raw.exists():
        for line in raw.read_text().splitlines():
            row = json.loads(line)
            assert row['key'] not in done
            assert not row['cached_full_replay'] or row['cached_full_replay']['pass'], 'Cannot resume past a failed engineering check'
            done[row['key']] = row
    if marker.exists():
        saved = json.loads(marker.read_text())
        assert saved['status'] == 'PASS' and saved['rows'] == len(done) == expected
        assert saved['raw_sha256'] == base.sha(raw.read_bytes())
        print('ALREADY_COMPLETE', raw, flush=True)
        return
    solver_cache = ExactSolverCache(writer_module._solve_a0, verify_every=64)
    writer_module._solve_a0 = solver_cache
    receiver = Receiver(m, pair['receiver'], task)
    assert receiver.model.config.model_type == 'qwen3', 'Dense cache stage only'
    assert receiver.words == pdata['words']
    start = time.time()
    failures = []
    for si, scene in enumerate(selected):
        for world in scene['worlds']:
            packet_record = pdata['packets'][f"{scene['scene']}|{world['world']}"]
            actual = base.deserialize(bytes.fromhex(packet_record['hex']))
            literal = literal_packet(world, receiver.words)
            for arm, gamma in configs:
                keys = [row_key(scene['scene'], world['world'], qi, arm, gamma) for qi in (0, 1)]
                if all(k in done for k in keys):
                    continue
                source = literal if arm.startswith('literal') else actual
                is_write = arm.endswith('_write')
                if arm == 'full_text':
                    extra = ' '.join(world['evidence'])
                elif arm in ('actual_text', 'literal_text'):
                    extra = base.packet_text(source, receiver.words, scene['aliases'], weights=True)
                else:
                    extra = None
                rows = receiver.evaluate_world(scene, source if is_write else None, gamma,
                                               extra, arm == 'zero', args.phase == 'engineering', m['numerics'])
                for qi, (key, result) in enumerate(zip(keys, rows)):
                    row = dict(key=key, pair=args.pair, phase=args.phase, scene=scene['scene'],
                               family=scene['family'], domain=scene['domain'], template=scene['template'],
                               split='exposed_development', world=world['world'], question=qi,
                               aliases=scene['aliases'], gold=world['answers'][qi],
                               arm=arm, gamma=gamma, magnitude=receiver.mag,
                               actual_sender_packet_sha256=packet_record['sha256'],
                               actual_sender_payload_bytes=packet_record['bytes'],
                               condition_packet_sha256=base.sha(base.serialize(source)) if arm.startswith(('actual', 'literal')) else None,
                               represented_packet_bytes=len(base.serialize(source)) if arm.startswith(('actual', 'literal')) else 0,
                               wire_bytes=len(base.serialize(source)) if is_write else len(extra.encode()) if extra else 0,
                               extra_text=extra, extra_text_bytes=len(extra.encode()) if extra else 0,
                               **result)
                    row['score64'] = task.score_answer(row['text64'], scene['aliases'], row['gold'])
                    row['score256'] = task.score_answer(row['text256'], scene['aliases'], row['gold'])
                    if arm == 'zero':
                        ref = done[row_key(scene['scene'], world['world'], qi, 'none', 0)]
                        assert all(row[k] == ref[k] for k in ('prefix_cache_sha256', 'logits_sha256', 'tokens')), 'Zero write changed state/output'
                    if result['cached_full_replay'] and not result['cached_full_replay']['pass']:
                        failures.append(key)
                    if key not in done:
                        with raw.open('a') as f:
                            f.write(json.dumps(row) + '\n')
                            f.flush()
                        done[key] = row
                    else:
                        assert all(done[key][k] == row[k] for k in ('prefix_cache_sha256', 'logits_sha256', 'tokens')), 'Partial-row resume drift'
                if failures:
                    atomic_json(root / f'engineering_p{args.pair}.json', dict(status='FAIL', keys=failures, raw=str(raw)))
                    raise RuntimeError(f'Cached/full engineering gate failed: {failures}')
        print('SCENE', args.phase, args.pair, args.shard, si + 1, len(selected),
              'rows', len(done), 'seconds', round(time.time() - start, 1), flush=True)
    assert len(done) == expected, (len(done), expected)
    receipt = dict(status='PASS', stage=args.phase, pair=args.pair, shard=args.shard,
                   scenes=len(selected), rows=expected, raw_sha256=base.sha(raw.read_bytes()),
                   solver_hits=solver_cache.hits, solver_verified=solver_cache.verified,
                   manifest_sha256=base.sha(Path(args.manifest).read_bytes()), elapsed_seconds=time.time() - start)
    atomic_json(marker, receipt)
    if args.phase == 'engineering':
        atomic_json(root / f'engineering_p{args.pair}.json', receipt)
    print('COMPLETE', json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--phase', choices=['engineering', 'development'], required=True)
    parser.add_argument('--pair', type=int, choices=[0, 1], required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    m = json.loads(args.manifest.read_text())
    assert 0 <= args.shard < args.shards
    for path, digest in m['pins'].items():
        assert base.sha(Path(path).read_bytes()) == digest, path
    torch.set_num_threads(4)
    # Numerical regime is versioned in the manifest. Do not change thresholds
    # when a faster arithmetic path fails the cached/full replay check.
    torch.backends.cuda.matmul.allow_tf32 = m['numerics']['allow_tf32']
    torch.backends.cudnn.allow_tf32 = m['numerics']['allow_tf32']
    run(m, args)


if __name__ == '__main__':
    main()
