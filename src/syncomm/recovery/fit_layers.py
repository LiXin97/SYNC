"""R0/R1 fitter: receiver-local word contrasts at an ARBITRARY set of decoder layers, one pass per context.

Same recipe as the frozen fitter `artifacts/mechanism0920/word_memory_fit_v1/snapshot/fit_word_memory.py` in every
scientific respect — the same twelve generic contexts (widths 2 and 4, both objects, every rank), the same word-final
token read-out, the same token-length-matched ` unknown` neutral comparison, the same fixed deterministic mean over the
twelve contrasts, the same fitting budget, and no scene, question, answer or sender state anywhere. The single change is
that the capture hook is installed on a declared LIST of layers instead of the receiver's three-layer band, so that one
forward serves every layer the depth scan needs. Identical arithmetic per layer, so the shared layers must reproduce the
frozen fit bit-for-bit; `--verify-against` asserts exactly that.

Per word it also stores the twelve per-context contrasts, which is what the R2 local scale
    c[l] = median over admitted (w,t) of ||delta[l,w,t]|| / ||d[l,w]||
is computed from (scripts/recovery_local_scale.py), so no extra forward is ever needed for the scale rule.

Usage (one receiver, one shard):
  python3 -m syncomm.recovery.fit_layers --manifest <root>/manifest.json --receiver qwen3-32b --shard 0 --shards 4
"""
import argparse
import contextlib
import importlib.util
import json
import os
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
from memory_layout import render, token_spans, NEUTRAL_WORD   # noqa: E402  (frozen layout, unchanged)

CANONICAL = dict(aliases=['XOPA', 'YUME'], questions=['FITTING_QUESTION_A?', 'FITTING_QUESTION_B?'])
CONTEXT_WIDTHS = (2, 4)


def construction(receiver, word, width, obj, index):
    """Byte-identical to the frozen fitter's `construction`."""
    selected = [[NEUTRAL_WORD] * width for _ in (0, 1)]
    selected[obj][index] = word
    message, chars = render(CANONICAL['aliases'], selected, width=width)
    texts, sequences, _, n = receiver.tokenized(CANONICAL, message)
    enc = receiver.tok(texts[0], return_offsets_mapping=True, add_special_tokens=not receiver.chat)
    assert enc['input_ids'] == sequences[0]
    spans = token_spans(enc['offset_mapping'], chars, texts[0].index(message))
    chosen = next(s for s in spans if s['object'] == obj and s['index'] == index)
    assert chosen['word'] == word and max(chosen['positions']) < n
    source = sequences[0][:n]
    filler = receiver.tok(' unknown', add_special_tokens=False)['input_ids']
    assert len(filler) == 1
    neutral = list(source)
    for pos in chosen['positions']:
        neutral[pos] = filler[0]
    assert all(a == b for i, (a, b) in enumerate(zip(source, neutral)) if i not in chosen['positions'])
    assert len(source) == len(neutral)
    return dict(source=source, neutral=neutral, position=chosen['last_position'],
                word_positions=chosen['positions'], width=width, object=obj, index=index)


@contextlib.contextmanager
def capture(model, layers, position, output):
    handles = []
    for layer in layers:
        def hook(module, args, result, layer=layer):
            h = result[0] if isinstance(result, tuple) else result
            output[layer] = h[0, position].detach().float().cpu().clone()
        handles.append(base.decoder_layers(model)[layer].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def read(model, layers, ids, position):
    states = {}
    t = torch.tensor([ids], device='cuda:0')
    with capture(model, layers, position, states):
        model(t, attention_mask=torch.ones_like(t), use_cache=False)
    assert set(states) == set(layers)
    return torch.stack([states[layer] for layer in layers])


def load_receiver(m, key):
    receiver = prior.Receiver.__new__(prior.Receiver)
    receiver.model, receiver.tok = base.load_model(m['models'][key])
    receiver.chat = key not in m['base_models']
    receiver.task = parent.Scaffold
    return receiver


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--receiver', required=True)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    p.add_argument('--verify-against', type=Path, default=None, help='a frozen directions_p*.pt to reproduce at its layers')
    a = p.parse_args()
    m = json.loads(a.manifest.read_text())
    for path, sha in m['pins'].items():
        assert base.sha(Path(path).read_bytes()) == sha, path
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    key = a.receiver
    receiver = load_receiver(m, key)
    n_blocks = len(base.decoder_layers(receiver.model))
    layers = m['layers'][key]
    assert n_blocks == m['n_blocks'][key], (n_blocks, m['n_blocks'][key])
    assert layers == sorted(set(layers)) and all(0 < l < n_blocks for l in layers)
    words = json.loads(Path(m['words_file']).read_text())
    assert len(words) == 221 and len(set(words)) == len(words) and NEUTRAL_WORD not in words
    target = Path(m['root']) / 'fits' / key
    target.mkdir(parents=True, exist_ok=True)
    manifest_sha = base.sha(a.manifest.read_bytes())
    reference = None
    if a.verify_against is not None:
        blob = torch.load(a.verify_against, map_location='cpu', weights_only=False)
        reference = dict(layers=blob['layers'], words=blob['words'], directions=blob['directions'].float())
        assert reference['words'] == words
        assert set(reference['layers']) <= set(layers), (reference['layers'], layers)
    cache = {}
    started = time.time(); records = []; checked = 0
    addresses = list(range(a.shard, len(words), a.shards))
    for address in addresses:
        word = words[address]; dest = target / f'w{address:03d}.pt'
        if dest.exists():
            stored = torch.load(dest, map_location='cpu', weights_only=False)
            assert stored['manifest_sha256'] == manifest_sha and stored['address'] == address and stored['word'] == word and stored['layers'] == layers
            records.append(dict(address=address, path=str(dest), sha256=base.sha(dest.read_bytes())))
            continue
        contrasts = []; contexts = []
        for width in CONTEXT_WIDTHS:
            for obj in (0, 1):
                for index in range(width):
                    c = construction(receiver, word, width, obj, index)
                    cachekey = (tuple(c['neutral']), c['position'])
                    if cachekey not in cache:
                        cache[cachekey] = read(receiver.model, layers, c['neutral'], c['position'])
                    source_state = read(receiver.model, layers, c['source'], c['position'])
                    delta = source_state - cache[cachekey]
                    assert torch.isfinite(delta).all()
                    contrasts.append(delta); contexts.append(c)
        assert len(contrasts) == 12
        stacked = torch.stack(contrasts)                      # [12, L, hidden]
        direction = stacked.double().mean(dim=0).float()      # [L, hidden]  (frozen fitter's exact expression)
        if reference is not None:
            for li, layer in enumerate(reference['layers']):
                mine = direction[layers.index(layer)]; theirs = reference['directions'][li, address]
                if not torch.equal(mine, theirs):
                    diff = float((mine - theirs).abs().max())
                    assert diff <= 1e-4, f'layer {layer} word {word}: multi-layer capture changed the fit by {diff}'
                checked += 1
        blob = dict(manifest_sha256=manifest_sha, receiver=key, address=address, word=word, layers=layers,
                    contexts=contexts, contrasts=stacked, direction=direction,
                    construction='Mean of 12 receiver-only one-word-minus-token-length-matched-neutral contrasts at the final word token; '
                                 'widths 2 and 4, all object/rank positions; identical recipe to word_memory_fit_v1, captured at several layers in one pass.')
        temporary = dest.with_suffix('.tmp'); torch.save(blob, temporary); os.replace(temporary, dest)
        records.append(dict(address=address, path=str(dest), sha256=base.sha(dest.read_bytes())))
        print('WORD', address, word, 'norms', [round(float(x), 2) for x in direction.norm(dim=1)], flush=True)
    receipt = dict(status='COMPLETE_RECEIVER_LOCAL_FIT_ONLY', receiver=key, layers=layers, n_blocks=n_blocks,
                   manifest_sha256=manifest_sha, shard=a.shard, shards=a.shards, words=len(records), rows=records,
                   verified_against=str(a.verify_against) if reference else None, verified_cells=checked,
                   elapsed_seconds=time.time() - started, no_communication_outcomes=True)
    prior.atomic_json(Path(m['root']) / 'fits' / f'fit_{key}_s{a.shard}.done.json', receipt)
    print('COMPLETE', key, a.shard, len(records), 'verified', checked, flush=True)


if __name__ == '__main__':
    main()
