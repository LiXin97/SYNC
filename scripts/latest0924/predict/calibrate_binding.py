"""Experiment F calibration probe v2 (amendment 1): can the receiver BIND a symbol written at relative depth d to its object?

Task-independent: synthetic two-object prompt (fixed names XOPA/YUME, all slots neutral) from the frozen renderer and
receiver scaffold; no scene, bank, answer or outcome. The 221 inventory symbols are paired by a fixed permutation
(seed 20260925; the last symbol is dropped): 110 pairs (s_a, s_b). For each pair and depth d, write h0 + d_{s_a} at
XOPA slot 0 and h0 + d_{s_b} at YUME slot 0 (final tokens, gamma 1, raw fitted directions, the evaluator's own write
hook, single layer at depth d), then ask the generic questions "Which object is {s_a}?" (gold XOPA) and
"Which object is {s_b}?" (gold YUME) from the cached prefix. Answers are decoded and scored with the frozen parser.
Metrics per depth: bind_pair_success (both answers correct; PRIMARY), bind_answer_accuracy, mean signed alias
log-prob gap toward gold, and the no-write reference.

Usage: PYTHONPATH=$R/src $PY scripts/latest0924/predict/calibrate_binding.py --receiver KEY --out <json>
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import syncomm.recovery.evaluate_depth as ED  # noqa: E402
from syncomm.latest import adapter  # noqa: E402
from calibrate import DEPTHS, spec  # noqa: E402

ALIASES = ['XOPA', 'YUME']
BUDGET = 16


def pairs(words):
    idx = list(range(len(words)))
    random.Random(20260925).shuffle(idx)
    return [(idx[2 * i], idx[2 * i + 1]) for i in range(len(idx) // 2)]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--receiver', required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    adapter.install(ED)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    sp = spec(a.receiver)
    words = json.loads((ROOT / 'artifacts/receiver_recovery/words_221.json').read_text())
    receiver = ED.build_receiver(sp, a.receiver, words)
    assert len(ED.base.decoder_layers(receiver.model)) == sp['n_blocks']
    blob = torch.load(sp['directions'], map_location='cpu', weights_only=False)
    assert blob['words'] == words
    order = {l: i for i, l in enumerate(blob['layers'])}
    D = blob['directions'].float().to('cuda:0')
    layers = sp['depth_layers']
    P = pairs(words)[: a.limit or None]
    out = dict(receiver=a.receiver, depths=DEPTHS, layers=layers, n_pairs=len(P), budget=BUDGET,
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               directions_sha256=hashlib.sha256(Path(sp['directions']).read_bytes()).hexdigest(), per_depth={}, no_write={})
    started = time.time()
    per = {str(d): [] for d in DEPTHS}; none = []
    for pi, (ia, ib) in enumerate(P):
        scene = dict(aliases=ALIASES, questions=[f'Which object is {words[ia]}?', f'Which object is {words[ib]}?'])
        ni = ED.prompt_inputs(receiver, scene, None)
        pos = {(s['object'], s['index']): s['last_position'] for s in ni['slots']}
        positions = sorted([pos[0, 0], pos[1, 0]])
        neutral_states = {}
        with ED.capture_hooks(receiver.model, layers, neutral_states):
            neutral_out = ED.forward(receiver, ni['prefix'])

        def answers(out_obj):
            cache = out_obj.past_key_values; res = []
            for qi, suffix in enumerate(ni['suffixes']):
                branch = ED.prior.clone_cache(cache)
                end = receiver.model(torch.tensor([suffix], device='cuda:0'), past_key_values=branch, use_cache=True,
                                     attention_mask=torch.ones((1, len(ni['prefix']) + len(suffix)), device='cuda:0', dtype=torch.long))
                v = receiver.decode(end.logits[0, -1].float(), end.past_key_values, scene, BUDGET)
                gold = ALIASES[qi]
                ok = ED.task.score_answer(v['text64'], ALIASES, gold)
                ok = ok['correct'] if isinstance(ok, dict) else bool(ok)
                res.append(dict(correct=ok, gap_to_gold=v['gap'] if qi == 0 else -v['gap'], text=v['text64']))
            return res

        none.append(answers(neutral_out))
        for d, layer in zip(DEPTHS, layers):
            st = neutral_states[layer].clone()
            st[0, pos[0, 0]] += D[order[layer], ia].to(st)
            st[0, pos[1, 0]] += D[order[layer], ib].to(st)
            rep = {}
            with ED.write_hooks(receiver.model, [layer], {layer: st}, positions, rep):
                o = ED.forward(receiver, ni['prefix'])
            per[str(d)].append(answers(o))
        if pi % 10 == 0:
            print(a.receiver, 'pair', pi, len(P), f'{time.time() - started:.0f}s',
                  {d: round(sum(all(x['correct'] for x in r) for r in per[d]) / len(per[d]), 3) for d in per}, flush=True)

    def summ(rs):
        return dict(bind_pair_success=sum(all(x['correct'] for x in r) for r in rs) / len(rs),
                    bind_answer_accuracy=sum(x['correct'] for r in rs for x in r) / (2 * len(rs)),
                    mean_gap_to_gold=sum(x['gap_to_gold'] for r in rs for x in r) / (2 * len(rs)))
    for d, layer in zip(DEPTHS, layers):
        out['per_depth'][str(d)] = dict(layer=layer, **summ(per[str(d)]))
    out['no_write'] = summ(none)
    out['answers'] = dict(no_write=none, **per)
    out['elapsed_seconds'] = time.time() - started
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out) + '\n')
    print('DONE', a.receiver, {d: round(v['bind_pair_success'], 3) for d, v in out['per_depth'].items()}, 'no-write', round(out['no_write']['bind_pair_success'], 3))


if __name__ == '__main__':
    main()
