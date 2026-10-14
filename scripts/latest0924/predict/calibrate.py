"""Experiment F calibration probe: can a later token read a symbol written at relative depth d?

Task-independent: a synthetic two-object prompt built by the frozen renderer (fixed nonsense names XOPA/YUME, every slot
neutral), the frozen receiver prompt scaffold, and a generic readout question ("What word is listed for XOPA?") instead
of any task question. No scene, bank, answer or outcome is read. For each inventory symbol s and depth d we write
h' = h0 + d_s at XOPA's first slot (final token), at the single layer of depth d (gamma 1, raw fitted direction, the
evaluator's own write hook), and read the next-token logits at the answer position. The symbol's score is its logit
gain over the neutral prompt (max over the with/without-leading-space first token). AUC_s = fraction of the other
inventory symbols (with disjoint first tokens) whose gain is lower (ties count half). Metrics per depth: mean AUC
(PRIMARY), top-1 rate, and the geometry ratio ||d_s|| / ||h0 at the slot|| (secondary).

Usage (GPU node, via the adapter so all receivers load as in Experiments A/B):
  PYTHONPATH=$R/src $PY scripts/latest0924/predict/calibrate.py --receiver KEY --out artifacts/latest0924/predict/calibration/KEY.json
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
import syncomm.recovery.evaluate_depth as ED  # noqa: E402
from syncomm.latest import adapter  # noqa: E402

DEPTHS = [0.2, 0.35, 0.5, 0.65, 0.8]
DEV = {'qwen3-32b': 'Qwen3-32B', 'qwen3-14b-base': 'Qwen3-14B-Base', 'glm-4-9b': 'GLM-4-9B-0414', 'gemma-3-27b-it': 'gemma-3-27b-it'}
PROBE_SCENE = dict(aliases=['XOPA', 'YUME'], questions=['What word is listed for XOPA?', 'What word is listed for YUME?'])


def spec(key):
    if key in DEV:
        m = json.loads((ROOT / f'artifacts/receiver_recovery/r1/{key}/manifest.json').read_text())
        layers = {c[0]: c[2][0] for c in m['configs'] if c[0] != 'band'}
        return dict(models={key: m['models'][key]}, base_models=m['base_models'], n_blocks=m['n_blocks'],
                    directions=m['directions'], depth_layers=[layers[f'd{int(round(d * 100))}'] for d in DEPTHS])
    fm = json.loads((ROOT / 'artifacts/latest0924/predict/fit_manifest.json').read_text())
    return dict(models={key: fm['models'][key]}, base_models=[], n_blocks=fm['n_blocks'][key],
                directions=str(ROOT / f'artifacts/latest0924/predict/fits/directions_{key}.pt'), depth_layers=fm['layers'][key])


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--receiver', required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    installed = adapter.install(ED)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    sp = spec(a.receiver)
    words = json.loads((ROOT / 'artifacts/receiver_recovery/words_221.json').read_text())
    receiver = ED.build_receiver(sp, a.receiver, words)
    assert len(ED.base.decoder_layers(receiver.model)) == sp['n_blocks']
    blob = torch.load(sp['directions'], map_location='cpu', weights_only=False)
    assert blob['words'] == words
    order = {l: i for i, l in enumerate(blob['layers'])}
    D = blob['directions'].float()
    ni = ED.prompt_inputs(receiver, PROBE_SCENE, None)
    ids = ni['prefix'] + ni['suffixes'][0]
    slot = [s for s in ni['slots'] if s['object'] == 0 and s['index'] == 0][0]
    pos = slot['last_position']
    tok = receiver.tok
    variants = []
    for w in words:
        v = {tok.encode(w, add_special_tokens=False)[0], tok.encode(' ' + w, add_special_tokens=False)[0]}
        variants.append(sorted(v))
    layers = sp['depth_layers']
    neutral_states = {}
    with ED.capture_hooks(receiver.model, layers, neutral_states):
        base_logits = ED.forward(receiver, ids).logits[0, -1].float()
    out = dict(receiver=a.receiver, depths=DEPTHS, layers=layers, probe_scene=PROBE_SCENE, write_slot=dict(object=0, index=0, position=pos),
               n_prompt_tokens=len(ids), adapter_priors=len(installed['priors']),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               directions_sha256=hashlib.sha256(Path(sp['directions']).read_bytes()).hexdigest(), per_depth={})
    started = time.time()
    for d, layer in zip(DEPTHS, layers):
        h0 = neutral_states[layer]
        # identity write must be exactly inert
        rep = {}
        with ED.write_hooks(receiver.model, [layer], {layer: h0}, [pos], rep):
            idl = ED.forward(receiver, ids).logits[0, -1].float()
        identity_max = float((idl - base_logits).abs().max())
        gains, geom = [], []
        dirs = D[order[layer]]
        for si in range(len(words)):
            st = h0.clone(); st[0, pos] += dirs[si].to(st)
            rep = {}
            with ED.write_hooks(receiver.model, [layer], {layer: st}, [pos], rep):
                lg = ED.forward(receiver, ids).logits[0, -1].float()
            g = (lg - base_logits).cpu()
            gains.append(g)
            geom.append(float(dirs[si].norm() / h0[0, pos].float().norm()))
        aucs, top1 = [], []
        for si in range(len(words)):
            g = gains[si]
            score = lambda j: max(float(g[t]) for t in variants[j])
            own = score(si)
            others = [j for j in range(len(words)) if j != si and not set(variants[j]) & set(variants[si])]
            vals = [score(j) for j in others]
            aucs.append(sum(1.0 if v < own else 0.5 if v == own else 0.0 for v in vals) / len(vals))
            top1.append(all(v < own for v in vals))
        out['per_depth'][str(d)] = dict(layer=layer, auc_mean=sum(aucs) / len(aucs), top1_rate=sum(top1) / len(top1),
                                        geometry_ratio_mean=sum(geom) / len(geom), identity_max_logit_diff=identity_max,
                                        auc_per_symbol=aucs)
        print(a.receiver, d, layer, 'AUC', round(out['per_depth'][str(d)]['auc_mean'], 4), 'top1', round(sum(top1) / len(top1), 3),
              'identity', identity_max, f'{time.time() - started:.0f}s', flush=True)
        assert identity_max == 0.0, 'identity write is not inert'
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    print('DONE', a.out)


if __name__ == '__main__':
    main()
