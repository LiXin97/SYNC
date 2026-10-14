"""Screen ONE adapter model through the patched FROZEN evaluator primitives before any grid cell uses it.

What it does (GPU mode, default). Loads the receiver exactly as src/syncomm/recovery/evaluate_depth.build_receiver does
(after adapter.install), renders one real R3 scene with the frozen prompt_inputs, and checks, at the relative-depth-0.2
layer the experiment writes to:
  structure      decoder ModuleList found, length == declared n_layers (== the expA lock), depth-0.2 layer and the
                 0.4/0.5/0.6 band from the frozen rounding (== the lock's selected_layer / original_band)
  hooks          a capture hook and a write hook both fire at the depth-0.2 layer
  identity       the frozen identity write (captured neutral state written back at the active slots) leaves the prefix
                 logits bit-identical and the prefix cache fingerprint identical to the no-message run
  write          a fitted-style write (the receiver's own fitted directions if artifacts/latest0924/fits has them, else
                 random unit directions at the residual norm) changes the logits
  frozen pass    evaluate_confirm.scene_rows in 'engineering' phase on that scene with the complete-fact source and arms
                 none, identity_write, original, selected, packet_text, full_text, sham0, other_world, native: the
                 evaluator's own cached-prefix + cloned-cache question continuation versus a complete forward
                 (max logit difference <= 0.05, alias gap difference <= 0.1, same parsed first-line answer at 64/256
                 tokens), its identity-inertness and sham norm-matching assertions, and the original cache unchanged
  answer format  the frozen greedy decode yields a parseable alias (first line) in >= 3 of the 4 full_text rows; a
                 failure here means the chat rendering / thinking control is wrong for this model
  fit prompts    (only if fits/<key>/w000.pt exists) the twelve fitting contexts rebuilt now are token-identical to the
                 stored ones, and word 0's direction recomputed now is compared with the stored direction (relative
                 difference reported; > 1e-3 is a warning, not a failure: the fit may have run in another environment)
It records load time, per-prefix forward time, peak GPU memory per device, parameter dtypes, the cache class and layer
types, and writes <out>/<key>.json with status PASS / FAIL (check failed) / ERROR (exception; see `stage`).
Skip-if-done: if <out>/<key>.json exists the script exits 0 (use --retry-errors to rerun an ERROR verdict).

CPU mode (--cpu; no GPU, no weights): meta-device instantiation and decoder path, checkpoint key/shape coverage of the
declared loader (emulated with transformers' own renaming rules on safetensors headers), tokenizer + chat rendering
through the frozen prompt pipeline on every R3 scene (neutral, complete-fact source, padded packet text, evidence text),
all 221 x 12 fitting constructions, stop tokens, the tekken conversion against mistral_common when available, a tiny
random-weight model of the same architecture through the frozen hooks and the adapter cache functions, and byte
identity of the cache functions with the frozen ones on a synthetic dense DynamicCache. Writes <out>/<key>.cpu.json.

Usage (node; absolute paths; interpreter = adapter.MODELS[key]['python']; one model per invocation):
  R=.
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/src $PY $R/scripts/latest0924/screen.py --model qwen3.5-2b --out $R/artifacts/latest0924/screen
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$R/src $PY $R/scripts/latest0924/screen.py --model qwen3.5-122b-a10b --out ...
  PYTHONPATH=$R/src $PY $R/scripts/latest0924/screen.py --cpu --model KEY --out DIR
"""
try:
    from syncomm.models import resolve_model
except ImportError:  # run outside the package: references must already be local paths
    def resolve_model(ref):
        return str(ref)
import argparse
import contextlib
import copy
import json
import os
import platform
import socket
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

BANK = ROOT / 'artifacts/receiver_recovery/r3_bank/bank.json'
WORDS = ROOT / 'artifacts/receiver_recovery/words_221.json'
R3_MANIFEST = ROOT / 'artifacts/receiver_recovery/r3/qwen3-32b/manifest.json'
CANDIDATE_A = ROOT / 'artifacts/receiver_recovery/r3_packets/candidateA_r3.json'
LOCK = ROOT / 'gates/latest0924/expA_protocol.lock.json'
FITS = ROOT / 'artifacts/latest0924/fits'
DEFAULT_SCENE = '000ab0c6e25c722375c5'   # the first R3 engineering scene
LIMITS = dict(max_logit_difference=.05, max_alias_gap_difference=.1)   # evaluate_confirm's engineering thresholds
LITERAL_ARMS = ['none', 'identity_write', 'original', 'selected', 'packet_text', 'full_text', 'sham0', 'other_world',
                'native']


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n')
    os.replace(tmp, path)


def ensure_sklearn():
    """syncomm.solver imports sklearn at module level; CPU mode on a box without it gets a stub (the solver never runs)."""
    try:
        import sklearn  # noqa: F401
        return False
    except ImportError:
        import importlib.machinery
        import types
        mods = {n: types.ModuleType(n) for n in ('sklearn', 'sklearn.exceptions', 'sklearn.linear_model',
                                                 'sklearn.metrics')}

        class ConvergenceWarning(UserWarning):
            pass

        def unavailable(*a, **k):
            raise RuntimeError('sklearn stub (screen --cpu)')
        mods['sklearn.exceptions'].ConvergenceWarning = ConvergenceWarning
        mods['sklearn.linear_model'].ElasticNet = unavailable
        mods['sklearn.metrics'].roc_curve = unavailable   # transformers' candidate_generator imports it
        for name, mod in mods.items():   # torch._dynamo calls find_spec on loaded module names: give each a spec
            mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=name == 'sklearn')
            if name == 'sklearn':
                mod.__path__ = []
            else:
                setattr(mods['sklearn'], name.split('.')[1], mod)
        sys.modules.update(mods)
        return True


def frozen_modules():
    """Import the three frozen evaluators and install the adapter into all of them."""
    import importlib
    from syncomm.latest import adapter as A
    ED = importlib.import_module('syncomm.recovery.evaluate_depth')
    EC = importlib.import_module('syncomm.recovery.evaluate_confirm')
    FL = importlib.import_module('syncomm.recovery.fit_layers')
    installed = A.install(ED, EC, FL)
    assert len(installed['priors']) >= 2, installed
    return A, ED, EC, FL


def lock_entry(key):
    if not LOCK.exists():
        return None
    return json.loads(LOCK.read_text()).get('receivers', {}).get(key)


def env_record():
    import torch
    import transformers
    rec = dict(python=sys.executable, transformers=transformers.__version__, torch=torch.__version__,
               host=socket.gethostname(), platform=platform.platform(), utc=utc(),
               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    import importlib.util
    rec['fla_importable'] = importlib.util.find_spec('fla') is not None
    rec['kernels_importable'] = importlib.util.find_spec('kernels') is not None
    if torch.cuda.is_available():
        rec['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return rec


def fake_receiver(ED, tok, words, key):
    r = ED.prior.Receiver.__new__(ED.prior.Receiver)
    r.tok, r.chat, r.task, r.words, r.cache, r.key = tok, True, ED.parent.Scaffold, list(words), {}, key
    return r


# ======================================================================================================================
# GPU mode
# ======================================================================================================================
def gpu_memory():
    import torch
    return {f'cuda:{i}': dict(peak_gib=round(torch.cuda.max_memory_allocated(i) / 2**30, 2),
                              now_gib=round(torch.cuda.memory_allocated(i) / 2**30, 2))
            for i in range(torch.cuda.device_count())}


def directions_for(key, layers, words, hidden, neutral_norm, device):
    """Real fitted directions [len(layers), 221, hidden] if the fit exists at every layer, else random unit directions
    scaled to the median neutral residual norm at the write sites (a fitted-style write of realistic size)."""
    import torch
    path = FITS / f'directions_{key}.pt'
    if path.exists():
        blob = torch.load(path, map_location='cpu', weights_only=False)
        if blob.get('words') == words and set(layers) <= set(blob['layers']):
            order = {l: i for i, l in enumerate(blob['layers'])}
            return blob['directions'].float()[[order[l] for l in layers]].to(device), f'fitted {path.name}'
    g = torch.Generator().manual_seed(20260924)
    d = torch.randn(len(layers), len(words), hidden, generator=g)
    d = d / d.norm(dim=-1, keepdim=True) * float(neutral_norm)
    return d.to(device), f'synthetic random unit x median neutral norm {float(neutral_norm):.3g}'


def gpu_screen(a, key, verdict):
    import torch
    stage = 'import'
    verdict['stage'] = stage
    A, ED, EC, FL = frozen_modules()
    spec = A.MODELS[key]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checks, reasons, info = verdict['checks'], verdict['reasons'], verdict['info']
    info['env_check'] = A.check_env(key)
    words = json.loads(WORDS.read_text())
    bank = json.loads(BANK.read_text())
    scene = next(s for s in bank if s['scene'] == a.scene)
    r3 = json.loads(R3_MANIFEST.read_text())

    verdict['stage'] = 'load'
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    t0 = time.time()
    m = dict(models={key: resolve_model(spec['path'])}, base_models=[])
    receiver = ED.build_receiver(m, key, words)        # base.load_model (patched) + frozen band
    info['load_seconds_total'] = round(time.time() - t0, 1)
    model, tok = receiver.model, receiver.tok
    info['adapter_load'] = model._syncomm_latest
    info['memory_after_load'] = gpu_memory()
    dt = Counter()
    for p in model.parameters():
        dt[str(p.dtype)] += p.numel()
    info['param_dtype_numel'] = dict(dt)
    info['param_devices'] = sorted({str(p.device) for p in model.parameters()})
    info['model_class'] = type(model).__name__
    lin = [m_ for m_ in model.modules() if hasattr(m_, 'chunk_gated_delta_rule')]
    if lin:
        f = lin[0].chunk_gated_delta_rule
        info['gated_delta_impl'] = f'{getattr(f, "__module__", "?")}.{getattr(f, "__name__", type(f).__name__)}'

    verdict['stage'] = 'structure'
    layers = ED.base.decoder_layers(model)
    n = len(layers)
    L = ED.base.tap_layer(model, .2)
    band = receiver.band
    info.update(n_layers=n, depth02_layer=L, band=band, layer_class=type(layers[0]).__name__)
    lk = lock_entry(key)
    checks['structure'] = dict(n_layers=n, declared=spec['n_layers'], lock=lk and dict(
        n_layers=lk['n_layers'], selected_layer=lk['selected_layer'], original_band=lk['original_band']))
    ok = n == spec['n_layers'] and (lk is None or (n == lk['n_layers'] and L == lk['selected_layer']
                                                   and band == lk['original_band']))
    checks['structure']['pass_'] = ok
    if not ok:
        reasons.append(f'structure: n={n} L02={L} band={band} vs declared/lock {checks["structure"]}')

    verdict['stage'] = 'prompt'
    ni = ED.prompt_inputs(receiver, scene, None)
    text0, _ = receiver.prompt(scene, 0, ni['message'])
    info['prompt'] = dict(scene=scene['scene'], prefix_tokens=ni['n'], suffix_tokens=[len(s) for s in ni['suffixes']],
                          rendered_head=text0[:200], rendered_tail=text0[-260:])
    world = scene['worlds'][0]
    packet = EC.literal_packet_generic(world, words)
    active, positions, mapping = ED.entry_positions(ED.packet_entries(packet, words), ni['slots'])
    info['write_positions'] = positions

    verdict['stage'] = 'hooks'
    store = {}
    with torch.inference_mode():
        with ED.capture_hooks(model, [L], store):
            torch.cuda.synchronize()
            t0 = time.time()
            neutral = ED.forward(receiver, ni['prefix'])
            torch.cuda.synchronize()
            info['prefix_forward_seconds_first'] = round(time.time() - t0, 3)
        times = []
        for _ in range(2):
            torch.cuda.synchronize()
            t0 = time.time()
            ED.forward(receiver, ni['prefix'])
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        info['prefix_forward_seconds'] = round(sum(times) / len(times), 3)
        fp0 = ED.prior.cache_fingerprint(neutral.past_key_values)
        info['cache'] = A.cache_summary(neutral.past_key_values)
        hid = store[L].shape[-1] if L in store else None
        checks['capture_hook'] = dict(pass_=L in store and list(store[L].shape) == [1, ni['n'], hid],
                                      shape=list(store[L].shape) if L in store else None)
        rep = {}
        with ED.write_hooks(model, [L], {L: store[L]}, positions, rep):
            ident = ED.forward(receiver, ni['prefix'])
        fp1 = ED.prior.cache_fingerprint(ident.past_key_values)
        checks['identity_write'] = dict(hook_fired=str(L) in rep, norm=rep.get(str(L), {}).get('norm'),
                                        logits_bit_identical=bool(torch.equal(ident.logits, neutral.logits)),
                                        cache_fingerprint_equal=fp1 == fp0)
        checks['identity_write']['pass_'] = (checks['identity_write']['hook_fired'] and
                                             checks['identity_write']['norm'] is not None and
                                             checks['identity_write']['norm'] <= 1e-6 and
                                             checks['identity_write']['logits_bit_identical'] and
                                             checks['identity_write']['cache_fingerprint_equal'])
        site_norm = store[L][0, positions].float().norm(dim=-1).median()
        all_layers = sorted(set(band) | {L})
        D, source = directions_for(key, all_layers, words, hid, site_norm, 'cuda:0')
        info['directions'] = source
        order = {l: i for i, l in enumerate(all_layers)}
        dsel = D[[order[L]]]
        states = ED.fitted_states({L: store[L]}, [L], dsel, [1.0], active, mapping, 1)
        rep2 = {}
        with ED.write_hooks(model, [L], states, positions, rep2):
            fitted = ED.forward(receiver, ni['prefix'])
        move = float((fitted.logits[0, -1].float() - neutral.logits[0, -1].float()).abs().max())
        checks['fitted_write'] = dict(hook_fired=str(L) in rep2, norm=rep2.get(str(L), {}).get('norm'),
                                      max_prefix_logit_change=move,
                                      cache_fingerprint_differs=ED.prior.cache_fingerprint(fitted.past_key_values) != fp0)
        checks['fitted_write']['pass_'] = bool(str(L) in rep2 and rep2[str(L)]['norm'] > 0 and move > 1e-3)
        del neutral, ident, fitted
    for name in ('capture_hook', 'identity_write', 'fitted_write'):
        if not checks[name]['pass_']:
            reasons.append(f'{name}: {checks[name]}')

    verdict['stage'] = 'frozen_engineering_pass'
    em = dict(original_layers=band, selected_layers=[L], gamma=r3['gamma'], permutations=r3['permutations'],
              row_split='latest0924_screen', receiver=key)
    dirs = {'original': D[[order[l] for l in band]], 'selected': dsel}
    torch.cuda.synchronize()
    t0 = time.time()
    try:
        rows = EC.scene_rows(receiver, scene, em, 'engineering', {'literal': None}, dirs, {'literal': list(LITERAL_ARMS)})
        err = None
    except AssertionError as e:
        rows, err = [], f'AssertionError: {e}'
    torch.cuda.synchronize()
    info['frozen_pass_seconds'] = round(time.time() - t0, 1)
    per_arm = {}
    for r in rows:
        c = r['engineering_checks']
        d = per_arm.setdefault(r['arm'], dict(rows=0, pass_=0, max_logit_difference=0., max_alias_gap_difference=0.,
                                              parsed_equal=True, answers=[], answered=0, correct=0, stops=Counter()))
        d['rows'] += 1
        d['pass_'] += bool(c['pass'])
        d['max_logit_difference'] = max(d['max_logit_difference'], c['max_logit_difference'])
        d['max_alias_gap_difference'] = max(d['max_alias_gap_difference'], c['alias_gap_difference'])
        d['parsed_equal'] &= bool(c['parsed_equal'])
        d['answers'].append(dict(world=r['world'], question=r['question'], gold=r['gold'], text64=r['text64'][:80],
                                 stop=r['stop'], n_tokens=len(r['tokens'])))
        d['answered'] += bool(r['score64']['answered'])
        d['correct'] += bool(r['score64']['correct'])
        d['stops'][r['stop']] += 1
    for d in per_arm.values():
        d['stops'] = dict(d['stops'])
    checks['frozen_engineering_pass'] = dict(error=err, rows=len(rows), expected_rows=4 * len(LITERAL_ARMS),
                                             per_arm=per_arm)
    ok = err is None and len(rows) == 4 * len(LITERAL_ARMS) and all(d['pass_'] == d['rows'] for d in per_arm.values())
    checks['frozen_engineering_pass']['pass_'] = ok
    if not ok:
        bad = {k: (d['pass_'], d['rows'], round(d['max_logit_difference'], 4), round(d['max_alias_gap_difference'], 4),
                   d['parsed_equal']) for k, d in per_arm.items() if d['pass_'] != d['rows']}
        reasons.append(f'frozen engineering pass: error={err} failing arms (pass, rows, max logit diff, max gap diff, '
                       f'parsed equal)={bad}')
    ft = per_arm.get('full_text', dict(answered=0, rows=0, correct=0))
    checks['answer_format'] = dict(full_text_answered=ft['answered'], full_text_rows=ft['rows'],
                                   full_text_correct=ft['correct'], pass_=ft['rows'] == 4 and ft['answered'] >= 3)
    if not checks['answer_format']['pass_']:
        reasons.append(f"answer format: only {ft['answered']}/{ft['rows']} full_text answers parse as an alias "
                       f"(chat rendering / thinking control?) e.g. {[x['text64'] for x in ft.get('answers', [])][:2]}")
    info['memory_after_frozen_pass'] = gpu_memory()

    verdict['stage'] = 'fit_prompts'
    checks['fit_prompts'] = fit_consistency(FL, receiver, key)
    fc = checks['fit_prompts']
    if fc.get('status') == 'FAIL':
        reasons.append(f"fit prompts: {fc.get('detail')}")
    elif fc.get('status') == 'WARN':
        verdict['warnings'].append(f"fit direction drift: {fc.get('detail')}")
    info['memory_peak'] = gpu_memory()
    verdict['stage'] = 'done'


def fit_consistency(FL, receiver, key):
    import torch
    path = FITS / key / 'w000.pt'
    if not path.exists():
        return dict(status='SKIPPED', detail=f'{path} absent')
    stored = torch.load(path, map_location='cpu', weights_only=False)
    same, first_diff = 0, None
    rebuilt = []
    for ctx in stored['contexts']:
        c = FL.construction(receiver, stored['word'], ctx['width'], ctx['object'], ctx['index'])
        rebuilt.append(c)
        eq = c['source'] == ctx['source'] and c['neutral'] == ctx['neutral'] and c['position'] == ctx['position']
        same += eq
        if not eq and first_diff is None:
            src_diff = next((i for i, (x, y) in enumerate(zip(c['source'], ctx['source'])) if x != y),
                            min(len(c['source']), len(ctx['source'])))
            first_diff = dict(width=ctx['width'], object=ctx['object'], index=ctx['index'],
                              now_len=len(c['source']), stored_len=len(ctx['source']), first_token_diff=src_diff)
    if same != len(stored['contexts']):
        return dict(status='FAIL', contexts=len(stored['contexts']), identical=same,
                    detail=f'{len(stored["contexts"]) - same} of {len(stored["contexts"])} fitting contexts are '
                           f'tokenized differently now than when {path} was fitted: {first_diff}')
    layers = stored['layers']
    with torch.inference_mode():
        contrasts = [FL.read(receiver.model, layers, c['source'], c['position']) -
                     FL.read(receiver.model, layers, c['neutral'], c['position']) for c in rebuilt]
    direction = torch.stack(contrasts).double().mean(dim=0).float()
    diff = (direction - stored['direction'].float()).abs()
    rel = float((diff.norm(dim=1) / stored['direction'].float().norm(dim=1)).max())
    status = 'OK' if rel <= 1e-3 else 'WARN'
    return dict(status=status, contexts=len(rebuilt), identical=same, layers=layers, word=stored['word'],
                max_abs_diff=float(diff.max()), max_relative_l2_diff=rel, bit_identical=bool(torch.equal(
                    direction, stored['direction'].float())),
                detail=f'word 0 direction recomputed now vs stored: max relative L2 difference {rel:.3g}')


# ======================================================================================================================
# CPU mode
# ======================================================================================================================
def meta_model(A, key):
    import torch
    import transformers
    spec = A.MODELS[key]
    cfg = transformers.AutoConfig.from_pretrained(resolve_model(resolve_model(spec['path'])))
    if spec['loader'] == 'text_only_key_mapping':
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
        tc = A._text_config(cfg)
        with torch.device('meta'):
            return MODEL_FOR_CAUSAL_LM_MAPPING[type(tc)]._from_config(tc, dtype=torch.float32)
    if spec['loader'] == 'gpt_oss_dequant' and getattr(cfg, 'quantization_config', None) is not None:
        del cfg.quantization_config   # a dequantized load has plain parameters of these shapes
    with torch.device('meta'):
        return transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)


def checkpoint_shapes(path):
    path = Path(path)
    idx = json.loads((path / 'model.safetensors.index.json').read_text())['weight_map']
    shapes = {}
    for shard in sorted(set(idx.values())):
        with open(path / shard, 'rb') as f:
            header = json.loads(f.read(int.from_bytes(f.read(8), 'little')))
        shapes.update({k: v['shape'] for k, v in header.items() if k != '__metadata__'})
    assert set(shapes) == set(idx), 'index and shard headers disagree'
    return shapes


def coverage(A, key, model):
    """Emulate from_pretrained's key renaming (transformers' own rules) on the checkpoint's safetensors headers."""
    import re
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key
    spec = A.MODELS[key]
    if spec['loader'] == 'gpt_oss_dequant':
        return dict(status='SKIPPED', detail='MXFP4 blocks/scales are converted by the quantizer; the GPU load is strict')
    shapes = checkpoint_shapes(resolve_model(spec['path']))
    conv = get_model_conversion_mapping(model, spec['key_mapping'], None)
    ren = [c for c in conv if isinstance(c, WeightRenaming)]
    cvt = [c for c in conv if isinstance(c, WeightConverter)]
    meta = model.state_dict()
    target = {}
    for k in shapes:
        t, pat = rename_source_key(k, ren, cvt, model.base_model_prefix, meta)
        target.setdefault(t, []).append((k, pat))
    ignore = [re.compile(p) for p in list(spec['ignorable_unexpected']) +
              list(getattr(model, '_keys_to_ignore_on_load_unexpected', None) or [])]
    tied = set()
    if getattr(A._text_config(model.config), 'tie_word_embeddings', False) or getattr(model.config, 'tie_word_embeddings', False):
        tw = getattr(model, '_tied_weights_keys', None)
        tied = set(tw) if tw else {'lm_head.weight'}
    missing = sorted(p for p in meta if p not in target and p not in tied)
    unexpected = sorted(t for t, srcs in target.items() if t not in meta and
                        not all(any(p.search(s) for p in ignore) for s, _ in srcs))
    mism = []
    for t, srcs in target.items():
        if t in meta and len(srcs) == 1 and srcs[0][1] is None and list(meta[t].shape) != shapes[srcs[0][0]]:
            mism.append((t, shapes[srcs[0][0]], list(meta[t].shape)))
    return dict(status='PASS' if not (missing or unexpected or mism) else 'FAIL', checkpoint_keys=len(shapes),
                model_params=len(meta), matched=sum(t in meta for t in target), missing=missing[:20],
                n_missing=len(missing), unexpected=unexpected[:20], n_unexpected=len(unexpected),
                shape_mismatch=mism[:20], n_converted=sum(1 for srcs in target.values() for _, p in srcs if p))


class RecordingTokenizer:
    """Delegates to the real tokenizer and records every (text, add_special_tokens) it is asked to encode."""

    def __init__(self, tok):
        self._tok, self.calls = tok, set()

    def __call__(self, text, *args, **kwargs):
        if isinstance(text, str):
            self.calls.add((text, bool(kwargs.get('add_special_tokens', True))))
        return self._tok(text, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tok, name)


def raw_tokenizer_equality(A, key, tok, calls):
    """The HF tokenizer vs the checkpoint's own tokenizer.json (tokenizers library) on every text the pipeline encoded."""
    path = Path(A.MODELS[key]['path']) / 'tokenizer.json'
    if not path.exists():
        return dict(status='SKIPPED', detail='no tokenizer.json (tekken: see tekken_equivalence)')
    import tokenizers
    raw = tokenizers.Tokenizer.from_file(str(path))
    bad = []
    for text, special in sorted(calls):
        a = tok(text, add_special_tokens=special)['input_ids']
        b = raw.encode(text, add_special_tokens=special).ids
        if a != b:
            i = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            bad.append(dict(text=text[max(0, i - 3):][:60], hf=tok.convert_ids_to_tokens(a[i:i + 4]),
                            raw=[raw.id_to_token(t) for t in b[i:i + 4]], hf_len=len(a), raw_len=len(b)))
    return dict(status='PASS' if not bad else 'FAIL', texts=len(calls), mismatches=len(bad), examples=bad[:5])


def prompt_pipeline(A, ED, EC, FL, key, tok, words):
    """Every R3 scene through the frozen prompt construction the evaluators and fitter use (tokenizer-only)."""
    bank = json.loads(BANK.read_text())
    rec_tok = RecordingTokenizer(tok)
    r = fake_receiver(ED, rec_tok, words, key)
    cand = json.loads(CANDIDATE_A.read_text())
    counts = Counter()
    failures = []
    lengths = []
    for scene in bank:
        try:
            ni = ED.prompt_inputs(r, scene, None)
            lengths.append(ni['n'] + max(len(s) for s in ni['suffixes']))
            counts['neutral'] += 1
            for wi, world in enumerate(scene['worlds']):
                for source in ('literal', 'candidateA'):
                    if source == 'literal':
                        packet = EC.literal_packet_generic(world, words)
                    else:
                        rec = cand['packets'][f"{scene['scene']}|{wi}"]
                        packet = ED.base.deserialize(bytes.fromhex(rec['hex']))
                    active, positions, mapping = ED.entry_positions(ED.packet_entries(packet, words), ni['slots'])
                    wbo = [[e['word'] for e in sorted(active, key=lambda x: x['index']) if e['object'] == o] for o in (0, 1)]
                    si = ED.prompt_inputs(r, scene, wbo)
                    assert (ni['prefix'][:ni['lo']] == si['prefix'][:si['lo']] and
                            ni['prefix'][ni['hi']:] == si['prefix'][si['hi']:]), 'source/target differ outside message'
                    for extra in (ED.render(scene['aliases'], wbo, width=ED.WIDTH)[0], ' '.join(world['evidence'])):
                        r.tokenized(scene, extra)
                    counts[f'source_{source}'] += 1
        except Exception as e:  # noqa: BLE001  record and continue
            failures.append(dict(scene=scene['scene'], error=f'{type(e).__name__}: {str(e)[:200]}'))
    fits_ok, fit_fail = 0, []
    for word in words:
        for width in FL.CONTEXT_WIDTHS:
            for obj in (0, 1):
                for index in range(width):
                    try:
                        FL.construction(r, word, width, obj, index)
                        fits_ok += 1
                    except Exception as e:  # noqa: BLE001
                        fit_fail.append(dict(word=word, width=width, object=obj, index=index,
                                             error=f'{type(e).__name__}: {str(e)[:160]}'))
    raw = raw_tokenizer_equality(A, key, tok, rec_tok.calls)
    return dict(scenes=len(bank), counts=dict(counts), failures=failures[:20], n_failures=len(failures),
                max_prompt_tokens=max(lengths) if lengths else None, fit_constructions_ok=fits_ok,
                fit_constructions_expected=len(words) * 12, fit_failures=fit_fail[:20], n_fit_failures=len(fit_fail),
                unknown_single_token=len(tok(' unknown', add_special_tokens=False)['input_ids']) == 1,
                raw_tokenizer_equality=raw,
                pass_=not failures and not fit_fail and raw['status'] in ('PASS', 'SKIPPED'))


def tekken_equivalence(A, key, tok, texts):
    """Our mistral_common-free conversion vs transformers' convert_tekken_tokenizer and mistral_common's chat encoding."""
    try:
        import mistral_common  # noqa: F401
    except ImportError:
        return dict(status='SKIPPED', detail='mistral_common not importable here')
    from mistral_common.protocol.instruct.messages import SystemMessage, UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from transformers.integrations.mistral import convert_tekken_tokenizer
    path = Path(A.MODELS[key]['path']) / 'tekken.json'
    ref = convert_tekken_tokenizer(str(path))
    mt = MistralTokenizer.from_file(str(path))
    vocab_equal = ref.get_vocab() == tok.get_vocab()
    enc_equal, chat_equal = 0, 0
    for t in texts:
        enc_equal += ref(t, add_special_tokens=False)['input_ids'] == tok(t, add_special_tokens=False)['input_ids']
        want = mt.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=t)])).tokens
        got = tok(A.render_chat(tok, key, [dict(role='user', content=t)]), add_special_tokens=False)['input_ids']
        chat_equal += want == got
    sys_want = mt.encode_chat_completion(ChatCompletionRequest(
        messages=[SystemMessage(content='You are terse.'), UserMessage(content=texts[0])])).tokens
    sys_got = tok(A.render_chat(tok, key, [dict(role='system', content='You are terse.'),
                                           dict(role='user', content=texts[0])]), add_special_tokens=False)['input_ids']
    ok = vocab_equal and enc_equal == len(texts) and chat_equal == len(texts) and sys_want == sys_got
    return dict(status='PASS' if ok else 'FAIL', vocab_equal=vocab_equal, texts=len(texts), plain_encoding_equal=enc_equal,
                chat_encoding_equal_mistral_common=chat_equal, system_chat_equal=sys_want == sys_got)


SHRINK = dict(hidden_size=64, intermediate_size=128, num_attention_heads=2, num_key_value_heads=1, vocab_size=512,
              linear_num_key_heads=2, linear_num_value_heads=2, num_experts=4, num_experts_per_tok=2,
              num_local_experts=4, experts_per_token=2, moe_intermediate_size=32, shared_expert_intermediate_size=32,
              top_k_experts=2, num_global_key_value_heads=1, vocab_size_per_layer_input=512,
              hidden_size_per_layer_input=0)
PATTERN = {'qwen3_5': ['linear_attention'] * 3 + ['full_attention'],
           'qwen3_5_moe': ['linear_attention'] * 3 + ['full_attention'],
           'gpt_oss': ['sliding_attention', 'full_attention'] * 2,
           'olmo3': ['sliding_attention'] * 3 + ['full_attention'],
           'gemma4': ['sliding_attention'] * 3 + ['full_attention']}


def tiny_harness(A, ED, key):
    """Random-weight model of the same architecture (few layers, small width, sliding window 8 so 24-token prefixes
    overflow it) through the frozen capture/write hooks and the adapter cache functions, on CPU in fp32."""
    import torch
    import transformers
    spec = A.MODELS[key]
    tc = copy.deepcopy(A._text_config(transformers.AutoConfig.from_pretrained(resolve_model(resolve_model(spec['path'])))))
    for k, v in SHRINK.items():
        if hasattr(tc, k):
            setattr(tc, k, v)
    pat = PATTERN.get(spec['family'])
    tc.num_hidden_layers = len(pat) if pat else 4
    if pat is not None and getattr(tc, 'layer_types', None) is not None:
        tc.layer_types = list(pat)
    if getattr(tc, 'sliding_window', None):
        tc.sliding_window = 8
    if getattr(tc, 'quantization_config', None) is not None:
        del tc.quantization_config
    for name in ('pad_token_id', 'bos_token_id', 'eos_token_id'):   # special ids must fit the shrunk vocabulary
        v = getattr(tc, name, None)
        if isinstance(v, int) and v >= tc.vocab_size:
            setattr(tc, name, 0)
    torch.manual_seed(0)
    extra = dict(experts_implementation=spec['experts_implementation']) if spec['experts_implementation'] else {}
    model = transformers.AutoModelForCausalLM.from_config(tc, dtype=torch.float32, attn_implementation='eager',
                                                          **extra).eval()
    frozen_fp = ED.prior._syncomm_frozen['cache_fingerprint']
    rec = dict(model_class=type(model).__name__, layer_types=getattr(tc, 'layer_types', None),
               sliding_window=getattr(tc, 'sliding_window', None),
               experts_implementation=getattr(model.config, '_experts_implementation', None))
    n, k = 24, 6
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(5, tc.vocab_size, (1, n + k), generator=g)
    prefix, suffix = ids[:, :n], ids[:, n:]
    L = A.tap_layer(model, .2)
    positions = [5, 9, 13]

    def fwd(x):
        return model(x, attention_mask=torch.ones_like(x), use_cache=True)
    with torch.inference_mode():
        store = {}
        with ED.capture_hooks(model, [L], store):
            neutral = fwd(prefix)
        fp0 = A.cache_fingerprint(neutral.past_key_values)
        rec['cache'] = A.cache_summary(neutral.past_key_values)
        try:
            rec['frozen_fingerprint_same'] = frozen_fp(neutral.past_key_values) == fp0
        except Exception as e:  # noqa: BLE001
            rec['frozen_fingerprint_same'] = f'frozen function raises {type(e).__name__}'
        rep = {}
        with ED.write_hooks(model, [L], {L: store[L]}, positions, rep):
            ident = fwd(prefix)
        rec['identity'] = dict(hook_fired=str(L) in rep, logits_bit_identical=bool(torch.equal(ident.logits, neutral.logits)),
                               fingerprint_equal=A.cache_fingerprint(ident.past_key_values) == fp0)
        d = torch.randn(store[L].shape[-1], generator=g)
        states = store[L].clone()
        for p in positions:
            states[0, p] += 3.0 * d / d.norm() * store[L][0, p].norm()
        with ED.write_hooks(model, [L], {L: states}, positions, {}):
            fitted = fwd(prefix)
        rec['fitted_max_logit_move'] = float((fitted.logits[0, -1] - neutral.logits[0, -1]).abs().max())
        for tag, outp, st in (('none', neutral, None), ('fitted', fitted, states)):
            cache = outp.past_key_values
            fp = A.cache_fingerprint(cache)
            hooks = (lambda: ED.write_hooks(model, [L], {L: st}, positions, {})) if st is not None else contextlib.nullcontext
            branch = A.clone_cache(cache)
            end = model(suffix, past_key_values=branch, use_cache=True, attention_mask=torch.ones(1, n + k, dtype=torch.long))
            with hooks():
                full = fwd(ids)
            nxt = torch.tensor([[int(end.logits[0, -1].argmax())]])
            step = model(nxt, past_key_values=end.past_key_values, use_cache=True)
            with hooks():
                full2 = fwd(torch.cat([ids, nxt], 1))
            end2 = model(suffix, past_key_values=A.clone_cache(cache), use_cache=True,
                         attention_mask=torch.ones(1, n + k, dtype=torch.long))
            rec[f'cached_vs_full_{tag}'] = dict(
                last=float((full.logits[0, -1] - end.logits[0, -1]).abs().max()),
                all_suffix=float((full.logits[0, n:] - end.logits[0]).abs().max()),
                next_token=float((full2.logits[0, -1] - step.logits[0, -1]).abs().max()),
                original_unchanged=A.cache_fingerprint(cache) == fp,
                second_clone_identical=bool(torch.equal(end.logits, end2.logits)))
    rec['pass_'] = bool(rec['identity']['hook_fired'] and rec['identity']['logits_bit_identical']
                        and rec['identity']['fingerprint_equal'] and rec['fitted_max_logit_move'] > 1e-3
                        and all(rec[f'cached_vs_full_{t}']['last'] < 1e-3 and rec[f'cached_vs_full_{t}']['next_token'] < 1e-3
                                and rec[f'cached_vs_full_{t}']['original_unchanged']
                                and rec[f'cached_vs_full_{t}']['second_clone_identical'] for t in ('none', 'fitted')))
    return rec


def synthetic_dense(A, ED):
    import torch
    from transformers import DynamicCache
    ffp, fcl = ED.prior._syncomm_frozen['cache_fingerprint'], ED.prior._syncomm_frozen['clone_cache']
    g = torch.Generator().manual_seed(1)
    res = {}
    for dtype in (torch.float32, torch.bfloat16):
        c = DynamicCache()
        for li in range(3):
            for step in (7, 1, 3):
                c.update(torch.randn(1, 2, step, 16, generator=g).to(dtype), torch.randn(1, 2, step, 16, generator=g).to(dtype), li)
        mine, frozen = A.cache_fingerprint(c), ffp(c)
        ca, cf = A.clone_cache(c), fcl(c)
        ca.update(torch.ones(1, 2, 1, 16, dtype=dtype), torch.ones(1, 2, 1, 16, dtype=dtype), 0)
        c2 = copy.deepcopy(c)
        c2.layers[1].values[0, 0, 0, 0] += 1
        res[str(dtype)] = dict(byte_identical=mine == frozen, clone_fingerprints_equal=A.cache_fingerprint(cf) == frozen,
                               original_unchanged_after_clone_update=A.cache_fingerprint(c) == mine,
                               one_element_changes_fingerprint=A.cache_fingerprint(c2) != mine)
    res['pass_'] = all(all(v.values()) for v in res.values() if isinstance(v, dict))
    return res


def cpu_screen(a, key, verdict):
    import torch
    verdict['info']['sklearn_stubbed'] = ensure_sklearn()
    verdict['stage'] = 'import'
    A, ED, EC, FL = frozen_modules()
    spec = A.MODELS[key]
    checks, reasons, info = verdict['checks'], verdict['reasons'], verdict['info']
    import transformers
    info['declared_env'] = dict(env=spec['env'], python=spec['python'], transformers_here=transformers.__version__,
                                matches=transformers.__version__.startswith(A.ENV_TRANSFORMERS[spec['env']]))
    words = json.loads(WORDS.read_text())
    verdict['stage'] = 'meta'
    model = meta_model(A, key)
    layers = A.decoder_layers(model)
    lk = lock_entry(key)
    L, band = A.tap_layer(model, .2), sorted(set(A.tap_layer(model, x) for x in (.4, .5, .6)))
    try:
        frozen_same = A._ORIGINALS['decoder_layers'](model) is layers
    except Exception as e:  # noqa: BLE001
        frozen_same = f'frozen decoder_layers raises {type(e).__name__}'
    checks['meta'] = dict(model_class=type(model).__name__, n_layers=len(layers), declared=spec['n_layers'],
                          layer_class=type(layers[0]).__name__, depth02_layer=L, band=band,
                          frozen_decoder_layers_same_object=frozen_same,
                          param_billions=round(sum(p.numel() for p in model.parameters()) / 1e9, 3),
                          lock=lk and dict(selected_layer=lk['selected_layer'], original_band=lk['original_band']))
    checks['meta']['pass_'] = (len(layers) == spec['n_layers'] and
                               (lk is None or (L == lk['selected_layer'] and band == lk['original_band'])))
    from syncomm.e015 import core   # the frozen rounding, compared on dummy stacks of every depth 2..128
    mismatches = []
    for n_ in range(2, 129):
        dummy = torch.nn.Module()
        dummy.model = torch.nn.Module()
        dummy.model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(n_)])
        dummy.config = type('Cfg', (), {'num_hidden_layers': n_})()
        mismatches += [(n_, f) for f in (.2, .4, .5, .6) if core.tap_layer(dummy, f) != A.tap_layer(dummy, f)]
    checks['tap_layer_rounding'] = dict(depths='2..128', fractions=[.2, .4, .5, .6], mismatches=mismatches[:20],
                                        pass_=not mismatches)
    verdict['stage'] = 'coverage'
    try:
        checks['checkpoint_coverage'] = coverage(A, key, model)
    except Exception as e:  # noqa: BLE001
        checks['checkpoint_coverage'] = dict(status='ERROR', detail=f'{type(e).__name__}: {e}',
                                             tb=traceback.format_exc()[-1500:])
    del model
    verdict['stage'] = 'tokenizer'
    tok = A.load_tokenizer(key)
    r = fake_receiver(ED, tok, words, key)
    scene = next(s for s in json.loads(BANK.read_text()) if s['scene'] == a.scene)
    ni = ED.prompt_inputs(r, scene, None)
    text0, _ = r.prompt(scene, 0, ni['message'])
    stops = dict(turn_end=spec['turn_end'], turn_end_id=tok.convert_tokens_to_ids(spec['turn_end']))
    if spec['eos_token_ids']:
        stops['declared_stop_tokens'] = tok.convert_ids_to_tokens(list(spec['eos_token_ids']))
    checks['chat'] = dict(render=spec['chat_render'], kwargs=A.chat_kwargs(key), rendered_head=text0[:200],
                          rendered_tail=text0[-260:], prefix_tokens=ni['n'], stops=stops,
                          tokenizer_class=type(tok).__name__)
    ok = isinstance(stops['turn_end_id'], int) and stops['turn_end_id'] != tok.unk_token_id
    if spec['eos_token_ids']:
        ok = ok and stops['declared_stop_tokens'][0] == spec['turn_end']
    if spec['chat_render'] == 'template':
        msgs = [dict(role='user', content='Q?')]
        frozen_call = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        checks['chat']['adapter_equals_frozen_call'] = A.render_chat(tok, key, msgs) == frozen_call
        ok = ok and checks['chat']['adapter_equals_frozen_call']
    else:   # the frozen Receiver.prompt call must be routed to the adapter rendering
        msgs = [dict(role='user', content='Q?')]
        routed = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        checks['chat']['frozen_call_routed_to_adapter'] = routed == A.render_chat(tok, key, msgs)
        ok = ok and checks['chat']['frozen_call_routed_to_adapter']
    checks['chat']['pass_'] = bool(ok)
    verdict['stage'] = 'prompt_pipeline'
    checks['prompt_pipeline'] = prompt_pipeline(A, ED, EC, FL, key, tok, words)
    if spec['tokenizer'] == 'tekken':
        verdict['stage'] = 'tekken'
        bank = json.loads(BANK.read_text())
        texts = [ED.parent.Scaffold.receiver_body(s, 0, ' '.join(s['worlds'][0]['evidence']))[0] for s in bank[:40]]
        checks['tekken_equivalence'] = tekken_equivalence(A, key, tok, texts)
    verdict['stage'] = 'tiny_model'
    try:
        checks['tiny_model'] = tiny_harness(A, ED, key)
    except Exception as e:  # noqa: BLE001
        checks['tiny_model'] = dict(pass_=False, error=f'{type(e).__name__}: {e}', tb=traceback.format_exc()[-2000:])
    verdict['stage'] = 'synthetic_dense_cache'
    checks['synthetic_dense_cache'] = synthetic_dense(A, ED)
    for name, c in checks.items():
        ok = c.get('pass_', c.get('status') in ('PASS', 'SKIPPED'))
        if not ok:
            reasons.append(f'{name}: {json.dumps({k: v for k, v in c.items() if k != "tb"}, default=str)[:600]}')
    verdict['stage'] = 'done'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--scene', default=DEFAULT_SCENE)
    ap.add_argument('--cpu', action='store_true', help='CPU-only checks (no GPU, no weights)')
    ap.add_argument('--retry-errors', action='store_true', help='rerun if the existing verdict is ERROR')
    a = ap.parse_args()
    from syncomm.latest import adapter as A
    key = A.resolve_key(a.model)
    if key is None:
        raise SystemExit(f'unknown model {a.model}; known: {sorted(A.MODELS)}')
    dest = a.out / (f'{key}.cpu.json' if a.cpu else f'{key}.json')
    if dest.exists():
        prev = json.loads(dest.read_text())
        if not (a.retry_errors and prev.get('status') == 'ERROR'):
            print(f'[screen] SKIP {key}: verdict exists ({prev.get("status")}) -> {dest}', flush=True)
            return 0
    verdict = dict(model=key, mode='cpu' if a.cpu else 'gpu', status=None, reasons=[], warnings=[], checks={}, info={},
                   spec={k: v for k, v in A.MODELS[key].items()}, started=utc(), stage='start')
    try:
        verdict['env'] = env_record()
        (cpu_screen if a.cpu else gpu_screen)(a, key, verdict)
        verdict['status'] = 'FAIL' if verdict['reasons'] else 'PASS'
    except Exception as e:  # noqa: BLE001  an exception is a verdict too (ERROR), with where it happened
        verdict['status'] = 'ERROR'
        verdict['reasons'].append(f"exception at stage {verdict['stage']}: {type(e).__name__}: {str(e)[:1500]}")
        verdict['traceback'] = traceback.format_exc()[-6000:]
    verdict['finished'] = utc()
    atomic_json(dest, verdict)
    print(f"[screen] {key} {verdict['mode']} {verdict['status']} -> {dest}", flush=True)
    for r in verdict['reasons']:
        print('  reason:', r[:800], flush=True)
    for w in verdict['warnings']:
        print('  warning:', w[:400], flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
