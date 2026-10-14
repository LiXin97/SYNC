"""Adapter that lets the FROZEN receiver-recovery evaluators run unchanged on 2025-2026 receiver checkpoints.

What it does. `src/syncomm/recovery/evaluate_depth.py`, `evaluate_confirm.py` and `fit_layers.py` reach model code only
through two frozen helper modules:

  base  = artifacts/counterfactual0917/v2/snapshot/counterfactual_run.py  load_model, decoder_layers, tap_layer,
                                                                           Receiver.prompt (chat rendering), packet_text
  prior = artifacts/mechanism0919/discovery_v3/snapshot/run.py            cache_fingerprint, clone_cache,
                                                                           Receiver.tokenized, Receiver.decode

`install()` swaps five of those names (base.load_model, base.decoder_layers, base.tap_layer, prior.cache_fingerprint,
prior.clone_cache) for the versions below. Two per-model properties are fixed at load time rather than by patching
frozen code: the tokenizer's chat rendering (only for the three models whose template cannot give a plain non-thinking
prompt through the frozen call) and the generation config's stop tokens (only where the checkpoint ships none). The
evaluators' own code, arms, hooks, checks and thresholds are untouched. A path that is not in MODELS is loaded by the
frozen loader verbatim; on a dense DynamicCache the cache functions produce byte-identical output to the frozen ones;
`tap_layer` is the frozen rounding; `decoder_layers` returns the same ModuleList wherever the frozen one succeeds.

Model-specific assumptions found in the frozen path, and what this module does about each:
  load_model     frozen: AutoModelForCausalLM (AutoModelForImageTextToText for model_type qwen3_5 only), one GPU
                 ('cuda:0'), eager attention, fp32. Here: per-model loader (text-only where the checkpoint is a
                 multimodal wrapper and the text-only load is clean), eager attention everywhere (every architecture
                 below implements it; gpt-oss's attention sinks live in its eager path), fp32 everywhere,
                 device_map='auto' over `gpus` cards with a per-card cap only for Qwen3.5-122B-A10B (~466 GiB in fp32),
                 gpt-oss MXFP4 experts dequantized (Mxfp4Config(dequantize=True): no MXFP4 kernels, no hub download),
                 Mistral-Small-3.2's tekken tokenizer converted without mistral_common. MoE experts pinned to
                 experts_implementation='eager' (the per-expert fp32 loop): transformers' default 'grouped_mm' casts
                 activations to the weight dtype and, on B200 with torch >= 2.10, would run an unverified grouped
                 kernel instead of the frozen fp32-eager regime. The load is refused if any
                 weight is missing or mismatched, or an unexpected checkpoint key is outside the declared ignorable
                 (vision tower / projector / multi-token-prediction) patterns. Floating tensors narrower than the
                 requested dtype are upcast after loading (only gpt-oss has any: its dequantized experts stay BF16).
  decoder_layers frozen: model.model.layers, else model.model.language_model.layers. Here: the same two paths first,
                 then other wrappers, and the ModuleList length must equal the text config's num_hidden_layers.
  cache          frozen fingerprint/clone read `.keys`/`.values` of every cache layer ("standard dense Qwen
                 DynamicCache only"). Qwen3.5/3.6 caches (transformers 5.12) hold LinearAttentionLayer entries with
                 `conv_states`/`recurrent_states` and no keys/values; gpt-oss, OLMo-3 and Gemma-4 have sliding-window
                 layers. Here: every state tensor of every layer is hashed (keys, values, conv_states,
                 recurrent_states), the same JSON record layout and seq length as the frozen function, and the clone is
                 asserted to have distinct storage for every state tensor. LinearAttentionLayer updates its states
                 IN PLACE (copy_), so a shared-storage clone would silently mutate the prefix state; the assertion
                 matters more here than for concatenating KV layers.
  chat template  frozen call: apply_chat_template([user], tokenize=False, add_generation_prompt=True,
                 enable_thinking=False). Correct as is for Qwen3.5/3.6 (appends '<think>\\n\\n</think>\\n\\n'),
                 Gemma-4 (appends the empty thought channel), OLMo-3 and Granite-4.1 (no thinking mode). Not correct for
                   gpt-oss   template writes today's date into the system message (strftime_now) and leaves the
                             model to open an analysis channel. Rendered with reasoning_effort='low', the date pinned to
                             FIXED_DATE, and the final channel pre-opened ('<|channel|>final<|message|>').
                   Seed-OSS  default thinking_budget=-1 lets the model think without limit. Rendered with the default
                             template (NO system turn, so the question-independent prefix is byte-identical to the
                             frozen call's, i.e. to the fitting contexts) followed by the template's own budget-0
                             assistant prefill, taken verbatim from a thinking_budget=0 rendering.
                   Mistral   the checkpoint has no HF tokenizer or template (tekken.json + mistral_common only).
                             Rendered as '<s>[INST]{user}[/INST]' (v11 instruct format, no default system prompt).
                 For these three the tokenizer instance's apply_chat_template is wrapped so the frozen
                 Receiver.prompt call returns the rendering above; every other call passes through.
  special tokens add_special_tokens=not chat (False) is right for every template here: each rendering carries its own
                 BOS where the model expects one (Gemma '<bos>', Mistral '<s>'); the others use none.
  decode / EOS   frozen decode stops on generation_config.eos_token_id (else tok.eos_token_id) or a newline.
                 Qwen3.5-2B/4B/9B ship no generation_config.json, so the default stop set would be the text config's
                 '<|endoftext|>' only; it is set to [<|im_end|>, <|endoftext|>] = [248046, 248044], exactly what the
                 27B/3.6 checkpoints ship. For every model the chat turn-end token is asserted to be a stop token.

Environments (node): 'system' = python3 with transformers 5.3.0; 'tflatest' = tf-latest venv, transformers 5.12.1 over
the same system packages (no flash-linear-attention, so Qwen3.5 uses the torch gated-delta fallback in both). Qwen3.5/3.6
MUST use 'tflatest': in transformers 5.3.0 the Gated-DeltaNet forward ignores the cached conv/recurrent state whenever
more than one token is fed on top of a cache (initial_state=None), so the evaluators' cached-prefix + multi-token
question continuation would silently differ from a full forward, and its cache class has no `.layers`. Granite-4.1
MUST use 'tflatest': 5.3.0 tokenizes it differently from its own tokenizer.json. Gemma-4 is not in 5.3.0. gpt-oss and
Mistral-Small-3.2 (no fits yet) use 'tflatest'; OLMo-3 and Seed-OSS stay on 'system', where their fits were computed and
whose tokenization equals tokenizer.json in both versions. The running transformers version is checked against the
declared env at load time.

Nothing here has run on a GPU when this file was written (2026-09-24); scripts/latest0924/screen.py is the GPU check.
"""
from __future__ import annotations
try:
    from syncomm.models import resolve_model
except ImportError:  # run outside the package: references must already be local paths
    def resolve_model(ref):
        return str(ref)

import base64
import copy
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
MODELS_DIR = Path('/path/to/models')
SYSTEM_PYTHON = os.environ.get('SYNC_PYTHON', sys.executable)                                           # node: transformers 5.3.0, torch 2.10.0+cu128
TFLATEST_PYTHON = os.environ.get('SYNC_PYTHON_TF512', os.environ.get('SYNC_PYTHON', sys.executable))  # node: transformers 5.12.1, same torch
ENV_TRANSFORMERS = {'system': '5.3.', 'tflatest': '5.12.'}
PYTHONS = {'system': SYSTEM_PYTHON, 'tflatest': TFLATEST_PYTHON}
FIXED_DATE = '2026-09-24'            # gpt-oss system message date (same pin as src/syncomm/latest/sender.py)
GPT_OSS_REASONING = 'low'            # lowest effort the harmony template offers; the final channel is pre-opened anyway
FROZEN_BASE = ROOT / 'artifacts/counterfactual0917/v2/snapshot'

QWEN35_IGNORABLE = (r'^model\.visual\.', r'^mtp\.')
QWEN_STOPS = [248046, 248044]        # <|im_end|>, <|endoftext|>: the stop set shipped by Qwen3.5-27B / Qwen3.6


def _spec(dirname, family, env, n_layers, loader, chat_render, turn_end, notes, gpus=1, roles=('receiver',),
          eos_token_ids=None, ignorable_unexpected=(), key_mapping=None, tokenizer='auto', max_memory_gib=None,
          dtype='float32'):
    moe = family in ('qwen3_5_moe', 'gpt_oss', 'gemma4')
    return dict(path=str(MODELS_DIR / dirname), family=family, chat=True, env=env, python=PYTHONS[env], dtype=dtype,
                gpus=gpus, n_layers=n_layers, loader=loader, chat_render=chat_render, turn_end=turn_end, roles=roles,
                eos_token_ids=eos_token_ids, ignorable_unexpected=tuple(ignorable_unexpected), key_mapping=key_mapping,
                tokenizer=tokenizer, max_memory_gib=max_memory_gib,
                experts_implementation='eager' if moe else None, notes=notes)


_Q35 = ('Text-only Qwen3_5{moe}ForCausalLM through AutoModelForCausalLM (transformers maps the composite config to the text '
        'config and drops model.visual.* / mtp.*). Hybrid: {lin} Gated-DeltaNet linear-attention + {full} full-attention '
        'layers; cache = DynamicCache of LinearAttentionLayer + DynamicLayer. tflatest required (5.3.0 cached-continuation '
        'bug).')

# Keys follow gates/latest0924/expA_protocol.lock.json. fp32 everywhere (nothing needs BF16 to fit on 8x183 GB).
MODELS = {
    'qwen3.5-2b': _spec('Qwen3.5-2B', 'qwen3_5', 'tflatest', 24, 'qwen3_5_text', 'template', '<|im_end|>',
                        _Q35.format(moe='', lin=18, full=6) + ' No generation_config.json: stops set to [248046, 248044].',
                        eos_token_ids=QWEN_STOPS, ignorable_unexpected=QWEN35_IGNORABLE),
    'qwen3.5-4b': _spec('Qwen3.5-4B', 'qwen3_5', 'tflatest', 32, 'qwen3_5_text', 'template', '<|im_end|>',
                        _Q35.format(moe='', lin=24, full=8) + ' No generation_config.json: stops set to [248046, 248044].',
                        eos_token_ids=QWEN_STOPS, ignorable_unexpected=QWEN35_IGNORABLE),
    'qwen3.5-9b': _spec('Qwen3.5-9B', 'qwen3_5', 'tflatest', 32, 'qwen3_5_text', 'template', '<|im_end|>',
                        _Q35.format(moe='', lin=24, full=8) + ' No generation_config.json: stops set to [248046, 248044].',
                        eos_token_ids=QWEN_STOPS, ignorable_unexpected=QWEN35_IGNORABLE),
    'qwen3.5-27b': _spec('Qwen3.5-27B', 'qwen3_5', 'tflatest', 64, 'qwen3_5_text', 'template', '<|im_end|>',
                         _Q35.format(moe='', lin=48, full=16) + ' ~104 GiB fp32 on one card.',
                         ignorable_unexpected=QWEN35_IGNORABLE),
    'qwen3.6-27b': _spec('Qwen3.6-27B', 'qwen3_5', 'tflatest', 64, 'qwen3_5_text', 'template', '<|im_end|>',
                         _Q35.format(moe='', lin=48, full=16) + ' ~104 GiB fp32 on one card. Also a sender.',
                         roles=('receiver', 'sender'), ignorable_unexpected=QWEN35_IGNORABLE),
    'qwen3.6-35b-a3b': _spec('Qwen3.6-35B-A3B', 'qwen3_5_moe', 'tflatest', 40, 'qwen3_5_text', 'template', '<|im_end|>',
                             _Q35.format(moe='Moe', lin=30, full=10) + ' 256 experts, fused expert tensors in the '
                             'checkpoint. ~134 GiB fp32 on one card (tight: ~40 GiB left).',
                             ignorable_unexpected=QWEN35_IGNORABLE),
    'qwen3.5-122b-a10b': _spec('Qwen3.5-122B-A10B', 'qwen3_5_moe', 'tflatest', 48, 'qwen3_5_text', 'template',
                               '<|im_end|>', _Q35.format(moe='Moe', lin=36, full=12) + ' 250 GB BF16 checkpoint = ~466 '
                               'GiB fp32: device_map auto over 4 cards capped at 150 GiB each (embeddings on cuda:0, '
                               'where the evaluators put their inputs).', gpus=4, max_memory_gib=150,
                               ignorable_unexpected=QWEN35_IGNORABLE),
    'gpt-oss-20b': _spec('gpt-oss-20b', 'gpt_oss', 'tflatest', 24, 'gpt_oss_dequant', 'gpt_oss_final', '<|return|>',
                         'GptOssForCausalLM, 32-expert MoE, alternating 128-token sliding-window / full attention '
                         '(12+12), attention sinks (eager path). MXFP4 experts dequantized (exact in bf16, then held in '
                         'fp32; ~80 GiB). Harmony prompt: reasoning low, date pinned, final channel pre-opened; R3 '
                         'prompts (~220 tokens) overflow the 128-token window, so sliding layers truncate. tflatest: no '
                         'fit exists yet and its sender runs on 5.12.1 (5.3.0 tokenizes it identically). Also a sender.',
                         roles=('receiver', 'sender')),
    'olmo-3-7b-instruct': _spec('OLMo-3-7B-Instruct', 'olmo3', 'system', 32, 'causal_lm', 'template', '<|im_end|>',
                                'Olmo3ForCausalLM, dense; 24 sliding-window (4096) + 8 full-attention layers; prompts are '
                                'far below 4096 tokens so no sliding layer ever truncates here.'),
    'seed-oss-36b-instruct': _spec('Seed-OSS-36B-Instruct', 'seed_oss', 'system', 64, 'causal_lm',
                                   'seed_budget0_prefill', '<seed:eos>',
                                   'SeedOssForCausalLM, dense, full attention; ~135 GiB fp32 on one card. Thinking '
                                   'suppressed by the template\'s own budget-0 assistant prefill; no system turn, so the '
                                   'prefix equals the frozen rendering (the one the running fits used).'),
    'mistral-small-3.2-24b': _spec('Mistral-Small-3.2-24B-Instruct-2506', 'mistral3', 'tflatest', 40,
                                   'text_only_key_mapping', 'mistral_v11', '</s>',
                                   'Text-only MistralForCausalLM from the Mistral3 conditional-generation (vision-language) checkpoint '
                                   '(language_model.model.* -> model.*, language_model.lm_head -> lm_head; vision tower '
                                   'and projector dropped). No HF tokenizer in the checkpoint: tekken.json v11 converted to '
                                   'a fast tokenizer here without mistral_common. ~90 GiB fp32. tflatest: no fit '
                                   'exists yet.',
                                   tokenizer='tekken',
                                   key_mapping={r'^language_model\.model\.': 'model.',
                                                r'^language_model\.lm_head\.': 'lm_head.'},
                                   ignorable_unexpected=(r'^vision_tower\.', r'^multi_modal_projector\.')),
    'granite-4.1-8b': _spec('granite-4.1-8b', 'granite', 'tflatest', 40, 'causal_lm', 'template', '<|end_of_text|>',
                            'GraniteForCausalLM, dense, full attention (embedding/residual/logit multipliers are inside '
                            'the modules, so layer-output hooks see the residual stream the model uses). tflatest '
                            'REQUIRED: transformers 5.3.0 rebuilds its GPT2Tokenizer with the GPT-2 byte-level regex '
                            'instead of the checkpoint\'s cl100k-style split, so it tokenizes differently from '
                            'tokenizer.json (e.g. ".\\n" -> two tokens); 5.12.1 matches tokenizer.json exactly.'),
    'gemma-4-26b-a4b-it': _spec('gemma-4-26B-A4B-it', 'gemma4', 'tflatest', 30, 'causal_lm', 'template', '<turn|>',
                                'Gemma4ForConditionalGeneration (AutoModelForCausalLM maps gemma4 to it; vision tower is '
                                'kept, text-only forwards). 128-expert MoE, 25 sliding-window (1024) + 5 full-attention '
                                'layers, no KV sharing. Optional receiver; sender. transformers 5.3.0 cannot load it.',
                                roles=('receiver', 'sender')),
}
_ORIGINALS = {}


# ----------------------------------------------------------------------------------------------------------------------
# lookup, environment
# ----------------------------------------------------------------------------------------------------------------------
def resolve_key(path_or_key):
    """MODELS key for a key or a checkpoint path (compared by realpath); None if the model is not an adapter model."""
    s = str(path_or_key)
    if s in MODELS:
        return s
    want = os.path.realpath(s)
    for key, spec in MODELS.items():
        if os.path.realpath(spec['path']) == want:
            return key
    return None


def check_env(key):
    """The running transformers must be the declared environment's (override: SYNCOMM_LATEST_ALLOW_ENV_MISMATCH=1)."""
    import transformers
    spec = MODELS[key]
    want = ENV_TRANSFORMERS[spec['env']]
    ok = transformers.__version__.startswith(want)
    if not ok and os.environ.get('SYNCOMM_LATEST_ALLOW_ENV_MISMATCH') != '1':
        raise RuntimeError(f"{key} is declared for env '{spec['env']}' (transformers {want}x, {spec['python']}); this is "
                           f"transformers {transformers.__version__} at {sys.executable}. Set "
                           f"SYNCOMM_LATEST_ALLOW_ENV_MISMATCH=1 only to screen it elsewhere on purpose.")
    return dict(declared=spec['env'], transformers=transformers.__version__, python=sys.executable, match=ok)


# ----------------------------------------------------------------------------------------------------------------------
# tokenizer and chat rendering
# ----------------------------------------------------------------------------------------------------------------------
def _fixed_strftime(fmt, _date=FIXED_DATE):
    return datetime.strptime(_date, '%Y-%m-%d').strftime(fmt)


def chat_kwargs(key):
    """The apply_chat_template kwargs this adapter's rendering uses (post-processing described in render_chat)."""
    mode = MODELS[key]['chat_render']
    if mode == 'template':
        return dict(enable_thinking=False)
    if mode == 'gpt_oss_final':
        return dict(reasoning_effort=GPT_OSS_REASONING)
    return {}


def _mistral_v11(messages):
    msgs = list(messages)
    text = '<s>'
    if msgs and msgs[0]['role'] == 'system':
        text += '[SYSTEM_PROMPT]' + msgs[0]['content'] + '[/SYSTEM_PROMPT]'
        msgs = msgs[1:]
    if [m['role'] for m in msgs] != ['user']:
        raise ValueError('mistral_v11 rendering supports [system,] user only')
    return text + '[INST]' + msgs[0]['content'] + '[/INST]'


def render_chat(tok, key, messages, _apply=None):
    """Plain non-thinking chat prompt (text, generation prompt included) for a MODELS chat model."""
    mode = MODELS[key]['chat_render']
    # never route through our own wrapper (it would call back into render_chat)
    apply = _apply or getattr(tok, '_syncomm_original_apply_chat_template', None) or tok.apply_chat_template
    messages = [dict(m) for m in messages]
    if mode == 'template':
        try:
            return apply(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:  # the project rules s6: a template/tokenizer that rejects the kwarg
            return apply(messages, tokenize=False, add_generation_prompt=True)
    if mode == 'gpt_oss_final':
        text = apply(messages, tokenize=False, add_generation_prompt=True, reasoning_effort=GPT_OSS_REASONING,
                     strftime_now=_fixed_strftime)
        assert text.count('Current date: ' + FIXED_DATE) == 1, 'gpt-oss date pin did not take effect'
        assert text.count('Reasoning: ' + GPT_OSS_REASONING) == 1
        assert text.endswith('<|start|>assistant'), text[-80:]
        return text + '<|channel|>final<|message|>'
    if mode == 'seed_budget0_prefill':
        head = '<seed:bos>assistant\n'
        text = apply(messages, tokenize=False, add_generation_prompt=True)
        ref = apply(messages, tokenize=False, add_generation_prompt=True, thinking_budget=0)
        assert text.endswith(head) and head in ref, (text[-80:], ref[-200:])
        prefill = ref[ref.rindex(head) + len(head):]
        assert prefill.startswith('<seed:think>') and prefill.endswith('</seed:think>'), prefill
        return text + prefill
    if mode == 'mistral_v11':
        return _mistral_v11(messages)
    raise ValueError(mode)


def _install_chat(tok, key):
    """Route the frozen Receiver.prompt call (tokenize=False, add_generation_prompt=True) through render_chat."""
    if MODELS[key]['chat_render'] == 'template':
        return  # the frozen call is already the right one; leave the tokenizer untouched
    original = tok.apply_chat_template

    def apply_chat_template(conversation, *args, **kwargs):
        if (not args and kwargs.get('tokenize') is False and kwargs.get('add_generation_prompt')
                and isinstance(conversation, (list, tuple)) and conversation
                and all(isinstance(m, dict) and 'role' in m for m in conversation)):
            return render_chat(tok, key, conversation, _apply=original)
        return original(conversation, *args, **kwargs)

    tok._syncomm_original_apply_chat_template = original
    tok.apply_chat_template = apply_chat_template
    tok._syncomm_chat_render = MODELS[key]['chat_render']


def _tekken_tokenizer(tekken_file):
    """transformers.integrations.mistral.convert_tekken_tokenizer without mistral_common (same vocab, merges, specials).

    Mirrors mistral_common's Tekkenizer.from_file: special tokens padded with <SPECIAL_i> up to
    default_num_special_tokens, the first default_vocab_size - num_special vocab entries as byte ranks, ids = rank +
    num_special. Equality with the mistral_common route is checked by `screen.py --cpu`.
    """
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.mistral import MistralConverter
    data = json.loads(Path(tekken_file).read_text())
    cfg = data['config']
    n_special, vocab_size = cfg['default_num_special_tokens'], cfg['default_vocab_size']
    specials = list(data['special_tokens'])
    specials += [dict(rank=i, token_str=f'<SPECIAL_{i}>', is_control=True) for i in range(len(specials), n_special)]
    assert len({t['token_str'] for t in specials}) == len(specials) == n_special
    ranks = {}
    for i, x in enumerate(data['vocab'][:vocab_size - n_special]):
        assert x['rank'] == i
        b = base64.b64decode(x['token_bytes'])
        assert i >= 256 or b == bytes([i])
        ranks[b] = i
    assert len(ranks) == vocab_size - n_special
    all_special = [t['token_str'] for t in sorted(specials, key=lambda t: t['rank'])]
    vocab = {s: i for i, s in enumerate(all_special)}
    vocab.update(ranks)
    tok = PreTrainedTokenizerFast(tokenizer_object=MistralConverter(
        vocab=vocab, additional_special_tokens=all_special, pattern=cfg['pattern']).converted())
    tok.add_special_tokens({'additional_special_tokens': all_special})
    for name, value in (('bos_token', '<s>'), ('eos_token', '</s>'), ('pad_token', '<pad>'), ('unk_token', '<unk>')):
        if value in all_special:
            tok.add_special_tokens({name: value})
    return tok


def load_tokenizer(key):
    spec = MODELS[key]
    if spec['tokenizer'] == 'tekken':
        tok = _tekken_tokenizer(Path(spec['path']) / 'tekken.json')
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(resolve_model(spec['path']))
    _install_chat(tok, key)
    tok._syncomm_key = key
    return tok


# ----------------------------------------------------------------------------------------------------------------------
# model loading
# ----------------------------------------------------------------------------------------------------------------------
def _text_config(config):
    get = getattr(config, 'get_text_config', None)
    if get is not None:
        try:
            return get(decoder=True)
        except TypeError:
            return get()
    return getattr(config, 'text_config', None) or config


def _placement(spec):
    if spec['gpus'] == 1:
        return dict(device_map='cuda:0')
    have = torch.cuda.device_count()
    assert have >= spec['gpus'], f"needs {spec['gpus']} visible GPUs, sees {have}"
    return dict(device_map='auto', max_memory={i: f"{spec['max_memory_gib']}GiB" for i in range(spec['gpus'])})


def _dtype(spec, requested):
    if spec['dtype'] == 'bfloat16' and requested == torch.float32:
        return torch.bfloat16   # declared per model only where fp32 cannot fit (none at present)
    return requested


def _check_loading(key, info):
    spec = MODELS[key]
    missing = sorted(info.get('missing_keys') or [])
    unexpected = sorted(info.get('unexpected_keys') or [])
    mismatched = [str(x) for x in (info.get('mismatched_keys') or [])]
    conversion = {str(k): str(v) for k, v in (info.get('conversion_errors') or {}).items()}
    errors = [str(e) for e in (info.get('error_msgs') or [])]
    patterns = [re.compile(p) for p in spec['ignorable_unexpected']]
    stray = [k for k in unexpected if not any(p.search(k) for p in patterns)]
    report = dict(n_missing=len(missing), missing=missing[:20], n_unexpected=len(unexpected),
                  unexpected_sample=unexpected[:10], stray_unexpected=stray[:20], n_mismatched=len(mismatched),
                  mismatched=mismatched[:20], conversion_errors=dict(list(conversion.items())[:20]), error_msgs=errors[:5])
    bad = missing or stray or mismatched or conversion or errors
    if bad and os.environ.get('SYNCOMM_LATEST_STRICT_LOAD', '1') != '0':
        raise RuntimeError(f'{key}: weights did not load cleanly: {json.dumps(report)[:2000]}')
    report['clean'] = not bad
    return report


def _normalize_stops(model, tok, key):
    spec = MODELS[key]
    gc = model.generation_config
    before = gc.eos_token_id
    if spec['eos_token_ids'] is not None:
        names = tok.convert_ids_to_tokens(list(spec['eos_token_ids']))
        assert names[0] == spec['turn_end'], (names, spec['turn_end'])
        gc.eos_token_id = list(spec['eos_token_ids'])
    eos = gc.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [tok.eos_token_id])   # the frozen decode's own rule
    turn_end = tok.convert_tokens_to_ids(spec['turn_end'])
    assert isinstance(turn_end, int) and turn_end in eos, f'{key}: turn end {spec["turn_end"]}={turn_end} not in stops {eos}'
    return dict(before=before, after=gc.eos_token_id, stop_ids=sorted(eos), turn_end_id=turn_end)


def _upcast_floats(model, dtype):
    """Bring every floating parameter/buffer to `dtype` when that is an UPCAST (never downcasts). Needed for gpt-oss:
    transformers loads the renamed MXFP4 keys with _dtype=None and Mxfp4Dequantize returns BF16, so the expert matrices
    stay BF16 in an otherwise fp32 model and GptOssExperts computes fp32 @ bf16 (a dtype error). BF16 -> fp32 is exact."""
    width = torch.finfo(dtype).bits if dtype.is_floating_point else None
    moved = Counter()
    with torch.no_grad():
        for tensor in list(model.parameters()) + list(model.buffers()):
            if tensor.is_floating_point() and tensor.dtype != dtype and torch.finfo(tensor.dtype).bits < width:
                moved[str(tensor.dtype)] += tensor.numel()
                tensor.data = tensor.data.to(dtype)
    return dict(moved)


def load_model(path_or_key, dtype=torch.float32):
    """(model, tok) with base.load_model's conventions: eval, no grad, pad token, left padding, eager attention.

    Paths not in MODELS go to the frozen loader unchanged (after install(); before it, KeyError).
    """
    key = resolve_key(path_or_key)
    if key is None:
        if 'load_model' in _ORIGINALS:
            return _ORIGINALS['load_model'](path_or_key, dtype)
        raise KeyError(f'{path_or_key} is not an adapter model: {sorted(MODELS)}')
    import transformers
    spec = MODELS[key]
    env = check_env(key)
    tok = load_tokenizer(key)
    dtype = _dtype(spec, dtype)
    kwargs = dict(dtype=dtype, attn_implementation='eager', output_loading_info=True, **_placement(spec))
    if spec['experts_implementation']:
        kwargs['experts_implementation'] = spec['experts_implementation']
    loader, cls = spec['loader'], transformers.AutoModelForCausalLM
    if loader == 'gpt_oss_dequant':
        kwargs['quantization_config'] = transformers.Mxfp4Config(dequantize=True)
    elif loader == 'text_only_key_mapping':
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
        config = _text_config(transformers.AutoConfig.from_pretrained(resolve_model(spec['path'])))
        cls = MODEL_FOR_CAUSAL_LM_MAPPING[type(config)]
        kwargs.update(config=config, key_mapping=dict(spec['key_mapping']))
    else:
        assert loader in ('causal_lm', 'qwen3_5_text'), loader
    started = time.time()
    model, info = cls.from_pretrained(resolve_model(spec['path']), **kwargs)
    load_seconds = time.time() - started
    if loader in ('qwen3_5_text', 'text_only_key_mapping'):
        assert 'ConditionalGeneration' not in type(model).__name__, type(model).__name__
    loading = _check_loading(key, info)
    upcast = _upcast_floats(model, dtype)
    model.eval()
    model.requires_grad_(False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    stops = _normalize_stops(model, tok, key)
    n = len(decoder_layers(model))
    assert n == spec['n_layers'], (key, n, spec['n_layers'])
    model._syncomm_latest = dict(key=key, loader=loader, model_class=type(model).__name__, dtype=str(dtype),
                               placement={k: (str(v) if k != 'max_memory' else v) for k, v in _placement(spec).items()},
                               attn_implementation=getattr(model.config, '_attn_implementation', None),
                               experts_implementation=getattr(_text_config(model.config), '_experts_implementation', None),
                               load_seconds=round(load_seconds, 1), loading=loading, stops=stops, env=env,
                               chat_render=spec['chat_render'], n_layers=n, upcast_numel_by_dtype=upcast)
    return model, tok


# ----------------------------------------------------------------------------------------------------------------------
# decoder layers and depth rounding
# ----------------------------------------------------------------------------------------------------------------------
_LAYER_PATHS = ('model.layers', 'model.language_model.layers', 'language_model.model.layers', 'model.model.layers',
                'language_model.layers', 'model.decoder.layers', 'transformer.h', 'layers')


def _n_text_layers(config):
    n = getattr(_text_config(config), 'num_hidden_layers', None)
    return n if isinstance(n, int) else None


def decoder_layers(model):
    """The ModuleList of text decoder blocks (frozen search order first; length must equal num_hidden_layers)."""
    want = _n_text_layers(model.config)
    for path in _LAYER_PATHS:
        obj = model
        for part in path.split('.'):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, torch.nn.ModuleList) and len(obj) > 0 and (want is None or len(obj) == want):
            return obj
    raise AttributeError(f'no text decoder ModuleList of length {want} on {type(model).__name__}; tried {_LAYER_PATHS}')


def tap_layer(model, frac):
    """Exactly syncomm.e015.core.tap_layer: int(round(frac * n)) clipped to [1, n - 1] (Python round: half to even)."""
    n = len(decoder_layers(model))
    li = int(round(frac * n))
    return max(1, min(n - 1, li))


# ----------------------------------------------------------------------------------------------------------------------
# cache fingerprint / clone
# ----------------------------------------------------------------------------------------------------------------------
_KV = ('keys', 'values')
_LINEAR = ('conv_states', 'recurrent_states')


def _digest(tensor):
    """Byte-identical to prior.digest_tensor."""
    tensor = tensor.detach().contiguous().cpu()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _nonempty(t):
    return isinstance(t, torch.Tensor) and t.numel() > 0


def layer_states(i, layer):
    """[(name, tensor)] for one cache layer: keys/values (frozen order) then conv/recurrent states; all or nothing."""
    out = []
    for group in (_KV, _LINEAR):
        tensors = [getattr(layer, name, None) for name in group]
        if any(_nonempty(t) for t in tensors):
            assert all(_nonempty(t) for t in tensors), f'cache layer {i} ({type(layer).__name__}): partial {group}'
            out.extend(zip(group, tensors))
    assert out, f'cache layer {i} ({type(layer).__name__}) holds no state tensor'
    return out


def _layers(cache):
    layers = getattr(cache, 'layers', None)
    if layers is None:
        raise TypeError(f'{type(cache).__name__} has no .layers (e.g. transformers 5.3 Qwen3_5DynamicCache); use the '
                        f'tf-latest environment for this model')
    return layers


def cache_fingerprint(cache):
    """sha256 over every state tensor of every layer + seq length; equals the frozen function on a dense cache."""
    records = []
    for i, layer in enumerate(_layers(cache)):
        for name, t in layer_states(i, layer):
            records.append((i, name, list(t.shape), str(t.dtype), _digest(t)))
    assert records
    raw = json.dumps([int(cache.get_seq_length()), records], sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _same(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    return a == b


def clone_cache(cache):
    """Deep copy with distinct storage for every state tensor (asserted), same seq length and per-layer flags."""
    cloned = copy.deepcopy(cache)
    assert type(cloned) is type(cache)
    assert cloned.get_seq_length() == cache.get_seq_length()
    left_layers, right_layers = _layers(cache), _layers(cloned)
    assert len(left_layers) == len(right_layers)
    for i, (left, right) in enumerate(zip(left_layers, right_layers)):
        assert type(left) is type(right)
        mine, theirs = dict(layer_states(i, left)), dict(layer_states(i, right))
        assert mine.keys() == theirs.keys()
        for name in mine:
            assert mine[name].data_ptr() != theirs[name].data_ptr(), f'layer {i} {name} shares storage'
        for attr in ('has_previous_state', 'cumulative_length', 'sliding_window'):
            if hasattr(left, attr):
                assert _same(getattr(left, attr), getattr(right, attr)), (i, attr)
    return cloned


def cache_summary(cache):
    """Cache class, per-layer class counts and state shapes (for the screen's record)."""
    from collections import Counter
    layers = _layers(cache)
    shapes = {}
    for i, layer in enumerate(layers):
        sig = type(layer).__name__ + ':' + ','.join(f'{n}{list(t.shape)}' for n, t in layer_states(i, layer))
        shapes.setdefault(sig, []).append(i)
    return dict(cache_class=type(cache).__name__, seq_length=int(cache.get_seq_length()),
                layer_classes=dict(Counter(type(l).__name__ for l in layers)),
                state_signatures={k: dict(n=len(v), layers=v[:8]) for k, v in shapes.items()})


# ----------------------------------------------------------------------------------------------------------------------
# installation into the frozen modules
# ----------------------------------------------------------------------------------------------------------------------
def frozen_base():
    """The frozen `base` module object (sys.modules['counterfactual_run']), the one the evaluators import."""
    if str(FROZEN_BASE) not in sys.path:
        sys.path.insert(0, str(FROZEN_BASE))
    import counterfactual_run
    return counterfactual_run


def _find_priors(modules):
    names = ('syncomm.recovery.evaluate_depth', 'syncomm.recovery.evaluate_confirm', 'syncomm.recovery.fit_layers')
    found = []
    for mod in list(modules) + [sys.modules.get(n) for n in names]:
        if mod is None:
            continue
        for cand in (getattr(mod, 'prior', None), getattr(getattr(mod, 'parent', None), 'prior', None)):
            if cand is not None and hasattr(cand, 'clone_cache') and all(cand is not f for f in found):
                found.append(cand)
    return found


def install(*modules):
    """Patch base (always) and every frozen `prior` reachable from `modules` (or the imported recovery modules).

    Call AFTER importing the evaluator module: `prior` is created by exec_module inside framing_reference and is not in
    sys.modules, so it can only be reached through the evaluator's attributes.
    """
    base = frozen_base()
    if 'load_model' not in _ORIGINALS:
        _ORIGINALS.update(load_model=base.load_model, decoder_layers=base.decoder_layers, tap_layer=base.tap_layer)
    base.load_model, base.decoder_layers, base.tap_layer = load_model, decoder_layers, tap_layer
    priors = _find_priors(modules)
    for p in priors:
        assert p.base is base, 'frozen prior is bound to a different counterfactual_run module'
        if not hasattr(p, '_syncomm_frozen'):
            p._syncomm_frozen = dict(cache_fingerprint=p.cache_fingerprint, clone_cache=p.clone_cache)
        p.cache_fingerprint, p.clone_cache = cache_fingerprint, clone_cache
    writer = sys.modules.get('syncomm.writer')   # SlotWrite (not used by the three evaluators) resolves layers here
    if writer is not None and hasattr(writer, 'decoder_layers'):
        writer.decoder_layers = decoder_layers
    if not priors:
        print('[adapter] WARNING: no frozen prior module found; import the evaluator before install()', flush=True)
    return dict(base=base, priors=priors)
