"""Latest-generation automatic senders (2025-2026 checkpoints) run with candidate A's FROZEN prompt, parser and packets.

What this does. For every (scene, world) of a bank, the sender model reads that world's object descriptions and the
public 221-word inventory, and returns JSON choosing up to four inventory words per object. The prompt text, the parse
rule and the packet construction are imported unchanged from the frozen candidate A
(src/syncomm/routeb/sender_candidate_a.py, a dated decision record; the four-object template from
src/syncomm/recovery/sender_candidate_a4.py when a bank has four aliases). Nothing about the prompt is tuned per sender.
What differs from candidate A is only (a) the checkpoint, (b) that checkpoint's own chat template, and (c) for a model
that emits a reasoning block before its answer, which part of the output is handed to the frozen parser (see
ANSWER RULES below). The packets file has exactly candidate A's structure (`packets[f'{scene}|{world}']` with
hex/sha256/bytes/empty_lanes/evidence_sha256, the same header keys, `log`, `unique_evidence_inputs`, `summary`), so
src/syncomm/recovery/evaluate_confirm.py and the quality audits consume it unchanged; the log carries extra fields.

Senders (all BF16 weights, greedy, max_new_tokens = candidate A's 256, batch size 1, one GPU each):
  gemma-4-26b-a4b-it  Gemma4ForConditionalGeneration (the checkpoint's declared class; AutoModelForCausalLM maps to it).
                      Chat template defaults: thinking OFF (no <|think|> in the system turn; the generation prompt
                      already ends with the empty thought block '<|channel>thought\\n<channel|>'). Native system role.
  qwen3.6-27b         Qwen3_5ForConditionalGeneration on the direct route (declared class; the adapter route loads the
                      text-only Qwen3_5ForCausalLM). Hybrid 48 linear + 16 full attention layers; tf-latest has no
                      flash-linear-attention, so the torch gated-delta fallback runs (slower). enable_thinking=False,
                      which makes the template append '<think>\\n\\n</think>\\n\\n' to the generation prompt.
  gpt-oss-20b         GptOssForCausalLM. The checkpoint is MXFP4; it is loaded with Mxfp4Config(dequantize=True), i.e.
                      BF16 weights (also avoids a hub download of MXFP4 kernels on an offline node). Harmony format
                      from the tokenizer's template with reasoning_effort='low' (reasoning cannot be switched off for
                      this model). The template's 'Current date:' line comes from strftime_now(); it is pinned to
                      GPTOSS_FIXED_DATE so the rendered prompt is identical on every day, shard and resume.
Environment: transformers >= 5.5 is needed for gemma4 (the checkpoint config says 5.5.0.dev0), so the node's system
python3 (transformers 5.3.0) cannot run Gemma-4; the adapter also declares Qwen3.5/3.6 for tf-latest (5.3.0 Gated-DeltaNet
cache bug). Recommended for all three senders: python (5.12.1 over
the node's system torch 2.10). The CPU prompt renders used 5.12.1 from the a separate conda env (same jinja template
path). Loading (--loader auto): `syncomm.latest.adapter.load_model(key, dtype=bfloat16)` when the adapter knows the
checkpoint, else transformers directly with the class named above. The adapter is receiver-oriented: eager attention,
text-only Qwen3_5ForCausalLM, and it pins gpt-oss to transformers 5.3.0 (under tf-latest its env check raises and the
sender falls back to the direct loader, recorded as `adapter_error`). Only the adapter's MODEL is used; its tokenizer
(which rewrites gpt-oss prompts) is not. The frozen chat kwargs below stay authoritative; any adapter disagreement is
recorded, never adopted. Use --loader direct to load all three senders the same way.

ANSWER RULES (what the frozen parse_selection sees; decided before any output of these senders):
  gemma4   tokens after the LAST '<channel|>' in the generated ids (the whole output when absent), special tokens skipped.
  qwen3_5  tokens after the LAST '</think>' in the generated ids (the whole output when absent), special tokens skipped.
  gpt_oss  the FIRST harmony message on the 'final' channel: tokens after its '<|message|>' up to '<|return|>',
           '<|end|>' or '<|call|>' (or the end of generation), special tokens skipped. No final channel (e.g. the
           analysis used up the 256-token budget) -> empty answer text -> parse status 'malformed' -> empty lanes, the
           world stays in the denominator. JSON inside the analysis channel is never used.
  gpt_oss_final_prefill  (only with --gptoss-mode final_prefill, a pre-declared alternative, NOT the default): the
           prompt ends with '<|channel|>final<|message|>' (no analysis, as the adapter renders gpt-oss receivers); the
           answer is the generated tokens up to the first '<|return|>'/'<|end|>'/'<|call|>'.
  Candidate A's rule is the gemma/qwen rule with no marker: the whole greedy output. `raw_output` below is always the
  full generation decoded with special tokens skipped (candidate A's field); `answer_text` is what was parsed.

Usage (GPU node; absolute paths, explicit PYTHONPATH; R=.,
PY=python):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/src $PY -m syncomm.latest.sender --sender qwen3.6-27b \\
      --bank $R/artifacts/mechanism0919/discovery_v1/exposed_development.json --split dev96 \\
      --out $R/artifacts/latest0924/sender_packets/qwen3.6-27b_dev96.json [--shard i --shards n] [--limit N]
  # after all shards of one run finished:
  $PY -m syncomm.latest.sender --merge --shards n --out <same --out> [--sender KEY --bank ... --split ...]
  # CPU only, no weights: render the exact prompt of one world (+ token counts over banks):
  $PY -m syncomm.latest.sender --render-only $R/artifacts/latest0924/sender_prompts --sender KEY --bank ... --split ...
Resumable: finished (scene|world) keys in --out (or its shard file) are skipped; the header must match exactly
(prompt hash, inventory, bank hash, sender, chat kwargs, rendered scaffold), otherwise the run refuses to start.
Writes are atomic (tmp + os.replace) every --save-every worlds and at the end.
"""
try:
    from syncomm.models import resolve_model
except ImportError:  # run outside the package: references must already be local paths
    def resolve_model(ref):
        return str(ref)
import argparse
import hashlib
import json
import os
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from syncomm.routeb.sender_candidate_a import (  # noqa: E402  frozen, imported not copied
    SYSTEM, USER_TEMPLATE, MAX_NEW_TOKENS, parse_selection, to_packet)
from syncomm.recovery.sender_candidate_a4 import SYSTEM4, USER_TEMPLATE4  # noqa: E402  frozen four-object template

MODELS_DIR = Path('/path/to/models')
# sha256(SYSTEM + '\n' + USER_TEMPLATE) as written into every candidate A / A4 packets file; guards the frozen text.
FROZEN_PROMPT_SHA256 = {2: '328c8b5df9d196724cdabd6cc7a291c0cfc7ca13ab524bd3cb70239de7183e0b',
                        4: 'bc796522ac1d8739be671c27f7f847810dc5163855999553fa380dd583fe81bf'}
DEFAULT_WORDS_FROM = ROOT / 'artifacts/receiver_recovery/r3_packets/candidateA_r3.json'
GPTOSS_FIXED_DATE = '2026-09-24'

# The frozen per-sender settings of this study. `chat_kwargs` go to tok.apply_chat_template; `answer` names the rule above.
SENDERS = {
    'gemma-4-26b-a4b-it': dict(path=str(MODELS_DIR / 'gemma-4-26B-A4B-it'), family='gemma4',
                               model_class='Gemma4ForConditionalGeneration', chat_kwargs={}, answer='gemma4',
                               dequantize_mxfp4=False, gpus=1),
    'qwen3.6-27b': dict(path=str(MODELS_DIR / 'Qwen3.6-27B'), family='qwen3_5',
                        model_class='Qwen3_5ForConditionalGeneration', chat_kwargs=dict(enable_thinking=False),
                        answer='qwen3_5', dequantize_mxfp4=False, gpus=1),
    'gpt-oss-20b': dict(path=str(MODELS_DIR / 'gpt-oss-20b'), family='gpt_oss', model_class='GptOssForCausalLM',
                        chat_kwargs=dict(reasoning_effort='low'), answer='gpt_oss', dequantize_mxfp4=True, gpus=1,
                        fixed_date=GPTOSS_FIXED_DATE),
}
ALIASES = {'gemma-4-26b-a4b-it': 'gemma-4-26b-a4b-it', 'gemma-4-26b-a4b': 'gemma-4-26b-a4b-it', 'gemma4-26b': 'gemma-4-26b-a4b-it',
           'qwen3.6-27b': 'qwen3.6-27b', 'qwen3_6-27b': 'qwen3.6-27b', 'qwen36-27b': 'qwen3.6-27b',
           'gpt-oss-20b': 'gpt-oss-20b', 'gptoss-20b': 'gpt-oss-20b', 'gpt_oss_20b': 'gpt-oss-20b'}


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_text(s):
    return sha_bytes(s.encode())


def resolve_sender(key):
    k = ALIASES.get(key.strip().lower())
    if k is None:
        raise SystemExit(f'unknown sender {key!r}; known: {sorted(SENDERS)}')
    return k


def load_adapter():
    """The concurrently written adapter, if importable; None otherwise (fall back to transformers directly)."""
    try:
        from syncomm.latest import adapter  # noqa: F401
        return adapter
    except Exception as e:  # absent, or broken mid-edit: never block on it
        print(f'[sender] adapter unavailable ({type(e).__name__}: {e}); using the direct loader', flush=True)
        return None


def adapter_key(adapter, spec):
    """Map this study's sender to the adapter's MODELS key by checkpoint path (keys may be spelled differently)."""
    models = getattr(adapter, 'MODELS', None) or {}
    want = os.path.realpath(spec['path'])
    for k, v in models.items():
        p = v.get('path') if isinstance(v, dict) else getattr(v, 'path', None)
        if p and os.path.realpath(str(p)) == want:
            return k, v
    return None, None


def frozen_template(n_aliases):
    if n_aliases == 2:
        system, user = SYSTEM, USER_TEMPLATE
    elif n_aliases == 4:
        system, user = SYSTEM4, USER_TEMPLATE4
    else:
        raise SystemExit(f'no frozen template for {n_aliases} objects')
    sha = sha_text(system + '\n' + user)
    assert sha == FROZEN_PROMPT_SHA256[n_aliases], f'frozen {n_aliases}-object prompt text changed: {sha}'
    return system, user, sha


def user_message(user_template, words, aliases, description):
    kw = {f'alias{i}': a for i, a in enumerate(aliases)}
    return user_template.format(n=len(words), vocab=', '.join(words), description=description, **kw)


def chat_kwargs_for(spec):
    kw = dict(spec['chat_kwargs'])
    if spec.get('fixed_date'):
        date = spec['fixed_date']
        kw['strftime_now'] = lambda fmt, _d=date: datetime.strptime(_d, '%Y-%m-%d').strftime(fmt)
    return kw


def render(tok, spec, system, user):
    """Exact prompt text through the model's own chat template, and its decomposition head + user + tail."""
    msgs = [dict(role='system', content=system), dict(role='user', content=user)]
    # transformers 5.x returns a dict when tokenizing, so render text first (the project rules trap), as candidate A does.
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **chat_kwargs_for(spec))
    if spec.get('fixed_date'):
        assert text.count(f"Current date: {spec['fixed_date']}\n") == 1, 'date pin did not take effect'
    text += spec.get('prefill', '')
    assert text.count(user) == 1, 'user message not verbatim (or not unique) in the rendered prompt'
    i = text.index(user)
    head, tail = text[:i], text[i + len(user):]
    assert head.count(system) == 1, 'system message not verbatim in the rendered prompt'
    return text, head, tail


def tokenize_prompt(tok, text, spec):
    ids = tok(text, add_special_tokens=False)['input_ids']  # the template writes its own BOS where it wants one
    if tok.bos_token_id is not None and len(ids) > 1:
        assert not (ids[0] == tok.bos_token_id and ids[1] == tok.bos_token_id), 'double BOS'
    return ids


def special_id(tok, token):
    i = tok.convert_tokens_to_ids(token)
    assert isinstance(i, int) and i != tok.unk_token_id and tok.convert_ids_to_tokens(i) == token, f'{token!r} is not one token'
    return i


def extract_answer(tok, rule, gen_ids):
    """The ANSWER RULES of the docstring. Returns (answer_text, info)."""
    ids = list(gen_ids)
    info = {}
    if rule in ('gemma4', 'qwen3_5'):
        marker = special_id(tok, '<channel|>' if rule == 'gemma4' else '</think>')
        pos = [i for i, t in enumerate(ids) if t == marker]
        info['reasoning_marker_in_output'] = bool(pos)
        start = pos[-1] + 1 if pos else 0
        info['answer_source'] = 'after_last_marker' if pos else 'full_output'
        info['reasoning_tokens'] = start
        return tok.decode(ids[start:], skip_special_tokens=True), info
    if rule == 'gpt_oss':
        ch, msg = special_id(tok, '<|channel|>'), special_id(tok, '<|message|>')
        stops = {special_id(tok, t) for t in ('<|return|>', '<|end|>', '<|call|>')}
        channels = []
        i = 0
        while i < len(ids):
            if ids[i] == ch:
                j = i + 1
                while j < len(ids) and ids[j] != msg:
                    j += 1
                header = tok.decode(ids[i + 1:j], skip_special_tokens=False).strip()
                m = re.match(r'[A-Za-z_]+', header)  # 'final', 'final <|constrain|>json', 'commentary to=...'
                name = m.group(0) if m else ''
                if j < len(ids):  # a complete header; body runs to the next stop token
                    k = j + 1
                    while k < len(ids) and ids[k] not in stops:
                        k += 1
                    channels.append(dict(name=name, header=header, body_start=j + 1, body_end=k, closed=k < len(ids)))
                    i = k
                    continue
                channels.append(dict(name=name, header=header, body_start=None, body_end=None, closed=False))
                break
            i += 1
        info['channels'] = [dict(name=c['name'], header=c['header'], closed=c['closed'],
                                 tokens=(c['body_end'] - c['body_start']) if c['body_start'] is not None else 0) for c in channels]
        final = next((c for c in channels if c['name'] == 'final' and c['body_start'] is not None), None)
        info['final_channel_found'] = final is not None
        info['answer_source'] = 'final_channel' if final else 'none'
        info['reasoning_tokens'] = final['body_start'] if final else len(ids)
        if final is None:
            return '', info
        info['final_closed'] = final['closed']
        return tok.decode(ids[final['body_start']:final['body_end']], skip_special_tokens=True), info
    if rule == 'gpt_oss_final_prefill':  # the prompt already ends with '<|channel|>final<|message|>'
        stops = {special_id(tok, t) for t in ('<|return|>', '<|end|>', '<|call|>')}
        k = next((i for i, t in enumerate(ids) if t in stops), len(ids))
        info.update(answer_source='prefilled_final_channel', final_channel_found=True, final_closed=k < len(ids),
                    reasoning_tokens=0, tokens_after_final=len(ids) - k - 1 if k < len(ids) else 0)
        return tok.decode(ids[:k], skip_special_tokens=True), info
    raise AssertionError(rule)


def bank_worlds(bank, shard, shards):
    for si, scene in enumerate(bank):
        if si % shards != shard:
            continue
        for world in scene['worlds']:
            yield si, scene, world


def env_record():
    rec = dict(python=sys.executable, python_version=sys.version.split()[0], hostname=socket.gethostname(),
               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    for mod in ('torch', 'transformers', 'tokenizers', 'accelerate'):
        try:
            rec[mod] = __import__(mod).__version__
        except Exception:
            rec[mod] = None
    try:
        import importlib.util
        rec['fla_available'] = importlib.util.find_spec('fla') is not None
    except Exception:
        rec['fla_available'] = None
    return rec


def shard_path(out, shard, shards):
    return out if shards == 1 else out.with_name(f'{out.stem}.shard{shard}of{shards}{out.suffix}')


def atomic_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=1) + '\n')
    os.replace(tmp, path)


# Header fields that must be identical for a resume or a merge to be valid.
IDENTITY_KEYS = ('sender', 'sender_key', 'sender_checkpoint', 'prompt_sha256', 'system', 'user_template', 'max_new_tokens',
                 'dtype', 'split', 'bank_sha256', 'words', 'dictionary_sha256', 'chat_template_kwargs',
                 'chat_template_sha256', 'rendered_prompt_head', 'rendered_prompt_tail', 'answer_rule')


def summarize(out):
    logs = out['log']
    st = [v['parse_status'] for v in logs.values()]
    n = max(1, len(st))
    out['unique_evidence_inputs'] = len({v['evidence_sha256'] for v in out['packets'].values()})
    out['summary'] = dict(
        n=len(st), ok=st.count('ok'), partial=st.count('partial'), empty=st.count('empty'), malformed=st.count('malformed'),
        mean_decode_seconds=sum(v['decode_seconds'] for v in logs.values()) / n,
        mean_bytes=sum(v['bytes'] for v in out['packets'].values()) / max(1, len(out['packets'])),
        mean_prompt_tokens=sum(v['prompt_tokens'] for v in logs.values()) / n,
        mean_generated_tokens=sum(v['generated_tokens'] for v in logs.values()) / n,
        max_generated_tokens=max((v['generated_tokens'] for v in logs.values()), default=0),
        hit_max_new_tokens=sum(1 for v in logs.values() if v.get('stop_reason') == 'max_new_tokens'),
        reasoning_marker_in_output=sum(1 for v in logs.values() if v.get('answer_info', {}).get('reasoning_marker_in_output')),
        final_channel_missing=sum(1 for v in logs.values() if v.get('answer_info', {}).get('final_channel_found') is False),
        greedy_argmax_mismatches=sum(v.get('greedy_argmax_mismatches') or 0 for v in logs.values()))


def load_model(spec, key, adapter, device, loader='auto'):
    """Adapter first (if present and --loader allows), else transformers directly. Returns (model, record).
    Whatever the route, the loaded weights must be BF16 and unquantized, or the run stops."""
    import torch
    rec = dict(route=None, adapter_key=None)
    model = None
    if loader in ('auto', 'adapter') and adapter is not None and hasattr(adapter, 'load_model'):
        akey, _ = adapter_key(adapter, spec)
        if akey is not None:
            try:
                # the adapter defaults to fp32 (receiver numerics); the sender is BF16 like candidate A. Only the model is
                # taken: the adapter's tokenizer wraps apply_chat_template (gpt-oss: final channel pre-opened), and the
                # sender renders through its own plain AutoTokenizer instead.
                got = adapter.load_model(akey, dtype=torch.bfloat16)
                model = got[0] if isinstance(got, (tuple, list)) else (got.get('model') if isinstance(got, dict) else got)
                rec.update(route='adapter', adapter_key=akey, adapter_record=getattr(model, '_syncomm_latest', None))
            except Exception as e:  # incompatible signature or failure: fall back, and say so in the record
                if loader == 'adapter':
                    raise
                rec['adapter_error'] = f'{type(e).__name__}: {e}'
                print(f'[sender] adapter.load_model failed ({rec["adapter_error"]}); using the direct loader', flush=True)
                model = None
                torch.cuda.empty_cache()
    elif loader == 'adapter':
        raise SystemExit('--loader adapter requested but the adapter (or this sender in it) is unavailable')
    if model is None:
        import transformers
        cls = getattr(transformers, spec['model_class'])
        kwargs = dict(dtype=torch.bfloat16, device_map={'': device}, output_loading_info=True)
        if spec['dequantize_mxfp4']:
            kwargs['quantization_config'] = transformers.Mxfp4Config(dequantize=True)
        model, info = cls.from_pretrained(resolve_model(spec['path']), **kwargs)
        missing = list(info.get('missing_keys') or [])
        mismatched = list(info.get('mismatched_keys') or [])
        rec.update(route='direct', missing_keys=missing[:50], n_missing_keys=len(missing),
                   unexpected_keys_sample=sorted(info.get('unexpected_keys') or [])[:20],
                   n_unexpected_keys=len(info.get('unexpected_keys') or []), n_mismatched_keys=len(mismatched))
        if missing or mismatched:
            raise SystemExit(f'weights did not load cleanly: missing {missing[:10]} mismatched {mismatched[:10]}')
    model.eval()
    dtypes = {}
    for p in model.parameters():
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
    rec['param_dtype_numel'] = dtypes
    rec['model_class'] = type(model).__name__
    rec['devices'] = sorted({str(p.device) for p in model.parameters()})
    floats = {k: v for k, v in dtypes.items() if 'float' in k}
    assert floats and max(floats, key=floats.get) == 'torch.bfloat16', f'weights are not BF16: {dtypes}'
    assert not any(k in dtypes for k in ('torch.uint8', 'torch.int8')), f'quantized weights still present: {dtypes}'
    rec['attn_implementation'] = getattr(model.config, '_attn_implementation', None)
    return model, rec


def run(a):
    key = resolve_sender(a.sender)
    spec = dict(SENDERS[key])
    if a.gptoss_mode == 'final_prefill':
        # Pre-declared ALTERNATIVE (not the default): no analysis at all, as the adapter renders gpt-oss for receivers.
        # Changes the rendered tail and the answer rule, both identity keys, so it can never be mixed with the default.
        if spec['family'] != 'gpt_oss':
            raise SystemExit('--gptoss-mode final_prefill applies to gpt-oss-20b only')
        spec.update(prefill='<|channel|>final<|message|>', answer='gpt_oss_final_prefill')
    words_src = json.loads(a.words_from.read_text())
    words = words_src['words'] if isinstance(words_src, dict) else words_src
    assert len(words) == 221 and len(set(words)) == 221
    index = {w: i for i, w in enumerate(words)}
    bank_bytes = a.bank.read_bytes()
    bank = json.loads(bank_bytes)
    if a.limit:
        bank = bank[:a.limit]
    n_alias = {len(s['aliases']) for s in bank}
    assert len(n_alias) == 1, f'mixed alias counts in bank: {n_alias}'
    n_alias = n_alias.pop()
    system, user_template, prompt_sha = frozen_template(n_alias)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(resolve_model(spec['path']))
    assert tok.chat_template, 'tokenizer has no chat template'
    # reference rendering (first world of the bank) fixes the scaffold every later world must reproduce
    s0, w0 = bank[0], bank[0]['worlds'][0]
    user0 = user_message(user_template, words, s0['aliases'], ' '.join(w0['evidence']))
    text0, head0, tail0 = render(tok, spec, system, user0)

    adapter = load_adapter()
    adapter_rec = None
    if adapter is not None:
        akey, aspec = adapter_key(adapter, spec)
        adapter_rec = dict(key=akey, spec={k: str(v) for k, v in (aspec or {}).items()} if isinstance(aspec, dict) else str(aspec))
        if akey is not None and hasattr(adapter, 'chat_kwargs'):
            try:
                akw = dict(adapter.chat_kwargs(akey))
            except Exception as e:
                akw = f'error: {e}'
            adapter_rec['chat_kwargs'] = akw
            if isinstance(akw, dict) and akw != spec['chat_kwargs']:
                # the frozen kwargs are used either way; record whether the adapter's would render a different prompt
                alt = render(tok, dict(spec, chat_kwargs=akw), system, user0)[0]
                adapter_rec['chat_kwargs_differ'] = True
                adapter_rec['chat_kwargs_render_identical'] = (alt == text0)
                if alt != text0:
                    print(f'[sender] WARNING adapter chat_kwargs {akw} would render a different prompt than the frozen '
                          f'{spec["chat_kwargs"]}; using the frozen ones', flush=True)
        want_py = (aspec or {}).get('python') if isinstance(aspec, dict) else None
        if want_py and os.path.realpath(want_py) != os.path.realpath(sys.executable):
            print(f'[sender] note: adapter declares python {want_py} for {akey}; running {sys.executable}', flush=True)
    chat_kw_record = dict(spec['chat_kwargs'], **({'strftime_now_pinned_date': spec['fixed_date']} if spec.get('fixed_date') else {}),
                          **({'appended_after_template': spec['prefill']} if spec.get('prefill') else {}))

    if a.render_only:
        return render_only(a, key, spec, tok, bank, words, system, user_template, prompt_sha, head0, tail0, chat_kw_record, adapter_rec)

    import torch
    out_path = shard_path(a.out, a.shard, a.shards)
    header = dict(
        sender=f'latest0924/{key}', sender_checkpoint=spec['path'],
        mechanism='candidate A frozen fixed-prompt JSON word-index selector on a new checkpoint; its own chat template; greedy; no retry',
        prompt_sha256=prompt_sha, system=system, user_template=user_template, max_new_tokens=MAX_NEW_TOKENS, dtype='bfloat16',
        split=a.split, bank=str(a.bank), bank_sha256=sha_bytes(bank_bytes), words=words,
        dictionary_sha256=words_src.get('dictionary_sha256') if isinstance(words_src, dict) else None,
        adr=a.adr, packets={}, log={},
        sender_key=key, words_from=str(a.words_from), chat_template_kwargs=chat_kw_record,
        chat_template_sha256=sha_text(tok.chat_template), rendered_prompt_head=head0, rendered_prompt_tail=tail0,
        answer_rule=spec['answer'], description_join="' '.join(world['evidence']) (as candidate A)",
        decoding=dict(do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None, batch_size=1,
                      max_new_tokens=MAX_NEW_TOKENS, eos_token_id=None, attention_mask='all ones'),
        shard=a.shard, shards=a.shards, limit=a.limit or None, script=str(Path(__file__).resolve()),
        script_sha256=sha_bytes(Path(__file__).read_bytes()), adapter=adapter_rec, created_utc=utc())
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        diff = [k for k in IDENTITY_KEYS if prev.get(k) != header.get(k)]
        if diff:
            raise SystemExit(f'{out_path} exists with a different header ({diff}); refusing to mix runs — use another --out')
        header['packets'], header['log'] = prev['packets'], prev['log']
        header['created_utc'] = prev.get('created_utc', header['created_utc'])
        header['resumed_utc'] = (prev.get('resumed_utc') or []) + [utc()]
    out = header
    todo = [(si, s, w) for si, s, w in bank_worlds(bank, a.shard, a.shards) if f"{s['scene']}|{w['world']}" not in out['packets']]
    print(f'[sender {key}] shard {a.shard}/{a.shards}: {len(out["packets"])} done, {len(todo)} to do -> {out_path}', flush=True)
    if not todo:
        summarize(out); out['completed_utc'] = out.get('completed_utc') or utc(); atomic_write(out_path, out)
        print(f'[sender {key}] DONE (nothing to do)', json.dumps(out['summary']), flush=True)
        return

    t_load = time.time()
    model, load_rec = load_model(spec, key, adapter, a.device, a.loader)
    load_rec['load_seconds'] = time.time() - t_load
    out['model_load'] = load_rec
    out['env'] = env_record()
    if torch.cuda.is_available():
        out['env']['device_name'] = torch.cuda.get_device_name(0)
    # one entry per launch, so a resumed file shows every host/env that contributed packets
    out['sessions'] = (out.get('sessions') or []) + [dict(utc=utc(), packets_before=len(out['packets']), env=out['env'],
                                                          model_load=load_rec)]
    eos = model.generation_config.eos_token_id
    eos_set = set(eos if isinstance(eos, list) else [eos]) if eos is not None else {tok.eos_token_id}
    pad = tok.pad_token_id if tok.pad_token_id is not None else sorted(eos_set)[0]
    out['decoding']['eos_token_id'] = sorted(eos_set)
    out['decoding']['pad_token_id'] = pad
    print(f'[sender {key}] loaded {load_rec["model_class"]} via {load_rec["route"]} in {load_rec["load_seconds"]:.0f}s; eos {sorted(eos_set)}', flush=True)

    n_new = 0
    for si, scene, world in todo:
        k = f"{scene['scene']}|{world['world']}"
        aliases = scene['aliases']
        description = ' '.join(world['evidence'])
        user = user_message(user_template, words, aliases, description)
        text_in, head, tail = render(tok, spec, system, user)
        assert head == head0 and tail == tail0, f'{k}: chat scaffold differs from the reference rendering'
        ids_list = tokenize_prompt(tok, text_in, spec)
        ids = torch.tensor([ids_list], device=a.device)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 num_beams=1, temperature=None, top_p=None, top_k=None, pad_token_id=pad,
                                 return_dict_in_generate=True, output_logits=True)
        torch.cuda.synchronize()
        secs = time.time() - t0
        new = gen.sequences[0, ids.shape[1]:].tolist()
        # greedy check: every generated token must be the argmax of the raw logits at its step (no hidden processor)
        mism = sum(int(int(lg[0].argmax()) != t) for lg, t in zip(gen.logits, new)) if gen.logits is not None else None
        del gen
        raw = tok.decode(new, skip_special_tokens=True)
        raw_special = tok.decode(new, skip_special_tokens=False)
        answer, ainfo = extract_answer(tok, spec['answer'], new)
        selection, status, dropped = parse_selection(answer, aliases, index)
        packet, data = to_packet(selection, aliases, words, index)
        stop = ('eos' if new and new[-1] in eos_set else 'max_new_tokens' if len(new) >= MAX_NEW_TOKENS else 'other')
        out['packets'][k] = dict(hex=data.hex(), sha256=sha_bytes(data), bytes=len(data),
                                 empty_lanes=sum(1 for ln in packet.lanes if not ln.concept_ids),
                                 evidence_sha256=sha_text(description))
        out['log'][k] = dict(raw_output=raw, parse_status=status, selection=selection, dropped=dropped, decode_seconds=secs,
                             prompt_tokens=len(ids_list), generated_tokens=len(new),
                             answer_text=answer, answer_info=ainfo, raw_output_special=raw_special,
                             generated_ids=new, stop_reason=stop, last_token=(tok.convert_ids_to_tokens(new[-1]) if new else None),
                             greedy_argmax_mismatches=mism, prompt_sha256=sha_text(text_in), scene_index=si)
        n_new += 1
        if n_new % a.save_every == 0:
            atomic_write(out_path, out)
            print(f'[sender {key}] {len(out["packets"])} packets, last {status} {len(new)} tok {secs:.1f}s', flush=True)
    summarize(out)
    out['completed_utc'] = utc()
    atomic_write(out_path, out)
    print(f'[sender {key}] DONE', json.dumps(out['summary']), flush=True)


def render_only(a, key, spec, tok, bank, words, system, user_template, prompt_sha, head0, tail0, chat_kw_record, adapter_rec):
    """CPU: write the exact rendered prompt of one world, check the scaffold on every world, and count prompt tokens."""
    d = Path(a.render_only)
    d.mkdir(parents=True, exist_ok=True)
    si = next(i for i, s in enumerate(bank) if s['scene'] == a.scene) if a.scene else 0
    scene = bank[si]
    world = scene['worlds'][a.world]
    user = user_message(user_template, words, scene['aliases'], ' '.join(world['evidence']))
    text, head, tail = render(tok, spec, system, user)
    ids = tokenize_prompt(tok, text, spec)
    counts = []
    for _, s, w in bank_worlds(bank, 0, 1):
        t, h, tl = render(tok, spec, system, user_message(user_template, words, s['aliases'], ' '.join(w['evidence'])))
        assert h == head0 and tl == tail0, f"{s['scene']}|{w['world']}: scaffold differs"
        counts.append(len(tokenize_prompt(tok, t, spec)))
    checks = {}
    if spec['family'] == 'gemma4':  # thinking is off by default: identical to passing enable_thinking=False explicitly
        t_off = tok.apply_chat_template([dict(role='system', content=system), dict(role='user', content=user)],
                                        add_generation_prompt=True, tokenize=False, enable_thinking=False)
        checks['default_equals_enable_thinking_false'] = (t_off == text)
        checks['ends_with_empty_thought_block'] = text.endswith('<|turn>model\n<|channel>thought\n<channel|>')
        checks['native_system_turn'] = text.startswith('<bos><|turn>system\n' + system + '<turn|>\n')
    if spec['family'] == 'qwen3_5':
        checks['ends_with_closed_empty_think'] = text.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
        checks['native_system_turn'] = text.startswith('<|im_start|>system\n' + system + '<|im_end|>\n')
    if spec['family'] == 'gpt_oss':
        checks['reasoning_low_line'] = '\nReasoning: low\n' in text
        checks['pinned_date_line'] = f"\nCurrent date: {spec['fixed_date']}\n" in text
        checks['system_text_in_developer_instructions'] = ('<|start|>developer<|message|># Instructions\n\n' + system + '\n\n<|end|>') in text
        checks['ends_with_assistant_start'] = text.endswith('<|end|><|start|>assistant' + spec.get('prefill', ''))
        checks['default_model_identity'] = 'You are ChatGPT, a large language model trained by OpenAI.' in text
    stem = f'{key}{"-final_prefill" if spec.get("prefill") else ""}__{a.split}__{scene["scene"]}_w{world["world"]}'
    (d / f'{stem}.txt').write_text(text)
    meta = dict(sender=key, checkpoint=spec['path'], split=a.split, bank=str(a.bank), bank_sha256=sha_bytes(a.bank.read_bytes()),
                scene=scene['scene'], world=world['world'], aliases=scene['aliases'], evidence=world['evidence'],
                prompt_file=f'{stem}.txt', rendered_prompt_sha256=sha_text(text), frozen_prompt_sha256=prompt_sha,
                chat_template_kwargs=chat_kw_record, chat_template_sha256=sha_text(tok.chat_template),
                tokenizer_class=type(tok).__name__, prompt_tokens=len(ids), first_ids=ids[:8], last_ids=ids[-12:],
                last_tokens=tok.convert_ids_to_tokens(ids[-12:]), rendered_prompt_head=head, rendered_prompt_tail=tail,
                scaffold_identical_across_bank=True, n_worlds_checked=len(counts),
                prompt_tokens_over_bank=dict(n=len(counts), min=min(counts), max=max(counts), mean=sum(counts) / len(counts), total=sum(counts)),
                template_checks=checks, answer_rule=spec['answer'], adapter=adapter_rec, env=env_record(), utc=utc(),
                script_sha256=sha_bytes(Path(__file__).read_bytes()))
    atomic_write(d / f'{stem}.json', meta)
    print(f'[render {key}] {stem}: {len(ids)} prompt tokens; bank {meta["prompt_tokens_over_bank"]}; checks {checks}', flush=True)


def merge(a):
    parts = [shard_path(a.out, i, a.shards) for i in range(a.shards)]
    missing = [str(p) for p in parts if not p.exists()]
    if missing:
        raise SystemExit(f'missing shard files: {missing}')
    docs = [json.loads(p.read_text()) for p in parts]
    ref = docs[0]
    for p, d in zip(parts[1:], docs[1:]):
        diff = [k for k in IDENTITY_KEYS if d.get(k) != ref.get(k)]
        if diff:
            raise SystemExit(f'{p} header differs from shard 0 in {diff}')
    bank = json.loads(Path(ref['bank']).read_bytes())
    assert sha_bytes(Path(ref['bank']).read_bytes()) == ref['bank_sha256'], 'bank changed since generation'
    if ref.get('limit'):
        bank = bank[:ref['limit']]
    expected = [f"{s['scene']}|{w['world']}" for s in bank for w in s['worlds']]
    out = {k: v for k, v in ref.items() if k not in ('packets', 'log', 'summary', 'shard', 'unique_evidence_inputs',
                                                     'env', 'model_load', 'completed_utc', 'resumed_utc')}
    out['packets'], out['log'] = {}, {}
    for i, d in enumerate(docs):
        for k in d['packets']:
            assert k not in out['packets'], f'{k} in two shards'
            out['packets'][k], out['log'][k] = d['packets'][k], d['log'][k]
    absent = [k for k in expected if k not in out['packets']]
    if absent:
        raise SystemExit(f'{len(absent)} worlds not generated yet, e.g. {absent[:3]}')
    out['packets'] = {k: out['packets'][k] for k in expected}  # bank order, like candidate A
    out['log'] = {k: out['log'][k] for k in expected}
    out['shard'] = None
    out['merged_from'] = [dict(path=str(p), sha256=sha_bytes(p.read_bytes()), sessions=d.get('sessions'),
                               n=len(d['packets'])) for p, d in zip(parts, docs)]
    out.pop('sessions', None)
    versions = {json.dumps({m: (s.get('env') or {}).get(m) for m in ('python', 'torch', 'transformers', 'tokenizers')}, sort_keys=True)
                for d in docs for s in (d.get('sessions') or [])}
    out['merged_env_versions'] = sorted(versions)
    if len(versions) > 1:
        print(f'[sender] WARNING shards ran under {len(versions)} different python/library versions: {sorted(versions)}', flush=True)
    out['merged_utc'] = utc()
    summarize(out)
    atomic_write(a.out, out)
    print('[sender] MERGED', a.out, json.dumps(out['summary']), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sender', help=f'one of {sorted(SENDERS)}')
    ap.add_argument('--bank', type=Path)
    ap.add_argument('--split', help='name recorded in the header, e.g. dev96 or r3_confirmation')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--words-from', type=Path, default=DEFAULT_WORDS_FROM,
                    help='packets file (or JSON list) whose 221-word list is the inventory; default: candidate A r3 packets')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--loader', choices=('auto', 'adapter', 'direct'), default='auto',
                    help='auto: syncomm.latest.adapter.load_model if present, else transformers directly')
    ap.add_argument('--gptoss-mode', choices=('analysis_low', 'final_prefill'), default='analysis_low',
                    help='gpt-oss only. analysis_low (default, the frozen rule): reasoning_effort=low, parse the final '
                         'channel. final_prefill: pre-declared alternative with the final channel pre-opened (no analysis)')
    ap.add_argument('--limit', type=int, default=0, help='first N scenes only (smoke)')
    ap.add_argument('--save-every', type=int, default=8)
    ap.add_argument('--adr', default=None, help='decision record path to write into the header (candidate A field)')
    ap.add_argument('--merge', action='store_true', help='combine --shards shard files of --out into --out')
    ap.add_argument('--render-only', default=None, help='CPU: directory for the rendered prompt of one world; no weights loaded')
    ap.add_argument('--scene', default=None, help='--render-only: scene id (default: first scene)')
    ap.add_argument('--world', type=int, default=0, help='--render-only: world index')
    a = ap.parse_args()
    if a.merge:
        return merge(a)
    if not (a.sender and a.bank and a.split):
        ap.error('--sender, --bank and --split are required')
    assert 0 <= a.shard < a.shards
    run(a)


if __name__ == '__main__':
    main()
