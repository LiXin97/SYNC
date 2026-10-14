"""B6 / G1 automatic sender candidate A — fixed-prompt word-index selector. Frozen in a dated decision record.

Reads one world's private description and the public 221-word inventory, asks an instruction-following model for a
JSON selection of up to four inventory words per object, and serializes it as a J-COM packet with equal 8-bit weights.
No retries, no correction, no gold. Writes a packets file in the counterfactual0917 format so the B1-family evaluators
consume it unchanged (`packets[scene|world] = {hex, sha256, bytes, …}`, `words`).

Usage (GPU):
  PYTHONPATH=src python3 -m syncomm.routeb.sender_candidate_a --bank <bank.json> --words-from <packets.json> \
      --out <packets_out.json> [--model /path/to/models/Llama-3.1-8B-Instruct] [--limit N]
"""
try:
    from syncomm.models import resolve_model
except ImportError:  # run outside the package: references must already be local paths
    def resolve_model(ref):
        return str(ref)
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import torch

from syncomm.packet import Lane, Packet, serialize, deserialize, inventory_digest

MODEL = '/path/to/models/Llama-3.1-8B-Instruct'
SYSTEM = ('You compress object descriptions into a fixed public vocabulary. You will be given a vocabulary of words '
          'and a description of two objects. For each object, choose up to four vocabulary words that the description '
          'explicitly supports. Use only words from the vocabulary, spelled exactly as given. Do not invent facts. '
          'Answer with one JSON object and nothing else.')
USER_TEMPLATE = ('Vocabulary ({n} words, in order):\n{vocab}\n\n'
                 'Objects: {alias0}, {alias1}\n'
                 'Description:\n{description}\n\n'
                 'Reply with exactly this JSON shape, listing up to four vocabulary words per object that the description '
                 'explicitly supports (fewer is fine, an empty list if nothing applies):\n'
                 '{{"{alias0}": ["word", "word"], "{alias1}": ["word"]}}')
MAX_NEW_TOKENS = 256
WIDTH = 4


def first_json_object(text):
    """First balanced {...} in the output, or None."""
    start = text.find('{')
    while start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find('{', start + 1)
    return None


def parse_selection(text, aliases, index):
    """Frozen parse rule from the ADR. Returns (lanes as {alias: [word...]}, status, dropped)."""
    obj = first_json_object(text)
    if not isinstance(obj, dict):
        return {a: [] for a in aliases}, 'malformed', {'reason': 'no json object'}
    out, dropped, statuses = {}, {}, []
    for a in aliases:
        v = obj.get(a)
        if not isinstance(v, list):
            out[a] = []; statuses.append('missing'); dropped[a] = 'missing or non-list'; continue
        kept, bad = [], []
        for w in v:
            if not isinstance(w, str):
                bad.append(repr(w)); continue
            key = w.strip().lower()
            if key in index and key not in kept:
                kept.append(key)
            elif key not in index:
                bad.append(w)
        out[a] = kept[:WIDTH]
        dropped[a] = dict(not_in_inventory=bad, beyond_four=kept[WIDTH:])
        statuses.append('ok' if kept else 'empty')
    status = 'ok' if all(s == 'ok' for s in statuses) else ('empty' if all(s in ('empty', 'missing') for s in statuses) else 'partial')
    return out, status, dropped


def to_packet(selection, aliases, words, index):
    lanes = []
    for slot, a in enumerate(aliases):
        ids = tuple(sorted(index[w] for w in selection[a]))
        lanes.append(Lane(slot, ids, tuple(255 for _ in ids)))
    packet = Packet(inventory_digest(words), 8, tuple(lanes))
    data = serialize(packet)
    assert serialize(deserialize(data)) == data
    return packet, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--words-from', type=Path, required=True, help='an existing packets file; its 221-word list is the inventory')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    words = json.loads(a.words_from.read_text())['words']
    assert len(words) == 221 and len(set(words)) == 221
    index = {w: i for i, w in enumerate(words)}
    bank = json.loads(a.bank.read_text())
    if a.limit:
        bank = bank[:a.limit]
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(resolve_model(a.model))
    model = AutoModelForCausalLM.from_pretrained(resolve_model(a.model), torch_dtype=torch.bfloat16, device_map='cuda:0').eval()
    prompt_sha = hashlib.sha256((SYSTEM + '\n' + USER_TEMPLATE).encode()).hexdigest()
    out = dict(sender='candidate_A/llama-3.1-8b-instruct', sender_checkpoint=a.model, mechanism='fixed-prompt JSON word-index selector; greedy; no retry',
               prompt_sha256=prompt_sha, system=SYSTEM, user_template=USER_TEMPLATE, max_new_tokens=MAX_NEW_TOKENS, dtype='bfloat16',
               split='dev', bank=str(a.bank), bank_sha256=hashlib.sha256(a.bank.read_bytes()).hexdigest(), words=words,
               dictionary_sha256=json.loads(a.words_from.read_text()).get('dictionary_sha256'),
               adr='a dated decision record', packets={}, log={})
    if a.out.exists():  # resume: keep finished (scene, world) entries
        prev = json.loads(a.out.read_text())
        if prev.get('prompt_sha256') == prompt_sha and prev.get('words') == words:
            out['packets'] = prev['packets']; out['log'] = prev.get('log', {})
    vocab = ', '.join(words)
    n_done = 0
    for scene in bank:
        aliases = scene['aliases']
        for world in scene['worlds']:
            key = f"{scene['scene']}|{world['world']}"
            if key in out['packets']:
                continue
            description = ' '.join(world['evidence'])
            msgs = [dict(role='system', content=SYSTEM),
                    dict(role='user', content=USER_TEMPLATE.format(n=len(words), vocab=vocab, alias0=aliases[0], alias1=aliases[1], description=description))]
            # transformers 5.x: apply_chat_template returns a dict when tokenizing, so render text first (the project rules trap)
            text_in = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            ids = tok(text_in, return_tensors='pt', add_special_tokens=False)['input_ids'].to('cuda:0')
            t0 = time.time()
            with torch.inference_mode():
                gen = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None, top_p=None, pad_token_id=tok.eos_token_id)
            secs = time.time() - t0
            text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
            selection, status, dropped = parse_selection(text, aliases, index)
            packet, data = to_packet(selection, aliases, words, index)
            out['packets'][key] = dict(hex=data.hex(), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                                       empty_lanes=sum(1 for ln in packet.lanes if not ln.concept_ids),
                                       evidence_sha256=hashlib.sha256(description.encode()).hexdigest())
            out['log'][key] = dict(raw_output=text, parse_status=status, selection=selection, dropped=dropped, decode_seconds=secs,
                                   prompt_tokens=int(ids.shape[1]), generated_tokens=int(gen.shape[1] - ids.shape[1]))
            n_done += 1
            if n_done % 16 == 0:
                a.out.write_text(json.dumps(out, indent=1) + '\n')
                print(f'[candidate_a] {len(out["packets"])} packets, last status {status}, {secs:.1f}s', flush=True)
    out['unique_evidence_inputs'] = len({v['evidence_sha256'] for v in out['packets'].values()})
    st = [v['parse_status'] for v in out['log'].values()]
    out['summary'] = dict(n=len(st), ok=st.count('ok'), partial=st.count('partial'), empty=st.count('empty'), malformed=st.count('malformed'),
                          mean_decode_seconds=sum(v['decode_seconds'] for v in out['log'].values()) / max(1, len(st)),
                          mean_bytes=sum(v['bytes'] for v in out['packets'].values()) / max(1, len(out['packets'])))
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    print('[candidate_a] DONE', json.dumps(out['summary']), flush=True)


if __name__ == '__main__':
    main()
