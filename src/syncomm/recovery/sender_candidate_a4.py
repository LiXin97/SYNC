"""R4 four-object sender adaptation of candidate A (plan §9) — the ONLY change from the frozen two-object candidate A
(src/syncomm/routeb/sender_candidate_a.py, ADR-b6-candidate-a-20260922) is that the system text says "four objects" and the
user template lists four aliases and a four-key JSON shape. Model, greedy decoding, max_new_tokens, inventory, parse rule
(first balanced JSON object, exact alias keys, inventory words only, dedupe, first four kept, no retry, malformed/empty ->
empty lane in the denominator), equal q=255 and ascending address ordering are imported unchanged. Frozen before any
four-object output; no prompt search.

Usage (GPU): PYTHONPATH=src python3 -m syncomm.recovery.sender_candidate_a4 --bank <r4 bank> --words-from <packets> --out <file>
"""
try:
    from syncomm.models import resolve_model
except ImportError:  # run outside the package: references must already be local paths
    def resolve_model(ref):
        return str(ref)
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from syncomm.routeb.sender_candidate_a import MODEL, MAX_NEW_TOKENS, parse_selection, to_packet

SYSTEM4 = ('You compress object descriptions into a fixed public vocabulary. You will be given a vocabulary of words '
           'and a description of four objects. For each object, choose up to four vocabulary words that the description '
           'explicitly supports. Use only words from the vocabulary, spelled exactly as given. Do not invent facts. '
           'Answer with one JSON object and nothing else.')
USER_TEMPLATE4 = ('Vocabulary ({n} words, in order):\n{vocab}\n\n'
                  'Objects: {alias0}, {alias1}, {alias2}, {alias3}\n'
                  'Description:\n{description}\n\n'
                  'Reply with exactly this JSON shape, listing up to four vocabulary words per object that the description '
                  'explicitly supports (fewer is fine, an empty list if nothing applies):\n'
                  '{{"{alias0}": ["word", "word"], "{alias1}": ["word"], "{alias2}": ["word"], "{alias3}": ["word"]}}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', type=Path, required=True); ap.add_argument('--words-from', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--model', default=MODEL)
    a = ap.parse_args()
    words = json.loads(a.words_from.read_text())['words']
    assert len(words) == 221 and len(set(words)) == 221
    index = {w: i for i, w in enumerate(words)}
    bank = json.loads(a.bank.read_text())
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(resolve_model(a.model))
    model = AutoModelForCausalLM.from_pretrained(resolve_model(a.model), torch_dtype=torch.bfloat16, device_map='cuda:0').eval()
    prompt_sha = hashlib.sha256((SYSTEM4 + '\n' + USER_TEMPLATE4).encode()).hexdigest()
    out = dict(sender='candidate_A4/llama-3.1-8b-instruct', sender_checkpoint=a.model,
               mechanism='four-object adaptation of the frozen fixed-prompt JSON word-index selector; greedy; no retry',
               prompt_sha256=prompt_sha, system=SYSTEM4, user_template=USER_TEMPLATE4, max_new_tokens=MAX_NEW_TOKENS, dtype='bfloat16',
               split='r4_four_object', bank=str(a.bank), bank_sha256=hashlib.sha256(a.bank.read_bytes()).hexdigest(), words=words,
               dictionary_sha256=json.loads(a.words_from.read_text()).get('dictionary_sha256'), packets={}, log={})
    if a.out.exists():
        prev = json.loads(a.out.read_text())
        if prev.get('prompt_sha256') == prompt_sha and prev.get('words') == words:
            out['packets'] = prev['packets']; out['log'] = prev.get('log', {})
    vocab = ', '.join(words); n_done = 0
    for scene in bank:
        al = scene['aliases']; assert len(al) == 4
        for world in scene['worlds']:
            key = f"{scene['scene']}|{world['world']}"
            if key in out['packets']:
                continue
            description = ' '.join(world['evidence'])
            msgs = [dict(role='system', content=SYSTEM4),
                    dict(role='user', content=USER_TEMPLATE4.format(n=len(words), vocab=vocab, alias0=al[0], alias1=al[1], alias2=al[2], alias3=al[3], description=description))]
            text_in = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            ids = tok(text_in, return_tensors='pt', add_special_tokens=False)['input_ids'].to('cuda:0')
            t0 = time.time()
            with torch.inference_mode():
                gen = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None, top_p=None, pad_token_id=tok.eos_token_id)
            secs = time.time() - t0
            text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
            selection, status, dropped = parse_selection(text, al, index)
            packet, data = to_packet(selection, al, words, index)
            out['packets'][key] = dict(hex=data.hex(), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                                       empty_lanes=sum(1 for ln in packet.lanes if not ln.concept_ids),
                                       evidence_sha256=hashlib.sha256(description.encode()).hexdigest())
            out['log'][key] = dict(raw_output=text, parse_status=status, selection=selection, dropped=dropped, decode_seconds=secs,
                                   prompt_tokens=int(ids.shape[1]), generated_tokens=int(gen.shape[1] - ids.shape[1]))
            n_done += 1
            if n_done % 16 == 0:
                a.out.write_text(json.dumps(out, indent=1) + '\n')   # resumable partial writes, as in the frozen two-object sender
                print(f'[candidate_a4] {len(out["packets"])} packets, last status {status}, {secs:.1f}s', flush=True)
    st = [v['parse_status'] for v in out['log'].values()]
    out['summary'] = dict(n=len(st), ok=st.count('ok'), partial=st.count('partial'), empty=st.count('empty'), malformed=st.count('malformed'),
                          mean_decode_seconds=sum(v['decode_seconds'] for v in out['log'].values()) / max(1, len(st)),
                          mean_bytes=sum(v['bytes'] for v in out['packets'].values()) / max(1, len(out['packets'])))
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    print('[candidate_a4] DONE', json.dumps(out['summary']), flush=True)


if __name__ == '__main__':
    main()
