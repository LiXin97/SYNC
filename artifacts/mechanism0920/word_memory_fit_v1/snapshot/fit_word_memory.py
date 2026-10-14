"""Fit receiver-only contextual word contrasts, without communication outcomes.

Every inventory word is retained. No scene bank, real question, answer,
source-model state or sender-specific outcome is used by the fitting procedure.
"""
import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'README.md').exists())
PARENT=ROOT/'artifacts/mechanism0920/framing_reference_v1/snapshot/framing_reference.py'
spec=importlib.util.spec_from_file_location('frozen_framing',PARENT)
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
prior,base=parent.prior,parent.base
from memory_layout import render,token_spans,NEUTRAL_WORD

CANONICAL=dict(aliases=['XOPA','YUME'],questions=['FITTING_QUESTION_A?','FITTING_QUESTION_B?'])


def construction(receiver,word,width,obj,index):
    selected=[[NEUTRAL_WORD]*width for _ in (0,1)]
    selected[obj][index]=word
    message,chars=render(CANONICAL['aliases'],selected,width=width)
    texts,sequences,_,n=receiver.tokenized(CANONICAL,message)
    enc=receiver.tok(texts[0],return_offsets_mapping=True,add_special_tokens=not receiver.chat)
    assert enc['input_ids']==sequences[0]
    spans=token_spans(enc['offset_mapping'],chars,texts[0].index(message))
    chosen=next(s for s in spans if s['object']==obj and s['index']==index)
    assert chosen['word']==word and max(chosen['positions'])<n
    source=sequences[0][:n]
    filler=receiver.tok(' unknown',add_special_tokens=False)['input_ids']
    assert len(filler)==1
    neutral=list(source)
    for pos in chosen['positions']:
        neutral[pos]=filler[0]
    assert all(a==b for i,(a,b) in enumerate(zip(source,neutral)) if i not in chosen['positions'])
    assert len(source)==len(neutral)
    # Prefix equality across the two different dummy questions is enforced by
    # the inherited tokenized() split. No question token enters either forward.
    return dict(source=source,neutral=neutral,position=chosen['last_position'],
                word_positions=chosen['positions'],width=width,object=obj,index=index)


@contextlib.contextmanager
def capture(receiver,position,output):
    handles=[]
    for layer in receiver.band:
        def hook(module,args,result,layer=layer):
            h=result[0] if isinstance(result,tuple) else result
            output[layer]=h[0,position].detach().float().cpu().clone()
        handles.append(base.decoder_layers(receiver.model)[layer].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:handle.remove()


@torch.inference_mode()
def read(receiver,ids,position):
    states={};t=torch.tensor([ids],device='cuda:0')
    with capture(receiver,position,states):
        receiver.model(t,attention_mask=torch.ones_like(t),use_cache=False)
    assert set(states)==set(receiver.band)
    return torch.stack([states[layer] for layer in receiver.band])


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--phase',choices=['engineering','fit'],default='fit');p.add_argument('--pair',type=int,required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=4)
    a=p.parse_args();m=json.loads(a.manifest.read_text());root=Path(m['root'])
    for path,sha in m['pins'].items():assert base.sha(Path(path).read_bytes())==sha,path
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    key=m['pairs'][a.pair]['receiver']
    receiver=prior.Receiver.__new__(prior.Receiver)
    receiver.model,receiver.tok=base.load_model(m['models'][key])
    receiver.chat=key!='qwen3-14b-base';receiver.task=parent.Scaffold
    receiver.band=sorted(set(base.tap_layer(receiver.model,x) for x in (.4,.5,.6)))
    words=json.loads(Path(m['words_file']).read_text())
    assert len(words)==221 and len(set(words))==len(words) and NEUTRAL_WORD not in words
    target=root/'rows'/f'p{a.pair}';target.mkdir(parents=True,exist_ok=True)
    cache={};records=[];started=time.time();manifest_sha=base.sha(a.manifest.read_bytes())
    addresses=[words.index(word) for word in m['engineering_words']] if a.phase=='engineering' else list(range(a.shard,len(words),a.shards))
    for address in addresses:
        word=words[address];dest=target/f'w{address:03d}.pt'
        if dest.exists():
            stored=torch.load(dest,map_location='cpu',weights_only=False)
            assert stored['manifest_sha256']==manifest_sha and stored['address']==address and stored['word']==word
            records.append(dict(address=address,path=str(dest),sha256=base.sha(dest.read_bytes())))
            continue
        contrasts=[];contexts=[]
        for width in m['context_widths']:
            for obj in (0,1):
                for index in range(width):
                    c=construction(receiver,word,width,obj,index)
                    cachekey=(tuple(c['neutral']),c['position'])
                    if cachekey not in cache:cache[cachekey]=read(receiver,c['neutral'],c['position'])
                    source_state=read(receiver,c['source'],c['position'])
                    if a.phase=='engineering':
                        replay=read(receiver,c['source'],c['position'])
                        if not torch.equal(source_state,replay):
                            prior.atomic_json(root/f'engineering_failure_p{a.pair}.json',dict(word=word,context=c,max_difference=float((source_state-replay).abs().max()),clause='exact hidden-state replay'))
                            raise AssertionError('Fitting source states do not replay exactly')
                    delta=source_state-cache[cachekey]
                    assert torch.isfinite(delta).all()
                    contrasts.append(delta);contexts.append(c)
        assert len(contrasts)==12
        stacked=torch.stack(contrasts)
        # One fixed deterministic average; no behavior-based row admission.
        direction=stacked.double().mean(dim=0).float()
        blob=dict(manifest_sha256=manifest_sha,pair=a.pair,receiver=key,address=address,word=word,layers=receiver.band,
                  contexts=contexts,contrasts=stacked,direction=direction,
                  construction='Mean of12receiver-only one-word-minus-token-length-matched-neutral contrasts at final word token; widths2and4, all object/rank positions.')
        temporary=dest.with_suffix('.tmp');torch.save(blob,temporary);os.replace(temporary,dest)
        records.append(dict(address=address,path=str(dest),sha256=base.sha(dest.read_bytes())))
        print('WORD',address,word,'contexts',len(contrasts),'norms',direction.norm(dim=1).tolist(),flush=True)
    receipt=dict(status='COMPLETE_RECEIVER_LOCAL_FIT_ONLY',phase=a.phase,manifest_sha256=manifest_sha,pair=a.pair,shard=a.shard,
                 words=len(records),rows=records,elapsed_seconds=time.time()-started,
                 no_communication_outcomes=True)
    prior.atomic_json(root/f'{a.phase}_p{a.pair}_s{a.shard}.done.json',receipt)
    print('COMPLETE',a.pair,a.shard,len(records),flush=True)


if __name__=='__main__':
    main()
