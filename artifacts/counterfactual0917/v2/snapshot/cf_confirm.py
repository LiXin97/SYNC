"""Locked post-calibration confirmation; separate from the historical pipeline."""
from pathlib import Path
import argparse,contextlib,copy,hashlib,json,sys
import torch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'README.md').exists())
sys.path.insert(0,str(ROOT/'artifacts/counterfactual0917/v2/snapshot'))
import counterfactual_run as base
import syncomm
syncomm.__path__.append(str(ROOT/'src/syncomm'))
from syncomm.counterfactual_controls import matching
import syncomm.writer as writer_module
sys.path.insert(0,str(ROOT/'infra/pi0917'))
from exact_solver_cache import ExactSolverCache
sys.path.insert(0,str(ROOT/'src/syncomm'))
import content_counterfactual as task


def swapped(packet):
    lanes={x.slot_id:x for x in packet.lanes}
    return base.Packet(packet.inventory_digest,packet.qbits,
        tuple(base.Lane(i,lanes[1-i].concept_ids,lanes[1-i].q) for i in (0,1)))


class Receiver(base.Receiver):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.norm_targets=None;self.norm_cache={}

    @contextlib.contextmanager
    def hooks(self,positions,packet,gamma,writer,zero,report):
        if self.norm_targets is None or gamma==0 or packet is None or zero:
            with super().hooks(positions,packet,gamma,writer,zero,report) as hs:
                yield hs
            return
        hs=[]
        try:
            band=self.band if writer=='band' else self.band[:1]
            for li in band:
                for ln in packet.lanes:
                    if not ln.concept_ids:continue
                    q=base.dequantize(ln.q,packet.qbits).cuda()
                    weights=q*(self.mag*gamma/q.sum()) if q.sum()>0 else q
                    norms={p:self.norm_targets[f'{li}|{ln.slot_id}|{p}'] for p in positions[ln.slot_id]}
                    hs.append(base.ObservedWrite(self.model,li,[positions[ln.slot_id]],'replace',
                        D=self.D,support=list(ln.concept_ids),a_hat=weights,lam1=.05,lam2=.001,
                        target_norm=norms,report=report,span=ln.slot_id))
            yield hs
        finally:
            for h in hs:h.remove()

    def matched(self,scene,qi,packet,gamma,targets,replay=False):
        key=base.sha(json.dumps(targets,sort_keys=True).encode())
        old=self.cache;self.cache=self.norm_cache.setdefault(key,{})
        self.norm_targets=targets
        try:return self.evaluate(scene,qi,packet,gamma,replay=replay)
        finally:self.norm_targets=None;self.cache=old


def run(m,args,bank):
    choices=json.loads(Path(m['selection']).read_text())['choices'][str(args.pair)]
    pair=m['pairs'][args.pair];split='dev' if args.phase=='smoke' else 'eval'
    pdata=json.loads((Path(m['root'])/'packets'/f"{split}_{pair['sender']}.json").read_text())
    match=json.loads((Path(m['root'])/f"matching_{split}_{pair['sender']}.json").read_text())
    bank=[s for s in bank if s['split']==split]
    selected=bank[args.shard::args.shards]
    if args.phase=='smoke':selected=selected[:2]
    out=Path(m['root'])/'confirmation_raw'/f'{args.phase}_p{args.pair}_s{args.shard}.jsonl'
    out.parent.mkdir(exist_ok=True);done={}
    if out.exists():
        for line in out.open():
            r=json.loads(line);assert r['key'] not in done;done[r['key']]=r
    if out.with_suffix('.done.json').exists():
        completed=json.loads(out.with_suffix('.done.json').read_text())
        assert completed['status']=='PASS' and completed['rows']==len(done)
        print('Already complete:',out,flush=True)
        return
    cache=ExactSolverCache(writer_module._solve_a0,verify_every=64)
    writer_module._solve_a0=cache
    receiver=Receiver(m,pair['receiver'],task)
    assert receiver.words==pdata['words']
    gamma=choices['band']['gamma'];single=choices['single']['gamma']
    configs=[('none',0,'band'),('zero',0,'band'),('oracle',0,'band'),
        ('selected',0,'band'),('exact_text',0,'band'),('write',gamma,'band'),
        ('swap',gamma,'band'),('matched_norm',gamma,'band'),
        ('single',single,'single'),('reference',8,'band'),
        ('donor',gamma,'band'),('donor_swap',gamma,'band')]
    expected=0
    for si,scene in enumerate(selected):
        donor_id=match['assignment'].get(scene['scene'])
        for world in scene['worlds']:
            rec=pdata['packets'][f"{scene['scene']}|{world['world']}"]
            packet=base.deserialize(bytes.fromhex(rec['hex']))
            for qi in (0,1):
                for arm,g,w in configs:
                    if arm.startswith('donor') and donor_id is None:continue
                    if arm=='reference' and args.phase=='eval' and scene['scene'] not in m['reference_scenes']:
                        continue
                    expected+=1
                    key='|'.join(map(str,[scene['scene'],world['world'],qi,arm,g,w]))
                    if key in done:continue
                    incoming=packet if arm in ['write','swap','matched_norm','single','reference'] else None
                    donor_rec=None
                    if arm.startswith('donor'):
                        donor_rec=pdata['packets'][f"{donor_id}|{world['world']}"]
                        incoming=base.deserialize(bytes.fromhex(donor_rec['hex']))
                    if arm in ['swap','donor_swap']:incoming=swapped(incoming)
                    extra=(' '.join(world['evidence']) if arm=='oracle' else
                        base.packet_text(packet,receiver.words,scene['aliases'],arm=='exact_text')
                        if arm in ['selected','exact_text'] else None)
                    replay=args.phase=='smoke' and si==0
                    targets=None
                    if arm=='matched_norm' and world['world']==1 and g>0:
                        ref=done['|'.join(map(str,[scene['scene'],0,qi,'write',g,'band']))]
                        _,positions=receiver.prompt(scene,qi,None)
                        targets={f'{li}|{i}|{p}':ref['sites'].get(f'{li}|{i}|{p}',{}).get('actual',0.)
                                 for li in receiver.band for i,pos in enumerate(positions) for p in pos}
                        if targets:
                            result=receiver.matched(scene,qi,incoming,g,targets,replay=replay)
                        else:
                            result=receiver.evaluate(scene,qi,incoming,g,w,replay=replay)
                    else:
                        result=receiver.evaluate(scene,qi,incoming,g,w,extra,zero=arm=='zero',replay=replay)
                    if arm=='zero':
                        ref=done['|'.join(map(str,[scene['scene'],world['world'],qi,'none',0,'band']))]
                        assert result['logits_sha256']==ref['logits_sha256'] and result['tokens']==ref['tokens']
                    errors=[]
                    if targets:
                        assert set(result['sites'])<=set(targets)
                        errors=[abs(result['sites'].get(k,{}).get('actual',0.)-v)/max(v,1e-8)
                                for k,v in targets.items()]
                    row=dict(key=key,pair=args.pair,scene=scene['scene'],family=scene['family'],
                        split=split,domain=scene['domain'],template=scene['template'],
                        world=world['world'],question=qi,aliases=scene['aliases'],gold=world['answers'][qi],
                        arm=arm,gamma=g,writer=w,magnitude=receiver.mag,
                        sender_packet_sha256=rec['sha256'],sender_payload_bytes=rec['bytes'],
                        empty_lanes=rec['empty_lanes'],donor_scene=donor_id if arm.startswith('donor') else None,
                        donor_packet_sha256=donor_rec['sha256'] if donor_rec else None,
                        norm_target_world=0 if arm=='matched_norm' else None,
                        norm_matching_active=bool(targets),norm_relative_error_max=max(errors,default=None),**result)
                    for budget in (64,256):row[f'score{budget}']=task.score_answer(row[f'text{budget}'],scene['aliases'],row['gold'])
                    with out.open('a') as f:f.write(json.dumps(row)+'\n')
                    done[key]=row
        print('CONFIRM',args.pair,args.shard,args.phase,si+1,'/',len(selected),flush=True)
    assert len(done)==expected,(len(done),expected)
    base.write_json(out.with_suffix('.done.json'),dict(status='PASS',rows=expected,scenes=len(selected),
        matched_scenes=sum(s['scene'] in match['assignment'] for s in selected),phase=args.phase,
        pair=args.pair,shard=args.shard,solver_cache=dict(hits=cache.hits,misses=cache.misses,verified=cache.verified)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--phase',choices=['extract','match','smoke','eval'],required=True)
    ap.add_argument('--sender');ap.add_argument('--pair',type=int,default=0)
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1)
    args=ap.parse_args();m=json.loads(args.manifest.read_text())
    assert args.phase in m.get('allowed_phases',['extract','match','smoke','eval'])
    for p,h in m['pins'].items():assert base.sha(Path(p).read_bytes())==h,p
    assert Path(m['selection']).exists(),'No confirmation before development selection'
    bank=json.loads(Path(m['bank']).read_text());torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=args.phase!='extract'
    if args.phase=='extract':
        args.split='eval';base.extract(m,args,[s for s in bank if s['split']=='eval'])
    elif args.phase=='match':
        inventory=json.loads(Path(m['inventory']).read_text())
        types={c['word']:c['type'] for c in inventory['concepts']}
        for split in ['dev','eval']:
            pdata=json.loads((Path(m['root'])/'packets'/f'{split}_{args.sender}.json').read_text())
            result=matching([s for s in bank if s['split']==split],pdata['packets'],pdata['words'],types)
            base.write_json(Path(m['root'])/f'matching_{split}_{args.sender}.json',result)
            print('MATCH',split,args.sender,len(result['assignment']),'eligible',flush=True)
    else:run(m,args,bank)


if __name__=='__main__':main()
