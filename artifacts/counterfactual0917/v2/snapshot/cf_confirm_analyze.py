"""Read the complete fixed confirmation; no interim decisions or row filtering."""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,json,sys
import numpy as np
R=next(p for p in Path(__file__).resolve().parents if (p/'README.md').exists())
sys.path.insert(0,str(R/'src'))
from syncomm.counterfactual_analysis import collapse,summarize,family_values,holm
from syncomm.counterfactual_statistics import stable_bootstrap
from syncomm.counterfactual_controls import keep_cycles
from syncomm.content_counterfactual import score_answer


def interval(records,metric):
    return stable_bootstrap(family_values(records,metric))


def contrast(records,treatment,reference,metric,cohort=None):
    a={r['scene']:r for r in records if r['condition']==treatment}
    b={r['scene']:r for r in records if r['condition']==reference}
    selected=set(a)&set(b) if cohort is None else set(cohort)
    assert selected and selected<=set(a) and selected<=set(b)
    d=[dict(scene=s,family=a[s]['family'],value=a[s][metric]-b[s][metric]) for s in sorted(selected)]
    return dict(scenes=len(d),**interval(d,'value'))


def main(root,manifest,shards):
    m=json.loads(manifest.read_text());selection=json.loads(Path(m['selection']).read_text())['choices']
    scenes={s['scene']:s for s in json.loads(Path(m['bank']).read_text()) if s['split']=='eval'}
    assert len(scenes)==720 and len({s['family'] for s in scenes.values()})==30
    results={};primary=[];secondary=[];pins={}
    for pair in range(4):
        rows=[]
        for shard in range(shards):
            p=root/'confirmation_raw'/f'eval_p{pair}_s{shard}.jsonl'
            marker=json.loads(p.with_suffix('.done.json').read_text());assert marker['status']=='PASS'
            rr=[json.loads(l) for l in p.open()];assert len(rr)==marker['rows'];rows+=rr
            pins[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        index={r['key']:r for r in rows};assert len(index)==len(rows)
        by=defaultdict(dict)
        for r in rows:
            assert r['pair']==pair and r['split']=='eval'
            s=scenes[r['scene']];assert r['gold']==s['worlds'][r['world']]['answers'][r['question']]
            for budget in [64,256]:assert r[f'score{budget}']==score_answer(r[f'text{budget}'],s['aliases'],r['gold'])
            by[r['scene'],r['world'],r['question']][r['arm']]=r
            if r['arm']=='zero':
                k=r['key'].replace('|zero|','|none|');ref=index[k]
                assert r['tokens']==ref['tokens'] and r['logits_sha256']==ref['logits_sha256']
                assert r['hook_writes']>0 and r['sites'] and all(x['actual']==0 for x in r['sites'].values())
            if r['gamma']>0 and r['arm'] in ['write','single','reference','swap'] and r['empty_lanes']<2:
                assert r['hook_writes']>0
            key=r['key'].split('|');key[2]=str(1-r['question']);other=index['|'.join(key)]
            assert r['sender_packet_sha256']==other['sender_packet_sha256']
            if r['arm'] not in ['oracle','selected','exact_text']:
                key=r['key'].split('|');key[1]=str(1-r['world']);other=index['|'.join(key)]
                assert r['prompt_sha256']==other['prompt_sha256']
                if r['gamma']==0 and r['arm'] not in ['zero']:
                    assert r['logits_sha256']==other['logits_sha256'] and r['tokens']==other['tokens']
        choice=selection[str(pair)];g=choice['band']['gamma'];gs=choice['single']['gamma']
        match=json.loads((root/f"matching_eval_{m['pairs'][pair]['sender']}.json").read_text())
        for sid in scenes:
            expected={'none','zero','oracle','selected','exact_text','write','swap','matched_norm','single'}
            if sid in m['reference_scenes']:expected.add('reference')
            if sid in match['assignment']:expected|={'donor','donor_swap'}
            for w in [0,1]:
                for q in [0,1]:assert set(by[sid,w,q])==expected,(sid,w,q)
        records=collapse(rows,scenes);long=collapse(rows,scenes,256)
        summary=summarize(records);summary256=summarize(long)
        oracle=summary['oracle:0:band']['accuracy'];none=summary['none:0:band']['accuracy']
        oracle_valid=choice['oracle_valid'] and oracle>=.8 and oracle-none>=.2
        cond=f'write:{g}:band';single=f'single:{gs}:single'
        for metric,minimum in [('accuracy',.05),('all_four',.05),('related_gap',0.)]:
            c=contrast(records,cond,'none:0:band',metric)
            valid=oracle_valid and choice['band']['eligible']
            primary.append(dict(pair=pair,metric=metric,minimum=minimum,eligible=valid,
                p_for_family=c['p'] if valid else 1.,**c))
        for metric in ['accuracy','all_four']:
            c=contrast(records,single,'none:0:band',metric)
            valid=oracle_valid and choice['single']['eligible']
            secondary.append(dict(pair=pair,metric=metric,minimum=.05,eligible=valid,
                p_for_family=c['p'] if valid else 1.,**c))
        table={}
        for name,stats in summary.items():
            rr=[x for x in records if x['condition']==name]
            if stats['families']>=10:
                table[name]={k:interval(rr,k) for k in ['accuracy','coverage','relevant_both','unrelated_both','all_four','related_gap','unrelated_abs_gap']}
            else:table[name]=dict(status='Too few material families for interval reporting',point=stats)
        # Actual write norms, with no pairing fallback or per-outcome decisions.
        eligible={};details={}
        for sid in match['assignment']:
            worst=0.;active=False
            for w in [0,1]:
                for q in [0,1]:
                    cell=by[sid,w,q]
                    for real,donor in [('write','donor'),('swap','donor_swap')]:
                        a,b=cell[real]['sites'],cell[donor]['sites']
                        for site in set(a)|set(b):
                            x=a.get(site,{}).get('actual',0.);y=b.get(site,{}).get('actual',0.)
                            active|=x>0 or y>0
                            worst=max(worst,abs(x-y)/max(x,1e-8))
            eligible[sid]=g>0 and active and worst<=.05;details[sid]=worst
        strict=keep_cycles(match['assignment'],eligible)
        strict_records=[]
        for sid in strict:
            score=0.;correct=0.
            for w in [0,1]:
                for q in [0,1]:
                    c=by[sid,w,q];sign=1 if c['write']['gold']==scenes[sid]['aliases'][0] else -1
                    score+=sign*(c['write']['gap']-c['swap']['gap']-c['donor']['gap']+c['donor_swap']['gap'])/4
                    correct+=(int(c['write']['score64']['correct'])-int(c['swap']['score64']['correct'])-
                              int(c['donor']['score64']['correct'])+int(c['donor_swap']['score64']['correct']))/4
            strict_records.append(dict(scene=sid,family=scenes[sid]['family'],score=score,correctness=correct))
        strict_families=len({x['family'] for x in strict_records})
        donor_report=dict(candidates=len(match['assignment']),candidate_families=len({scenes[s]['family'] for s in match['assignment']}),
            actual_matched=len(strict),actual_families=strict_families,
            status='No nonzero intervention selected' if g==0 else 'Descriptive strict subset',
            maximum_relative_norm_error=details,retained_assignment=strict)
        if strict_families>=10:
            donor_report['score_interaction']=interval(strict_records,'score')
            donor_report['correctness_interaction']=interval(strict_records,'correctness')
        norm_rows=[r for r in rows if r['arm']=='matched_norm' and r['world']==1]
        norm_good={sid:all(r['norm_matching_active'] and r['norm_relative_error_max']<=.01
                         for r in norm_rows if r['scene']==sid) for sid in scenes}
        subgroup={}
        for field in ['color','taste','material']:
            subset=[r for r in records if scenes[r['scene']]['changed_field']==field]
            subgroup[field]=summarize(subset)
        shared=set(m['reference_scenes'])
        reference_report={metric:{'reference_minus_none':contrast(records,'reference:8:band','none:0:band',metric,shared),
                                   'selected_minus_reference':contrast(records,cond,'reference:8:band',metric,shared)}
                          for metric in ['accuracy','coverage','related_gap']}
        results[str(pair)]=dict(pair=m['pairs'][pair],rows=len(rows),scenes=720,families=30,
            choice=choice,oracle_valid=bool(oracle_valid),summary64=summary,summary256=summary256,
            intervals=table,strict_donor=donor_report,norm_match=dict(eligible_scenes=sum(norm_good.values()),total=720,
              max_relative_error=max((r['norm_relative_error_max'] or 0 for r in norm_rows),default=0.),
              status='Not a nonzero intervention' if g==0 else 'World1 matched to world0'),
            by_changed_field=subgroup,reference120=reference_report,
            by_domain={domain:summarize([r for r in records if r['domain']==domain])
                       for domain in ['candy','block']},
            unrelated_abs_gap_quantiles={c:[float(v) for v in np.quantile(
                [r['unrelated_abs_gap'] for r in records if r['condition']==c],[.5,.9,.95])]
                for c in summary},
            budget64_truncated=sum(r['truncated64'] for r in rows))
        print('ANALYZED',pair,flush=True)
    for family in [primary,secondary]:
        adjusted=holm([x['p_for_family'] for x in family])
        for r,p in zip(family,adjusted):
            r['holm_p']=p;r['pass']=bool(r['eligible'] and r['mean']>=r['minimum'] and r['lo']>0 and p<.05)
    report=dict(study='COUNTERFACTUAL0917',status='COMPLETE',results=results,
        primary_family=primary,single_layer_secondary_family=secondary,raw_sha256=pins,
        statistical_scope='Fixed vocabulary,30 held-out material-combination families; repeated worlds/questions/templates clustered.3 bootstrap seeds are Monte Carlo stability only. Small or degenerate intervals do not imply population equivalence.')
    dest=root/'confirmation_analysis.json'
    if dest.exists():assert json.loads(dest.read_text())==report
    else:dest.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('Primary passes',sum(x['pass'] for x in primary),'of12; single passes',sum(x['pass'] for x in secondary),'of8')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--shards',type=int,default=8)
    a=ap.parse_args();main(a.root,a.manifest,a.shards)
