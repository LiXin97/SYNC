"""Outcome-blind packet matching for the explicit-fact extension (CPU only)."""
from collections import Counter
from functools import reduce
import operator


def decode_q8(payload):
    raw=bytes.fromhex(payload)
    assert raw[0]==1 and raw[5]==8 and reduce(operator.xor,raw,0)==0
    pos=7;lanes={}
    for _ in range(raw[6]):
        slot,k=raw[pos:pos+2];pos+=2
        ids=list(raw[pos:pos+k]);pos+=k
        q=list(raw[pos:pos+k]);pos+=k
        assert slot not in lanes and ids==sorted(set(ids))
        lanes[slot]=dict(ids=ids,q=q)
    assert pos==len(raw)-1 and set(lanes)=={0,1}
    return lanes


def matching(scenes, packets, words, word_types, tolerance=26):
    """Both worlds match bytes/support size/type and sorted q8 profiles.

    Actual receiver write norms are deliberately NOT inferred from packet
    weights. They are checked separately after all controls have run. Drop
    entire cycles if a site-norm requirement fails, preserving message roles.
    """
    from .strict_matching import strict_derangement
    scenes={s['scene']:s for s in scenes}
    decoded={(sid,w):decode_q8(packets[f'{sid}|{w}']['hex']) for sid in scenes for w in (0,1)}
    feasible={k:[] for k in scenes};costs={}
    keys=sorted(scenes)
    for sid in keys:
        s=scenes[sid]
        for did in keys:
            d=scenes[did]
            if s['family']==d['family'] or s['domain']!=d['domain'] or s['changed_field']!=d['changed_field']:
                continue
            differences=[];ok=True
            for w in (0,1):
                if packets[f'{sid}|{w}']['bytes'] != packets[f'{did}|{w}']['bytes']:
                    ok=False;break
                for lane in (0,1):
                    a,b=decoded[sid,w][lane],decoded[did,w][lane]
                    if len(a['ids'])!=len(b['ids']):ok=False;break
                    if Counter(word_types[words[i]] for i in a['ids'])!=Counter(word_types[words[i]] for i in b['ids']):
                        ok=False;break
                    diff=[abs(x-y) for x,y in zip(sorted(a['q']),sorted(b['q']))]
                    if max(diff,default=0)>tolerance:ok=False;break
                    differences.extend(diff)
                if not ok:break
            if ok:
                feasible[sid].append(did);costs[sid,did]=sum(differences)
    assignment,excluded=strict_derangement(keys,feasible,lambda a,b:costs[a,b])
    return dict(assignment=assignment,excluded=excluded,
                candidates={k:len(v) for k,v in feasible.items()},
                rule='Different material family, same domain/changed field; both worlds, each object: identical support counts/types/bytes, sorted q8 differences <=26. Norm eligibility checked from actual writes, whole cycles only.')


def keep_cycles(assignment, eligible):
    assert set(assignment)==set(assignment.values())
    keep=set();seen=set()
    for first in sorted(assignment):
        if first in seen:continue
        cycle=[];current=first
        while current not in seen:
            cycle.append(current);seen.add(current);current=assignment[current]
        assert current==first
        if all(eligible.get(k,False) for k in cycle):keep.update(cycle)
    return {k:assignment[k] for k in sorted(keep)}
