"""Pure tensor construction for a separately evaluated receiver-local writer."""
import torch


def targets(neutral_states, layers, directions, entries, slots, gamma, permutation=None):
    assert gamma >= 0 and directions.ndim == 3 and directions.shape[0] == len(layers)
    assert set(layers) == set(neutral_states)
    mapping={(s['object'],s['index']):s['last_position'] for s in slots}
    assert len(mapping)==len(slots)
    assert all(len(s['positions'])==1 for s in slots), 'Deployment placeholders must be one token'
    active=[e for e in entries if e['weight']>0]
    positions=sorted(mapping[e['object'],e['index']] for e in active)
    assert len(set(positions))==len(positions)
    result={layer:h.clone() for layer,h in neutral_states.items()}
    for li,layer in enumerate(layers):
        for e in active:
            address=e['address'];direction=directions[li,address]
            if permutation is not None:
                reference=direction.norm()
                direction=directions[li,int(permutation[address])]
                norm=direction.norm()
                assert norm>0 or reference==0
                direction=direction*(reference/norm) if norm>0 else torch.zeros_like(direction)
            pos=mapping[e['object'],e['index']]
            result[layer][0,pos] += gamma*e['relative_weight']*direction.to(result[layer])
    return result,positions
