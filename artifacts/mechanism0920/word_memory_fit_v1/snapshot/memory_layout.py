"""Public word-memory layout and packet-to-position mapping; no model or gold.

Implementation preparation for the candidate writer. Not a launched experiment
or a statement that the candidate has passed an engineering/behavioral gate.
"""
from __future__ import annotations

WIDTH = 4
NEUTRAL_WORD = 'unknown'


def render(aliases, object_words=None, width=WIDTH):
    """Return text and exact character spans, with four public slots/object.

    A no-message call depends only on aliases. Actual packet lengths/weights
    cannot silently enter that prefix. Multi-token words remain single entries;
    the tokenizer, not this renderer, determines their token spans.
    """
    assert width in (2,4)
    assert len(aliases) == 2 and len(set(aliases)) == 2
    if object_words is None:
        object_words = [[], []]
    assert len(object_words) == 2
    text, spans = 'Selected words and exact relative weights (integers divided by 255): ', []
    for obj, (alias, words) in enumerate(zip(aliases, object_words)):
        assert len(words) <= width
        assert isinstance(alias, str) and alias and '\n' not in alias
        if obj:
            text += '; '
        text += alias + ': '
        for index in range(width):
            if index:
                text += ', '
            word = words[index] if index < len(words) else NEUTRAL_WORD
            assert isinstance(word, str) and word and '\n' not in word
            start = len(text)
            text += word
            spans.append(dict(object=obj, index=index, word=word, start=start, end=len(text)))
            text += ' [255/255]'
    text += '.'
    return text, spans


def packet_entries(packet, words):
    """Map the unchanged packet to ordered public sites; preserve empty lanes."""
    assert packet.qbits == 8
    entries = []
    assert len({lane.slot_id for lane in packet.lanes}) == len(packet.lanes)
    for lane in packet.lanes:
        assert lane.slot_id in (0,1)
        assert len(lane.concept_ids) == len(lane.q) <= WIDTH
        assert len(set(lane.concept_ids)) == len(lane.concept_ids)
        maximum = max(lane.q, default=0)
        for rank,(address,weight) in enumerate(zip(lane.concept_ids,lane.q)):
            assert 0 <= address < len(words) and 0 <= weight <= 255
            entries.append(dict(object=lane.slot_id,index=rank,address=address,word=words[address],
                                weight=weight,relative_weight=weight/maximum if maximum else 0.))
    return entries


def words_by_object(entries):
    result=[[],[]]
    for obj in (0,1):
        selected=sorted((entry for entry in entries if entry['object']==obj),key=lambda e:e['index'])
        assert [entry['index'] for entry in selected] == list(range(len(selected)))
        result[obj]=[entry['word'] for entry in selected]
    return result


def token_spans(offsets, char_spans, message_start):
    result=[]
    for span in char_spans:
        start=message_start+span['start'];end=message_start+span['end']
        positions=[i for i,(a,b) in enumerate(offsets) if a<end and b>start]
        assert positions and positions == list(range(positions[0],positions[-1]+1))
        result.append(dict(**span,positions=positions,last_position=positions[-1]))
    occupied=[p for s in result for p in s['positions']]
    assert len(set(occupied))==len(occupied), 'A token crossed memory-entry boundaries'
    return result
