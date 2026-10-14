"""Experiment E (latest0924): message layouts for the frozen-setting transfer test. No model, no gold.

Every renderer returns (text, spans) with one span per (object, slot index) in the frozen convention of
artifacts/mechanism0920/word_memory_fit_v1/snapshot/memory_layout.py: dict(object, index, word, start, end). Unused slots
hold the neutral word `unknown`. Weights are never printed differently from the frozen layout's constant 255 because
Experiment E sends complete facts only (all weights 255).

  L0  standard: the frozen layout, generalized to n objects (byte-identical to memory_layout.render for two objects)
  L1  slot permutation: the standard template, but each object's two facts sit in slots 3 and 2 (reversed order) and
      slots 0 and 1 hold `unknown`; the neutral prefix is identical to L0
  L2  new template: new header wording, separators and weight annotation, and the objects listed in reverse order
  L3  new object count: the standard template with three objects (the third is a distractor with distinct values)
"""
from __future__ import annotations

WIDTH = 4
NEUTRAL = 'unknown'
HEADER_STANDARD = 'Selected words and exact relative weights (integers divided by 255): '
HEADER_L2 = 'Received concepts, each followed by its strength out of 255 | '
CONDITIONS = ('L0', 'L1', 'L2', 'L3')


def _check(aliases, object_words, width):
    assert width == WIDTH and len(aliases) == len(object_words) and len(set(aliases)) == len(aliases)
    for alias, words in zip(aliases, object_words):
        assert isinstance(alias, str) and alias and '\n' not in alias
        assert len(words) <= width and all(isinstance(w, str) and w and '\n' not in w for w in words)


def render_standard(aliases, object_words=None, width=WIDTH):
    object_words = object_words if object_words is not None else [[] for _ in aliases]
    _check(aliases, object_words, width)
    text, spans = HEADER_STANDARD, []
    for obj, (alias, words) in enumerate(zip(aliases, object_words)):
        if obj:
            text += '; '
        text += alias + ': '
        for index in range(width):
            if index:
                text += ', '
            word = words[index] if index < len(words) else NEUTRAL
            start = len(text)
            text += word
            spans.append(dict(object=obj, index=index, word=word, start=start, end=len(text)))
            text += ' [255/255]'
    text += '.'
    return text, spans


def render_l2(aliases, object_words=None, width=WIDTH):
    object_words = object_words if object_words is not None else [[] for _ in aliases]
    _check(aliases, object_words, width)
    text, spans = HEADER_L2, []
    for k, obj in enumerate(reversed(range(len(aliases)))):
        if k:
            text += ' | '
        text += aliases[obj] + ' -> '
        words = object_words[obj]
        for index in range(width):
            if index:
                text += ' ; '
            word = words[index] if index < len(words) else NEUTRAL
            start = len(text)
            text += word
            spans.append(dict(object=obj, index=index, word=word, start=start, end=len(text)))
            text += ' (255)'
    text += ' |'
    return text, spans


RENDER = {'L0': render_standard, 'L1': render_standard, 'L2': render_l2, 'L3': render_standard}


def fact_words(world, words_inventory):
    """Each object's complete facts, in ascending inventory index (the frozen literal-packet order)."""
    out = []
    for facts in world['facts']:
        ws = sorted(set(facts.values()), key=words_inventory.index)
        assert len(ws) == len(facts) == 2
        out.append(ws)
    return out


def condition_slots(condition, words_by_object):
    """-> (object_words for the renderer, entries [(object, slot index, word)]) for one world."""
    if condition == 'L1':
        object_words = [[NEUTRAL, NEUTRAL, ws[1], ws[0]] for ws in words_by_object]
        entries = [(o, 3, ws[0]) for o, ws in enumerate(words_by_object)] + [(o, 2, ws[1]) for o, ws in enumerate(words_by_object)]
    else:
        object_words = [list(ws) for ws in words_by_object]
        entries = [(o, i, w) for o, ws in enumerate(words_by_object) for i, w in enumerate(ws)]
    return object_words, sorted(entries)
