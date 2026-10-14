"""Pure construction checks for the separately versioned discovery study."""


def literal_codes(facts, words):
    """Facts only: never inspect a question, gold answer, or model output."""
    index = {word: i for i, word in enumerate(words)}
    assert len(index) == len(words)
    assert len(facts) == 2
    result = []
    for obj in facts:
        assert set(obj) in ({'color', 'taste'}, {'color', 'material'})
        values = list(obj.values())
        assert len(set(values)) == 2
        missing = sorted(set(values) - set(index))
        if missing:
            raise ValueError(f'Literal attributes absent from inventory: {missing}')
        ids = tuple(sorted(index[word] for word in values))
        result.append((ids, (255, 255)))
    return result


def prefix_length(texts, input_ids, offsets, positions):
    """Split existing full tokenizations; never retokenize across the split."""
    lengths = []
    for text, ids, spans, sites in zip(texts, input_ids, offsets, positions):
        marker = text.rfind('\nQ: ')
        assert marker >= 0, 'Missing final question separator'
        # A carrier's terminal period may share a token with the newline.
        # Include the public, question-independent Q: delimiter while keeping
        # every word of the actual question in the continuation.
        boundary = marker + len('\nQ:')
        # Zero-width special tokens at the start remain in the prefix.
        n = next(i for i, (_, end) in enumerate(spans) if end > boundary)
        assert n > 1 and n < len(ids)
        assert max(p for obj in sites for p in obj) < n
        lengths.append(n)
    assert len(lengths) == 2 and lengths[0] == lengths[1]
    n = lengths[0]
    assert input_ids[0][:n] == input_ids[1][:n], 'Question-dependent prefix tokens'
    assert positions[0] == positions[1], 'Question-dependent write sites'
    return n


def conditions(gammas):
    result = [('none', 0), ('zero', 0), ('full_text', 0),
              ('actual_text', 0), ('literal_text', 0)]
    result += [(f'{source}_write', gamma)
               for source in ('actual', 'literal') for gamma in gammas]
    assert len(result) == len(set(result))
    return result


def row_key(scene, world, question, arm, gamma):
    return '|'.join(map(str, (scene, world, question, arm, gamma)))
