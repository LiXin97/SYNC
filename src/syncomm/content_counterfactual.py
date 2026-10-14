"""Explicit-fact counterfactuals; pure data/score logic, with no model access.

The independent material family is an unordered pair of colors and an unordered
pair of tastes/materials. Aliases, templates, worlds and questions are repeats
inside that family, never additional independent semantic samples.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re

COLORS = ('green', 'orange', 'red', 'yellow')
TASTES = ('bitter', 'salty', 'sour', 'sweet')
MATERIALS = ('metallic', 'wooden')
DEV_ALIASES = ('LORP', 'MIV', 'ZANE', 'KRUL', 'FEP', 'VOTH', 'BLIX', 'QUAN')
EVAL_ALIASES = ('DRAM', 'NEXO', 'TIRP', 'GLAV', 'MOSK', 'YERN', 'PLUX', 'SKAR',
                'VANT', 'TOLM', 'DALP', 'EMOR')
FRAMES = (
    '{a} is a {color} {kind}. It is {other}.',
    'Private record: {a} has a {color} color and is {other}. It is a {kind}.',
    'The {kind} named {a} is {other}; its color is {color}.',
    'For {a}, the recorded color is {color}. This {kind} is {other}.',
    '{a}: a {color} {kind} that is {other}.',
)
FEWSHOT = ('Answer with exactly one object name on one line.\n'
    'DEMOX is large. DEMOY is small.\nQ: Which object is large?\nA: DEMOX\n\n'
    'DEMOX is smooth. DEMOY is rough.\nQ: Which object is rough?\nA: DEMOY\n\n')


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def families():
    out = []
    for domain, values in [('candy', TASTES), ('block', MATERIALS)]:
        group = []
        for colors, others in itertools.product(itertools.combinations(COLORS, 2),
                                                itertools.combinations(values, 2)):
            rec = dict(domain=domain, colors=list(colors), others=list(others))
            rec['family'] = digest(rec)[:16]
            group.append(rec)
        group.sort(key=lambda x: digest(['split20260917', x['family']]))
        for i, rec in enumerate(group):
            rec['split'] = 'dev' if i < (10 if domain == 'candy' else 2) else 'eval'
            out.append(rec)
    return sorted(out, key=lambda x: x['family'])


def question(field, value):
    if field == 'color':
        return f'Which object has a {value} color?'
    return f'Which object is {value}?'


def build_bank():
    scenes = []
    for f in families():
        templates = (0, 1) if f['split'] == 'dev' else (2, 3, 4)
        aliases = DEV_ALIASES if f['split'] == 'dev' else EVAL_ALIASES
        # Eight orientation/changed-field combinations in each confirmation
        # template. Dev uses four per template (eight per family).
        orientations = list(itertools.product((0, 1), repeat=3))
        for ti, template in enumerate(templates):
            chosen = orientations if f['split'] == 'eval' else orientations[ti::2]
            for changed, flip_color, flip_other in chosen:
                key = [f['family'], template, changed, flip_color, flip_other]
                scene_id = digest(key)[:20]
                first = int(scene_id[:4], 16) % len(aliases)
                second = (first + 1 + int(scene_id[4:8], 16) % (len(aliases) - 1)) % len(aliases)
                names = [aliases[first], aliases[second]]
                fields = ['color', 'taste' if f['domain'] == 'candy' else 'material']
                values = [f['colors'], f['others']]
                original = [dict(zip(fields, [values[0][i ^ flip_color],
                                              values[1][i ^ flip_other]])) for i in (0, 1)]
                target = fields[changed]
                unrelated = fields[1 - changed]
                worlds = []
                queries = [question(target, values[changed][0]),
                           question(unrelated, values[1 - changed][0])]
                for world in (0, 1):
                    facts = [dict(x) for x in original]
                    if world:
                        facts[0][target], facts[1][target] = facts[1][target], facts[0][target]
                    evidence = [FRAMES[template].format(a=names[i], color=x['color'],
                                kind=f['domain'], other=x[fields[1]]) for i, x in enumerate(facts)]
                    answers = [names[next(i for i, x in enumerate(facts) if x[field] == val)]
                               for field, val in [(target, values[changed][0]),
                                                  (unrelated, values[1 - changed][0])]]
                    worlds.append(dict(world=world, facts=facts, evidence=evidence, answers=answers))
                scenes.append(dict(scene=scene_id, family=f['family'], split=f['split'],
                    domain=f['domain'], template=template, aliases=names, fields=fields,
                    changed_field=target, queried_values=[values[changed][0], values[1-changed][0]],
                    questions=queries, worlds=worlds))
    validate_bank(scenes)
    return scenes


def validate_bank(scenes):
    assert len({s['scene'] for s in scenes}) == len(scenes)
    split_families = {sp: {s['family'] for s in scenes if s['split'] == sp} for sp in ['dev', 'eval']}
    assert not split_families['dev'] & split_families['eval']
    templates = {sp: {s['template'] for s in scenes if s['split'] == sp} for sp in ['dev', 'eval']}
    assert not templates['dev'] & templates['eval']
    for s in scenes:
        w0, w1 = s['worlds']
        assert w0['answers'][0] != w1['answers'][0]
        assert w0['answers'][1] == w1['answers'][1]
        assert len(set(s['aliases'])) == 2
        for i in (0, 1):
            changed = [k for k in s['fields'] if w0['facts'][i][k] != w1['facts'][i][k]]
            assert changed == [s['changed_field']]
            for w in s['worlds']:
                assert s['aliases'][i] in w['evidence'][i]
                assert 'Which' not in w['evidence'][i] and '?' not in w['evidence'][i]
        for q in range(2):
            # No world-specific content can enter the no-message/latent prompt.
            public = {k: v for k, v in s.items() if k != 'worlds'}
            assert receiver_body(s, q) == receiver_body(public, q)
    return {'scenes': {sp: sum(s['split'] == sp for s in scenes) for sp in split_families},
            'families': {sp: len(v) for sp, v in split_families.items()},
            'template_families': {sp: sorted(v) for sp, v in templates.items()},
            'lexical_concepts_shared_across_splits': True,
            'worlds_and_questions_are_repeated_measures': True}


def receiver_body(scene, qi, extra=None):
    carriers = [f'{alias} is an object.' for alias in scene['aliases']]
    text = FEWSHOT + ' '.join(carriers) + '\n'
    if extra:
        text += extra + '\n'
    text += 'Q: ' + scene['questions'][qi] + '\nA:'
    return text, carriers


def score_answer(text, aliases, gold):
    # Generation stops after the first nonempty line. Do not reward an alias
    # embedded in an explanation, list, or repeated question.
    line = next((x.strip() for x in text.splitlines() if x.strip()), '')
    clean = line.strip(' \t\r.,:;!?\"\'`').upper()
    valid = clean in aliases
    return {'answered': valid, 'correct': valid and clean == gold,
            'answer': clean if valid else None}


def score_orientation(scene, gap_world0, gap_world1, query=0):
    sign = 1 if scene['worlds'][0]['answers'][query] == scene['aliases'][0] else -1
    return sign * (gap_world0 - gap_world1)


def scene_behavior(scene, answers):
    """answers[(world, question)] contains a parsed alias or None."""
    ok = {(w, q): answers[w, q] == scene['worlds'][w]['answers'][q]
          for w in (0, 1) for q in (0, 1)}
    return {'accuracy': sum(ok.values()) / 4,
            'relevant_both': float(ok[0, 0] and ok[1, 0]),
            'unrelated_both': float(ok[0, 1] and ok[1, 1]),
            'all_four': float(all(ok.values())),
            'unrelated_consistent': float(answers[0, 1] is not None and
                                         answers[0, 1] == answers[1, 1])}
