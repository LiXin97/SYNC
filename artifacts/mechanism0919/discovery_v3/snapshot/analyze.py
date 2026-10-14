"""Recompute exposed-development behavior from raw responses; no test verdict."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
TASK_PATH = ROOT / 'artifacts/counterfactual0917/v2/snapshot/content_counterfactual.py'
spec = importlib.util.spec_from_file_location('locked_task', TASK_PATH)
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(m):
    root = Path(m['root'])
    bank = {s['scene']: s for s in json.loads(Path(m['bank']).read_text())}
    all_results = {}
    for pair in (0, 1):
        groups, seen, source_files, prefix_states, public_checks = {}, set(), [], {}, {}
        for shard in range(4):
            raw = root / 'raw' / f'development_p{pair}_s{shard}.jsonl'
            marker = json.loads(raw.with_suffix('.done.json').read_text())
            assert marker['status'] == 'PASS' and marker['raw_sha256'] == sha(raw)
            count = 0
            for line in raw.open():
                r = json.loads(line)
                assert r['key'] not in seen
                seen.add(r['key']); count += 1
                s = bank[r['scene']]
                assert r['gold'] == s['worlds'][r['world']]['answers'][r['question']]
                for budget in (64, 256):
                    score = task.score_answer(r[f'text{budget}'], s['aliases'], r['gold'])
                    assert score == r[f'score{budget}']
                    key = (r['arm'], r['gamma'], budget)
                    cell = groups.setdefault(key, {}).setdefault(r['scene'], {})
                    cell[r['world'], r['question']] = score
                assert r['prefix_cache_sha256'] == r['branch_initial_cache_sha256']
                prefix_key = (r['scene'], r['world'], r['arm'], r['gamma'])
                prefix_value = (r['prefix_token_sha256'], r['prefix_cache_sha256'])
                assert prefix_states.setdefault(prefix_key, prefix_value) == prefix_value
                if r['arm'] in ('none', 'zero'):
                    public_key = (r['scene'], r['question'])
                    public_value = (r['prompt_sha256'], r['prefix_cache_sha256'], r['logits_sha256'], tuple(r['tokens']))
                    assert public_checks.setdefault(public_key, public_value) == public_value
            assert count == marker['rows'] == 1440
            source_files.append(dict(path=str(raw), sha256=sha(raw), rows=count))
        summaries = []
        for (arm, gamma, budget), cells in sorted(groups.items()):
            assert len(cells) == 96
            by_family = {}
            for scene, records in cells.items():
                assert set(records) == {(w, q) for w in (0, 1) for q in (0, 1)}
                s = bank[scene]
                answers = {key: v['answer'] for key, v in records.items()}
                row = task.scene_behavior(s, answers)
                row['coverage'] = sum(v['answered'] for v in records.values()) / 4
                by_family.setdefault(s['family'], []).append(row)
            metrics = {}
            for name in ('accuracy', 'relevant_both', 'unrelated_both', 'all_four', 'coverage'):
                x = np.array([np.mean([r[name] for r in rows]) for _, rows in sorted(by_family.items())])
                intervals = []
                for seed in m['analysis']['bootstrap_seeds']:
                    rng = np.random.default_rng(seed)
                    samples = x[rng.integers(0, len(x), (20000, len(x)))].mean(1)
                    intervals.append(np.quantile(samples, [.025, .975]).tolist())
                metrics[name] = dict(mean=float(x.mean()), lo=min(v[0] for v in intervals),
                                     hi=max(v[1] for v in intervals), families=len(x))
            summaries.append(dict(arm=arm, gamma=gamma, budget=budget, scenes=96, **metrics))
        controls = {r['arm']: r for r in summaries if r['budget'] == 64 and r['gamma'] == 0}
        instruments = {arm: controls[arm]['all_four']['mean'] >= m['instrument_joint_min']
                       for arm in ('literal_text', 'full_text')}
        selection = {}
        for source in ('actual', 'literal'):
            eligible = [r for r in summaries if r['budget'] == 64 and
                        (r['arm'] == f'{source}_write' or r['arm'] == 'none') and r['coverage']['mean'] >= .90]
            if eligible:
                best = max(eligible, key=lambda r: (r['all_four']['mean'], r['accuracy']['mean'], -r['gamma']))
                selection[source] = dict(arm=best['arm'], gamma=best['gamma'],
                                         joint=best['all_four']['mean'], accuracy=best['accuracy']['mean'])
            else:
                selection[source] = dict(eligible=False, reason='No candidate reaches 90% coverage')
        all_results[str(pair)] = dict(rows=len(seen), source_files=source_files,
                                     instrument_descriptive_prerequisites=instruments,
                                     development_selection=selection, conditions=summaries)
    return dict(status='COMPLETE_RAW_RECOMPUTED_DEVELOPMENT_ONLY',
                interpretation='Previously exposed 12 development families; descriptive intervals and selection, not new confirmation. Text and cached native state are one reference, not independent methods.',
                bootstrap='Equal material-family weighting; union of three Monte Carlo seed intervals. Seeds are not independent model replications.',
                pairs=all_results)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--manifest', type=Path, required=True); a = p.parse_args()
    m = json.loads(a.manifest.read_text())
    for path, expected in m['pins'].items():
        assert sha(Path(path)) == expected, path
    result = summarize(m)
    dest = Path(m['root']) / 'development_analysis.json'
    dest.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=result['status'], rows=sum(p['rows'] for p in result['pairs'].values()))), flush=True)


if __name__ == '__main__':
    main()
