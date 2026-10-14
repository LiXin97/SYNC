"""Experiment F: held-out depth-scan manifests for the frozen evaluator (src/syncomm/recovery/evaluate_depth.py).
Complete facts, 92 development scenes, single fitted write (gamma 1) and native reference at depths 0.2..0.8, plus the
shared none/text/full-text arms. Refuses to run unless predictions.json is committed in git (blindness guard).
Usage: python3 scripts/latest0924/predict/build_depth_manifests.py"""
import hashlib, json, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; P = ROOT / 'artifacts/latest0924/predict'
LOCK = json.loads((ROOT / 'gates/latest0924/expF_predict.lock.json').read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    pred = P / 'predictions.json'
    assert pred.exists(), 'predictions.json missing'
    # blindness guard: the predictions' sha must be recorded in a committed file before any held-out outcome row
    rec = ROOT / 'scripts/latest0924/predict/predictions_record.json'
    assert rec.exists() and json.loads(rec.read_text())['predictions_sha256'] == sha(pred), 'predictions not recorded'
    tracked = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '--error-unmatch', str(rec)], capture_output=True).returncode == 0
    clean = subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', 'HEAD', '--', str(rec)]).returncode == 0
    assert tracked and clean, 'predictions record must be committed and unmodified'
    fm = json.loads((P / 'fit_manifest.json').read_text())
    r1 = json.loads((ROOT / 'artifacts/receiver_recovery/r1/qwen3-14b-base/manifest.json').read_text())
    for k in LOCK['heldout_receivers']:
        root = P / 'depth' / k; path = root / 'manifest.json'
        if path.exists():
            print('EXISTS', path); continue
        layers = fm['layers'][k]; directions = P / 'fits' / f'directions_{k}.pt'
        scale_file = P / 'fits' / f'scale_{k}.json'
        scale = {str(l): 1.0 for l in layers}  # only read by 'local' arms, which this scan does not run
        m = dict(study=f'LATEST0924_EXPF_DEPTH_{k}'.upper(), lock='gates/latest0924/expF_predict.lock.json',
                 purpose=f'Experiment F held-out depth scan on {k}: complete facts, single fitted write (gamma 1) and native reference at depths 0.2..0.8, 92 development scenes. Scored against predictions committed beforehand.',
                 receiver=k, root=str(root), models={k: fm['models'][k]}, base_models=[], n_blocks=fm['n_blocks'][k],
                 all_layers=layers, configs=[[f'd{int(round(d * 100))}', [d], [l]] for d, l in zip(LOCK['depths'], layers)],
                 directions=str(directions), words_file=r1['words_file'], bank=r1['bank'], bank_size=r1['bank_size'],
                 bank_split=r1['bank_split'], engineering_scene_ids=r1['engineering_scene_ids'], gammas=[1],
                 local_scale_labels=[], match_labels=[], scale=scale, numerics=r1['numerics'],
                 row_split='latest0924_expF_heldout', plan='gates/latest0924/expF_predict.lock.json',
                 pins={str(p): sha(p) for p in [r1['bank'], r1['words_file'], directions, ROOT / 'src/syncomm/recovery/evaluate_depth.py',
                                                ROOT / 'gates/latest0924/expF_predict.lock.json', pred]})
        root.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(m, indent=1) + '\n'); print('WROTE', path)


if __name__ == '__main__':
    main()
