"""Experiment F: fit the locked threshold rule on the development cells and write held-out predictions.

Reads only calibration outputs (artifacts/latest0924/predict/calibration/*.json) and the development depth profiles
(artifacts/latest0924/predict/dev_depth_profiles.json, from the R1 depth scan). No held-out outcome exists or is read.
Writes artifacts/latest0924/predict/predictions.json. Usage: python3 scripts/latest0924/predict/make_predictions.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / 'artifacts/latest0924/predict'
LOCK = json.loads((ROOT / 'gates/latest0924/expF_predict.lock.json').read_text())
DEPTHS = LOCK['depths']


def fit_threshold(values, labels):
    """Maximize balanced accuracy over midpoints between consecutive sorted values; ties -> midpoint of tied range."""
    xs = sorted(set(values))
    cands = [(xs[i] + xs[i + 1]) / 2 for i in range(len(xs) - 1)] + [xs[0] - 1e-9, xs[-1] + 1e-9]
    def ba(t):
        tp = sum(1 for v, y in zip(values, labels) if y and v >= t); fn = sum(1 for v, y in zip(values, labels) if y and v < t)
        tn = sum(1 for v, y in zip(values, labels) if not y and v < t); fp = sum(1 for v, y in zip(values, labels) if not y and v >= t)
        return 0.5 * (tp / (tp + fn) + tn / (tn + fp))
    scores = [(ba(t), t) for t in cands]
    best = max(s for s, _ in scores)
    tied = sorted(t for s, t in scores if s == best)
    return (tied[0] + tied[-1]) / 2, best, tied


def main():
    dev = json.loads((P / 'dev_depth_profiles.json').read_text())
    amend = LOCK['amendments'][-1]
    metric = amend['primary_metric']
    cal = {k: json.loads((P / amend['calibration_dir'] / f'{k}.json').read_text()) for k in LOCK['development_receivers'] + LOCK['heldout_receivers']}
    readout = {k: json.loads((P / 'calibration' / f'{k}.json').read_text()) for k in cal if (P / 'calibration' / f'{k}.json').exists()}
    dv, dl, dd = [], [], []
    for k in LOCK['development_receivers']:
        for d in DEPTHS:
            dv.append(cal[k]['per_depth'][str(d)][metric]); dl.append(dev[k][f'd{int(round(d * 100))}'] >= 50.0); dd.append(d)
    tau, ba_dev, tied = fit_threshold(dv, dl)
    # depth-only comparator: usable iff depth <= d_star (fit on negated depth so the same ">= t" rule applies)
    t_neg, ba_depth, _ = fit_threshold([-d for d in dd], dl)
    d_star = -t_neg
    preds = {}
    for k in LOCK['heldout_receivers']:
        preds[k] = {str(d): dict(metric=cal[k]['per_depth'][str(d)][metric],
                                 bind_answer_accuracy=cal[k]['per_depth'][str(d)]['bind_answer_accuracy'],
                                 readout_auc=(readout[k]['per_depth'][str(d)]['auc_mean'] if k in readout else None),
                                 geometry_ratio_mean=(readout[k]['per_depth'][str(d)]['geometry_ratio_mean'] if k in readout else None),
                                 predicted_usable=cal[k]['per_depth'][str(d)][metric] >= tau,
                                 depth_only_usable=d <= d_star) for d in DEPTHS}
    out = dict(lock_sha256=hashlib.sha256((ROOT / 'gates/latest0924/expF_predict.lock.json').read_bytes()).hexdigest(),
               metric=metric, tau=tau, dev_balanced_accuracy=ba_dev, tied_thresholds=tied, d_star=d_star,
               dev_balanced_accuracy_depth_only=ba_depth,
               development_cells=[dict(receiver=k, depth=d, metric=cal[k]['per_depth'][str(d)][metric], J=dev[k][f'd{int(round(d * 100))}'])
                                  for k in LOCK['development_receivers'] for d in DEPTHS],
               calibration_sha256={k: hashlib.sha256((P / amend['calibration_dir'] / f'{k}.json').read_bytes()).hexdigest() for k in cal},
               predictions=preds)
    (P / 'predictions.json').write_text(json.dumps(out, indent=1) + '\n')
    print('tau', round(tau, 4), 'dev BA', round(ba_dev, 3), '| depth-only d*', d_star, 'dev BA', round(ba_depth, 3))
    for k in LOCK['heldout_receivers']:
        print(f'{k:24s}', ' '.join(f"{d}:{'U' if preds[k][str(d)]['predicted_usable'] else '.'}({preds[k][str(d)]['metric']:.3f})" for d in DEPTHS))


if __name__ == '__main__':
    main()
