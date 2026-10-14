"""Stack the per-word R0 fits into one directions tensor per receiver, and compute the R2 generic local scale.

directions[l, address] = the frozen fitter's mean of the twelve generic contrasts at layer l (read straight from the
per-word blobs, never recomputed), and

    c[l] = median over admitted (word, template) of ||delta[l,w,t]|| / ||d[l,w]||     (epsilon = 1e-8)

using every inventory row the existing fitting rule admits (it admits all of them; the mean is unconditional), so the
scale never sees a task word, an answer or a receiver outcome. Rows whose mean direction norm is below epsilon are
reported and excluded from the median, not chosen by behaviour.
"""
import argparse, hashlib, json
from pathlib import Path
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
R = ROOT


def rooted(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
EPS = 1e-8


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', type=Path, required=True); ap.add_argument('--receiver', required=True)
    a = ap.parse_args(); m = json.loads(a.manifest.read_text())
    words = json.loads(rooted(m['words_file']).read_text()); layers = m['layers'][a.receiver]
    src = rooted(m['root']) / 'fits' / a.receiver
    D = None; ratios = [[] for _ in layers]; excluded = []
    for addr, word in enumerate(words):
        blob = torch.load(src / f'w{addr:03d}.pt', map_location='cpu', weights_only=False)
        assert blob['word'] == word and blob['address'] == addr and blob['layers'] == layers
        d = blob['direction']                      # [L, hidden]
        if D is None:
            D = torch.zeros(len(layers), len(words), d.shape[1])
        D[:, addr] = d
        per = blob['contrasts'].norm(dim=2)        # [12, L]
        base_norm = d.norm(dim=1)                  # [L]
        for li in range(len(layers)):
            if float(base_norm[li]) < EPS:
                excluded.append(dict(word=word, layer=layers[li], direction_norm=float(base_norm[li]))); continue
            ratios[li] += [float(x) / float(base_norm[li]) for x in per[:, li]]
    scale = {str(l): float(torch.tensor(ratios[li]).median()) for li, l in enumerate(layers)}
    out = rooted(m['root']) / 'fits' / f'directions_{a.receiver}.pt'
    torch.save(dict(directions=D, words=words, layers=layers, receiver=a.receiver,
                    source=str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src), n_words=len(words),
                    construction='per-word means from the R0 multi-layer fit (same recipe as word_memory_fit_v1)'), out)
    meta = dict(receiver=a.receiver, layers=layers, file=str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out), sha256=sha(out),
                direction_norm_median_per_layer={str(l): float(D[li].norm(dim=1).median()) for li, l in enumerate(layers)},
                local_scale=scale, n_ratio_samples={str(l): len(ratios[li]) for li, l in enumerate(layers)},
                excluded_rows=excluded, epsilon=EPS,
                rule='c[l] = median over (word, template) of ||delta[l,w,t]|| / ||d[l,w]||; >= 1 by construction')
    (rooted(m['root']) / 'fits' / f'scale_{a.receiver}.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(a.receiver, 'directions', sha(out)[:12], 'scale', {k: round(v, 3) for k, v in scale.items()},
          'dir-norm median', {k: round(v, 1) for k, v in meta['direction_norm_median_per_layer'].items()}, 'excluded', len(excluded))


if __name__ == '__main__':
    main()
