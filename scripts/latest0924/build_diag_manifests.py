"""Write the D1/D2 manifests (gates/latest0924/diag_protocol.lock.json) from the R1 depth-scan manifests; never overwrite."""
import hashlib, json
from pathlib import Path
R = Path('.')
LOCK = R / 'gates/latest0924/diag_protocol.lock.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock = json.loads(LOCK.read_text())
for rec, L in lock['layers'].items():
    r1 = json.loads((R / f'artifacts/receiver_recovery/r1/{rec}/manifest.json').read_text())
    n = r1['n_blocks']
    arms = [dict(arm='none'), dict(arm='text'), dict(arm='identity_write', diagnostic='check')]
    for src in ('fitted', 'native'):
        arms.append(dict(arm='write', source=src, layer=L['d20'], diagnostic='D1_A'))
        for k in ('d35', 'd50', 'd65'):
            arms.append(dict(arm='write', source=src, layer=L[k], diagnostic='D1_B'))
            arms.append(dict(arm='early_cut', source=src, cut=L[k], diagnostic='D1_C'))
        arms.append(dict(arm='early_cut', source=src, cut=L['d20'], diagnostic='check_cut_at_write_layer'))
        arms.append(dict(arm='early_cut', source=src, cut=n - 1, diagnostic='check_cut_at_last_layer'))
    d2 = rec in ('glm-4-9b', 'qwen3-14b-base')
    d2_layers = [L['d20'], L['d35'], L['d50']] if d2 else []
    for l in d2_layers:
        arms.append(dict(arm='write', source='matched', layer=l, diagnostic='D2_matched'))
    root = R / 'artifacts/latest0924/diag' / rec
    files = [r1['bank'], r1['words_file'], r1['directions'], str(LOCK), str(R / 'src/syncomm/latest/diagnostics.py'), str(R / 'src/syncomm/recovery/evaluate_depth.py')]
    m = dict(study=f'LATEST0924_DIAG_{rec}'.upper(), lock=str(LOCK.relative_to(R)), receiver=rec, root=str(root),
             models=r1['models'], base_models=r1['base_models'], n_blocks=n, directions=r1['directions'],
             words_file=r1['words_file'], bank=r1['bank'], bank_size=r1['bank_size'], engineering_scene_ids=r1['engineering_scene_ids'],
             row_split='latest0924_diag_dev', layers=L, d2_layers=d2_layers, arms=arms, numerics=r1['numerics'],
             pins={f: sha(f) for f in files})
    root.mkdir(parents=True, exist_ok=True)
    p = root / 'manifest.json'
    if p.exists():
        print('EXISTS', p); continue
    p.write_text(json.dumps(m, indent=1) + '\n'); print('WROTE', p, len(arms), 'arms')
