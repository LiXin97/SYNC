"""Run one FROZEN recovery evaluator unchanged, with src/syncomm/latest/adapter.py installed first.

Usage (GPU node, absolute paths, the model's declared interpreter -- see adapter.MODELS[key]['python']):
  PYTHONPATH=$R/src $PY $R/scripts/latest0924/run_patched.py <module> <the module's own args...>

  <module> is syncomm.recovery.evaluate_depth | syncomm.recovery.evaluate_confirm | syncomm.recovery.fit_layers
  (short names evaluate_depth / evaluate_confirm / fit_layers are accepted). Examples:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R/src $PY $R/scripts/latest0924/run_patched.py fit_layers \\
      --manifest $R/artifacts/latest0924/fit_manifest.json --receiver gpt-oss-20b --shard 0 --shards 1
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$R/src $PY $R/scripts/latest0924/run_patched.py evaluate_confirm \\
      --manifest <cell>/manifest.json --phase engineering

The evaluator module is imported first (its frozen `prior` only exists as an attribute of the imported module), then
adapter.install() patches base/prior, then the module's main() runs with the remaining argv. Nothing else changes: the
manifest pins, arms, checks and outputs are the evaluator's own. A receiver path that is not in adapter.MODELS is
loaded by the frozen loader.
"""
import importlib
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'README.md').exists())
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

SHORT = {'evaluate_depth': 'syncomm.recovery.evaluate_depth', 'evaluate_confirm': 'syncomm.recovery.evaluate_confirm',
         'fit_layers': 'syncomm.recovery.fit_layers'}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        return 0 if len(sys.argv) >= 2 else 2
    name = SHORT.get(sys.argv[1], sys.argv[1])
    if name not in SHORT.values():
        print(f'[run_patched] WARNING: {name} is not one of the three frozen evaluators', flush=True)
    from syncomm.latest import adapter
    module = importlib.import_module(name)
    installed = adapter.install(module)
    assert installed['priors'], f'no frozen prior reachable from {name}'
    print(f'[run_patched] {name}: adapter installed (base + {len(installed["priors"])} prior module(s))', flush=True)
    sys.argv = [name] + sys.argv[2:]
    return module.main()


if __name__ == '__main__':
    sys.exit(main())
