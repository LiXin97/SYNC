# SYNC: Share the Symbol, Not the Vector

**Training-Free Fact Communication Across Language Models**

[Xin Li](https://lixin.ai)<sup>1</sup>, [Mengbing Liu](https://liumengbing.com/)<sup>1</sup>, Hao Jiang<sup>1</sup>, [Xin Gao](https://gaoxin492.github.io/)<sup>2</sup>, Annan Wang<sup>1</sup>, Xingwei Qu<sup>3</sup>, Yichi Zhang, [Chau Yuen](https://blogs.ntu.edu.sg/chau-yuen/)<sup>1</sup>

<sup>1</sup>Nanyang Technological University &nbsp; <sup>2</sup>Yale University &nbsp; <sup>3</sup>University of Manchester

[Project page](https://lixin.ai/SYNC/) · [Paper (arXiv:2610.19696)](https://arxiv.org/abs/2610.19696) · [Data (per-row records)](https://huggingface.co/datasets/XINLI1997/SYNC-data) · [Fitted word directions](https://huggingface.co/XINLI1997/SYNC-directions) · [Hugging Face collection](https://huggingface.co/collections/XINLI1997/sync-share-the-symbol-not-the-vector-6acf26edd5d52fb91ae87350)

SYNC (**Sy**mbol-Indexed **N**eural **C**ommunication) lets language models with different architectures pass facts
to each other without fine-tuning or a learned translator. From descriptions only it can see, the sender selects words
from a shared list and sends their indices for each object, without seeing the questions. The receiver writes the
described facts into its hidden states using its own word directions, vectors fitted once from its own activations.
Both models stay frozen.

This repository contains the code, the preregistration locks, the scene banks, every sender message and the analysis
outputs behind the paper's tables and figures. The Python package is `syncomm`.

---

## 1. Installation

Python 3.11 or newer.

```bash
git clone https://github.com/LiXin97/SYNC.git
cd SYNC
pip install -e .            # torch, transformers, huggingface_hub, numpy, scipy, scikit-learn, matplotlib
```

Checking the tables (Section 3) needs only `numpy` and `matplotlib`. Running models needs a GPU. The newer
architectures (Qwen3.5, Qwen3.6, Gemma-4) need a recent `transformers` (we used 5.12 for them and 5.3 for the others).

## 2. Models

Run manifests record checkpoints as `/path/to/models/<Name>`. Every model is loaded through
`syncomm.models.resolve_model`, which maps the name to a Hugging Face ID and downloads it
(`src/syncomm/models.py` lists all IDs). On a machine without internet access, put the checkpoints in one directory
and set

```bash
export SYNC_MODEL_ROOT=/your/models      # contains Qwen3-32B/, gemma-3-27b-it/, ...
```

Llama and Gemma checkpoints are gated on Hugging Face: accept their licenses and set `HF_TOKEN`. The 27-32B
receivers in FP32 need a GPU with more than 128 GB of memory, or several GPUs.

## 3. Check the paper's numbers (CPU only, a few minutes)

```bash
python scripts/build_latest_views.py --check              # Tables 1-3, Figure 6 data and their appendix tables
python scripts/build_receiver_recovery_views.py --check   # Figures 4-5, development-receiver tables
python scripts/build_word_interface_views.py --check      # four-object and single-context/embedding tables
```

Each command regenerates its LaTeX tables from the analysis outputs in `artifacts/` and asserts byte equality with
`paper/tables/`. `artifacts/paper_builds/*_provenance.json` record the SHA-256 of every input and generated table.

### Recount from the per-row records

The per-row records (one row per scene, world, question and condition: parsed answer, correctness, coverage and
generated text; 1.6 million rows, 2.3 GB) are on Hugging Face. Download them into the repository root; they land in
the `artifacts/**/raw/` folders the scripts expect:

```bash
hf download XINLI1997/SYNC-data --repo-type dataset --local-dir .
python scripts/latest0924/layout/recount_layout.py        # recounts all 96 layout cells
python scripts/latest0924/predict/score.py                # recomputes Table 2 from the depth-scan rows
```

`score.py` rewrites `artifacts/latest0924/predict/score.json`: every statistic and per-cell prediction comes out
identical; only the recorded lock digest changes, to the edited lock's digest listed in `SANITIZATION.json`.

The rows were slimmed to fit: token ids, slot tables, per-layer patch reports and most cache/logit hashes were removed.
Every point estimate can be recounted from them; the engineering checks that need the removed fields (identity
writes, cached-versus-complete agreement, edit-norm matching) are recorded in the analysis outputs and audit files.
Countries rows omit message contents.

## 4. Where each result comes from

| Paper item | Analysis output | Produced by |
|---|---|---|
| Table 1, Appendix A (new senders and receivers) | `artifacts/latest0924/expA/analysis/expA_results.json` | `scripts/latest0924/analyze_expA.py` |
| Figures 4-5, Appendices F and J (development receivers, native states) | `artifacts/receiver_recovery/results_package_v2/` | runners in `src/syncomm/recovery/` |
| Table 2, Appendix B (depth prediction) | `artifacts/latest0924/predict/score.json` | `scripts/latest0924/predict/score.py` |
| Table 3 left, Appendix C.1 (new layouts) | `artifacts/latest0924/layout/analysis/layout_results.json` | `scripts/latest0924/layout/analyze_layout.py` |
| Table 3 right, Appendix C.2 (Countries) | `artifacts/latest0924/realtask/analysis/expD_results.json` | `scripts/latest0924/realtask/analyze.py` |
| Figure 6, Appendix D (graded weights) | `artifacts/latest0924/graded/analysis.json` | `scripts/latest0924/analyze_graded.py` |
| Other text formats (Section 4.5, Appendix D) | `artifacts/latest0924/graded_textfmt/analysis_v2.json` | `scripts/latest0924/textfmt/analyze_textfmt_v2.py` |
| Late-writing diagnostics (Appendix G) | `artifacts/latest0924/diag/analysis/` | `scripts/latest0924/analyze_diag.py` |
| Four-object questions, simpler representations (Appendix K) | `artifacts/word_interface_followup/results_package_v1/` | `src/syncomm/wordif/evaluate_localrep.py` and the recovery runners |

## 5. Method code and re-running experiments

| Component | File |
|---|---|
| Fitting receiver-local symbol directions (twelve contexts) | `src/syncomm/recovery/fit_layers.py` (frozen reference: `artifacts/mechanism0920/word_memory_fit_v1/snapshot/`) |
| Writing at a chosen depth, cache reuse, controls, scoring | `src/syncomm/recovery/evaluate_confirm.py`, `src/syncomm/recovery/evaluate_depth.py` |
| Loading newer architectures (chat rendering, cache handling) | `src/syncomm/latest/adapter.py`, entry point `scripts/latest0924/run_patched.py` |
| Prompted senders and the message parser | `src/syncomm/latest/sender.py`, `src/syncomm/recovery/sender_candidate_a4.py`, `scripts/latest0924/sender_gate.py` |
| Graded weights, text formats, layouts | `src/syncomm/latest/evaluate_graded*.py`, `src/syncomm/latest/evaluate_layout.py`, `src/syncomm/latest/layout_render.py` |
| Depth probes | `scripts/latest0924/predict/calibrate.py` (repeat probe), `scripts/latest0924/predict/calibrate_binding.py` (binding probe) |
| Countries task and 275-symbol inventory | `src/syncomm/bench/build_ac.py`, `scripts/latest0924/realtask/build_countries_bank.py`, `make_inventory_variants.py` |
| Model loading | `src/syncomm/models.py` |

Each experiment directory holds the manifests that were run. For example, for one receiver of Table 1
(`<key>` such as `qwen3.5-2b`):

```bash
export PYTHONPATH=$PWD/src
# 1. fit the receiver's word directions (one file per word under artifacts/latest0924/fits/<key>/), then stack them
python scripts/latest0924/run_patched.py fit_layers --manifest artifacts/latest0924/fit_manifest.json --receiver <key> --shard 0 --shards 1
python scripts/recovery_stack_directions.py --manifest artifacts/latest0924/fit_manifest.json --receiver <key>
# 2. evaluate it on a sender's messages (engineering checks first, then the development/confirmation rows)
python scripts/latest0924/run_patched.py evaluate_confirm --manifest artifacts/latest0924/expA/<sender>/<receiver>/manifest.json --phase engineering
python scripts/latest0924/run_patched.py evaluate_confirm --manifest artifacts/latest0924/expA/<sender>/<receiver>/manifest.json --phase development
# 3. (optional) regenerate a sender's messages
python -m syncomm.latest.sender --sender gpt-oss-20b --bank artifacts/receiver_recovery/r3_bank/bank.json --split r3_confirmation --out <packets.json>
```

The fitted directions used in the paper can be downloaded instead of step 1 (see Section 7). Run all commands from
the repository root. Runners verify the SHA-256 of every pinned input before they start; the pins of the direction
files match the published tensors, so a refit that is not bit-identical (another GPU or library version) must update
the `pins` entry of its manifest. Each model declares the `transformers` environment it ran in
(`src/syncomm/latest/adapter.py`): set `SYNC_PYTHON` to the interpreter with transformers 5.3 and `SYNC_PYTHON_TF512`
to one with transformers 5.12 (Qwen3.5, Qwen3.6, Gemma-4); both default to the current interpreter.
All runs used FP32, eager attention, batch size one and greedy decoding.

### Data generation

Every scene bank is generated mechanically from fixed word lists and a seeded random generator; no third-party text is
used. The banks that were run are included (so the generators need not be re-run):

- `artifacts/receiver_recovery/r3_bank/` (572 confirmation scenes), `artifacts/mechanism0919/discovery_v1/exposed_development.json` (development scenes)
- `artifacts/latest0924/graded/bank.json` (144-scene subset), `artifacts/latest0924/layout/` (new layouts)
- `artifacts/word_interface_followup/p0b_banks/` (four-object bank)
- inventory: `artifacts/receiver_recovery/words_221.json`

The generators are `scripts/latest0924/build_graded_bank.py`, `scripts/latest0924/layout/build_layout_bank.py`,
`scripts/latest0924/realtask/build_countries_bank.py` and `scripts/p0b_build_banks.py`. The Countries task follows
the protocol of Ramesh & Li (2025) and is rebuilt by `python -m syncomm.bench.build_ac --tasks countries --splits dev`
(writes `data/bench/`, which `build_countries_bank.py` reads).

Sender prompts and every message are in `artifacts/latest0924/sender_prompts/`, `sender_packets/`, `senders/`,
`sender_gate/` and `artifacts/receiver_recovery/r3_packets/`.

## 6. Preregistration

Every experiment was fixed in a lock file before any outcome row existed:

- `gates/latest0924/*.lock.json`: question, receivers, arms, endpoints, decision rules, bootstrap settings, UTC time,
  and SHA-256 pins of the scripts, banks and manifests used. Amendments are listed inside each lock with their time and
  the condition under which they were made.
- `artifacts/latest0924/*/DESIGN.md`: the corresponding protocol text.
- `artifacts/receiver_recovery/locks/`, `artifacts/word_interface_followup/locks/`: locks of the development studies.
- `artifacts/latest0924/predict/predictions.json` and `scripts/latest0924/predict/predictions_record.json` hold the
  depth predictions for the nine held-out receivers, fixed before their depth scans.

`SANITIZATION.json` lists every file edited after the analyses ran (removal of machine-specific paths and host names,
and the package rename for this release), with the SHA-256 recorded when the analyses ran and the SHA-256 of the file
shipped here. Digests inside the records were updated accordingly, so the checks in Section 3 pass on this tree.

## 7. What is not included

- Model weights (downloaded from their publishers, see Section 2).
- Fitted word directions: regenerated by `fit_layers` and `scripts/recovery_stack_directions.py` (Section 5); the
  tensors used in the paper are on Hugging Face at
  [XINLI1997/SYNC-directions](https://huggingface.co/XINLI1997/SYNC-directions), each under its base model's license.
- Code of the earlier sparse-extraction studies summarized in the later appendices.

## 8. License

Code: Apache License 2.0 (`LICENSE`). The models used in the experiments are not part of this repository and keep
their own licenses (`NOTICE`, `licenses/`). The per-row records on Hugging Face are released under CC BY 4.0; the
fitted word directions are derived from the base models and follow each base model's license.

## 9. Citation

```bibtex
@misc{li2026sync,
  title         = {Share the Symbol, Not the Vector: Training-Free Fact Communication Across Language Models},
  author        = {Xin Li and Mengbing Liu and Hao Jiang and Xin Gao and Annan Wang and Xingwei Qu and Yichi Zhang and Chau Yuen},
  year          = {2026},
  eprint        = {2610.19696},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.19696}
}
```
