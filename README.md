<div align="center">

# SYNC

### Share the Symbol, Not the Vector: Training-Free Fact Communication Across Language Models

[Xin Li](https://lixin.ai)<sup>1</sup>, [Mengbing Liu](https://liumengbing.com/)<sup>1</sup>, Hao Jiang<sup>1</sup>,
[Xin Gao](https://gaoxin492.github.io/)<sup>2</sup>, Annan Wang<sup>1</sup>, Xingwei Qu<sup>3</sup>, Yichi Zhang,
[Chau Yuen](https://blogs.ntu.edu.sg/chau-yuen/)<sup>1</sup><br> <sup>1</sup>Nanyang Technological University &nbsp;
<sup>2</sup>Yale University &nbsp; <sup>3</sup>University of Manchester

[![arXiv](https://img.shields.io/badge/arXiv-2610.19696-b31b1b?logo=arxiv)](https://arxiv.org/abs/2610.19696)
[![Project page](https://img.shields.io/badge/Project-page-059669)](https://lixin.ai/SYNC/)
[![Word directions](https://img.shields.io/badge/%F0%9F%A4%97%20Word%20directions-SYNC--directions-ffcc4d)](https://huggingface.co/XINLI1997/SYNC-directions)
[![Data](https://img.shields.io/badge/%F0%9F%A4%97%20Data-SYNC--data-ffcc4d)](https://huggingface.co/datasets/XINLI1997/SYNC-data)
[![Code license](https://img.shields.io/badge/code-Apache--2.0-blue)](LICENSE)
[![Data license](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey)](#license)

SYNC (**Sy**mbol-Indexed **N**eural **C**ommunication) lets language models with different architectures pass facts to
each other without fine-tuning or a learned translator.

</div>

## News

- **2026-10-14** The paper is on arXiv: [2610.19696](https://arxiv.org/abs/2610.19696). Code, analysis outputs,
  preregistration locks and scene banks are public; the [project page](https://lixin.ai/SYNC/) is online.

## The idea in 30 seconds

- **The sender sends word indices, not vectors.** From descriptions only it can see, the sender selects words from a
  shared list and sends their indices for each object, without seeing the questions.
- **The receiver uses its own word directions.** The receiver writes the described facts into its hidden states using
  its own word directions, vectors fitted once from its own activations.
- **No training.** Both models stay frozen.

## Quick start

```bash
git clone https://github.com/LiXin97/SYNC.git
cd SYNC
pip install -e .            # torch, transformers, huggingface_hub, numpy, scipy, scikit-learn, matplotlib
```

Python 3.11 or newer. Checking the tables needs only `numpy` and `matplotlib`. Running models needs a GPU. The newer
architectures (Qwen3.5, Qwen3.6, Gemma-4) need a recent `transformers` (we used 5.12 for them and 5.3 for the others).

**Check the paper's numbers (CPU only, a few minutes):**

```bash
python scripts/build_latest_views.py --check              # Tables 1-3, Figure 6 data and their appendix tables
python scripts/build_receiver_recovery_views.py --check   # Figures 4-5, development-receiver tables
python scripts/build_word_interface_views.py --check      # four-object and single-context/embedding tables
```

Expected: each command regenerates its LaTeX tables from the analysis outputs in `artifacts/` and asserts byte
equality with `paper/tables/`. `artifacts/paper_builds/*_provenance.json` record the SHA-256 of every input and
generated table.

Recounting from the per-row records, where each result comes from, re-running the experiments, data generation and the
preregistration locks: [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Models and data

All on Hugging Face, in the [SYNC
collection](https://huggingface.co/collections/XINLI1997/sync-share-the-symbol-not-the-vector-6acf26edd5d52fb91ae87350).

| Item | What is in it | License |
|---|---|---|
| [🤗 SYNC-data](https://huggingface.co/datasets/XINLI1997/SYNC-data) | per-row records: one row per scene, world, question and condition (parsed answer, correctness, coverage and generated text); 1.6 million rows, 2.3 GB | CC BY 4.0 |
| [🤗 SYNC-directions](https://huggingface.co/XINLI1997/SYNC-directions) | the fitted word directions used in the paper | each base model's license |

The language models themselves are not included; they are downloaded from their publishers
([docs/REPRODUCE.md](docs/REPRODUCE.md#models)).

## Repository layout

This repository contains the code, the preregistration locks, the scene banks, every sender message and the analysis
outputs behind the paper's tables and figures. The Python package is `syncomm`.

```
src/syncomm/        the Python package: method code and model loading
scripts/            table builders, analysis scripts and runners
artifacts/          analysis outputs, scene banks, sender prompts and messages, run manifests
gates/              preregistration lock files
paper/tables/       the paper's LaTeX tables, which the checks compare against
licenses/, NOTICE   licenses of the models used in the experiments
SANITIZATION.json   every file edited after the analyses ran
docs/REPRODUCE.md   models, recounts, result map, re-running, data generation, preregistration
```

## Citation

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

## License

Code: Apache License 2.0 ([LICENSE](LICENSE)). The models used in the experiments are not part of this repository and
keep their own licenses ([NOTICE](NOTICE), [licenses/](licenses/)). The per-row records on Hugging Face are released
under CC BY 4.0; the fitted word directions are derived from the base models and follow each base model's license.

**Not included.**

- Model weights (downloaded from their publishers, see [docs/REPRODUCE.md](docs/REPRODUCE.md#models)).
- Fitted word directions: regenerated by `fit_layers` and `scripts/recovery_stack_directions.py`
  ([docs/REPRODUCE.md](docs/REPRODUCE.md#method-code-and-re-running-experiments)); the
  tensors used in the paper are on Hugging Face at
  [XINLI1997/SYNC-directions](https://huggingface.co/XINLI1997/SYNC-directions), each under its base model's license.
- Code of the earlier sparse-extraction studies summarized in the later appendices.
