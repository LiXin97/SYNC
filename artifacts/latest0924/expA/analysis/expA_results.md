# Experiment A (latest0924): frozen analysis

Generated 2026-09-24T20:30:27+00:00 by `scripts/latest0924/analyze_expA.py` (sha256 `f71fc6a5a16a955f`) under `gates/latest0924/expA_protocol.lock.json` (sha256 `1d6327d15123520c`). All numbers are recomputed from the raw rows listed in `expA_results.json` → `inputs` (sha256 of every file read).

Point estimates are family-macro means (36 families) at 64 tokens; intervals are 95% percentile cluster-bootstrap intervals enclosing all three seeds (20260924, 20260925, 20260926; 20,000 draws each). Contrast intervals enclose both clustering levels (36 families, 12 parent templates = domain x template); p = the larger two-sided bootstrap p of the two levels. All values in percentage points. Cells that are not complete are never estimated.

## (a) Sender gemma-4-26b-a4b-it → cross-model receivers

Sender gate: PASS. H1: **9 of 10 evaluated** cross-model receivers show recovery (Holm m = 10; 12 planned). H2 tally: similar: 2, text better: 4, unresolved: 4.

| receiver | status | J(selected) [95%] | J(text) | J(original) | reassigned J range / coverage | coverage(selected) | selected − mean reassigned [both levels] p, Holm thr | H1 | selected − text [both levels] p | H2 |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | COMPLETE | 81.2 [76.7, 85.4] | 86.7 | 4.7 | 1.7–3.2 / 99.4 | 99.9 | +78.6 [+73.0, +84.1] p=1.7e-05, thr 0.01 | RECOVERY | -5.6 [-9.1, -2.3] p=0.00017 | text better |
| qwen3.5-4b | COMPLETE | 96.0 [93.9, 97.9] | 96.7 | 3.1 | 0.7–2.6 / 97.8 | 99.8 | +94.2 [+91.8, +96.2] p=1.7e-05, thr 0.013 | RECOVERY | -0.7 [-3.7, +2.1] p=0.63 | unresolved |
| qwen3.5-9b | COMPLETE | 93.4 [89.7, 96.5] | 94.7 | 4.9 | 2.1–4.2 / 93.9 | 99.3 | +90.4 [+85.1, +94.9] p=1.7e-05, thr 0.017 | RECOVERY | -1.4 [-3.1, +0.4] p=0.13 | unresolved |
| qwen3.5-27b | COMPLETE | 98.3 [96.2, 99.8] | 98.6 | 77.8 | 1.1–3.6 / 86.3 | 99.5 | +96.0 [+93.2, +98.0] p=1.7e-05, thr 0.0083 | RECOVERY | -0.3 [-1.6, +0.5] p=0.77 | similar |
| qwen3.6-27b | COMPLETE | 97.7 [96.0, 99.1] | 98.2 | 82.5 | 2.6–3.7 / 93.9 | 99.9 | +94.7 [+92.3, +96.9] p=1.7e-05, thr 0.025 | RECOVERY | -0.5 [-1.2, +0.0] p=0.086 | similar |
| qwen3.6-35b-a3b | ENGINEERING FAILURE | - | - | - | - | - | - | NOT EVALUABLE | - | NOT EVALUABLE |
| qwen3.5-122b-a10b | ENGINEERING FAILURE (ADR-latest0924-moe-gate) | - | - | - | - | - | - | NOT EVALUABLE | - | NOT EVALUABLE |
| gpt-oss-20b | COMPLETE | 85.3 [81.5, 89.0] | 97.7 | 4.5 | 1.6–1.8 / 99.2 | 99.9 | +83.7 [+79.1, +88.2] p=1.7e-05, thr 0.005 | RECOVERY | -12.4 [-16.1, -8.7] p=1.7e-05 | text better |
| olmo-3-7b-instruct | COMPLETE | 76.0 [68.6, 82.8] | 87.8 | 6.3 | 1.7–4.0 / 98.5 | 93.4 | +72.9 [+61.3, +82.6] p=1.7e-05, thr 0.0071 | RECOVERY | -11.8 [-18.4, -5.5] p=0.00013 | text better |
| seed-oss-36b-instruct | COMPLETE | 35.1 [26.1, 44.3] | 89.5 | 0.0 | 0.0–0.0 / 0.7 | 72.0 | +35.1 [+20.9, +49.0] p=1.7e-05, thr 0.05 | NO RECOVERY (J_selected_ge_0_50, coverage_selected_ge_0_90) | -54.4 [-63.4, -45.2] p=1.7e-05 | text better |
| mistral-small-3.2-24b | COMPLETE | 96.9 [94.3, 99.0] | 96.7 | 0.7 | 1.6–2.3 / 89.4 | 100.0 | +95.0 [+92.2, +97.2] p=1.7e-05, thr 0.0063 | RECOVERY | +0.2 [-3.8, +4.3] p=0.95 | unresolved |
| granite-4.1-8b | COMPLETE | 92.0 [87.0, 96.4] | 94.2 | 14.8 | 0.0–0.2 / 12.4 | 97.5 | +91.9 [+84.2, +97.8] p=1.7e-05, thr 0.0056 | RECOVERY | -2.3 [-4.7, -0.3] p=0.016 | unresolved |

Sensitivity (Holm m = all 12 planned cross-model receivers, unevaluated at p = 1): no verdict changes.

## (a) Sender qwen3.6-27b → cross-model receivers

Sender gate: PASS. H1: **8 of 9 evaluated** cross-model receivers show recovery (Holm m = 9; 11 planned). H2 tally: similar: 1, text better: 6, unresolved: 2.

| receiver | status | J(selected) [95%] | J(text) | J(original) | reassigned J range / coverage | coverage(selected) | selected − mean reassigned [both levels] p, Holm thr | H1 | selected − text [both levels] p | H2 |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | COMPLETE | 79.3 [74.4, 83.9] | 83.8 | 4.0 | 1.4–3.0 / 99.0 | 99.7 | +77.1 [+71.2, +82.5] p=1.7e-05, thr 0.013 | RECOVERY | -4.5 [-8.0, -1.6] p=0.0025 | text better |
| qwen3.5-4b | COMPLETE | 93.4 [88.7, 97.2] | 94.3 | 3.1 | 0.7–2.3 / 97.7 | 99.8 | +91.8 [+84.3, +96.4] p=1.7e-05, thr 0.017 | RECOVERY | -0.9 [-4.0, +1.7] p=0.55 | unresolved |
| qwen3.5-9b | COMPLETE | 93.6 [88.7, 97.6] | 96.2 | 4.9 | 2.1–3.5 / 93.8 | 99.7 | +90.7 [+82.2, +96.0] p=1.7e-05, thr 0.025 | RECOVERY | -2.6 [-4.2, -1.2] p=3.3e-05 | text better |
| qwen3.5-27b | COMPLETE | 96.4 [92.2, 99.8] | 96.7 | 75.7 | 0.9–3.2 / 85.9 | 99.9 | +94.4 [+87.3, +98.3] p=1.7e-05, thr 0.01 | RECOVERY | -0.3 [-1.2, +0.3] p=0.7 | similar |
| qwen3.6-35b-a3b | ENGINEERING FAILURE | - | - | - | - | - | - | NOT EVALUABLE | - | NOT EVALUABLE |
| qwen3.5-122b-a10b | ENGINEERING FAILURE (ADR-latest0924-moe-gate) | - | - | - | - | - | - | NOT EVALUABLE | - | NOT EVALUABLE |
| gpt-oss-20b | COMPLETE | 83.8 [79.5, 87.9] | 95.8 | 4.7 | 1.2–1.7 / 99.1 | 99.9 | +82.2 [+75.6, +88.0] p=1.7e-05, thr 0.0056 | RECOVERY | -12.0 [-16.4, -7.7] p=1.7e-05 | text better |
| olmo-3-7b-instruct | COMPLETE | 72.0 [64.8, 78.7] | 86.1 | 7.0 | 2.1–4.4 / 98.4 | 93.2 | +68.7 [+57.6, +78.7] p=1.7e-05, thr 0.0083 | RECOVERY | -14.1 [-19.7, -8.6] p=1.7e-05 | text better |
| seed-oss-36b-instruct | COMPLETE | 36.0 [26.2, 45.9] | 87.5 | 0.0 | 0.0–0.0 / 1.3 | 71.9 | +36.0 [+20.0, +52.0] p=1.7e-05, thr 0.05 | NO RECOVERY (J_selected_ge_0_50, coverage_selected_ge_0_90) | -51.5 [-63.8, -39.1] p=1.7e-05 | text better |
| mistral-small-3.2-24b | COMPLETE | 94.1 [88.9, 98.3] | 96.0 | 0.9 | 1.2–2.6 / 86.5 | 100.0 | +92.3 [+84.6, +97.3] p=1.7e-05, thr 0.0071 | RECOVERY | -1.9 [-4.7, +0.0] p=0.074 | unresolved |
| granite-4.1-8b | COMPLETE | 90.6 [84.0, 96.0] | 94.4 | 14.7 | 0.0–0.0 / 12.9 | 97.6 | +90.6 [+80.0, +98.1] p=1.7e-05, thr 0.0063 | RECOVERY | -3.8 [-8.6, -0.5] p=0.011 | text better |

Sensitivity (Holm m = all 11 planned cross-model receivers, unevaluated at p = 1) changes: H1 [], H2 ['granite-4.1-8b'].

## (a) Sender gpt-oss-20b → cross-model receivers

Sender gate: PASS. H1: **8 of 9 evaluated** cross-model receivers show recovery (Holm m = 9; 11 planned). H2 tally: similar: 3, text better: 3, unresolved: 3.

| receiver | status | J(selected) [95%] | J(text) | J(original) | reassigned J range / coverage | coverage(selected) | selected − mean reassigned [both levels] p, Holm thr | H1 | selected − text [both levels] p | H2 |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | COMPLETE | 77.5 [70.9, 83.5] | 81.5 | 4.2 | 1.9–2.6 / 99.8 | 100.0 | +75.2 [+65.8, +83.8] p=1.7e-05, thr 0.01 | RECOVERY | -4.0 [-7.1, -1.1] p=0.0043 | text better |
| qwen3.5-4b | COMPLETE | 89.9 [83.9, 94.8] | 88.0 | 2.8 | 0.5–2.3 / 98.5 | 99.5 | +88.3 [+79.3, +94.4] p=1.7e-05, thr 0.013 | RECOVERY | +1.9 [-0.9, +5.3] p=0.23 | unresolved |
| qwen3.5-9b | COMPLETE | 86.6 [79.6, 92.4] | 88.5 | 4.9 | 2.1–3.3 / 94.1 | 97.9 | +83.8 [+72.7, +92.2] p=1.7e-05, thr 0.017 | RECOVERY | -1.9 [-4.3, +0.0] p=0.055 | unresolved |
| qwen3.5-27b | COMPLETE | 88.8 [81.6, 94.7] | 89.1 | 67.4 | 0.9–2.5 / 87.4 | 98.3 | +87.0 [+75.2, +95.6] p=1.7e-05, thr 0.0083 | RECOVERY | -0.3 [-0.9, +0.0] p=0.26 | similar |
| qwen3.6-27b | COMPLETE | 90.6 [84.1, 95.7] | 89.5 | 74.2 | 2.6–3.1 / 94.4 | 99.2 | +87.7 [+77.4, +94.9] p=1.7e-05, thr 0.025 | RECOVERY | +1.1 [-0.2, +2.6] p=0.11 | similar |
| qwen3.6-35b-a3b | ENGINEERING FAILURE | - | - | - | - | - | - | NOT EVALUABLE | - | NOT EVALUABLE |
| qwen3.5-122b-a10b | ENGINEERING FAILURE (ADR-latest0924-moe-gate) | - | - | - | - | - | - | NOT EVALUABLE | - | NOT EVALUABLE |
| olmo-3-7b-instruct | COMPLETE | 77.1 [69.8, 83.6] | 84.6 | 5.1 | 1.7–3.5 / 98.6 | 96.6 | +74.4 [+64.6, +83.2] p=1.7e-05, thr 0.0071 | RECOVERY | -7.6 [-14.9, -1.8] p=0.0025 | text better |
| seed-oss-36b-instruct | COMPLETE | 24.7 [18.2, 31.4] | 82.5 | 0.0 | 0.0–0.0 / 0.5 | 65.9 | +24.7 [+15.3, +34.0] p=1.7e-05, thr 0.05 | NO RECOVERY (J_selected_ge_0_50, coverage_selected_ge_0_90) | -57.8 [-70.0, -43.5] p=1.7e-05 | text better |
| mistral-small-3.2-24b | COMPLETE | 89.0 [82.6, 94.2] | 88.8 | 0.7 | 1.9–2.1 / 92.6 | 99.5 | +87.0 [+77.4, +93.9] p=1.7e-05, thr 0.0063 | RECOVERY | +0.2 [-2.9, +3.0] p=0.86 | similar |
| granite-4.1-8b | COMPLETE | 83.1 [75.0, 90.1] | 86.0 | 12.6 | 0.0–0.2 / 11.9 | 94.3 | +83.0 [+70.6, +92.7] p=1.7e-05, thr 0.0056 | RECOVERY | -3.0 [-6.5, -0.0] p=0.043 | unresolved |

Sensitivity (Holm m = all 11 planned cross-model receivers, unevaluated at p = 1): no verdict changes.

## (a′) Same-model pairs (reported separately; excluded from every cross-model count, no Holm)

| pair | status | J(selected) [95%] | J(text) | J(original) | coverage(selected) | selected − mean reassigned | H1 criteria (unadjusted) | selected − text | H2-like (unadjusted) |
|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b -> gpt-oss-20b | COMPLETE | 77.5 [71.2, 83.1] | 90.9 | 5.1 | 99.8 | +75.8 [+68.1, +82.6] p=1.7e-05 | RECOVERY (same-model) | -13.4 [-17.3, -9.8] p=1.7e-05 | text better |
| qwen3.6-27b -> qwen3.6-27b | COMPLETE | 96.0 [91.3, 99.8] | 96.0 | 77.1 | 100.0 | +93.2 [+85.2, +97.9] p=1.7e-05 | RECOVERY (same-model) | +0.0 [+0.0, +0.0] p=1 | similar |

## (b) Engineering audit per cell (recomputed from the engineering rows; receipts not trusted)

| sender → receiver | gate | rows / expected | receipt | max logit diff (at) | max alias gap | parsed unequal | rows failing applicable gate | stored pass ≠ recomputed | identity not inert | verdict | cell status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-26b-a4b-it → qwen3.5-2b | lock | 208 / 208 | valid | 8.392e-05 (013a221b|w1|q0|literal|packet_text) | 7.069e-05 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| gemma-4-26b-a4b-it → qwen3.5-4b | lock | 208 / 208 | valid | 8.774e-05 (01c022ca|w1|q1|literal|full_text) | 0.000103 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| gemma-4-26b-a4b-it → qwen3.5-9b | lock | 208 / 208 | valid | 0.000102 (006580f7|w0|q0|gemma-4-26b-a4b-it|sham1) | 6.962e-05 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| gemma-4-26b-a4b-it → qwen3.5-27b | lock | 208 / 208 | valid | 0.0003061 (006580f7|w1|q1|literal|native) | 0.0002083 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| gemma-4-26b-a4b-it → qwen3.6-27b | lock | 208 / 208 | valid | 0.003591 (000ab0c6|w0|q0|gemma-4-26b-a4b-it|original) | 0.0003794 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| gemma-4-26b-a4b-it → qwen3.6-35b-a3b | lock | 52 / 208 | missing | 0.1359 (000ab0c6|w1|q0|gemma-4-26b-a4b-it|sham1) | 0.01312 | 0 | 2 | 0 | 0/4 | FAIL | ENGINEERING FAILURE |
| gemma-4-26b-a4b-it → qwen3.5-122b-a10b | ADR | 52 / 208 | missing | 0.8013 (000ab0c6|w0|q1|gemma-4-26b-a4b-it|sham1) | 0.1677 | 0 | 1 | 0 | 0/4 | FAIL | ENGINEERING FAILURE (ADR-latest0924-moe-gate) |
| gemma-4-26b-a4b-it → gpt-oss-20b | lock | 208 / 208 | valid | 8.965e-05 (000ab0c6|w1|q1|gemma-4-26b-a4b-it|original) | 5.341e-05 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| gemma-4-26b-a4b-it → olmo-3-7b-instruct | lock | 208 / 208 | valid | 0.0001206 (006580f7|w1|q0|gemma-4-26b-a4b-it|original) | 9.155e-05 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| gemma-4-26b-a4b-it → seed-oss-36b-instruct | lock | 208 / 208 | valid | 0.000309 (006580f7|w0|q0|gemma-4-26b-a4b-it|other_world) | 0.0002718 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| gemma-4-26b-a4b-it → mistral-small-3.2-24b | lock | 208 / 208 | valid | 0.0001507 (01c022ca|w1|q1|literal|full_text) | 5.722e-05 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| gemma-4-26b-a4b-it → granite-4.1-8b | lock | 208 / 208 | valid | 0.0002441 (013a221b|w1|q0|gemma-4-26b-a4b-it|sham2) | 0.0001146 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| qwen3.6-27b → qwen3.5-2b | lock | 144 / 144 | valid | 7.915e-05 (01c022ca|w1|q1|qwen3.6-27b|sham0) | 6.545e-05 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| qwen3.6-27b → qwen3.5-4b | lock | 144 / 144 | valid | 8.774e-05 (01c022ca|w1|q1|qwen3.6-27b|original) | 5.341e-05 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| qwen3.6-27b → qwen3.5-9b | lock | 144 / 144 | valid | 7.82e-05 (006580f7|w1|q0|qwen3.6-27b|packet_text) | 7.534e-05 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| qwen3.6-27b → qwen3.5-27b | lock | 144 / 144 | valid | 0.0003421 (006580f7|w0|q1|qwen3.6-27b|original) | 0.0002232 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| qwen3.6-27b → qwen3.6-27b | lock | 144 / 144 | valid | 0.003591 (000ab0c6|w0|q0|qwen3.6-27b|original) | 0.0003794 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| qwen3.6-27b → qwen3.6-35b-a3b | lock | 0 / 144 | - | - | - | - | - | - | - | NOT STARTED | ENGINEERING FAILURE |
| qwen3.6-27b → qwen3.5-122b-a10b | ADR | 0 / 144 | - | - | - | - | - | - | - | NOT STARTED | ENGINEERING FAILURE (ADR-latest0924-moe-gate) |
| qwen3.6-27b → gpt-oss-20b | lock | 144 / 144 | valid | 8.965e-05 (000ab0c6|w1|q1|qwen3.6-27b|original) | 6.39e-05 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| qwen3.6-27b → olmo-3-7b-instruct | lock | 144 / 144 | valid | 9.584e-05 (006580f7|w0|q1|qwen3.6-27b|sham0) | 9.537e-05 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| qwen3.6-27b → seed-oss-36b-instruct | lock | 144 / 144 | valid | 0.0001831 (013a221b|w0|q0|qwen3.6-27b|packet_text) | 0.0001335 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| qwen3.6-27b → mistral-small-3.2-24b | lock | 144 / 144 | valid | 0.0001106 (01c022ca|w0|q1|qwen3.6-27b|selected) | 4.28e-05 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| qwen3.6-27b → granite-4.1-8b | lock | 144 / 144 | valid | 0.0002441 (013a221b|w1|q0|qwen3.6-27b|sham2) | 0.0001869 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| gpt-oss-20b → qwen3.5-2b | lock | 144 / 144 | valid | 8.392e-05 (013a221b|w1|q0|gpt-oss-20b|packet_text) | 7.069e-05 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| gpt-oss-20b → qwen3.5-4b | lock | 144 / 144 | valid | 8.774e-05 (01c022ca|w1|q1|gpt-oss-20b|original) | 5.341e-05 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| gpt-oss-20b → qwen3.5-9b | lock | 144 / 144 | valid | 0.000102 (006580f7|w0|q0|gpt-oss-20b|sham1) | 7.057e-05 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| gpt-oss-20b → qwen3.5-27b | lock | 144 / 144 | valid | 0.0003134 (013a221b|w0|q1|gpt-oss-20b|sham0) | 0.0002232 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| gpt-oss-20b → qwen3.6-27b | lock | 144 / 144 | valid | 0.003591 (000ab0c6|w0|q0|gpt-oss-20b|original) | 0.0003794 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| gpt-oss-20b → qwen3.6-35b-a3b | lock | 0 / 144 | - | - | - | - | - | - | - | NOT STARTED | ENGINEERING FAILURE |
| gpt-oss-20b → qwen3.5-122b-a10b | ADR | 0 / 144 | - | - | - | - | - | - | - | NOT STARTED | ENGINEERING FAILURE (ADR-latest0924-moe-gate) |
| gpt-oss-20b → gpt-oss-20b | lock | 144 / 144 | valid | 8.965e-05 (000ab0c6|w1|q1|gpt-oss-20b|original) | 3.91e-05 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| gpt-oss-20b → olmo-3-7b-instruct | lock | 144 / 144 | valid | 0.0001206 (006580f7|w1|q0|gpt-oss-20b|original) | 9.155e-05 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| gpt-oss-20b → seed-oss-36b-instruct | lock | 144 / 144 | valid | 0.000309 (006580f7|w0|q0|gpt-oss-20b|other_world) | 0.0002718 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| gpt-oss-20b → mistral-small-3.2-24b | lock | 144 / 144 | valid | 0.0001106 (01c022ca|w0|q1|gpt-oss-20b|selected) | 5.722e-05 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |
| gpt-oss-20b → granite-4.1-8b | lock | 144 / 144 | valid | 0.0002174 (000ab0c6|w1|q1|gpt-oss-20b|sham1) | 0.0001146 | 0 | 0 | 0 | 0/16 | PASS | COMPLETE |

**qwen3.6-35b-a3b: ENGINEERING FAILURE** (gemma-4-26b-a4b-it -> qwen3.6-35b-a3b: recomputed gate/identity failure on engineering rows; qwen3.6-27b -> qwen3.6-35b-a3b: ENGINEERING_FAILED marker with failure note; gpt-oss-20b -> qwen3.6-35b-a3b: ENGINEERING_FAILED marker with failure note). No outcome rows under any sender.

> Note: Receiver-level engineering failure (expA_protocol.lock.json engineering_gate): gemma-4-26b-a4b-it x qwen3.6-35b-a3b engineering max cached-vs-complete logit difference 0.1359 (sham1) and 0.0530 (literal/native) > 0.05; parsed answers equal on all 52 rows; alias-gap differences <= 0.0131. Recorded 2026-09-24T14:4xZ. No outcome rows for this receiver under any sender.

Recomputed from artifacts/latest0924/expA/gemma-4-26b-a4b-it/qwen3.6-35b-a3b/raw/engineering_qwen3.6-35b-a3b_s0.jsonl: 52 rows; max logit difference 0.1359 at 000ab0c6|w1|q0|gemma-4-26b-a4b-it|sham1; max alias-gap difference 0.0131; parsed-unequal rows 0; rows failing: lock/evaluator gate 2, lock-literal gate (logit + parsed) 2, ADR gate 0; applicable = lock/evaluator (max logit <= 0.05, alias gap <= 0.1, parsed equal).

**qwen3.5-122b-a10b: ENGINEERING FAILURE** (gemma-4-26b-a4b-it -> qwen3.5-122b-a10b: recomputed gate/identity failure on engineering rows; qwen3.6-27b -> qwen3.5-122b-a10b: ENGINEERING_FAILED marker with failure note; gpt-oss-20b -> qwen3.5-122b-a10b: ENGINEERING_FAILED marker with failure note). No outcome rows under any sender.

> Note: Receiver-level engineering failure under ADR-latest0924-moe-gate (amended gate: equal parsed answers and alias-gap difference <= 0.1): gemma-4-26b-a4b-it x qwen3.5-122b-a10b engineering row sham0 world0 q1 alias-gap difference 0.1677 > 0.1 (parsed answers equal, both 'unknown'); max logit differences up to 0.80 (reported only). Recorded 2026-09-24T14:58:28Z. No Experiment A outcome rows for this receiver under any sender.

Recomputed from artifacts/latest0924/expA/gemma-4-26b-a4b-it/qwen3.5-122b-a10b/raw/engineering_qwen3.5-122b-a10b_s0.jsonl: 52 rows; max logit difference 0.8013 at 000ab0c6|w0|q1|gemma-4-26b-a4b-it|sham1; max alias-gap difference 0.1677; parsed-unequal rows 0; rows failing: lock/evaluator gate 3, lock-literal gate (logit + parsed) 3, ADR gate 1; applicable = ADR-latest0924-moe-gate (parsed equal and alias gap <= 0.1).

## (c) Scale ladder (H3)

### Sender gemma-4-26b-a4b-it

**H3 primary (gap(122B) − gap(2B)): NOT EVALUABLE (qwen3.5-122b-a10b engineering failure, ADR-latest0924-moe-gate)**

| ladder point | params | status | J(selected) | J(text) | gap = selected − text [both levels] |
|---|---|---|---|---|---|
| qwen3.5-2b | 2e+09 | COMPLETE | 81.2 | 86.7 | -5.6 [-9.1, -2.3] |
| qwen3.5-4b | 4e+09 | COMPLETE | 96.0 | 96.7 | -0.7 [-3.7, +2.1] |
| qwen3.5-9b | 9e+09 | COMPLETE | 93.4 | 94.7 | -1.4 [-3.1, +0.4] |
| qwen3.5-27b | 3e+10 | COMPLETE | 98.3 | 98.6 | -0.3 [-1.6, +0.5] |
| qwen3.5-122b-a10b | 1e+11 | ENGINEERING FAILURE (ADR-latest0924-moe-gate) | - | - | - |
| qwen3.6-27b | 3e+10 | COMPLETE | 97.7 | 98.2 | -0.5 [-1.2, +0.0] |
| qwen3.6-35b-a3b | 4e+10 | ENGINEERING FAILURE | - | - | - |

Secondary (DESCRIPTIVE ONLY, no significance claim): Spearman(log params, gap) over 5 available points = +0.872.
Descriptive gap(qwen3.5-27b) − gap(qwen3.5-2b), NOT the preregistered primary: +5.2 [+1.9, +8.9] (family 5.2 [2.5, 8.0]; template 5.2 [1.9, 8.9]).

### Sender qwen3.6-27b

**H3 primary (gap(122B) − gap(2B)): NOT EVALUABLE (qwen3.5-122b-a10b engineering failure, ADR-latest0924-moe-gate)**

| ladder point | params | status | J(selected) | J(text) | gap = selected − text [both levels] |
|---|---|---|---|---|---|
| qwen3.5-2b | 2e+09 | COMPLETE | 79.3 | 83.8 | -4.5 [-8.0, -1.6] |
| qwen3.5-4b | 4e+09 | COMPLETE | 93.4 | 94.3 | -0.9 [-4.0, +1.7] |
| qwen3.5-9b | 9e+09 | COMPLETE | 93.6 | 96.2 | -2.6 [-4.2, -1.2] |
| qwen3.5-27b | 3e+10 | COMPLETE | 96.4 | 96.7 | -0.3 [-1.2, +0.3] |
| qwen3.5-122b-a10b | 1e+11 | ENGINEERING FAILURE (ADR-latest0924-moe-gate) | - | - | - |
| qwen3.6-27b | 3e+10 | same-model pair (excluded) | - | - | - |
| qwen3.6-35b-a3b | 4e+10 | ENGINEERING FAILURE | - | - | - |

Secondary (DESCRIPTIVE ONLY, no significance claim): Spearman(log params, gap) over 4 available points = +0.800.
Descriptive gap(qwen3.5-27b) − gap(qwen3.5-2b), NOT the preregistered primary: +4.2 [+0.9, +7.8] (family 4.2 [1.1, 7.4]; template 4.2 [0.9, 7.8]).

### Sender gpt-oss-20b

**H3 primary (gap(122B) − gap(2B)): NOT EVALUABLE (qwen3.5-122b-a10b engineering failure, ADR-latest0924-moe-gate)**

| ladder point | params | status | J(selected) | J(text) | gap = selected − text [both levels] |
|---|---|---|---|---|---|
| qwen3.5-2b | 2e+09 | COMPLETE | 77.5 | 81.5 | -4.0 [-7.1, -1.1] |
| qwen3.5-4b | 4e+09 | COMPLETE | 89.9 | 88.0 | +1.9 [-0.9, +5.3] |
| qwen3.5-9b | 9e+09 | COMPLETE | 86.6 | 88.5 | -1.9 [-4.3, +0.0] |
| qwen3.5-27b | 3e+10 | COMPLETE | 88.8 | 89.1 | -0.3 [-0.9, +0.0] |
| qwen3.5-122b-a10b | 1e+11 | ENGINEERING FAILURE (ADR-latest0924-moe-gate) | - | - | - |
| qwen3.6-27b | 3e+10 | COMPLETE | 90.6 | 89.5 | +1.1 [-0.2, +2.6] |
| qwen3.6-35b-a3b | 4e+10 | ENGINEERING FAILURE | - | - | - |

Secondary (DESCRIPTIVE ONLY, no significance claim): Spearman(log params, gap) over 5 available points = +0.359.
Descriptive gap(qwen3.5-27b) − gap(qwen3.5-2b), NOT the preregistered primary: +3.7 [+0.5, +7.0] (family 3.7 [0.9, 6.6]; template 3.7 [0.5, 7.0]).

## (d) Completeness per cell

| sender → receiver | cell status | engineering | dev shards (receipted / present, declared S) | dev rows present / expected | notes |
|---|---|---|---|---|---|
| gemma-4-26b-a4b-it → qwen3.5-2b | COMPLETE | PASS | 2 / 2, S=[2] | 29744 / 29744 | - |
| gemma-4-26b-a4b-it → qwen3.5-4b | COMPLETE | PASS | 2 / 2, S=[2] | 29744 / 29744 | - |
| gemma-4-26b-a4b-it → qwen3.5-9b | COMPLETE | PASS | 2 / 2, S=[2] | 29744 / 29744 | - |
| gemma-4-26b-a4b-it → qwen3.5-27b | COMPLETE | PASS | 3 / 3, S=[3] | 29744 / 29744 | - |
| gemma-4-26b-a4b-it → qwen3.6-27b | COMPLETE | PASS | 3 / 3, S=[3] | 29744 / 29744 | - |
| gemma-4-26b-a4b-it → qwen3.6-35b-a3b | ENGINEERING FAILURE | FAIL | 0 / 0, S=? | 0 / 29744 | - |
| gemma-4-26b-a4b-it → qwen3.5-122b-a10b | ENGINEERING FAILURE (ADR-latest0924-moe-gate) | FAIL | 0 / 0, S=? | 0 / 29744 | - |
| gemma-4-26b-a4b-it → gpt-oss-20b | COMPLETE | PASS | 2 / 2, S=[2] | 29744 / 29744 | - |
| gemma-4-26b-a4b-it → olmo-3-7b-instruct | COMPLETE | PASS | 2 / 2, S=[2] | 29744 / 29744 | - |
| gemma-4-26b-a4b-it → seed-oss-36b-instruct | COMPLETE | PASS | 3 / 3, S=[3] | 29744 / 29744 | - |
| gemma-4-26b-a4b-it → mistral-small-3.2-24b | COMPLETE | PASS | 3 / 3, S=[3] | 29744 / 29744 | - |
| gemma-4-26b-a4b-it → granite-4.1-8b | COMPLETE | PASS | 2 / 2, S=[2] | 29744 / 29744 | - |
| qwen3.6-27b → qwen3.5-2b | COMPLETE | PASS | 2 / 2, S=[2] | 20592 / 20592 | - |
| qwen3.6-27b → qwen3.5-4b | COMPLETE | PASS | 2 / 2, S=[2] | 20592 / 20592 | - |
| qwen3.6-27b → qwen3.5-9b | COMPLETE | PASS | 2 / 2, S=[2] | 20592 / 20592 | - |
| qwen3.6-27b → qwen3.5-27b | COMPLETE | PASS | 3 / 3, S=[3] | 20592 / 20592 | - |
| qwen3.6-27b → qwen3.6-27b (same-model) | COMPLETE | PASS | 3 / 3, S=[3] | 20592 / 20592 | - |
| qwen3.6-27b → qwen3.6-35b-a3b | ENGINEERING FAILURE | NOT_STARTED | 0 / 0, S=? | 0 / 20592 | - |
| qwen3.6-27b → qwen3.5-122b-a10b | ENGINEERING FAILURE (ADR-latest0924-moe-gate) | NOT_STARTED | 0 / 0, S=? | 0 / 20592 | - |
| qwen3.6-27b → gpt-oss-20b | COMPLETE | PASS | 2 / 2, S=[2] | 20592 / 20592 | - |
| qwen3.6-27b → olmo-3-7b-instruct | COMPLETE | PASS | 2 / 2, S=[2] | 20592 / 20592 | - |
| qwen3.6-27b → seed-oss-36b-instruct | COMPLETE | PASS | 3 / 3, S=[3] | 20592 / 20592 | - |
| qwen3.6-27b → mistral-small-3.2-24b | COMPLETE | PASS | 3 / 3, S=[3] | 20592 / 20592 | - |
| qwen3.6-27b → granite-4.1-8b | COMPLETE | PASS | 2 / 2, S=[2] | 20592 / 20592 | - |
| gpt-oss-20b → qwen3.5-2b | COMPLETE | PASS | 2 / 2, S=[2] | 20592 / 20592 | - |
| gpt-oss-20b → qwen3.5-4b | COMPLETE | PASS | 2 / 2, S=[2] | 20592 / 20592 | - |
| gpt-oss-20b → qwen3.5-9b | COMPLETE | PASS | 2 / 2, S=[2] | 20592 / 20592 | - |
| gpt-oss-20b → qwen3.5-27b | COMPLETE | PASS | 4 / 4, S=[4] | 20592 / 20592 | - |
| gpt-oss-20b → qwen3.6-27b | COMPLETE | PASS | 4 / 4, S=[4] | 20592 / 20592 | - |
| gpt-oss-20b → qwen3.6-35b-a3b | ENGINEERING FAILURE | NOT_STARTED | 0 / 0, S=? | 0 / 20592 | - |
| gpt-oss-20b → qwen3.5-122b-a10b | ENGINEERING FAILURE (ADR-latest0924-moe-gate) | NOT_STARTED | 0 / 0, S=? | 0 / 20592 | - |
| gpt-oss-20b → gpt-oss-20b (same-model) | COMPLETE | PASS | 2 / 2, S=[2] | 20592 / 20592 | - |
| gpt-oss-20b → olmo-3-7b-instruct | COMPLETE | PASS | 2 / 2, S=[2] | 20592 / 20592 | - |
| gpt-oss-20b → seed-oss-36b-instruct | COMPLETE | PASS | 6 / 6, S=[6] | 20592 / 20592 | - |
| gpt-oss-20b → mistral-small-3.2-24b | COMPLETE | PASS | 3 / 3, S=[3] | 20592 / 20592 | - |
| gpt-oss-20b → granite-4.1-8b | COMPLETE | PASS | 2 / 2, S=[2] | 20592 / 20592 | - |

## (e) Complete facts (source `literal`, first-sender cells) — descriptive

| receiver | status | J selected [95%] | J native | J packet_text | J full_text | coverage selected | selected − text | selected − native | selected − full_text |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | COMPLETE | 86.3 [82.3, 89.9] | 85.9 [81.9, 89.6] | 89.6 [86.3, 92.7] | 97.2 [93.9, 99.8] | 100.0 | -3.3 [-5.6, -1.0] | +0.4 [-2.0, +2.7] | -11.0 [-17.6, -3.0] |
| qwen3.5-4b | COMPLETE | 99.0 [97.7, 99.8] | 99.7 [99.1, 100.0] | 98.4 [96.7, 99.7] | 100.0 [100.0, 100.0] | 100.0 | +0.5 [-0.7, +1.7] | -0.7 [-1.7, +0.2] | -1.0 [-2.3, +0.0] |
| qwen3.5-9b | COMPLETE | 96.2 [93.9, 98.1] | 99.8 [99.5, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.7 | -3.8 [-7.0, -1.2] | -3.7 [-6.8, -1.2] | -3.8 [-7.0, -1.2] |
| qwen3.5-27b | COMPLETE | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| qwen3.6-27b | COMPLETE | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| qwen3.6-35b-a3b | ENGINEERING FAILURE | - | - | - | - | - | - | - | - |
| qwen3.5-122b-a10b | ENGINEERING FAILURE (ADR-latest0924-moe-gate) | - | - | - | - | - | - | - | - |
| gpt-oss-20b | COMPLETE | 86.1 [81.9, 89.9] | 90.6 [87.3, 93.6] | 99.1 [98.1, 99.8] | 100.0 [100.0, 100.0] | 99.8 | -13.1 [-17.3, -8.9] | -4.5 [-7.5, -1.8] | -13.9 [-19.0, -9.3] |
| olmo-3-7b-instruct | COMPLETE | 91.6 [89.3, 93.9] | 90.4 [86.8, 93.7] | 94.6 [92.3, 96.7] | 96.0 [91.3, 99.5] | 98.1 | -3.0 [-5.3, -0.6] | +1.2 [-2.1, +4.9] | -4.4 [-8.9, +2.2] |
| seed-oss-36b-instruct | COMPLETE | 30.6 [22.2, 39.4] | 67.8 [59.5, 75.7] | 99.8 [99.5, 100.0] | 100.0 [100.0, 100.0] | 68.9 | -69.3 [-81.7, -56.5] | -37.2 [-45.2, -29.2] | -69.4 [-82.1, -56.5] |
| mistral-small-3.2-24b | COMPLETE | 97.6 [95.0, 99.5] | 98.4 [96.9, 99.7] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 | -2.4 [-5.0, -0.5] | -0.9 [-3.6, +1.2] | -2.4 [-5.0, -0.5] |
| granite-4.1-8b | COMPLETE | 98.1 [96.9, 99.1] | 97.6 [95.0, 99.5] | 100.0 [100.0, 100.0] | 99.8 [99.5, 100.0] | 99.5 | -1.9 [-3.1, -0.9] | +0.5 [-1.6, +3.0] | -1.7 [-3.1, -0.5] |

## (f) H4 depth (descriptive): selected − original

| sender → receiver | J(selected) | J(original) | selected − original [both levels] | family [95%] | template [95%] |
|---|---|---|---|---|---|
| gemma-4-26b-a4b-it → qwen3.5-2b | 81.2 | 4.7 | +76.4 [+72.3, +80.5] | 76.4 [72.4, 80.5] | 76.4 [72.3, 80.4] |
| gemma-4-26b-a4b-it → qwen3.5-4b | 96.0 | 3.1 | +92.9 [+89.4, +96.2] | 92.9 [90.3, 95.3] | 92.9 [89.4, 96.2] |
| gemma-4-26b-a4b-it → qwen3.5-9b | 93.4 | 4.9 | +88.5 [+82.5, +93.9] | 88.5 [84.3, 92.2] | 88.5 [82.5, 93.9] |
| gemma-4-26b-a4b-it → qwen3.5-27b | 98.3 | 77.8 | +20.4 [+12.9, +28.9] | 20.4 [14.8, 26.2] | 20.4 [12.9, 28.9] |
| gemma-4-26b-a4b-it → qwen3.6-27b | 97.7 | 82.5 | +15.2 [+8.2, +23.6] | 15.2 [9.9, 21.0] | 15.2 [8.2, 23.6] |
| gemma-4-26b-a4b-it → qwen3.6-35b-a3b | NOT EVALUABLE (ENGINEERING FAILURE) | - | - | - | - |
| gemma-4-26b-a4b-it → qwen3.5-122b-a10b | NOT EVALUABLE (ENGINEERING FAILURE (ADR-latest0924-moe-gate)) | - | - | - | - |
| gemma-4-26b-a4b-it → gpt-oss-20b | 85.3 | 4.5 | +80.8 [+76.1, +85.5] | 80.8 [76.7, 84.7] | 80.8 [76.1, 85.5] |
| gemma-4-26b-a4b-it → olmo-3-7b-instruct | 76.0 | 6.3 | +69.7 [+57.5, +79.9] | 69.7 [61.6, 77.2] | 69.8 [57.5, 79.9] |
| gemma-4-26b-a4b-it → seed-oss-36b-instruct | 35.1 | 0.0 | +35.1 [+20.9, +49.0] | 35.1 [26.1, 44.3] | 35.1 [20.9, 49.0] |
| gemma-4-26b-a4b-it → mistral-small-3.2-24b | 96.9 | 0.7 | +96.2 [+93.2, +98.4] | 96.2 [93.4, 98.3] | 96.2 [93.2, 98.4] |
| gemma-4-26b-a4b-it → granite-4.1-8b | 92.0 | 14.8 | +77.1 [+67.9, +85.0] | 77.1 [70.1, 83.7] | 77.1 [67.9, 85.0] |
| qwen3.6-27b → qwen3.5-2b | 79.3 | 4.0 | +75.2 [+70.1, +79.8] | 75.2 [70.5, 79.8] | 75.2 [70.1, 79.6] |
| qwen3.6-27b → qwen3.5-4b | 93.4 | 3.1 | +90.3 [+82.4, +96.0] | 90.3 [85.1, 94.4] | 90.3 [82.4, 96.0] |
| qwen3.6-27b → qwen3.5-9b | 93.6 | 4.9 | +88.7 [+80.2, +94.9] | 88.7 [83.8, 92.9] | 88.7 [80.2, 94.9] |
| qwen3.6-27b → qwen3.5-27b | 96.4 | 75.7 | +20.6 [+12.1, +30.1] | 20.6 [14.6, 26.8] | 20.6 [12.1, 30.1] |
| qwen3.6-27b → qwen3.6-27b (same-model) | 96.0 | 77.1 | +18.9 [+10.8, +27.8] | 18.9 [13.0, 25.0] | 18.9 [10.8, 27.8] |
| qwen3.6-27b → qwen3.6-35b-a3b | NOT EVALUABLE (ENGINEERING FAILURE) | - | - | - | - |
| qwen3.6-27b → qwen3.5-122b-a10b | NOT EVALUABLE (ENGINEERING FAILURE (ADR-latest0924-moe-gate)) | - | - | - | - |
| qwen3.6-27b → gpt-oss-20b | 83.8 | 4.7 | +79.1 [+72.0, +85.3] | 79.1 [74.5, 83.5] | 79.0 [72.0, 85.3] |
| qwen3.6-27b → olmo-3-7b-instruct | 72.0 | 7.0 | +65.0 [+53.3, +75.6] | 65.0 [56.9, 72.7] | 65.0 [53.3, 75.6] |
| qwen3.6-27b → seed-oss-36b-instruct | 36.0 | 0.0 | +36.0 [+20.0, +52.0] | 36.0 [26.2, 45.9] | 36.0 [20.0, 52.0] |
| qwen3.6-27b → mistral-small-3.2-24b | 94.1 | 0.9 | +93.2 [+85.4, +98.4] | 93.2 [88.0, 97.2] | 93.2 [85.4, 98.4] |
| qwen3.6-27b → granite-4.1-8b | 90.6 | 14.7 | +75.9 [+65.0, +85.1] | 75.9 [68.5, 82.9] | 75.9 [65.0, 85.1] |
| gpt-oss-20b → qwen3.5-2b | 77.5 | 4.2 | +73.3 [+65.2, +80.4] | 73.3 [67.5, 78.5] | 73.3 [65.2, 80.4] |
| gpt-oss-20b → qwen3.5-4b | 89.9 | 2.8 | +87.1 [+77.7, +93.9] | 87.1 [81.1, 92.2] | 87.1 [77.7, 93.9] |
| gpt-oss-20b → qwen3.5-9b | 86.6 | 4.9 | +81.7 [+70.9, +90.4] | 81.7 [74.6, 87.8] | 81.6 [70.9, 90.4] |
| gpt-oss-20b → qwen3.5-27b | 88.8 | 67.4 | +21.4 [+10.7, +33.4] | 21.4 [14.4, 28.7] | 21.5 [10.7, 33.4] |
| gpt-oss-20b → qwen3.6-27b | 90.6 | 74.2 | +16.3 [+7.7, +26.4] | 16.3 [10.2, 22.8] | 16.4 [7.7, 26.4] |
| gpt-oss-20b → qwen3.6-35b-a3b | NOT EVALUABLE (ENGINEERING FAILURE) | - | - | - | - |
| gpt-oss-20b → qwen3.5-122b-a10b | NOT EVALUABLE (ENGINEERING FAILURE (ADR-latest0924-moe-gate)) | - | - | - | - |
| gpt-oss-20b → gpt-oss-20b (same-model) | 77.5 | 5.1 | +72.4 [+64.3, +79.4] | 72.4 [65.9, 78.4] | 72.4 [64.3, 79.4] |
| gpt-oss-20b → olmo-3-7b-instruct | 77.1 | 5.1 | +72.0 [+62.3, +80.9] | 72.0 [64.5, 78.8] | 72.0 [62.3, 80.9] |
| gpt-oss-20b → seed-oss-36b-instruct | 24.7 | 0.0 | +24.7 [+15.3, +34.0] | 24.7 [18.2, 31.4] | 24.7 [15.3, 34.0] |
| gpt-oss-20b → mistral-small-3.2-24b | 89.0 | 0.7 | +88.3 [+78.2, +95.5] | 88.3 [81.9, 93.6] | 88.3 [78.2, 95.5] |
| gpt-oss-20b → granite-4.1-8b | 83.1 | 12.6 | +70.5 [+59.4, +80.0] | 70.5 [62.9, 77.5] | 70.4 [59.4, 80.0] |

## (g) Other endpoints (complete cells, sender source)

| sender → receiver | arm | J | accuracy | coverage | changed R | unchanged U | J@256 | coverage@256 |
|---|---|---|---|---|---|---|---|---|
| gemma-4-26b-a4b-it → qwen3.5-2b | none | 0.0 | 50.3 | 100.0 | 0.0 | 50.7 | 0.0 | 100.0 |
| gemma-4-26b-a4b-it → qwen3.5-2b | original | 4.7 | 54.5 | 99.8 | 8.0 | 52.4 | 4.7 | 99.8 |
| gemma-4-26b-a4b-it → qwen3.5-2b | selected | 81.2 | 94.2 | 99.9 | 88.0 | 92.0 | 81.2 | 99.9 |
| gemma-4-26b-a4b-it → qwen3.5-2b | packet_text | 86.7 | 96.0 | 100.0 | 91.6 | 94.4 | 86.7 | 100.0 |
| gemma-4-26b-a4b-it → qwen3.5-2b | sham0 | 2.6 | 50.4 | 99.3 | 6.3 | 44.9 | 2.6 | 99.3 |
| gemma-4-26b-a4b-it → qwen3.5-2b | sham1 | 3.2 | 49.8 | 99.1 | 5.6 | 46.0 | 3.2 | 99.1 |
| gemma-4-26b-a4b-it → qwen3.5-2b | sham2 | 1.7 | 50.3 | 99.7 | 5.1 | 43.7 | 1.7 | 99.7 |
| gemma-4-26b-a4b-it → qwen3.5-2b | other_world | 0.0 | 50.2 | 99.9 | 0.0 | 92.0 | 0.0 | 99.9 |
| gemma-4-26b-a4b-it → qwen3.5-2b | other_world vs SUPPLIED gold (consistency) | 81.2 | 94.2 | - | - | - | 81.2 | - |
| gemma-4-26b-a4b-it → qwen3.5-4b | none | 0.0 | 49.9 | 99.8 | 0.0 | 50.0 | 0.0 | 99.8 |
| gemma-4-26b-a4b-it → qwen3.5-4b | original | 3.1 | 52.2 | 99.7 | 4.7 | 50.4 | 3.1 | 99.7 |
| gemma-4-26b-a4b-it → qwen3.5-4b | selected | 96.0 | 98.9 | 99.8 | 97.6 | 98.3 | 96.0 | 99.8 |
| gemma-4-26b-a4b-it → qwen3.5-4b | packet_text | 96.7 | 99.1 | 99.7 | 98.4 | 98.2 | 96.7 | 99.7 |
| gemma-4-26b-a4b-it → qwen3.5-4b | sham0 | 2.6 | 48.7 | 95.9 | 4.7 | 46.3 | 2.6 | 95.9 |
| gemma-4-26b-a4b-it → qwen3.5-4b | sham1 | 2.1 | 49.9 | 99.2 | 4.6 | 44.6 | 2.1 | 99.2 |
| gemma-4-26b-a4b-it → qwen3.5-4b | sham2 | 0.7 | 49.0 | 98.3 | 2.1 | 45.0 | 0.7 | 98.3 |
| gemma-4-26b-a4b-it → qwen3.5-4b | other_world | 0.0 | 50.0 | 99.8 | 0.0 | 98.3 | 0.0 | 99.8 |
| gemma-4-26b-a4b-it → qwen3.5-4b | other_world vs SUPPLIED gold (consistency) | 96.0 | 98.9 | - | - | - | 96.0 | - |
| gemma-4-26b-a4b-it → qwen3.5-9b | none | 0.0 | 24.1 | 49.5 | 0.0 | 23.8 | 0.0 | 49.5 |
| gemma-4-26b-a4b-it → qwen3.5-9b | original | 4.9 | 54.6 | 99.8 | 11.0 | 51.0 | 4.9 | 99.8 |
| gemma-4-26b-a4b-it → qwen3.5-9b | selected | 93.4 | 98.2 | 99.3 | 96.3 | 97.0 | 93.4 | 99.3 |
| gemma-4-26b-a4b-it → qwen3.5-9b | packet_text | 94.7 | 98.4 | 98.6 | 97.2 | 97.5 | 94.7 | 98.6 |
| gemma-4-26b-a4b-it → qwen3.5-9b | sham0 | 4.2 | 45.9 | 90.3 | 10.7 | 34.8 | 4.2 | 90.3 |
| gemma-4-26b-a4b-it → qwen3.5-9b | sham1 | 2.5 | 46.7 | 97.4 | 7.0 | 32.9 | 2.5 | 97.4 |
| gemma-4-26b-a4b-it → qwen3.5-9b | sham2 | 2.1 | 47.7 | 94.0 | 8.9 | 37.6 | 2.1 | 94.0 |
| gemma-4-26b-a4b-it → qwen3.5-9b | other_world | 0.0 | 49.7 | 99.3 | 0.0 | 97.0 | 0.0 | 99.3 |
| gemma-4-26b-a4b-it → qwen3.5-9b | other_world vs SUPPLIED gold (consistency) | 93.4 | 98.2 | - | - | - | 93.4 | - |
| gemma-4-26b-a4b-it → qwen3.5-27b | none | 0.0 | 50.0 | 99.9 | 0.0 | 50.0 | 0.0 | 99.9 |
| gemma-4-26b-a4b-it → qwen3.5-27b | original | 77.8 | 92.7 | 99.4 | 84.5 | 90.6 | 77.8 | 99.4 |
| gemma-4-26b-a4b-it → qwen3.5-27b | selected | 98.3 | 99.4 | 99.5 | 99.1 | 99.1 | 98.3 | 99.5 |
| gemma-4-26b-a4b-it → qwen3.5-27b | packet_text | 98.6 | 99.5 | 99.6 | 99.7 | 99.0 | 98.6 | 99.6 |
| gemma-4-26b-a4b-it → qwen3.5-27b | sham0 | 2.1 | 41.4 | 76.6 | 12.4 | 31.1 | 2.1 | 76.6 |
| gemma-4-26b-a4b-it → qwen3.5-27b | sham1 | 3.6 | 46.0 | 94.5 | 8.8 | 36.6 | 3.6 | 94.5 |
| gemma-4-26b-a4b-it → qwen3.5-27b | sham2 | 1.1 | 41.6 | 87.9 | 4.2 | 31.5 | 1.1 | 87.9 |
| gemma-4-26b-a4b-it → qwen3.5-27b | other_world | 0.0 | 49.7 | 99.5 | 0.0 | 99.1 | 0.0 | 99.5 |
| gemma-4-26b-a4b-it → qwen3.5-27b | other_world vs SUPPLIED gold (consistency) | 98.3 | 99.4 | - | - | - | 98.3 | - |
| gemma-4-26b-a4b-it → qwen3.6-27b | none | 0.0 | 49.6 | 100.0 | 0.0 | 49.1 | 0.0 | 100.0 |
| gemma-4-26b-a4b-it → qwen3.6-27b | original | 82.5 | 94.3 | 97.5 | 89.9 | 91.3 | 82.5 | 97.5 |
| gemma-4-26b-a4b-it → qwen3.6-27b | selected | 97.7 | 99.3 | 99.9 | 98.9 | 98.8 | 97.7 | 99.9 |
| gemma-4-26b-a4b-it → qwen3.6-27b | packet_text | 98.2 | 99.4 | 100.0 | 99.5 | 98.8 | 98.2 | 100.0 |
| gemma-4-26b-a4b-it → qwen3.6-27b | sham0 | 2.6 | 45.8 | 90.5 | 15.6 | 31.7 | 2.6 | 90.5 |
| gemma-4-26b-a4b-it → qwen3.6-27b | sham1 | 3.7 | 46.5 | 97.3 | 13.1 | 32.7 | 3.7 | 97.3 |
| gemma-4-26b-a4b-it → qwen3.6-27b | sham2 | 2.6 | 45.6 | 94.1 | 7.7 | 32.0 | 2.6 | 94.1 |
| gemma-4-26b-a4b-it → qwen3.6-27b | other_world | 0.0 | 49.8 | 99.9 | 0.0 | 98.8 | 0.0 | 99.9 |
| gemma-4-26b-a4b-it → qwen3.6-27b | other_world vs SUPPLIED gold (consistency) | 97.7 | 99.3 | - | - | - | 97.7 | - |
| gemma-4-26b-a4b-it → gpt-oss-20b | none | 0.0 | 49.3 | 97.0 | 0.0 | 49.9 | 0.0 | 97.0 |
| gemma-4-26b-a4b-it → gpt-oss-20b | original | 4.5 | 54.1 | 99.9 | 8.6 | 53.2 | 4.5 | 99.9 |
| gemma-4-26b-a4b-it → gpt-oss-20b | selected | 85.3 | 95.3 | 99.9 | 90.4 | 94.1 | 85.3 | 99.9 |
| gemma-4-26b-a4b-it → gpt-oss-20b | packet_text | 97.7 | 99.3 | 100.0 | 98.8 | 98.8 | 97.7 | 100.0 |
| gemma-4-26b-a4b-it → gpt-oss-20b | sham0 | 1.8 | 51.3 | 99.0 | 6.1 | 47.4 | 1.8 | 99.0 |
| gemma-4-26b-a4b-it → gpt-oss-20b | sham1 | 1.7 | 51.2 | 99.3 | 4.9 | 48.1 | 1.7 | 99.3 |
| gemma-4-26b-a4b-it → gpt-oss-20b | sham2 | 1.6 | 50.1 | 99.3 | 4.0 | 45.5 | 1.6 | 99.3 |
| gemma-4-26b-a4b-it → gpt-oss-20b | other_world | 0.0 | 50.0 | 99.9 | 0.0 | 94.1 | 0.0 | 99.9 |
| gemma-4-26b-a4b-it → gpt-oss-20b | other_world vs SUPPLIED gold (consistency) | 85.3 | 95.3 | - | - | - | 85.3 | - |
| gemma-4-26b-a4b-it → olmo-3-7b-instruct | none | 0.0 | 45.7 | 93.8 | 0.0 | 45.5 | 0.0 | 93.8 |
| gemma-4-26b-a4b-it → olmo-3-7b-instruct | original | 6.3 | 54.6 | 97.5 | 13.3 | 50.0 | 6.3 | 97.5 |
| gemma-4-26b-a4b-it → olmo-3-7b-instruct | selected | 76.0 | 91.3 | 93.4 | 84.4 | 86.7 | 76.0 | 93.4 |
| gemma-4-26b-a4b-it → olmo-3-7b-instruct | packet_text | 87.8 | 95.8 | 96.3 | 91.4 | 94.4 | 87.8 | 96.3 |
| gemma-4-26b-a4b-it → olmo-3-7b-instruct | sham0 | 4.0 | 49.1 | 98.4 | 10.5 | 38.4 | 4.0 | 98.4 |
| gemma-4-26b-a4b-it → olmo-3-7b-instruct | sham1 | 1.7 | 48.5 | 98.9 | 5.8 | 38.1 | 1.7 | 98.9 |
| gemma-4-26b-a4b-it → olmo-3-7b-instruct | sham2 | 3.5 | 48.3 | 98.3 | 8.2 | 37.2 | 3.5 | 98.3 |
| gemma-4-26b-a4b-it → olmo-3-7b-instruct | other_world | 0.0 | 46.4 | 93.4 | 0.0 | 86.7 | 0.0 | 93.4 |
| gemma-4-26b-a4b-it → olmo-3-7b-instruct | other_world vs SUPPLIED gold (consistency) | 76.0 | 91.3 | - | - | - | 76.0 | - |
| gemma-4-26b-a4b-it → seed-oss-36b-instruct | none | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gemma-4-26b-a4b-it → seed-oss-36b-instruct | original | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gemma-4-26b-a4b-it → seed-oss-36b-instruct | selected | 35.1 | 72.0 | 72.0 | 59.0 | 66.1 | 35.1 | 72.0 |
| gemma-4-26b-a4b-it → seed-oss-36b-instruct | packet_text | 89.5 | 96.9 | 96.9 | 93.7 | 95.8 | 89.5 | 96.9 |
| gemma-4-26b-a4b-it → seed-oss-36b-instruct | sham0 | 0.0 | 0.7 | 1.1 | 0.0 | 0.7 | 0.0 | 1.1 |
| gemma-4-26b-a4b-it → seed-oss-36b-instruct | sham1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gemma-4-26b-a4b-it → seed-oss-36b-instruct | sham2 | 0.0 | 0.7 | 1.1 | 0.3 | 0.2 | 0.0 | 1.1 |
| gemma-4-26b-a4b-it → seed-oss-36b-instruct | other_world | 0.0 | 35.6 | 72.0 | 0.0 | 66.1 | 0.0 | 72.0 |
| gemma-4-26b-a4b-it → seed-oss-36b-instruct | other_world vs SUPPLIED gold (consistency) | 35.1 | 72.0 | - | - | - | 35.1 | - |
| gemma-4-26b-a4b-it → mistral-small-3.2-24b | none | 0.0 | 38.0 | 75.7 | 0.0 | 38.0 | 0.0 | 75.7 |
| gemma-4-26b-a4b-it → mistral-small-3.2-24b | original | 0.7 | 50.3 | 100.0 | 1.2 | 49.5 | 0.7 | 100.0 |
| gemma-4-26b-a4b-it → mistral-small-3.2-24b | selected | 96.9 | 99.1 | 100.0 | 98.6 | 98.3 | 96.9 | 100.0 |
| gemma-4-26b-a4b-it → mistral-small-3.2-24b | packet_text | 96.7 | 99.1 | 99.7 | 98.1 | 98.6 | 96.7 | 99.7 |
| gemma-4-26b-a4b-it → mistral-small-3.2-24b | sham0 | 1.6 | 41.6 | 85.5 | 7.2 | 30.8 | 1.6 | 85.5 |
| gemma-4-26b-a4b-it → mistral-small-3.2-24b | sham1 | 1.7 | 42.2 | 86.4 | 5.0 | 31.3 | 1.7 | 86.4 |
| gemma-4-26b-a4b-it → mistral-small-3.2-24b | sham2 | 2.3 | 47.8 | 96.2 | 5.6 | 39.0 | 2.3 | 96.2 |
| gemma-4-26b-a4b-it → mistral-small-3.2-24b | other_world | 0.0 | 49.8 | 100.0 | 0.0 | 98.3 | 0.0 | 100.0 |
| gemma-4-26b-a4b-it → mistral-small-3.2-24b | other_world vs SUPPLIED gold (consistency) | 96.9 | 99.1 | - | - | - | 96.9 | - |
| gemma-4-26b-a4b-it → granite-4.1-8b | none | 0.0 | 0.4 | 1.0 | 0.0 | 0.2 | 0.0 | 1.0 |
| gemma-4-26b-a4b-it → granite-4.1-8b | original | 14.8 | 59.5 | 59.9 | 33.8 | 50.2 | 14.8 | 59.9 |
| gemma-4-26b-a4b-it → granite-4.1-8b | selected | 92.0 | 97.5 | 97.5 | 95.5 | 96.5 | 92.0 | 97.5 |
| gemma-4-26b-a4b-it → granite-4.1-8b | packet_text | 94.2 | 98.2 | 98.2 | 96.8 | 97.4 | 94.2 | 98.2 |
| gemma-4-26b-a4b-it → granite-4.1-8b | sham0 | 0.2 | 6.5 | 10.7 | 3.0 | 3.3 | 0.2 | 10.7 |
| gemma-4-26b-a4b-it → granite-4.1-8b | sham1 | 0.0 | 6.6 | 13.1 | 0.5 | 2.6 | 0.0 | 13.1 |
| gemma-4-26b-a4b-it → granite-4.1-8b | sham2 | 0.0 | 5.2 | 13.3 | 0.2 | 0.3 | 0.0 | 13.3 |
| gemma-4-26b-a4b-it → granite-4.1-8b | other_world | 0.0 | 48.7 | 97.5 | 0.0 | 96.5 | 0.0 | 97.5 |
| gemma-4-26b-a4b-it → granite-4.1-8b | other_world vs SUPPLIED gold (consistency) | 92.0 | 97.5 | - | - | - | 92.0 | - |
| qwen3.6-27b → qwen3.5-2b | none | 0.0 | 50.3 | 100.0 | 0.0 | 50.7 | 0.0 | 100.0 |
| qwen3.6-27b → qwen3.5-2b | original | 4.0 | 54.1 | 99.8 | 8.0 | 51.9 | 4.0 | 99.8 |
| qwen3.6-27b → qwen3.5-2b | selected | 79.3 | 93.6 | 99.7 | 86.8 | 91.3 | 79.3 | 99.7 |
| qwen3.6-27b → qwen3.5-2b | packet_text | 83.8 | 95.0 | 99.7 | 89.4 | 93.4 | 83.8 | 99.7 |
| qwen3.6-27b → qwen3.5-2b | sham0 | 2.1 | 50.3 | 99.2 | 5.2 | 46.0 | 2.1 | 99.2 |
| qwen3.6-27b → qwen3.5-2b | sham1 | 3.0 | 49.4 | 98.8 | 4.9 | 45.7 | 3.0 | 98.8 |
| qwen3.6-27b → qwen3.5-2b | sham2 | 1.4 | 49.3 | 99.0 | 3.7 | 43.5 | 1.4 | 99.0 |
| qwen3.6-27b → qwen3.5-2b | other_world | 0.0 | 50.1 | 99.7 | 0.0 | 91.3 | 0.0 | 99.7 |
| qwen3.6-27b → qwen3.5-2b | other_world vs SUPPLIED gold (consistency) | 79.3 | 93.6 | - | - | - | 79.3 | - |
| qwen3.6-27b → qwen3.5-4b | none | 0.0 | 49.9 | 99.8 | 0.0 | 50.0 | 0.0 | 99.8 |
| qwen3.6-27b → qwen3.5-4b | original | 3.1 | 52.3 | 99.7 | 4.9 | 50.6 | 3.1 | 99.7 |
| qwen3.6-27b → qwen3.5-4b | selected | 93.4 | 98.1 | 99.8 | 95.8 | 97.4 | 93.4 | 99.8 |
| qwen3.6-27b → qwen3.5-4b | packet_text | 94.3 | 98.4 | 99.7 | 96.5 | 97.7 | 94.3 | 99.7 |
| qwen3.6-27b → qwen3.5-4b | sham0 | 1.9 | 48.4 | 95.8 | 4.7 | 46.0 | 1.9 | 95.8 |
| qwen3.6-27b → qwen3.5-4b | sham1 | 2.3 | 50.1 | 99.0 | 4.6 | 45.5 | 2.3 | 99.0 |
| qwen3.6-27b → qwen3.5-4b | sham2 | 0.7 | 49.3 | 98.2 | 2.5 | 45.8 | 0.7 | 98.2 |
| qwen3.6-27b → qwen3.5-4b | other_world | 0.0 | 50.0 | 99.8 | 0.0 | 97.4 | 0.0 | 99.8 |
| qwen3.6-27b → qwen3.5-4b | other_world vs SUPPLIED gold (consistency) | 93.4 | 98.1 | - | - | - | 93.4 | - |
| qwen3.6-27b → qwen3.5-9b | none | 0.0 | 24.1 | 49.5 | 0.0 | 23.8 | 0.0 | 49.5 |
| qwen3.6-27b → qwen3.5-9b | original | 4.9 | 54.6 | 99.8 | 11.2 | 50.7 | 4.9 | 99.8 |
| qwen3.6-27b → qwen3.5-9b | selected | 93.6 | 98.2 | 99.7 | 96.2 | 97.4 | 93.6 | 99.7 |
| qwen3.6-27b → qwen3.5-9b | packet_text | 96.2 | 98.8 | 99.3 | 97.6 | 98.6 | 96.2 | 99.3 |
| qwen3.6-27b → qwen3.5-9b | sham0 | 3.5 | 45.9 | 90.5 | 10.3 | 34.5 | 3.5 | 90.5 |
| qwen3.6-27b → qwen3.5-9b | sham1 | 3.0 | 47.4 | 97.5 | 8.1 | 35.0 | 3.0 | 97.5 |
| qwen3.6-27b → qwen3.5-9b | sham2 | 2.1 | 46.9 | 93.3 | 8.7 | 37.9 | 2.1 | 93.3 |
| qwen3.6-27b → qwen3.5-9b | other_world | 0.0 | 49.9 | 99.7 | 0.0 | 97.4 | 0.0 | 99.7 |
| qwen3.6-27b → qwen3.5-9b | other_world vs SUPPLIED gold (consistency) | 93.6 | 98.2 | - | - | - | 93.6 | - |
| qwen3.6-27b → qwen3.5-27b | none | 0.0 | 50.0 | 99.9 | 0.0 | 50.0 | 0.0 | 99.9 |
| qwen3.6-27b → qwen3.5-27b | original | 75.7 | 92.2 | 99.3 | 83.1 | 89.9 | 75.7 | 99.3 |
| qwen3.6-27b → qwen3.5-27b | selected | 96.4 | 99.0 | 99.9 | 97.9 | 98.4 | 96.4 | 99.9 |
| qwen3.6-27b → qwen3.5-27b | packet_text | 96.7 | 99.1 | 100.0 | 98.4 | 98.3 | 96.7 | 100.0 |
| qwen3.6-27b → qwen3.5-27b | sham0 | 1.9 | 41.5 | 77.8 | 12.3 | 31.1 | 1.9 | 77.8 |
| qwen3.6-27b → qwen3.5-27b | sham1 | 3.2 | 45.3 | 93.3 | 7.4 | 36.8 | 3.2 | 93.3 |
| qwen3.6-27b → qwen3.5-27b | sham2 | 0.9 | 40.4 | 86.5 | 3.3 | 32.2 | 0.9 | 86.5 |
| qwen3.6-27b → qwen3.5-27b | other_world | 0.0 | 50.0 | 99.9 | 0.0 | 98.4 | 0.0 | 99.9 |
| qwen3.6-27b → qwen3.5-27b | other_world vs SUPPLIED gold (consistency) | 96.4 | 99.0 | - | - | - | 96.4 | - |
| qwen3.6-27b → qwen3.6-27b | none | 0.0 | 49.6 | 100.0 | 0.0 | 49.1 | 0.0 | 100.0 |
| qwen3.6-27b → qwen3.6-27b | original | 77.1 | 92.8 | 97.1 | 86.7 | 89.3 | 77.1 | 97.1 |
| qwen3.6-27b → qwen3.6-27b | selected | 96.0 | 99.0 | 100.0 | 97.7 | 98.3 | 96.0 | 100.0 |
| qwen3.6-27b → qwen3.6-27b | packet_text | 96.0 | 99.0 | 100.0 | 97.7 | 98.3 | 96.0 | 100.0 |
| qwen3.6-27b → qwen3.6-27b | sham0 | 2.6 | 45.8 | 90.3 | 15.4 | 31.5 | 2.6 | 90.3 |
| qwen3.6-27b → qwen3.6-27b | sham1 | 3.7 | 45.2 | 96.2 | 11.5 | 32.4 | 3.7 | 96.2 |
| qwen3.6-27b → qwen3.6-27b | sham2 | 2.1 | 44.2 | 93.2 | 7.5 | 32.2 | 2.1 | 93.2 |
| qwen3.6-27b → qwen3.6-27b | other_world | 0.0 | 50.1 | 100.0 | 0.0 | 98.3 | 0.0 | 100.0 |
| qwen3.6-27b → qwen3.6-27b | other_world vs SUPPLIED gold (consistency) | 96.0 | 99.0 | - | - | - | 96.0 | - |
| qwen3.6-27b → gpt-oss-20b | none | 0.0 | 49.3 | 97.0 | 0.0 | 49.9 | 0.0 | 97.0 |
| qwen3.6-27b → gpt-oss-20b | original | 4.7 | 53.9 | 99.9 | 8.4 | 52.5 | 4.7 | 99.9 |
| qwen3.6-27b → gpt-oss-20b | selected | 83.8 | 94.7 | 99.9 | 89.2 | 93.7 | 83.8 | 99.9 |
| qwen3.6-27b → gpt-oss-20b | packet_text | 95.8 | 98.8 | 100.0 | 97.4 | 98.3 | 95.8 | 100.0 |
| qwen3.6-27b → gpt-oss-20b | sham0 | 1.2 | 50.4 | 98.4 | 5.8 | 46.5 | 1.2 | 98.4 |
| qwen3.6-27b → gpt-oss-20b | sham1 | 1.7 | 51.3 | 99.3 | 5.6 | 47.6 | 1.7 | 99.3 |
| qwen3.6-27b → gpt-oss-20b | sham2 | 1.7 | 50.7 | 99.7 | 3.7 | 46.2 | 1.7 | 99.7 |
| qwen3.6-27b → gpt-oss-20b | other_world | 0.0 | 50.1 | 99.9 | 0.0 | 93.7 | 0.0 | 99.9 |
| qwen3.6-27b → gpt-oss-20b | other_world vs SUPPLIED gold (consistency) | 83.8 | 94.7 | - | - | - | 83.8 | - |
| qwen3.6-27b → olmo-3-7b-instruct | none | 0.0 | 45.7 | 93.8 | 0.0 | 45.5 | 0.0 | 93.8 |
| qwen3.6-27b → olmo-3-7b-instruct | original | 7.0 | 54.4 | 97.5 | 13.8 | 49.7 | 7.0 | 97.5 |
| qwen3.6-27b → olmo-3-7b-instruct | selected | 72.0 | 90.1 | 93.2 | 82.2 | 84.9 | 72.0 | 93.2 |
| qwen3.6-27b → olmo-3-7b-instruct | packet_text | 86.1 | 95.4 | 96.8 | 91.1 | 93.4 | 86.1 | 96.8 |
| qwen3.6-27b → olmo-3-7b-instruct | sham0 | 4.4 | 48.9 | 98.3 | 10.1 | 37.9 | 4.4 | 98.3 |
| qwen3.6-27b → olmo-3-7b-instruct | sham1 | 2.1 | 48.8 | 98.8 | 5.1 | 40.6 | 2.1 | 98.8 |
| qwen3.6-27b → olmo-3-7b-instruct | sham2 | 3.3 | 48.5 | 98.2 | 8.2 | 39.0 | 3.3 | 98.2 |
| qwen3.6-27b → olmo-3-7b-instruct | other_world | 0.0 | 46.2 | 93.2 | 0.0 | 84.9 | 0.0 | 93.2 |
| qwen3.6-27b → olmo-3-7b-instruct | other_world vs SUPPLIED gold (consistency) | 72.0 | 90.1 | - | - | - | 72.0 | - |
| qwen3.6-27b → seed-oss-36b-instruct | none | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| qwen3.6-27b → seed-oss-36b-instruct | original | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| qwen3.6-27b → seed-oss-36b-instruct | selected | 36.0 | 71.9 | 71.9 | 59.2 | 65.6 | 36.0 | 71.9 |
| qwen3.6-27b → seed-oss-36b-instruct | packet_text | 87.5 | 96.1 | 96.1 | 92.7 | 94.8 | 87.5 | 96.1 |
| qwen3.6-27b → seed-oss-36b-instruct | sham0 | 0.0 | 0.7 | 1.1 | 0.0 | 0.7 | 0.0 | 1.1 |
| qwen3.6-27b → seed-oss-36b-instruct | sham1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| qwen3.6-27b → seed-oss-36b-instruct | sham2 | 0.0 | 1.5 | 2.7 | 0.2 | 1.1 | 0.0 | 2.7 |
| qwen3.6-27b → seed-oss-36b-instruct | other_world | 0.0 | 35.5 | 71.9 | 0.0 | 65.6 | 0.0 | 71.9 |
| qwen3.6-27b → seed-oss-36b-instruct | other_world vs SUPPLIED gold (consistency) | 36.0 | 71.9 | - | - | - | 36.0 | - |
| qwen3.6-27b → mistral-small-3.2-24b | none | 0.0 | 38.0 | 75.7 | 0.0 | 38.0 | 0.0 | 75.7 |
| qwen3.6-27b → mistral-small-3.2-24b | original | 0.9 | 50.2 | 100.0 | 1.4 | 49.5 | 0.9 | 100.0 |
| qwen3.6-27b → mistral-small-3.2-24b | selected | 94.1 | 98.4 | 100.0 | 96.9 | 97.2 | 94.1 | 100.0 |
| qwen3.6-27b → mistral-small-3.2-24b | packet_text | 96.0 | 98.7 | 99.6 | 97.9 | 98.1 | 96.0 | 99.6 |
| qwen3.6-27b → mistral-small-3.2-24b | sham0 | 1.2 | 40.9 | 83.4 | 6.7 | 29.7 | 1.2 | 83.4 |
| qwen3.6-27b → mistral-small-3.2-24b | sham1 | 1.6 | 39.0 | 80.5 | 4.2 | 29.4 | 1.6 | 80.5 |
| qwen3.6-27b → mistral-small-3.2-24b | sham2 | 2.6 | 48.3 | 95.5 | 6.5 | 41.4 | 2.6 | 95.5 |
| qwen3.6-27b → mistral-small-3.2-24b | other_world | 0.0 | 49.9 | 100.0 | 0.0 | 97.2 | 0.0 | 100.0 |
| qwen3.6-27b → mistral-small-3.2-24b | other_world vs SUPPLIED gold (consistency) | 94.1 | 98.4 | - | - | - | 94.1 | - |
| qwen3.6-27b → granite-4.1-8b | none | 0.0 | 0.4 | 1.0 | 0.0 | 0.2 | 0.0 | 1.0 |
| qwen3.6-27b → granite-4.1-8b | original | 14.7 | 58.4 | 58.8 | 33.8 | 47.8 | 14.7 | 58.8 |
| qwen3.6-27b → granite-4.1-8b | selected | 90.6 | 97.2 | 97.6 | 95.0 | 95.7 | 90.6 | 97.6 |
| qwen3.6-27b → granite-4.1-8b | packet_text | 94.4 | 98.3 | 98.6 | 97.2 | 97.2 | 94.4 | 98.6 |
| qwen3.6-27b → granite-4.1-8b | sham0 | 0.0 | 5.5 | 9.8 | 1.8 | 3.0 | 0.0 | 9.8 |
| qwen3.6-27b → granite-4.1-8b | sham1 | 0.0 | 6.9 | 13.6 | 0.9 | 3.5 | 0.0 | 13.6 |
| qwen3.6-27b → granite-4.1-8b | sham2 | 0.0 | 5.8 | 15.3 | 0.5 | 1.4 | 0.0 | 15.3 |
| qwen3.6-27b → granite-4.1-8b | other_world | 0.0 | 48.8 | 97.6 | 0.0 | 95.7 | 0.0 | 97.6 |
| qwen3.6-27b → granite-4.1-8b | other_world vs SUPPLIED gold (consistency) | 90.6 | 97.2 | - | - | - | 90.6 | - |
| gpt-oss-20b → qwen3.5-2b | none | 0.0 | 50.3 | 100.0 | 0.0 | 50.7 | 0.0 | 100.0 |
| gpt-oss-20b → qwen3.5-2b | original | 4.2 | 54.4 | 99.9 | 8.9 | 51.9 | 4.2 | 99.9 |
| gpt-oss-20b → qwen3.5-2b | selected | 77.5 | 93.0 | 100.0 | 85.3 | 90.9 | 77.5 | 100.0 |
| gpt-oss-20b → qwen3.5-2b | packet_text | 81.5 | 94.5 | 100.0 | 87.8 | 92.7 | 81.5 | 100.0 |
| gpt-oss-20b → qwen3.5-2b | sham0 | 2.6 | 51.1 | 100.0 | 7.2 | 45.1 | 2.6 | 100.0 |
| gpt-oss-20b → qwen3.5-2b | sham1 | 2.3 | 50.2 | 99.6 | 5.4 | 44.8 | 2.3 | 99.6 |
| gpt-oss-20b → qwen3.5-2b | sham2 | 1.9 | 50.4 | 100.0 | 4.6 | 44.6 | 1.9 | 100.0 |
| gpt-oss-20b → qwen3.5-2b | other_world | 0.0 | 50.3 | 100.0 | 0.0 | 90.9 | 0.0 | 100.0 |
| gpt-oss-20b → qwen3.5-2b | other_world vs SUPPLIED gold (consistency) | 77.5 | 93.0 | - | - | - | 77.5 | - |
| gpt-oss-20b → qwen3.5-4b | none | 0.0 | 49.9 | 99.8 | 0.0 | 50.0 | 0.0 | 99.8 |
| gpt-oss-20b → qwen3.5-4b | original | 2.8 | 52.4 | 99.9 | 4.7 | 50.4 | 2.8 | 99.9 |
| gpt-oss-20b → qwen3.5-4b | selected | 89.9 | 96.9 | 99.5 | 93.0 | 95.8 | 89.9 | 99.5 |
| gpt-oss-20b → qwen3.5-4b | packet_text | 88.0 | 96.3 | 98.5 | 92.5 | 94.8 | 88.0 | 98.5 |
| gpt-oss-20b → qwen3.5-4b | sham0 | 2.3 | 49.5 | 97.9 | 4.4 | 47.2 | 2.3 | 97.9 |
| gpt-oss-20b → qwen3.5-4b | sham1 | 1.9 | 49.6 | 99.5 | 3.7 | 43.7 | 1.9 | 99.5 |
| gpt-oss-20b → qwen3.5-4b | sham2 | 0.5 | 48.7 | 98.1 | 1.9 | 44.1 | 0.5 | 98.1 |
| gpt-oss-20b → qwen3.5-4b | other_world | 0.2 | 50.2 | 99.5 | 0.2 | 95.8 | 0.2 | 99.5 |
| gpt-oss-20b → qwen3.5-4b | other_world vs SUPPLIED gold (consistency) | 89.9 | 96.9 | - | - | - | 89.9 | - |
| gpt-oss-20b → qwen3.5-9b | none | 0.0 | 24.1 | 49.5 | 0.0 | 23.8 | 0.0 | 49.5 |
| gpt-oss-20b → qwen3.5-9b | original | 4.9 | 54.8 | 99.4 | 11.0 | 50.9 | 4.9 | 99.4 |
| gpt-oss-20b → qwen3.5-9b | selected | 86.6 | 95.5 | 97.9 | 90.9 | 94.8 | 86.6 | 97.9 |
| gpt-oss-20b → qwen3.5-9b | packet_text | 88.5 | 96.0 | 97.0 | 92.8 | 94.9 | 88.5 | 97.0 |
| gpt-oss-20b → qwen3.5-9b | sham0 | 2.8 | 46.4 | 92.4 | 10.5 | 35.0 | 2.8 | 92.4 |
| gpt-oss-20b → qwen3.5-9b | sham1 | 3.3 | 46.6 | 96.4 | 8.0 | 32.4 | 3.3 | 96.4 |
| gpt-oss-20b → qwen3.5-9b | sham2 | 2.1 | 46.6 | 93.4 | 8.0 | 35.5 | 2.1 | 93.4 |
| gpt-oss-20b → qwen3.5-9b | other_world | 0.0 | 49.6 | 97.9 | 0.0 | 94.8 | 0.0 | 97.9 |
| gpt-oss-20b → qwen3.5-9b | other_world vs SUPPLIED gold (consistency) | 86.6 | 95.5 | - | - | - | 86.6 | - |
| gpt-oss-20b → qwen3.5-27b | none | 0.0 | 50.0 | 99.9 | 0.0 | 50.0 | 0.0 | 99.9 |
| gpt-oss-20b → qwen3.5-27b | original | 67.4 | 89.2 | 98.1 | 77.5 | 85.9 | 67.4 | 98.1 |
| gpt-oss-20b → qwen3.5-27b | selected | 88.8 | 96.4 | 98.3 | 93.2 | 95.3 | 88.8 | 98.3 |
| gpt-oss-20b → qwen3.5-27b | packet_text | 89.1 | 96.5 | 98.4 | 93.5 | 95.3 | 89.1 | 98.4 |
| gpt-oss-20b → qwen3.5-27b | sham0 | 2.5 | 43.8 | 81.6 | 11.1 | 33.1 | 2.5 | 81.6 |
| gpt-oss-20b → qwen3.5-27b | sham1 | 2.1 | 44.3 | 92.9 | 5.8 | 34.0 | 2.1 | 92.9 |
| gpt-oss-20b → qwen3.5-27b | sham2 | 0.9 | 41.8 | 87.8 | 3.7 | 32.2 | 0.9 | 87.8 |
| gpt-oss-20b → qwen3.5-27b | other_world | 0.0 | 49.4 | 98.3 | 0.0 | 95.3 | 0.0 | 98.3 |
| gpt-oss-20b → qwen3.5-27b | other_world vs SUPPLIED gold (consistency) | 88.8 | 96.4 | - | - | - | 88.8 | - |
| gpt-oss-20b → qwen3.6-27b | none | 0.0 | 49.6 | 100.0 | 0.0 | 49.1 | 0.0 | 100.0 |
| gpt-oss-20b → qwen3.6-27b | original | 74.2 | 91.6 | 96.2 | 84.9 | 87.1 | 74.2 | 96.2 |
| gpt-oss-20b → qwen3.6-27b | selected | 90.6 | 97.1 | 99.2 | 94.1 | 96.1 | 90.6 | 99.2 |
| gpt-oss-20b → qwen3.6-27b | packet_text | 89.5 | 96.5 | 98.7 | 93.5 | 95.6 | 89.5 | 98.7 |
| gpt-oss-20b → qwen3.6-27b | sham0 | 3.1 | 46.5 | 92.9 | 16.5 | 31.1 | 3.1 | 92.9 |
| gpt-oss-20b → qwen3.6-27b | sham1 | 2.6 | 46.1 | 97.3 | 11.2 | 31.0 | 2.6 | 97.3 |
| gpt-oss-20b → qwen3.6-27b | sham2 | 3.0 | 45.6 | 93.0 | 7.7 | 30.9 | 3.0 | 93.0 |
| gpt-oss-20b → qwen3.6-27b | other_world | 0.0 | 50.0 | 99.2 | 0.0 | 96.1 | 0.0 | 99.2 |
| gpt-oss-20b → qwen3.6-27b | other_world vs SUPPLIED gold (consistency) | 90.6 | 97.1 | - | - | - | 90.6 | - |
| gpt-oss-20b → gpt-oss-20b | none | 0.0 | 49.3 | 97.0 | 0.0 | 49.9 | 0.0 | 97.0 |
| gpt-oss-20b → gpt-oss-20b | original | 5.1 | 54.7 | 99.9 | 9.6 | 53.4 | 5.1 | 99.9 |
| gpt-oss-20b → gpt-oss-20b | selected | 77.5 | 92.9 | 99.8 | 85.7 | 89.9 | 77.5 | 99.8 |
| gpt-oss-20b → gpt-oss-20b | packet_text | 90.9 | 97.1 | 99.8 | 93.9 | 96.2 | 90.9 | 99.8 |
| gpt-oss-20b → gpt-oss-20b | sham0 | 1.6 | 50.5 | 99.2 | 6.3 | 45.0 | 1.6 | 99.2 |
| gpt-oss-20b → gpt-oss-20b | sham1 | 2.1 | 50.7 | 99.2 | 3.8 | 47.4 | 2.1 | 99.2 |
| gpt-oss-20b → gpt-oss-20b | sham2 | 1.4 | 50.0 | 98.7 | 4.0 | 45.0 | 1.4 | 98.7 |
| gpt-oss-20b → gpt-oss-20b | other_world | 0.0 | 50.0 | 99.8 | 0.2 | 89.9 | 0.0 | 99.8 |
| gpt-oss-20b → gpt-oss-20b | other_world vs SUPPLIED gold (consistency) | 77.5 | 92.9 | - | - | - | 77.5 | - |
| gpt-oss-20b → olmo-3-7b-instruct | none | 0.0 | 45.7 | 93.8 | 0.0 | 45.5 | 0.0 | 93.8 |
| gpt-oss-20b → olmo-3-7b-instruct | original | 5.1 | 53.4 | 97.2 | 11.6 | 48.8 | 5.1 | 97.2 |
| gpt-oss-20b → olmo-3-7b-instruct | selected | 77.1 | 92.5 | 96.6 | 86.0 | 87.7 | 77.1 | 96.6 |
| gpt-oss-20b → olmo-3-7b-instruct | packet_text | 84.6 | 94.7 | 97.6 | 90.0 | 92.7 | 84.6 | 97.6 |
| gpt-oss-20b → olmo-3-7b-instruct | sham0 | 2.8 | 49.9 | 98.7 | 10.7 | 39.0 | 2.8 | 98.7 |
| gpt-oss-20b → olmo-3-7b-instruct | sham1 | 1.7 | 48.0 | 98.7 | 5.8 | 37.2 | 1.7 | 98.7 |
| gpt-oss-20b → olmo-3-7b-instruct | sham2 | 3.5 | 48.8 | 98.5 | 9.8 | 36.9 | 3.5 | 98.5 |
| gpt-oss-20b → olmo-3-7b-instruct | other_world | 0.0 | 48.1 | 96.6 | 0.0 | 87.7 | 0.0 | 96.6 |
| gpt-oss-20b → olmo-3-7b-instruct | other_world vs SUPPLIED gold (consistency) | 77.1 | 92.5 | - | - | - | 77.1 | - |
| gpt-oss-20b → seed-oss-36b-instruct | none | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gpt-oss-20b → seed-oss-36b-instruct | original | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gpt-oss-20b → seed-oss-36b-instruct | selected | 24.7 | 65.9 | 65.9 | 50.8 | 60.5 | 24.7 | 65.9 |
| gpt-oss-20b → seed-oss-36b-instruct | packet_text | 82.5 | 93.7 | 93.7 | 88.1 | 92.8 | 82.5 | 93.7 |
| gpt-oss-20b → seed-oss-36b-instruct | sham0 | 0.0 | 1.0 | 1.6 | 0.2 | 0.7 | 0.0 | 1.6 |
| gpt-oss-20b → seed-oss-36b-instruct | sham1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gpt-oss-20b → seed-oss-36b-instruct | sham2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gpt-oss-20b → seed-oss-36b-instruct | other_world | 0.0 | 33.0 | 65.9 | 0.0 | 60.5 | 0.0 | 65.9 |
| gpt-oss-20b → seed-oss-36b-instruct | other_world vs SUPPLIED gold (consistency) | 24.7 | 65.9 | - | - | - | 24.7 | - |
| gpt-oss-20b → mistral-small-3.2-24b | none | 0.0 | 38.0 | 75.7 | 0.0 | 38.0 | 0.0 | 75.7 |
| gpt-oss-20b → mistral-small-3.2-24b | original | 0.7 | 50.3 | 99.6 | 1.2 | 49.3 | 0.7 | 99.6 |
| gpt-oss-20b → mistral-small-3.2-24b | selected | 89.0 | 96.6 | 99.5 | 92.9 | 95.1 | 89.0 | 99.5 |
| gpt-oss-20b → mistral-small-3.2-24b | packet_text | 88.8 | 96.2 | 98.3 | 93.0 | 94.9 | 88.8 | 98.3 |
| gpt-oss-20b → mistral-small-3.2-24b | sham0 | 1.9 | 45.3 | 91.1 | 8.1 | 35.0 | 1.9 | 91.1 |
| gpt-oss-20b → mistral-small-3.2-24b | sham1 | 2.1 | 43.8 | 89.5 | 5.8 | 32.5 | 2.1 | 89.5 |
| gpt-oss-20b → mistral-small-3.2-24b | sham2 | 1.9 | 47.2 | 97.3 | 5.1 | 36.2 | 1.9 | 97.3 |
| gpt-oss-20b → mistral-small-3.2-24b | other_world | 0.2 | 50.2 | 99.5 | 0.2 | 95.1 | 0.2 | 99.5 |
| gpt-oss-20b → mistral-small-3.2-24b | other_world vs SUPPLIED gold (consistency) | 89.0 | 96.6 | - | - | - | 89.0 | - |
| gpt-oss-20b → granite-4.1-8b | none | 0.0 | 0.4 | 1.0 | 0.0 | 0.2 | 0.0 | 1.0 |
| gpt-oss-20b → granite-4.1-8b | original | 12.6 | 57.2 | 57.6 | 31.7 | 46.9 | 12.6 | 57.6 |
| gpt-oss-20b → granite-4.1-8b | selected | 83.1 | 93.9 | 94.3 | 89.2 | 92.1 | 83.1 | 94.3 |
| gpt-oss-20b → granite-4.1-8b | packet_text | 86.0 | 94.8 | 94.9 | 90.5 | 94.1 | 86.0 | 94.9 |
| gpt-oss-20b → granite-4.1-8b | sham0 | 0.2 | 7.3 | 12.2 | 3.7 | 4.0 | 0.2 | 12.2 |
| gpt-oss-20b → granite-4.1-8b | sham1 | 0.0 | 5.7 | 12.3 | 0.0 | 1.8 | 0.0 | 12.3 |
| gpt-oss-20b → granite-4.1-8b | sham2 | 0.0 | 4.9 | 11.1 | 0.2 | 0.3 | 0.0 | 11.1 |
| gpt-oss-20b → granite-4.1-8b | other_world | 0.0 | 47.5 | 94.3 | 0.0 | 92.1 | 0.0 | 94.3 |
| gpt-oss-20b → granite-4.1-8b | other_world vs SUPPLIED gold (consistency) | 83.1 | 93.9 | - | - | - | 83.1 | - |

## Resolutions of lock ambiguities (stated, not silently chosen)

1. Parent templates: the bank field `template` takes only 4 values (0-3); the lock says 12 parent templates. The 12 clusters are domain x template (`{domain}/T{template}`), the convention of every earlier audit of this bank (scripts/audit_r3_20260923.py) and of Experiment B (scripts/latest0924/analyze_graded.py). Families nest in them.
2. Engineering gate: the lock text names max logit difference <= 0.05 with equal parsed answers; the pinned evaluator (the manifest pins src/syncomm/recovery/evaluate_confirm.py) also requires alias-gap difference <= 0.1. The script applies the evaluator's three-clause gate (stricter) and also reports the lock-literal two-clause count. qwen3.5-122b-a10b uses ADR-latest0924-moe-gate (parsed answers equal and alias gap <= 0.1).
3. Engineering failure is receiver-level: a failure in any sender's cell makes the receiver an engineering failure for every sender (the lock: "A receiver failing the gate is reported as an engineering failure with no outcome rows").
4. "Lower limit > 0 after Holm" (H1, H2): Holm (alpha 0.05) is applied to p = max(two-sided bootstrap p at the family level, at the template level), each p pooled over the 3 seeds x 20,000 draws; the decision additionally requires the unadjusted 95% seed-enclosing interval to exclude 0 in the stated direction at BOTH levels. This is the rule already used for Experiment B (scripts/latest0924/analyze_graded.py). Holm-adjusted p values are also reported.
5. Holm family size m: the receivers EVALUATED for that sender (complete, engineering-passed, audit-passed, cross-model). As a sensitivity, the verdict is also computed with m = every planned cross-model receiver of that sender (unevaluated receivers enter with p = 1); disagreements are listed.
6. H1 point criteria use family-macro point estimates: J(selected) >= 0.50 and coverage(selected) >= 0.90 at 64 tokens.
7. H2 order: "internal better" / "text better" are checked before "similar" (the lock lists them first); a pair that is both significant and within +/-3 points is labelled by its significant direction and flagged within_margin=true. "Similar" uses the unadjusted 95% seed-enclosing intervals at both levels, inclusive bounds [-0.03, +0.03].
8. Same-model pairs (lock same_model_pairs) are reported in their own table with unadjusted intervals and p (no Holm), are excluded from every cross-model count and Holm family, and are excluded from the H3 secondary ladder.
9. H3 secondary: gap = family-macro J(selected) - J(packet_text) at 64 tokens; Spearman with average ranks over the available ladder points (qwen3.5-27b and qwen3.6-27b tie at 27e9); computed only when >= 3 points exist. Missing points (engineering failures, incomplete cells) are listed, never imputed.
10. "Reassigned success/valid" in the main table = the range of J over sham0-sham2 / the mean coverage over sham0-sham2.
11. Cells whose development shards are not all receipted are INCOMPLETE and are reported as NOT RUN for inference (the lock's execution_order), with the rows present.

## Not claims (from the lock)

- No bandwidth, latency or privacy advantage is tested.
- The bank is finite and constructed; results are not population estimates.
- A prompted sender does not show reliable extraction from hidden states.
- Same-model pairs do not count as cross-model communication.

Same-model rule: Same-model pairs are reported separately and excluded from cross-family and cross-model claims.

Limits: worlds, questions, arms and senders are repeated measures on the same scenes and add no independent N; bootstrap seeds are not model replications; the no-message J is structurally 0; the bank is finite and constructed and was previously evaluated with four older receivers and the Llama-3.1-8B-Instruct sender (lock `bank.disclosure`).

Ignored paths (not lock cells): `artifacts/latest0924/expA/_superseded/gemma-4-26b-a4b-it__granite-4.1-8b__tf53dirs`, `artifacts/latest0924/expA/_superseded`, `artifacts/latest0924/expA/analysis`
