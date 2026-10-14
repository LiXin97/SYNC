# Experiment E: frozen-setting transfer to new layouts

Lock `gates/latest0924/expE_layout.lock.json` sha `8d513833e9b7`; Holm m = 18, non-inferiority margin 5 points.

| receiver | status | eng | L0 repro | cond | J internal | J text | J sham0 | int(c)-int(L0) [CI] | verdict | int-text [CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-32b | COMPLETE | PASS | 1152/1152 | L0 | 100.0 | 99.3 | 1.4 |  |  | +0.7 [+0.0, +2.1] |
| qwen3-32b | COMPLETE | PASS | 1152/1152 | L1 | 100.0 | 100.0 | 1.4 | +0.0 [+0.0, +0.0] | transfers | +0.0 [+0.0, +0.0] |
| qwen3-32b | COMPLETE | PASS | 1152/1152 | L2 | 98.6 | 99.3 | 0.0 | -1.4 [-3.5, +0.0] | transfers | -0.7 [-2.8, +1.4] |
| qwen3-32b | COMPLETE | PASS | 1152/1152 | L3 | 99.3 | 99.3 | 0.0 | -0.7 [-2.1, +0.0] | transfers | +0.0 [+0.0, +0.0] |
| qwen3.5-27b | COMPLETE | PASS | 1152/1152 | L0 | 100.0 | 100.0 | 2.1 |  |  | +0.0 [+0.0, +0.0] |
| qwen3.5-27b | COMPLETE | PASS | 1152/1152 | L1 | 100.0 | 100.0 | 3.5 | +0.0 [+0.0, +0.0] | transfers | +0.0 [+0.0, +0.0] |
| qwen3.5-27b | COMPLETE | PASS | 1152/1152 | L2 | 100.0 | 100.0 | 2.1 | +0.0 [+0.0, +0.0] | transfers | +0.0 [+0.0, +0.0] |
| qwen3.5-27b | COMPLETE | PASS | 1152/1152 | L3 | 100.0 | 100.0 | 1.4 | +0.0 [+0.0, +0.0] | transfers | +0.0 [+0.0, +0.0] |
| qwen3.6-27b | COMPLETE | PASS | 1152/1152 | L0 | 100.0 | 100.0 | 4.2 |  |  | +0.0 [+0.0, +0.0] |
| qwen3.6-27b | COMPLETE | PASS | 1152/1152 | L1 | 100.0 | 100.0 | 2.8 | +0.0 [+0.0, +0.0] | transfers | +0.0 [+0.0, +0.0] |
| qwen3.6-27b | COMPLETE | PASS | 1152/1152 | L2 | 100.0 | 100.0 | 5.6 | +0.0 [+0.0, +0.0] | transfers | +0.0 [+0.0, +0.0] |
| qwen3.6-27b | COMPLETE | PASS | 1152/1152 | L3 | 100.0 | 100.0 | 3.5 | +0.0 [+0.0, +0.0] | transfers | +0.0 [+0.0, +0.0] |
| mistral-small-3.2-24b | COMPLETE | PASS | 1152/1152 | L0 | 97.9 | 100.0 | 2.8 |  |  | -2.1 [-4.9, +0.0] |
| mistral-small-3.2-24b | COMPLETE | PASS | 1152/1152 | L1 | 97.2 | 100.0 | 2.1 | -0.7 [-3.5, +2.1] | transfers | -2.8 [-5.6, -0.7] |
| mistral-small-3.2-24b | COMPLETE | PASS | 1152/1152 | L2 | 89.6 | 100.0 | 2.1 | -8.3 [-15.3, -2.1] | inconclusive | -10.4 [-17.4, -4.2] |
| mistral-small-3.2-24b | COMPLETE | PASS | 1152/1152 | L3 | 95.1 | 100.0 | 2.1 | -2.8 [-8.3, +2.1] | inconclusive | -4.9 [-9.7, -0.7] |
| qwen3.5-9b | COMPLETE | PASS | 1152/1152 | L0 | 93.8 | 100.0 | 0.7 |  |  | -6.2 [-12.5, -1.4] |
| qwen3.5-9b | COMPLETE | PASS | 1152/1152 | L1 | 99.3 | 100.0 | 1.4 | +5.6 [+0.7, +11.8] | transfers | -0.7 [-2.1, +0.0] |
| qwen3.5-9b | COMPLETE | PASS | 1152/1152 | L2 | 93.8 | 100.0 | 1.4 | +0.0 [-4.9, +4.9] | inconclusive | -6.2 [-13.2, -0.7] |
| qwen3.5-9b | COMPLETE | PASS | 1152/1152 | L3 | 95.1 | 100.0 | 1.4 | +1.4 [-2.8, +6.2] | transfers | -4.9 [-9.7, -0.7] |
| granite-4.1-8b | COMPLETE | PASS | 1152/1152 | L0 | 96.5 | 100.0 | 0.0 |  |  | -3.5 [-6.9, -0.7] |
| granite-4.1-8b | COMPLETE | PASS | 1152/1152 | L1 | 99.3 | 100.0 | 0.0 | +2.8 [+0.0, +6.2] | transfers | -0.7 [-2.1, +0.0] |
| granite-4.1-8b | COMPLETE | PASS | 1152/1152 | L2 | 91.7 | 99.3 | 0.0 | -4.9 [-9.7, +0.0] | inconclusive | -7.6 [-11.8, -4.2] |
| granite-4.1-8b | COMPLETE | PASS | 1152/1152 | L3 | 97.9 | 100.0 | 0.0 | +1.4 [-2.1, +4.9] | transfers | -2.1 [-4.9, +0.0] |
