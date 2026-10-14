# Experiment B (graded weights): internal vs text — analysis

Generated 2026-09-24T16:08:19.756993+00:00 by `scripts/latest0924/analyze_graded.py` (sha256 `352a8d3bb0303db8`). Status: **PASS_INDEPENDENT_GRADED_AUDIT**. Intervals: family-level (36 families) 95% cluster bootstrap, 20,000 draws x 3 seeds, enclosing interval; contrasts also report the template level (12) and the interval enclosing both.

## Verdicts (primary: internal - text, pooled over L = 64, 128, 191; Holm across the four declared receivers)

| receiver | verdict | interpretable | estimate [enclosing 95%] (points) | p (max of levels) | Holm threshold | failed controls |
|---|---|---|---|---|---|---|
| qwen3.5-27b | internal better than text | True | 33.8 [20.8, 48.1] | 1.667e-05 | 0.025 | - |
| qwen3.5-122b-a10b | NOT RUN | - | - | - | - | - |
| gpt-oss-20b | internal better than text | False | 46.1 [37.3, 54.6] | 1.667e-05 | 0.0125 | positive_L0_internal |
| qwen3-32b | internal better than text | True | 36.1 [31.0, 41.4] | 1.667e-05 | 0.01667 | - |

## qwen3-32b (PASS_INDEPENDENT_GRADED_ROOT_AUDIT)

Scenes 144, rows 6912, clusters {'family': 36, 'template': 12}, tiers [1, 2], layer [13], gamma 1.

| route | metric | L=0 | L=64 | L=128 | L=191 | L=255 |
|---|---|---|---|---|---|---|
| internal | J | 100.0 [100.0, 100.0] | 99.3 [97.9, 100.0] | 72.9 [66.0, 79.9] | 4.9 [1.4, 9.0] | 0.0 [0.0, 0.0] |
| internal | coverage | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.0 [98.1, 99.7] | 99.0 [97.6, 100.0] | 99.0 [97.6, 100.0] |
| internal | R | 100.0 [100.0, 100.0] | 99.3 [97.9, 100.0] | 72.9 [66.0, 79.9] | 4.9 [1.4, 9.0] | 0.0 [0.0, 0.0] |
| internal | U | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| internal | x_accuracy | 100.0 [100.0, 100.0] | 99.7 [99.0, 100.0] | 86.1 [82.3, 89.6] | 51.0 [49.0, 53.5] | 49.0 [47.6, 50.0] |
| internal | J256 | 100.0 [100.0, 100.0] | 99.3 [97.9, 100.0] | 72.9 [66.0, 79.9] | 4.9 [1.4, 9.0] | 0.0 [0.0, 0.0] |
| text | J | 99.3 [97.9, 100.0] | 31.2 [24.3, 38.2] | 19.4 [14.6, 25.0] | 18.1 [12.5, 23.6] | 0.0 [0.0, 0.0] |
| text | coverage | 99.8 [99.5, 100.0] | 98.3 [96.9, 99.3] | 98.8 [97.6, 99.7] | 98.1 [96.2, 99.5] | 97.6 [95.5, 99.3] |
| text | R | 99.3 [97.9, 100.0] | 31.2 [24.3, 38.2] | 19.4 [14.6, 25.0] | 18.8 [13.2, 25.0] | 0.0 [0.0, 0.0] |
| text | U | 100.0 [100.0, 100.0] | 99.3 [97.9, 100.0] | 100.0 [100.0, 100.0] | 99.3 [97.9, 100.0] | 99.3 [97.9, 100.0] |
| text | x_accuracy | 99.7 [99.0, 100.0] | 63.2 [58.7, 67.4] | 55.9 [52.4, 59.4] | 56.6 [52.8, 60.4] | 47.9 [46.2, 49.3] |
| text | J256 | 99.3 [97.9, 100.0] | 31.2 [24.3, 38.2] | 19.4 [14.6, 25.0] | 18.1 [12.5, 23.6] | 0.0 [0.0, 0.0] |
| none | J / coverage | 0.0 [0.0, 0.0] / 94.4 [91.0, 97.2] | | | | |
| sham0_L128 | J / coverage | 0.0 [0.0, 0.0] / 55.6 [47.0, 64.1] | | | | |

| contrast | family | template | enclosing | p (max) |
|---|---|---|---|---|
| primary_internal_minus_text_pooled | 36.1 [31.0, 41.4] | 36.1 [31.0, 41.2] | [31.0, 41.4] | 1.667e-05 |
| internal_minus_text_L000 | 0.7 [0.0, 2.1] | 0.7 [0.0, 2.1] | [0.0, 2.1] | 0.7208 |
| internal_minus_text_L064 | 68.1 [60.4, 75.7] | 68.1 [61.8, 74.3] | [60.4, 75.7] | 1.667e-05 |
| internal_minus_text_L128 | 53.5 [45.1, 61.8] | 53.5 [45.8, 61.1] | [45.1, 61.8] | 1.667e-05 |
| internal_minus_text_L191 | -13.2 [-20.1, -6.9] | -13.2 [-18.8, -6.9] | [-20.1, -6.9] | 6.667e-05 |
| internal_minus_text_L255 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | [0.0, 0.0] | 1 |
| did_internal_minus_text_vs_L0 | 35.4 [29.6, 41.2] | 35.4 [29.6, 41.0] | [29.6, 41.2] | 1.667e-05 |
| graded_cost_internal | -41.0 [-43.8, -38.2] | -41.0 [-43.5, -38.7] | [-43.8, -38.2] | 1.667e-05 |
| graded_cost_text | -76.4 [-81.5, -71.3] | -76.4 [-80.8, -71.5] | [-81.5, -71.3] | 1.667e-05 |
| internal_minus_sham0_L128 | 72.9 [66.0, 79.9] | 72.9 [66.7, 78.5] | [66.0, 79.9] | 1.667e-05 |

Controls: positive_L0_internal: PASS (J=1.0, coverage=1.0); negative_L255_internal: PASS (J_sum=0, R_sum=0, x_accuracy_max_scene=0.5); positive_L0_text: PASS (J=0.9930555555555556, coverage=0.9982638888888888); negative_L255_text: PASS (J_sum=0, R_sum=0, x_accuracy_max_scene=0.5); no_message: PASS (J_sum=0); l255_outputs_identical_across_worlds: PASS (pairs=576, identical=576)

Continuity with the paper's R3 complete-fact rows: internal_L000_vs_R3_literal_selected: {'answers': 576, 'identical_prompt': 576, 'identical_tokens': 576, 'identical_logits_sha256': 576}; text_L000_vs_R3_literal_packet_text: {'answers': 576, 'identical_prompt': 576, 'identical_tokens': 576, 'identical_logits_sha256': 576}

Throughput: development_qwen3-32b_s0: 0.714 s/row (torch.float32, adapter True, GPUs 1); development_qwen3-32b_s1: 0.629 s/row (torch.float32, adapter True, GPUs 1); engineering_qwen3-32b_s0: 1.387 s/row (torch.float32, adapter True, GPUs 1)

## qwen3.5-27b (PASS_INDEPENDENT_GRADED_ROOT_AUDIT)

Scenes 144, rows 6912, clusters {'family': 36, 'template': 12}, tiers [1, 2], layer [13], gamma 1.

| route | metric | L=0 | L=64 | L=128 | L=191 | L=255 |
|---|---|---|---|---|---|---|
| internal | J | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 95.1 [90.3, 99.3] | 50.7 [41.0, 60.4] | 0.0 [0.0, 0.0] |
| internal | coverage | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| internal | R | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 95.1 [90.3, 99.3] | 50.7 [41.0, 60.4] | 0.0 [0.0, 0.0] |
| internal | U | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| internal | x_accuracy | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 97.6 [95.1, 99.7] | 75.3 [70.5, 80.2] | 50.0 [50.0, 50.0] |
| internal | J256 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 95.1 [90.3, 99.3] | 50.7 [41.0, 60.4] | 0.0 [0.0, 0.0] |
| text | J | 100.0 [100.0, 100.0] | 64.6 [54.9, 74.3] | 41.0 [31.9, 50.7] | 38.9 [27.8, 50.0] | 0.0 [0.0, 0.0] |
| text | coverage | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| text | R | 100.0 [100.0, 100.0] | 64.6 [54.9, 74.3] | 41.0 [31.9, 50.7] | 38.9 [27.8, 50.0] | 0.0 [0.0, 0.0] |
| text | U | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| text | x_accuracy | 100.0 [100.0, 100.0] | 82.3 [77.4, 87.2] | 70.1 [65.6, 75.0] | 68.4 [63.2, 74.0] | 50.0 [50.0, 50.0] |
| text | J256 | 100.0 [100.0, 100.0] | 64.6 [54.9, 74.3] | 41.0 [31.9, 50.7] | 38.9 [27.8, 50.0] | 0.0 [0.0, 0.0] |
| none | J / coverage | 0.0 [0.0, 0.0] / 100.0 [100.0, 100.0] | | | | |
| sham0_L128 | J / coverage | 2.1 [0.0, 4.9] / 83.9 [79.2, 88.4] | | | | |

| contrast | family | template | enclosing | p (max) |
|---|---|---|---|---|
| primary_internal_minus_text_pooled | 33.8 [22.5, 45.1] | 33.8 [20.8, 48.1] | [20.8, 48.1] | 1.667e-05 |
| internal_minus_text_L000 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | [0.0, 0.0] | 1 |
| internal_minus_text_L064 | 35.4 [25.7, 45.1] | 35.4 [24.3, 47.2] | [24.3, 47.2] | 1.667e-05 |
| internal_minus_text_L128 | 54.2 [41.7, 66.0] | 54.2 [39.6, 68.8] | [39.6, 68.8] | 1.667e-05 |
| internal_minus_text_L191 | 11.8 [-4.9, 28.5] | 11.8 [-5.6, 31.9] | [-5.6, 31.9] | 0.2185 |
| internal_minus_text_L255 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | [0.0, 0.0] | 1 |
| did_internal_minus_text_vs_L0 | 33.8 [22.5, 45.1] | 33.8 [20.8, 48.1] | [20.8, 48.1] | 1.667e-05 |
| graded_cost_internal | -18.1 [-21.8, -14.4] | -18.1 [-20.8, -15.0] | [-21.8, -14.4] | 1.667e-05 |
| graded_cost_text | -51.9 [-61.1, -42.6] | -51.9 [-63.9, -40.5] | [-63.9, -40.5] | 1.667e-05 |
| internal_minus_sham0_L128 | 93.1 [87.5, 97.2] | 93.1 [84.7, 99.3] | [84.7, 99.3] | 1.667e-05 |

Controls: positive_L0_internal: PASS (J=1.0, coverage=1.0); negative_L255_internal: PASS (J_sum=0, R_sum=0, x_accuracy_max_scene=0.5); positive_L0_text: PASS (J=1.0, coverage=1.0); negative_L255_text: PASS (J_sum=0, R_sum=0, x_accuracy_max_scene=0.5); no_message: PASS (J_sum=0); l255_outputs_identical_across_worlds: PASS (pairs=576, identical=576)

Throughput: development_qwen3.5-27b_s0: 0.996 s/row (torch.float32, adapter True, GPUs 1); development_qwen3.5-27b_s1: 0.801 s/row (torch.float32, adapter True, GPUs 1); engineering_qwen3.5-27b_s0: 1.577 s/row (torch.float32, adapter True, GPUs 1)

## gpt-oss-20b (PASS_INDEPENDENT_GRADED_ROOT_AUDIT)

Scenes 144, rows 6912, clusters {'family': 36, 'template': 12}, tiers [1, 2], layer [5], gamma 1.

| route | metric | L=0 | L=64 | L=128 | L=191 | L=255 |
|---|---|---|---|---|---|---|
| internal | J | 84.0 [77.8, 89.6] | 76.4 [67.4, 84.7] | 63.2 [56.2, 70.1] | 34.0 [25.0, 43.1] | 0.0 [0.0, 0.0] |
| internal | coverage | 99.8 [99.5, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| internal | R | 88.2 [83.3, 93.1] | 82.6 [75.0, 89.6] | 66.7 [59.0, 73.6] | 35.4 [26.4, 44.4] | 0.0 [0.0, 0.0] |
| internal | U | 93.1 [88.2, 97.2] | 91.0 [84.7, 96.5] | 93.8 [88.9, 97.9] | 95.1 [91.0, 98.6] | 95.8 [91.7, 99.3] |
| internal | x_accuracy | 94.1 [91.7, 96.5] | 91.3 [87.5, 94.8] | 83.3 [79.5, 86.8] | 67.7 [63.2, 72.2] | 50.0 [50.0, 50.0] |
| internal | J256 | 84.0 [77.8, 89.6] | 76.4 [67.4, 84.7] | 63.2 [56.2, 70.1] | 34.0 [25.0, 43.1] | 0.0 [0.0, 0.0] |
| text | J | 99.3 [97.9, 100.0] | 15.3 [9.0, 22.9] | 11.1 [5.6, 18.1] | 9.0 [4.2, 14.6] | 0.0 [0.0, 0.0] |
| text | coverage | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.8 [99.5, 100.0] | 99.8 [99.5, 100.0] | 100.0 [100.0, 100.0] |
| text | R | 99.3 [97.9, 100.0] | 15.3 [9.0, 22.9] | 11.1 [5.6, 18.1] | 9.0 [4.2, 14.6] | 0.0 [0.0, 0.0] |
| text | U | 100.0 [100.0, 100.0] | 99.3 [97.9, 100.0] | 99.3 [97.9, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| text | x_accuracy | 99.7 [99.0, 100.0] | 56.9 [53.8, 60.4] | 55.2 [52.4, 58.7] | 53.8 [51.4, 56.6] | 50.0 [50.0, 50.0] |
| text | J256 | 99.3 [97.9, 100.0] | 15.3 [9.0, 22.9] | 11.1 [5.6, 18.1] | 9.0 [4.2, 14.6] | 0.0 [0.0, 0.0] |
| none | J / coverage | 0.0 [0.0, 0.0] / 98.3 [96.9, 99.7] | | | | |
| sham0_L128 | J / coverage | 0.0 [0.0, 0.0] / 99.7 [99.0, 100.0] | | | | |

| contrast | family | template | enclosing | p (max) |
|---|---|---|---|---|
| primary_internal_minus_text_pooled | 46.1 [38.9, 53.0] | 46.1 [37.3, 54.6] | [37.3, 54.6] | 1.667e-05 |
| internal_minus_text_L000 | -15.3 [-21.5, -9.7] | -15.3 [-22.9, -8.3] | [-22.9, -8.3] | 1.667e-05 |
| internal_minus_text_L064 | 61.1 [51.4, 70.8] | 61.1 [48.6, 72.9] | [48.6, 72.9] | 1.667e-05 |
| internal_minus_text_L128 | 52.1 [43.8, 60.4] | 52.1 [45.1, 59.7] | [43.8, 60.4] | 1.667e-05 |
| internal_minus_text_L191 | 25.0 [16.0, 34.0] | 25.0 [14.6, 35.4] | [14.6, 35.4] | 1.667e-05 |
| internal_minus_text_L255 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | [0.0, 0.0] | 1 |
| did_internal_minus_text_vs_L0 | 61.3 [54.2, 68.3] | 61.3 [54.6, 68.1] | [54.2, 68.3] | 1.667e-05 |
| graded_cost_internal | -26.2 [-31.5, -20.8] | -26.2 [-29.6, -22.2] | [-31.5, -20.8] | 1.667e-05 |
| graded_cost_text | -87.5 [-92.6, -81.5] | -87.5 [-91.4, -82.6] | [-92.6, -81.5] | 1.667e-05 |
| internal_minus_sham0_L128 | 63.2 [56.2, 70.1] | 63.2 [56.9, 68.8] | [56.2, 70.1] | 1.667e-05 |

Controls: positive_L0_internal: FAIL (J=0.8402777777777778, coverage=0.9982638888888888); negative_L255_internal: PASS (J_sum=0, R_sum=0, x_accuracy_max_scene=0.5); positive_L0_text: PASS (J=0.9930555555555556, coverage=1.0); negative_L255_text: PASS (J_sum=0, R_sum=0, x_accuracy_max_scene=0.5); no_message: PASS (J_sum=0); l255_outputs_identical_across_worlds: PASS (pairs=576, identical=576)

Throughput: development_gpt-oss-20b_s0: 0.647 s/row (torch.float32, adapter True, GPUs 1); engineering_gpt-oss-20b_s0: 1.231 s/row (torch.float32, adapter True, GPUs 1)

## Limits

Scene-paired repeated measures: routes and levels live on the same scenes and add no independent N. Intervals are cluster-bootstrap envelopes over three seeds at the family and template levels; seeds are not model replications. The L=255 and no-message rows are structural zeros of four-answer success for a deterministic receiver, not evidence about communication. Complete facts only (no sender). The bank is finite and constructed; the default evaluation set is a pre-declared balanced subset of it.
