# Experiment B text-format follow-up — analysis v2 (F1-F5 + embedding)

Lock `gates/latest0924/expB_textfmt.lock.json` sha `35ce0909cae1`; Holm family of 16; intervals enclose family (36) and template (12) cluster bootstraps, 20,000 draws x 3 seeds.

| contrast | estimate [enclosing 95%] | p | Holm thr | verdict |
|---|---|---|---|---|
| qwen3-32b:pct | 23.4 [18.1, 29.9] | 1.67e-05 | 0.004167 | internal still better |
| qwen3-32b:verbal | 25.2 [18.8, 31.7] | 1.67e-05 | 0.005 | internal still better |
| qwen3-32b:instr | 29.4 [21.3, 37.5] | 1.67e-05 | 0.003571 | internal still better |
| qwen3-32b:instr_internal | 26.4 [18.1, 35.0] | 1.67e-05 | 0.003846 | internal still better |
| qwen3-32b:sorted | 31.9 [24.3, 38.7] | 1.67e-05 | 0.004545 | internal still better |
| qwen3-32b:hedged | 15.5 [9.7, 21.3] | 1.67e-05 | 0.003333 | internal still better |
| qwen3-32b:internal_minus_emb | 7.9 [3.9, 11.8] | 6.67e-05 | 0.01 | internal better |
| qwen3-32b:emb_minus_text | 28.2 [22.7, 33.6] | 1.67e-05 | 0.003125 | emb better |
| qwen3.5-27b:pct | 38.4 [24.3, 53.5] | 1.67e-05 | 0.00625 | internal still better |
| qwen3.5-27b:verbal | 25.2 [18.3, 31.9] | 1.67e-05 | 0.008333 | internal still better |
| qwen3.5-27b:instr | 14.8 [4.9, 25.0] | 0.00233 | 0.0125 | internal still better |
| qwen3.5-27b:instr_internal | 15.0 [4.4, 25.9] | 0.0038 | 0.01667 | internal still better |
| qwen3.5-27b:sorted | 13.0 [6.9, 19.2] | 1.67e-05 | 0.007143 | internal still better |
| qwen3.5-27b:hedged | -8.8 [-14.4, -2.8] | 0.0047 | 0.025 | text better |
| qwen3.5-27b:internal_minus_emb | 38.9 [31.9, 45.8] | 1.67e-05 | 0.005556 | internal better |
| qwen3.5-27b:emb_minus_text | -5.1 [-13.9, 3.9] | 0.247 | 0.05 | unresolved |

## qwen3-32b

| route | L=0 | L=64 | L=128 | L=191 | L=255 |
|---|---|---|---|---|---|
| internal J | 100.0 | 99.3 | 72.9 | 4.9 | 0.0 |
| text_expB J | 99.3 | 31.2 | 19.4 | 18.1 | 0.0 |
| textpct J | 100.0 | 38.2 | 35.4 | 33.3 | 0.0 |
| textverbal J | 99.3 | 49.3 | 31.9 | 20.1 | 0.0 |
| textinstr J | 100.0 | 34.0 | 29.9 | 25.0 | 0.0 |
| internal_instr J | 100.0 | 97.2 | 68.8 | 2.1 | 0.0 |
| textsorted J | 99.3 | 30.6 | 25.0 | 25.7 | 0.0 |
| texthedged J | 97.9 | 60.4 | 42.4 | 27.8 | 0.0 |
| emb J | 99.3 | 97.9 | 55.6 | 0.0 | 0.0 |

Controls: positive_L0_textpct: PASS; negative_L255_textpct: PASS; positive_L0_textverbal: PASS; negative_L255_textverbal: PASS; positive_L0_textinstr: PASS; negative_L255_textinstr: PASS; positive_L0_internal_instr: PASS; negative_L255_internal_instr: PASS; positive_L0_internal: PASS; negative_L255_internal: PASS; positive_L0_text_expB: PASS; negative_L255_text_expB: PASS; positive_L0_textsorted: PASS; negative_L255_textsorted: PASS; positive_L0_texthedged: PASS; negative_L255_texthedged: PASS; positive_L0_emb: PASS; negative_L255_emb: PASS; no_message_none: PASS; no_message_none_instr: PASS; no_message_none_a1: PASS; l255_outputs_identical_across_worlds: PASS

Engineering: {'rows': 720, 'failing': [], 'max_logit_difference': 0.000316619873046875, 'continuity_with_expB': {'internal_L064:identical': 32, 'text_L064:identical': 32, 'emb_L000_vs_expB_text_L000:identical': 576}}

Text-format gain over Experiment B text (pooled L=64/128/191): textpct_minus_text_expB 12.7 [5.8, 18.5]; textverbal_minus_text_expB 10.9 [6.2, 16.0]; textinstr_minus_text_expB 6.7 [0.5, 13.2]; textsorted_minus_text_expB 4.2 [-4.6, 13.0]; texthedged_minus_text_expB 20.6 [13.4, 27.5]; emb_minus_text_expB 28.2 [22.7, 33.6]

## qwen3.5-27b

| route | L=0 | L=64 | L=128 | L=191 | L=255 |
|---|---|---|---|---|---|
| internal J | 100.0 | 100.0 | 95.1 | 50.7 | 0.0 |
| text_expB J | 100.0 | 64.6 | 41.0 | 38.9 | 0.0 |
| textpct J | 100.0 | 59.7 | 37.5 | 33.3 | 0.0 |
| textverbal J | 100.0 | 84.7 | 51.4 | 34.0 | 0.0 |
| textinstr J | 100.0 | 77.1 | 66.0 | 58.3 | 0.0 |
| internal_instr J | 100.0 | 100.0 | 92.4 | 54.2 | 0.0 |
| textsorted J | 100.0 | 92.4 | 68.1 | 46.5 | 0.0 |
| texthedged J | 100.0 | 100.0 | 94.4 | 77.8 | 0.0 |
| emb J | 100.0 | 100.0 | 29.2 | 0.0 | 0.0 |

Controls: positive_L0_textpct: PASS; negative_L255_textpct: PASS; positive_L0_textverbal: PASS; negative_L255_textverbal: PASS; positive_L0_textinstr: PASS; negative_L255_textinstr: PASS; positive_L0_internal_instr: PASS; negative_L255_internal_instr: PASS; positive_L0_internal: PASS; negative_L255_internal: PASS; positive_L0_text_expB: PASS; negative_L255_text_expB: PASS; positive_L0_textsorted: PASS; negative_L255_textsorted: PASS; positive_L0_texthedged: PASS; negative_L255_texthedged: PASS; positive_L0_emb: PASS; negative_L255_emb: PASS; no_message_none: PASS; no_message_none_instr: PASS; no_message_none_a1: PASS; l255_outputs_identical_across_worlds: PASS

Engineering: {'rows': 720, 'failing': [], 'max_logit_difference': 0.00028777122497558594, 'continuity_with_expB': {'internal_L064:identical': 32, 'text_L064:identical': 32, 'emb_L000_vs_expB_text_L000:identical': 576}}

Text-format gain over Experiment B text (pooled L=64/128/191): textpct_minus_text_expB -4.6 [-10.9, 2.5]; textverbal_minus_text_expB 8.6 [-2.3, 19.2]; textinstr_minus_text_expB 19.0 [10.2, 27.5]; textsorted_minus_text_expB 20.8 [7.2, 34.7]; texthedged_minus_text_expB 42.6 [30.6, 55.3]; emb_minus_text_expB -5.1 [-13.9, 3.9]
