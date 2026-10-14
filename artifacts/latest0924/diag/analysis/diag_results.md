# D1/D2 diagnostics (development material; diagnostic, not confirmation)

Lock `gates/latest0924/diag_protocol.lock.json` sha `f33369444e50`. Four-answer success, family-macro (%), 95% paired family-bootstrap enclosing interval (20000 draws x seeds (20260924, 20260925, 20260926)).

## qwen3-14b-base  (92 scenes, 12 families, 8832 rows)

Checks: fitted_cut_at_write_layer_cache_identical_to_A=True, fitted_cut_at_write_layer_tokens_equal_A=True, fitted_cut_at_last_layer_tokens_equal_no_message=True, native_cut_at_write_layer_cache_identical_to_A=True, native_cut_at_write_layer_tokens_equal_A=True, native_cut_at_last_layer_tokens_equal_no_message=True, identity_write_tokens_equal_no_message=True, identity_write_logits_equal_no_message=True, engineering_receipt=True

No message 0.0; complete-fact text 100.0.

| source | L | A (d20) | B (write at L) | C (d20, cut L) | C-B [95% CI] | A-C [95% CI] | verdict |
|---|---|---|---|---|---|---|---|
| fitted | d35 | 100.0 | 87.5 | 97.9 | +10.4 [+0.0, +22.9] | +2.1 [+0.0, +5.2] | unresolved/partial |
| fitted | d50 | 100.0 | 6.2 | 59.4 | +53.1 [+39.6, +65.6] | +40.6 [+29.2, +52.1] | lower_layer_reading_needed_at_L |
| fitted | d65 | 100.0 | 1.0 | 0.0 | -1.0 [-3.1, +0.0] | +100.0 [+100.0, +100.0] | lower_layer_reading_needed_at_L |
| native | d35 | 100.0 | 96.9 | 97.9 | +1.0 [+0.0, +3.1] | +2.1 [+0.0, +5.2] | unresolved/partial |
| native | d50 | 100.0 | 22.9 | 52.1 | +29.2 [+16.7, +41.7] | +47.9 [+31.2, +63.5] | lower_layer_reading_needed_at_L |
| native | d65 | 100.0 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | +100.0 [+100.0, +100.0] | lower_layer_reading_needed_at_L |

| depth | fixed | matched | native | text | G_layout [CI] | G_joint [CI] | G_delivery [CI] |
|---|---|---|---|---|---|---|---|
| d20 | 100.0 | 100.0 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| d35 | 87.5 | 85.4 | 96.9 | 100.0 | -2.1 [-8.3, +3.1] | +11.5 [+2.1, +22.9] | +3.1 [+0.0, +6.2] |
| d50 | 6.2 | 11.5 | 22.9 | 100.0 | +5.2 [-2.1, +12.5] | +11.5 [+5.2, +18.8] | +77.1 [+67.7, +85.4] |

## glm-4-9b  (92 scenes, 12 families, 8832 rows)

Checks: fitted_cut_at_write_layer_cache_identical_to_A=True, fitted_cut_at_write_layer_tokens_equal_A=True, fitted_cut_at_last_layer_tokens_equal_no_message=True, native_cut_at_write_layer_cache_identical_to_A=True, native_cut_at_write_layer_tokens_equal_A=True, native_cut_at_last_layer_tokens_equal_no_message=True, identity_write_tokens_equal_no_message=True, identity_write_logits_equal_no_message=True, engineering_receipt=True

No message 0.0; complete-fact text 99.0.

| source | L | A (d20) | B (write at L) | C (d20, cut L) | C-B [95% CI] | A-C [95% CI] | verdict |
|---|---|---|---|---|---|---|---|
| fitted | d35 | 88.5 | 8.3 | 72.9 | +64.6 [+52.1, +77.1] | +15.6 [+6.2, +27.1] | unresolved/partial |
| fitted | d50 | 88.5 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | +88.5 [+79.2, +95.8] | lower_layer_reading_needed_at_L |
| fitted | d65 | 88.5 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | +88.5 [+79.2, +95.8] | lower_layer_reading_needed_at_L |
| native | d35 | 90.6 | 66.7 | 75.0 | +8.3 [+3.1, +13.5] | +15.6 [+8.3, +24.0] | unresolved/partial |
| native | d50 | 90.6 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | +90.6 [+81.2, +97.9] | lower_layer_reading_needed_at_L |
| native | d65 | 90.6 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | +90.6 [+81.2, +97.9] | lower_layer_reading_needed_at_L |

| depth | fixed | matched | native | text | G_layout [CI] | G_joint [CI] | G_delivery [CI] |
|---|---|---|---|---|---|---|---|
| d20 | 88.5 | 87.5 | 90.6 | 99.0 | -1.0 [-5.2, +3.1] | +3.1 [-2.1, +8.3] | +8.3 [+1.0, +17.7] |
| d35 | 8.3 | 40.6 | 66.7 | 99.0 | +32.3 [+24.0, +39.6] | +26.0 [+11.5, +39.6] | +32.3 [+20.8, +45.8] |
| d50 | 0.0 | 0.0 | 0.0 | 99.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +99.0 [+96.9, +100.0] |

## gemma-3-27b-it  (92 scenes, 12 families, 7728 rows)

Checks: fitted_cut_at_write_layer_cache_identical_to_A=True, fitted_cut_at_write_layer_tokens_equal_A=True, fitted_cut_at_last_layer_tokens_equal_no_message=True, native_cut_at_write_layer_cache_identical_to_A=True, native_cut_at_write_layer_tokens_equal_A=True, native_cut_at_last_layer_tokens_equal_no_message=True, identity_write_tokens_equal_no_message=True, identity_write_logits_equal_no_message=True, engineering_receipt=True

No message 0.0; complete-fact text 100.0.

| source | L | A (d20) | B (write at L) | C (d20, cut L) | C-B [95% CI] | A-C [95% CI] | verdict |
|---|---|---|---|---|---|---|---|
| fitted | d35 | 100.0 | 85.4 | 97.9 | +12.5 [+5.2, +21.9] | +2.1 [+0.0, +6.2] | unresolved/partial |
| fitted | d50 | 100.0 | 0.0 | 2.1 | +2.1 [+0.0, +5.2] | +97.9 [+94.8, +100.0] | lower_layer_reading_needed_at_L |
| fitted | d65 | 100.0 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | +100.0 [+100.0, +100.0] | lower_layer_reading_needed_at_L |
| native | d35 | 100.0 | 96.9 | 97.9 | +1.0 [+0.0, +3.1] | +2.1 [+0.0, +6.2] | unresolved/partial |
| native | d50 | 100.0 | 0.0 | 2.1 | +2.1 [+0.0, +5.2] | +97.9 [+94.8, +100.0] | lower_layer_reading_needed_at_L |
| native | d65 | 100.0 | 0.0 | 1.0 | +1.0 [+0.0, +3.1] | +99.0 [+96.9, +100.0] | lower_layer_reading_needed_at_L |

## qwen3-32b  (92 scenes, 12 families, 7728 rows)

Checks: fitted_cut_at_write_layer_cache_identical_to_A=True, fitted_cut_at_write_layer_tokens_equal_A=True, fitted_cut_at_last_layer_tokens_equal_no_message=True, native_cut_at_write_layer_cache_identical_to_A=True, native_cut_at_write_layer_tokens_equal_A=True, native_cut_at_last_layer_tokens_equal_no_message=True, identity_write_tokens_equal_no_message=True, identity_write_logits_equal_no_message=True, engineering_receipt=True

No message 0.0; complete-fact text 100.0.

| source | L | A (d20) | B (write at L) | C (d20, cut L) | C-B [95% CI] | A-C [95% CI] | verdict |
|---|---|---|---|---|---|---|---|
| fitted | d35 | 100.0 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | unresolved/partial |
| fitted | d50 | 100.0 | 96.9 | 100.0 | +3.1 [+0.0, +6.2] | +0.0 [+0.0, +0.0] | unresolved/partial |
| fitted | d65 | 100.0 | 1.0 | 6.2 | +5.2 [+0.0, +11.5] | +93.8 [+87.5, +99.0] | lower_layer_reading_needed_at_L |
| native | d35 | 100.0 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | unresolved/partial |
| native | d50 | 100.0 | 97.9 | 100.0 | +2.1 [+0.0, +5.2] | +0.0 [+0.0, +0.0] | unresolved/partial |
| native | d65 | 100.0 | 2.1 | 7.3 | +5.2 [+0.0, +10.4] | +92.7 [+87.5, +96.9] | lower_layer_reading_needed_at_L |

