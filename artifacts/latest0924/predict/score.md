# Experiment F: blind held-out validation

tau = 0.7864 (primary metric bind_pair_success); depth-only comparator d* = 0.425

Balanced accuracy 1.000 [1.000, 1.000]; accuracy 1.000; Spearman 0.904 [0.769, 0.946]; depth-only BA 0.868 [0.814, 0.922]; difference +0.132 [+0.078, +0.186]. Locked success criterion met: True.

| receiver | d=0.2 | d=0.35 | d=0.5 | d=0.65 | d=0.8 |
|---|---|---|---|---|---|
| qwen3.5-2b | 89.6 (U, 0.945) | 35.4 (., 0.209) | 0.0 (., 0.018) | 2.1 (., 0.018) | 0.0 (., 0.018) |
| qwen3.5-4b | 100.0 (U, 0.982) | 7.3 (., 0.055) | 0.0 (., 0.000) | 0.0 (., 0.000) | 0.0 (., 0.000) |
| qwen3.5-9b | 100.0 (U, 0.991) | 34.4 (., 0.282) | 0.0 (., 0.000) | 3.1 (., 0.000) | 0.0 (., 0.000) |
| qwen3.5-27b | 100.0 (U, 1.000) | 100.0 (U, 1.000) | 18.8 (., 0.309) | 0.0 (., 0.018) | 0.0 (., 0.009) |
| qwen3.6-27b | 100.0 (U, 1.000) | 100.0 (U, 1.000) | 85.4 (U, 0.927) | 11.5 (., 0.182) | 0.0 (., 0.009) |
| granite-4.1-8b | 100.0 (U, 1.000) | 99.0 (U, 0.991) | 21.9 (., 0.764) | 0.0 (., 0.009) | 0.0 (., 0.000) |
| olmo-3-7b-instruct | 100.0 (U, 0.982) | 20.8 (., 0.409) | 2.1 (., 0.109) | 2.1 (., 0.100) | 0.0 (., 0.118) |
| gpt-oss-20b | 89.6 (U, 0.991) | 26.0 (., 0.409) | 2.1 (., 0.000) | 1.0 (., 0.000) | 0.0 (., 0.000) |
| mistral-small-3.2-24b | 99.0 (U, 0.973) | 10.4 (., 0.364) | 0.0 (., 0.018) | 0.0 (., 0.018) | 0.0 (., 0.018) |

Cells: observed J (predicted usable U / not ., calibration metric).

Non-blind check vs Experiment A middle band (already observed): agreement 0.89.
