# Sender gate — candidate_A4/llama-3.1-8b-instruct on bank.json (r4_four_object)

**Verdict: NO_LOCK**

Packets `artifacts/receiver_recovery/r4_packets/candidateA4_r4.json` (sha256 `b854cd94b944…`); bank `artifacts/receiver_recovery/r4_bank/bank.json` (sha256 `5a1bf14ca502…`, 192 scenes / 384 worlds). Generated 2026-09-24T13:10:53Z by `scripts/latest0924/sender_gate.py` (sha256 `8fd8227bb403…`).

Integrity: **OK** — header {'bank_sha256_matches': True, 'prompt_sha256_frozen': True, 'inventory_matches_reference': True, 'inventory_size': 221}; nonzero counts none; file summary matches recount: True.

| metric | all |
|---|---|
| messages (scenes) | 384 (192) |
| parse fully / partially / failed | 384 / 0 / 0 |
| parse fully (%) [95% CI] | 100.0 [100.0, 100.0] |
| fact recall (Table 17) | 2968/3072 (96.6) [95.1, 98.0] |
| complete objects (Table 17) | 1432/1536 (93.2) [90.2, 96.0] |
| complete packets | 349/384 (90.9) [86.7, 94.5] |
| wrong-object lanes (words) | 1 (1) |
| unsupported-attribute lanes | 1 |
| extra non-attribute lanes | 855 |
| empty lanes | 0 |
| mean words per object | 2.50 {'2': 772, '3': 757, '4': 7} |
| precision / exact set (B6 defs) | 77.2 / 44.3 |
| mean bytes (Table 17) | 36.0 |
| generated tokens mean / max | 56.888 / 83 |
| hit max_new_tokens / no final channel | n/a / n/a |
| greedy argmax mismatches | n/a |

Recall counts required attribute words assigned to the correct object; complete objects contain both required facts; extra selected words are allowed (Table 17 definitions, engineering scenes included in `all`). Intervals: scene-clustered percentile bootstrap, 2000 draws, seed 0. Message content only — not a receiver outcome.
