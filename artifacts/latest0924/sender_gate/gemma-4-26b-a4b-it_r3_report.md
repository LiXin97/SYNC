# Sender gate — latest0924/gemma-4-26b-a4b-it on bank.json (r3_confirmation)

**Verdict: NO_LOCK**

Packets `artifacts/latest0924/senders/gemma-4-26b-a4b-it_r3.json` (sha256 `a55a1c330b6a…`); bank `artifacts/receiver_recovery/r3_bank/bank.json` (sha256 `c88ff3197717…`, 576 scenes / 1152 worlds). Generated 2026-09-24T14:15:13Z by `scripts/latest0924/sender_gate.py` (sha256 `8fd8227bb403…`).

Integrity: **OK** — header {'bank_sha256_matches': True, 'prompt_sha256_frozen': True, 'inventory_matches_reference': True, 'inventory_size': 221}; nonzero counts none; file summary matches recount: True.

| metric | all | evaluated |
|---|---|---|
| messages (scenes) | 1152 (576) | 1144 (572) |
| parse fully / partially / failed | 1152 / 0 / 0 | 1144 / 0 / 0 |
| parse fully (%) [95% CI] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| fact recall (Table 17) | 4526/4608 (98.2) [97.7, 98.7] | 4495/4576 (98.2) [97.7, 98.7] |
| complete objects (Table 17) | 2222/2304 (96.4) [95.4, 97.3] | 2207/2288 (96.5) [95.5, 97.4] |
| complete packets | 1078/1152 (93.6) [91.8, 95.2] | 1071/1144 (93.6) [91.9, 95.3] |
| wrong-object lanes (words) | 0 (0) | 0 (0) |
| unsupported-attribute lanes | 0 | 0 |
| extra non-attribute lanes | 1168 | 1160 |
| empty lanes | 0 | 0 |
| mean words per object | 2.47 {'1': 35, '2': 1148, '3': 1121} | 2.47 {'1': 35, '2': 1139, '3': 1114} |
| precision / exact set (B6 defs) | 79.5 / 47.8 | 79.5 / 47.8 |
| mean bytes (Table 17) | 21.9 | 21.9 |
| generated tokens mean / max | 29.661 / 36 | 29.663 / 36 |
| hit max_new_tokens / no final channel | 0 / n/a | 0 / n/a |
| greedy argmax mismatches | 0 | 0 |

Recall counts required attribute words assigned to the correct object; complete objects contain both required facts; extra selected words are allowed (Table 17 definitions, engineering scenes included in `all`). Intervals: scene-clustered percentile bootstrap, 2000 draws, seed 0. Message content only — not a receiver outcome.
