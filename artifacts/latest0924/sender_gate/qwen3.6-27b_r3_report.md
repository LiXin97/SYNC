# Sender gate — latest0924/qwen3.6-27b on bank.json (r3_confirmation)

**Verdict: NO_LOCK**

Packets `artifacts/latest0924/senders/qwen3.6-27b_r3.json` (sha256 `833301468928…`); bank `artifacts/receiver_recovery/r3_bank/bank.json` (sha256 `c88ff3197717…`, 576 scenes / 1152 worlds). Generated 2026-09-24T14:15:20Z by `scripts/latest0924/sender_gate.py` (sha256 `8fd8227bb403…`).

Integrity: **OK** — header {'bank_sha256_matches': True, 'prompt_sha256_frozen': True, 'inventory_matches_reference': True, 'inventory_size': 221}; nonzero counts none; file summary matches recount: True.

| metric | all | evaluated |
|---|---|---|
| messages (scenes) | 1152 (576) | 1144 (572) |
| parse fully / partially / failed | 1152 / 0 / 0 | 1144 / 0 / 0 |
| parse fully (%) [95% CI] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| fact recall (Table 17) | 4506/4608 (97.8) [97.0, 98.5] | 4474/4576 (97.8) [97.1, 98.5] |
| complete objects (Table 17) | 2202/2304 (95.6) [94.1, 97.0] | 2186/2288 (95.5) [94.1, 96.9] |
| complete packets | 1100/1152 (95.5) [94.0, 96.9] | 1092/1144 (95.5) [94.1, 96.9] |
| wrong-object lanes (words) | 0 (0) | 0 (0) |
| unsupported-attribute lanes | 0 | 0 |
| extra non-attribute lanes | 1570 | 1558 |
| empty lanes | 0 | 0 |
| mean words per object | 2.64 {'1': 40, '2': 756, '3': 1508} | 2.64 {'1': 40, '2': 752, '3': 1496} |
| precision / exact set (B6 defs) | 74.2 / 30.1 | 74.2 / 30.2 |
| mean bytes (Table 17) | 22.5 | 22.5 |
| generated tokens mean / max | 27.122 / 33 | 27.114 / 33 |
| hit max_new_tokens / no final channel | 0 / n/a | 0 / n/a |
| greedy argmax mismatches | 0 | 0 |

Recall counts required attribute words assigned to the correct object; complete objects contain both required facts; extra selected words are allowed (Table 17 definitions, engineering scenes included in `all`). Intervals: scene-clustered percentile bootstrap, 2000 draws, seed 0. Message content only — not a receiver outcome.
