# Sender gate — latest0924/gpt-oss-20b on bank.json (r3_confirmation)

**Verdict: NO_LOCK**

Packets `artifacts/latest0924/senders/gpt-oss-20b_r3.json` (sha256 `5298eed435e4…`); bank `artifacts/receiver_recovery/r3_bank/bank.json` (sha256 `c88ff3197717…`, 576 scenes / 1152 worlds). Generated 2026-09-24T14:15:28Z by `scripts/latest0924/sender_gate.py` (sha256 `8fd8227bb403…`).

Integrity: **OK** — header {'bank_sha256_matches': True, 'prompt_sha256_frozen': True, 'inventory_matches_reference': True, 'inventory_size': 221}; nonzero counts none; file summary matches recount: True.

| metric | all | evaluated |
|---|---|---|
| messages (scenes) | 1152 (576) | 1144 (572) |
| parse fully / partially / failed | 1139 / 5 / 8 | 1131 / 5 / 8 |
| parse fully (%) [95% CI] | 98.9 [98.2, 99.5] | 98.9 [98.2, 99.5] |
| fact recall (Table 17) | 4337/4608 (94.1) [93.0, 95.2] | 4305/4576 (94.1) [92.9, 95.1] |
| complete objects (Table 17) | 2056/2304 (89.2) [87.3, 91.1] | 2040/2288 (89.2) [87.0, 91.1] |
| complete packets | 988/1152 (85.8) [83.2, 88.1] | 980/1144 (85.7) [83.1, 88.0] |
| wrong-object lanes (words) | 0 (0) | 0 (0) |
| unsupported-attribute lanes | 0 | 0 |
| extra non-attribute lanes | 462 | 458 |
| empty lanes | 21 | 21 |
| mean words per object | 2.08 {'0': 21, '1': 155, '2': 1739, '3': 388, '4': 1} | 2.08 {'0': 21, '1': 155, '2': 1727, '3': 384, '4': 1} |
| precision / exact set (B6 defs) | 90.3 / 72.4 | 90.3 / 72.4 |
| mean bytes (Table 17) | 20.3 | 20.3 |
| generated tokens mean / max | 103.824 / 256 | 103.767 / 256 |
| hit max_new_tokens / no final channel | 6 / 6 | 6 / 6 |
| greedy argmax mismatches | 0 | 0 |

Recall counts required attribute words assigned to the correct object; complete objects contain both required facts; extra selected words are allowed (Table 17 definitions, engineering scenes included in `all`). Intervals: scene-clustered percentile bootstrap, 2000 draws, seed 0. Message content only — not a receiver outcome.
