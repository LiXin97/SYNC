# Sender gate — candidate_A/llama-3.1-8b-instruct on bank.json (dev)

**Verdict: NO_LOCK**

Packets `artifacts/receiver_recovery/r3_packets/candidateA_r3.json` (sha256 `3db70ae5cbda…`); bank `artifacts/receiver_recovery/r3_bank/bank.json` (sha256 `c88ff3197717…`, 576 scenes / 1152 worlds). Generated 2026-09-24T13:10:33Z by `scripts/latest0924/sender_gate.py` (sha256 `8fd8227bb403…`).

Integrity: **OK** — header {'bank_sha256_matches': True, 'prompt_sha256_frozen': True, 'inventory_matches_reference': True, 'inventory_size': 221}; nonzero counts none; file summary matches recount: True.

| metric | all | evaluated |
|---|---|---|
| messages (scenes) | 1152 (576) | 1144 (572) |
| parse fully / partially / failed | 1152 / 0 / 0 | 1144 / 0 / 0 |
| parse fully (%) [95% CI] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| fact recall (Table 17) | 4570/4608 (99.2) [98.7, 99.6] | 4538/4576 (99.2) [98.8, 99.5] |
| complete objects (Table 17) | 2269/2304 (98.5) [97.7, 99.2] | 2253/2288 (98.5) [97.7, 99.1] |
| complete packets | 1132/1152 (98.3) [97.4, 99.0] | 1124/1144 (98.3) [97.4, 99.0] |
| wrong-object lanes (words) | 5 (5) | 5 (5) |
| unsupported-attribute lanes | 5 | 5 |
| extra non-attribute lanes | 1466 | 1456 |
| empty lanes | 0 | 0 |
| mean words per object | 2.66 {'1': 5, '2': 862, '3': 1355, '4': 82} | 2.66 {'1': 5, '2': 856, '3': 1345, '4': 82} |
| precision / exact set (B6 defs) | 74.6 / 36.3 | 74.6 / 36.3 |
| mean bytes (Table 17) | 22.6 | 22.6 |
| generated tokens mean / max | 33.866 / 51 | 33.869 / 51 |
| hit max_new_tokens / no final channel | n/a / n/a | n/a / n/a |
| greedy argmax mismatches | n/a | n/a |

Recall counts required attribute words assigned to the correct object; complete objects contain both required facts; extra selected words are allowed (Table 17 definitions, engineering scenes included in `all`). Intervals: scene-clustered percentile bootstrap, 2000 draws, seed 0. Message content only — not a receiver outcome.
