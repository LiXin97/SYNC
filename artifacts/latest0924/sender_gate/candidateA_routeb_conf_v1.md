# Sender gate — candidate_A/llama-3.1-8b-instruct on bank.json (dev)

**Verdict: NO_LOCK**

Packets `artifacts/routeb0922/conf_packets/candidateA_conf.json` (sha256 `3643f9f8101c…`); bank `artifacts/routeb0922/confirmation_bank_v1/bank.json` (sha256 `5ddc27dec7e0…`, 576 scenes / 1152 worlds). Generated 2026-09-24T13:10:37Z by `scripts/latest0924/sender_gate.py` (sha256 `8fd8227bb403…`).

Integrity: **OK** — header {'bank_sha256_matches': True, 'prompt_sha256_frozen': True, 'inventory_matches_reference': True, 'inventory_size': 221}; nonzero counts none; file summary matches recount: True.

| metric | all |
|---|---|
| messages (scenes) | 1152 (576) |
| parse fully / partially / failed | 1152 / 0 / 0 |
| parse fully (%) [95% CI] | 100.0 [100.0, 100.0] |
| fact recall (Table 17) | 4601/4608 (99.8) [99.7, 100.0] |
| complete objects (Table 17) | 2297/2304 (99.7) [99.5, 99.9] |
| complete packets | 1145/1152 (99.4) [99.0, 99.8] |
| wrong-object lanes (words) | 0 (0) |
| unsupported-attribute lanes | 0 |
| extra non-attribute lanes | 1235 |
| empty lanes | 0 |
| mean words per object | 2.53 {'2': 1076, '3': 1228} |
| precision / exact set (B6 defs) | 78.8 / 46.4 |
| mean bytes (Table 17) | 22.1 |
| generated tokens mean / max | 28.865 / 68 |
| hit max_new_tokens / no final channel | n/a / n/a |
| greedy argmax mismatches | n/a |

Recall counts required attribute words assigned to the correct object; complete objects contain both required facts; extra selected words are allowed (Table 17 definitions, engineering scenes included in `all`). Intervals: scene-clustered percentile bootstrap, 2000 draws, seed 0. Message content only — not a receiver outcome.
