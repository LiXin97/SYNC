# Sender gate — latest0924/gpt-oss-20b on exposed_development.json (dev96)

**Verdict: PASS** (population `all`, lock `gates/latest0924/sender_gate.lock.json` sha256 `96990ed60bc4…`)

Packets `artifacts/latest0924/sender_packets/gpt-oss-20b_dev96.json` (sha256 `f6cc37dec857…`); bank `artifacts/mechanism0919/discovery_v1/exposed_development.json` (sha256 `97b7622418ba…`, 96 scenes / 192 worlds). Generated 2026-09-24T13:38:23Z by `scripts/latest0924/sender_gate.py` (sha256 `8fd8227bb403…`).

Integrity: **OK** — header {'bank_sha256_matches': True, 'prompt_sha256_frozen': True, 'inventory_matches_reference': True, 'inventory_size': 221}; nonzero counts none; file summary matches recount: True.

| metric | all |
|---|---|
| messages (scenes) | 192 (96) |
| parse fully / partially / failed | 191 / 0 / 1 |
| parse fully (%) [95% CI] | 99.5 [98.4, 100.0] |
| fact recall (Table 17) | 733/768 (95.4) [92.7, 97.8] |
| complete objects (Table 17) | 351/384 (91.4) [86.5, 95.8] |
| complete packets | 173/192 (90.1) [84.4, 94.8] |
| wrong-object lanes (words) | 0 (0) |
| unsupported-attribute lanes | 0 |
| extra non-attribute lanes | 24 |
| empty lanes | 2 |
| mean words per object | 1.97 {'0': 2, '1': 18, '2': 353, '3': 11} |
| precision / exact set (B6 defs) | 96.8 / 88.5 |
| mean bytes (Table 17) | 19.9 |
| generated tokens mean / max | 99.802 / 124 |
| hit max_new_tokens / no final channel | 0 / 0 |
| greedy argmax mismatches | 0 |

| threshold | op | value | observed | pass |
|---|---|---|---|---|
| parse_fully_rate | >= | 0.99 | 0.995 | PASS |
| required_fact_recall | >= | 0.95 | 0.954 | PASS |
| complete_packet_rate | >= | 0.9 | 0.901 | PASS |

Recall counts required attribute words assigned to the correct object; complete objects contain both required facts; extra selected words are allowed (Table 17 definitions, engineering scenes included in `all`). Intervals: scene-clustered percentile bootstrap, 2000 draws, seed 0. Message content only — not a receiver outcome.
