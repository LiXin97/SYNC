# Sender gate — latest0924/gemma-4-26b-a4b-it on exposed_development.json (dev96)

**Verdict: PASS** (population `all`, lock `gates/latest0924/sender_gate.lock.json` sha256 `96990ed60bc4…`)

Packets `artifacts/latest0924/sender_packets/gemma-4-26b-a4b-it_dev96.json` (sha256 `0f45bdee3049…`); bank `artifacts/mechanism0919/discovery_v1/exposed_development.json` (sha256 `97b7622418ba…`, 96 scenes / 192 worlds). Generated 2026-09-24T13:37:04Z by `scripts/latest0924/sender_gate.py` (sha256 `8fd8227bb403…`).

Integrity: **OK** — header {'bank_sha256_matches': True, 'prompt_sha256_frozen': True, 'inventory_matches_reference': True, 'inventory_size': 221}; nonzero counts none; file summary matches recount: True.

| metric | all |
|---|---|
| messages (scenes) | 192 (96) |
| parse fully / partially / failed | 192 / 0 / 0 |
| parse fully (%) [95% CI] | 100.0 [100.0, 100.0] |
| fact recall (Table 17) | 768/768 (100.0) [100.0, 100.0] |
| complete objects (Table 17) | 384/384 (100.0) [100.0, 100.0] |
| complete packets | 192/192 (100.0) [100.0, 100.0] |
| wrong-object lanes (words) | 0 (0) |
| unsupported-attribute lanes | 0 |
| extra non-attribute lanes | 64 |
| empty lanes | 0 |
| mean words per object | 2.17 {'2': 320, '3': 64} |
| precision / exact set (B6 defs) | 92.3 / 83.3 |
| mean bytes (Table 17) | 20.7 |
| generated tokens mean / max | 27.917 / 32 |
| hit max_new_tokens / no final channel | 0 / n/a |
| greedy argmax mismatches | 0 |

| threshold | op | value | observed | pass |
|---|---|---|---|---|
| parse_fully_rate | >= | 0.99 | 1.000 | PASS |
| required_fact_recall | >= | 0.95 | 1.000 | PASS |
| complete_packet_rate | >= | 0.9 | 1.000 | PASS |

Recall counts required attribute words assigned to the correct object; complete objects contain both required facts; extra selected words are allowed (Table 17 definitions, engineering scenes included in `all`). Intervals: scene-clustered percentile bootstrap, 2000 draws, seed 0. Message content only — not a receiver outcome.
