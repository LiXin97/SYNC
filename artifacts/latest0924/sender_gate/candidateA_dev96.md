# Sender gate — candidate_A/llama-3.1-8b-instruct on exposed_development.json (dev)

**Verdict: NO_LOCK**

Packets `artifacts/routeb0921/b6a_packets/candidateA_dev96.json` (sha256 `aed8ace8bc37…`); bank `artifacts/mechanism0919/discovery_v1/exposed_development.json` (sha256 `97b7622418ba…`, 96 scenes / 192 worlds). Generated 2026-09-24T13:10:26Z by `scripts/latest0924/sender_gate.py` (sha256 `8fd8227bb403…`).

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
| extra non-attribute lanes | 56 |
| empty lanes | 0 |
| mean words per object | 2.15 {'2': 328, '3': 56} |
| precision / exact set (B6 defs) | 93.2 / 85.4 |
| mean bytes (Table 17) | 20.6 |
| generated tokens mean / max | 27.812 / 32 |
| hit max_new_tokens / no final channel | n/a / n/a |
| greedy argmax mismatches | n/a |

Recall counts required attribute words assigned to the correct object; complete objects contain both required facts; extra selected words are allowed (Table 17 definitions, engineering scenes included in `all`). Intervals: scene-clustered percentile bootstrap, 2000 draws, seed 0. Message content only — not a receiver outcome.
