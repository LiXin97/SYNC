# Word-interface follow-up — DELIVERY NOTE (results_package_v1, 2026-09-23)

Scope: the two P0 packages of `the study protocol` only (PI-authorized 2026-09-23). Four receivers at their
frozen depth-0.2 layer (qwen3-32b L13, qwen3-14b-base L8, glm-4-9b L8, gemma-3-27b-it L12), raw fitted write, the frozen
prompted senders (candidate A two-object; candidate A4 four-object). Every number below is read from this package's
audited tables (`conditions_long`, `contrasts_long`, `claim_gate.json`); raw rows are indexed in `predictions_index.json`.
J = all four answers of a scene correct (2 worlds × 2 questions) at the 64-token budget; failed sender parses and invalid
receiver outputs stay in the denominator. Intervals are scene-paired cluster-bootstrap envelopes (20,000 draws × 3 seeds);
"conservative" = the worse of the two declared cluster levels (families/groups and templates, analysed separately).

## Status

| Stage | Rows | Audit | Independent recount |
|---|---|---|---|
| P0-A development (exposed bank, dose selection only) | 4 × 22,080 | PASS | — (dry-run 0 differences) |
| P0-A confirmation (new 576-scene two-object bank) | 32,256 / 32,256 / 41,472 / 32,256 | PASS | 0 estimate / interval / Holm differences; 2 p-value entries differ by 1.0e-4 from floating-point ties at zero — adjudicated, see below |
| P0-B development (96 scenes) | 4 × 5,760 | PASS | — (dry-run 0 differences) |
| P0-B evaluation (new 576-scene conjunction bank) | 4 × 34,560 | PASS | PASS, 0 differences |

Every lock (`locks/`) was committed before the outputs it governs. Nothing was cut; all receivers, including failures, are
in the tables.

## P0-A — does averaging twelve contexts matter? (confirmation, 576 new scenes)

J at the selected doses (γ = 1 everywhere except glm single_t0 γ = 4, glm emb1 γ = 2):

| Receiver | source | twelve-context | single t0 | single t6 | single t11 | packet as text (= emb1@1) | evidence text |
|---|---|---|---|---|---|---|---|
| qwen3-32b | automatic | 0.953 | 0.872 | 0.951 | 0.953 | 0.948 | — |
| | complete facts | 0.988 | 0.946 | 0.988 | 0.986 | 0.993 | 0.965 |
| qwen3-14b-base | automatic | 0.990 | 0.970 | 0.991 | 0.990 | 0.993 | — |
| | complete facts | 0.991 | 0.972 | 0.998 | 0.990 | 1.000 | 1.000 |
| glm-4-9b | automatic | 0.806 | 0.099 | 0.828 | 0.835 | 0.872 (emb1@2 0.889) | — |
| | complete facts | 0.764 | 0.089 | 0.755 | 0.767 | 0.927 (emb1@2 0.932) | 0.986 |
| gemma-3-27b-it | automatic | 0.993 | 0.997 | 0.993 | 0.993 | 0.991 | — |
| | complete facts | 1.000 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |

Holm families (four receivers each, conservative p):
- **A1, twelve − mean(three singles):** significant on glm (automatic +0.218, conservative lower bound +0.171; complete facts
  +0.227, +0.173) and qwen3-32b (+0.028, +0.010; +0.014, +0.006); not on qwen3-14b-base (+0.006, p 0.031 > Holm
  threshold; +0.005) or gemma (−0.001; +0.001) — both at ceiling.
- **What drives A1:** the first-position context t0 alone. twelve − t0 is +0.707 / +0.675 on glm, +0.082 / +0.042 on
  qwen3-32b; twelve − t6 and twelve − t11 are within ±0.03 and non-significant on every receiver. Reading: a single
  well-placed context reproduces the twelve-context direction; averaging protects against a bad single context.
- **A2, twelve − embedding:** on this bank every active word is one token in its field context for all receivers (576/576),
  so emb1@1 is *exactly* the packet-as-text route (full token-span embedding replacement = text, identity-checked at
  |Δlogit| ≤ 1e-4, measured 0.0). A2 at γ = 1 is therefore twelve-context versus text and is labelled so. Significant only
  on glm and **negative** (automatic −0.083, complete facts −0.168: the text/embedding route beats the internal write);
  n.s. elsewhere. glm's emb1@2 (the only non-identical embedding cell) is not different from text (+0.017, n.s.).
- **Not tested here:** the multi-token mean-pooling adaptation never fired (no multi-token word in either bank); it is not a
  complete-word representation and no claim is made about it.

## P0-B — conjunction queries, four objects (evaluation, 576 new scenes)

Each single colour and each single material is held by exactly two objects (asserted per scene); only the colour +
material conjunction identifies the answer.

| Receiver | automatic: internal | text | evidence | reassignments (max) | valid | other world → its own gold | complete facts: internal / text |
|---|---|---|---|---|---|---|---|
| qwen3-32b | **0.953** (549) | 0.941 | 0.986 | 0.000 | 0.995 | 0.953 | 0.969 / 0.977 |
| qwen3-14b-base | **0.957** (551) | 0.969 | 1.000 | 0.000 | 1.000 | 0.957 | 0.977 / 0.984 |
| glm-4-9b | **0.418** (241) | 0.703 | 0.877 | 0.000 | 0.991 | 0.418 | 0.446 / 0.892 |
| gemma-3-27b-it | **0.944** (544) | 0.977 | 0.997 | 0.002 | 1.000 | 0.944 | 0.951 / 0.990 |

- Internal − mean reassignment: +0.953 / +0.957 / +0.418 / +0.944 (conservative lower bounds +0.929 / +0.934 / +0.361 /
  +0.921); Holm rejects in both families (B_actual, B_literal) for all four receivers.
- **Pre-declared gate (J ≥ 0.5, valid ≥ 0.9, positive lower bound, Holm):** PASS qwen3-32b, qwen3-14b-base, gemma-3-27b-it;
  **FAIL glm-4-9b** (J 0.418 < 0.5 although the contrast is significant). Recount reproduces all four verdicts.
- Internal vs text (automatic, descriptive): qwen3-32b +0.012 and qwen3-14b-base −0.012 (n.s.); gemma −0.033 (text better,
  significant); glm −0.285 (text much better). No-message J = 0 is structural.
- Sender content (evaluation, 1,152 packets): parse 1,150 ok / 2 partial; fact recall 0.9986; complete packets 0.991
  (1,142); 4 packets place a value in another object's lane. glm's shortfall is on the receiver side (its complete-fact
  internal write is 0.446 vs text 0.892).
- 256-token budget: identical J for the internal route on every receiver.
- Scope: controlled conjunctive use of a finite, reused vocabulary; not general reasoning, relational communication or
  agent-task utility.

## Deviations and incidents

1. **P0-A report fix before any development row:** the emb1 edit-norm report read 0 (FP32 `.float()` view aliasing); the
   edit itself was applied. Runner fixed (reporting only), manifests rebuilt, amendment in `p0a_dev.lock.json`; the partial
   engineering rows were preserved as `*.superseded_reportbug`.
2. **Two node incidents, no data lost:** a spot pre-emption (12:21–12:39Z) and my dispatcher crash (15:03–15:26Z, a
   job-list rewrite read mid-rename on the shared mount; ≈3 GPU-h lost). Every raw file was checked to end on complete
   lines; runners resume and re-verify recomputed rows. A duplicate race candidate ran ≈3 min (11:03–11:06Z) with no work.
3. **P0-A recount:** the first recount ordered clusters by sorted key (the R3/R4/P0-B convention) while the P0-A audit
   uses first appearance — 154 bound/p-only differences, 0 estimate or decision differences (kept as
   `p0a_confirm_recount_sortedkey_sensitivity.json`). With the audit's order, 2 p entries (qwen3-14b-base complete-fact A1,
   template level, p ≈ 0.398) still differ by 1.0e-4: 1,609 of 60,000 bootstrap means are 0 up to 1e-17 and the two
   summation orders leave 3 vs 0 exact zeros. Verified by `scripts/verify_p0a_recount_ties.py`; adjudication in
   `p0a_confirm_recount_adjudication.json`. The recount status string is left as produced; no tolerance was changed.
4. Scheduling-only reorders of the job list (longest shards first); no protocol, dose or bank change after any output.

## Resources

One 8-GPU node at a time (one spot pre-emption, 12:21–12:39Z); the node was released after the last cell.
