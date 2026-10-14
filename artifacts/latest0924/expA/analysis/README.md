# Experiment A analysis (latest0924)

Frozen analysis of Experiment A under `gates/latest0924/expA_protocol.lock.json`.
Script: `scripts/latest0924/analyze_expA.py`. It is CPU-only (numpy + stdlib), imports nothing from the runner, and never imports torch.

## Run

```bash
cd .
python3 scripts/latest0924/analyze_expA.py
# -> artifacts/latest0924/expA/analysis/expA_results.json and expA_results.md   (about 30 s on the dev box)
```

Options (for tests only): `--lock`, `--expa-root`, `--adr-glob`, `--out-dir`.
- The script is safe to run while cells are still running. A cell counts only when every development shard has a valid receipt. A cell without one is reported as INCOMPLETE, is treated as NOT RUN for inference, and is never extrapolated.
- Outputs are a deterministic function of the input files, apart from the `generated_utc` field. `expA_results.json` → `inputs` records the sha256 and byte count of every file read: the lock, manifests, pinned inputs, the bank, sender packets, sender gates, the ADR, raw rows, receipts and failure notes. It records `script.sha256` as well.
- If the lock no longer contains the constants copied into the script (seeds, 20,000 draws, 0.50 / 0.90, [-3, +3], 0.05, ...), the script exits with code 3 and writes nothing.

## Cell status vocabulary

| status | meaning |
|---|---|
| COMPLETE | Engineering gate re-checked and passed, all development shards receipted, key set exact, integrity audit passed. This is the only status that is estimated. |
| INCOMPLETE - NOT RUN per lock | Rows present (shown) but not every shard has a valid receipt. Not estimated. |
| NOT RUN (no development rows) | The cell has a manifest but no development rows. |
| ENGINEERING FAILURE | The receiver failed the engineering gate in any sender's cell (re-computed from the rows, or an ENGINEERING_FAILED marker with its note). Receiver-level: no outcome rows under any sender. `(ADR-latest0924-moe-gate)` marks qwen3.5-122b-a10b, which is judged under the amended gate. |
| AUDIT_FAIL | Integrity check failed: an invalid receipt, a changed pin, a stored score ≠ re-parse, identity write not inert, a reassignment norm mismatch, and so on. The reasons are in the completeness table and the JSON. Not estimated. |
| NOT_BUILT | The lock lists the pair but no manifest exists. |

A `.done.json` receipt is never taken as proof that the engineering gate passed. Every engineering row is re-checked:
- **Default gate:** max logit difference ≤ 0.05, alias-gap difference ≤ 0.1, and equal parsed answers.
- **qwen3.5-122b-a10b (ADR gate):** equal parsed answers and alias-gap difference ≤ 0.1.
- **Every cell:** identity_write rows must reproduce the none rows' tokens and logits_sha256.

## Tables in `expA_results.md`

- **(a) Main table, one per sender (cross-model receivers only).**
  - J(selected) is the primary endpoint: four-answer success at 64 tokens, with invalid answers counted wrong, family-macro over the 36 families. The interval is a 95% percentile cluster bootstrap enclosing all three seeds.
  - J(text) is the same message as `packet_text`. J(original) is the old middle-band write.
  - *Reassigned J range / coverage* gives the min–max J over sham0–sham2 and their mean coverage. coverage(selected) is the fraction of answers that parse to an alias.
  - *Selected − mean reassigned* is the scene-paired H1 contrast. Its interval encloses both clustering levels and all seeds. p is the larger two-sided bootstrap p of the two levels. "thr" is the Holm threshold across that sender's evaluated receivers.
  - **H1** = RECOVERY iff J ≥ 0.50, coverage ≥ 0.90, Holm rejects, and lower limits > 0 at both levels. When it fails, the failed criteria are listed.
  - *Selected − text* and **H2** (internal better / text better / similar / unresolved) follow the lock's H2 rule.
  - Below each table is a sensitivity line: verdicts recomputed with Holm m = all planned cross-model receivers, with unevaluated receivers entering at p = 1.
- **(a′) Same-model pairs.** Reported separately with unadjusted p and no Holm. They are excluded from every cross-model count and Holm family.
- **(b) Engineering audit per cell.**
  - The table gives: rows vs expected, receipt validity, max logit difference (and where it occurs), max alias gap, parsed-unequal rows, rows failing the applicable gate, stored `pass` flags that disagree with the re-computation, identity rows that are not inert, and the verdict.
  - Each engineering-failure receiver is followed by its failure note, verbatim, and the numbers re-computed from its rows under the lock gate, the lock-literal gate and the ADR gate.
- **(c) Scale ladder (H3).**
  - **H3 primary** is gap(122B) − gap(2B). It is NOT EVALUABLE while qwen3.5-122b-a10b is an engineering failure, and no other model is substituted.
  - The ladder table gives each point's gap = J(selected) − J(packet_text).
  - The Spearman correlation between log parameter count and gap is DESCRIPTIVE ONLY and is computed only when at least 3 points exist.
  - gap(qwen3.5-27b) − gap(qwen3.5-2b) is shown with intervals and labelled NOT the preregistered primary.
- **(d) Completeness per cell.** Status, engineering verdict, receipted vs present shards (declared S), rows present vs expected (572 scenes × 4 × arms), and notes, including errors and warnings.
- **(e) Complete facts.** From source `literal` in the first-sender (Gemma) cells: selected vs native vs packet_text vs full_text, descriptive.
- **(f) H4 depth.** Selected − original per cell, descriptive, with family, template and enclosing intervals.
- **(g) Other endpoints.** For each complete cell and arm: J, single-answer accuracy, coverage, and the changed component R (the changed-field question right in both worlds). Also the unchanged component U, and J and coverage at 256 tokens. `other_world` appears twice: scored against its own gold, and against the supplied (other world's) gold. The second is the lock's consistency check, and it should equal J(selected).
- **Resolutions.** Every point where the lock was ambiguous, and how this script resolved it. For example, the 12 parent templates are domain × template, because the bank's `template` field has only 4 values.

## Validation performed when the script was written (2026-09-24)

- **Independent recount on real rows.** A separate snippet computed values from the stored `score64` fields and the rows' own family, domain and template fields. It agreed to 1e-12 on:
  - J at family and template level, and coverage, for 5 (cell, arm) pairs;
  - the family CI of J(selected);
  - the means, CIs and p values of two contrasts at both levels.
- **Synthetic trees with planted outcomes, built outside the repo.** The script recovered:
  - every planted J, coverage and template value exactly;
  - every H1/H2 verdict, the Holm m, the Spearman ρ, and gap(27B) − gap(2B);
  - the statuses INCOMPLETE, NOT_BUILT and NOT_STARTED, and ENGINEERING FAILURE under the ADR gate;
  - receiver-level propagation of an engineering failure.

  In a fault-injection tree it detected:
  - a receipt that says PASS over an engineering row with a logit difference of 0.07;
  - an identity write that is not inert (engineering and development);
  - a receipt raw-sha mismatch, a stored-score mismatch, a changed pin, and a reassignment norm mismatch.
