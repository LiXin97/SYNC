# Experiment B: graded weights, internal vs text (preregistration DRAFT)

> **Status: DRAFT, not binding.** This protocol becomes binding only when the controller writes
> `gates/latest0924/expB_graded.lock.json` **before the first development row of any receiver**. The lock must hold the
> sha256 of this file, `bank.json`, the five `packets_L*.json`, `construction_report.json`, the builder, runner and
> analysis scripts, the adapter, and each receiver's `manifest.json`, plus the tier set and cost cap chosen under
> §8. Engineering rows can exist before the lock because they are excluded from every estimate. After the lock,
> changes need a dated decision record, and only before the affected test. Missed thresholds are reported as
> misses.
>
> Prepared 2026-09-24 on CPU. **Nothing here has run on a GPU.** The Qwen3-32B CPU dry-run reproduces the paper's
> R3 prompts, write positions and packets exactly (§9). No row of this experiment exists.

## 1. Question

Every message in the paper uses weight 255, and the paper says that graded confidence remains untested. With complete
facts and no sender, suppose a message gives an object two competing values for one attribute with unequal 8-bit
weights. Does the receiver follow the stronger weight more reliably when the weights arrive as **write magnitudes**
(internal route) or as **printed numbers** (text route)?

## 2. Design

### 2.1 Scenes
- Source: `artifacts/receiver_recovery/r3_bank/bank.json` (sha256 `c88ff319…`), the paper's two-object confirmation
  bank. It has 576 scenes: 36 families × 16 scenes, 12 parent templates and 3 domains (candy colour/taste, block
  paint/material, coat fabric/buttons). In world 1, one attribute X (the bank's `changed_field`) is swapped between
  objects A and B, and the other attribute Y is unchanged.
- The four engineering scenes of the R3 manifest are excluded from every estimate: `000ab0c6e25c722375c5`,
  `006580f74136c64f5667`, `013a221b69bb9e33fe59` and `01c022ca70dbfbd8c0a1`. That leaves 572 evaluated scenes.
- **Evaluation set: tiers 1+2 = 144 scenes (4 per family, 12 per template, 48 per domain).** Tiers depend only on
  each scene's design bits (changed attribute, the two flips, and the queried value), never on an outcome.
  `build_graded_bank.py::select_tiers` states the rule:
  - The candidates are 4-point patterns of the 2^4 within-family design. A pattern must be balanced on all eight
    design columns: c, fa, fb, q, the X/Y flips fX/fY, and the world-0 answer objects sX/sY. It must also avoid the
    two design points that the engineering scenes occupy.
  - Among those, take the pattern with the most fully crossed column pairs; ties go to the lexicographically
    smallest.
  - Split that pattern into two 2-point halves, balanced first on c, then sY, q and fX. The halves are tiers 1 and 2.
  - Tier 3 is the best disjoint second pattern (144 scenes). Tier 4 is the remaining 284 scenes.
  - The resulting tiers 1+2 are balanced 72/72 on every column. Each domain × changed-attribute cell has 24 scenes.
- Why a subset: §8 gives the costs. All 572 scenes would cost about 5.5 GPU-h per 32B-class receiver, above the
  ~2 GPU-h cap per receiver. §8 fixes the fallback (tier 1 only) and the extension (tiers 1–3) in advance.

### 2.2 Messages (complete facts, one packet per world, one per level)
For level L ∈ {0, 64, 128, 191, 255}, object *o* carries:
- its true X value at weight 255;
- the **other** object's X value at weight L (omitted when L = 0);
- its Y value at weight 255.

Words are listed in ascending inventory index, as in the frozen packet format. Serialization is
`syncomm.packet.serialize` (qbits 8), and the file is byte-identical to the frozen snapshot copy. Example (tier 1,
scene `1521805b851ccbda9306`, "Which object has a red color?" / "Which object is salty?"):

| L | world 0 (gold X = BANQ) | world 1 (gold X = FELK) |
|---|---|---|
| 0 | BANQ: red, sweet · FELK: salty, yellow | BANQ: sweet, yellow · FELK: red, salty |
| 128 | BANQ: red 255, sweet 255, yellow 128 · FELK: red 128, salty 255, yellow 255 | BANQ: red 128, sweet 255, yellow 255 · FELK: red 255, salty 255, yellow 128 |
| 255 | BANQ: red, sweet, yellow · FELK: red, salty, yellow (all 255) | byte-identical to world 0 |

Properties checked by the builder (`construction_report.json`, all PASS):
- **L = 0** is byte-identical to the paper's complete-fact packet (1,152/1,152).
- **0 < L < 255:** the two worlds carry the **same words in the same slots**, and only the two X weights swap
  (1,728/1,728). In every world exactly one of the two lanes lists its strong X value first, so word order carries no
  information about strength (3,456/3,456).
- **L = 255:** the two worlds' packets and texts are byte-identical (576/576), so X is ambiguous by construction.
- **Weight gold:** the object with the strictly larger weight for the queried value always equals the bank's gold.
  The only exception is the X question at L = 255, where the weights tie. Each row re-checks this.
- The renderer `render_weighted` is byte-identical to the frozen `memory_layout.render`, in both text and character
  spans, whenever every weight is 255. This held in 6,880 cases: every scene, every world, the neutral layout, and
  4,000 seeded random word lists. With graded weights it differs from `render` only in the two weak slots' digits
  (3,456/3,456).

### 2.3 Arms (per scene × world; four answers per arm per scene)

| arm | receiver input | notes |
|---|---|---|
| `none` | neutral prefix: four `unknown [255/255]` slots per object | structural J = 0 |
| `identity_write` | the captured neutral states written back at all eight slots | engineering scenes only; must be exactly inert |
| `internal_L{L}` | neutral prefix; at the depth-0.2 layer, h' = h⁰ + γ·(q_w/q_max)·d_w at each active slot's final token, γ = 1, raw fitted directions | the visible prompt is the neutral layout, with a constant `[255/255]` and `unknown` in every slot, so no weight text leaks. q_max = 255 always, so the weak word is written at L/255 |
| `text_L{L}` | the same words, rendered with their **actual** weights `[w/255]` in the same slots; no write | uses the header "Selected words and exact relative weights (integers divided by 255):" |
| `sham0_L128` | `internal_L128` with each word's direction replaced by another word's (the R3 permutation 0, seed 202609201, a derangement), and the per-layer edit norm matched to `internal_L128` | one level only; §5 explains why |

Development rows per scene: 12 arms × 4 answers = 48. Questions, prompts, parser (strict first line) and budgets (64
primary, 256 sensitivity) are unchanged from the paper. The bank's question wording is also unchanged, e.g. "Which
object has a red color?".

### 2.4 Receivers

| key | checkpoint | layer (depth 0.2) / blocks | directions | python on node | GPUs | adapter |
|---|---|---|---|---|---|---|
| qwen3-32b (continuity) | Qwen3-32B | 13 / 64 | `artifacts/receiver_recovery/fits/directions_qwen3-32b.pt` (paper R3) | `/usr/bin/python3` (transformers 5.3.0) | 1 | optional (frozen loader; the adapter's cache functions are byte-identical on dense caches) |
| qwen3.5-27b | Qwen3.5-27B | 13 / 64 | `artifacts/latest0924/fits/directions_qwen3.5-27b.pt` (exists) | tf-latest (5.12.1), required by the adapter | 1 | required |
| qwen3.5-122b-a10b | Qwen3.5-122B-A10B | 10 / 48 | `…/directions_qwen3.5-122b-a10b.pt` (**pending**) | tf-latest | 4 (device_map auto, 150 GiB cap) | required |
| gpt-oss-20b | gpt-oss-20b | 5 / 24 | `…/directions_gpt-oss-20b.pt` (**pending**) | `/usr/bin/python3` | 1 | required (harmony prompt, final channel pre-opened, MXFP4 dequantized, FP32) |

Directions come from the frozen twelve-context recipe (`src/syncomm/recovery/fit_layers.py`). Numerics are FP32 eager
with TF32 off, batch 1 and greedy decoding. Every receipt records the adapter's per-model load report: dtype,
placement, stop tokens and environment.

## 3. Endpoints

- **Primary (per receiver):** Δ = mean over L ∈ {64, 128, 191} of [J_internal(L) − J_text(L)].
  - J is **four-answer success**: all of (world 0, world 1) × (X question, Y question) correct at 64 tokens. Invalid
    or truncated answers count as errors.
  - Δ is paired by scene and averaged family-macro over the families present.
- **Secondary (descriptive, no multiplicity correction):**
  1. Per-level curves per route: J, coverage, accuracy, R (X question right in both worlds), U (Y question right in
     both worlds), X and Y accuracy, and J at 256 tokens.
  2. Per-level internal − text.
  3. **Difference-in-differences** vs L = 0: mean over the pooled levels of
     [(J_int(L) − J_int(0)) − (J_text(L) − J_text(0))]. This separates a gap in reading the weights from a delivery
     gap that already exists without them (e.g. GLM's +13.5-point text advantage in the paper).
  4. The graded cost per route, mean over the pooled levels of J(L) − J(0).
  5. internal − sham0 at L = 128.
  6. Qwen3-32B continuity: L = 0 rows vs the paper's R3 `literal/selected` and `literal/packet_text` rows. This
     reports identical prompts, tokens and logits hashes.

## 4. Statistics
- Cluster bootstrap over the 36 families and, separately, over the 12 parent templates, keeping each scene intact:
  - 20,000 draws for each of the seeds 20260924, 20260925 and 20260926 (the Experiment A seeds);
  - the 95% percentile interval enclosing all three seeds.
- For each contrast, the reported interval encloses both cluster levels, and p is the larger of the two-sided
  bootstrap p values.
- **Holm** at familywise α = 0.05 across the fixed family of four receivers {qwen3.5-27b, qwen3.5-122b-a10b,
  gpt-oss-20b, qwen3-32b} on the primary contrast. A receiver that is not run stays in the family with p = 1.
- Expected precision, from a CPU sketch with independent Bernoulli scene outcomes on the real 144-scene cluster
  structure (enclosing half-width):

  | scenario | 144 scenes | 72 scenes | 288 scenes |
  |---|---|---|---|
  | internal .6 vs text .9 | ≈ 6 points | ≈ 8 | ≈ 4 |
  | both .9 | ≈ 4 | ≈ 6 | ≈ 3 |
  | both .98 | ≈ 2 | ≈ 2.4 | ≈ 1.3 |
  | both .5 | ≈ 7 | ≈ 10 | ≈ 5 |

  Differences of about 10 points or more are therefore detectable. "Similar within ±3" is reachable only when both
  routes are near ceiling (or floor).

## 5. Controls, analytic floors and thresholds

- **Engineering gate** (per receiver; the 4 engineering scenes; every arm plus `identity_write`). A receiver that
  fails is an engineering failure with no development rows.
  - The identity write reproduces the `none` logits hash and tokens exactly, and its realized edit norm is 0.
  - A cached-prefix continuation vs a complete forward: max logit difference ≤ 0.05, alias-gap difference ≤ 0.1,
    and equal parses at both budgets.
  - The L = 255 outputs are identical across worlds.
  - For every internal slot in FP32, the requested edit norm equals (q_w/q_max)·‖d_w‖ within 1%.
- **No message:** J = 0 is structural. Both worlds see identical input.
- **Positive control, L = 0** (the paper's complete-fact condition): **J ≥ 0.90 and coverage ≥ 0.90, per route**.
  In R3, Qwen3-32B reached J = 0.998 on both routes.
- **Negative control, L = 255, with an exact analytic floor.** Both worlds receive byte-identical messages on both
  routes. A deterministic receiver therefore gives the same X answer in both worlds, and the two golds differ, so:
  - **J = 0 and R = 0 exactly**;
  - X accuracy is ≤ 1/2 in every scene (exactly 1/2 when the X answer is valid);
  - only the Y answers can be right in both worlds.

  Threshold: J_sum = R_sum = 0 on both routes, and every L = 255 answer is identical across worlds. Any violation is
  an integrity failure (nondeterminism or leakage). It is reported, and the receiver is marked not interpretable.
- **Floor for ignoring the weights (analytic, 0 < L < 255).** The two worlds' messages differ only in the weights. A
  receiver that ignores them gives world-identical X answers, so R = J = 0. A receiver whose X answer flips at random
  per world has E[R] = 1/4 and E[J] ≤ U/4. So J clearly above U(L)/4 cannot come from weight-blind behaviour.
- **Reassignment control (`sham0_L128`).** At graded levels the X words are the same in both worlds, so internal
  success can only come from word-specific magnitude reading. Reassignment keeps the magnitude pattern, the slots
  and the edit norm but breaks word identity. J should be about 0, far below `internal_L128`. It checks at one level
  that graded internal success depends on the specific words' directions (the paper checked this only at weight
  255). Cost: +4 answers per scene (+8%). It is reported descriptively and does not enter the primary decision.

## 6. Decision rules (per receiver, primary Δ, enclosing interval, Holm)

| call | rule |
|---|---|
| **internal better than text** | Holm rejects AND enclosing lower limit > 0 |
| **text better than internal** | Holm rejects AND enclosing upper limit < 0 |
| **similar** | enclosing interval inside [−3, +3] points (no equivalence claim beyond this margin) |
| **unresolved** | otherwise |

- **Interpretability.** A call counts as evidence about conveying weights only if all of these hold:
  - the engineering gate passed;
  - both routes pass the L = 0 positive control;
  - the L = 255 and no-message structural zeros hold.

  Otherwise the numbers are still reported, labelled "not interpretable (failed: …)", and the receiver stays in the
  Holm family.
- **Reading the primary with the DiD** (descriptive):
  - If the primary favours one route but the DiD interval contains 0 and the L = 0 contrast already shows that
    advantage, report it as a delivery gap, not a difference in weight reading.
  - If the DiD excludes 0 in the same direction, the weights themselves are read better on that route.
- **Across receivers:** report the count of receivers per call. There is no pooling across receivers and no
  scale-trend claim from four receivers.

## 7. What cannot be concluded
- This is not a sender test. The facts are complete and the weights are set by construction.
- It says nothing about whether weights are **calibrated** confidence. It tests a single two-value competition for
  one attribute, with one weak level per packet.
- The internal route's visible layout says `[255/255]` for every slot, a constant cue that may compete with the write
  magnitude. The internal route cannot be separated from this layout choice. Likewise the text route uses one format,
  the integers /255 under the header "exact relative weights"; percentages, words and other formats are untested.
- There is no equivalence claim beyond ±3 points. At 144 scenes, "similar" is attainable only near ceiling or floor
  (§4).
- The evaluation set is a pre-declared, balanced subset of a finite, constructed bank that has already been evaluated
  with other receivers (disclosed). The results are not population estimates.
- Routes and levels are scene-paired repeated measures and add no independent N. Bootstrap seeds are not model
  replications.
- There is no bandwidth, latency or privacy claim. The rows carry wire bytes: 20 B (L = 0) or 24 B (L > 0) internally,
  vs ≈ 213–229 B as text.
- For Qwen3.5-122B-A10B, the 4-GPU device map and any adapter deviation are part of that receiver's condition.

## 8. Cost, tier rule and execution

**Rows** (tiers 1+2): 144 × 48 = 6,912 development rows, plus 4 × 52 = 208 engineering rows. An engineering row costs
about 2× a development row because of the complete-forward check. Measured: R3 Qwen3-32B took 1.38 vs 0.70 s/row, and
R4 took 1.84 vs 1.05.

| receiver | s/row basis | GPUs | tiers 1+2 (144) | tier 1 (72) | all 572 |
|---|---|---|---|---|---|
| qwen3-32b | **measured 0.698** (R3 receipts: 22,370 s / 32,032 rows) | 1 | **≈ 1.5 GPU-h** (1.34 dev + 0.08 eng + load) | ≈ 0.8 | ≈ 5.5 |
| qwen3.5-27b | not measured; assume 0.7–1.4 | 1 | ≈ 1.5–2.9 | ≈ 0.8–1.5 | ≈ 5.5–11 |
| gpt-oss-20b | not measured; assume 0.3–0.9 | 1 | ≈ 0.7–1.9 | ≈ 0.4–1.0 | ≈ 2.4–7 |
| qwen3.5-122b-a10b | not measured; assume 1–2.5 s wall | **4** | **≈ 8–20** | ≈ 4–10 | ≫ |

**Tier rule** (declared now; it uses only a receiver's engineering receipt, never an outcome):
- r̂ = the engineering `seconds_per_computed_row` / 2.
- Projected cost P(T) = n_GPUs × (rows(T) × r̂ + 1,800) / 3,600 GPU-h.
- Run tiers 1+2 if P ≤ C; else tier 1 if P ≤ C; else the receiver is **NOT RUN** in Experiment B (Holm p = 1).
- C = 2.0 GPU-h per receiver unless the lock sets another value. **Qwen3.5-122B-A10B will almost certainly miss
  C = 2.0 on 4 GPUs. Whether to raise C for it is a PI decision to record in the lock.**
- The formal-task rule applies: a task averaging below 30% GPU utilization over 3 h is killed. A 4-GPU device-map job
  may sit near that line, so co-schedule it with single-GPU jobs on the same task.

**Execution** (absolute paths; formal node; `J=.`,
`SYS=/usr/bin/python3`, `TFL=python`; `PY` = `SYS` for qwen3-32b
and gpt-oss-20b, `TFL` for the two Qwen3.5 receivers):

1. When a receiver's directions exist **and the adapter is final**:
   ```
   PYTHONPATH=$J/src $PY -m syncomm.latest.evaluate_graded make-manifest --receiver KEY [--tiers 1 2]
   ```
   This writes `$J/artifacts/latest0924/graded/KEY/manifest.json` with status READY_FOR_LOCK. It refuses to
   overwrite a different manifest. Optional CPU re-check:
   ```
   … dry-run --manifest … --report …/KEY/dryrun.json [--compare-r3 $J/artifacts/receiver_recovery/r3/qwen3-32b]
   ```
2. Controller: write the lock (see the banner above).
3. Engineering, one GPU (four for 122B):
   ```
   mkdir -p $J/artifacts/latest0924/graded/KEY/logs
   CUDA_VISIBLE_DEVICES=g PYTHONPATH=$J/src $PY -m syncomm.latest.evaluate_graded run --manifest $J/artifacts/latest0924/graded/KEY/manifest.json --phase engineering > $J/artifacts/latest0924/graded/KEY/logs/eng.log 2>&1
   ```
   The gate must pass, then apply the tier rule.
4. Development, sharded; resumable, re-launch the same line:
   ```
   … run --manifest … --phase development --shard s --shards S > …/logs/dev_s{s}.log 2>&1
   ```
5. Analysis (CPU):
   ```
   $SYS $J/scripts/latest0924/analyze_graded.py --roots $J/artifacts/latest0924/graded/{qwen3-32b,qwen3.5-27b,qwen3.5-122b-a10b,gpt-oss-20b} --output $J/artifacts/latest0924/graded/analysis.json --markdown $J/artifacts/latest0924/graded/analysis.md --r3-root qwen3-32b=$J/artifacts/receiver_recovery/r3/qwen3-32b
   ```

## 9. Verification done on CPU (2026-09-24), and what remains GPU-only

- **Builder:** all checks in §2.2 PASS. The build is deterministic: a rebuild reports every file identical.
- **Dry-run on the Qwen3-32B tokenizer, no weights:** 148 scenes (144 + 4 engineering) and 7,120 planned rows.
  - Every L = 0 internal answer (592/592) has the same prompt tokens, question tokens, write positions, active entries
    and packet hex as the paper's R3 `literal/selected` rows.
  - Every L = 0 text prompt (592/592) is token-identical to R3 `literal/packet_text`, and every neutral prompt
    (592/592) to R3 `none`.
  - Outside the message block, text-route prompts are token-identical to the neutral prompt.
  - Each slot's tokens contain exactly its word.
  - The L = 255 text prompts are token-identical across worlds.
- **Dry-runs on the other three tokenizers, through the adapter's own tokenizer loader:** Qwen3.5-27B, Qwen3.5-122B-A10B
  and gpt-oss-20b all pass. gpt-oss is rendered in the harmony format with the final channel pre-opened. Every
  receiver has eight single-token ` unknown` slots. Neutral prefixes are 182–186 tokens (Qwen3), 191–195 (Qwen3.5)
  and 206–210 (gpt-oss).
- **Across transformers versions:** for each of the four receivers, 2,664 prompts rendered and tokenized under 4.57.6
  (the dry-run environment) and under 5.12.1 (the node's tf-latest version) are identical in both text and token IDs
  (2,664/2,664 each). The node's 5.3.0, which gpt-oss and Qwen3-32B use, was not checked directly. The R3 comparison
  above covers 5.3.0 for Qwen3-32B.
- **CPU smoke test of `scene_rows`**, with a tiny random-weight Qwen3-architecture model, the real tokenizer and the
  adapter installed. This is **not a real receiver**, and its answers are meaningless:
  - 4 engineering scenes (208 rows) and 3 development scenes (144 rows) ran through the runner's own code;
  - every engineering check passed (max cached-vs-complete logit difference 4.8e-7);
  - the identity write was exactly inert;
  - L = 255 outputs were identical across worlds (28/28 pairs);
  - reassignment norms matched;
  - internal slot edits equal (q/q_max)·‖d‖ to within 2e-7 relative, with weak slots at 0.251/0.502/0.749.

  The analysis script accepted these rows through all of its integrity checks and flagged the random model's failed
  positive control correctly ("not interpretable").
- **Analysis script:** tested on synthetic roots with planted outcomes (scratchpad only). It recovered the planted
  curves, e.g. a planted 25/50/75% internal success read back as 23.6/50.7/82.6%. It gave "similar" at exactly 0. It
  rejected tampered gold, a mismatched reassignment norm, an L = 255 mismatch across worlds and altered text.
- **GPU-only, untested:** real-model loading through the adapter (FP32, the 4-GPU map for 122B, dequantized gpt-oss
  experts), hook placement on the real architectures, the identity-write inertness, the cached-vs-complete agreement
  on hybrid and sliding-window caches, determinism of the L = 255 pairs (MoE routing in gpt-oss/122B is the main
  risk), throughput, and every outcome.
- **Adapter still changing:** `src/syncomm/latest/adapter.py` changed several times while this was prepared. Each
  manifest pins it, so generate the manifests (§8 step 1) only after the adapter is final. `run` refuses any pin
  mismatch, and `dry-run` only reports it.

## 10. Changes from the brief, and why
1. **L = 0 omits the weak word instead of sending weight 0.** An explicit zero-weight entry would still take a slot
   index, which moves the Y word's slot internally, and it would print `[0/255]` in the text route. That is a
   different condition from the paper's. Omitting it makes L = 0 byte-identical to the paper's packet, so it is a
   true positive control.
2. **The L = 255 floor is exact:** J = R = 0, not "near". It works as an integrity check, not a noisy baseline. The
   informative floor for the graded levels is the weight-blind bound in §5.
3. **A pre-declared balanced subset** (tiers 1+2, 144 scenes) replaces all 572 scenes, for cost. §8 fixes the
   fallback and extension in advance.
4. **One reassignment arm at L = 128** was added, for about 8% extra cost.
5. **The DiD was added as a key secondary**, so that a delivery gap already present at L = 0 is not read as a
   difference in weight reading.
6. **Qwen3-32B is inside the Holm family** (m = 4, fixed), which is conservative. In R3 it was outside as the anchor.
   Here it answers the same new question.
7. **The identity write covers all eight slots**, a superset of any arm's positions. This is a stronger inertness
   test than using one packet's positions.
8. **The seeds are the Experiment A seeds**, for consistency within the latest0924 study.

## 11. Files (sha256 at the time of writing)
- `bank.json` `b701b10c838b6402…`; `packets_L0/64/128/191/255.json` `9af3c050… / 9b50a983… / 4dc7f370… / ffda40e7…
  / 466536ed…`; `construction_report.json` `c43d6e7b…`.
- Scripts: `scripts/latest0924/build_graded_bank.py`, `src/syncomm/latest/evaluate_graded.py` and
  `scripts/latest0924/analyze_graded.py`. The lock pins their final digests.
