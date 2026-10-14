# Receiver recovery — results package (2026-09-23T02:13:43.150704+00:00)

Every number is read from an audit JSON; the R3 audit was reproduced by an independent second count with zero disagreements. Labels: R1/R2 = development (exposed dev-96); R2 rescreen = exposed fibre/metal bank; R3 = confirmation on the locked new 576-scene bank.

## What succeeded
The bounded receiver-side change is the WRITE SITE: the fitted word-contrast write moved from the original three-layer mid band to ONE early layer at relative depth 0.2, raw scale, gamma 1, with dictionary, layout, sender and parser unchanged. On the new bank, with the frozen automatic sender:

| receiver | original band | selected (depth 0.2) | identical packet as text | mean reassignment | selected - original | targets |
|---|---|---|---|---|---|---|
| qwen3-32b | 0.944 | **0.970** (cov 0.996) | 0.970 | 0.012 | +0.026 [+0.012, +0.042] | useful=True, substantial=False (anchor, outside the Holm family) |
| qwen3-14b-base | 0.310 | **0.977** (cov 1.000) | 0.977 | 0.023 | +0.667 [+0.631, +0.702] | useful=True, substantial=True |
| glm-4-9b | 0.005 | **0.743** (cov 0.923) | 0.878 | 0.006 | +0.738 [+0.665, +0.808] | useful=True, substantial=True |
| gemma-3-27b-it | 0.305 | **0.977** (cov 1.000) | 0.979 | 0.030 | +0.673 [+0.617, +0.726] | useful=True, substantial=True |

All six frozen Holm tests (selected-original and selected-mean reassignment on qwen3-14b-base, glm-4-9b, gemma-3-27b-it) reject; intervals clear zero at both the 36-family and 12-template levels. The other world's own packet, current question held fixed, is scored correct against the SUPPLIED gold at exactly the selected rate on every receiver (fact-following, not disturbance). Every no-message row is 0.000 and the active identity write is exactly inert.

## Why it works (development mechanism, R0-R2)
R1: native fact states and fitted directions share one depth profile on every receiver — depth 0.2 works, 0.35 partially, 0.5/0.65/0.8 fail — so the failing factor was the write layer, not the dictionary construction. R0/R2: the generic local scale is c[l] = 1.02-1.25 and reproduces raw gamma=1 exactly; at the working depth the fitted and native realized edit norms already agree within 3-7 %, and matching them moves J by <= 1 pp. The recovery is a site effect, not a scale effect. Gemma's pre-declared transfer rule (native-selected depth, local scale, gamma 1), frozen before any Gemma fitted cell was read, reaches development J = 1.000.

## What did not succeed or is not claimed
- The internal write does not beat text: selected - identical packet as text is +0.000 [-0.005, +0.005] (qwen3-32b), +0.000 [-0.007, +0.009] (qwen3-14b-base), -0.002 [-0.005, +0.000] (gemma-3-27b-it) and -0.135 [-0.183, -0.089] (glm-4-9b, text ahead).
- glm-4-9b recovers most but not all the way (0.743 vs text 0.878; on the exposed rescreen it missed the J and coverage clauses).
- The R2 local-scale hypothesis advanced no cell; averaging-shrinkage is excluded as an explanation.
- The R3 bank is new in evidence constructions, names and scene composition; 14 of 72 two-object combinations, 7 of 36 value cells and all single-object fact tuples repeat (finite 4x4 vocabulary), and the question wordings are the developed ones. It tests unseen constructions and compositions, not unseen facts.
- Native transplantation reads scene-specific states: an instrument, not a deployable method.

## Cost (R5)
- qwen3-32b: selected write 1301 ms (p90 1456), identical packet as text 1018 ms, no message 1039 ms per item (two questions); packet 20 B vs text 226 B. No engineering advantage claimed.
- qwen3-14b-base: selected write 640 ms (p90 717), identical packet as text 515 ms, no message 540 ms per item (two questions); packet 20 B vs text 226 B. No engineering advantage claimed.
- glm-4-9b: selected write 651 ms (p90 736), identical packet as text 545 ms, no message 471 ms per item (two questions); packet 20 B vs text 226 B. No engineering advantage claimed.
- gemma-3-27b-it: selected write 1300 ms (p90 1406), identical packet as text 1051 ms, no message 1054 ms per item (two questions); packet 20 B vs text 226 B. No engineering advantage claimed.

## R4 — four-object extension (descriptive; no test family or target was declared)
Four painted blocks per scene, 188 scenes / 24 family cells, four candidate aliases (chance J = 0.004). Frozen two-object receiver configurations; the sender changed only in its object count (384/384 parsed, required-fact recall 0.966, complete per-object recovery 0.932, wrong-object 0.0007).

| receiver | none | original band | selected (depth 0.2) | identical packet as text | mean reassignment | complete facts selected / text / native | selected - original |
|---|---|---|---|---|---|---|---|
| qwen3-32b | 0.000 | 0.800 | **0.869** (cov 0.953) | 0.869 | 0.000 | 0.973 / 0.968 / 0.978 | +0.069 [+0.026, +0.113] |
| gemma-3-27b-it | 0.000 | 0.262 | **0.911** (cov 0.993) | 0.911 | 0.002 | 1.000 / 1.000 / 1.000 | +0.650 [+0.558, +0.731] |

R4 is a separate protocol and sample count and is never pooled with R3; the same finite vocabulary applies, and no open-world or general relational claim follows.

## Not run
- Nothing further in the plan: R0-R5 and the conditional R4 are complete.

## Files
conditions_long.*, contrasts_long.*, norms_long.*, selection.json, cost.json, asset_map.json, provenance.json, locks/.
