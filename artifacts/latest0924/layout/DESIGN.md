# Experiment E (latest0924): frozen-setting transfer to new layouts, slot positions and object counts

Binding once `gates/latest0924/expE_layout.lock.json` is committed, before any GPU row (engineering included).

## Question
With everything that defines the interface frozen — each receiver's existing twelve-context word-minus-unknown
directions, the single write at its depth-0.2 layer, gamma 1, raw scale, the frozen parser, questions and budgets — does
delivery survive changes to the message structure that the directions were never fitted or tuned on?

## Material
Complete facts (every weight 255), no model sender, so delivery is isolated. Scenes: the 144 prespecified two-object
confirmation scenes of Experiment B (tiers 1+2 of `artifacts/latest0924/graded/bank.json`, 4 per material family,
balanced on every design factor) and its 4 engineering scenes. For L3 each scene gets a third object C (bank builder
`scripts/latest0924/layout/build_layout_bank.py`): for each field, the first value in sorted order of the domain's
value pool that neither A nor B uses; C's facts are identical in both worlds and never match a queried value; C's alias
is drawn deterministically from the aliases of the R3 confirmation bank. Questions, queried values and golds are
unchanged, so the four-answer counterfactual test is unchanged.

## Conditions (identical for the internal and text routes; renderer `src/syncomm/latest/layout_render.py`)
- **L0 standard**: `Selected words and exact relative weights (integers divided by 255): A: w1 [255/255], w2 [255/255],
  unknown [255/255], unknown [255/255]; B: ... .` Facts in ascending inventory order in slots 0, 1. Byte-identical to
  the frozen renderer; must reproduce the Experiment A complete-fact rows (R3 rows for qwen3-32b).
- **L1 slot permutation**: same template and neutral prefix as L0, but each object's facts sit in slots 3 and 2 in
  reversed order (`A: unknown [255/255], unknown [255/255], w2 [255/255], w1 [255/255]`).
- **L2 new template**: `Received concepts, each followed by its strength out of 255 | B -> w1 (255) ; w2 (255) ;
  unknown (255) ; unknown (255) | A -> ... |` — new header, separators and weight annotation, objects in reverse order.
- **L3 new object count**: the standard template with three objects (`A: ...; B: ...; C: ... .`); twelve slots.
- **L4 relational roles**: out of scope. It needs a new message format and question family whose construction and
  validation cannot be done soundly before the deadline; it is not run.

## Receivers
qwen3-32b (continuity; frozen loader, system python), qwen3.5-27b, qwen3.6-27b, mistral-small-3.2-24b, qwen3.5-9b,
granite-4.1-8b (adapter, tf-latest). Directions and layers exactly as in Experiments A/B.

## Arms per condition and world
none (structural zero); internal; text; sham0 (the R3 reassignment 0 with per-layer edit norm matched to internal).
Engineering phase adds identity_write (all slots, must be exactly inert) and checks cached vs complete forward on
every arm (max logit difference <= 0.05, alias-gap difference <= 0.1, equal parses at 64 and 256 tokens).

## Endpoints and decision rule
- J = four-answer success per scene at 64 tokens; family-macro.
- **Primary** per receiver x condition c in {L1, L2, L3}: Delta_c = J_internal(c) - J_internal(L0), paired by scene.
  Cluster bootstrap over the 36 families and over the 12 domain x template clusters, 20,000 draws x seeds 20260924-6,
  interval enclosing seeds and levels.
- **Transfer rule (non-inferiority, margin 5 points)**: H0 Delta_c <= -5 points; one-sided p = max over levels and
  seeds of the bootstrap fraction of Delta* <= -5; Holm across all 18 receiver x condition cells (alpha 0.05).
  "transfers" if Holm rejects; "fails to transfer" if the enclosing upper limit < -5; otherwise "inconclusive". A
  receiver failing its engineering gate is "not interpretable" for every condition.
- Secondary, descriptive: internal - text per condition; internal - sham0 per condition; text(c) - text(L0); the ratio
  J_internal(c) / J_internal(L0).
- Checks: receipts and exact row sets; engineering rows all pass; identity inert; L0 rows identical in tokens and
  logits to the Experiment A (R3) complete-fact rows.
- Results are reported whichever way they come out; nothing is tuned after the lock.

## Limits
Complete facts only; the vocabulary and question forms are those of the confirmation bank; a constructed bank subset,
not a population; relational roles untested.
