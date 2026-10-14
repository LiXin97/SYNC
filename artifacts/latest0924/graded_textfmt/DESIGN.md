# Experiment B follow-up: stronger text formats for graded weights (preregistration)

Binding once gates/latest0924/expB_textfmt.lock.json is committed (before any GPU row of this study).
Authority: PI, 2026-09-25 ("[non-English note removed]"), after Experiment B found internal > text by +36.1 (Qwen3-32B) and +33.8 (Qwen3.5-27B)
points with a single text format.

## Question
Do stronger text renderings of the same graded weights close the gap to the internal (write-strength) route?

## Fixed (identical to Experiment B, gates/latest0924/expB_graded.lock.json)
Receivers qwen3-32b and qwen3.5-27b; bank artifacts/latest0924/graded/bank.json; tiers 1+2 (144 scenes, 36 families,
12 domain x template clusters); packets_L{0,64,128,191,255}.json (own value 255, competing value L, absent at L=0);
4 engineering scenes; questions, strict first-line parser, budgets 64 (primary) / 256; FP32 eager, TF32 off, greedy,
batch 1; selected layer (13 for both), gamma 1, raw fitted directions; python/adapter as in Experiment B.

## New arms (text route only, plus instructed internal)
- F1 `textpct_L{L}`: message "Selected words and exact relative weights (percent): A: w1 [p1], ..." with p = round(100 w/255)%
  (255 -> 100%, 191 -> 75%, 128 -> 50%, 64 -> 25%); unfilled slots "unknown [100%]".
- F2 `textverbal_L{L}`: "Selected words and relative weights (strong, fairly strong, moderate, weak): A: w1 [strong], ..."
  with 255 strong, 191 fairly strong, 128 moderate, 64 weak; unfilled slots "unknown [strong]".
- F3 `textinstr_L{L}`: Experiment B's exact "[w/255]" message, and the receiver prompt carries one extra line inserted
  immediately before "Message:\n" (after the frozen few-shot block):
  "If an object lists two values for the same attribute, the value with the higher weight is the true one."
- F3 internal `internal_instr_L{L}`: Experiment B's internal write with the same instruction line in the neutral prompt.
- `none`, `none_instr`; engineering only: `identity_write`, `identity_write_instr` (exactly inert), `internal_L064` and
  `text_L064` (continuity with Experiment B engineering rows; reported, not gating).
Every other prompt byte is unchanged (dry run checks: instructed prompts differ only by the instruction; text prompts
are token-identical to the neutral prompt outside the message block; each slot token contains exactly its word).
Internal comparator for F1/F2/F3: Experiment B's own development internal_L* rows (same scenes), not rerun.

## Endpoints (Holm across all 8, alpha 0.05)
Per receiver r in {qwen3-32b, qwen3.5-27b}:
- pct: mean over L in {64,128,191} of J_internal(L) - J_textpct(L)
- verbal: ... - J_textverbal(L)
- instr: ... - J_textinstr(L)
- instr_internal: mean over L of J_internal_instr(L) - J_textinstr(L)
J = four-answer success (both worlds x both questions) at 64 tokens, family-macro, paired by scene.
Labels: internal still better (Holm reject and enclosing lower > 0); text better (Holm reject and upper < 0);
similar (enclosing interval within [-3, +3] points); otherwise unresolved.
Interpretability per contrast: both arms pass the L=0 positive control (J >= 0.90 and coverage >= 0.90) and the L=255
structural zero (J = 0; identical packets give world-identical outputs); otherwise NOT INTERPRETABLE, numbers reported.
Descriptive: each format's gain over Experiment B's text route; per-level curves; J at 256 tokens.

## Statistics
Experiment B's, imported unchanged from scripts/latest0924/analyze_graded.py: cluster bootstrap over 36 families and
over 12 templates, 20,000 draws x seeds 20260924-6, enclosing interval; p = larger of the two levels' two-sided p.

## Execution
One 8-GPU formal node; qwen3-32b shards 0-3 on GPUs 0-3, qwen3.5-27b shards 0-3 on GPUs 4-7; engineering on shard 0
first (other shards wait for its receipt); 22 development arms x 4 answers x 144 scenes = 12,672 rows per receiver.
Node released when all rows are written.

## Reporting
Results are reported whichever way they come out, including if a text format closes or reverses the gap.
