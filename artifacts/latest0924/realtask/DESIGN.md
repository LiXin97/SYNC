# Experiment D: the final interface on a realistic task (Countries)

**Question.** Does the final SINC interface (prompted sender selects inventory symbols; receiver writes its own
twelve-context symbol directions once at relative depth 0.2, gamma 1; prefix cache reused per question) transfer facts
on a realistic task where the receiver must combine the communicated fact with its own world knowledge?

**Task.** Countries (Ramesh & Li 2025, rebuilt mechanically in `src/syncomm/bench/build_ac.py`): a sender privately sees
"X is at [the] LANDMARK"; the receiver must say where X is. We use the 400 **dev** items only (the test split was touched
once by the earlier utility study and stays untouched). The final interface was never tuned on Countries.

**Format.** The frozen two-object counterfactual format, so the frozen sender prompt, receiver prompt, parser, scorer
and evaluator (`src/syncomm/recovery/evaluate_confirm.py`) run unchanged: two objects with synthetic uppercase aliases
(R3 alias pairs), world 0 = A at L1, B at L2 (different countries), world 1 = landmarks swapped; questions "Which object
is in C1?" and "Which object is in C2?". Four-answer success needs both questions right in both worlds; without a
message it is zero by construction. The receiver must map each landmark to its country itself.
Scenes: 24 sender-gate scenes, 4 implementation-check scenes, 172 evaluation scenes (pairs of dev items, seed 20260925).

**Inventory.** All 195 landmarks and 80 countries in the task builder's fixed lists, lowercased: 275 symbols, from the
task definition only. Most landmarks are multi-word symbols (untested so far in the paper).
Two single-assertion variants accept 275 instead of 221 (`fit_layers_inv275.py`, `sender_inv275.py`); nothing else changes.

**Sender.** gemma-4-26b-a4b-it with the frozen two-object candidate-A prompt (BF16, greedy, 256 tokens, direct loader),
gate on the 24 gate scenes: parsed >= 0.99, landmark recall >= 0.90, wrong-object lanes <= 0.02.

**Receivers.** qwen3.5-27b, qwen3.6-27b, mistral-small-3.2-24b (adapter, tf-latest) and qwen3-32b (continuity, frozen
loader, system python); directions fitted for the 275-symbol inventory at depth 0.2 and the original band.

**Arms.** none; identity_write (engineering only); original (middle band 0.4/0.5/0.6); selected (early write);
packet_text (same symbols as text); full_text (the evidence sentences); sham0-2 (symbol-direction reassignments,
edit-norm matched, derangements seeds 202609251-3); other_world (consistency).

**Endpoints and rule.** J = four-answer success, family (country pair) macro. Intervals: cluster bootstrap over
country pairs and over continent pairs, 20,000 draws x seeds 20260924-6, enclosing interval, p = larger of the two.
POSITIVE per receiver: Holm across evaluated receivers rejects both selected - none and selected - mean(reassignment),
both lower limits > 0. Secondary: selected - packet_text, selected - full_text, selected - original, accuracy,
coverage, J at 256. Engineering gate per receiver as in Experiment A (identity inert; cached vs complete max logit
<= 0.05, alias gap <= 0.1, equal parses); a failing receiver is an engineering failure with no outcomes.
Results are reported whichever way they come out.
