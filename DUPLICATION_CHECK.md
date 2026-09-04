# Duplication check against Neel Nanda's team (2026-09-04)

What their team has published on this eval, so that this repo neither repeats it nor misdescribes it.

## 1. The July review (LessWrong, 2026-07-06)
"We also tried replicating the quantitative evals in section 10.2 … Poetry, and arithmetic both failed to replicate, but this is
plausibly due to experimenter error or worse model capabilities." Lens: 25 Pile prompts, 128 tokens, first 4 skipped, penultimate
layer. Baselines included reading "the Chinese token for the intermediate". Dataset and scorer for that attempt: not public.

## 2. `camilablank/workspace-bench` (GitHub + HF dataset; pushed 2026-08-14..25; scholars Camila Blank, Agam Bhatia)
- An 11-family benchmark on Qwen3.6-27B comparing a generative "oracle lens" (AO, not distributed) against the J-lens.
- Its J-lens is `neuronpedia/jacobian-lens qwen3.6-27b n1000 wikitext` — the same hosted lens this repo reads through the API.
- `order_ops` was renamed `arithmetic_intermediates` and redesigned: chat template with "Compute (24 + 30) * 8. Reply with only the
  final number, nothing else."; 2–3-digit intermediates mostly with no single-token form ("structural"), read at one frozen
  (layer, position) inside the assistant scaffold (L56/L60, p−7/p−8) with 10 sampled readouts; metric `net = value − cross`.
  Reason given for abandoning the bare prompt: "a bare `Compute (7*3)+8 =` makes the model *write* `21`, which would prove nothing."
  Admissibility gate: ≥ 8/10 rollouts assert the answer and 0 rollouts emit an intermediate.
- Their item schema has `single_token_reachable`; the README names `十八`/`eighteen` as one-token routes; their deterministic
  matcher scores CJK by substring and numbers only in answer position. So single-token reachability and Chinese numerals are
  known to them by August. Whether the July scorer credited Chinese numerals is still not public.
- Their headline single-token banks (multihop, multilingual, poetry, typo, association, basic-readout) ARE bare prompts read at the
  final token — the same readout rule this repo uses — but order-ops is not among them.

## 3. What this repo does that is not in either
- The paper's own 55 order-ops items, bare prompt, readout at the token before the answer, J-lens pass@k with digit/word/CJK synonyms,
  plus 50 rule-generated held-out items with wrong-precedence decoys — i.e. the 10.2 eval as written, on Qwen.
- Two lenses on one forward pass (hosted n1000 vs a 25-prompt Pile fit by the July recipe) — the instrument comparison.
- The position factor (with/without trailing space) and the digit-count × CJK contrast as preregistered hypotheses.

## 4. What this repo adopted from them, before scoring
- The leak gate (addendum 2), run via the API's raw `prompt` generation (addendum 3) and on the box with seeds. Every headline is
  reported on admissible items. If Qwen writes the intermediate on these bare prompts, the v1 rank-1 hits are next-token prediction
  and the note says so.

## 5. Not duplicated
Nothing here re-runs workspace-bench. Their design answers "can a generative readout recover multi-digit intermediates the
J-lens cannot represent"; this repo answers "does the paper's single-token order-ops eval pass on Qwen3.6-27B, and what
separates that from the July null".
