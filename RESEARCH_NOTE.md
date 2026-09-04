# The order-of-operations Jacobian-lens eval on Qwen3.6-27B: the probing half replicates in one pass

Jeffrey W. Shorthill (ORCID 0009-0004-3954-2752) · research note, 2026-09-04 · not peer reviewed

**AI assistance.** The API calls, scoring script, and first draft of this note were produced by a Claude Code agent
(Claude Fable 5.1) on 2026-09-04. The author is responsible for the claims; see "Author verification" at the end.

## Summary

Neel Nanda's review of Anthropic's global-workspace paper (LessWrong, 2026-07-06) reports that, on Qwen 3.6 27B,
"Poetry, and arithmetic both failed to replicate, but this is plausibly due to experimenter error or worse model
capabilities." The arithmetic eval in question is the paper's section-10.2 `lens-eval-order-ops` set. Run on the same
model through Neuronpedia's hosted Jacobian lens, with the paper's items and readout rule, the unspoken intermediate
reaches lens rank 1 in 24 of 55 items and rank 3 or better in 42 of 55, against a matched uninvolved-digit baseline of
8% and 28%. Two-digit intermediates are found almost only as CJK numerals, because Qwen tokenizes digits singly.
This is one greedy pass, probing only, with a different (larger) lens than Neel's team used. It does not show that
their result was wrong; it shows the effect is present on this model under these scoring choices and names three
things that could separate the two runs.

## What the eval is

`data/evaluations/lens-eval-order-ops.json` in `anthropics/jacobian-lens` (commit 581d3986, 2026-07-02; file sha256
b203206d…): 55 items, each a bare expression such as `(2 + 3) * 4 = ` with `intermediates` [number, operation] and a
`target` answer that is never scored. The README's rule: readout at the single token immediately preceding the target,
across all layers; rank is the minimum over single-token synonyms (numbers in digit and word form, operations in symbol
and word form); metric pass@k is the fraction of intermediates whose minimum-over-layers rank is ≤ k. The intermediate
(the 5 in `(2 + 3) * 4 = 20`) is never emitted, which is what makes this a workspace test rather than a next-token test.

## Method

- Model: `Qwen/Qwen3.6-27B` (64 layers, checkpoint natively bf16, no quantization config), served by Neuronpedia.
- Lens: Neuronpedia's hosted `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt`, fitted on
  1000 WikiText prompts, credited to Mateusz Piotrowski (Anthropic Interpretability). This is not the 25-prompt Pile fit
  Neel's team used.
- API: `POST https://www.neuronpedia.org/api/lens/prompt`, `type ["JACOBIAN_LENS"]`, `topN 8`, `temperature 0`,
  `filterNonWordTokens false`, `prependBos true`. Prompts were sent as raw `inputTokenIds` from the Qwen3.6-27B tokenizer
  with no chat template. Every prompt ends `= `, so the last prompt token is the space after ` =`; that is the readout
  position. No completion was returned on this path, so no correctness filter was applied.
- Synonym sets: digits `"5"`; English words (`five`, `twelve`, `twenty-one`/`twentyone`); CJK numerals (`五`, `十二`,
  `二十一`). Operations: symbol, English word forms, and CJK (`乘`, `加`, `减`, `除` and compounds). Matching is on the
  stripped, lower-cased token string. Rank capped at 8 by `topN`.
- Controls: (a) the same readout one token earlier (position control); (b) for every item, the best rank of each digit
  0–9 that appears nowhere in the prompt, target, or intermediate (uninvolved-digit baseline, 298 readings, clustered by
  item); (c) a wrong-precedence decoy for 14 items, the value obtained by evaluating the expression in the wrong order
  (for `2 + 3 * 4`, 5 instead of 12); (d) the same scoring with CJK synonyms removed.
- One pass, greedy, no preregistration. Run 2026-09-04 ≈16:25 UTC.

## Results

| readout | pass@1 | pass@3 | pass@5 | pass@8 | MRR |
|---|---|---|---|---|---|
| number intermediate, digit + word + CJK | 24/55 = 0.44 [CP 0.30–0.58] | 42/55 = 0.76 [0.63–0.87] | 0.84 | 0.85 | 0.60 |
| number intermediate, no CJK | 17/55 = 0.31 [0.19–0.45] | 37/55 = 0.67 [0.53–0.79] | 0.76 | 0.78 | 0.49 |
| number intermediate, one token earlier | 4/55 = 0.07 [0.02–0.18] | 6/55 = 0.11 [0.04–0.22] | 0.18 | 0.27 | 0.12 |
| operation intermediate | 24/55 = 0.44 | 34/55 = 0.62 | 0.65 | 0.73 | — |
| uninvolved digits (n = 298) | 0.08 | 0.28 | 0.50 | 0.80 | 0.25 |

CP = 95% Clopper-Pearson. MRR counts rank > 8 as 0. Paired sign test, intermediate rank against each item's median
uninvolved-digit rank: better in 41 items, worse in 12, tied in 2, two-sided p = 8×10⁻⁵. The layer of the best number
hit has median 55 (IQR 52–60) of 64. Rank histogram for intermediates: 24 at rank 1, 11 at 2, 7 at 3, 4 at 5, 1 at 7,
8 absent from the top 8. The eight misses: `17 - 3 * 4`, `10 + 5 - 3`, `((2 * 3) + 6) / 4`, `2 * 6 / 4 * 3`,
`20 - 5 - 3 - 2`, `10 + 3 - 5 + 2`, `(((1 + 2) * 3) + 3) / 4`, `(2 * 5) // 3`.

**Carrier.** Of the 33 single-digit intermediates, 32 are found as the digit token. Of the 22 two-digit intermediates,
at rank 1: 7 are found only as the CJK numeral, 2 as both CJK and English word, 0 as the word alone, 13 not at rank 1.
At rank ≤ 3: 5 CJK only, 1 word only, 6 both, 10 neither. Removing CJK from the synonym set drops pass@1 from 24 to 17.
The seven CJK-only rank-1 items: `3 * 4 + 2` (十二), `4 * 5 - 8` (二十), `(10 + 5) / 3` (十五), `(14 - 2) / 4` (十二),
`(7 + 4) % 3` (十一), `(2 * 8) % 5` (十六), `Calculate: 3 times 4 plus 2 equals` (十二).

**Decoy.** Across the 14 items with a computable wrong-precedence value, the correct intermediate ranks better in 9,
the decoy in 2, tie in 3.

**Operations** echo the prompt symbol (`-` at layer 0 counts under min-over-layers), so the operation row is weaker
evidence than the number row and is reported for completeness.

## What this does not show

- It is the probing half only. No causal swap against an answer-token baseline was run.
- One pass, no correctness filter, hosted lens. The paper adapts items to what the model can do; that filter was not
  applied here because the raw-token path returned no completion.
- The lens differs from Neel's team's lens (1000 WikiText prompts by Anthropic vs 25 Pile prompts). A stronger lens is
  itself a candidate explanation for the difference and cannot be separated from the scoring explanation with this run.
- The paper's own qualitative example, `calc: ( 4 + 17 ) * 2 + 7 =`, sent through the chat template, produced prose
  ("To solve the expression") and none of 21/42/49 in any form at the `=` position (`raw_calc_paperprompt.json`). The
  chat template moves the readout position and changes the task; the eval must be run on raw prompts.
- Two earlier arithmetic runs by the author (July 25, `Show your steps briefly` prompts) read the number the model is
  about to write at each `=` of its own chain of thought. Those are next-emission readouts, not unspoken intermediates,
  and are not evidence on this eval.

## Candidate explanations for the discrepancy with Neel's team

1. **Tokenization and synonym coverage.** Qwen has no single ASCII token for a two-digit number. If a scorer targets
   the digit or English form, two-digit items read as misses; here CJK numerals carried 7 of the 9 two-digit rank-1
   hits. Neel's review reads as using the Chinese token as a separate baseline condition rather than as a synonym.
2. **Readout position.** Hits sit at the token before the answer. One token earlier, pass@1 is 0.07. A prompt ending in
   `=` without the trailing space, or an off-by-one index, would remove the effect.
3. **Lens quality.** 1000-prompt WikiText fit vs 25-prompt Pile fit.

Each is testable on their side in minutes: add CJK numerals to the synonym set, confirm the readout index, and run their
25-prompt lens on these 55 raw responses' prompts.

## Reproduction

Directory `cc-lens/outputs/orderops_qwen36_27b_20260904/`: `order_ops.json` (the eval items, unmodified),
`raw/NN_name.json` (55 API responses), `score_orderops.py` (recomputes every number above from `raw/`),
`scored_rows.json` (per-item ranks), `raw_calc_paperprompt.json` + `req_calc.json` (the chat-format check),
`README.md`. Run `python3 score_orderops.py` in that directory.

## Author verification (to complete before this note is shared)

1. Open `raw/00_parens-add-mult.json`; confirm the last prompt token is `' '` and that `5` is rank 1 at layer 43.
2. Open `raw/11_parens-add-div.json`; confirm `十五` is rank 1 at layer 51 and neither `15` nor `fifteen` is in the
   top 8 at that position.
3. Run `python3 score_orderops.py` and confirm the table above line for line.
