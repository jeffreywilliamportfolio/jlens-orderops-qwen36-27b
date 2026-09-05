# Order-of-operations lens eval on Qwen3.6-27B (Neuronpedia), 2026-09-04

Post-hoc check run by Fable during the review of the July-25 arithmetic J-space runs. n=1 pass, greedy, no prereg.

- Eval: `lens-eval-order-ops.json` from `anthropics/jacobian-lens` (55 items, prompt + `intermediates` [number, operation] + `target`),
  the arithmetic member of the paper's section-10.2 quantitative evals and the one Neel Nanda's review (2026-07-06) reports as
  "failed to replicate" on Qwen 3.6 27B.
- Instrument: Neuronpedia `POST /api/lens/prompt`, `modelId qwen3.6-27b` (meta reports `Qwen/Qwen3.6-27B`), hosted Jacobian lens (`neuronpedia/jacobian-lens` `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt`, fitted on 1000 WikiText prompts by Mateusz Piotrowski, Anthropic Interpretability — NOT a 25-prompt Pile fit like Neel's),
  `topN 8`, `temperature 0`, `filterNonWordTokens false`, prompts sent as raw `inputTokenIds` (Qwen3.6-27B tokenizer, no chat
  template, `prependBos true`). Generation with `inputTokenIds` returned no tokens, so no correctness filter was applied.
- Readout rule (paper's): single position = the token immediately preceding `target` (the last prompt token), all layers;
  rank = min over single-token synonyms (digit, English word, CJK numeral for numbers; symbol/word/CJK for operations);
  metric = pass@k = fraction of intermediates with min-over-layers rank <= k. topN 8 caps measurable rank at 8.
- Raw per-item responses in `raw/`; scored table in `orderops_results.json`; scorer = the inline script in the session transcript
  (fe4f48cb, 2026-09-04 ~10:30 MDT) — numbers below are reproducible from `raw/` with the synonym sets stated there.

| readout | pass@1 | pass@3 | pass@5 | pass@8 |
|---|---|---|---|---|
| number intermediate @ '=' (digit+word+CJK) | 0.44 | 0.76 | 0.84 | 0.85 |
| number intermediate @ '=' (no CJK) | 0.31 | 0.67 | 0.76 | 0.78 |
| number intermediate @ token before '=' (position control) | 0.07 | 0.11 | 0.18 | 0.27 |
| operation intermediate @ '=' | 0.44 | 0.62 | 0.65 | 0.73 |
| uninvolved single digits @ '=' (n=298; digit-cloud baseline) | 0.08 | 0.28 | 0.50 | 0.80 |

- Layer of best number hit: median L55 (IQR 52–60) of 64.
- Carrier of the hit: single-digit intermediates ride the digit token (32/55 overall); **two-digit intermediates are carried only by
  CJK numerals (13) or an English word (1)**; 8 have no hit in top-8. Qwen splits digits, so "12" cannot be a single ASCII token.
- Wrong-order decoy (the intermediate you would get by ignoring precedence), 14 items: correct intermediate ranks better 9,
  decoy better 2, tie 3.
- Caveats: one pass, unseeded-irrelevant (greedy), hosted lens (n=1000 WikiText fit by Anthropic, a different and likely stronger instrument than Neel's 25-prompt lens; lens quality is therefore a live explanation for the discrepancy), no correctness filter, operation
  symbols echo the prompt (min-over-layers lets a layer-0 '-' count). pass@8 is at the digit-cloud floor and says nothing.

Also here: `raw_calc_paperprompt.json` — one chat-format call with the paper's Opus example "calc: ( 4 + 17 ) * 2 + 7 =": under the
chat template the model starts prose ("To solve the expression") and none of 21/42/49 (any form) appears at the '=' position.

## Teardown record
Box 49890170 (1× RTX PRO 6000 Blackwell 96 GB, vastai/pytorch:cuda-12.8.1-auto, $1.70/h) up 2026-09-04 19:27Z → destroyed 2026-09-05 ~01:58Z
on Jeffrey's word, ≈ 6.5 h ≈ $11. Ran: order-ops v2 arm C (5-prompt Neel-recipe lens fit, two-lens read of 210 prompts, seeded leak
gate on 105), then the evalaware / sandbag / hybrid pokes. Everything pulled before destroy (v2_box, v2_box_full, lens .pt, logs, and the
three poke output dirs). Astra's box 49901204 untouched.
