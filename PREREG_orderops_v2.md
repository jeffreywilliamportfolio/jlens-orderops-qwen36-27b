# Preregistration — order-of-operations lens eval v2 on Qwen3.6-27B (FROZEN 2026-09-04 13:23 MDT on Jeffrey's word; hashes in FREEZE.sha256)

Author: Jeffrey W. Shorthill. Drafted by a Claude Code agent 2026-09-04; frozen only after Jeffrey's predictions are entered
and this file's sha256 is recorded in FREEZE.sha256 **before any API or GPU call**.

## What v1 was
A single unregistered pass (2026-09-04 ≈16:25 UTC) of the paper's 55 items through Neuronpedia's hosted 27B lens:
number intermediate pass@1 24/55, pass@3 42/55 (RESEARCH_NOTE.md). v1 is exploratory and is not cited as a frozen result.

## Frozen inputs
| file | sha256 (first 16) | what |
|---|---|---|
| order_ops.json | b203206d16ff6281 | the paper's 55 items, unmodified (anthropics/jacobian-lens commit 581d3986) |
| items_new50.json | 6f08fc447b0930b5 | 50 held-out items (25 single-digit, 25 two-digit intermediates), seed 20260904, generator rules inside the file; each also carries a `prompt_nospace` variant |
| run_v2_api.py | (see FREEZE.sha256) | the API runner for arms A and B, written before the freeze, unchanged after |
| score_orderops.py | 5bb4e0718cea342a | the scorer (synonym sets, readout rule, metrics) — reused unchanged; items_new20 scored by the same code |

## Fixed pipeline
- Prompts sent exactly as in the item files (trailing space kept). Readout = last prompt token, all layers.
- Rank = min over single-token synonyms: digit, English word, CJK numeral; matched on stripped lower-cased strings (a leading space is stripped, so ' twelve' and 'twelve' both match; only forms that exist as one token in the Qwen3.6-27B tokenizer can ever appear). Two-digit intermediates with no single-token form in any script are unreachable by construction; the held-out generator excludes them, and any such item in the paper's set is reported separately.
  API arm: topN is capped at 8 by Neuronpedia (verified on a throwaway prompt 2026-09-04), so rank > 8 is recorded as absent.
  Box arm: exact rank of each synonym over the full vocabulary from the lens logits (no cap); reported both raw and capped at 8 for comparison with the API arm.
- Metrics: pass@1, pass@3, MRR; uninvolved-digit baseline (digits absent from prompt, target, intermediate) per item;
  paired sign test (intermediate rank vs item's median uninvolved rank); target rank at the same position reported beside the intermediate rank (spoken vs unspoken); decoy comparison (intermediate vs wrong-precedence value).
- Persistence descriptives (reported, not hypothesis-tested): per item, the number of layers at which the intermediate is at rank ≤ 3, and the first/last such layer (layer span); on the box, the exact rank of intermediate, target, and decoy at every layer (rank trajectory), plotted per item and as a median curve. A value held across many layers at a fixed rank beyond the API's top-8 horizon is reported as such.
- Decoding: greedy everywhere.

## Arms
- **A. API, hosted 1000-prompt lens** (Neuronpedia qwen3.6-27b): 55 + 50 items, lens read on the raw-token path.
- **B. Correctness filter**: for every item, the model's greedy answer. On the box: greedy continuation of the raw prompt
  (≤ 6 tokens); correct = first integer emitted == target. If only the API is available: chat endpoint, numCompletionTokens 8,
  same rule, flagged as template-mediated. Results reported unfiltered AND filtered to correct items.
- **C. Box, two lenses, one forward pass** (if the box is rented): (C1) the hosted 1000-prompt lens .pt downloaded and applied
  locally; (C2) a lens fitted here by Neel's recipe — 25 Pile prompts, 128 tokens, first 4 tokens skipped, target = penultimate
  layer, plain autograd. Same 105 prompts, same scorer. C1 must reproduce arm A's pass@1 within ±3 items or the local pipeline
  is not trusted and C2 is not reported.

## Hypotheses and decision rules (stated before any call)
- **H1 (paper set, filtered)**: on items the model answers correctly, number pass@1 ≥ 0.35 and pass@3 ≥ 0.65 with the
  hosted lens; paired sign test p < 0.01. Rule: both thresholds met → "replicates"; else "does not replicate as scored".
- **H2 (held-out 50)**: pass@1 and pass@3 on items_new50 within the 95% CP interval of the paper-set values. Rule: inside →
  "generalises beyond the paper's items"; outside on the low side → "paper items favourable"; report the numbers either way.
- **H3 (carrier)**: among two-digit intermediates with a rank-1 hit, ≥ 2/3 are CJK-only or CJK+word. Rule as stated.
- **H4 (lens, arm C only)**: 25-prompt lens pass@1 on the 105 items ≥ half of the 1000-prompt lens pass@1 → "effect survives
  Neel's lens; lens quality does not explain the null"; < half → "lens quality is a sufficient explanation".
- **H6 (position, all arms)**: every item is also read with the trailing space removed (`prompt_nospace`), readout at the `=` token. Rule: if pass@1 falls by more than half relative to the with-space form → "readout position is a sufficient explanation for a null"; if within half → "position does not explain it".
- **H7 (digit count, held-out set)**: two-digit vs single-digit pass@1 with CJK in the synonym set, and the same contrast with CJK removed. Rule: if the two-digit rate without CJK falls below half of the two-digit rate with CJK, tokenization is a sufficient explanation for two-digit misses.
- **H5 (decoy)**: correct intermediate beats decoy in > 60% of decidable items, pooled over 105.

## Predictions (Jeffrey, verbatim, before freezing)
- H1 (Jeffrey, 2026-09-04 12:51 MDT): pass@1 = 0.50 on correctly answered items ("based on the 24/55 number … the additional size wikitext corpus isn't going to change its math capabilities"); pass@3 = 0.75 ("right around there matching this morning's").
- H2 (Jeffrey, 2026-09-04 13:07 MDT): "my prediction stands that the numbers should match the paper's numbers" → held-out pass@1 = 0.50, pass@3 = 0.75, same as H1; i.e. the paper's items were not special once every intermediate is reachable by a single token.
- H6 (Jeffrey, 2026-09-04 13:21 MDT): ≈ 1.0. "It's always going to be in either the =_ token or the = token itself, the intermediate" — the readout follows the last token, so ending the prompt at "=" moves the intermediate to "=" rather than losing it.
- H7 (Jeffrey, 2026-09-04 13:22 MDT): two-digit pass@1 with the Chinese numerals ≈ 0.50 ("to match the 9 out of 22" — note 9/22 is 0.41; recorded as stated, 0.50 ± 0.1); without them ≈ 0.05 ("very few").
- H3 (Jeffrey, 2026-09-04 13:09 MDT): 0.90 of two-digit rank-1 hits involve the Chinese numeral ("based on this morning's observations": 9/9).
- H4 (Jeffrey, 2026-09-04 13:14 MDT): "the 1000 wikitext corpus increases the survival effect", but "only marginally" → 25-prompt lens pass@1 ≈ 0.9 of the 1000-prompt lens (well above the 0.5 rule line).
- H5 (Jeffrey, 2026-09-04 13:17 MDT): "matches this morning or increases it maybe 5%" → decoy win rate ≈ 0.85 of decided items (this morning 9/11 = 0.82).
Agent's predictions (recorded now, blind to Jeffrey's): H1 0.45 / 0.78; H2 0.40 / 0.72; H3 0.85; H4 0.6; H5 0.70; H6 0.25; H7 0.40 with CJK / 0.10 without.

## Not in scope
Causal swaps; other models; any change to the synonym sets after freezing (a change means a new prereg).
