# RESULTS_v2 — order-of-operations lens eval v2 on Qwen3.6-27B, scored against the frozen preregistration

Produced by `score_v2.py` (sha256 `cca14ad0348e5b03…`; box dir `v2_box`, no --box-full-dir), stdlib only, no network. Every number below is computed from the files listed here; the prereg and addenda are quoted, not paraphrased. The addendum-5 C2 lens is a **5-prompt fit** and is labelled **neel5 (5-prompt fit)** throughout; its on-disk directory is `neel25/`. No number here is an n=25 result.

## 1. Data provenance

| source | files found | expected | status |
|---|---|---|---|
| `v2_api/raw_A` | 210 | 210 | complete |
| `v2_api/raw_C` | 43 | 105 | partial (41%) |
| `v2_api/raw_B` | 105 | 105 | complete |
| `v2_api/raw_B2` | 0 | 105 | MISSING |
| `v2_box/hosted_n1000` | 210 | 210 | complete |
| `v2_box/neel25` | 210 | 210 | complete |
| `v2_box/continuations` | 210 | 210 | complete |
| `v2_box/manifest.json` | present | 1 | present |

Missing sources: `v2_api/raw_B2`.
Box manifest: hosted (n1000): n_prompts=1000, layers 0..62, sha256 1718c8c52dd8…; neel5 (5-prompt fit): n_prompts=5, layers 0..61, sha256 fd9af5fdb732…; model config sha 69db4eb7196b…; torch 2.11.0+cu128; GPU NVIDIA RTX PRO 6000 Blackwell Workstation Edition; started 2026-09-04T22:28:05Z, finished 2026-09-04T22:55:11Z; sampling 10×T=0.8 top_p=0.95.

Frozen-file re-verification (`FREEZE.sha256`):

| file | sha256 | check |
|---|---|---|
| `PREREG_orderops_v2.md` | `5cd441ec9edb5a3d…` | PASS |
| `order_ops.json` | `b203206d16ff6281…` | PASS |
| `items_new50.json` | `6f08fc447b0930b5…` | PASS |
| `score_orderops.py` | `5bb4e0718cea342a…` | PASS |
| `run_v2_api.py` | `ce81daaab02e0abd…` | PASS |
| `PREREG_orderops_v2_ADDENDUM.md` | `e7f887ac4c5a6d9c…` | PASS |
| `run_v2_apiB2.py` | `5be011619a03ff32…` | PASS |
| `PREREG_orderops_v2_ADDENDUM2.md` | `4260a48ac8ed0d65…` | PASS |
| `PREREG_orderops_v2_ADDENDUM3.md` | `9c895643aceb7afc…` | PASS |
| `run_v2_apiC.py` | `b0f8b196cdb78809…` | PASS |
| `PREREG_orderops_v2_ADDENDUM4.md` | `0f7a26998745fd3d…` | PASS |
| `PREREG_orderops_v2_ADDENDUM5.md` | `5a48d65cac82171e…` | PASS |

**FREEZE re-verification: PASS** (12 files).

Items: 55 paper + 50 held-out = 105. Arm A reads present: 210/210 (space 105, nospace 105). Correctness records (space): 105 items (105 box, 0 API); API arm C files 43 (43 paper, 0 held-out). Leak gate: 43 items gated (0 box, 43 API), 15 admissible. Decoys: 50 held-out (from the item file), 17 paper (derived by `box/read_items.py::derive_decoy`, the wrong-precedence rule of RESEARCH_NOTE.md — `score_orderops.py` carries no decoy dict; the 14 v1 pairs lived in the session-transcript inline scorer).

Correctness normalisation: targets and generated text are passed through a number-word map (zero..ninety-nine, hyphen or space) before comparison. The API arm-C flags were computed by digit-string equality, which mis-scores the six word-target paper items (word-add-mult, word-mult-sub, word-parens, word-sub-mult, word-add-add, word-div-sub); they were recomputed here from the stored text: `correct_greedy` changed for 3 of 43 API items (word-add-mult: False→True, word-mult-sub: False→True, word-parens: False→True); `admissible` changed for 2 (word-add-mult: False→True, word-mult-sub: False→True). Box `greedy.correct` is trusted as word-aware; a local re-parse disagrees on 10 of 210 box continuations: paper_div-sub-left_space, paper_div-sub-left_nospace, paper_nested-mult-add-div_space, paper_nested-mult-add-div_nospace, paper_mult-div-left_space, paper_mult-div-left_nospace.

Other schema notes: (i) layer 63 of every 64-layer response is the model's own next-token top-8 (no Jacobian): the lens rank here is the min over layers 0–62, and the frozen scorer's `best_rank` (all 64 layers) is shown once as a sensitivity line. (ii) The nospace readout token is ` =` for 97 prompts and ` equals` for 8 (the word-form paper items). (iii) The API gate ran on the with-space variant only; item admissibility is a per-item property applied to both variants.

## 2. Hypotheses

### H1 — paper set, hosted lens, correctness-filtered

> **Frozen rule:** "on items the model answers correctly, number pass@1 ≥ 0.35 and pass@3 ≥ 0.65 with the hosted lens; paired sign test p < 0.01. Rule: both thresholds met → \"replicates\"; else \"does not replicate as scored\"."
> Jeffrey: pass@1 = 0.50, pass@3 = 0.75 on correctly answered items. Agent: 0.45 / 0.78.

Arm A (API, hosted n1000 lens, `space` variant, 55 paper items):

| column | n | pass@1 [CP] | pass@3 [CP] | pass@5 | pass@8 | MRR | sign test better/worse/tie, two-sided p |
|---|---|---|---|---|---|---|---|
| unfiltered | 55 | 24/55 = 0.44 [0.30, 0.58] | 41/55 = 0.75 [0.61, 0.85] | 0.82 | 0.85 | 0.58 | 42/12/1, p=5.2e-05 |
| correct-only | 47 | 16/47 = 0.34 [0.21, 0.49] | 33/47 = 0.70 [0.55, 0.83] | 0.79 | 0.83 | 0.51 | 34/12/1, p=1.6e-03 |
| admissible-only | 15 | 3/15 = 0.20 [0.04, 0.48] | 9/15 = 0.60 [0.32, 0.84] | 0.73 | 0.80 | 0.37 | 9/5/1, p=4.2e-01 |

- Verdict on the frozen column (correct-only, n=47 of 55 paper items with a correctness record): **does not replicate as scored** (pass@1 0.34 vs 0.35, pass@3 0.70 vs 0.65, p=1.6e-03 vs 0.01).
- Unfiltered (n=55): **replicates** (pass@1 0.44 vs 0.35, pass@3 0.75 vs 0.65, p=5.2e-05 vs 0.01).
- Admissible-only (addendum-2 headline, n=15): **does not replicate as scored** (pass@1 0.20 vs 0.35, pass@3 0.60 vs 0.65, p=4.2e-01 vs 0.01). Addendum-2 rule: 15 admissible of 105 (< 30) → **H1/H2 not testable as designed** on the admissible column (gate incomplete: 43/105 gated).

Box C1 (hosted lens applied locally, same items; pipeline trusted by the C1-vs-A gate):

*top-8-sliced ranks (API parity)*

| column | n | pass@1 [CP] | pass@3 [CP] | pass@5 | pass@8 | MRR | sign test better/worse/tie, two-sided p |
|---|---|---|---|---|---|---|---|
| unfiltered | 55 | 23/55 = 0.42 [0.29, 0.56] | 41/55 = 0.75 [0.61, 0.85] | 0.82 | 0.85 | 0.58 | 42/12/1, p=5.2e-05 |
| correct-only | 47 | 15/47 = 0.32 [0.19, 0.47] | 33/47 = 0.70 [0.55, 0.83] | 0.79 | 0.83 | 0.50 | 34/12/1, p=1.6e-03 |
| admissible-only | 15 | 3/15 = 0.20 [0.04, 0.48] | 9/15 = 0.60 [0.32, 0.84] | 0.73 | 0.80 | 0.37 | 9/5/1, p=4.2e-01 |

- correct-only verdict (top-8-sliced ranks (API parity)): **does not replicate as scored** (pass@1 0.32 vs 0.35, pass@3 0.70 vs 0.65, p=1.6e-03 vs 0.01).
*exact full-vocab ranks*

| column | n | pass@1 [CP] | pass@3 [CP] | pass@5 | pass@8 | MRR | sign test better/worse/tie, two-sided p |
|---|---|---|---|---|---|---|---|
| unfiltered | 55 | 24/55 = 0.44 [0.30, 0.58] | 41/55 = 0.75 [0.61, 0.85] | 0.82 | 0.85 | 0.58 | 42/12/1, p=5.2e-05 |
| correct-only | 47 | 16/47 = 0.34 [0.21, 0.49] | 33/47 = 0.70 [0.55, 0.83] | 0.79 | 0.83 | 0.51 | 34/12/1, p=1.6e-03 |
| admissible-only | 15 | 3/15 = 0.20 [0.04, 0.48] | 9/15 = 0.60 [0.32, 0.84] | 0.73 | 0.80 | 0.37 | 9/5/1, p=4.2e-01 |

- correct-only verdict (exact full-vocab ranks): **does not replicate as scored** (pass@1 0.34 vs 0.35, pass@3 0.70 vs 0.65, p=1.6e-03 vs 0.01).

### H2 — held-out 50

> **Frozen rule:** "pass@1 and pass@3 on items_new50 within the 95% CP interval of the paper-set values. Rule: inside → \"generalises beyond the paper's items\"; outside on the low side → \"paper items favourable\"; report the numbers either way."
> Jeffrey: held-out pass@1 = 0.50, pass@3 = 0.75 ("should match the paper's numbers"). Agent: 0.40 / 0.72.

Arm A (API, hosted lens, `space` variant, 50 held-out items):

| column | n | pass@1 [CP] | pass@3 [CP] | pass@5 | pass@8 | MRR | sign test better/worse/tie, two-sided p |
|---|---|---|---|---|---|---|---|
| unfiltered | 50 | 24/50 = 0.48 [0.34, 0.63] | 36/50 = 0.72 [0.58, 0.84] | 0.78 | 0.82 | 0.62 | 39/11/0, p=9.0e-05 |
| correct-only | 50 | 24/50 = 0.48 [0.34, 0.63] | 36/50 = 0.72 [0.58, 0.84] | 0.78 | 0.82 | 0.62 | 39/11/0, p=9.0e-05 |
| admissible-only | 0 | – | – | – | – | – | – |

- unfiltered: held-out pass@1 0.48 vs paper CP [0.30, 0.58] → inside; pass@3 0.72 vs paper CP [0.61, 0.85] → inside → **generalises beyond the paper's items**.
- correct-only: held-out pass@1 0.48 vs paper CP [0.21, 0.49] → inside; pass@3 0.72 vs paper CP [0.55, 0.83] → inside → **generalises beyond the paper's items**.
- admissible-only: not scorable (held-out n=0, paper n=15) — no held-out item has been gated yet.
- Addendum-2 headline column (admissible-only): **not testable as designed** (15 admissible < 30).

Box C1 on the held-out 50 (exact ranks):

| column | n | pass@1 [CP] | pass@3 [CP] | pass@5 | pass@8 | MRR | sign test better/worse/tie, two-sided p |
|---|---|---|---|---|---|---|---|
| unfiltered | 50 | 24/50 = 0.48 [0.34, 0.63] | 36/50 = 0.72 [0.58, 0.84] | 0.78 | 0.82 | 0.62 | 39/11/0, p=9.0e-05 |
| correct-only | 50 | 24/50 = 0.48 [0.34, 0.63] | 36/50 = 0.72 [0.58, 0.84] | 0.78 | 0.82 | 0.62 | 39/11/0, p=9.0e-05 |
| admissible-only | 0 | – | – | – | – | – | – |

### H3 — carrier of two-digit hits

> **Frozen rule:** "among two-digit intermediates with a rank-1 hit, ≥ 2/3 are CJK-only or CJK+word. Rule as stated."
> Jeffrey: 0.90 of two-digit rank-1 hits involve the Chinese numeral. Agent: 0.85.

- Arm A, all (47 two-digit items, 22 at rank 1): CJK-only or CJK+word 20/22 = 0.91; breakdown {'CJK+word': 3, 'CJK-only': 17, 'word-only': 2} → **holds** (≥ 2/3).
- Arm A, paper (22 two-digit items, 9 at rank 1): CJK-only or CJK+word 9/9 = 1.00; breakdown {'CJK+word': 2, 'CJK-only': 7} → holds (≥ 2/3).
- Arm A, new (25 two-digit items, 13 at rank 1): CJK-only or CJK+word 11/13 = 0.85; breakdown {'CJK-only': 10, 'CJK+word': 1, 'word-only': 2} → holds (≥ 2/3).
- Box hosted (n1000) (exact ranks), all: 47 two-digit, 22 at rank 1; CJK-only or CJK+word 20/22 = 0.91; {'CJK+word': 3, 'CJK-only': 17, 'word-only': 2}.
- Box neel5 (5-prompt fit) (exact ranks), all: 47 two-digit, 24 at rank 1; CJK-only or CJK+word 20/24 = 0.83; {'CJK-only': 19, 'word-only': 4, 'CJK+word': 1}.

### H4 — lens quality (box only: hosted (n1000) vs neel5 (5-prompt fit) on the same forward pass)

> **Frozen rule:** "25-prompt lens pass@1 on the 105 items ≥ half of the 1000-prompt lens pass@1 → \"effect survives Neel's lens; lens quality does not explain the null\"; < half → \"lens quality is a sufficient explanation\"."
> **Addendum 5 (asymmetric reading, n=5):** "If the effect survives it, H4 is answered in the direction 'lens quality does not explain the null' with more force than the frozen rule required. If the effect fails under it, H4 is inconclusive, not negative, and the note says 'not tested at n=25'."
> Jeffrey: ratio ≈ 0.9. Agent: 0.6.

| ranks | column | hosted (n1000) pass@1 | neel5 (5-prompt fit) pass@1 | ratio neel5/hosted | rule (≥ 0.5) |
|---|---|---|---|---|---|
| top-8-sliced (API parity) | unfiltered | 47/105 = 0.45 [0.35, 0.55] | 44/105 = 0.42 [0.32, 0.52] | 0.94 | survives |
| top-8-sliced (API parity) | correct-only | 39/97 = 0.40 [0.30, 0.51] | 36/97 = 0.37 [0.28, 0.48] | 0.92 | survives |
| top-8-sliced (API parity) | admissible-only | 3/15 = 0.20 [0.04, 0.48] | 4/15 = 0.27 [0.08, 0.55] | 1.33 | survives |
| exact full-vocab | unfiltered | 48/105 = 0.46 [0.36, 0.56] | 45/105 = 0.43 [0.33, 0.53] | 0.94 | survives |
| exact full-vocab | correct-only | 40/97 = 0.41 [0.31, 0.52] | 37/97 = 0.38 [0.28, 0.49] | 0.93 | survives |
| exact full-vocab | admissible-only | 3/15 = 0.20 [0.04, 0.48] | 4/15 = 0.27 [0.08, 0.55] | 1.33 | survives |

neel5 (5-prompt fit) exact pass@1 restricted to its Jacobian-carrying layers (0–61; layer 62 is the fit target, read without a Jacobian): 43/105 = 0.41 [0.31, 0.51].

**Verdict (exact ranks, unfiltered, ratio 0.94 ≥ 0.5): the effect survives the 5-prompt lens → "lens quality does not explain the null", with more force than the frozen rule required (addendum 5). The lens is a 5-prompt fit, not n=25.**
Top-8-sliced ratio (API parity): 0.94. Admissible-only column: ratio 1.33.

### H5 — decoy (wrong-precedence value)

> **Frozen rule:** "correct intermediate beats decoy in > 60% of decidable items, pooled over 105."
> Jeffrey: win rate ≈ 0.85 of decided items (v1: 9/11 = 0.82). Agent: 0.70.

- Arm A, pooled 105, `space`: 67 items with a decoy; win/loss/tie 49/8/10 (ties with both absent from the top-8: 5; decoys ≥ 100 with no single-token form: 2). Win rate of decided items 49/57 = 0.86 [0.74, 0.94] → > 0.60 → **holds**; win rate of all decoy items 49/67 = 0.73 → > 0.60 → holds.
- Arm A, paper: 17 items with a decoy; win/loss/tie 12/2/3 (ties with both absent from the top-8: 1; decoys ≥ 100 with no single-token form: 0). Win rate of decided items 12/14 = 0.86 [0.57, 0.98] → > 0.60 → **holds**; win rate of all decoy items 12/17 = 0.71 → > 0.60 → holds.
- Arm A, held-out: 50 items with a decoy; win/loss/tie 37/6/7 (ties with both absent from the top-8: 4; decoys ≥ 100 with no single-token form: 2). Win rate of decided items 37/43 = 0.86 [0.72, 0.95] → > 0.60 → **holds**; win rate of all decoy items 37/50 = 0.74 → > 0.60 → holds.
- Arm A, pooled, correct-only: 63 items with a decoy; win/loss/tie 47/8/8 (ties with both absent from the top-8: 5; decoys ≥ 100 with no single-token form: 2). Win rate of decided items 47/55 = 0.85 [0.73, 0.94] → > 0.60 → **holds**; win rate of all decoy items 47/63 = 0.75 → > 0.60 → holds.
- Arm A, pooled, admissible-only: 5 items with a decoy; win/loss/tie 5/0/0 (ties with both absent from the top-8: 0; decoys ≥ 100 with no single-token form: 0). Win rate of decided items 5/5 = 1.00 [0.48, 1.00] → > 0.60 → **holds**; win rate of all decoy items 5/5 = 1.00 → > 0.60 → holds.
- Box hosted (n1000), exact ranks, pooled: 58 items with a decoy; win/loss/tie 43/8/7 (ties with both absent from the top-8: 1; decoys ≥ 100 with no single-token form: 0). Win rate of decided items 43/51 = 0.84 [0.71, 0.93] → > 0.60 → **holds**; win rate of all decoy items 43/58 = 0.74 → > 0.60 → holds.
- Box neel5 (5-prompt fit), exact ranks, pooled: 58 items with a decoy; win/loss/tie 48/6/4 (ties with both absent from the top-8: 1; decoys ≥ 100 with no single-token form: 0). Win rate of decided items 48/54 = 0.89 [0.77, 0.96] → > 0.60 → **holds**; win rate of all decoy items 48/58 = 0.83 → > 0.60 → holds.

The frozen text says "decidable items"; Jeffrey's prediction is phrased over *decided* items (ties excluded, v1's 9/11). Both denominators are given; the decided-items rate is the one compared with the prediction.

### H6 — readout position (space vs nospace)

> **Frozen rule:** "every item is also read with the trailing space removed (`prompt_nospace`), readout at the `=` token. Rule: if pass@1 falls by more than half relative to the with-space form → \"readout position is a sufficient explanation for a null\"; if within half → \"position does not explain it\"."
> Jeffrey: ≈ 1.0 ("it's always going to be in either the =_ token or the = token itself"). Agent: 0.25.

Correct-only here = items whose greedy continuation is correct under BOTH variants (each variant carries its own `correct_greedy`); admissible-only is the per-item gate.

| source | set | column | space pass@1 | nospace pass@1 | ratio nospace/space | space pass@3 | nospace pass@3 | both hit / space only / nospace only / neither |
|---|---|---|---|---|---|---|---|---|
| arm A | all | unfiltered | 48/105 = 0.46 [0.36, 0.56] | 4/105 = 0.04 [0.01, 0.09] | 0.08 | 0.73 | 0.07 | 4/44/0/57 |
| arm A | all | correct-only | 34/77 = 0.44 [0.33, 0.56] | 1/77 = 0.01 [0.00, 0.07] | 0.03 | 0.78 | 0.04 | 1/33/0/43 |
| arm A | all | admissible-only | 3/15 = 0.20 [0.04, 0.48] | 0/15 = 0.00 [0.00, 0.22] | 0.00 | 0.60 | 0.00 | 0/3/0/12 |
| arm A | paper | unfiltered | 24/55 = 0.44 [0.30, 0.58] | 4/55 = 0.07 [0.02, 0.18] | 0.17 | 0.75 | 0.11 | 4/20/0/31 |
| arm A | paper | correct-only | 13/37 = 0.35 [0.20, 0.53] | 1/37 = 0.03 [0.00, 0.14] | 0.08 | 0.76 | 0.05 | 1/12/0/24 |
| arm A | paper | admissible-only | 3/15 = 0.20 [0.04, 0.48] | 0/15 = 0.00 [0.00, 0.22] | 0.00 | 0.60 | 0.00 | 0/3/0/12 |
| arm A | new | unfiltered | 24/50 = 0.48 [0.34, 0.63] | 0/50 = 0.00 [0.00, 0.07] | 0.00 | 0.72 | 0.02 | 0/24/0/26 |
| arm A | new | correct-only | 21/40 = 0.53 [0.36, 0.68] | 0/40 = 0.00 [0.00, 0.09] | 0.00 | 0.80 | 0.03 | 0/21/0/19 |
| box hosted (n1000) top8 | all | unfiltered | 47/105 = 0.45 [0.35, 0.55] | 5/105 = 0.05 [0.02, 0.11] | 0.11 | 0.73 | 0.07 | 5/42/0/58 |
| box hosted (n1000) exact | all | unfiltered | 48/105 = 0.46 [0.36, 0.56] | 5/105 = 0.05 [0.02, 0.11] | 0.10 | 0.73 | 0.07 | 5/43/0/57 |
| box neel5 (5-prompt fit) top8 | all | unfiltered | 44/105 = 0.42 [0.32, 0.52] | 4/105 = 0.04 [0.01, 0.09] | 0.09 | 0.56 | 0.06 | 4/40/0/61 |
| box neel5 (5-prompt fit) exact | all | unfiltered | 45/105 = 0.43 [0.33, 0.53] | 4/105 = 0.04 [0.01, 0.09] | 0.09 | 0.57 | 0.06 | 4/41/0/60 |

**Verdict (arm A, pooled 105, unfiltered): ratio 0.08 → "readout position is a sufficient explanation for a null" (fell by more than half).** Correct-only ratio 0.03. Admissible-only ratio 0.00.

### H7 — digit count and CJK synonyms (held-out 50)

> **Frozen rule:** "two-digit vs single-digit pass@1 with CJK in the synonym set, and the same contrast with CJK removed. Rule: if the two-digit rate without CJK falls below half of the two-digit rate with CJK, tokenization is a sufficient explanation for two-digit misses."
> Jeffrey: two-digit pass@1 with CJK ≈ 0.50 (± 0.1), without ≈ 0.05. Agent: 0.40 / 0.10.

| source | set | column | two-digit, with CJK | two-digit, no CJK | single-digit, with CJK | single-digit, no CJK | ratio two-digit noCJK/CJK | rule (< 0.5) |
|---|---|---|---|---|---|---|---|---|
| arm A | new | unfiltered | 13/25 = 0.52 [0.31, 0.72] | 3/25 = 0.12 [0.03, 0.31] | 11/25 = 0.44 [0.24, 0.65] | 11/25 = 0.44 [0.24, 0.65] | 0.23 | tokenization sufficient |
| arm A | new | correct-only | 13/25 = 0.52 [0.31, 0.72] | 3/25 = 0.12 [0.03, 0.31] | 11/25 = 0.44 [0.24, 0.65] | 11/25 = 0.44 [0.24, 0.65] | 0.23 | tokenization sufficient |
| arm A | paper | unfiltered | 9/22 = 0.41 [0.21, 0.64] | 2/22 = 0.09 [0.01, 0.29] | 15/33 = 0.45 [0.28, 0.64] | 15/33 = 0.45 [0.28, 0.64] | 0.22 | tokenization sufficient |
| arm A | paper | correct-only | 7/20 = 0.35 [0.15, 0.59] | 2/20 = 0.10 [0.01, 0.32] | 9/27 = 0.33 [0.17, 0.54] | 9/27 = 0.33 [0.17, 0.54] | 0.29 | tokenization sufficient |
| arm A | paper | admissible-only | 3/8 = 0.38 [0.09, 0.76] | 2/8 = 0.25 [0.03, 0.65] | 0/7 = 0.00 [0.00, 0.41] | 0/7 = 0.00 [0.00, 0.41] | 0.67 | not below half |
| arm A | all | unfiltered | 22/47 = 0.47 [0.32, 0.62] | 5/47 = 0.11 [0.04, 0.23] | 26/58 = 0.45 [0.32, 0.58] | 26/58 = 0.45 [0.32, 0.58] | 0.23 | tokenization sufficient |
| arm A | all | correct-only | 20/45 = 0.44 [0.30, 0.60] | 5/45 = 0.11 [0.04, 0.24] | 20/52 = 0.38 [0.25, 0.53] | 20/52 = 0.38 [0.25, 0.53] | 0.25 | tokenization sufficient |
| arm A | all | admissible-only | 3/8 = 0.38 [0.09, 0.76] | 2/8 = 0.25 [0.03, 0.65] | 0/7 = 0.00 [0.00, 0.41] | 0/7 = 0.00 [0.00, 0.41] | 0.67 | not below half |
| box hosted (n1000) exact | new | unfiltered | 13/25 = 0.52 [0.31, 0.72] | 3/25 = 0.12 [0.03, 0.31] | 11/25 = 0.44 [0.24, 0.65] | 11/25 = 0.44 [0.24, 0.65] | 0.23 | tokenization sufficient |
| box neel5 (5-prompt fit) exact | new | unfiltered | 13/25 = 0.52 [0.31, 0.72] | 3/25 = 0.12 [0.03, 0.31] | 8/25 = 0.32 [0.15, 0.54] | 8/25 = 0.32 [0.15, 0.54] | 0.23 | tokenization sufficient |

**Verdict (arm A, held-out 50, unfiltered): two-digit pass@1 with CJK 13/25 = 0.52, without CJK 3/25 = 0.12 (ratio 0.23) → "tokenization is a sufficient explanation for two-digit misses". Single-digit pass@1 11/25 = 0.44 with CJK, 11/25 without.**

## 3. Controls

### 3a. Layer-63 next-token control (addendum 4)

> "(a) the fraction of rank-1 items whose intermediate is ALSO in the model's top-8 next tokens at the readout position; (b) the 'workspace-only' subset: items where the intermediate reaches rank ≤ 3 at some source layer 0–62 but is absent from the model's top-8 at layer 63 — hits the logit lens would not show; (c) the same for the decoy."

| source | set | variant | rank-1 items | of which in L63 top-8 (a) | rank ≤ 3 items | workspace-only (b) | decoy rank-1 | decoy in L63 top-8 | decoy rank ≤ 3 | decoy workspace-only (c) | intermediate in L63 top-8 (all items) | L63 rank 1 (next token IS the intermediate) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arm A | all | space | 48 | 24 (0.50) | 77 | 30 (0.39) | 5 | 4 | 19 | 4 | 54/105 | 4/105 |
| arm A | all | nospace | 4 | 3 (0.75) | 7 | 3 (0.43) | 0 | 0 | 1 | 0 | 4/105 | 0/105 |
| arm A | paper | space | 24 | 14 (0.58) | 41 | 15 (0.37) | 2 | 2 | 5 | 0 | 30/55 | 4/55 |
| arm A | paper | nospace | 4 | 3 (0.75) | 6 | 2 (0.33) | 0 | 0 | 1 | 0 | 4/55 | 0/55 |
| arm A | new | space | 24 | 10 (0.42) | 36 | 15 (0.42) | 3 | 2 | 14 | 4 | 24/50 | 0/50 |
| arm A | new | nospace | 0 | – | 1 | 1 (1.00) | 0 | 0 | 0 | 0 | 0/50 | 0/50 |
| box hosted (n1000) top8 | all | space | 47 | 24 (0.51) | 77 | 28 (0.36) | 7 | 6 | 19 | 4 | 56/105 | 4/105 |
| box hosted (n1000) exact | all | space | 48 | 25 (0.52) | 77 | 28 (0.36) | 7 | 6 | 20 | 5 | 56/105 | 4/105 |
| box neel5 (5-prompt fit) top8 | all | space | 44 | 20 (0.45) | 59 | 27 (0.46) | 2 | 2 | 4 | 2 | 56/105 | 4/105 |
| box neel5 (5-prompt fit) exact | all | space | 45 | 21 (0.47) | 60 | 28 (0.47) | 2 | 2 | 4 | 2 | 56/105 | 4/105 |

Workspace-only items (arm A, pooled, space; rank ≤ 3 at a source layer, absent from the layer-63 top-8): 30 — `paper_mult-parens-add`, `paper_add-mult-right`, `paper_mult-add-left`, `paper_mult-sub-left`, `paper_sub-add-left-right`, `paper_parens-add-div`, `paper_parens-sub-div`, `paper_chain-add-mult-add`, `paper_word-add-mult`, `paper_mod-add`, `paper_mod-mult`, `paper_nested4-sub-add-mult-sub`, `paper_word-add-add`, `paper_mixed-mult-add`, `paper_floordiv-add`, `new_new01`, `new_new10`, `new_new12`, `new_new15`, `new_new18`, `new_new19`, `new_new20`, `new_new26`, `new_new27`, `new_new28`, `new_new30`, `new_new34`, `new_new38`, `new_new41`, `new_new46`.
Sensitivity — the frozen scorer's `best_rank` over all 64 layers (layer 63 included): pass@1 48/105 and pass@3 79/105 vs 48/105 and 77/105 over layers 0–62; layer 63 improves the rank for 7 items. On the box, `exact` L63 ranks are full-vocab next-token ranks; the top-8 rule above is applied to them for parity.

### 3b. Uninvolved-digit baseline

| source | set | variant | readings (digits × items) | pass@1 | pass@3 | pass@8 | per-item mean fraction hit @1 / @3 | intermediate pass@1 / pass@3 (same items) |
|---|---|---|---|---|---|---|---|---|
| arm A | all | space | 548 | 0.06 | 0.28 | 0.74 | 0.06 / 0.29 | 0.46 / 0.73 |
| arm A | all | nospace | 548 | 0.00 | 0.00 | 0.05 | 0.00 / 0.00 | 0.04 / 0.07 |
| arm A | paper | space | 298 | 0.07 | 0.26 | 0.75 | 0.07 / 0.24 | 0.44 / 0.75 |
| arm A | paper | nospace | 298 | 0.00 | 0.00 | 0.06 | 0.00 / 0.00 | 0.07 / 0.11 |
| arm A | new | space | 250 | 0.05 | 0.32 | 0.74 | 0.05 / 0.33 | 0.48 / 0.72 |
| arm A | new | nospace | 250 | 0.00 | 0.00 | 0.03 | 0.00 / 0.00 | 0.00 / 0.02 |
| box hosted (n1000) top8 | all | space | 548 | 0.08 | 0.30 | 0.75 | 0.07 / 0.29 | 0.45 / 0.73 |
| box hosted (n1000) exact | all | space | 548 | median exact rank of uninvolved digits 6 | – | – | – | median exact rank of the intermediate 2 |
| box neel5 (5-prompt fit) top8 | all | space | 548 | 0.00 | 0.02 | 0.07 | 0.00 / 0.01 | 0.42 / 0.56 |
| box neel5 (5-prompt fit) exact | all | space | 548 | median exact rank of uninvolved digits 51 | – | – | – | median exact rank of the intermediate 3 |

### 3c. Decoy win/loss/tie

See H5 above (arm A pooled: win/loss/tie 49/8/10 of 67 items with a decoy).

### 3d. Target (spoken) rank beside the intermediate (unspoken) rank at the readout

| source | set | variant | n | target pass@1 | target pass@3 | intermediate pass@1 | intermediate pass@3 | target rank < intermediate rank / equal / > | operation pass@1 |
|---|---|---|---|---|---|---|---|---|---|
| arm A | all | space | 105 | 0.59 | 0.80 | 0.46 | 0.73 | 42/33/30 | 0.41 |
| arm A | all | nospace | 105 | 0.09 | 0.13 | 0.04 | 0.07 | 24/65/16 | 0.00 |
| arm A | paper | space | 55 | 0.78 | 0.93 | 0.44 | 0.75 | 27/20/8 | 0.44 |
| arm A | paper | nospace | 55 | 0.16 | 0.22 | 0.07 | 0.11 | 19/24/12 | 0.00 |
| arm A | new | space | 50 | 0.38 | 0.66 | 0.48 | 0.72 | 15/13/22 | 0.38 |
| arm A | new | nospace | 50 | 0.00 | 0.04 | 0.00 | 0.02 | 5/41/4 | 0.00 |
| box hosted (n1000) exact | all | space | 105 | 0.59 | 0.80 | 0.46 | 0.73 | 42/30/33 | 0.41 |
| box neel5 (5-prompt fit) exact | all | space | 105 | 0.79 | 0.83 | 0.43 | 0.57 | 53/33/19 | 0.40 |

Word-target paper items have their target mapped to the number (`fourteen` → the 14 synonyms) for this row; the operation row echoes the prompt symbol at early layers (frozen scorer caveat).

## 4. Persistence descriptives (reported, not hypothesis-tested)

### 4a. Layer span at rank ≤ 3 (source layers 0–62)

| source | set | variant | items | items with any rank ≤ 3 layer | median span (layers, among those) | span ≥ 3 layers | median first / last layer | median best layer (rank-1 items) |
|---|---|---|---|---|---|---|---|---|
| arm A | all | space | 105 | 77 | 4 | 55 | 54 / 62 | 55 |
| arm A | all | nospace | 105 | 7 | 1 | 2 | 60 / 62 | 61 |
| arm A | paper | space | 55 | 41 | 4 | 27 | 54 / 62 | 55 |
| arm A | paper | nospace | 55 | 6 | 1 | 1 | 61 / 62 | 61 |
| arm A | new | space | 50 | 36 | 5 | 28 | 54 / 62 | 55 |
| arm A | new | nospace | 50 | 1 | 3 | 1 | 54 / 62 | – |
| box hosted (n1000) top8 | all | space | 105 | 77 | 4 | 56 | 53 / 62 | 55 |
| box hosted (n1000) exact | all | space | 105 | 77 | 4 | 56 | 53 / 62 | 55 |
| box neel5 (5-prompt fit) top8 | all | space | 105 | 59 | 3 | 41 | 55 / 58 | 55 |
| box neel5 (5-prompt fit) exact | all | space | 105 | 60 | 3 | 41 | 55 / 58 | 55 |

Span histogram (arm A, pooled, space; number of source layers with rank ≤ 3, 10 = ≥ 10): 0:28, 1:16, 2:6, 3:10, 4:9, 5:13, 6:5, 7:5, 8:2, 9:3, 10:8.

### 4b. Box: median exact rank by layer (intermediate / target / decoy), every 4 layers

| layer | hosted (n1000) inter | hosted (n1000) target | hosted (n1000) decoy | neel5 (5-prompt fit) inter | neel5 (5-prompt fit) target | neel5 (5-prompt fit) decoy |
|---|---|---|---|---|---|---|
| 0 | 3253 | 3850 | 3818 | 43582 | 31200 | 42947 |
| 4 | 20752 | 21667 | 17034 | 78349 | 54832 | 94676 |
| 8 | 52426 | 35349 | 72368 | 16345 | 17839 | 17738 |
| 12 | 52131 | 38701 | 52594 | 81165 | 81542 | 64037 |
| 16 | 3139 | 2688 | 5091 | 26725 | 33133 | 49502 |
| 20 | 237 | 242 | 306 | 45192 | 37064 | 52442 |
| 24 | 142 | 150 | 180 | 7298 | 5864 | 16406 |
| 28 | 115 | 125 | 147 | 1366 | 1908 | 2372 |
| 32 | 177 | 189 | 214 | 1262 | 984 | 1944 |
| 36 | 62 | 315 | 197 | 2043 | 3816 | 2578 |
| 40 | 38 | 139 | 101 | 1195 | 2363 | 1864 |
| 44 | 65 | 146 | 109 | 906 | 1774 | 1552 |
| 48 | 145 | 311 | 286 | 1717 | 2932 | 3666 |
| 52 | 29 | 182 | 294 | 567 | 4056 | 9563 |
| 56 | 7 | 17 | 66 | 24 | 46 | 1154 |
| 60 | 42 | 2 | 350 | 629 | 1 | 8706 |
| 62 | 6 | 12 | 29 | 22 | 3 | 90 |
| 63 (next-token, no Jacobian) | 8 | 32 | 183 | 8 | 32 | 183 |

Items per curve: hosted (n1000) n=105, neel5 (5-prompt fit) n=105. Full 64-number curves (median exact rank):
- hosted (n1000) inter: 3253 30337 53363 169931 20752 31144 15884 102941 52426 78884 14406 98653 52131 37224 2298 8831 3139 1474 782 372 237 231 181 158 142 175 120 161 115 148 161 210 177 135 118 59 62 74 71 40 38 75 160 70 65 61 77 53 145 163 68 75 29 15 10 7 7 14 10 23 42 12 6 8
- hosted (n1000) target: 3850 37319 63771 174413 21667 38927 9430 86537 35349 60264 11437 77095 38701 21448 1551 7374 2688 1423 522 356 242 316 188 175 150 221 142 173 125 139 158 212 189 244 253 344 315 203 184 112 139 187 295 187 146 125 168 168 311 531 285 358 182 129 99 50 17 9 3 2 2 3 12 32
- hosted (n1000) decoy: 3818 36901 59248 177068 17034 33125 28532 104552 72368 66790 19294 96450 52594 37296 3902 9752 5091 2392 843 508 306 262 170 170 180 274 162 208 147 205 180 282 214 184 232 237 197 132 136 100 101 134 300 202 109 122 140 181 286 454 302 288 294 183 163 128 66 113 90 127 350 132 29 183
- neel5 (5-prompt fit) inter: 43582 46979 158309 110684 78349 23163 33542 13863 16345 24856 136283 99281 81165 56208 48151 55068 26725 50413 37084 57925 45192 23477 13875 8254 7298 3164 6334 3013 1366 828 614 1046 1262 1435 1927 1839 2043 1615 1685 1132 1195 1351 1835 1567 906 750 506 862 1717 1537 1176 946 567 632 166 33 24 63 93 593 629 281 22 8
- neel5 (5-prompt fit) target: 31200 45779 157967 98513 54832 28896 36733 15233 17839 23673 115972 78406 81542 60273 48678 35282 33133 21916 20846 33876 37064 16234 8355 5783 5864 4365 5671 3430 1908 1289 715 1129 984 1862 4082 3192 3816 2679 3386 2777 2363 2666 2378 2696 1774 1303 977 1742 2932 2652 4026 4012 4056 4498 1782 99 46 15 3 1 1 2 3 32
- neel5 (5-prompt fit) decoy: 42947 46211 166907 104242 94676 22683 36713 13388 17738 27270 127860 64500 64037 59968 46052 56726 49502 55536 43956 67884 52442 33062 21158 17928 16406 10963 16970 5388 2372 1484 1264 1896 1944 1628 3177 2780 2578 2103 3068 1650 1864 2088 2272 2276 1552 1093 818 1332 3666 3396 4350 5876 9563 12502 9478 2858 1154 1694 3463 15592 8706 6186 90 183

## 5. C1-reproduces-A gate (± 3 items on pass@1, `space` variant)

> "C1 must reproduce arm A's pass@1 within ±3 items or the local pipeline is not trusted and C2 is not reported." (logic of `box/check_c1_vs_a.py`; ranks over layers 0–62, top-8-sliced on both sides; the script's own all-64-layer `best_rank_topk` is shown beside it.)

- space: n=105 items; pass@1 A=48, C1(top-8)=47 (C1 top-64: 47), |diff|=1 (tolerance 3); per-item pass@1 agreement 104/105; all-64-layer rule as in check_c1_vs_a.py: A=48, C1=47 → **PASS: local pipeline trusted; C2 may be reported**.
  - disagreements (tag, A rank, C1 rank): paper_parens-add-mult 1/2
- nospace: n=105 items; pass@1 A=4, C1(top-8)=5 (C1 top-64: 5), |diff|=1 (tolerance 3); per-item pass@1 agreement 104/105; all-64-layer rule as in check_c1_vs_a.py: A=4, C1=5 → **PASS: local pipeline trusted; C2 may be reported** (secondary variant; the gate is decided on `space`).
  - disagreements (tag, A rank, C1 rank): paper_redundant-parens-div 2/1

## 6. Leak gate (addenda 2–3)

> admissible ⇔ ≥ 8/10 sampled continuations (T=0.8, 32 new tokens) write the target as the first number AND no continuation (sampled or greedy) writes the intermediate as a number before the target. "If fewer than 30 of the 105 items are admissible, H1/H2 are 'not testable as designed'."

- API arm C (with-space variant; unseeded; flags recomputed word-aware): n gated 43 (43 paper, 0 held-out); greedy correct 32/43; greedy writes the intermediate first 5/43; assert_rate ≥ 0.8 22/43 (median assert_rate 0.8); leak_any 25/43; **admissible 15/43** (stored flags said 13) — 15 < 30 → H1/H2 not testable as designed (gate 43/105 complete).
  - leak example `paper_parens-add-mult` (intermediate 5, target 20): `5 * 4 = 20 is the correct answer.⏎⏎And if we write⏎```javascript⏎var x = 2 + 3 * `
  - leak example `paper_mult-parens-add` (intermediate 7, target 14): `2 * 7 = 14\n现在，请你解决一下这个数学题：\n2 + 3 * (4 + 5)⏎⏎`
  - leak example `paper_sub-mult-right` (intermediate 12, target 5): `17 - 12 = 5？⏎⏎<think>⏎Here's a thinking process:⏎⏎1.  **Analyze User Input:**⏎   -`
- Box sampled pass (space variant; seeds 0–9, top_p 0.95): no gated items.

Headline gate used for the admissible-only columns (box preferred, else API): 43 gated, 15 admissible. Correctness records used for the correct-only columns (space): 105 items; correct 97.

Secondary, template-mediated (chat endpoint): arm B (8 tokens) n=105: first number == target 2, another number first 102, no number within 8 tokens 1 (addendum 1: mostly undetermined by construction). Arm B2 (48 tokens) n=0 (no files — `v2_api/raw_B2/` is empty).

## 7. Summary

On the paper's 55 items read through Neuronpedia's hosted n1000 lens at the space-after-`=` position, the unspoken intermediate reaches rank 1 in 24/55 (0.44) and rank ≤ 3 in 41/55 (0.75), against a paired uninvolved-digit sign test of 42/12/1 (p=5e-05). Restricted to the 47 paper items the model answers correctly (word-aware; 55 of 55 paper items have a correctness record), pass@1 is 0.34 and pass@3 0.70, so on the column the rule names H1 does not replicate as scored. The addendum-2 headline column has 15 admissible items of 43 gated (105 planned); below the 30-item line, H1 and H2 are not testable as designed on that column while the gate is incomplete. The held-out 50 give pass@1 0.48 and pass@3 0.72 unfiltered, inside the paper-set CP intervals (H2: generalises beyond the paper's items). Of the 22 two-digit rank-1 hits, 20 ride the CJK numeral (H3 holds at 0.91). The correct intermediate beats the wrong-precedence decoy in 49 of 57 decided items (0.86; H5 holds), with 10 ties. Of the three candidate explanations for the discrepancy with Neel's team: readout position is a sufficient explanation for a null (nospace/space pass@1 ratio 0.08, H6); tokenization/synonym coverage is a sufficient explanation for two-digit misses (two-digit pass@1 0.52 with CJK vs 0.12 without, H7); lens quality is ruled out as an explanation — the effect survives a 5-prompt lens at ratio 0.94 (H4; a 5-prompt fit, not n=25). The addendum-4 control shows that 24 of the 48 rank-1 intermediates are also in the model's own top-8 next tokens at the readout, and 30 of the 77 rank ≤ 3 items are workspace-only (absent from the layer-63 top-8); for those items the lens read is not reducible to next-token prediction, while the leak gate's finding that the intermediate opens narrated work in sampled continuations qualifies every hit that also sits in the top-8. The API leak gate has reached 43 of 105 items: greedy writes the intermediate first in 5, but at T=0.8 the intermediate leaks in 25, leaving 15 admissible.

## Appendix — per-item ranks (arm A, layers 0–62, top-8; 99 = absent)

| item | intermediate | target | decoy | space rank (noCJK) | best layer | span ≤3 (n, first–last) | L63 rank | nospace rank | target rank | decoy rank | median uninvolved | correct (space) | assert / leak / admissible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper_parens-add-mult | 5 | 20 | 12 | 1 (1) | 43 | 7, 39–62 | 2 | 99 | 1 | 99 | 8 | yes(b) | 0.9 / 1 / 0 |
| paper_parens-sub-mult | 5 | 15 | 12 | 3 (3) | 62 | 1, 62–62 | 2 | 99 | 99 | 99 | 53 | yes(b) | 1.0 / 0 / 1 |
| paper_mult-parens-add | 7 | 14 | 6 | 1 (1) | 52 | 7, 50–58 | 99 | 99 | 2 | 6 | 6 | yes(b) | 0.5 / 1 / 0 |
| paper_mult-parens-sub | 3 | 6 | 12 | 3 (3) | 62 | 1, 62–62 | 2 | 99 | 1 | 99 | 53.5 | yes(b) | 0.8 / 0 / 1 |
| paper_add-mult-right | 12 | 14 | 5 | 1 (1) | 55 | 4, 55–58 | 99 | 99 | 2 | 2 | 2 | yes(b) | 1.0 / 0 / 1 |
| paper_mult-add-left | 12 | 14 | 6 | 1 (2) | 55 | 4, 55–58 | 99 | 99 | 1 | 2 | 4.5 | yes(b) | 0.8 / 0 / 1 |
| paper_sub-mult-right | 12 | 5 | 14 | 99 (99) | – | 0 | 99 | 99 | 1 | 99 | 3.5 | yes(b) | 0.7 / 1 / 0 |
| paper_mult-sub-left | 20 | 12 | – | 1 (2) | 55 | 5, 54–58 | 99 | 99 | 1 | – | 6.5 | yes(b) | 0.7 / 1 / 0 |
| paper_add-sub-left-right | 15 | 12 | 2 | 99 (99) | – | 0 | 99 | 99 | 1 | 5 | 6 | yes(b) | 0.6 / 1 / 0 |
| paper_sub-add-left-right | 10 | 13 | 8 | 2 (2) | 14 | 1, 14–14 | 99 | 99 | 1 | 3 | 7.5 | yes(b) | 0.9 / 1 / 0 |
| paper_div-parens-add | 5 | 4 | 10 | 3 (3) | 62 | 1, 62–62 | 4 | 99 | 1 | 4 | 7 | yes(b) | 0.7 / 1 / 0 |
| paper_parens-add-div | 15 | 5 | – | 1 (99) | 51 | 2, 51–53 | 99 | 99 | 1 | – | 5 | yes(b) | 0.6 / 1 / 0 |
| paper_parens-sub-div | 12 | 3 | – | 1 (99) | 51 | 1, 51–51 | 99 | 99 | 1 | – | 6.5 | yes(b) | 0.6 / 1 / 0 |
| paper_div-sub-left | 4 | 3 | – | 1 (1) | 57 | 9, 54–62 | 3 | 5 | 1 | – | 7.5 | yes(b) | 0.5 / 1 / 0 |
| paper_chain-add-mult-add | 6 | 11 | – | 2 (2) | 53 | 4, 52–56 | 99 | 99 | 2 | – | 7 | yes(b) | 0.8 / 1 / 0 |
| paper_chain-mult-sub-add | 6 | 7 | – | 1 (1) | 58 | 5, 58–62 | 1 | 8 | 2 | – | 7 | no(b) | 0.0 / 1 / 0 |
| paper_chain-sub-mult-sub | 6 | 5 | – | 2 (2) | 52 | 3, 52–58 | 8 | 99 | 1 | – | 4.5 | yes(b) | 0.4 / 1 / 0 |
| paper_nested-add-mult-sub | 9 | 5 | – | 3 (3) | 62 | 1, 62–62 | 2 | 99 | 1 | – | 4.5 | yes(b) | 1.0 / 0 / 1 |
| paper_nested-mult-add-div | 12 | 3 | – | 99 (99) | – | 0 | 99 | 99 | 1 | – | 7 | yes(b) | 0.8 / 0 / 1 |
| paper_nested-sub-add-mult | 5 | 15 | – | 3 (3) | 62 | 1, 62–62 | 3 | 99 | 1 | – | 8 | yes(b) | 0.9 / 0 / 1 |
| paper_word-add-mult | 12 | fourteen | – | 1 (1) | 55 | 3, 55–57 | 99 | 7 | 1 | – | 2.5 | yes(b) | 0.8 / 0 / 1 |
| paper_word-mult-sub | 15 | seven | – | 5 (6) | 56 | 0 | 99 | 99 | 1 | – | 3.5 | yes(b) | 0.8 / 0 / 1 |
| paper_word-parens | 5 | twenty | – | 2 (2) | 53 | 4, 52–62 | 3 | 99 | 2 | – | 6 | yes(b) | 0.2 / 0 / 0 |
| paper_mod-add | 11 | 2 | 1 | 1 (4) | 51 | 3, 51–61 | 99 | 1 | 2 | 1 | 6 | no(b) | 0.0 / 1 / 0 |
| paper_mod-mult | 16 | 1 | 3 | 1 (3) | 55 | 5, 55–59 | 99 | 4 | 1 | 4 | 4 | no(b) | 0.3 / 1 / 0 |
| paper_square-sub | 3 | 9 | – | 1 (1) | 39 | 15, 37–62 | 1 | 5 | 1 | – | 6 | no(b) | 0.1 / 1 / 0 |
| paper_mult-mult-left | 6 | 24 | 12 | 3 (3) | 62 | 1, 62–62 | 4 | 99 | 1 | 99 | 6.5 | yes(b) | 1.0 / 0 / 1 |
| paper_div-div-left | 6 | 3 | 2 | 1 (1) | 60 | 3, 60–62 | 2 | 99 | 3 | 5 | 53.5 | no(b) | 0.1 / 1 / 0 |
| paper_mult-div-left | 24 | 8 | – | 7 (99) | 57 | 0 | 99 | 99 | 1 | – | 4 | yes(b) | 0.9 / 0 / 1 |
| paper_div-mult-left | 4 | 8 | 6 | 5 (5) | 62 | 0 | 5 | 99 | 1 | 4 | 7 | yes(b) | 0.8 / 1 / 0 |
| paper_mult-div-mult | 12 | 9 | – | 99 (99) | – | 0 | 99 | 99 | 1 | – | 7.5 | yes(b) | 0.7 / 1 / 0 |
| paper_add-add-add | 3 | 10 | – | 3 (3) | 47 | 1, 47–47 | 4 | 99 | 1 | – | 6 | yes(b) | 1.0 / 0 / 1 |
| paper_sub-sub-sub | 15 | 10 | – | 99 (99) | – | 0 | 99 | 99 | 2 | – | 7 | yes(b) | 0.9 / 0 / 1 |
| paper_add-sub-add-sub | 13 | 10 | – | 99 (99) | – | 0 | 99 | 99 | 1 | – | 6 | yes(b) | 0.8 / 0 / 1 |
| paper_chain-mult-mult-add | 6 | 26 | – | 4 (4) | 62 | 0 | 3 | 99 | 5 | – | 8 | yes(b) | 0.9 / 1 / 0 |
| paper_nested-add-add-mult | 9 | 18 | – | 6 (6) | 62 | 0 | 5 | 99 | 5 | – | 4.5 | yes(b) | 0.6 / 0 / 0 |
| paper_nested-mult-sub-add | 7 | 9 | – | 2 (2) | 59 | 4, 59–62 | 3 | 8 | 1 | – | 6.5 | yes(b) | 0.7 / 1 / 0 |
| paper_nested-div-add-sub | 8 | 5 | – | 5 (5) | 59 | 0 | 8 | 99 | 1 | – | 5 | yes(b) | 0.9 / 0 / 1 |
| paper_nested-sub-div-mult | 3 | 15 | – | 2 (2) | 62 | 1, 62–62 | 2 | 7 | 1 | – | 53.5 | yes(b) | 0.6 / 1 / 0 |
| paper_nested4-add-mult-add-div | 12 | 3 | – | 99 (99) | – | 0 | 99 | 99 | 1 | – | 6 | yes(b) | 0.3 / 0 / 0 |
| paper_nested4-sub-add-mult-sub | 12 | 10 | – | 3 (99) | 54 | 3, 54–56 | 99 | 99 | 1 | – | 3.5 | yes(b) | 0.8 / 1 / 0 |
| paper_redundant-parens-mult | 6 | 11 | 8 | 1 (1) | 56 | 5, 55–62 | 2 | 99 | 1 | 6 | 6 | yes(b) | 0.9 / 1 / 0 |
| paper_redundant-parens-div | 3 | 2 | – | 1 (1) | 55 | 12, 39–62 | 1 | 2 | 1 | – | 6.5 | no(b) | 0.4 / 1 / 0 |
| paper_word-sub-mult | 6 | four | – | 1 (1) | 56 | 5, 50–56 | 6 | 8 | 1 | – | 3 | yes(b) | – |
| paper_word-add-add | 7 | twelve | – | 3 (3) | 53 | 2, 53–54 | 99 | 99 | 1 | – | 2 | yes(b) | – |
| paper_word-div-sub | 5 | three | – | 1 (1) | 54 | 6, 54–62 | 3 | 1 | 1 | – | 6 | yes(b) | – |
| paper_mixed-mult-add | 12 | 14 | – | 1 (2) | 55 | 3, 55–57 | 99 | 99 | 1 | – | 6 | yes(b) | – |
| paper_mixed-parens-mult | 5 | 15 | – | 3 (3) | 62 | 1, 62–62 | 2 | 99 | 1 | – | 5 | yes(b) | – |
| paper_mod-sub | 9 | 1 | 3 | 1 (1) | 51 | 12, 51–62 | 1 | 1 | 4 | 1 | 4 | no(b) | – |
| paper_square-div | 3 | 9 | – | 1 (1) | 55 | 11, 39–62 | 3 | 5 | 1 | – | 5 | no(b) | – |
| paper_square-mult | 4 | 16 | – | 1 (1) | 55 | 11, 41–62 | 2 | 99 | 1 | – | 6 | yes(b) | – |
| paper_pystar-add | 3 | 9 | – | 1 (1) | 47 | 10, 40–62 | 2 | 1 | 1 | – | 53.5 | yes(b) | – |
| paper_pystar-sub | 4 | 16 | – | 1 (1) | 52 | 9, 40–62 | 2 | 3 | 1 | – | 52 | yes(b) | – |
| paper_floordiv-add | 11 | 3 | – | 2 (4) | 51 | 1, 51–51 | 99 | 99 | 1 | – | 4.5 | yes(b) | – |
| paper_floordiv-mult | 10 | 3 | – | 99 (99) | – | 0 | 99 | 99 | 1 | – | 7 | yes(b) | – |
| new_new01 | 12 | 22 | 16 | 1 (3) | 54 | 5, 54–58 | 99 | 99 | 2 | 2 | 2.5 | yes(b) | – |
| new_new02 | 4 | 8 | 20 | 3 (3) | 62 | 1, 62–62 | 4 | 99 | 1 | 99 | 52.5 | yes(b) | – |
| new_new03 | 6 | 14 | 10 | 1 (1) | 55 | 7, 53–62 | 2 | 5 | 2 | 4 | 4 | yes(b) | – |
| new_new04 | 6 | 42 | 35 | 1 (1) | 52 | 9, 51–62 | 3 | 99 | 99 | 99 | 6.5 | yes(b) | – |
| new_new05 | 6 | 13 | 8 | 2 (2) | 55 | 1, 55–55 | 4 | 99 | 5 | 5 | 52.5 | yes(b) | – |
| new_new06 | 8 | 4 | 1 | 1 (1) | 53 | 11, 52–62 | 2 | 8 | 1 | 3 | 4 | yes(b) | – |
| new_new07 | 2 | 11 | 10 | 6 (6) | 62 | 0 | 2 | 99 | 1 | 2 | 7.5 | yes(b) | – |
| new_new08 | 14 | 42 | 21 | 99 (99) | – | 0 | 99 | 99 | 99 | 99 | 5 | yes(b) | – |
| new_new09 | 8 | 11 | 5 | 1 (1) | 55 | 7, 53–62 | 2 | 8 | 1 | 4 | 3 | yes(b) | – |
| new_new10 | 14 | 23 | 16 | 1 (8) | 57 | 2, 54–57 | 99 | 99 | 2 | 3 | 3 | yes(b) | – |
| new_new11 | 6 | 17 | 28 | 1 (1) | 52 | 8, 51–58 | 7 | 99 | 3 | 99 | 5 | yes(b) | – |
| new_new12 | 6 | 4 | 14 | 1 (1) | 55 | 5, 53–57 | 99 | 99 | 1 | 99 | 4 | yes(b) | – |
| new_new13 | 2 | 14 | 49 | 2 (2) | 52 | 4, 52–62 | 2 | 99 | 1 | 99 | 7 | yes(b) | – |
| new_new14 | 8 | 13 | 7 | 1 (1) | 59 | 8, 55–62 | 2 | 99 | 1 | 3 | 5.5 | yes(b) | – |
| new_new15 | 36 | 41 | 17 | 1 (99) | 57 | 3, 56–58 | 99 | 99 | 99 | 99 | 3.5 | yes(b) | – |
| new_new16 | 3 | 7 | 11 | 2 (2) | 59 | 6, 56–62 | 2 | 99 | 1 | 99 | 6 | yes(b) | – |
| new_new17 | 18 | 90 | 40 | 99 (99) | – | 0 | 99 | 99 | 2 | 99 | 3 | yes(b) | – |
| new_new18 | 12 | 24 | 20 | 1 (3) | 55 | 3, 53–55 | 99 | 99 | 3 | 1 | 3.5 | yes(b) | – |
| new_new19 | 16 | 27 | 60 | 1 (2) | 55 | 4, 55–58 | 99 | 99 | 2 | 99 | 3 | yes(b) | – |
| new_new20 | 13 | 65 | 45 | 1 (2) | 52 | 5, 52–56 | 99 | 99 | 99 | 99 | 52 | yes(b) | – |
| new_new21 | 9 | 54 | 6 | 2 (2) | 51 | 3, 51–62 | 2 | 99 | 99 | 2 | 53 | yes(b) | – |
| new_new22 | 14 | 42 | 24 | 99 (99) | – | 0 | 99 | 99 | 99 | 99 | 3.5 | yes(b) | – |
| new_new23 | 6 | 18 | 15 | 1 (1) | 53 | 6, 39–62 | 3 | 99 | 2 | 99 | 8 | yes(b) | – |
| new_new24 | 1 | 2 | 14 | 2 (2) | 62 | 1, 62–62 | 2 | 99 | 1 | 99 | 7.5 | yes(b) | – |
| new_new25 | 21 | 168 | 96 | 7 (99) | 55 | 0 | 99 | 99 | 99 | 99 | 5 | yes(b) | – |
| new_new26 | 72 | 64 | 4 | 1 (99) | 55 | 5, 54–58 | 99 | 99 | 99 | 3 | 2 | yes(b) | – |
| new_new27 | 15 | 90 | 36 | 1 (99) | 55 | 2, 54–55 | 99 | 99 | 1 | 99 | 3 | yes(b) | – |
| new_new28 | 28 | 33 | 12 | 2 (99) | 58 | 1, 58–58 | 99 | 99 | 99 | 99 | 2.5 | yes(b) | – |
| new_new29 | 21 | 84 | 48 | 99 (99) | – | 0 | 99 | 99 | 99 | 99 | 5 | yes(b) | – |
| new_new30 | 30 | 29 | 4 | 1 (1) | 56 | 4, 55–58 | 99 | 99 | 1 | 3 | 3 | yes(b) | – |
| new_new31 | 24 | 29 | 64 | 5 (99) | 56 | 0 | 99 | 99 | 3 | 99 | 1.5 | yes(b) | – |
| new_new32 | 6 | 48 | 16 | 2 (2) | 51 | 10, 51–62 | 2 | 99 | 99 | 99 | 52 | yes(b) | – |
| new_new33 | 9 | 3 | 2 | 1 (1) | 51 | 4, 51–62 | 2 | 3 | 1 | 1 | 4 | yes(b) | – |
| new_new34 | 40 | 51 | 128 | 1 (2) | 55 | 5, 54–58 | 99 | 99 | 99 | 99 | 4 | yes(b) | – |
| new_new35 | 17 | 10 | 2 | 99 (99) | – | 0 | 99 | 99 | 1 | 4 | 7 | yes(b) | – |
| new_new36 | 4 | 28 | 49 | 2 (2) | 62 | 3, 53–62 | 2 | 4 | 2 | 99 | 7 | yes(b) | – |
| new_new37 | 6 | 18 | 15 | 4 (4) | 61 | 0 | 4 | 99 | 5 | 99 | 6.5 | yes(b) | – |
| new_new38 | 30 | 37 | 100 | 1 (1) | 55 | 5, 55–59 | 99 | 99 | 99 | 99 | 2.5 | yes(b) | – |
| new_new39 | 19 | 10 | 1 | 99 (99) | – | 0 | 99 | 99 | 3 | 1 | 8 | yes(b) | – |
| new_new40 | 18 | 15 | 9 | 99 (99) | – | 0 | 99 | 99 | 2 | 2 | 51.5 | yes(b) | – |
| new_new41 | 22 | 132 | 60 | 1 (99) | 56 | 2, 55–56 | 99 | 99 | 99 | 4 | 5 | yes(b) | – |
| new_new42 | 4 | 36 | 72 | 2 (2) | 61 | 5, 55–62 | 2 | 99 | 2 | 99 | 7 | yes(b) | – |
| new_new43 | 14 | 12 | 3 | 99 (99) | – | 0 | 99 | 99 | 1 | 6 | 3 | yes(b) | – |
| new_new44 | 5 | 40 | 16 | 1 (1) | 53 | 6, 53–62 | 2 | 99 | 1 | 99 | 6.5 | yes(b) | – |
| new_new45 | 7 | 16 | 14 | 1 (1) | 59 | 5, 57–62 | 3 | 99 | 99 | 99 | 53 | yes(b) | – |
| new_new46 | 30 | 40 | 60 | 1 (1) | 56 | 6, 52–57 | 99 | 99 | 2 | 99 | 5 | yes(b) | – |
| new_new47 | 14 | 11 | 8 | 99 (99) | – | 0 | 99 | 99 | 1 | 5 | 6 | yes(b) | – |
| new_new48 | 6 | 3 | 1 | 2 (2) | 62 | 2, 55–62 | 2 | 99 | 1 | 3 | 5.5 | yes(b) | – |
| new_new49 | 4 | 3 | 1 | 2 (2) | 46 | 7, 46–62 | 2 | 99 | 1 | 3 | 7.5 | yes(b) | – |
| new_new50 | 4 | 2 | 8 | 4 (4) | 58 | 0 | 5 | 99 | 1 | 8 | 8 | yes(b) | – |

