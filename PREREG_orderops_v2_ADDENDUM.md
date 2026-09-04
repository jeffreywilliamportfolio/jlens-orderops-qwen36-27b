# Addendum 1 — 2026-09-04 13:27 MDT, written before any v2 result was scored

Observed during the first minutes of arm B (chat-endpoint correctness, numCompletionTokens 8, as frozen): the model opens with
prose ("Let's break down the calculation step by") and the answer does not arrive within 8 tokens, so arm B as frozen will mostly
return "undetermined". Fix, without touching the frozen files: a supplementary arm **B2**, identical except numCompletionTokens 48.
Correctness rule for B2: the completion is correct if the last integer it contains equals the target, or the first integer
following an "=" in the completion equals the target; both readings are stored. Arm B (8 tokens) is still recorded and reported.
The box arm's greedy continuation of the raw prompt remains the primary correctness measure. No other change.
