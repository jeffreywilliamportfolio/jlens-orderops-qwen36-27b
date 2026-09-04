# Addendum 3 — 2026-09-04 13:52 MDT, written before any v2 result was scored

Discovered on a throwaway probe (`(1 + 1) * 2 = `, not an item): Neuronpedia's lens endpoint accepts a raw `prompt` field
(no chat template) and generates from it; the greedy continuation was "4\n(1 + " — the answer, not the intermediate.

**Arm C (API leak gate and raw-prompt correctness), supersedes arms B/B2 as the correctness measure on the API side:**
for every item, both variants: one greedy continuation of the raw prompt (12 new tokens) → `correct_greedy` (first number
written equals the target) and `greedy_writes_intermediate_first`; and, for the with-space variant only, 10 sampled
continuations (temperature 0.8, 32 new tokens; the API is unseeded, so these are not replayable) → `assert_rate` and
`leak_any` exactly as defined in addendum 2. `admissible` = assert_rate ≥ 0.8 and not leak_any. Arms B and B2 remain
recorded as the template-mediated secondary. The box arm repeats the gate with seeds and top_p 0.95 as specified.
Runner: `run_v2_apiC.py` (hashed below). No change to the frozen files.
