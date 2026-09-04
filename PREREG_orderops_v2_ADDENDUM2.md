# Addendum 2 — 2026-09-04 13:40 MDT, written before any v2 result was scored

Source: `camilablank/workspace-bench` (Neel Nanda's scholars; pushed 2026-08-14..25), family `arithmetic_intermediates`
(renamed from `order_ops`). Its README states the objection our design must answer: "a bare `Compute (7*3)+8 =` makes the model
*write* `21`, which would prove nothing." Their admissibility gate: 10 sampled rollouts (temp 0.8, top_p 0.95, max_new 32);
pass = ≥ 8/10 assert the answer AND 0 rollouts emit any scored intermediate. Greedy is recorded, never the gate.

**Leak gate (box arm, both prompt variants), adopted as stated:** for every item, 10 sampled continuations of the raw prompt
(no chat template; temp 0.8, top_p 0.95, max_new 32, seeds 0–9) plus the greedy continuation. An item is **admissible** iff
≥ 8/10 sampled continuations assert the target (first number written equals the target) AND no continuation (sampled or greedy)
contains the intermediate as a number before the target. Items whose greedy continuation writes the intermediate first (e.g.
"5 * 4 = 20") are reported as **leaked**: for them the "intermediate" is the next token and any lens hit is next-token
prediction, not workspace content.

**Reporting rule:** every H1–H7 number is reported (a) as frozen, on all items, and (b) restricted to admissible items. The
(b) numbers are the headline. If fewer than 30 of the 105 items are admissible, H1/H2 are "not testable as designed" and the
note says so. The API arms cannot run the gate (no raw-prompt generation), so their headline is deferred to (b) from the box.

**Also recorded from workspace-bench, for the note's discrepancy section:** their August runs read the same hosted
`neuronpedia/jacobian-lens qwen3.6-27b n1000 wikitext` lens (results_2026-08-13.json), their item schema carries a
`single_token_reachable` field and names `十八`/`eighteen` as one-token routes, and their scored intermediates are mostly
2–3-digit values with no single-token form, read by a generative "oracle lens" against J-lens token bags with a
`net = value − cross` metric. That design does not overlap ours; it supersedes the July 10.2-style attempt they reported as
failed. No change to the frozen files.
