# Addendum 4 — 2026-09-04 14:14 MDT, written before any headline v2 number was scored

Observed in the first 14 leak-gate items (arm C): greedy writes the answer, not the intermediate, in 14/14 (13/14 correct), but at
temperature 0.8 the intermediate appears as the first written number in ≥ 1 of 10 samples for 10/14 items ("15 - 3 = 12 …"). So
the intermediate is not the greedy next token, yet it carries next-token probability as the opening of narrated work. A mid-layer
lens hit could therefore reflect that next-token disposition rather than a held value — the objection behind workspace-bench's
redesign.

**Next-token control, reported beside every headline:** the 64th layer of the API response (and layer 63 on the box) is the
model's own logits with no Jacobian. For each item, record the intermediate's rank there (`rank_L63`) and the decoy's. Report:
(a) the fraction of rank-1 items whose intermediate is ALSO in the model's top-8 next tokens at the readout position;
(b) the "workspace-only" subset: items where the intermediate reaches rank ≤ 3 at some source layer 0–62 but is absent from
the model's top-8 at layer 63 — hits the logit lens would not show; (c) the same for the decoy. The prereg's H1–H7 are scored as
frozen; this control is a qualifier on how they may be interpreted, not a new hypothesis. No change to the frozen files.
