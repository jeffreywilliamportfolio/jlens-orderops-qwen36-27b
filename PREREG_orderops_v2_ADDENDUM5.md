# Addendum 5 — 2026-09-04 15:15 MDT, written before any box result was read

The Neel-recipe fit (arm C2) runs at ~25 min per prompt on this box: 48 of the 64 layers are gated DeltaNet on the pure-PyTorch
path (fused kernels are disallowed in this workspace), which limits the fit to dim_batch 4 (1,280 backward passes per prompt) and
makes each pass ~1 s. A 25-prompt fit would end after midnight. Jeffrey's call (15:13 MDT): time matters more than money; stop
at 10 prompts for an initial reading.

**Change:** C2 is fitted on the first **10** of the 25 preselected pile-10k documents (same order, same 128-token truncation,
same skip_first 4, same target block 62), finalized from the per-prompt checkpoint. Everything else in arm C is unchanged.

**Interpretation rule (stated now):** a 10-prompt lens is weaker than Neel's 25-prompt recipe. If the effect survives it, H4 is
answered in the direction "lens quality does not explain the null" with more force than the frozen rule required. If the effect
fails under it, H4 is **inconclusive**, not negative, and the note says "not tested at n=25". The frozen H4 rule (ratio ≥ 0.5 of the
hosted lens's pass@1) is applied to the 10-prompt lens and reported as such; no number is presented as an n=25 result.
