#!/usr/bin/env python3
"""Fit a Jacobian lens on Qwen3.6-27B by "Neel's recipe" (PREREG_orderops_v2 arm C2).

Recipe (prereg + the parameters Neuronpedia recorded for the externally fitted
NeelNanda/pile-10k lenses, e.g. deepseek-v4-flash/jlens/NeelNanda-pile-10k/config.yaml:
n_prompts 25, max_seq_len 128, skip_first 4, target = penultimate, estimator "standard",
weighting "uniform", capture_point block_output):

  * 25 prompts from NeelNanda/pile-10k, deterministic: the first 25 documents (row order)
    that tokenize (add_special_tokens=False, no BOS -- the Qwen tokenizer has none) to
    >= 132 tokens, each truncated to its first 128 tokens.
  * skip the first 4 source positions (upstream default is 16); the final position is
    dropped by upstream's valid_position_mask as always.
  * Jacobians from every layer's residual (block output = the output of model.layers[l],
    upstream's ActivationRecorder hook point) to the residual at the TARGET layer.
  * target_layer = -2  ->  upstream jlens.fitting._check_layer_indices resolves it to
    n_layers - 2 = 62 for this 64-block model: the OUTPUT of block index 62, i.e. the
    penultimate block's output residual (HF hidden_states index 63). Upstream's default
    target is n_layers - 1 = 63 (the final block's output, which after final norm +
    unembed IS the model's logits); "penultimate" is one block earlier, and
    jacobian_for_prompt's docstring names it as the better-conditioned choice. The same
    convention is used by agu18dec/qwen3.6-27b-pile-jacobians (target_block 62 =
    "penultimate", 63 = "final", n25/skip4 on this exact model). Upstream fits source
    layers 0..61 (it refuses source >= target); J_62 = I by definition and layer 63 is
    the model's own output -- read_items.py treats both that way.
  * plain autograd through the realized compute graph (no fused kernels; the bootstrap
    asserts fla / causal_conv1d are absent), bf16 weights, fp32 accumulation (upstream).
  * dim_batch 8, falling back to 4 on CUDA OOM (the per-prompt checkpoint makes the
    fallback resume where it stopped). Uniform weighting over prompts/positions and the
    standard estimator are exactly what upstream fit() does (running mean of per-prompt
    Jacobians; cotangents summed over later targets, mean over source positions).
  * output: <out>/Qwen3.6-27B_jacobian_lens_neel25.pt written by upstream
    JacobianLens.save: {"J": {layer: fp16 [d,d]}, "n_prompts", "source_layers", "d_model"}
    -- the SAME dict layout the hosted file has (box introspection 2026-09-04: J = 63
    fp16 5120x5120 matrices for layers 0..62, n_prompts 1000, source_layers [0..62],
    d_model 5120), so read_items.py reads both lenses through one loader. Ours carries
    layers 0..61 (target 62). Orientation, from upstream jlens.fitting: row i of J_l is
    d h_target[i] / d h_l, so the transport is J @ h == h @ J.T (JacobianLens.transport
    does `residual @ J_bar.T`); the final RMSNorm is applied AFTER the transport, inside
    unembed (lm_head(final_norm(x))). + provenance.json.

Dense model: no routing log.  --dry-run selects the prompts with the tokenizer only
(works offline from the HF cache) and prints doc ids / token counts, no model load.

Box paths (verbatim): python /venv/main/bin/python, model /workspace/models/qwen36-27b,
out /workspace/out/lens_neel, logs /workspace/logs.  Run:
  /venv/main/bin/python /workspace/orderops/box/fit_neel_recipe.py 2>&1 | tee /workspace/logs/fit_neel.log
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

BANNED = ("fla", "flash_linear_attention", "causal_conv1d")
DEFAULT_MODEL = "/workspace/models/qwen36-27b"
DEFAULT_OUT = "/workspace/out/lens_neel"
DEFAULT_TOKENIZER = "Qwen/Qwen3.6-27B"
DATASET = "NeelNanda/pile-10k"
LENS_BASENAME = "Qwen3.6-27B_jacobian_lens_neel25.pt"


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(*a):
    print(f"[{ts()}]", *a, flush=True)


def check_env():
    import importlib
    for mod in BANNED:
        try:
            importlib.import_module(mod)
        except ImportError:
            continue
        sys.exit(f"FATAL: {mod} is importable; the fused kernels must be absent "
                 "(workspace rule: pure-PyTorch DeltaNet path, plain autograd)")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- corpus
def pile10k_parquet() -> tuple[str, str]:
    """Path to the pile-10k train parquet + the dataset snapshot revision.

    Uses snapshot_download(repo_type="dataset"), which resolves from the local HF cache
    when HF_HUB_OFFLINE=1 (the dry run) and downloads otherwise (the box)."""
    from huggingface_hub import snapshot_download
    d = snapshot_download(DATASET, repo_type="dataset", allow_patterns=["data/*.parquet"])
    files = sorted(glob.glob(os.path.join(d, "data", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet under {d}/data")
    rev = os.path.basename(os.path.realpath(d))
    return files[0], rev


def load_pile10k():
    """(texts, metas, info). The datasets library first (the box: `load_dataset('NeelNanda/pile-10k', split='train')`),
    else the dataset's single train parquet from the HF hub cache (offline dry run). Same shard, same row order."""
    try:
        from datasets import load_dataset
        ds = load_dataset(DATASET, split="train")
        info = dict(loader="datasets.load_dataset", split="train", fingerprint=getattr(ds, "_fingerprint", None), n_rows=len(ds))
        return list(ds["text"]), list(ds["meta"]), info
    except Exception as e:                      # offline: hub-cache parquet
        import pyarrow.parquet as pq
        path, rev = pile10k_parquet()
        table = pq.read_table(path, columns=["text", "meta"])
        info = dict(loader="pyarrow(hub cache parquet)", fallback_reason=f"{type(e).__name__}: {str(e)[:160]}", parquet=path,
                    parquet_sha256=sha256_file(path), dataset_revision=rev, n_rows=table.num_rows)
        return table.column("text").to_pylist(), table.column("meta").to_pylist(), info


def select_prompts(tok, n_prompts: int, max_seq_len: int, min_doc_tokens: int):
    """First n_prompts docs (row order) with >= min_doc_tokens tokens, truncated to max_seq_len ids."""
    texts, metas, corpus = load_pile10k()
    picked = []
    for i, text in enumerate(texts):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= min_doc_tokens:
            picked.append(dict(doc_index=i, pile_set_name=(metas[i] or {}).get("pile_set_name"),
                               n_tokens_full=len(ids), input_ids=ids[:max_seq_len],
                               text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest()))
            if len(picked) == n_prompts:
                break
    if len(picked) < n_prompts:
        raise RuntimeError(f"only {len(picked)} docs with >= {min_doc_tokens} tokens")
    corpus["docs_scanned"] = picked[-1]["doc_index"] + 1
    return picked, corpus


# --------------------------------------------------------------------------- model loading + capture check
def load_text_model(path: str, dtype):
    """AutoModelForCausalLM resolves the text part of this Qwen3_5ForConditionalGeneration checkpoint (as on the
    Aug-21 box, transformers 5.5); if a newer transformers refuses, load the wrapper class itself -- jlens.from_hf
    finds the decoder at model.language_model either way."""
    from transformers import AutoModelForCausalLM
    try:
        hf = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map="cuda")
    except Exception as e:
        log(f"AutoModelForCausalLM failed ({type(e).__name__}: {str(e)[:200]}); loading Qwen3_5ForConditionalGeneration")
        from transformers import Qwen3_5ForConditionalGeneration
        hf = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=dtype, device_map="cuda")
    hf.eval()
    return hf


def verify_capture(lm, tok) -> dict:
    """Coordinator check: output_hidden_states must give n_layers+1 = 65 entries (embeddings + 64 blocks), and
    hidden_states[l+1] must equal the hooked output of block l (the capture point used everywhere here). The final
    entry is post-final-norm in transformers 5.x and is expected to DIFFER from the hooked final-block output."""
    import torch
    from jlens.hooks import ActivationRecorder
    ids = torch.tensor([tok("The quick brown fox jumps over the lazy dog.", add_special_tokens=False)["input_ids"]],
                       device=lm.input_device)
    with torch.no_grad(), ActivationRecorder(lm.layers, at=list(range(lm.n_layers))) as rec:
        out = lm._text_module(input_ids=ids, use_cache=False, output_hidden_states=True)
    hs = out.hidden_states
    assert len(hs) == lm.n_layers + 1, f"expected {lm.n_layers + 1} hidden_states, got {len(hs)}"
    mism = [l for l in (0, lm.n_layers // 2, lm.n_layers - 2) if not torch.equal(hs[l + 1], rec.activations[l])]
    final_equal = bool(torch.equal(hs[-1], rec.activations[lm.n_layers - 1]))
    res = dict(n_hidden_states=len(hs), hidden_states_lplus1_equals_block_l_output=(not mism), mismatched_layers=mism,
               final_entry_equals_hooked_final_block=final_equal)
    log(f"capture check: {len(hs)} hidden_states; hidden_states[l+1]==block-l output {'OK' if not mism else 'MISMATCH at ' + str(mism)}; "
        f"final entry {'EQUALS' if final_equal else 'differs from'} hooked block-{lm.n_layers - 1} output (post-norm: differs expected)")
    if mism:
        log("WARNING: hidden_states indexing differs from the hook capture; the pipeline uses the hooks (upstream convention) regardless")
    return res


# --------------------------------------------------------------------------- model wrapper
def make_lens_model(hf_model, tok, prompt_ids: dict[str, list[int]]):
    """HFLensModel whose encode() returns pre-tokenized ids, so the 128-token truncation is
    done on ids (re-tokenizing decoded text could shift BPE boundaries)."""
    from jlens.hf import HFLensModel
    import torch

    class PreTokenized(HFLensModel):
        def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
            ids = prompt_ids[text][:max_length]
            return torch.tensor([ids], dtype=torch.long, device=self.input_device)

    return PreTokenized(hf_model, tok, compile=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="local snapshot dir (box) or HF id")
    ap.add_argument("--tokenizer", default=None, help="tokenizer id/path (default: --model; dry-run default Qwen/Qwen3.6-27B)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-prompts", type=int, default=25)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--min-doc-tokens", type=int, default=132)
    ap.add_argument("--skip-first", type=int, default=4)
    ap.add_argument("--target-layer", type=int, default=-2, help="-2 -> block 62 of 64 (penultimate block output)")
    ap.add_argument("--dim-batch", type=int, default=8)
    ap.add_argument("--dim-batch-fallback", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--compile", action="store_true", help="per-block torch.compile (upstream fast path; off by default on the hybrid stack)")
    ap.add_argument("--dry-run", action="store_true", help="select prompts with the tokenizer only; no model load")
    args = ap.parse_args()

    check_env()
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()
    from transformers import AutoTokenizer
    tok_src = args.tokenizer or (DEFAULT_TOKENIZER if args.dry_run and not os.path.isdir(args.model) else args.model)
    tok = AutoTokenizer.from_pretrained(tok_src)
    assert tok.bos_token_id is None, "expected a BOS-less Qwen tokenizer; prompts are fitted without any prepended token"
    log(f"tokenizer {tok_src}: len {len(tok)}, bos {tok.bos_token_id}, eos {tok.eos_token_id}")

    log(f"selecting {args.n_prompts} prompts from {DATASET} (>= {args.min_doc_tokens} tokens, truncated to {args.max_seq_len}) …")
    picked, corpus = select_prompts(tok, args.n_prompts, args.max_seq_len, args.min_doc_tokens)
    for p in picked:
        head = tok.decode(p["input_ids"][:12]).replace("\n", "\\n")
        log(f"  doc {p['doc_index']:5d} {str(p['pile_set_name']):14s} full={p['n_tokens_full']:6d} used={len(p['input_ids'])}  {head!r}")
    sel_path = os.path.join(args.out, "prompts_selected.json")
    with open(sel_path, "w") as f:
        json.dump(dict(dataset=DATASET, corpus=corpus, rule=f"first {args.n_prompts} docs in row order with >= {args.min_doc_tokens} "
                       f"tokens (add_special_tokens=False, no BOS), truncated to {args.max_seq_len} ids", prompts=picked), f, indent=1)
    log(f"wrote {sel_path}  (docs scanned {corpus['docs_scanned']}; corpus loader {corpus['loader']})")
    if args.dry_run:
        log("DRY RUN: prompt selection only; doc ids:", [p["doc_index"] for p in picked])
        return

    import torch
    import jlens
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    logging.getLogger("jlens.fitting").setLevel(logging.DEBUG)   # per-100-pass lines: a wedge inside a prompt is visible

    log(f"loading {args.model} in {args.dtype} on cuda …")
    hf = load_text_model(args.model, getattr(torch, args.dtype))
    prompt_ids = {f"pile10k:{p['doc_index']}": p["input_ids"] for p in picked}
    lm = make_lens_model(hf, tok, prompt_ids)
    capture_check = verify_capture(lm, tok)
    if args.compile:
        for i in range(len(lm.layers)):
            lm.layers[i] = torch.compile(lm.layers[i], mode="default", dynamic=False)
    n_layers = lm.n_layers
    target = args.target_layer + n_layers if args.target_layer < 0 else args.target_layer
    log(f"wrapped {lm}; target_layer {args.target_layer} -> block {target} of {n_layers} (source layers 0..{target-1}); "
        f"GPU mem after load {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    prompts = list(prompt_ids)
    ckpt = os.path.join(args.out, "fit_ckpt.pt")
    per_prompt = []

    class Timed:
        """Sequence wrapper: timestamps each prompt start so a wedge is visible in the log."""
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __iter__(self):
            for i, p in enumerate(self.items):
                t0 = time.time()
                log(f"prompt {i+1}/{len(self.items)} start {p} ({len(prompt_ids[p])} ids) "
                    f"GPU {torch.cuda.memory_allocated()/2**30:.1f} GiB alloc / {torch.cuda.max_memory_allocated()/2**30:.1f} peak")
                yield p
                per_prompt.append(dict(prompt=p, elapsed_s=round(time.time() - t0, 1)))
                log(f"prompt {i+1}/{len(self.items)} done in {per_prompt[-1]['elapsed_s']}s")

    def run(dim_batch):
        return jlens.fit(lm, Timed(prompts), target_layer=args.target_layer, dim_batch=dim_batch,
                         max_seq_len=args.max_seq_len, skip_first=args.skip_first,
                         checkpoint_path=ckpt, checkpoint_every=1, resume=True)

    dim_batch_used, fallback_events = args.dim_batch, []
    try:
        lens = run(args.dim_batch)
    except torch.cuda.OutOfMemoryError as e:
        fallback_events.append(dict(at=ts(), dim_batch=args.dim_batch, error=str(e)[:300]))
        log(f"CUDA OOM at dim_batch {args.dim_batch}; falling back to {args.dim_batch_fallback} (resumes from checkpoint)")
        torch.cuda.empty_cache()
        dim_batch_used = args.dim_batch_fallback
        lens = run(args.dim_batch_fallback)

    out_pt = os.path.join(args.out, LENS_BASENAME)
    lens.save(out_pt)          # upstream format: {"J": {layer: fp16 [d,d]}, "n_prompts", "source_layers", "d_model"}
    elapsed = round(time.time() - t_start)
    jl_dir = os.path.dirname(os.path.dirname(os.path.abspath(jlens.__file__)))
    commit = subprocess.run(["git", "-C", jl_dir, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    import transformers
    prov = dict(
        kind="jacobian_lens_fit", recipe="neel_pile10k_n25_skip4_penultimate", when_utc=ts(),
        model=args.model, model_config_sha256=sha256_file(os.path.join(args.model, "config.json")) if os.path.isdir(args.model) else None,
        model_index_sha256=sha256_file(os.path.join(args.model, "model.safetensors.index.json")) if os.path.isfile(os.path.join(args.model, "model.safetensors.index.json")) else None,
        dtype=args.dtype, n_layers=n_layers, d_model=lm.d_model, target_layer_arg=args.target_layer, target_layer_resolved=target,
        source_layers=lens.source_layers, n_prompts_fitted=lens.n_prompts, skip_first=args.skip_first, max_seq_len=args.max_seq_len,
        min_doc_tokens=args.min_doc_tokens, dim_batch_requested=args.dim_batch, dim_batch_used=dim_batch_used, fallback_events=fallback_events,
        estimator="upstream jlens.fitting.jacobian_for_prompt: one-hot cotangents at every valid target position, gradient summed over later targets, mean over valid source positions; running mean over prompts (uniform weighting)",
        capture_point="block_output (forward hook on model.layers[l]; residual BEFORE final norm)", capture_check=capture_check,
        pt_layout="upstream JacobianLens.save: {'J': {layer: fp16 [d,d]}, 'n_prompts', 'source_layers', 'd_model'}; transport = h @ J.T",
        dataset=DATASET, corpus=corpus, prompts=[dict(doc_index=p["doc_index"], pile_set_name=p["pile_set_name"], n_tokens_full=p["n_tokens_full"],
                                                     n_tokens_used=len(p["input_ids"]), text_sha256=p["text_sha256"]) for p in picked],
        per_prompt_seconds=per_prompt, elapsed_s=elapsed, jlens_commit=commit, torch=torch.__version__, transformers=transformers.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), peak_gpu_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        lens_file=out_pt, lens_sha256=sha256_file(out_pt), lens_bytes=os.path.getsize(out_pt), compile=args.compile,
    )
    with open(os.path.join(args.out, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    log(f"DONE -> {out_pt} ({prov['lens_bytes']} bytes, sha256 {prov['lens_sha256'][:16]}…) in {elapsed}s; provenance.json written")


if __name__ == "__main__":
    main()
