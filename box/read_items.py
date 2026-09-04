#!/usr/bin/env python3
"""Arm C of PREREG_orderops_v2 (+ Addendum 2 leak gate) on the box: two lenses, one forward pass per prompt.

For every item in ../order_ops.json (55, paper) and ../items_new50.json (50, held-out), and both
prompt variants (`prompt` with trailing space; `prompt_nospace` = the same without it, exactly as
run_v2_api.py sent to Neuronpedia), this script

  1. tokenizes WITHOUT chat template: tok(prompt, add_special_tokens=False) -- byte-identical to
     the ids run_v2_api.py sent as inputTokenIds (cross-checked against ../v2_api/raw_A/*.json
     when present).  BOS: Neuronpedia's `prependBos: true` is implemented as
        bos = tokenizer.bos_token; if prepend_bos and bos: text = bos + text
     (neuronpedia_inference/endpoints/lens/prompt.py::_encode_raw_text) and the inputTokenIds
     path returns the ids untouched; the Qwen3.6-27B tokenizer has bos_token = None, so NOTHING
     was prepended -- confirmed by ../raw/00_parens-add-mult.json token 0 = '(' id 7 with
     prompt_len 11 = the 11 raw ids.  Default --bos none reproduces that; --bos endoftext
     prepends <|endoftext|> (248044) as a sensitivity variant only (not the prereg pipeline).
  2. runs ONE forward pass (bf16, no grad), hooking the output of every decoder block
     (jlens ActivationRecorder = upstream's capture point, residual BEFORE the final norm;
     HF output_hidden_states[-1] is post-norm and is NOT used), keeps the residual at every
     layer and position,
  3. applies BOTH lenses -- C1 hosted n1000 wikitext .pt, C2 the Neel-recipe .pt fitted by
     fit_neel_recipe.py -- to the same residuals through ONE reader (load_lens): both files are
     the upstream JacobianLens.save dict {"J": {layer: fp16 [5120,5120]}, "n_prompts",
     "source_layers", "d_model"} (hosted: layers 0..62, n_prompts 1000 -- box introspection
     2026-09-04; ours: layers 0..61). Readout per layer: logits_l = lm_head(final_norm(J_l @ h_l))
     -- orientation from upstream jlens.fitting (row i of J_l = d h_target[i] / d h_l, so
     J @ h == h @ J.T, which is what upstream JacobianLens.transport computes as
     `residual @ J_bar.T`; NOT h @ J), final RMSNorm AFTER the transport, inside unembed
     (upstream HFLensModel.unembed). Layers the lens does not carry (the lens's target layer,
     and layer 63 = the model's own output) get unembed(h_l) with no Jacobian -- exactly how
     Neuronpedia serves layer 63 in its 64-entry response.
     Records per layer at the LAST prompt token: the exact 1-based rank over the full logit
     vector (248320 = lm_head width, the API's reported vocab_size) of every single-token
     synonym of the intermediate, the target, the decoy and the operation, of the digit tokens
     0-9 (uninvolved-digit baseline), the top-64 ids/probs/strings; and at EVERY prompt
     position the top-8 (API parity, position controls).
  4. greedy continuation of the raw prompt (<= 6 new tokens, no template; arm B on the box),
  5. Addendum 2 leak gate: 10 sampled continuations (temperature 0.8, top_p 0.95, 32 new
     tokens, seeds 0..9 -- one CUDA generator per row, so each sample is seed-reproducible
     while the 10 rows are decoded as one batch), assert_rate, leak_any,
     greedy_writes_intermediate_first, admissible.

Synonym sets are NOT re-typed: W / C / OPS / forms() / syn() are extracted from
../score_orderops.py by AST and executed as-is (the file's sha256 goes in the manifest).
Token ids per synonym string s: tokenize s and " "+s, keep single-token results (rules
"tok"/"tok_space"); PLUS every vocab id whose decode().strip().lower() == s (rule "vocab",
the scorer's actual matching rule -- catches capitalised forms like " Five").  Each id
carries its rules so either definition can be scored.

Output (scorer-adaptable; mirrors the API response shape):
  /workspace/out/<lens>/<set>_<name>_<variant>.json      lens in {hosted_n1000, neel25}
      meta, _item, tokens[ {token,id,is_generated,results:[{type,top_tokens,top_probs,top_ids}]} ]
      (top-8 at every position, top-64 at the last), rank_exact{...}, continuations{...}, done
  /workspace/out/continuations/<set>_<name>_<variant>.json   the lens-independent gate data
  /workspace/out/manifest.json

--dry-run (offline, HF_HUB_OFFLINE=1, local tokenizer): builds every prompt id list and every
synonym id list, cross-checks them, prints them, loads no model.

Box (verbatim): /venv/main/bin/python /workspace/orderops/box/read_items.py 2>&1 | tee /workspace/logs/read_items.log
"""

from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

ENDOFTEXT_ID = 248044     # <|endoftext|>  (config bos/eos id; tokenizer.pad_token)
IM_END_ID = 248046        # <|im_end|>     (tokenizer.eos_token)
EOG_IDS = (ENDOFTEXT_ID, IM_END_ID)
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ITEMS_DIR = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_MODEL = "/workspace/models/qwen36-27b"
DEFAULT_TOKENIZER = "Qwen/Qwen3.6-27B"
DEFAULT_LENS_HOSTED = "/workspace/lens_hosted/qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
DEFAULT_LENS_NEEL = "/workspace/out/lens_neel/Qwen3.6-27B_jacobian_lens_neel25.pt"
DEFAULT_OUT = "/workspace/out"
NUM_RE = re.compile(r"(?<!\d)(\d+)(?!\d)")      # standalone integer (a digit run not embedded in a longer one)


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(*a):
    print(f"[{ts()}]", *a, flush=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- scorer definitions (verbatim, via AST)
def load_scorer_defs(path: str):
    """Execute only the W/C/OPS assignments and forms()/syn() defs of score_orderops.py."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    parts = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("W", "C", "OPS") for t in node.targets):
            parts.append(ast.get_source_segment(src, node))
        elif isinstance(node, ast.FunctionDef) and node.name in ("forms", "syn"):
            parts.append(ast.get_source_segment(src, node))
    ns: dict = {}
    exec("\n".join(parts), ns)          # noqa: S102 -- the frozen scorer's own definitions
    missing = [k for k in ("W", "C", "OPS", "forms", "syn") if k not in ns]
    if missing:
        raise RuntimeError(f"{path}: could not extract {missing}")
    return ns, hashlib.sha256(src.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- items
OP_PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "%": 2, "×": 2, "÷": 2, "x": 2}


def _apply(a: int, op: str, b: int):
    if op == "+": return a + b
    if op == "-": return a - b
    if op in ("*", "×", "x"): return a * b
    if op in ("/", "÷"): return a // b if b and a % b == 0 else None
    if op == "%": return a % b if b else None
    return None


def derive_decoy(prompt: str, intermediate: str, target: str):
    """Wrong-precedence decoy for a 3-operand expression `a op b op c` (parens optional), as in
    RESEARCH_NOTE.md ('for 2 + 3 * 4, 5 instead of 12'): the value of the pair that is NOT the
    correct first step. Returns None (undecidable) unless the parsed correct first step equals
    the item's intermediate and the decoy is a distinct non-negative integer."""
    tail = prompt.rstrip().rstrip("=").strip()
    tail = re.split(r"[.\n:;]", tail)[-1].strip()
    ops = r"[+\-*/%×÷x]"
    m1 = re.fullmatch(rf"\(\s*(\d+)\s*({ops})\s*(\d+)\s*\)\s*({ops})\s*(\d+)", tail)
    m2 = re.fullmatch(rf"(\d+)\s*({ops})\s*\(\s*(\d+)\s*({ops})\s*(\d+)\s*\)", tail)
    m3 = re.fullmatch(rf"(\d+)\s*({ops})\s*(\d+)\s*({ops})\s*(\d+)", tail)
    if m1:
        a, o1, b, o2, c = m1.groups(); first_is_left = True
    elif m2:
        a, o1, b, o2, c = m2.groups(); first_is_left = False
    elif m3:
        a, o1, b, o2, c = m3.groups(); first_is_left = OP_PREC[o2] <= OP_PREC[o1]
    else:
        return None
    a, b, c = int(a), int(b), int(c)
    left, right = _apply(a, o1, b), _apply(b, o2, c)
    first, other = (left, right) if first_is_left else (right, left)
    if first is None or str(first) != intermediate or other is None or other < 0:
        return None
    if str(other) in (intermediate, str(target)):
        return None
    return str(other)


def load_items(items_dir: str):
    paper = json.load(open(os.path.join(items_dir, "order_ops.json")))["items"]
    new = json.load(open(os.path.join(items_dir, "items_new50.json")))["items"]
    rows = []
    for setname, its in (("paper", paper), ("new", new)):
        for it in its:
            num = [k for k in it["intermediates"] if k.lstrip("-").isdigit()][0]
            op = [k for k in it["intermediates"] if not k.lstrip("-").isdigit()][0]
            nospace = it.get("prompt_nospace", it["prompt"].rstrip())
            if "prompt_nospace" in it and it["prompt_nospace"] != it["prompt"].rstrip():
                log(f"NOTE {setname}/{it['name']}: prompt_nospace != prompt.rstrip()")
            decoy_src = "item" if it.get("decoy") else "derived"
            decoy = str(it["decoy"]) if it.get("decoy") else derive_decoy(it["prompt"], num, str(it["target"]))
            used = set(re.findall(r"\d", it["prompt"])) | set(re.findall(r"\d", str(it["target"]))) | set(num)
            rows.append(dict(set=setname, name=it["name"], prompt=it["prompt"], prompt_nospace=nospace, target=str(it["target"]),
                             intermediates=list(it["intermediates"]), num=num, op=op, decoy=decoy, decoy_source=decoy_src if decoy else None,
                             two_digit=int(num) >= 10, uninvolved_digits=[str(d) for d in range(10) if str(d) not in used],
                             single_token_forms=it.get("single_token_forms")))
    return rows


# --------------------------------------------------------------------------- token ids for synonyms
def build_norm_index(tok):
    """vocab id -> decode().strip().lower(); inverted. The scorer's matching rule over the whole vocab."""
    index: dict[str, list[int]] = defaultdict(list)
    for i in range(len(tok)):
        s = tok.decode([i]).strip().lower()
        if s:
            index[s].append(i)
    return index


def synonym_ids(tok, syn_set, norm_index):
    entries: dict[int, dict] = {}
    for s in sorted(syn_set):
        for cand, tag in ((s, "tok"), (" " + s, "tok_space")):
            ids = tok(cand, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                entries.setdefault(ids[0], {"str": tok.decode(ids), "rules": set()})["rules"].add(tag)
        for i in norm_index.get(s, ()):
            entries.setdefault(i, {"str": tok.decode([i]), "rules": set()})["rules"].add("vocab")
    return [dict(id=i, str=e["str"], rules=sorted(e["rules"])) for i, e in sorted(entries.items())]


def build_keys(tok, scorer, item, norm_index):
    keys = {"intermediate": item["num"], "target": item["target"], "operation": item["op"]}
    if item["decoy"]:
        keys["decoy"] = item["decoy"]
    out = {}
    for role, key in keys.items():
        syn = scorer["syn"](key)
        out[role] = dict(key=key, synonyms=sorted(syn), ids=synonym_ids(tok, syn, norm_index))
    out["intermediate_nocjk_synonyms"] = sorted(scorer["syn"](item["num"], False))
    return out


# --------------------------------------------------------------------------- lens reader (one for both files)
def load_lens(path: str) -> dict:
    """Read a JacobianLens .pt: {"J": {layer: [d,d] fp16}, "n_prompts", "source_layers", "d_model"} (+ optional
    "provenance"). Layers are the block indices whose OUTPUT residual the matrix transports; transport = h @ J.T."""
    import torch
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:                      # weights_only refuses an unexpected pickled type: fall back
        log(f"NOTE: weights_only=True load failed for {path} ({type(e).__name__}); retrying with weights_only=False")
        ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "J" not in ck:
        raise ValueError(f"{path}: not a JacobianLens file (keys {sorted(ck) if isinstance(ck, dict) else type(ck)})")
    J = {int(l): t for l, t in ck["J"].items()}
    layers = sorted(J)
    d = int(ck.get("d_model") or J[layers[0]].shape[0])
    for l, t in J.items():
        if tuple(t.shape) != (d, d):
            raise ValueError(f"{path}: J[{l}] shape {tuple(t.shape)} != ({d},{d})")
    declared = ck.get("source_layers")
    if declared is not None and sorted(int(x) for x in declared) != layers:
        log(f"WARNING: {path}: source_layers {declared} != J keys {layers}; using J keys")
    return dict(path=path, J=J, source_layers=layers, n_prompts=int(ck.get("n_prompts") or 0), d_model=d, keys=sorted(ck),
                dtype_on_disk=str(J[layers[0]].dtype), provenance=ck.get("provenance"))


def load_text_model(path: str, dtype):
    """AutoModelForCausalLM resolves the text part of this Qwen3_5ForConditionalGeneration checkpoint; if a newer
    transformers refuses, load the wrapper class -- jlens.from_hf finds the decoder at model.language_model either way."""
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
    """output_hidden_states must give n_layers+1 = 65 entries (embeddings + 64 blocks) and hidden_states[l+1] must
    equal the hooked output of block l (the capture point used here). The final entry is post-final-norm in
    transformers 5.x and is expected to DIFFER from the hooked final-block output -- which is why hooks are used."""
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
    log(f"capture check: {len(hs)} hidden_states; hidden_states[l+1]==block-l output {'OK' if not mism else 'MISMATCH at ' + str(mism)}; "
        f"final entry {'EQUALS' if final_equal else 'differs from'} hooked block-{lm.n_layers - 1} output (post-norm: differs expected)")
    if mism:
        log("WARNING: hidden_states indexing differs from the hook capture; the pipeline uses the hooks (upstream convention) regardless")
    return dict(n_hidden_states=len(hs), hidden_states_lplus1_equals_block_l_output=(not mism), mismatched_layers=mism,
                final_entry_equals_hooked_final_block=final_equal)


# --------------------------------------------------------------------------- text metrics (Addendum 2)
def numbers_in(text: str) -> list[int]:
    return [int(x) for x in NUM_RE.findall(text)]


def first_number(text: str):
    m = NUM_RE.search(text)
    return int(m.group(1)) if m else None


def leaks(text: str, intermediate: int, target: int) -> bool:
    nums = numbers_in(text)
    prefix = nums[: nums.index(target)] if target in nums else nums
    return intermediate in prefix


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items-dir", default=DEFAULT_ITEMS_DIR, help="dir holding order_ops.json, items_new50.json, score_orderops.py")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", default=None, help="default: --model (dry-run: Qwen/Qwen3.6-27B from the HF cache)")
    ap.add_argument("--lens-hosted", default=DEFAULT_LENS_HOSTED)
    ap.add_argument("--lens-neel", default=DEFAULT_LENS_NEEL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--bos", choices=["none", "endoftext"], default="none",
                    help="none = what Neuronpedia's prependBos did for this BOS-less tokenizer (default); endoftext = sensitivity variant")
    ap.add_argument("--topk-last", type=int, default=64)
    ap.add_argument("--topk-all", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=6)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--sample-max-new", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--skip-sampling", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the first N items (smoke test)")
    ap.add_argument("--hash-shards", action="store_true", help="also sha256 every model shard into the manifest (minutes)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok_src = args.tokenizer or (DEFAULT_TOKENIZER if args.dry_run and not os.path.isdir(args.model) else args.model)
    tok = AutoTokenizer.from_pretrained(tok_src)
    assert tok.bos_token_id is None, "expected bos_token=None (Qwen); the BOS policy below assumes it"
    assert tok.convert_tokens_to_ids("<|endoftext|>") == ENDOFTEXT_ID and tok.convert_tokens_to_ids("<|im_end|>") == IM_END_ID
    scorer, scorer_sha = load_scorer_defs(os.path.join(args.items_dir, "score_orderops.py"))
    items = load_items(args.items_dir)
    if args.limit:
        items = items[: args.limit]
    log(f"tokenizer {tok_src} (len {len(tok)}); scorer sha256 {scorer_sha[:16]}…; {len(items)} items; bos={args.bos}")

    t0 = time.time()
    norm_index = build_norm_index(tok)
    log(f"vocab normalisation index built over {len(tok)} ids in {time.time()-t0:.1f}s")
    digit_ids = {str(d): synonym_ids(tok, {str(d)}, norm_index) for d in range(10)}
    bos_prefix = [ENDOFTEXT_ID] if args.bos == "endoftext" else []

    # ---- prompt id lists + cross-checks (dry-run prints these; the real run stores them)
    prepared, mismatches, form_misses, n_no_ids = [], [], [], 0
    for it in items:
        keys = build_keys(tok, scorer, it, norm_index)
        if not keys["intermediate"]["ids"]:
            n_no_ids += 1
        if it["single_token_forms"]:
            have = {e["str"] for e in keys["intermediate"]["ids"]}
            miss = [f for f in it["single_token_forms"] if f not in have]
            if miss:
                form_misses.append((it["name"], miss))
        for variant, prompt in (("space", it["prompt"]), ("nospace", it["prompt_nospace"])):
            raw_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            ref = os.path.join(args.items_dir, "v2_api", "raw_A", f"{it['set']}_{it['name']}_{variant}.json")
            if os.path.exists(ref):
                sent = json.load(open(ref)).get("_item", {}).get("input_ids")
                if sent is not None and sent != raw_ids:
                    mismatches.append((it["name"], variant, sent, raw_ids))
            prepared.append(dict(item=it, variant=variant, prompt=prompt, raw_ids=raw_ids, ids=bos_prefix + raw_ids, keys=keys))
    log(f"prepared {len(prepared)} prompts; ids differ from v2_api/raw_A in {len(mismatches)} cases; "
        f"intermediates with NO single-token id: {n_no_ids}; new50 single_token_forms not covered: {len(form_misses)}")
    for m in mismatches[:10]:
        log("  MISMATCH", m)
    for m in form_misses[:10]:
        log("  FORM MISS", m)

    if args.dry_run:
        fmt = lambda ids: " ".join(f"{e['id']}:{e['str']!r}:{'/'.join(e['rules'])}" for e in ids)
        for p in prepared:
            it, k = p["item"], p["keys"]
            print(f"{it['set']}/{it['name']} [{p['variant']}] prompt={p['prompt']!r}")
            print(f"   ids={p['ids']}  toks={[tok.decode([i]) for i in p['ids']]}")
            print(f"   intermediate {k['intermediate']['key']!r}: {fmt(k['intermediate']['ids'])}")
            if p["variant"] == "space":
                print(f"   target {k['target']['key']!r}: {fmt(k['target']['ids'])}")
                print(f"   decoy {k.get('decoy', {}).get('key')!r} ({it['decoy_source']}): {fmt(k['decoy']['ids']) if 'decoy' in k else '-'}")
                print(f"   operation {k['operation']['key']!r}: {fmt(k['operation']['ids'])}")
                print(f"   uninvolved digits {it['uninvolved_digits']}")
        n_decoy = sum(1 for it in items if it["decoy"])
        n_decoy_paper = sum(1 for it in items if it["decoy"] and it["set"] == "paper")
        print(f"\nDRY RUN SUMMARY: {len(items)} items / {len(prepared)} prompts; bos={args.bos}; "
              f"decoys available {n_decoy} (paper derived {n_decoy_paper}, new from file {n_decoy - n_decoy_paper}); "
              f"intermediates with no single-token id {n_no_ids}; id mismatches vs API {len(mismatches)}; form misses {len(form_misses)}; "
              f"digit token ids: " + "; ".join(f"{d}->{[e['id'] for e in v]}" for d, v in digit_ids.items()))
        return

    # ------------------------------------------------------------------ model + lenses
    import torch
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder

    os.makedirs(args.out, exist_ok=True)
    log(f"loading {args.model} bf16 on cuda …")
    hf = load_text_model(args.model, torch.bfloat16)
    lm = from_hf(hf, tok)
    n_layers, dev = lm.n_layers, lm.input_device
    log(f"{lm}; GPU {torch.cuda.memory_allocated()/2**30:.1f} GiB")
    capture_check = verify_capture(lm, tok)

    lenses = {}
    for name, path in (("hosted_n1000", args.lens_hosted), ("neel25", args.lens_neel)):
        if not os.path.exists(path):
            log(f"WARNING: lens {name} missing at {path}; skipping it")
            continue
        L = load_lens(path)
        if L["d_model"] != lm.d_model:
            sys.exit(f"lens {name}: d_model {L['d_model']} != model {lm.d_model}")
        J = {l: t.to(dev).float() for l, t in L["J"].items()}       # fp32 on GPU (~6.6 GB per lens); transport = h @ J.T
        del L["J"]
        lenses[name] = dict(L, sha256=sha256_file(path), bytes=os.path.getsize(path), J=J)
        log(f"lens {name}: layers {L['source_layers'][0]}..{L['source_layers'][-1]} ({len(L['source_layers'])}), n_prompts {L['n_prompts']}, "
            f"disk dtype {L['dtype_on_disk']}, keys {L['keys']}; GPU {torch.cuda.memory_allocated()/2**30:.1f} GiB")
    if not lenses:
        sys.exit("no lens loaded")
    for name in lenses:
        os.makedirs(os.path.join(args.out, name), exist_ok=True)
    os.makedirs(os.path.join(args.out, "continuations"), exist_ok=True)

    decode_cache: dict[int, str] = {}

    def dec(i: int) -> str:
        if i not in decode_cache:
            decode_cache[i] = tok.decode([i])
        return decode_cache[i]

    @torch.no_grad()
    def residuals(ids: torch.Tensor, at):
        with ActivationRecorder(lm.layers, at=list(at)) as rec:
            lm.forward(ids)
            return {l: rec.activations[l].detach() for l in at}       # [B, seq, d] bf16

    @torch.no_grad()
    def next_logits(ids: torch.Tensor) -> torch.Tensor:
        h = residuals(ids, [n_layers - 1])[n_layers - 1][:, -1, :]
        return lm.unembed(h).float()

    def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
        sp, si = probs.sort(-1, descending=True)
        remove = (sp.cumsum(-1) - sp) >= top_p           # prior mass already >= top_p -> drop (HF TopPLogitsWarper semantics)
        sp = sp.masked_fill(remove, 0.0)
        out = torch.zeros_like(probs).scatter_(-1, si, sp)
        return out / out.sum(-1, keepdim=True)

    @torch.no_grad()
    def generate(prompt_ids: list[int], n_new: int, seeds=None):
        """Greedy (seeds None) or per-row-seeded sampling, batched, full re-forward each step (prompts are
        <= ~30 tokens, so a cached step and a full forward cost the same: both read the 54 GB of weights)."""
        B = 1 if seeds is None else len(seeds)
        cur = torch.tensor([prompt_ids] * B, dtype=torch.long, device=dev)
        gens = None if seeds is None else [torch.Generator(device=dev).manual_seed(int(s)) for s in seeds]
        out = [[] for _ in range(B)]
        done = [False] * B
        for _ in range(n_new):
            logits = next_logits(cur)
            if gens is None:
                nxt = logits.argmax(-1)
            else:
                probs = top_p_filter(torch.softmax(logits / args.temperature, -1), args.top_p)
                nxt = torch.stack([torch.multinomial(probs[b], 1, generator=gens[b])[0] for b in range(B)])
            for b in range(B):
                if not done[b]:
                    t = int(nxt[b]); out[b].append(t)
                    if t in EOG_IDS:
                        done[b] = True
            if all(done):
                break
            cur = torch.cat([cur, nxt[:, None]], 1)
        res = []
        for b in range(B):
            ids = out[b]
            stop = dec(ids[-1]) if ids and ids[-1] in EOG_IDS else None
            text_ids = ids[:-1] if stop else ids
            res.append(dict(ids=ids, text=tok.decode(text_ids, skip_special_tokens=False), stopped_at=stop))
        return res

    lens_layers = {name: sorted(L["J"]) for name, L in lenses.items()}
    layer_list = list(range(n_layers))
    started = ts()
    manifest = dict(kind="orderops_v2_box_manifest", started_utc=started, model=args.model,
                    model_config_sha256=sha256_file(os.path.join(args.model, "config.json")),
                    model_index_sha256=sha256_file(os.path.join(args.model, "model.safetensors.index.json")) if os.path.exists(os.path.join(args.model, "model.safetensors.index.json")) else None,
                    model_hub_revision=(sorted(os.listdir(os.path.join(os.environ.get("HF_HOME", "/workspace/.hf_home"), "hub", "models--Qwen--Qwen3.6-27B", "snapshots")))[:1] or [None])[0]
                    if os.path.isdir(os.path.join(os.environ.get("HF_HOME", "/workspace/.hf_home"), "hub", "models--Qwen--Qwen3.6-27B", "snapshots")) else None,
                    model_shards={os.path.basename(p): os.path.getsize(p) for p in sorted(glob.glob(os.path.join(args.model, "*.safetensors")))},
                    model_shard_sha256={os.path.basename(p): sha256_file(p) for p in sorted(glob.glob(os.path.join(args.model, "*.safetensors")))} if args.hash_shards else None,
                    lenses={n: {k: v for k, v in L.items() if k != "J"} for n, L in lenses.items()},
                    torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                    transformers=__import__("transformers").__version__, n_layers=n_layers, d_model=lm.d_model, capture_check=capture_check,
                    readout="lm_head(final_norm(h @ J[l].T)) for l in layers_with_jacobian; lm_head(final_norm(h)) otherwise",
                    logit_vocab_size=int(lm.unembed(torch.zeros(1, lm.d_model, device=dev)).shape[-1]), n_real_tokens=len(tok),
                    bos_policy=args.bos, bos_prefix=bos_prefix, scorer_sha256=scorer_sha, items=len(items), prompts=len(prepared),
                    args={k: v for k, v in vars(args).items()}, rank_convention="1-based: 1 + #logits strictly greater, over the full lm_head width",
                    layers_with_jacobian=lens_layers, identity_layers={n: [l for l in layer_list if l not in lens_layers[n]] for n in lenses},
                    sampling=dict(n=args.n_samples, temperature=args.temperature, top_p=args.top_p, max_new=args.sample_max_new, seeds=list(range(args.n_samples)),
                                  note="one CUDA torch.Generator per row seeded 0..n-1; rows decoded as one batch, full re-forward per step"),
                    id_mismatches_vs_api=[m[:2] for m in mismatches], form_misses=form_misses)
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)

    # ------------------------------------------------------------------ main loop
    for pi, p in enumerate(prepared):
        it, variant, ids = p["item"], p["variant"], p["ids"]
        tag = f"{it['set']}_{it['name']}_{variant}"
        t0 = time.time()
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        acts = residuals(x, layer_list)                                   # ONE forward pass; both lenses read it
        seq = len(ids)
        last = seq - 1
        h_all = torch.stack([acts[l][0] for l in layer_list]).float()   # [L, seq, d]
        resid_norm = h_all[:, last, :].norm(dim=-1).tolist()

        # continuations (lens-independent): greedy + Addendum 2 samples, on the raw prompt ids
        greedy = generate(ids, args.max_new_tokens)[0]
        samples = [] if args.skip_sampling else generate(ids, args.sample_max_new, seeds=list(range(args.n_samples)))
        for s_i, s in enumerate(samples):
            s["seed"] = s_i
        tgt, inter = int(it["target"]), int(it["num"])
        greedy["first_number"] = first_number(greedy["text"]); greedy["correct"] = greedy["first_number"] == tgt
        for s in samples:
            s["first_number"] = first_number(s["text"]); s["asserts_target"] = s["first_number"] == tgt; s["leaks"] = leaks(s["text"], inter, tgt)
        greedy["leaks"] = leaks(greedy["text"], inter, tgt)
        assert_rate = (sum(s["asserts_target"] for s in samples) / len(samples)) if samples else None
        leak_any = bool(greedy["leaks"] or any(s["leaks"] for s in samples))
        cont = dict(set=it["set"], name=it["name"], variant=variant, prompt=p["prompt"], target=it["target"], intermediate=it["num"],
                    greedy=greedy, samples=samples, assert_rate=assert_rate, leak_any=leak_any,
                    greedy_writes_intermediate_first=(greedy["first_number"] == inter),
                    admissible=(assert_rate is not None and assert_rate >= 0.8 and not leak_any))
        json.dump(cont, open(os.path.join(args.out, "continuations", f"{tag}.json"), "w"), ensure_ascii=False, indent=1)

        for lname, L in lenses.items():
            J = L["J"]
            tokens_out = []
            per_pos_top = [[None] * n_layers for _ in range(seq)]
            rank_exact = dict(position=last, keys={}, digits={}, uninvolved_digits=it["uninvolved_digits"],
                              n_pad_in_top64_by_layer=[], resid_norm_by_layer=resid_norm, lens_applied_by_layer=[])
            key_ids = {role: [e["id"] for e in v["ids"]] for role, v in p["keys"].items() if isinstance(v, dict)}
            digit_id_lists = {d: [e["id"] for e in v] for d, v in digit_ids.items()}
            key_ranks = {role: [[] for _ in ids_] for role, ids_ in key_ids.items()}
            digit_ranks = {d: [[] for _ in ids_] for d, ids_ in digit_id_lists.items()}
            with torch.no_grad():
                for l in layer_list:
                    h = h_all[l]                                             # [seq, d] fp32
                    applied = l in J
                    if applied:
                        h = h @ J[l].T
                    logits = lm.unembed(h).float()                            # [seq, V]
                    probs = torch.softmax(logits, -1)
                    tk = probs.topk(args.topk_all, -1)
                    for pos in range(seq):
                        per_pos_top[pos][l] = (tk.indices[pos].tolist(), tk.values[pos].tolist())
                    tk64 = probs[last].topk(args.topk_last)
                    per_pos_top[last][l] = (tk64.indices.tolist(), tk64.values.tolist())
                    rank_exact["n_pad_in_top64_by_layer"].append(int((tk64.indices >= len(tok)).sum()))
                    rank_exact["lens_applied_by_layer"].append(applied)
                    row = logits[last]
                    for role, ids_ in key_ids.items():
                        if ids_:
                            sc = row[torch.tensor(ids_, device=dev)]
                            r = 1 + (row[None, :] > sc[:, None]).sum(1)
                            for j, rv in enumerate(r.tolist()):
                                key_ranks[role][j].append(rv)
                    for d, ids_ in digit_id_lists.items():
                        sc = row[torch.tensor(ids_, device=dev)]
                        r = 1 + (row[None, :] > sc[:, None]).sum(1)
                        for j, rv in enumerate(r.tolist()):
                            digit_ranks[d][j].append(rv)
            for pos in range(seq):
                tt = [[dec(i) for i in per_pos_top[pos][l][0]] for l in layer_list]
                tp = [per_pos_top[pos][l][1] for l in layer_list]
                ti = [per_pos_top[pos][l][0] for l in layer_list]
                tokens_out.append(dict(kind="token", position=pos, token=dec(ids[pos]), id=ids[pos], is_generated=False,
                                       results=[dict(type="JACOBIAN_LENS", top_tokens=tt, top_probs=tp, top_ids=ti)]))
            for role, v in p["keys"].items():
                if not isinstance(v, dict):
                    continue
                entries = [dict(e, ranks=key_ranks[role][j]) for j, e in enumerate(v["ids"])]
                min_by_layer = [min(e["ranks"][l] for e in entries) for l in layer_list] if entries else None
                rank_exact["keys"][role] = dict(key=v["key"], synonyms=v["synonyms"], ids=entries, min_rank_by_layer=min_by_layer,
                                                best_rank=min(min_by_layer) if min_by_layer else None,
                                                best_layer=min_by_layer.index(min(min_by_layer)) if min_by_layer else None)
            rank_exact["keys"]["intermediate_nocjk_synonyms"] = p["keys"]["intermediate_nocjk_synonyms"]
            for d, v in digit_ids.items():
                entries = [dict(e, ranks=digit_ranks[d][j]) for j, e in enumerate(v)]
                rank_exact["digits"][d] = dict(ids=entries, min_rank_by_layer=[min(e["ranks"][l] for e in entries) for l in layer_list])
            doc = dict(
                meta=dict(kind="meta", model=args.model, lens=lname, lens_path=L["path"], lens_sha256=L["sha256"], types=["JACOBIAN_LENS"],
                          layers_by_type={"JACOBIAN_LENS": layer_list}, layers_with_jacobian=lens_layers[lname],
                          top_n=args.topk_all, top_n_last=args.topk_last, prompt_len=seq, num_completion_tokens=0, temperature=0,
                          prepend_bos=False, bos_policy=args.bos, bos_prefix=bos_prefix, reuse_len=0),
                _item=dict(set=it["set"], name=it["name"], variant=variant, prompt=p["prompt"], target=it["target"], intermediates=it["intermediates"],
                           decoy=it["decoy"], decoy_source=it["decoy_source"], two_digit=it["two_digit"], input_ids=p["raw_ids"], model_input_ids=ids),
                tokens=tokens_out, rank_exact=rank_exact, continuations=cont,
                done=dict(kind="done", seq_len=seq, prompt_len=seq, vocab_size=manifest["logit_vocab_size"], completion=greedy["text"]))
            json.dump(doc, open(os.path.join(args.out, lname, f"{tag}.json"), "w"), ensure_ascii=False)
        ir = {n: rank_exact_best(os.path.join(args.out, n, f"{tag}.json")) for n in lenses}
        log(f"[{pi+1}/{len(prepared)}] {tag:40s} {time.time()-t0:5.1f}s  greedy={greedy['text']!r:>12} correct={greedy['correct']} "
            f"assert={assert_rate} leak={leak_any} adm={cont['admissible']}  best_rank(intermediate)={ir}")

    manifest["finished_utc"] = ts()
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    log(f"DONE {len(prepared)} prompts x {len(lenses)} lenses -> {args.out}")


def rank_exact_best(path):
    try:
        k = json.load(open(path))["rank_exact"]["keys"]["intermediate"]
        return (k["best_rank"], k["best_layer"])
    except Exception:
        return None


if __name__ == "__main__":
    main()
