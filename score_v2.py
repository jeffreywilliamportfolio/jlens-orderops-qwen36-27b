#!/usr/bin/env python3
"""score_v2.py -- score the frozen preregistration PREREG_orderops_v2.md (+ addenda 1-5) against whatever
data exists and write RESULTS_v2.md.  Python 3 stdlib only, no network, deterministic, idempotent.

Reads (never writes) the frozen files listed in FREEZE.sha256 and re-verifies their hashes.
Synonym sets W/C/OPS and forms()/syn()/best_rank()/cp() are pulled from score_orderops.py by AST and
executed as-is (never retyped); derive_decoy()/_apply()/OP_PREC come the same way from box/read_items.py;
best_rank_topk() is imported from box/check_c1_vs_a.py.

Frozen definitions applied here:
  readout      = last prompt token (variant `space`: the ' ' after '='; variant `nospace`: the ' =' / ' equals' token)
  rank         = min over single-token synonyms (digit / word / CJK, stripped lower-cased) at each layer,
                 then min over SOURCE layers 0..62.  Layer 63 is the model's own next-token top-8 (no Jacobian)
                 and is the addendum-4 control; it is NOT part of the lens rank.
  pass@k       = fraction with rank <= k;  CP = Clopper-Pearson 95%;  rank > top-N is recorded as 99 (absent).
  sign test    = intermediate rank vs the item's median uninvolved-digit rank (frozen scorer's formula).
  correct      = first number written == target, with targets AND generated text normalised through a number-word
                 map (zero..ninety-nine, hyphen or space) before comparison.  Box `continuations/*.json`
                 `greedy.correct` is word-aware and trusted; API raw_C flags were computed by digit-string
                 equality and are RECOMPUTED here from the stored greedy/samples text (six paper items carry
                 word targets: fourteen, seven, twenty, four, twelve, three).  Box preferred, else API.
  admissible   = assert_rate >= 0.8 and not leak_any (box sampled pass preferred -- from --box-full-dir when
                 given, else the box dir's own continuations if they carry samples -- else API arm C);
                 addendum-2 rule: < 30 admissible of 105 -> H1/H2 "not testable as designed".
Usage:
  python3 score_v2.py [--box-dir v2_box] [--box-full-dir v2_box_full] [--out RESULTS_v2.md]
  --box-dir       box output of the fast pass (lens reads + greedy; assert_rate/admissible None)   default v2_box
  --box-full-dir  box output of the sampled pass; admissibility is taken from here when given        default none
Box lens subdirectories are `hosted_n1000` and `neel25`; the second is the addendum-5 5-prompt fit and is
labelled "neel5 (5-prompt fit)" everywhere in the output.
"""
import argparse
import ast
import glob
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(HERE, *a)
N_LAYERS = 64
SOURCE_LAYERS = list(range(63))            # 0..62 -- the lens rank; 63 = next-token control
API_TOPN = 8
LENS_DIRS = OrderedDict([("hosted_n1000", "hosted (n1000)"), ("neel25", "neel5 (5-prompt fit)")])
NEEL = "neel25"                            # on-disk name of the 5-prompt lens
ABSENT = 99
WORD_TARGET_ITEMS = ("word-add-mult", "word-mult-sub", "word-parens", "word-sub-mult", "word-add-add", "word-div-sub")


# ----------------------------------------------------------------------------------------------- frozen code, by AST
def ast_extract(path, names, funcs, extra_ns=None):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    parts = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            parts.append(ast.get_source_segment(src, node))
        elif isinstance(node, ast.FunctionDef) and node.name in funcs:
            parts.append(ast.get_source_segment(src, node))
    ns = dict(extra_ns or {})
    ns.setdefault("math", math); ns.setdefault("re", re)
    exec("\n".join(parts), ns)          # noqa: S102 -- the frozen files' own definitions, verbatim
    missing = [k for k in list(names) + list(funcs) if k not in ns]
    if missing:
        raise RuntimeError(f"{path}: could not extract {missing}")
    return ns


SC = ast_extract(P("score_orderops.py"), ("W", "C", "OPS"), ("forms", "syn", "best_rank", "cp"))
BX = ast_extract(P("box", "read_items.py"), ("OP_PREC",), ("_apply", "derive_decoy"))
forms, syn, cp = SC["forms"], SC["syn"], SC["cp"]
derive_decoy = BX["derive_decoy"]

_spec = importlib.util.spec_from_file_location("check_c1_vs_a", P("box", "check_c1_vs_a.py"))
check_c1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(check_c1)
best_rank_topk = check_c1.best_rank_topk     # all-64-layer version, used verbatim for the gate's "as written" line


# ----------------------------------------------------------------------------------------------- word-aware numbers
WORD2NUM = {v: k for k, v in SC["W"].items()}                       # zero..twenty, thirty..ninety (from the frozen W)
_TENS = [SC["W"][t] for t in (20, 30, 40, 50, 60, 70, 80, 90)]
_ONES = [SC["W"][o] for o in range(1, 10)]
_SINGLE = sorted(SC["W"].values(), key=len, reverse=True)
NUM_TOKEN_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)"                                                    # 1: digit run (the frozen runner's regex)
    r"|\b(" + "|".join(_TENS) + r")(?:[- ](" + "|".join(_ONES) + r"))?\b"   # 2,3: twenty-one / twenty one / twenty
    r"|\b(" + "|".join(_SINGLE) + r")\b",                                   # 4: zero..nineteen, thirty..ninety
    re.IGNORECASE)


def numbers_in(text):
    """Every number written in `text`, in order, as canonical strings ('14', '-3', '2.5'); words mapped to digits."""
    out = []
    for m in NUM_TOKEN_RE.finditer(text or ""):
        if m.group(1):
            out.append(m.group(1))
        elif m.group(2):
            out.append(str(WORD2NUM[m.group(2).lower()] + (WORD2NUM[m.group(3).lower()] if m.group(3) else 0)))
        else:
            out.append(str(WORD2NUM[m.group(4).lower()]))
    return out


def first_num_wa(text):
    n = numbers_in(text)
    return n[0] if n else None


def leaks_wa(text, inter, target):
    """Addendum-2 leak: the intermediate is written as a number before the target (word-aware)."""
    for x in numbers_in(text):
        if x == target:
            return False
        if x == inter:
            return True
    return False


def norm_target(t):
    t = str(t).strip()
    if t.lstrip("-").isdigit():
        return t
    n = numbers_in(t.lower())
    return n[0] if n else t.lower()


# ----------------------------------------------------------------------------------------------- helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


BAD_FILES = []


def load_json(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception as e:                       # partial / truncated file from a run still in progress
        BAD_FILES.append((os.path.relpath(path, HERE), f"{type(e).__name__}: {str(e)[:80]}"))
        return None


def ranks_by_layer(tok_entry, S, k):
    """Per-layer rank (1-based) of the first decoded top-k string whose strip().lower() is in S; 99 = absent.
    Same matching rule as the frozen best_rank(), kept per layer and capped at k (API parity when k=8)."""
    out = []
    for layer in tok_entry["results"][0]["top_tokens"]:
        b = ABSENT
        for r, s in enumerate(layer[:k]):
            if s.strip().lower() in S:
                b = r + 1
                break
        out.append(b)
    return out


def lens_min(rbl, layers=SOURCE_LAYERS):
    vals = [rbl[l] for l in layers if l < len(rbl) and rbl[l] is not None]
    return min(vals) if vals else ABSENT


def span3(rbl, layers=SOURCE_LAYERS):
    hit = [l for l in layers if l < len(rbl) and rbl[l] is not None and rbl[l] <= 3]
    return (len(hit), hit[0], hit[-1]) if hit else (0, None, None)


def passk(vals, k):
    return sum(v <= k for v in vals)


def fmt_rate(k, n, ci=True):
    if n == 0:
        return "n/a (n=0)"
    s = f"{k}/{n} = {k/n:.2f}"
    if ci:
        lo, hi = cp(k, n)
        s += f" [{lo:.2f}, {hi:.2f}]"
    return s


def sign_test(pairs):
    """Frozen scorer lines 77-81: better/worse/tie vs the item's median uninvolved rank; two-sided binomial."""
    w = l = t = 0
    for a, m in pairs:
        w += a < m; l += a > m; t += a == m
    p = sum(math.comb(w + l, i) for i in range(0, l + 1)) / 2 ** (w + l) * 2 if w + l else 1.0
    return w, l, t, min(1.0, p)


def mrr(vals):
    return sum(1.0 / v for v in vals if v <= API_TOPN) / len(vals) if vals else float("nan")


def median_or_none(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def syn_target(target):
    """Target synonyms for the spoken-vs-unspoken comparison (a word target such as 'fourteen' is mapped to 14)."""
    t = norm_target(target)
    return syn(t) if t.lstrip("-").isdigit() else {t}


# ----------------------------------------------------------------------------------------------- items
def load_items():
    paper = load_json(P("order_ops.json"))["items"]
    new = load_json(P("items_new50.json"))["items"]
    items = []
    for setname, its in (("paper", paper), ("new", new)):
        for it in its:
            num = [k for k in it["intermediates"] if k.lstrip("-").isdigit()][0]
            op = [k for k in it["intermediates"] if not k.lstrip("-").isdigit()][0]
            used = set(re.findall(r"\d", it["prompt"])) | set(re.findall(r"\d", str(it["target"]))) | set(num)
            decoy = str(it["decoy"]) if it.get("decoy") else derive_decoy(it["prompt"], num, str(it["target"]))
            items.append(dict(
                set=setname, name=it["name"], prompt=it["prompt"], prompt_nospace=it.get("prompt_nospace", it["prompt"].rstrip()),
                num=num, op=op, target=str(it["target"]), target_num=norm_target(it["target"]), two_digit=int(num) >= 10,
                decoy=decoy, decoy_source=("item" if it.get("decoy") else ("derived" if decoy else None)),
                uninvolved=[str(x) for x in range(10) if str(x) not in used],
                S=syn(num), S_nocjk=syn(num, False), S_target=syn_target(it["target"]),
                S_decoy=syn(decoy) if decoy else None,
                forms={k: {x.lower() for x in v} for k, v in forms(int(num)).items()},
                word_target=not str(it["target"]).lstrip("-").isdigit()))
    return items


# ----------------------------------------------------------------------------------------------- lens reads
def finish_record(r):
    r["num"] = lens_min(r["rbl_num"]); r["num_nocjk"] = lens_min(r["rbl_num_nocjk"])
    r["target"] = lens_min(r["rbl_target"]); r["decoy"] = lens_min(r["rbl_decoy"]) if r.get("rbl_decoy") else None
    r["op"] = lens_min(r["rbl_op"]) if r.get("rbl_op") else None
    r["forms"] = {f: lens_min(v) for f, v in r["rbl_forms"].items()} if r.get("rbl_forms") else {}
    r["un"] = [lens_min(v) for v in r["rbl_un"].values()]
    r["un_median"] = statistics.median(r["un"]) if r["un"] else None
    r["L63_num"] = r["rbl_num"][63] if len(r["rbl_num"]) > 63 else None
    r["L63_target"] = r["rbl_target"][63] if len(r["rbl_target"]) > 63 else None
    r["L63_decoy"] = r["rbl_decoy"][63] if r.get("rbl_decoy") and len(r["rbl_decoy"]) > 63 else None
    r["span"] = span3(r["rbl_num"])
    src = [r["rbl_num"][l] for l in SOURCE_LAYERS if l < len(r["rbl_num"])]
    r["best_layer"] = src.index(min(src)) if src and min(src) < ABSENT else None


def read_lens_file(path, item, k, doc=None):
    """One lens read (API raw_A or box lens file) -> per-layer ranks (top-k sliced) + derived scalars."""
    d = doc if doc is not None else load_json(path)
    if d is None or "tokens" not in d:
        return None
    prompt_toks = [t for t in d["tokens"] if not t.get("is_generated")]
    if not prompt_toks:
        return None
    last = prompt_toks[-1]
    if len(last["results"][0]["top_tokens"]) != N_LAYERS:
        BAD_FILES.append((os.path.relpath(path, HERE), f"{len(last['results'][0]['top_tokens'])} layers, expected {N_LAYERS}"))
    r = dict(last_tok=last["token"], n_prompt_tokens=len(prompt_toks), doc=d)
    r["rbl_num"] = ranks_by_layer(last, item["S"], k)
    r["rbl_num_nocjk"] = ranks_by_layer(last, item["S_nocjk"], k)
    r["rbl_target"] = ranks_by_layer(last, item["S_target"], k)
    r["rbl_decoy"] = ranks_by_layer(last, item["S_decoy"], k) if item["S_decoy"] else None
    r["rbl_forms"] = {f: ranks_by_layer(last, S, k) for f, S in item["forms"].items()}
    r["rbl_un"] = {dg: ranks_by_layer(last, {dg}, k) for dg in item["uninvolved"]}
    r["rbl_op"] = ranks_by_layer(last, syn(item["op"]), k)
    finish_record(r)
    r["frozen_all64_num"] = best_rank_topk(last, item["S"], k)      # the scorer's own rule incl. layer 63 (sensitivity)
    return r


def read_box_exact(doc, item):
    """Exact full-vocab ranks from a box lens file's rank_exact block (64 entries per key, layer 63 = identity)."""
    rx = doc.get("rank_exact")
    if not rx or "keys" not in rx:
        return None
    K = rx["keys"]

    def mrl(role):
        v = K.get(role)
        return v.get("min_rank_by_layer") if isinstance(v, dict) else None

    r = dict()
    r["rbl_num"] = mrl("intermediate")
    if r["rbl_num"] is None:
        return None
    nocjk = set(K.get("intermediate_nocjk_synonyms") or item["S_nocjk"])
    ids = K["intermediate"].get("ids", [])
    ents = [e for e in ids if e.get("str", "").strip().lower() in nocjk]
    r["rbl_num_nocjk"] = [min(e["ranks"][l] for e in ents) for l in range(N_LAYERS)] if ents else [ABSENT] * N_LAYERS
    r["rbl_target"] = mrl("target") or [ABSENT] * N_LAYERS
    r["rbl_decoy"] = mrl("decoy")
    r["rbl_op"] = mrl("operation")
    fr = {}
    for f, S in item["forms"].items():
        es = [e for e in ids if e.get("str", "").strip().lower() in S]
        fr[f] = [min(e["ranks"][l] for e in es) for l in range(N_LAYERS)] if es else [ABSENT] * N_LAYERS
    r["rbl_forms"] = fr
    digits = rx.get("digits", {})
    r["rbl_un"] = {dg: digits[dg]["min_rank_by_layer"] for dg in item["uninvolved"] if dg in digits and digits[dg].get("min_rank_by_layer")}
    r["lens_applied"] = rx.get("lens_applied_by_layer")
    finish_record(r)
    if r["lens_applied"]:
        jl = [l for l in SOURCE_LAYERS if l < len(r["lens_applied"]) and r["lens_applied"][l]]
        r["num_jacobian_only"] = lens_min(r["rbl_num"], jl)
    return r


# ----------------------------------------------------------------------------------------------- continuations / gate
def api_c_record(rec, variant):
    """API arm C, recomputed word-aware from the stored text; the stored digit-string flags are kept for comparison."""
    v = rec.get("variants", {}).get(variant)
    if not v:
        return None
    inter, target = str(rec["intermediate"]), norm_target(rec["target"])
    g = v.get("greedy") or ""
    fn = first_num_wa(g)
    r = dict(source="api", greedy=g, correct=(fn == target), writes_inter_first=(fn == inter),
             stored_correct=v.get("correct_greedy"), stored_admissible=None,
             assert_rate=None, leak_any=None, admissible=None, samples=[], greedy_leaks=leaks_wa(g, inter, target))
    if variant == "space" and "samples" in rec:          # the API gate ran on the with-space variant only
        r["samples"] = [dict(text=s, leaks=leaks_wa(s, inter, target), asserts=(first_num_wa(s) == target)) for s in rec["samples"]]
        n = len(r["samples"])
        r["assert_rate"] = (sum(s["asserts"] for s in r["samples"]) / n) if n else None
        r["leak_any"] = bool(r["greedy_leaks"] or any(s["leaks"] for s in r["samples"]))
        r["admissible"] = (r["assert_rate"] is not None and r["assert_rate"] >= 0.8 and not r["leak_any"])
        r["stored_admissible"] = rec.get("admissible")
    return r


def box_cont_record(c):
    """Box continuation (greedy.correct is word-aware on the box and trusted); samples only if the sampled pass ran."""
    g = c.get("greedy") or {}
    samples = c.get("samples") or []
    inter, target = str(c.get("intermediate")), norm_target(c.get("target"))
    r = dict(source="box", greedy=g.get("text"), correct=g.get("correct"),
             writes_inter_first=c.get("greedy_writes_intermediate_first"),
             stored_correct=g.get("correct"), stored_admissible=c.get("admissible") if samples else None,
             assert_rate=c.get("assert_rate") if samples else None,
             leak_any=c.get("leak_any") if samples else None,
             admissible=(c.get("admissible") if samples and c.get("assert_rate") is not None else None),
             samples=[dict(text=s.get("text"), leaks=s.get("leaks"), asserts=s.get("asserts_target")) for s in samples],
             greedy_leaks=g.get("leaks"),
             recheck_correct=(first_num_wa(g.get("text") or "") == target))     # local word-aware re-parse, for a mismatch count only
    return r


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--box-dir", default="v2_box", help="box output of the fast pass (lens reads + greedy)")
    ap.add_argument("--box-full-dir", default=None, help="box output of the sampled pass; admissibility is taken from it when given")
    ap.add_argument("--out", default="RESULTS_v2.md")
    args = ap.parse_args()
    BOX_DIR = args.box_dir if os.path.isabs(args.box_dir) else P(args.box_dir)
    BOX_FULL = (args.box_full_dir if os.path.isabs(args.box_full_dir) else P(args.box_full_dir)) if args.box_full_dir else None
    OUT_PATH = args.out if os.path.isabs(args.out) else P(args.out)
    def rel(p):
        r = os.path.relpath(p, HERE)
        return p if r.startswith("..") else r

    out = []
    W = out.append
    items = load_items()
    n_items = len(items)

    # ---- provenance ---------------------------------------------------------------------------
    freeze_lines, freeze_ok = [], True
    for line in open(P("FREEZE.sha256"), encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        h, rest = line.split(None, 1); fn = rest.split()[0]
        ok = os.path.exists(P(fn)) and sha256_file(P(fn)) == h
        freeze_ok &= ok
        freeze_lines.append(f"| `{fn}` | `{h[:16]}…` | {'PASS' if ok else 'FAIL'} |")

    src_counts = OrderedDict()
    src_counts["v2_api/raw_A"] = (len(glob.glob(P("v2_api", "raw_A", "*.json"))), 2 * n_items)
    src_counts["v2_api/raw_C"] = (len(glob.glob(P("v2_api", "raw_C", "*.json"))), n_items)
    src_counts["v2_api/raw_B"] = (len(glob.glob(P("v2_api", "raw_B", "*.json"))), n_items)
    src_counts["v2_api/raw_B2"] = (len(glob.glob(P("v2_api", "raw_B2", "*.json"))), n_items)
    for d in LENS_DIRS:
        src_counts[f"{rel(BOX_DIR)}/{d}"] = (len(glob.glob(os.path.join(BOX_DIR, d, "*.json"))), 2 * n_items)
    src_counts[f"{rel(BOX_DIR)}/continuations"] = (len(glob.glob(os.path.join(BOX_DIR, "continuations", "*.json"))), 2 * n_items)
    if BOX_FULL:
        src_counts[f"{rel(BOX_FULL)}/continuations"] = (len(glob.glob(os.path.join(BOX_FULL, "continuations", "*.json"))), 2 * n_items)
        for d in LENS_DIRS:
            src_counts[f"{rel(BOX_FULL)}/{d}"] = (len(glob.glob(os.path.join(BOX_FULL, d, "*.json"))), 2 * n_items)
    manifest = load_json(os.path.join(BOX_DIR, "manifest.json")) if os.path.exists(os.path.join(BOX_DIR, "manifest.json")) else None
    manifest_full = load_json(os.path.join(BOX_FULL, "manifest.json")) if BOX_FULL and os.path.exists(os.path.join(BOX_FULL, "manifest.json")) else None
    missing_sources = [k for k, (n, _) in src_counts.items() if n == 0]
    box_present = any(src_counts[f"{rel(BOX_DIR)}/{d}"][0] for d in LENS_DIRS)

    # ---- load arm A ---------------------------------------------------------------------------
    A = {}                                    # (set,name,variant) -> record
    for it in items:
        for variant in ("space", "nospace"):
            f = P("v2_api", "raw_A", f"{it['set']}_{it['name']}_{variant}.json")
            if os.path.exists(f):
                r = read_lens_file(f, it, API_TOPN)
                if r:
                    r.pop("doc", None); A[(it["set"], it["name"], variant)] = r

    # ---- load box lenses + continuations ------------------------------------------------------
    BOX = {d: {} for d in LENS_DIRS}          # lens -> (set,name,variant) -> dict(top8=rec, top64=rec, exact=rec)
    BOX_CONT = {}                             # (set,name,variant) -> continuation record (greedy from box dir; gate from full dir)
    box_decoy_override = 0
    for it in items:
        for variant in ("space", "nospace"):
            key = (it["set"], it["name"], variant)
            tag = f"{it['set']}_{it['name']}_{variant}.json"
            fc = os.path.join(BOX_DIR, "continuations", tag)
            if os.path.exists(fc):
                c = load_json(fc)
                if c:
                    BOX_CONT[key] = box_cont_record(c)
            for d in LENS_DIRS:
                fb = os.path.join(BOX_DIR, d, tag)
                if not os.path.exists(fb):
                    continue
                doc = load_json(fb)
                top8 = read_lens_file(fb, it, API_TOPN, doc) if doc else None
                if not top8:
                    continue
                top8.pop("doc", None)
                top64 = read_lens_file(fb, it, 64, doc); top64.pop("doc", None)
                exact = read_box_exact(doc, it)
                bi = doc.get("_item", {})
                if bi.get("decoy") and str(bi["decoy"]) != (it["decoy"] or ""):
                    it["decoy"] = str(bi["decoy"]); it["decoy_source"] = bi.get("decoy_source") or "box"; it["S_decoy"] = syn(it["decoy"])
                    box_decoy_override += 1
                    top8 = read_lens_file(fb, it, API_TOPN, doc); top8.pop("doc", None)          # re-score with the box decoy
                    top64 = read_lens_file(fb, it, 64, doc); top64.pop("doc", None)
                    exact = read_box_exact(doc, it)
                BOX[d][key] = dict(top8=top8, top64=top64, exact=exact)
                if key not in BOX_CONT and doc.get("continuations"):
                    BOX_CONT[key] = box_cont_record(doc["continuations"])
            if BOX_FULL:                       # admissibility from the sampled pass
                ff = os.path.join(BOX_FULL, "continuations", tag)
                cf = load_json(ff) if os.path.exists(ff) else None
                if cf is None:
                    for d in LENS_DIRS:
                        fl = os.path.join(BOX_FULL, d, tag)
                        if os.path.exists(fl):
                            dl = load_json(fl)
                            if dl and dl.get("continuations"):
                                cf = dl["continuations"]; break
                if cf:
                    full = box_cont_record(cf)
                    if key in BOX_CONT:
                        for k in ("assert_rate", "leak_any", "admissible", "samples", "stored_admissible"):
                            BOX_CONT[key][k] = full[k]
                        BOX_CONT[key]["gate_source"] = "box-full"
                    else:
                        BOX_CONT[key] = full; full["gate_source"] = "box-full"

    # ---- continuations: API arm C --------------------------------------------------------------
    API_C = {}
    for it in items:
        f = P("v2_api", "raw_C", f"{it['set']}_{it['name']}.json")
        if os.path.exists(f):
            rec = load_json(f)
            if rec:
                API_C[(it["set"], it["name"])] = rec

    def cont_for(it, variant):
        key = (it["set"], it["name"], variant)
        if key in BOX_CONT:
            return BOX_CONT[key]
        if (it["set"], it["name"]) in API_C:
            return api_c_record(API_C[(it["set"], it["name"])], variant)
        return None

    def gate_for(it):
        """Item-level admissibility record (space variant): box sampled pass preferred, else API arm C."""
        b = BOX_CONT.get((it["set"], it["name"], "space"))
        if b and b["assert_rate"] is not None:
            return b
        if (it["set"], it["name"]) in API_C:
            a = api_c_record(API_C[(it["set"], it["name"])], "space")
            if a and a["assert_rate"] is not None:
                return a
        return None

    for it in items:
        it["cont"] = {v: cont_for(it, v) for v in ("space", "nospace")}
        it["gate"] = gate_for(it)
        it["admissible"] = it["gate"]["admissible"] if it["gate"] else None

    n_gated = sum(1 for it in items if it["gate"])
    n_adm = sum(1 for it in items if it["admissible"])
    h12_testable = n_adm >= 30

    # recomputation bookkeeping (API flags vs word-aware; box greedy.correct vs local re-parse)
    api_recs = [(it, api_c_record(API_C[(it["set"], it["name"])], "space")) for it in items if (it["set"], it["name"]) in API_C]
    api_corr_changed = [(it["name"], r["stored_correct"], r["correct"]) for it, r in api_recs if r["stored_correct"] != r["correct"]]
    api_adm_changed = [(it["name"], r["stored_admissible"], r["admissible"]) for it, r in api_recs if r["stored_admissible"] != r["admissible"]]
    box_corr_mismatch = [k for k, v in BOX_CONT.items() if v.get("correct") is not None and v.get("recheck_correct") is not None and v["correct"] != v["recheck_correct"]]

    # ---- arm B / B2 secondary ------------------------------------------------------------------
    B = dict(n=0, correct=0, wrong=0, undetermined=0)
    for f in sorted(glob.glob(P("v2_api", "raw_B", "*.json"))):
        d = load_json(f)
        if not d:
            continue
        B["n"] += 1
        fn = first_num_wa(d["_item"].get("generated", ""))
        if fn is None:
            B["undetermined"] += 1
        elif fn == norm_target(d["_item"]["target"]):
            B["correct"] += 1
        else:
            B["wrong"] += 1
    B2 = dict(n=0, last_int=0, first_after_eq=0, wa=0)
    for f in sorted(glob.glob(P("v2_api", "raw_B2", "*.json"))):
        d = load_json(f)
        if not d:
            continue
        B2["n"] += 1; B2["last_int"] += bool(d["_item"].get("correct_last_int")); B2["first_after_eq"] += bool(d["_item"].get("correct_first_after_eq"))
        nums = numbers_in(d["_item"].get("generated", "")); B2["wa"] += bool(nums) and nums[-1] == norm_target(d["_item"]["target"])

    # ---- column selectors ----------------------------------------------------------------------
    COLS = ("unfiltered", "correct-only", "admissible-only")

    def col_filter(col, variant):
        if col == "unfiltered":
            return lambda it: True
        if col == "correct-only":
            return lambda it: bool(it["cont"][variant] and it["cont"][variant]["correct"] is True)
        if col == "admissible-only":
            return lambda it: it["admissible"] is True
        raise ValueError(col)

    def recs(source, setname, variant, col, field="num"):
        """[(item, record, rank)] for source 'A' or ('box', lens, 'top8'|'top64'|'exact')."""
        f = col_filter(col, variant)
        rows = []
        for it in items:
            if setname != "all" and it["set"] != setname:
                continue
            if not f(it):
                continue
            key = (it["set"], it["name"], variant)
            if source == "A":
                r = A.get(key)
            else:
                _, lens, kind = source
                r = BOX[lens].get(key, {}).get(kind)
            if r is None or r.get(field) is None:
                continue
            rows.append((it, r, r[field]))
        return rows

    def block(rows):
        vals = [v for _, _, v in rows]
        n = len(vals)
        if n == 0:
            return dict(n=0)
        k1, k3, k5, k8 = (passk(vals, k) for k in (1, 3, 5, 8))
        pairs = [(v, r["un_median"]) for _, r, v in rows if r.get("un_median") is not None]
        w, l, t, p = sign_test(pairs)
        return dict(n=n, k1=k1, k3=k3, k5=k5, k8=k8, p1=k1 / n, p3=k3 / n, cp1=cp(k1, n), cp3=cp(k3, n), mrr=mrr(vals),
                    sign=(w, l, t, p), n_sign=len(pairs))

    def table3(source, setname, variant="space", field="num"):
        rows = ["| column | n | pass@1 [CP] | pass@3 [CP] | pass@5 | pass@8 | MRR | sign test better/worse/tie, two-sided p |",
                "|---|---|---|---|---|---|---|---|"]
        blocks = {}
        for col in COLS:
            b = block(recs(source, setname, variant, col, field)); blocks[col] = b
            if b["n"] == 0:
                rows.append(f"| {col} | 0 | – | – | – | – | – | – |")
            else:
                rows.append(f"| {col} | {b['n']} | {fmt_rate(b['k1'], b['n'])} | {fmt_rate(b['k3'], b['n'])} | {b['k5']/b['n']:.2f} | "
                            f"{b['k8']/b['n']:.2f} | {b['mrr']:.2f} | {b['sign'][0]}/{b['sign'][1]}/{b['sign'][2]}, p={b['sign'][3]:.1e} |")
        return rows, blocks

    def lens_ok(lens):
        return bool(BOX[lens]) and (lens != NEEL or c2_reportable)

    # ---- C1-reproduces-A gate (needed before H4) ----------------------------------------------
    gate = {}
    for variant in ("space", "nospace"):
        both = [(it, A[(it["set"], it["name"], variant)], BOX["hosted_n1000"][(it["set"], it["name"], variant)]["top8"])
                for it in items if (it["set"], it["name"], variant) in A and (it["set"], it["name"], variant) in BOX["hosted_n1000"]]
        if not both:
            gate[variant] = None
            continue
        a1 = sum(a["num"] <= 1 for _, a, _ in both); c1 = sum(c["num"] <= 1 for _, _, c in both)
        c1_t64 = sum(BOX["hosted_n1000"][(it["set"], it["name"], variant)]["top64"]["num"] <= 1 for it, _, _ in both)
        agree = sum((a["num"] <= 1) == (c["num"] <= 1) for _, a, c in both)
        a1_64 = sum(a["frozen_all64_num"] <= 1 for _, a, _ in both); c1_64 = sum(c["frozen_all64_num"] <= 1 for _, _, c in both)
        disagree = [(it["set"] + "_" + it["name"], a["num"], c["num"]) for it, a, c in both if (a["num"] <= 1) != (c["num"] <= 1)]
        gate[variant] = dict(n=len(both), a1=a1, c1=c1, c1_top64=c1_t64, diff=abs(a1 - c1), agree=agree, a1_64=a1_64, c1_64=c1_64,
                             passed=abs(a1 - c1) <= 3, disagree=disagree)
    gate_passed = bool(gate.get("space") and gate["space"]["passed"])
    c2_reportable = box_present and gate_passed and bool(BOX[NEEL])
    H4 = {}

    # ================================================================================ REPORT
    W("# RESULTS_v2 — order-of-operations lens eval v2 on Qwen3.6-27B, scored against the frozen preregistration")
    W("")
    W(f"Produced by `score_v2.py` (sha256 `{sha256_file(os.path.abspath(__file__))[:16]}…`; box dir `{rel(BOX_DIR)}`"
      + (f", box full dir `{rel(BOX_FULL)}`" if BOX_FULL else ", no --box-full-dir") + "), stdlib only, no network. "
      "Every number below is computed from the files listed here; the prereg and addenda are quoted, not paraphrased. "
      "The addendum-5 C2 lens is a **5-prompt fit** and is labelled **neel5 (5-prompt fit)** throughout; its on-disk directory is `neel25/`. No number here is an n=25 result.")
    W("")
    W("## 1. Data provenance")
    W("")
    W("| source | files found | expected | status |")
    W("|---|---|---|---|")
    for k, (n, exp) in src_counts.items():
        st = "MISSING" if n == 0 else ("complete" if n >= exp else f"partial ({n/exp:.0%})")
        W(f"| `{k}` | {n} | {exp} | {st} |")
    W(f"| `{rel(BOX_DIR)}/manifest.json` | {'present' if manifest else 'absent'} | 1 | {'present' if manifest else 'MISSING'} |")
    if BOX_FULL:
        W(f"| `{rel(BOX_FULL)}/manifest.json` | {'present' if manifest_full else 'absent'} | 1 | {'present' if manifest_full else 'MISSING'} |")
    W("")
    if missing_sources:
        W("Missing sources: " + ", ".join(f"`{m}`" for m in missing_sources)
          + (" — the box arm (C1/C2, H4, exact ranks, box leak gate) is not scored; every box-only section below says so" if not box_present else "") + ".")
    for lab, m in (("Box manifest", manifest), ("Box full-pass manifest", manifest_full)):
        if m:
            ml = m.get("lenses", {})
            W(f"{lab}: " + "; ".join(f"{LENS_DIRS.get(k, k)}: n_prompts={v.get('n_prompts')}, layers {v.get('source_layers', ['?'])[0]}..{v.get('source_layers', ['?'])[-1]}, sha256 {str(v.get('sha256', ''))[:12]}…" for k, v in ml.items())
              + f"; model config sha {str(m.get('model_config_sha256', ''))[:12]}…; torch {m.get('torch')}; GPU {m.get('gpu')}; started {m.get('started_utc')}, finished {m.get('finished_utc', '(not finished)')}"
              + f"; sampling {m.get('sampling', {}).get('n')}×T={m.get('sampling', {}).get('temperature')} top_p={m.get('sampling', {}).get('top_p')}.")
            fm = m.get("form_misses")
            if fm:
                W(f"{lab} `form_misses` (synonym strings with no single-token form in the tokenizer): {json.dumps(fm, ensure_ascii=False)[:600]}")
    if BAD_FILES:
        W("")
        W(f"Unreadable / malformed files skipped ({len(BAD_FILES)}): " + "; ".join(f"`{f}` ({e})" for f, e in BAD_FILES[:10]) + (" …" if len(BAD_FILES) > 10 else ""))
    W("")
    W("Frozen-file re-verification (`FREEZE.sha256`):")
    W("")
    W("| file | sha256 | check |"); W("|---|---|---|")
    out.extend(freeze_lines)
    W("")
    W(f"**FREEZE re-verification: {'PASS' if freeze_ok else 'FAIL'}** ({len(freeze_lines)} files).")
    W("")
    n_paper_dec = sum(1 for it in items if it["set"] == "paper" and it["decoy"])
    n_new_dec = sum(1 for it in items if it["set"] == "new" and it["decoy"])
    n_corr_recs = sum(1 for it in items if it["cont"]["space"])
    W(f"Items: 55 paper + 50 held-out = {n_items}. Arm A reads present: {len(A)}/{2*n_items} (space {sum(1 for k in A if k[2]=='space')}, nospace {sum(1 for k in A if k[2]=='nospace')}). "
      f"Correctness records (space): {n_corr_recs} items ({sum(1 for it in items if it['cont']['space'] and it['cont']['space']['source']=='box')} box, {sum(1 for it in items if it['cont']['space'] and it['cont']['space']['source']=='api')} API); API arm C files {len(API_C)} ({sum(1 for k in API_C if k[0]=='paper')} paper, {sum(1 for k in API_C if k[0]=='new')} held-out). "
      f"Leak gate: {n_gated} items gated ({sum(1 for it in items if it['gate'] and it['gate']['source']=='box')} box, {sum(1 for it in items if it['gate'] and it['gate']['source']=='api')} API), {n_adm} admissible. "
      f"Decoys: {n_new_dec} held-out (from the item file), {n_paper_dec} paper ("
      + (f"{box_decoy_override} taken from the box file's `_item.decoy`; the rest " if box_decoy_override else "")
      + "derived by `box/read_items.py::derive_decoy`, the wrong-precedence rule of RESEARCH_NOTE.md — `score_orderops.py` carries no decoy dict; the 14 v1 pairs lived in the session-transcript inline scorer).")
    W("")
    W("Correctness normalisation: targets and generated text are passed through a number-word map (zero..ninety-nine, hyphen or space) before comparison. "
      f"The API arm-C flags were computed by digit-string equality, which mis-scores the six word-target paper items ({', '.join(WORD_TARGET_ITEMS)}); "
      f"they were recomputed here from the stored text: `correct_greedy` changed for {len(api_corr_changed)} of {len(api_recs)} API items"
      + (" (" + ", ".join(f"{n}: {a}→{b}" for n, a, b in api_corr_changed[:8]) + (" …" if len(api_corr_changed) > 8 else "") + ")" if api_corr_changed else "")
      + f"; `admissible` changed for {len(api_adm_changed)}"
      + (" (" + ", ".join(f"{n}: {a}→{b}" for n, a, b in api_adm_changed[:8]) + ")" if api_adm_changed else "") + ". "
      + (f"Box `greedy.correct` is trusted as word-aware; a local re-parse disagrees on {len(box_corr_mismatch)} of {len(BOX_CONT)} box continuations" + (": " + ", ".join("_".join(k) for k in box_corr_mismatch[:6]) if box_corr_mismatch else "") + "." if BOX_CONT else ""))
    W("")
    W("Other schema notes: (i) layer 63 of every 64-layer response is the model's own next-token top-8 (no Jacobian): the lens rank here is the min over layers 0–62, and "
      f"the frozen scorer's `best_rank` (all 64 layers) is shown once as a sensitivity line. (ii) The nospace readout token is ` =` for {sum(1 for k,r in A.items() if k[2]=='nospace' and r['last_tok']==' =')} prompts and "
      f"` equals` for {sum(1 for k,r in A.items() if k[2]=='nospace' and r['last_tok']==' equals')} (the word-form paper items). "
      "(iii) The API gate ran on the with-space variant only; item admissibility is a per-item property applied to both variants.")
    W("")

    # ================================================================================ H1
    W("## 2. Hypotheses")
    W("")
    W("### H1 — paper set, hosted lens, correctness-filtered")
    W("")
    W("> **Frozen rule:** \"on items the model answers correctly, number pass@1 ≥ 0.35 and pass@3 ≥ 0.65 with the hosted lens; paired sign test p < 0.01. Rule: both thresholds met → \\\"replicates\\\"; else \\\"does not replicate as scored\\\".\"")
    W("> Jeffrey: pass@1 = 0.50, pass@3 = 0.75 on correctly answered items. Agent: 0.45 / 0.78.")
    W("")
    W("Arm A (API, hosted n1000 lens, `space` variant, 55 paper items):")
    W("")
    rows, H1 = table3("A", "paper")
    out.extend(rows)
    W("")

    def h1_verdict(b):
        if b["n"] == 0:
            return "no items in this column"
        ok = b["p1"] >= 0.35 and b["p3"] >= 0.65 and b["sign"][3] < 0.01
        return ("**replicates**" if ok else "**does not replicate as scored**") + f" (pass@1 {b['p1']:.2f} vs 0.35, pass@3 {b['p3']:.2f} vs 0.65, p={b['sign'][3]:.1e} vs 0.01)"

    n_paper_corr_recs = sum(1 for it in items if it["set"] == "paper" and it["cont"]["space"])
    W(f"- Verdict on the frozen column (correct-only, n={H1['correct-only']['n']} of {n_paper_corr_recs} paper items with a correctness record): {h1_verdict(H1['correct-only'])}.")
    W(f"- Unfiltered (n={H1['unfiltered']['n']}): {h1_verdict(H1['unfiltered'])}.")
    W(f"- Admissible-only (addendum-2 headline, n={H1['admissible-only']['n']}): {h1_verdict(H1['admissible-only'])}. "
      + (f"Addendum-2 rule: {n_adm} admissible of {n_items} (< 30) → **H1/H2 not testable as designed** on the admissible column" + (f" (gate incomplete: {n_gated}/{n_items} gated)" if n_gated < n_items else "") + "." if not h12_testable else f"Addendum-2 rule: {n_adm} admissible of {n_items} (≥ 30) → testable as designed."))
    if box_present and BOX["hosted_n1000"]:
        W("")
        W("Box C1 (hosted lens applied locally, same items; " + ("pipeline trusted by the C1-vs-A gate" if gate_passed else "**C1-vs-A gate FAILED — local pipeline not trusted; shown for the record only**") + "):")
        W("")
        for kind, lab in (("top8", "top-8-sliced ranks (API parity)"), ("exact", "exact full-vocab ranks")):
            rows, blk = table3(("box", "hosted_n1000", kind), "paper")
            W(f"*{lab}*"); W(""); out.extend(rows); W("")
            W(f"- correct-only verdict ({lab}): {h1_verdict(blk['correct-only'])}.")
    W("")

    # ================================================================================ H2
    W("### H2 — held-out 50")
    W("")
    W("> **Frozen rule:** \"pass@1 and pass@3 on items_new50 within the 95% CP interval of the paper-set values. Rule: inside → \\\"generalises beyond the paper's items\\\"; outside on the low side → \\\"paper items favourable\\\"; report the numbers either way.\"")
    W("> Jeffrey: held-out pass@1 = 0.50, pass@3 = 0.75 (\"should match the paper's numbers\"). Agent: 0.40 / 0.72.")
    W("")
    W("Arm A (API, hosted lens, `space` variant, 50 held-out items):")
    W("")
    rows, H2 = table3("A", "new")
    out.extend(rows)
    W("")
    H2V = {}
    for col in COLS:
        b, pb = H2[col], H1[col]
        if b["n"] == 0 or pb["n"] == 0:
            n_new_rec = sum(1 for it in items if it["set"] == "new" and it["cont"]["space"]); n_new_gate = sum(1 for it in items if it["set"] == "new" and it["gate"])
            why = ""
            if col == "correct-only" and b["n"] == 0:
                why = " — no held-out item has a correctness record yet" if n_new_rec == 0 else f" — {n_new_rec} held-out items have a record, none scored correct"
            elif col == "admissible-only" and b["n"] == 0:
                why = " — no held-out item has been gated yet" if n_new_gate == 0 else f" — {n_new_gate} held-out items gated, none admissible"
            W(f"- {col}: not scorable (held-out n={b['n']}, paper n={pb['n']}){why}.")
            continue
        in1 = pb["cp1"][0] <= b["p1"] <= pb["cp1"][1]; in3 = pb["cp3"][0] <= b["p3"] <= pb["cp3"][1]
        if in1 and in3:
            v = "**generalises beyond the paper's items**"
        elif (not in1 and b["p1"] < pb["cp1"][0]) or (not in3 and b["p3"] < pb["cp3"][0]):
            v = "**paper items favourable** (outside on the low side)"
        else:
            v = "outside on the **high** side (held-out easier than the paper's items)"
        H2V[col] = (in1, in3, v)
        W(f"- {col}: held-out pass@1 {b['p1']:.2f} vs paper CP [{pb['cp1'][0]:.2f}, {pb['cp1'][1]:.2f}] → {'inside' if in1 else 'outside'}; "
          f"pass@3 {b['p3']:.2f} vs paper CP [{pb['cp3'][0]:.2f}, {pb['cp3'][1]:.2f}] → {'inside' if in3 else 'outside'} → {v}.")
    W(f"- Addendum-2 headline column (admissible-only): " + ("testable" if h12_testable else f"**not testable as designed** ({n_adm} admissible < 30)") + ".")
    if box_present and BOX["hosted_n1000"]:
        W("")
        W("Box C1 on the held-out 50 (exact ranks):")
        W("")
        rows, _ = table3(("box", "hosted_n1000", "exact"), "new"); out.extend(rows)
    W("")

    # ================================================================================ H3
    W("### H3 — carrier of two-digit hits")
    W("")
    W("> **Frozen rule:** \"among two-digit intermediates with a rank-1 hit, ≥ 2/3 are CJK-only or CJK+word. Rule as stated.\"")
    W("> Jeffrey: 0.90 of two-digit rank-1 hits involve the Chinese numeral. Agent: 0.85.")
    W("")

    def carrier_stats(source, setname, variant="space"):
        rows = [(it, r, v) for it, r, v in recs(source, setname, variant, "unfiltered") if it["two_digit"]]
        hits = [(it, r) for it, r, v in rows if v == 1]
        cats = Counter()
        for it, r in hits:
            fs = {f for f, rk in r["forms"].items() if rk == 1}
            if fs == {"cjk"}: cats["CJK-only"] += 1
            elif fs == {"cjk", "word"}: cats["CJK+word"] += 1
            elif fs == {"word"}: cats["word-only"] += 1
            elif "cjk" in fs: cats["CJK+other"] += 1
            elif "digit" in fs: cats["digit"] += 1
            else: cats["other"] += 1
        num = cats["CJK-only"] + cats["CJK+word"]
        return len(rows), len(hits), cats, num

    for setname in ("all", "paper", "new"):
        n2, nh, cats, num = carrier_stats("A", setname)
        frac = f"{num}/{nh} = {num/nh:.2f}" if nh else "n/a (no rank-1 two-digit hits)"
        verdict = ("**holds** (≥ 2/3)" if nh and num / nh >= 2 / 3 else ("**fails** (< 2/3)" if nh else "not scorable"))
        W(f"- Arm A, {setname} ({n2} two-digit items, {nh} at rank 1): CJK-only or CJK+word {frac}; breakdown {dict(cats)} → {verdict if setname=='all' else verdict.replace('**','')}.")
    for lens in LENS_DIRS:
        if lens_ok(lens):
            n2, nh, cats, num = carrier_stats(("box", lens, "exact"), "all")
            W(f"- Box {LENS_DIRS[lens]} (exact ranks), all: {n2} two-digit, {nh} at rank 1; CJK-only or CJK+word {num}/{nh} = {num/nh:.2f}; {dict(cats)}." if nh else f"- Box {LENS_DIRS[lens]}: no rank-1 two-digit hits.")
    W("")

    # ================================================================================ H4
    W(f"### H4 — lens quality (box only: hosted (n1000) vs {LENS_DIRS[NEEL]} on the same forward pass)")
    W("")
    W("> **Frozen rule:** \"25-prompt lens pass@1 on the 105 items ≥ half of the 1000-prompt lens pass@1 → \\\"effect survives Neel's lens; lens quality does not explain the null\\\"; < half → \\\"lens quality is a sufficient explanation\\\".\"")
    W("> **Addendum 5 (asymmetric reading, n=5):** \"If the effect survives it, H4 is answered in the direction 'lens quality does not explain the null' with more force than the frozen rule required. If the effect fails under it, H4 is inconclusive, not negative, and the note says 'not tested at n=25'.\"")
    W("> Jeffrey: ratio ≈ 0.9. Agent: 0.6.")
    W("")
    if not box_present:
        W(f"**Not scored: `{rel(BOX_DIR)}/` has no lens reads.** H4 needs the two box lenses on one forward pass; nothing on the API side substitutes. Verdict: **inconclusive (not run)**.")
    elif not BOX[NEEL]:
        W(f"**Not scored: `{rel(BOX_DIR)}/{NEEL}/` (the {LENS_DIRS[NEEL]} reads) is absent.** Verdict: **inconclusive (not run)**.")
    elif not gate_passed:
        W("**C2 not reported:** the C1-reproduces-A gate failed (section 5), so per the frozen arm-C rule the local pipeline is not trusted. Verdict: **inconclusive (pipeline not trusted)**.")
    else:
        W(f"| ranks | column | hosted (n1000) pass@1 | {LENS_DIRS[NEEL]} pass@1 | ratio neel5/hosted | rule (≥ 0.5) |")
        W("|---|---|---|---|---|---|")
        for kind, lab in (("top8", "top-8-sliced (API parity)"), ("exact", "exact full-vocab")):
            for col in COLS:
                hb = block(recs(("box", "hosted_n1000", kind), "all", "space", col)); nb = block(recs(("box", NEEL, kind), "all", "space", col))
                if hb["n"] == 0 or nb["n"] == 0 or hb["k1"] == 0:
                    W(f"| {lab} | {col} | {fmt_rate(hb.get('k1',0), hb['n']) if hb['n'] else '–'} | {fmt_rate(nb.get('k1',0), nb['n']) if nb['n'] else '–'} | – | not scorable |")
                    continue
                ratio = (nb["k1"] / nb["n"]) / (hb["k1"] / hb["n"]); H4[(kind, col)] = ratio
                W(f"| {lab} | {col} | {fmt_rate(hb['k1'], hb['n'])} | {fmt_rate(nb['k1'], nb['n'])} | {ratio:.2f} | {'survives' if ratio >= 0.5 else 'fails'} |")
        jo = [r["exact"]["num_jacobian_only"] for k, r in BOX[NEEL].items() if k[2] == "space" and r.get("exact") and r["exact"].get("num_jacobian_only") is not None]
        if jo:
            W("")
            W(f"{LENS_DIRS[NEEL]} exact pass@1 restricted to its Jacobian-carrying layers (0–61; layer 62 is the fit target, read without a Jacobian): {fmt_rate(passk(jo,1), len(jo))}.")
        W("")
        key = ("exact", "unfiltered")
        if key in H4:
            r = H4[key]
            if r >= 0.5:
                W(f"**Verdict (exact ranks, unfiltered, ratio {r:.2f} ≥ 0.5): the effect survives the 5-prompt lens → \"lens quality does not explain the null\", with more force than the frozen rule required (addendum 5). The lens is a 5-prompt fit, not n=25.**")
            else:
                W(f"**Verdict (exact ranks, unfiltered, ratio {r:.2f} < 0.5): the effect fails under the 5-prompt lens → H4 is inconclusive, not negative (addendum 5); not tested at n=25.**")
            if ("top8", "unfiltered") in H4:
                W(f"Top-8-sliced ratio (API parity): {H4[('top8','unfiltered')]:.2f}. Admissible-only column: " + (f"ratio {H4[('exact','admissible-only')]:.2f}" if ("exact", "admissible-only") in H4 else "not scorable") + ".")
    W("")

    # ================================================================================ H5
    W("### H5 — decoy (wrong-precedence value)")
    W("")
    W("> **Frozen rule:** \"correct intermediate beats decoy in > 60% of decidable items, pooled over 105.\"")
    W("> Jeffrey: win rate ≈ 0.85 of decided items (v1: 9/11 = 0.82). Agent: 0.70.")
    W("")

    def decoy_stats(source, setname="all", variant="space", col="unfiltered"):
        rows = [(it, r) for it, r, v in recs(source, setname, variant, col) if it["decoy"] and r.get("decoy") is not None]
        w = sum(r["num"] < r["decoy"] for _, r in rows); l = sum(r["num"] > r["decoy"] for _, r in rows); t = len(rows) - w - l
        both_absent = sum(r["num"] >= ABSENT and r["decoy"] >= ABSENT for _, r in rows)
        unreach = sum(1 for it, _ in rows if int(it["decoy"]) >= 100)
        return dict(n=len(rows), w=w, l=l, t=t, both_absent=both_absent, unreach=unreach)

    def decoy_line(lab, s):
        if s["n"] == 0:
            return f"- {lab}: no items with a decoy."
        dec = s["w"] + s["l"]
        r_dec = f"{s['w']}/{dec} = {s['w']/dec:.2f} [{cp(s['w'],dec)[0]:.2f}, {cp(s['w'],dec)[1]:.2f}]" if dec else "n/a"
        r_all = f"{s['w']}/{s['n']} = {s['w']/s['n']:.2f}"
        v_dec = ("> 0.60 → **holds**" if dec and s["w"] / dec > 0.6 else "≤ 0.60 → **fails**") if dec else "not scorable"
        v_all = ("> 0.60 → holds" if s["w"] / s["n"] > 0.6 else "≤ 0.60 → fails")
        return (f"- {lab}: {s['n']} items with a decoy; win/loss/tie {s['w']}/{s['l']}/{s['t']} (ties with both absent from the top-8: {s['both_absent']}; "
                f"decoys ≥ 100 with no single-token form: {s['unreach']}). Win rate of decided items {r_dec} → {v_dec}; win rate of all decoy items {r_all} → {v_all}.")

    W(decoy_line("Arm A, pooled 105, `space`", decoy_stats("A")))
    W(decoy_line("Arm A, paper", decoy_stats("A", "paper")))
    W(decoy_line("Arm A, held-out", decoy_stats("A", "new")))
    W(decoy_line("Arm A, pooled, correct-only", decoy_stats("A", col="correct-only")))
    W(decoy_line("Arm A, pooled, admissible-only", decoy_stats("A", col="admissible-only")))
    for lens in LENS_DIRS:
        if lens_ok(lens):
            W(decoy_line(f"Box {LENS_DIRS[lens]}, exact ranks, pooled", decoy_stats(("box", lens, "exact"))))
    W("")
    W("The frozen text says \"decidable items\"; Jeffrey's prediction is phrased over *decided* items (ties excluded, v1's 9/11). Both denominators are given; the decided-items rate is the one compared with the prediction.")
    W("")

    # ================================================================================ H6
    W("### H6 — readout position (space vs nospace)")
    W("")
    W("> **Frozen rule:** \"every item is also read with the trailing space removed (`prompt_nospace`), readout at the `=` token. Rule: if pass@1 falls by more than half relative to the with-space form → \\\"readout position is a sufficient explanation for a null\\\"; if within half → \\\"position does not explain it\\\".\"")
    W("> Jeffrey: ≈ 1.0 (\"it's always going to be in either the =_ token or the = token itself\"). Agent: 0.25.")
    W("")
    W("Correct-only here = items whose greedy continuation is correct under BOTH variants (each variant carries its own `correct_greedy`); admissible-only is the per-item gate.")
    W("")
    W("| source | set | column | space pass@1 | nospace pass@1 | ratio nospace/space | space pass@3 | nospace pass@3 | both hit / space only / nospace only / neither |")
    W("|---|---|---|---|---|---|---|---|---|")
    H6 = {}

    def h6_row(source, lab, setname, col):
        sp = {(it["set"], it["name"]): v for it, r, v in recs(source, setname, "space", col)}
        ns = {(it["set"], it["name"]): v for it, r, v in recs(source, setname, "nospace", col)}
        keys = sorted(set(sp) & set(ns))
        if not keys:
            return None
        s1 = sum(sp[k] <= 1 for k in keys); n1 = sum(ns[k] <= 1 for k in keys)
        s3 = sum(sp[k] <= 3 for k in keys); n3 = sum(ns[k] <= 3 for k in keys)
        bb = sum(sp[k] <= 1 and ns[k] <= 1 for k in keys); so = sum(sp[k] <= 1 and ns[k] > 1 for k in keys); no = sum(sp[k] > 1 and ns[k] <= 1 for k in keys)
        ratio = (n1 / s1) if s1 else float("nan")
        W(f"| {lab} | {setname} | {col} | {fmt_rate(s1, len(keys))} | {fmt_rate(n1, len(keys))} | {ratio:.2f} | {s3/len(keys):.2f} | {n3/len(keys):.2f} | {bb}/{so}/{no}/{len(keys)-bb-so-no} |")
        return ratio

    for setname in ("all", "paper", "new"):
        for col in COLS:
            r = h6_row("A", "arm A", setname, col)
            if r is not None:
                H6[("A", setname, col)] = r
    for lens in LENS_DIRS:
        if lens_ok(lens):
            for kind in ("top8", "exact"):
                r = h6_row(("box", lens, kind), f"box {LENS_DIRS[lens]} {kind}", "all", "unfiltered")
                if r is not None:
                    H6[(lens, kind)] = r
    W("")
    if ("A", "all", "unfiltered") in H6:
        r = H6[("A", "all", "unfiltered")]
        W(f"**Verdict (arm A, pooled 105, unfiltered): ratio {r:.2f} → " + ("\"readout position is a sufficient explanation for a null\" (fell by more than half)" if r < 0.5 else "\"position does not explain it\" (within half)") + ".**"
          + (f" Correct-only ratio {H6[('A','all','correct-only')]:.2f}." if ("A", "all", "correct-only") in H6 else "")
          + (f" Admissible-only ratio {H6[('A','all','admissible-only')]:.2f}." if ("A", "all", "admissible-only") in H6 else ""))
    W("")

    # ================================================================================ H7
    W("### H7 — digit count and CJK synonyms (held-out 50)")
    W("")
    W("> **Frozen rule:** \"two-digit vs single-digit pass@1 with CJK in the synonym set, and the same contrast with CJK removed. Rule: if the two-digit rate without CJK falls below half of the two-digit rate with CJK, tokenization is a sufficient explanation for two-digit misses.\"")
    W("> Jeffrey: two-digit pass@1 with CJK ≈ 0.50 (± 0.1), without ≈ 0.05. Agent: 0.40 / 0.10.")
    W("")
    W("| source | set | column | two-digit, with CJK | two-digit, no CJK | single-digit, with CJK | single-digit, no CJK | ratio two-digit noCJK/CJK | rule (< 0.5) |")
    W("|---|---|---|---|---|---|---|---|---|")
    H7 = {}

    def h7_row(source, lab, setname, col):
        rows = recs(source, setname, "space", col)
        two = [r for it, r, v in rows if it["two_digit"]]; one = [r for it, r, v in rows if not it["two_digit"]]
        if not two:
            return None
        t1 = passk([r["num"] for r in two], 1); t1n = passk([r["num_nocjk"] for r in two], 1)
        o1 = passk([r["num"] for r in one], 1) if one else 0; o1n = passk([r["num_nocjk"] for r in one], 1) if one else 0
        ratio = (t1n / t1) if t1 else float("nan")
        W(f"| {lab} | {setname} | {col} | {fmt_rate(t1, len(two))} | {fmt_rate(t1n, len(two))} | {fmt_rate(o1, len(one)) if one else '–'} | {fmt_rate(o1n, len(one)) if one else '–'} | {ratio:.2f} | "
          + ("tokenization sufficient" if t1 and ratio < 0.5 else ("not below half" if t1 else "n/a")) + " |")
        return dict(t1=t1, n2=len(two), t1n=t1n, o1=o1, o1n=o1n, n1=len(one), ratio=ratio)

    for setname in ("new", "paper", "all"):
        for col in COLS:
            r = h7_row("A", "arm A", setname, col)
            if r:
                H7[("A", setname, col)] = r
    for lens in LENS_DIRS:
        if lens_ok(lens):
            r = h7_row(("box", lens, "exact"), f"box {LENS_DIRS[lens]} exact", "new", "unfiltered")
            if r:
                H7[(lens, "new")] = r
    W("")
    if ("A", "new", "unfiltered") in H7:
        r = H7[("A", "new", "unfiltered")]
        W(f"**Verdict (arm A, held-out 50, unfiltered): two-digit pass@1 with CJK {r['t1']}/{r['n2']} = {r['t1']/r['n2']:.2f}, without CJK {r['t1n']}/{r['n2']} = {r['t1n']/r['n2']:.2f} (ratio {r['ratio']:.2f}) → "
          + ("\"tokenization is a sufficient explanation for two-digit misses\"" if r["t1"] and r["ratio"] < 0.5 else ("not below half — tokenization alone does not explain two-digit misses" if r["t1"] else "not scorable (no two-digit rank-1 hits with CJK)"))
          + f". Single-digit pass@1 {r['o1']}/{r['n1']} = {r['o1']/r['n1']:.2f} with CJK, {r['o1n']}/{r['n1']} without.**")
    W("")

    # ================================================================================ CONTROLS
    W("## 3. Controls")
    W("")
    W("### 3a. Layer-63 next-token control (addendum 4)")
    W("")
    W("> \"(a) the fraction of rank-1 items whose intermediate is ALSO in the model's top-8 next tokens at the readout position; (b) the 'workspace-only' subset: items where the intermediate reaches rank ≤ 3 at some source layer 0–62 but is absent from the model's top-8 at layer 63 — hits the logit lens would not show; (c) the same for the decoy.\"")
    W("")
    W("| source | set | variant | rank-1 items | of which in L63 top-8 (a) | rank ≤ 3 items | workspace-only (b) | decoy rank-1 | decoy in L63 top-8 | decoy rank ≤ 3 | decoy workspace-only (c) | intermediate in L63 top-8 (all items) | L63 rank 1 (next token IS the intermediate) |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    CTRL = {}

    def ctrl_row(source, lab, setname, variant):
        rows = recs(source, setname, variant, "unfiltered")
        if not rows:
            return None
        r1 = [r for _, r, v in rows if v == 1]; a = sum(r["L63_num"] is not None and r["L63_num"] <= 8 for r in r1)
        r3 = [r for _, r, v in rows if v <= 3]; b = sum(r["L63_num"] is not None and r["L63_num"] > 8 for r in r3)
        d1 = [r for _, r, v in rows if r.get("decoy") is not None and r["decoy"] == 1]; da = sum(r["L63_decoy"] is not None and r["L63_decoy"] <= 8 for r in d1)
        d3 = [r for _, r, v in rows if r.get("decoy") is not None and r["decoy"] <= 3]; db = sum(r["L63_decoy"] is not None and r["L63_decoy"] > 8 for r in d3)
        in63 = sum(r["L63_num"] is not None and r["L63_num"] <= 8 for _, r, _ in rows); l63r1 = sum(r["L63_num"] == 1 for _, r, _ in rows)
        fa = f"{a} ({a/len(r1):.2f})" if r1 else "–"; fb = f"{b} ({b/len(r3):.2f})" if r3 else "–"
        W(f"| {lab} | {setname} | {variant} | {len(r1)} | {fa} | {len(r3)} | {fb} | {len(d1)} | {da} | {len(d3)} | {db} | {in63}/{len(rows)} | {l63r1}/{len(rows)} |")
        wo_names = [it["set"] + "_" + it["name"] for it, r, v in rows if v <= 3 and r["L63_num"] is not None and r["L63_num"] > 8]
        return dict(n=len(rows), r1=len(r1), a=a, r3=len(r3), b=b, d1=len(d1), da=da, d3=len(d3), db=db, in63=in63, wo=wo_names)

    for setname in ("all", "paper", "new"):
        for variant in ("space", "nospace"):
            c = ctrl_row("A", "arm A", setname, variant)
            if c:
                CTRL[("A", setname, variant)] = c
    for lens in LENS_DIRS:
        if lens_ok(lens):
            for kind in ("top8", "exact"):
                c = ctrl_row(("box", lens, kind), f"box {LENS_DIRS[lens]} {kind}", "all", "space")
                if c:
                    CTRL[(lens, kind)] = c
    W("")
    if ("A", "all", "space") in CTRL:
        c = CTRL[("A", "all", "space")]
        W(f"Workspace-only items (arm A, pooled, space; rank ≤ 3 at a source layer, absent from the layer-63 top-8): {len(c['wo'])} — " + ", ".join(f"`{n}`" for n in c["wo"][:40]) + (" …" if len(c["wo"]) > 40 else "") + ".")
        rows = recs("A", "all", "space", "unfiltered")
        k1_63 = sum(r["frozen_all64_num"] <= 1 for _, r, _ in rows); k1_62 = sum(v <= 1 for _, _, v in rows)
        k3_63 = sum(r["frozen_all64_num"] <= 3 for _, r, _ in rows); k3_62 = sum(v <= 3 for _, _, v in rows)
        changed = sum(r["frozen_all64_num"] < v for _, r, v in rows)
        W(f"Sensitivity — the frozen scorer's `best_rank` over all 64 layers (layer 63 included): pass@1 {k1_63}/{len(rows)} and pass@3 {k3_63}/{len(rows)} vs {k1_62}/{len(rows)} and {k3_62}/{len(rows)} over layers 0–62; layer 63 improves the rank for {changed} items. "
          "On the box, `exact` L63 ranks are full-vocab next-token ranks; the top-8 rule above is applied to them for parity.")
    W("")
    W("### 3b. Uninvolved-digit baseline")
    W("")
    W("| source | set | variant | readings (digits × items) | pass@1 | pass@3 | pass@8 | per-item mean fraction hit @1 / @3 | intermediate pass@1 / pass@3 (same items) |")
    W("|---|---|---|---|---|---|---|---|---|")

    def base_row(source, lab, setname, variant):
        rows = recs(source, setname, variant, "unfiltered")
        if not rows:
            return
        base = [u for _, r, _ in rows for u in r["un"]]
        if not base:
            return
        pb1 = [sum(u <= 1 for u in r["un"]) / len(r["un"]) for _, r, _ in rows if r["un"]]; pb3 = [sum(u <= 3 for u in r["un"]) / len(r["un"]) for _, r, _ in rows if r["un"]]
        vals = [v for _, _, v in rows]
        W(f"| {lab} | {setname} | {variant} | {len(base)} | {passk(base,1)/len(base):.2f} | {passk(base,3)/len(base):.2f} | {passk(base,8)/len(base):.2f} | {sum(pb1)/len(pb1):.2f} / {sum(pb3)/len(pb3):.2f} | {passk(vals,1)/len(vals):.2f} / {passk(vals,3)/len(vals):.2f} |")

    for setname in ("all", "paper", "new"):
        for variant in ("space", "nospace"):
            base_row("A", "arm A", setname, variant)
    for lens in LENS_DIRS:
        if lens_ok(lens):
            base_row(("box", lens, "top8"), f"box {LENS_DIRS[lens]} top8", "all", "space")
            rows = recs(("box", lens, "exact"), "all", "space", "unfiltered")
            base = [u for _, r, _ in rows for u in r["un"]]
            if base:
                W(f"| box {LENS_DIRS[lens]} exact | all | space | {len(base)} | median exact rank of uninvolved digits {statistics.median(base):.0f} | – | – | – | median exact rank of the intermediate {statistics.median([v for _,_,v in rows]):.0f} |")
    W("")
    W("### 3c. Decoy win/loss/tie")
    W("")
    ds_all = decoy_stats("A")
    W(f"See H5 above (arm A pooled: win/loss/tie {ds_all['w']}/{ds_all['l']}/{ds_all['t']} of {ds_all['n']} items with a decoy).")
    W("")
    W("### 3d. Target (spoken) rank beside the intermediate (unspoken) rank at the readout")
    W("")
    W("| source | set | variant | n | target pass@1 | target pass@3 | intermediate pass@1 | intermediate pass@3 | target rank < intermediate rank / equal / > | operation pass@1 |")
    W("|---|---|---|---|---|---|---|---|---|---|")

    def tgt_row(source, lab, setname, variant):
        rows = recs(source, setname, variant, "unfiltered")
        if not rows:
            return
        tg = [r["target"] for _, r, _ in rows]; iv = [v for _, _, v in rows]; op = [r["op"] for _, r, _ in rows if r.get("op") is not None]
        lt = sum(a < b for a, b in zip(tg, iv)); eq = sum(a == b for a, b in zip(tg, iv)); gt = len(rows) - lt - eq
        opc = f"{passk(op,1)/len(op):.2f}" if op else "–"
        W(f"| {lab} | {setname} | {variant} | {len(rows)} | {passk(tg,1)/len(rows):.2f} | {passk(tg,3)/len(rows):.2f} | {passk(iv,1)/len(rows):.2f} | {passk(iv,3)/len(rows):.2f} | {lt}/{eq}/{gt} | {opc} |")

    for setname in ("all", "paper", "new"):
        for variant in ("space", "nospace"):
            tgt_row("A", "arm A", setname, variant)
    for lens in LENS_DIRS:
        if lens_ok(lens):
            tgt_row(("box", lens, "exact"), f"box {LENS_DIRS[lens]} exact", "all", "space")
    W("")
    W("Word-target paper items have their target mapped to the number (`fourteen` → the 14 synonyms) for this row; the operation row echoes the prompt symbol at early layers (frozen scorer caveat).")
    W("")

    # ================================================================================ PERSISTENCE
    W("## 4. Persistence descriptives (reported, not hypothesis-tested)")
    W("")
    W("### 4a. Layer span at rank ≤ 3 (source layers 0–62)")
    W("")
    W("| source | set | variant | items | items with any rank ≤ 3 layer | median span (layers, among those) | span ≥ 3 layers | median first / last layer | median best layer (rank-1 items) |")
    W("|---|---|---|---|---|---|---|---|---|")

    def span_row(source, lab, setname, variant):
        rows = recs(source, setname, variant, "unfiltered")
        if not rows:
            return
        sp = [r["span"] for _, r, _ in rows]; hit = [s for s in sp if s[0] > 0]
        bl = [r["best_layer"] for _, r, v in rows if v == 1 and r["best_layer"] is not None]
        if hit:
            W(f"| {lab} | {setname} | {variant} | {len(rows)} | {len(hit)} | {statistics.median([s[0] for s in hit]):.0f} | {sum(s[0] >= 3 for s in hit)} | {statistics.median([s[1] for s in hit]):.0f} / {statistics.median([s[2] for s in hit]):.0f} | {f"{statistics.median(bl):.0f}" if bl else '–'} |")
        else:
            W(f"| {lab} | {setname} | {variant} | {len(rows)} | 0 | – | 0 | – | – |")

    for setname in ("all", "paper", "new"):
        for variant in ("space", "nospace"):
            span_row("A", "arm A", setname, variant)
    for lens in LENS_DIRS:
        if lens_ok(lens):
            span_row(("box", lens, "top8"), f"box {LENS_DIRS[lens]} top8", "all", "space")
            span_row(("box", lens, "exact"), f"box {LENS_DIRS[lens]} exact", "all", "space")
    W("")
    rows = recs("A", "all", "space", "unfiltered")
    if rows:
        dist = Counter(min(r["span"][0], 10) for _, r, _ in rows)
        W("Span histogram (arm A, pooled, space; number of source layers with rank ≤ 3, 10 = ≥ 10): " + ", ".join(f"{k}:{dist[k]}" for k in sorted(dist)) + ".")
    W("")
    W("### 4b. Box: median exact rank by layer (intermediate / target / decoy), every 4 layers")
    W("")
    if not box_present:
        W(f"Not available: `{rel(BOX_DIR)}/` has no lens reads (exact full-vocab ranks exist only on the box).")
    else:
        lens_list = [l for l in LENS_DIRS if lens_ok(l)]
        if not lens_list:
            W("No reportable box lens reads.")
        else:
            hdr = "| layer | " + " | ".join(f"{LENS_DIRS[l]} inter | {LENS_DIRS[l]} target | {LENS_DIRS[l]} decoy" for l in lens_list) + " |"
            W(hdr); W("|---|" + "---|" * (3 * len(lens_list)))
            curves = {}
            for l in lens_list:
                ex = [r["exact"] for k, r in BOX[l].items() if k[2] == "space" and r.get("exact")]
                curves[l] = dict(
                    inter=[median_or_none([e["rbl_num"][L] for e in ex if e["rbl_num"] and L < len(e["rbl_num"])]) for L in range(N_LAYERS)],
                    target=[median_or_none([e["rbl_target"][L] for e in ex if e["rbl_target"] and L < len(e["rbl_target"])]) for L in range(N_LAYERS)],
                    decoy=[median_or_none([e["rbl_decoy"][L] for e in ex if e.get("rbl_decoy") and L < len(e["rbl_decoy"])]) for L in range(N_LAYERS)],
                    n=len(ex))
            for L in list(range(0, 61, 4)) + [62, 63]:
                cells = []
                for l in lens_list:
                    c = curves[l]
                    cells += [f"{c['inter'][L]:.0f}" if c["inter"][L] is not None else "–", f"{c['target'][L]:.0f}" if c["target"][L] is not None else "–", f"{c['decoy'][L]:.0f}" if c["decoy"][L] is not None else "–"]
                tag = " (next-token, no Jacobian)" if L == 63 else ""
                W(f"| {L}{tag} | " + " | ".join(cells) + " |")
            W("")
            W("Items per curve: " + ", ".join(f"{LENS_DIRS[l]} n={curves[l]['n']}" for l in lens_list) + ". Full 64-number curves (median exact rank):")
            for l in lens_list:
                for role in ("inter", "target", "decoy"):
                    W(f"- {LENS_DIRS[l]} {role}: " + " ".join(f"{v:.0f}" if v is not None else "–" for v in curves[l][role]))
    W("")

    # ================================================================================ GATE
    W("## 5. C1-reproduces-A gate (± 3 items on pass@1, `space` variant)")
    W("")
    W("> \"C1 must reproduce arm A's pass@1 within ±3 items or the local pipeline is not trusted and C2 is not reported.\" (logic of `box/check_c1_vs_a.py`; ranks over layers 0–62, top-8-sliced on both sides; the script's own all-64-layer `best_rank_topk` is shown beside it.)")
    W("")
    if not box_present or not BOX["hosted_n1000"]:
        W(f"**Not run: `{rel(BOX_DIR)}/hosted_n1000/` is absent.** C2 cannot be reported until the box files are pulled and this gate passes.")
    else:
        for variant in ("space", "nospace"):
            g = gate[variant]
            if not g:
                W(f"- {variant}: no overlapping items.")
                continue
            W(f"- {variant}: n={g['n']} items; pass@1 A={g['a1']}, C1(top-8)={g['c1']} (C1 top-64: {g['c1_top64']}), |diff|={g['diff']} (tolerance 3); per-item pass@1 agreement {g['agree']}/{g['n']}; "
              f"all-64-layer rule as in check_c1_vs_a.py: A={g['a1_64']}, C1={g['c1_64']} → **{'PASS: local pipeline trusted; C2 may be reported' if g['passed'] else 'FAIL: do NOT report C2'}**" + (" (secondary variant; the gate is decided on `space`)" if variant == "nospace" else "") + ".")
            if g["disagree"]:
                W("  - disagreements (tag, A rank, C1 rank): " + "; ".join(f"{t} {a}/{c}" for t, a, c in g["disagree"][:20]) + (" …" if len(g["disagree"]) > 20 else ""))
    W("")

    # ================================================================================ LEAK GATE
    W("## 6. Leak gate (addenda 2–3)")
    W("")
    W("> admissible ⇔ ≥ 8/10 sampled continuations (T=0.8, 32 new tokens) write the target as the first number AND no continuation (sampled or greedy) writes the intermediate as a number before the target. \"If fewer than 30 of the 105 items are admissible, H1/H2 are 'not testable as designed'.\"")
    W("")

    def gate_summary(lab, getter):
        gs = [(it, getter(it)) for it in items]
        gs = [(it, g) for it, g in gs if g and g["assert_rate"] is not None]
        if not gs:
            W(f"- {lab}: no gated items.")
            return None
        n = len(gs); wif = sum(bool(g["writes_inter_first"]) for _, g in gs); la = sum(bool(g["leak_any"]) for _, g in gs); adm = sum(bool(g["admissible"]) for _, g in gs)
        ar = [g["assert_rate"] for _, g in gs]; hi = sum(a >= 0.8 for a in ar)
        corr = sum(g["correct"] is True for _, g in gs)
        stored_adm = sum(bool(g.get("stored_admissible")) for _, g in gs)
        W(f"- {lab}: n gated {n} ({sum(1 for it,_ in gs if it['set']=='paper')} paper, {sum(1 for it,_ in gs if it['set']=='new')} held-out); greedy correct {corr}/{n}; greedy writes the intermediate first {wif}/{n}; "
          f"assert_rate ≥ 0.8 {hi}/{n} (median assert_rate {statistics.median(ar):.1f}); leak_any {la}/{n}; **admissible {adm}/{n}**"
          + (f" (stored flags said {stored_adm})" if stored_adm != adm else "")
          + (f" — {adm} < 30 → H1/H2 not testable as designed (gate {n}/{n_items} complete)" if adm < 30 else " — ≥ 30 → testable") + ".")
        ex = []
        for it, g in gs:
            for smp in g["samples"]:
                if smp.get("leaks"):
                    ex.append((it["set"] + "_" + it["name"], it["num"], it["target"], (smp["text"] or "").replace("\n", "⏎")[:90]))
                    break
            if len(ex) >= 3:
                break
        for tag, num, tgt, txt in ex:
            W(f"  - leak example `{tag}` (intermediate {num}, target {tgt}): `{txt}`")
        return dict(n=n, wif=wif, la=la, adm=adm, corr=corr)

    G_api = gate_summary("API arm C (with-space variant; unseeded; flags recomputed word-aware)", lambda it: api_c_record(API_C[(it["set"], it["name"])], "space") if (it["set"], it["name"]) in API_C else None)
    G_box = gate_summary("Box sampled pass (space variant; seeds 0–9, top_p 0.95" + (f"; from `{rel(BOX_FULL)}`" if BOX_FULL else "") + ")", lambda it: BOX_CONT.get((it["set"], it["name"], "space")))
    if BOX_CONT and any(v["assert_rate"] is not None for k, v in BOX_CONT.items() if k[2] == "nospace"):
        gate_summary("Box sampled pass (nospace variant)", lambda it: BOX_CONT.get((it["set"], it["name"], "nospace")))
    W("")
    W(f"Headline gate used for the admissible-only columns (box preferred, else API): {n_gated} gated, {n_adm} admissible. "
      f"Correctness records used for the correct-only columns (space): {n_corr_recs} items; correct {sum(1 for it in items if it['cont']['space'] and it['cont']['space']['correct'] is True)}.")
    W("")
    W(f"Secondary, template-mediated (chat endpoint): arm B (8 tokens) n={B['n']}: first number == target {B['correct']}, another number first {B['wrong']}, no number within 8 tokens {B['undetermined']} (addendum 1: mostly undetermined by construction). "
      f"Arm B2 (48 tokens) n={B2['n']}" + (f": correct by last integer {B2['last_int']}, by first integer after '=' {B2['first_after_eq']}, by last number word-aware {B2['wa']}." if B2["n"] else " (no files — `v2_api/raw_B2/` is empty)."))
    W("")

    # ================================================================================ SUMMARY
    W("## 7. Summary")
    W("")
    s = []
    u, c, a = H1["unfiltered"], H1["correct-only"], H1["admissible-only"]
    s.append(f"On the paper's 55 items read through Neuronpedia's hosted n1000 lens at the space-after-`=` position, the unspoken intermediate reaches rank 1 in {u['k1']}/{u['n']} ({u['p1']:.2f}) and rank ≤ 3 in {u['k3']}/{u['n']} ({u['p3']:.2f}), against a paired uninvolved-digit sign test of {u['sign'][0]}/{u['sign'][1]}/{u['sign'][2]} (p={u['sign'][3]:.0e}).")
    if c["n"]:
        s.append(f"Restricted to the {c['n']} paper items the model answers correctly (word-aware; {n_paper_corr_recs} of 55 paper items have a correctness record{', arm C incomplete' if n_paper_corr_recs < 55 else ''}), pass@1 is {c['p1']:.2f} and pass@3 {c['p3']:.2f}, so on the column the rule names H1 {h1_verdict(c).split(' (')[0].strip('*')}.")
    else:
        s.append("No correctness records exist yet, so the H1 column the rule names cannot be scored.")
    s.append(f"The addendum-2 headline column has {n_adm} admissible items of {n_gated} gated ({n_items} planned)" + ("; below the 30-item line, H1 and H2 are not testable as designed on that column" + (" while the gate is incomplete" if n_gated < n_items else "") + "." if not h12_testable else "; on it, " + (f"pass@1 {a['p1']:.2f} / pass@3 {a['p3']:.2f}." if a["n"] else "no items.")))
    if H2["unfiltered"]["n"] and "unfiltered" in H2V:
        b = H2["unfiltered"]; in1, in3, v = H2V["unfiltered"]
        s.append(f"The held-out 50 give pass@1 {b['p1']:.2f} and pass@3 {b['p3']:.2f} unfiltered, {'inside' if in1 and in3 else 'outside'} the paper-set CP intervals (H2: {v.replace('**','')})"
                 + ("; the correctness-filtered H2 columns have no held-out records yet." if H2["correct-only"]["n"] == 0 else "."))
    n2, nh, cats, num = carrier_stats("A", "all")
    if nh:
        s.append(f"Of the {nh} two-digit rank-1 hits, {num} ride the CJK numeral (H3 {'holds' if num/nh >= 2/3 else 'fails'} at {num/nh:.2f}).")
    if ds_all["w"] + ds_all["l"]:
        s.append(f"The correct intermediate beats the wrong-precedence decoy in {ds_all['w']} of {ds_all['w']+ds_all['l']} decided items ({ds_all['w']/(ds_all['w']+ds_all['l']):.2f}; H5 {'holds' if ds_all['w']/(ds_all['w']+ds_all['l']) > 0.6 else 'fails'}), with {ds_all['t']} ties.")
    expl = []
    if ("A", "all", "unfiltered") in H6:
        r = H6[("A", "all", "unfiltered")]
        expl.append(f"readout position is {'a sufficient explanation' if r < 0.5 else 'ruled out as an explanation'} for a null (nospace/space pass@1 ratio {r:.2f}, H6)")
    else:
        expl.append("readout position is inconclusive (no nospace reads)")
    if ("A", "new", "unfiltered") in H7 and H7[("A", "new", "unfiltered")]["t1"]:
        r = H7[("A", "new", "unfiltered")]
        expl.append(f"tokenization/synonym coverage is {'a sufficient explanation' if r['ratio'] < 0.5 else 'not by itself sufficient'} for two-digit misses (two-digit pass@1 {r['t1']/r['n2']:.2f} with CJK vs {r['t1n']/r['n2']:.2f} without, H7)")
    else:
        expl.append("tokenization/synonym coverage is inconclusive (no two-digit rank-1 hits to contrast)")
    if c2_reportable and ("exact", "unfiltered") in H4:
        r = H4[("exact", "unfiltered")]
        if r >= 0.5:
            expl.append(f"lens quality is ruled out as an explanation — the effect survives a 5-prompt lens at ratio {r:.2f} (H4; a 5-prompt fit, not n=25)")
        else:
            expl.append(f"lens quality is inconclusive — the effect fails under the 5-prompt lens (ratio {r:.2f}), which addendum 5 reads as not tested at n=25")
    elif box_present and not gate_passed:
        expl.append("lens quality is inconclusive (the C1-reproduces-A gate failed, so the box pipeline is not trusted and C2 is not reported)")
    else:
        expl.append("lens quality is inconclusive (the box arm has not been scored; the 5-prompt C2 lens has not been read)")
    s.append("Of the three candidate explanations for the discrepancy with Neel's team: " + "; ".join(expl) + ".")
    if ("A", "all", "space") in CTRL:
        cc = CTRL[("A", "all", "space")]
        s.append(f"The addendum-4 control shows that {cc['a']} of the {cc['r1']} rank-1 intermediates are also in the model's own top-8 next tokens at the readout, and {cc['b']} of the {cc['r3']} rank ≤ 3 items are workspace-only (absent from the layer-63 top-8); for those items the lens read is not reducible to next-token prediction, while the leak gate's finding that the intermediate opens narrated work in sampled continuations qualifies every hit that also sits in the top-8.")
    if G_api:
        s.append(f"The API leak gate has reached {G_api['n']} of {n_items} items: greedy writes the intermediate first in {G_api['wif']}, but at T=0.8 the intermediate leaks in {G_api['la']}, leaving {G_api['adm']} admissible.")
    if G_box:
        s.append(f"The box sampled pass gates {G_box['n']} items with {G_box['adm']} admissible.")
    if not box_present:
        s.append(f"Nothing here is a box result: exact ranks, the box leak gate, the C1-vs-A gate and H4 all wait on `{rel(BOX_DIR)}/`.")
    W(" ".join(s))
    W("")

    # ================================================================================ per-item appendix
    def gate_cell(g):
        if not g:
            return "–"
        return f"{g['assert_rate']:.1f} / {int(bool(g['leak_any']))} / {int(bool(g['admissible']))}"

    def fmt_med(v):
        if v is None:
            return "–"
        return f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"

    def corr_cell(c):
        if not c:
            return "–"
        return ("yes" if c["correct"] else "no") + f"({c['source'][0]})"

    W("## Appendix — per-item ranks (arm A, layers 0–62, top-8; 99 = absent)")
    W("")
    W("| item | intermediate | target | decoy | space rank (noCJK) | best layer | span ≤3 (n, first–last) | L63 rank | nospace rank | target rank | decoy rank | median uninvolved | correct (space) | assert / leak / admissible |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for it in items:
        a = A.get((it["set"], it["name"], "space")); b = A.get((it["set"], it["name"], "nospace"))
        if not a and not b:
            continue
        sp = a["span"] if a else (0, None, None)
        W(f"| {it['set']}_{it['name']} | {it['num']} | {it['target']} | {it['decoy'] or '–'} | {a['num'] if a else '–'} ({a['num_nocjk'] if a else '–'}) | {a['best_layer'] if a and a['best_layer'] is not None else '–'} | "
          f"{sp[0]}{f', {sp[1]}–{sp[2]}' if sp[0] else ''} | {a['L63_num'] if a else '–'} | {b['num'] if b else '–'} | {a['target'] if a else '–'} | {a['decoy'] if a and a['decoy'] is not None else '–'} | "
          f"{fmt_med(a['un_median']) if a else '–'} | {corr_cell(it['cont']['space'])} | {gate_cell(it['gate'])} |")
    W("")

    text = "\n".join(out) + "\n"
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
