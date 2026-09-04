#!/usr/bin/env python3
"""Prereg gate for arm C: "C1 must reproduce arm A's pass@1 within +-3 items or the local pipeline is not
trusted and C2 is not reported."  Also prints the H4 ratio (C2 pass@1 / C1 pass@1; rule line 0.5).

Scores everything with the frozen scorer's own rule (synonym sets + forms()/syn() pulled from
../score_orderops.py by AST; readout = last prompt token; rank = min over single-token synonyms of the
decoded top-k strings, stripped + lower-cased; min over layers; pass@k = rank <= k), applied identically to
  A  : ../v2_api/raw_A/<set>_<name>_<variant>.json   (Neuronpedia, top-8)
  C1 : <box_out>/hosted_n1000/<set>_<name>_<variant>.json   top_tokens sliced to the first 8 (API parity)
  C2 : <box_out>/neel25/...                                  same
plus, for the box files, the exact-rank version (rank_exact.keys.intermediate.best_rank, no cap).

Usage (local, after rsync):  python3 check_c1_vs_a.py --box-out ../v2_box [--variant space]
"""
import argparse, ast, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS_DIR = os.path.abspath(os.path.join(HERE, ".."))


def load_scorer_defs(path):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    parts = [ast.get_source_segment(src, n) for n in tree.body
             if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("W", "C", "OPS") for t in n.targets))
             or (isinstance(n, ast.FunctionDef) and n.name in ("forms", "syn"))]
    ns = {}
    exec("\n".join(parts), ns)
    return ns


def best_rank_topk(tokens_entry, S, k):
    """Scorer's best_rank over decoded top-k strings (rank 99 = absent)."""
    b = 99
    for layer in tokens_entry["results"][0]["top_tokens"]:
        for r, s in enumerate(layer[:k]):
            if s.strip().lower() in S and r + 1 < b:
                b = r + 1
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box-out", default=os.path.join(ITEMS_DIR, "v2_box"))
    ap.add_argument("--variant", default="space", choices=["space", "nospace"])
    ap.add_argument("--tolerance", type=int, default=3)
    args = ap.parse_args()
    sc = load_scorer_defs(os.path.join(ITEMS_DIR, "score_orderops.py"))
    items = [("paper", it) for it in json.load(open(os.path.join(ITEMS_DIR, "order_ops.json")))["items"]] + \
            [("new", it) for it in json.load(open(os.path.join(ITEMS_DIR, "items_new50.json")))["items"]]
    rows = []
    for setname, it in items:
        num = [k for k in it["intermediates"] if k.lstrip("-").isdigit()][0]
        S = sc["syn"](num)
        tag = f"{setname}_{it['name']}_{args.variant}"
        row = dict(tag=tag, num=num)
        fa = os.path.join(ITEMS_DIR, "v2_api", "raw_A", tag + ".json")
        if os.path.exists(fa):
            d = json.load(open(fa))
            last = [t for t in d["tokens"] if not t["is_generated"]][-1]
            row["A"] = best_rank_topk(last, S, 8)
        for lens in ("hosted_n1000", "neel25"):
            fb = os.path.join(args.box_out, lens, tag + ".json")
            if os.path.exists(fb):
                d = json.load(open(fb))
                last = [t for t in d["tokens"] if not t["is_generated"]][-1]
                row[lens + "_top8"] = best_rank_topk(last, S, 8)
                row[lens + "_top64"] = best_rank_topk(last, S, 64)
                row[lens + "_exact"] = d["rank_exact"]["keys"]["intermediate"]["best_rank"]
                row[lens + "_admissible"] = d["continuations"]["admissible"]
        rows.append(row)

    def passk(key, k, subset=None):
        vals = [r[key] for r in rows if key in r and (subset is None or subset(r))]
        return sum(v <= k for v in vals), len(vals)

    print(f"variant={args.variant}  items={len(rows)}")
    for key in ("A", "hosted_n1000_top8", "hosted_n1000_top64", "hosted_n1000_exact", "neel25_top8", "neel25_top64", "neel25_exact"):
        p1, n = passk(key, 1); p3, _ = passk(key, 3)
        if n:
            print(f"  {key:22s} n={n:3d} pass@1 {p1:3d} ({p1/n:.2f})  pass@3 {p3:3d} ({p3/n:.2f})")
    a1, na = passk("A", 1); c1, nc = passk("hosted_n1000_top8", 1)
    if na and nc:
        both = [r for r in rows if "A" in r and "hosted_n1000_top8" in r]
        agree = sum((r["A"] <= 1) == (r["hosted_n1000_top8"] <= 1) for r in both)
        diff = abs(a1 - c1)
        print(f"\nC1-vs-A gate: pass@1 A={a1} C1(top8)={c1} |diff|={diff} (tolerance {args.tolerance}); per-item pass@1 agreement {agree}/{len(both)}"
              f" -> {'PASS: local pipeline trusted; C2 may be reported' if diff <= args.tolerance else 'FAIL: do NOT report C2'}")
        disagree = [(r["tag"], r["A"], r["hosted_n1000_top8"]) for r in both if (r["A"] <= 1) != (r["hosted_n1000_top8"] <= 1)]
        for t in disagree[:20]:
            print("   disagree", t)
    h1, nh = passk("hosted_n1000_exact", 1); n1, nn = passk("neel25_exact", 1)
    if nh and nn and h1:
        print(f"\nH4: neel25 pass@1 {n1}/{nn} vs hosted {h1}/{nh}: ratio {n1/h1:.2f} (rule line 0.5; Jeffrey predicted ~0.9)")
    adm = [r for r in rows if r.get("hosted_n1000_admissible")]
    if adm:
        print(f"\nAddendum 2: admissible items {len(adm)}/{len(rows)}" + ("  (< 30 -> H1/H2 'not testable as designed')" if len(adm) < 30 else ""))
        for key in ("A", "hosted_n1000_exact", "neel25_exact"):
            p1, n = passk(key, 1, lambda r: r.get("hosted_n1000_admissible")); p3, _ = passk(key, 3, lambda r: r.get("hosted_n1000_admissible"))
            if n:
                print(f"  admissible-only {key:20s} n={n:3d} pass@1 {p1/n:.2f} pass@3 {p3/n:.2f}")
    json.dump(rows, open(os.path.join(args.box_out, f"check_c1_vs_a_{args.variant}.json"), "w"), indent=1) if os.path.isdir(args.box_out) else None


if __name__ == "__main__":
    main()
