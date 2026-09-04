#!/usr/bin/env python3
"""Score the order-ops lens eval from raw/ (Neuronpedia lens responses, topN 8).
Readout = last prompt token (the token immediately preceding the answer). Rank = min over
single-token synonyms at each layer; per intermediate, min over layers. pass@k = fraction with rank <= k.
Usage: python3 score_orderops.py  (run from this directory)"""
import json, os, re, glob, math
from itertools import groupby
W={0:"zero",1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine",10:"ten",11:"eleven",12:"twelve",13:"thirteen",14:"fourteen",15:"fifteen",16:"sixteen",17:"seventeen",18:"eighteen",19:"nineteen",20:"twenty",30:"thirty",40:"forty",50:"fifty",60:"sixty",70:"seventy",80:"eighty",90:"ninety"}
C={0:"零",1:"一",2:"二",3:"三",4:"四",5:"五",6:"六",7:"七",8:"八",9:"九",10:"十"}
OPS={"multiplication":["*","×","x","multiplication","multiply","multiplying","times","product","乘","乘法","乘以"],
     "addition":["+","addition","add","adding","plus","sum","加","加法","加上"],
     "subtraction":["-","−","–","subtraction","subtract","subtracting","minus","difference","减","减法","减去"],
     "division":["/","÷","division","divide","dividing","divided","quotient","除","除法","除以"]}
def forms(n):
    d={"digit":{str(n)}}
    if n in W: d["word"]={W[n]}
    elif n<100: t,o=divmod(n,10); d["word"]={W[t*10]+"-"+W[o],W[t*10]+W[o]}
    if n<=10: d["cjk"]={C[n]}
    elif n<20: d["cjk"]={"十"+C[n%10]}
    elif n<100: t,o=divmod(n,10); d["cjk"]={C[t]+"十"+(C[o] if o else "")}
    return d
def syn(key, cjk=True):
    if key.lstrip("-").isdigit():
        f=forms(int(key)); s=set(f["digit"])|set(f.get("word",()))|(set(f.get("cjk",())) if cjk else set())
    else: s=set(OPS.get(key,[key]))
    return {x.lower() for x in s}
def best_rank(tok, S):
    b=99
    for layer in tok["results"][0]["top_tokens"]:
        for r,s in enumerate(layer):
            if s.strip().lower() in S and r+1<b: b=r+1
    return b
def cp(k,n,a=0.05):
    """Clopper-Pearson interval by bisection on the exact binomial CDF."""
    def cdf(k,n,p): return sum(math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(k+1))
    lo,hi=0.0,1.0
    if k>0:
        a_,b_=0.0,1.0
        for _ in range(60):
            m=(a_+b_)/2
            if 1-cdf(k-1,n,m) < a/2: a_=m
            else: b_=m
        lo=a_
    if k<n:
        a_,b_=0.0,1.0
        for _ in range(60):
            m=(a_+b_)/2
            if cdf(k,n,m) < a/2: b_=m
            else: a_=m
        hi=b_
    return lo,hi
items={it["name"]:it for it in json.load(open("order_ops.json"))["items"]}
rows=[]
for f in sorted(glob.glob("raw/*.json")):
    name=os.path.basename(f)[3:-5]; it=items[name]; d=json.load(open(f))
    prompt=[t for t in d["tokens"] if not t["is_generated"]]; last,prev=prompt[-1],prompt[-2]
    num=[k for k in it["intermediates"] if k.lstrip("-").isdigit()][0]; op=[k for k in it["intermediates"] if not k.lstrip("-").isdigit()][0]
    used=set(re.findall(r"\d",it["prompt"]))|set(re.findall(r"\d",str(it["target"])))|set(num)
    un=[best_rank(last,{str(x)}) for x in range(10) if str(x) not in used]
    fm=forms(int(num)); carrier=min({k:best_rank(last,{x.lower() for x in v}) for k,v in fm.items()}.items(), key=lambda kv:kv[1])
    rows.append(dict(name=name,prompt=it["prompt"],num=num,two_digit=int(num)>=10,last_tok=last["token"],
        num_last=best_rank(last,syn(num)),num_last_nocjk=best_rank(last,syn(num,False)),num_prev=best_rank(prev,syn(num)),
        op_last=best_rank(last,syn(op)),carrier=carrier[0] if carrier[1]<99 else "none",uninvolved=un))
n=len(rows)
def passk(vals,k): return sum(v<=k for v in vals)
print(f"items {n} | last prompt tokens: {sorted({r['last_tok'] for r in rows})}")
for lab,key in [("number @last (digit+word+CJK)","num_last"),("number @last (no CJK)","num_last_nocjk"),("number @prev token","num_prev"),("operation @last","op_last")]:
    v=[r[key] for r in rows]; k1,k3,k5,k8=(passk(v,k) for k in (1,3,5,8))
    lo1,hi1=cp(k1,n); lo3,hi3=cp(k3,n)
    print(f"{lab:32s} pass@1 {k1}/{n}={k1/n:.2f} [CP {lo1:.2f},{hi1:.2f}]  pass@3 {k3}/{n}={k3/n:.2f} [CP {lo3:.2f},{hi3:.2f}]  pass@5 {k5/n:.2f}  pass@8 {k8/n:.2f}")
base=[u for r in rows for u in r["uninvolved"]]; b1,b3=passk(base,1),passk(base,3)
print(f"uninvolved-digit baseline (n={len(base)} readings, clustered by item) pass@1 {b1/len(base):.2f} pass@3 {b3/len(base):.2f} pass@8 {passk(base,8)/len(base):.2f}")
pb1=[sum(u<=1 for u in r["uninvolved"])/len(r["uninvolved"]) for r in rows]; pb3=[sum(u<=3 for u in r["uninvolved"])/len(r["uninvolved"]) for r in rows]
print(f"per-item baseline (mean over items of fraction of uninvolved digits hit): pass@1 {sum(pb1)/n:.2f} pass@3 {sum(pb3)/n:.2f}")
# paired sign test: intermediate rank vs median uninvolved rank, per item
import statistics
w=l=t=0
for r in rows:
    m=statistics.median(r["uninvolved"]); w+=r["num_last"]<m; l+=r["num_last"]>m; t+=r["num_last"]==m
p=sum(math.comb(w+l,i) for i in range(0,l+1))/2**(w+l)*2 if w+l else 1.0
print(f"paired sign test (intermediate rank vs item's median uninvolved-digit rank): better {w} / worse {l} / tie {t}, two-sided p={min(1,p):.2e}")
td=[r for r in rows if r["two_digit"]]; from collections import Counter
print(f"two-digit intermediates n={len(td)}: carrier {dict(Counter(r['carrier'] for r in td))}; single-digit n={n-len(td)}: {dict(Counter(r['carrier'] for r in rows if not r['two_digit']))}")
json.dump(rows,open("scored_rows.json","w"),indent=1); print("wrote scored_rows.json")
