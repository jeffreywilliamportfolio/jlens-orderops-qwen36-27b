#!/usr/bin/env python3
"""Arm C (addendum 3): raw-prompt greedy continuation (both variants) + 10 sampled continuations (with-space) via the API's `prompt` field."""
import json, os, re, time, urllib.request
API="https://www.neuronpedia.org/api/lens/prompt"; MODEL="qwen3.6-27b"
KEY=[l.split("=",1)[1].strip().strip('"') for l in open("/Volumes/ExternalSSD/sae-tests/.env") if l.startswith("NEURONPEDIA_API_KEY=")][0]
D=os.path.dirname(os.path.abspath(__file__)); OUT=f"{D}/v2_api"; os.makedirs(f"{OUT}/raw_C",exist_ok=True)
while not (os.path.exists(f"{OUT}/run.log") and "DONE" in open(f"{OUT}/run.log").read()): time.sleep(20)
log=open(f"{OUT}/run_C.log","a")
def L(s): print(s,flush=True); log.write(s+"\n"); log.flush()
def post(body):
    rq=urllib.request.Request(API,data=json.dumps(body).encode(),headers={"Content-Type":"application/json","x-api-key":KEY},method="POST")
    for a in range(4):
        try: return json.loads(urllib.request.urlopen(rq,timeout=240).read())
        except Exception as e: L(f"  retry {a} {str(e)[:100]}"); time.sleep(5*(a+1))
    raise RuntimeError("API failed")
def gen_text(d): return "".join(t["token"] for t in d["tokens"] if t["is_generated"])
def first_num(s):
    m=re.search(r"-?\d+(?:\.\d+)?",s); return m.group(0) if m else None
def leaks(s,inter,target):
    nums=re.findall(r"-?\d+(?:\.\d+)?",s)
    for x in nums:
        if x==str(target): return False
        if x==str(inter): return True
    return False
items=[("paper",it) for it in json.load(open(f"{D}/order_ops.json"))["items"]]+[("new",it) for it in json.load(open(f"{D}/items_new50.json"))["items"]]
L(f"C START {time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}")
for setname,it in items:
    inter=[k for k in it["intermediates"] if k.lstrip("-").isdigit()][0]; target=str(it["target"])
    f=f"{OUT}/raw_C/{setname}_{it['name']}.json"
    if os.path.exists(f): continue
    rec={"set":setname,"name":it["name"],"target":target,"intermediate":inter,"variants":{}}
    for variant,prompt in (("space",it["prompt"]),("nospace",it["prompt"].rstrip())):
        d=post({"modelId":MODEL,"prompt":prompt,"type":["JACOBIAN_LENS"],"topN":1,"temperature":0,"numCompletionTokens":12,"filterNonWordTokens":True,"stream":False,"enableThinking":False})
        g=gen_text(d); rec["variants"][variant]={"greedy":g,"correct_greedy":first_num(g)==target,"greedy_writes_intermediate_first":first_num(g)==inter}
        time.sleep(0.5)
    samples=[]
    for i in range(10):
        d=post({"modelId":MODEL,"prompt":it["prompt"],"type":["JACOBIAN_LENS"],"topN":1,"temperature":0.8,"numCompletionTokens":32,"filterNonWordTokens":True,"stream":False,"enableThinking":False})
        samples.append(gen_text(d)); time.sleep(0.4)
    ar=sum(first_num(s)==target for s in samples)/10; la=any(leaks(s,inter,target) for s in samples) or leaks(rec["variants"]["space"]["greedy"],inter,target)
    rec.update({"samples":samples,"assert_rate":ar,"leak_any":la,"admissible":(ar>=0.8 and not la)})
    json.dump(rec,open(f,"w"),ensure_ascii=False)
    L(f"C {setname} {it['name']:26s} greedy={rec['variants']['space']['greedy'][:14]!r} correct={rec['variants']['space']['correct_greedy']} inter_first={rec['variants']['space']['greedy_writes_intermediate_first']} assert={ar:.1f} leak={la} admissible={rec['admissible']}")
L(f"C DONE {time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}")
