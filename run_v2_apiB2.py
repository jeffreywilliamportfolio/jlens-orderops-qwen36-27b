#!/usr/bin/env python3
"""Arm B2 (addendum 1): chat-endpoint correctness with numCompletionTokens 48. Waits for run_v2_api.py to finish."""
import json, os, re, time, urllib.request
API="https://www.neuronpedia.org/api/lens/prompt"; MODEL="qwen3.6-27b"
KEY=[l.split("=",1)[1].strip().strip('"') for l in open("/Volumes/ExternalSSD/sae-tests/.env") if l.startswith("NEURONPEDIA_API_KEY=")][0]
D=os.path.dirname(os.path.abspath(__file__)); OUT=f"{D}/v2_api"; os.makedirs(f"{OUT}/raw_B2",exist_ok=True)
while not (os.path.exists(f"{OUT}/run.log") and "DONE" in open(f"{OUT}/run.log").read()): time.sleep(20)
log=open(f"{OUT}/run_B2.log","a")
def L(s): print(s,flush=True); log.write(s+"\n"); log.flush()
def post(body):
    rq=urllib.request.Request(API,data=json.dumps(body).encode(),headers={"Content-Type":"application/json","x-api-key":KEY},method="POST")
    for a in range(4):
        try: return json.loads(urllib.request.urlopen(rq,timeout=240).read())
        except Exception as e: L(f"  retry {a} {str(e)[:100]}"); time.sleep(5*(a+1))
    raise RuntimeError("API failed")
items=[("paper",it) for it in json.load(open(f"{D}/order_ops.json"))["items"]]+[("new",it) for it in json.load(open(f"{D}/items_new50.json"))["items"]]
L(f"B2 START {time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}")
for setname,it in items:
    f=f"{OUT}/raw_B2/{setname}_{it['name']}.json"
    if os.path.exists(f): continue
    d=post({"modelId":MODEL,"chat":[{"role":"user","content":it["prompt"]}],"type":["JACOBIAN_LENS"],"topN":1,"temperature":0,"numCompletionTokens":48,"filterNonWordTokens":True,"stream":False,"enableThinking":False})
    gen="".join(t["token"] for t in d["tokens"] if t["is_generated"]); ints=re.findall(r"-?\d+",gen); after_eq=re.findall(r"=\s*\$?\s*(-?\d+)",gen)
    corr_last=bool(ints) and ints[-1]==str(it["target"]); corr_eq=bool(after_eq) and after_eq[0]==str(it["target"])
    d["_item"]={"set":setname,"name":it["name"],"prompt":it["prompt"],"target":it["target"],"generated":gen,"correct_last_int":corr_last,"correct_first_after_eq":corr_eq}
    json.dump(d,open(f,"w")); L(f"B2 {setname} {it['name']} target={it['target']} last={corr_last} eq={corr_eq} gen={gen[:70]!r}"); time.sleep(0.8)
L(f"B2 DONE {time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}")
