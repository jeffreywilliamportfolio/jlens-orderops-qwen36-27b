#!/usr/bin/env python3
"""v2 API arms. Arm A: lens read of every item, with-space and no-space, raw inputTokenIds, topN 8, greedy.
Arm B: correctness via the chat endpoint (template-mediated, flagged), numCompletionTokens 8, greedy.
Writes v2_api/raw_A/<set>_<name>_<space|nospace>.json and v2_api/raw_B/<set>_<name>.json plus v2_api/run.log."""
import json, os, sys, time, urllib.request
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
from transformers import AutoTokenizer
API="https://www.neuronpedia.org/api/lens/prompt"; MODEL="qwen3.6-27b"
KEY=[l.split("=",1)[1].strip().strip('"') for l in open("/Volumes/ExternalSSD/sae-tests/.env") if l.startswith("NEURONPEDIA_API_KEY=")][0]
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
D=os.path.dirname(os.path.abspath(__file__)); OUT=f"{D}/v2_api"; os.makedirs(f"{OUT}/raw_A",exist_ok=True); os.makedirs(f"{OUT}/raw_B",exist_ok=True)
log=open(f"{OUT}/run.log","a")
def L(*a):
    s=" ".join(str(x) for x in a); print(s,flush=True); log.write(s+"\n"); log.flush()
def post(body):
    rq=urllib.request.Request(API,data=json.dumps(body).encode(),headers={"Content-Type":"application/json","x-api-key":KEY},method="POST")
    for attempt in range(4):
        try: return json.loads(urllib.request.urlopen(rq,timeout=180).read())
        except Exception as e:
            L("  retry",attempt,str(e)[:100]); time.sleep(5*(attempt+1))
    raise RuntimeError("API failed 4x")
items=[("paper",it) for it in json.load(open(f"{D}/order_ops.json"))["items"]]+[("new",it) for it in json.load(open(f"{D}/items_new50.json"))["items"]]
L(f"START {time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())} items={len(items)}")
n=0
for setname,it in items:
    name=it["name"]
    for variant,prompt in (("space",it["prompt"]),("nospace",it["prompt"].rstrip())):
        f=f"{OUT}/raw_A/{setname}_{name}_{variant}.json"
        if os.path.exists(f): continue
        ids=tok(prompt,add_special_tokens=False)["input_ids"]
        d=post({"modelId":MODEL,"inputTokenIds":ids,"type":["JACOBIAN_LENS"],"topN":8,"temperature":0,"numCompletionTokens":0,"filterNonWordTokens":False,"stream":False,"enableThinking":False,"prependBos":True})
        d["_item"]={"set":setname,"name":name,"variant":variant,"prompt":prompt,"input_ids":ids}; json.dump(d,open(f,"w")); n+=1
        last=[t for t in d["tokens"] if not t["is_generated"]][-1]["token"]
        L(f"A {setname} {name} {variant:7s} last_tok={last!r}"); time.sleep(0.8)
    f=f"{OUT}/raw_B/{setname}_{name}.json"
    if not os.path.exists(f):
        d=post({"modelId":MODEL,"chat":[{"role":"user","content":it["prompt"]}],"type":["JACOBIAN_LENS"],"topN":1,"temperature":0,"numCompletionTokens":8,"filterNonWordTokens":True,"stream":False,"enableThinking":False})
        gen="".join(t["token"] for t in d["tokens"] if t["is_generated"]); d["_item"]={"set":setname,"name":name,"prompt":it["prompt"],"target":it["target"],"generated":gen}
        json.dump(d,open(f,"w")); n+=1; L(f"B {setname} {name} target={it['target']} gen={gen!r}"); time.sleep(0.8)
L(f"DONE {time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())} calls_this_run={n}")
