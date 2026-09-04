#!/bin/bash
# Orchestration only (not part of the frozen protocol): run the API arms sequentially, idempotently, waiting out the hourly quota.
# Order: A/B lens reads -> C raw-prompt leak gate -> B2 chat correctness. Each runner skips files that already exist.
cd "$(dirname "$0")"; PY=/Volumes/ExternalSSD/cc-lens/.venv-jlens/bin/python
KEY=$(grep -E '^NEURONPEDIA_API_KEY=' /Volumes/ExternalSSD/sae-tests/.env | cut -d= -f2- | tr -d '"')
log(){ echo "$(date -u +%H:%M:%SZ) $*" | tee -a v2_api/supervise.log; }
wait_quota(){ # block until a throwaway call returns 200; report the remaining-quota header when present
  while true; do
    out=$(curl -s -D - -o /dev/null -m 60 -H "Content-Type: application/json" -H "x-api-key: $KEY" -X POST https://www.neuronpedia.org/api/lens/prompt --data '{"modelId":"qwen3.6-27b","chat":[{"role":"user","content":"hi"}],"type":["JACOBIAN_LENS"],"topN":1,"temperature":0,"numCompletionTokens":0,"stream":false,"enableThinking":false}')
    code=$(echo "$out" | head -1 | awk '{print $2}'); rem=$(echo "$out" | grep -i "x-limit-remaining" | awk '{print $2}' | tr -d '\r')
    if [ "$code" = "200" ]; then log "quota ok (remaining=${rem:-?})"; return; fi
    log "quota exhausted (HTTP $code, remaining=${rem:-?}); sleeping 600s"; sleep 600
  done
}
run_arm(){ # $1 runner, $2 done-marker file, $3 done-marker text
  while ! grep -q "$3" "$2" 2>/dev/null; do
    wait_quota; log "launch $1"; $PY "$1" >> "v2_api/$(basename "$1" .py).out" 2>&1 || log "$1 exited non-zero; will resume"
    grep -q "$3" "$2" 2>/dev/null || sleep 120
  done; log "$1 complete"
}
run_arm run_v2_api.py   v2_api/run.log    "DONE"
run_arm run_v2_apiC.py  v2_api/run_C.log  "C DONE"
run_arm run_v2_apiB2.py v2_api/run_B2.log "B2 DONE"
log "ALL ARMS COMPLETE"
