# RUNBOOK — arm C of PREREG_orderops_v2 (+ Addendum 2 leak gate) on the live box

Box (facts 2026-09-04): Vast.ai image `vastai/pytorch:cuda-12.8.1-auto`, 1x RTX PRO 6000 96 GB, 220 GB disk.
Python: `/venv/main/bin/python` (torch 2.11.0+cu128, transformers 5.14.1, datasets 5.0.1, accelerate, safetensors);
system python3 has NO torch. Model already at `/workspace/models/qwen36-27b` (52 GB, 15 shards; marker `/workspace/.DONE_model`),
hosted lens already at `/workspace/lens_hosted/qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt`
(3.3 GB; marker `/workspace/.DONE_lens`), token at `/workspace/.hf_token`, `HF_HOME=/workspace/.hf_home`,
logs `/workspace/logs/`, outputs `/workspace/out/`.

Frozen inputs the box needs, rsynced from this directory: `order_ops.json`, `items_new50.json`, `score_orderops.py`
(sha256 5bb4e0718cea342a…, the scorer whose synonym sets read_items.py extracts by AST) and this `box/` folder.
The FREEZE.sha256 hashes must still match before anything runs (`shasum -a 256 -c FREEZE.sha256` locally).

Standing rules (CLAUDE.md): no apt; never two sessions/agents on the same box; never two GPU jobs at once (the fit and
the read are sequenced); pull everything before any stop; **the box is NOT destroyed or stopped without asking Jeffrey.**

| step | what | wall time (est.) | gate |
|---|---|---|---|
| 1 | first contact | ~1 min | `ssh … echo OK` within ~2 min of `running`, else the box is bad |
| 2 | network faithfulness | ~2 min | ≥ 40 MB/s real download, ≥ 25 MB/s upload |
| 3 | token push over stdin | seconds | `/workspace/.hf_token` mode 600, never echoed |
| 4 | rsync inputs + bootstrap | ~3–5 min | torch/CUDA import, fla/causal_conv1d ABSENT, lens sha256 match |
| 5 | fit (C2, Neel recipe) | **1–3 h** (measure prompt 1) | 25 prompts fitted, provenance.json written |
| 6 | read (C1 + C2 + arm B + Addendum 2) | **~20–35 min** | 210 prompts x 2 lenses + continuations |
| 7 | pull to `../v2_box/` | ~2 min | manifest + 420 lens files + 210 continuations present |
| 8 | C1-reproduces-A gate, then scoring | minutes, local | ±3 items on pass@1 |
| 9 | teardown | — | **ask Jeffrey first** |

## 1. First contact (~1 min)
```bash
ID=<instance id>; vastai show instances-v1 --raw | python3 -c 'import json,sys;[print(i["id"],i["actual_status"],i["ssh_host"],i["ssh_port"],i["gpu_name"],i["dph_total"]) for i in json.load(sys.stdin)]'
URL=$(vastai ssh-url $ID)   # proxy endpoint, e.g. ssh://root@ssh7.vast.ai:PORT — the reliable one
ssh -i ~/.ssh/vast_gptoss_sl -o IdentitiesOnly=yes -p PORT root@HOST 'echo OK; nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /workspace; supervisorctl status'
```
If the key is refused or the port resets after ~2 min of retries: stop debugging, tell Jeffrey (his rule: don't grind > ~3 min).
Stop the template llama-server if it is up: `supervisorctl stop llama` (bootstrap does this too).

## 2. Network faithfulness gate (~2 min)
The downloads are already complete on this box, so the real-workload download gate has effectively been passed
(record the rate from the download logs in `/workspace/logs/` if present: bytes / seconds). Upload gate, still required
before the pull:
```bash
dd if=/dev/zero bs=1M count=256 2>/dev/null | curl -s -o /dev/null -w '%{speed_upload}\n' -T - https://speed.cloudflare.com/__up
```
PASS ≥ 25 MB/s (25000000). FAIL → tell Jeffrey before pulling large artifacts; do not destroy on your own.

## 3. Token push over stdin (seconds)
```bash
TOKEN="$(grep -E '^HF_TOKEN=' /Volumes/ExternalSSD/qwen-huahua-speech-prompts/.env | cut -d= -f2- | tr -d '"')"
printf '%s' "$TOKEN" | ssh -i ~/.ssh/vast_gptoss_sl -o IdentitiesOnly=yes -p PORT root@HOST 'umask 077; cat > /workspace/.hf_token; chmod 600 /workspace/.hf_token'
unset TOKEN
```
(Already done on this box; re-run only if `/workspace/.hf_token` is missing. Never put the token on a command line or in a log.)

## 4. Push inputs + bootstrap (~3–5 min; markers already present so P2 returns immediately)
```bash
D=/Volumes/ExternalSSD/cc-lens/outputs/orderops_qwen36_27b_20260904
ssh -i ~/.ssh/vast_gptoss_sl -o IdentitiesOnly=yes -p PORT root@HOST 'mkdir -p /workspace/orderops/box /workspace/logs /workspace/out'
rsync -av -e "ssh -i ~/.ssh/vast_gptoss_sl -o IdentitiesOnly=yes -p PORT" \
  "$D/order_ops.json" "$D/items_new50.json" "$D/score_orderops.py" "$D/FREEZE.sha256" root@HOST:/workspace/orderops/
rsync -av -e "ssh -i ~/.ssh/vast_gptoss_sl -o IdentitiesOnly=yes -p PORT" "$D/box/" root@HOST:/workspace/orderops/box/
ssh -i ~/.ssh/vast_gptoss_sl -o IdentitiesOnly=yes -p PORT root@HOST \
  'cd /workspace/orderops && sha256sum -c <(grep -E "order_ops.json|items_new50.json|score_orderops.py" FREEZE.sha256) && bash box/bootstrap.sh 2>&1 | tee /workspace/logs/bootstrap.log'
```
bootstrap.sh: P0 preflight (nvidia-smi, llama-server stopped, token present, torch+CUDA on `/venv/main`), P1 only-if-needed
pip (nothing pre-installed is upgraded) + `anthropics/jacobian-lens` cloned at `581d3986` and installed `--no-deps -e`,
then the import gate incl. **fla / flash_linear_attention / causal_conv1d must be absent**; P2 waits on the two markers
(byte-growth watch that only warns — it never relaunches the operator's downloads); P3 verifies the lens
(3,303,032,772 bytes, sha256 `1718c8c5…`), the 15 shards + index, introspects the .pt (expect keys J/n_prompts/
source_layers/d_model, 63 fp16 5120x5120 matrices for layers 0..62) and writes `/workspace/bootstrap_provenance.json`.
Ends with `READY.`

## 5. Fit C2 — Neel's recipe (1–3 h; time prompt 1 and extrapolate)
```bash
ssh … root@HOST 'cd /workspace/orderops/box && nohup /venv/main/bin/python fit_neel_recipe.py > /workspace/logs/fit_neel.log 2>&1 & echo started'
ssh … root@HOST 'tail -5 /workspace/logs/fit_neel.log'      # every line is timestamped; per-100-pass DEBUG lines inside a prompt
```
What it does: 25 pile-10k docs (first 25 in row order with ≥ 132 tokens, truncated to 128 ids; the dry run picked doc
ids 0,1,2,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27 — `prompts_selected.json` on the box must list the
same), skip_first 4, target_layer −2 → block **62** (penultimate block output; source layers 0..61), dim_batch 8 with
automatic fallback to 4 on CUDA OOM (resumes from the per-prompt checkpoint `fit_ckpt.pt`), bf16 weights, plain autograd.
Cost per prompt = 1 forward + 640 backward passes (5120/8) through the retained graph; Neuronpedia's B200 fit of the
sibling 27B ran ~60 s/prompt at dim_batch 64 (80 passes), so expect **~2–6 min/prompt here → 1–3 h for 25**. If prompt 1
takes > 8 min, note it and let it run (checkpointing means nothing is lost); if it OOMs twice, tell Jeffrey.
Output: `/workspace/out/lens_neel/Qwen3.6-27B_jacobian_lens_neel25.pt` (upstream layout, fp16, ~3.2 GB) + `provenance.json`
(commit, doc ids, token counts, dtype, per-prompt seconds, GPU peak) + `prompts_selected.json`.
Optional sanity (seconds, CPU): `/venv/main/bin/python /Volumes/…/cc-lens/compare_lenses.py` is local-only; on the box use
`python -c "from jlens.lens import JacobianLens as L; print(L.load('/workspace/out/lens_neel/Qwen3.6-27B_jacobian_lens_neel25.pt'))"`.

## 6. Read C1 + C2 + arm B + Addendum 2 (~20–35 min)
```bash
ssh … root@HOST 'cd /workspace/orderops/box && nohup /venv/main/bin/python read_items.py > /workspace/logs/read_items.log 2>&1 & echo started'
ssh … root@HOST 'tail -3 /workspace/logs/read_items.log'
```
Smoke first if you want (adds ~3 min): `read_items.py --limit 3 --out /workspace/out_smoke`.
Per prompt (210 = 105 items x {space, nospace}): one forward pass (hooks on all 64 block outputs) → both lenses on the same
residuals (top-8 at every position, top-64 + exact full-vocab ranks of every synonym id at the last token, digit ranks for the
uninvolved-digit baseline) → greedy continuation ≤ 6 tokens (arm B on the box) → **Addendum 2**: 10 sampled continuations
(temperature 0.8, top_p 0.95, 32 new tokens, seeds 0..9 as one batch with a CUDA generator per row; no KV cache — for
≤ 30-token prompts a full re-forward costs the same as a cached step, ~32 forwards of batch 10 ≈ 1.5–2.5 s per prompt →
**~6–9 min of the total**) → `assert_rate`, `leak_any`, `greedy_writes_intermediate_first`, `admissible`.
Model load ~1–2 min, lens load ~1 min (2 x 6.6 GB fp32 on GPU next to the 52 GB weights), then ~5–8 s per prompt.
Outputs: `/workspace/out/{hosted_n1000,neel25}/<set>_<name>_<variant>.json`, `/workspace/out/continuations/…`,
`/workspace/out/manifest.json` (model config/index sha, lens sha256s, torch/transformers, GPU, timestamps, capture check).
BOS: `--bos none` (default) reproduces Neuronpedia's inputTokenIds path for this BOS-less tokenizer (token 0 of every
arm-A response is the first prompt token). `--bos endoftext` exists only as a sensitivity variant; not the prereg pipeline.

## 7. Pull results (~2 min for ~200 MB)
```bash
D=/Volumes/ExternalSSD/cc-lens/outputs/orderops_qwen36_27b_20260904
mkdir -p "$D/v2_box"
rsync -av -e "ssh -i ~/.ssh/vast_gptoss_sl -o IdentitiesOnly=yes -p PORT" \
  root@HOST:/workspace/out/ "$D/v2_box/"                       # lens files, continuations, manifest, lens_neel/ (incl. the 3.2 GB .pt)
rsync -av -e "ssh -i ~/.ssh/vast_gptoss_sl -o IdentitiesOnly=yes -p PORT" \
  root@HOST:/workspace/logs/ root@HOST:/workspace/bootstrap_provenance.json root@HOST:/workspace/lens_hosted/introspect.json "$D/v2_box/logs/"
```
If the 3.2 GB `lens_neel/*.pt` is not wanted locally yet, add `--exclude 'lens_neel/*.pt'` and pull it separately later — but
pull it before any stop/destroy (a stopped box's /workspace is unreachable).

## 8. C1-reproduces-A gate, then scoring (local)
```bash
cd "$D/box" && python3 check_c1_vs_a.py --box-out ../v2_box            # also --variant nospace
```
Prereg rule: C1 (hosted lens on the box, scored with the scorer's rule on the first 8 of each layer's top tokens) must match
arm A's pass@1 within ±3 items, else the local pipeline is not trusted and **C2 is not reported**. The script also prints
the H4 ratio (neel25 / hosted pass@1, rule line 0.5) and the Addendum-2 admissible count (< 30 → H1/H2 "not testable as
designed"). Full scoring = adapting `score_orderops.py` to read `v2_box/<lens>/` (same `tokens[...]["results"][0]["top_tokens"]`
shape, 64 layers; slice `[:8]` for the capped numbers, use `rank_exact` for the raw ranks and the persistence descriptives).

## 9. Teardown
**Do not stop or destroy the box without asking Jeffrey (standing rule 2026-08-07).** Before asking, confirm the pull is
complete (`ls v2_box/hosted_n1000 | wc -l` = 210, `v2_box/neel25` = 210, `v2_box/continuations` = 210, `manifest.json`,
`lens_neel/Qwen3.6-27B_jacobian_lens_neel25.pt` + `provenance.json`). After his word: `yes | vastai destroy instance $ID`,
`vastai show instances-v1 --raw` → active = 0, teardown record (hours, $/h, total) appended to the run's README.

## Budget note
Hours left = credit / dph_total (`vastai show user --raw`, `vastai show instances-v1 --raw`). Expected GPU time for steps
4–7: **~2–4.5 h** (fit dominates). A dim_batch-4 fallback roughly doubles the fit's backward count.
