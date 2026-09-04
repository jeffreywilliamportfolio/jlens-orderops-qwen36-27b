#!/bin/bash
# Arm-C bootstrap for PREREG_orderops_v2 on the live Vast.ai box
#   image vastai/pytorch:cuda-12.8.1-auto, 1x RTX PRO 6000 (96 GB), 220 GB disk.
# Run on the box as:   mkdir -p /workspace/logs && bash /workspace/orderops/box/bootstrap.sh 2>&1 | tee /workspace/logs/bootstrap.log
#
# The two downloads were ALREADY started by hand on this box (2026-09-04) and this script does NOT launch or
# relaunch them. It waits on their markers, with a byte-growth watch that only WARNS on a stall:
#   model  Qwen/Qwen3.6-27B  -> /workspace/models/qwen36-27b          marker /workspace/.DONE_model
#   lens   neuronpedia/jacobian-lens qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt
#          -> /workspace/lens_hosted/<that path>                      marker /workspace/.DONE_lens
#
# Python env (box fact, 2026-09-04): /venv/main (torch 2.11.0+cu128, transformers 5.14.1, datasets 5.0.1, accelerate,
# safetensors, huggingface_hub). System python3 has NO torch. Everything runs on /venv/main/bin/python; the only
# additions are hf_xet/pyarrow if missing and an editable, --no-deps install of jacobian-lens. (A nested
# `python -m venv --system-site-packages` made FROM /venv/main would inherit the BASE interpreter's site-packages,
# i.e. the torch-less system python -- so no nested venv here; nothing pre-installed is upgraded or replaced.)
# Follows /Volumes/ExternalSSD/CLAUDE.md: NO apt; reuse the image's torch;
# HF_HOME=/workspace/.hf_home, HF_XET_HIGH_PERFORMANCE=1; token read from /workspace/.hf_token (never echoed);
# torch+CUDA verified on the image python first, the full stack verified BEFORE the downloads are waited on;
# flash-linear-attention / causal-conv1d MUST be absent (pure-PyTorch DeltaNet path; the fit needs plain autograd).
#
# Never enable `set -x` in this file: HF_TOKEN is exported into the environment.
set -euo pipefail

# ----------------------------------------------------------------------------- pins / paths (verbatim from the live box)
MODEL_ID="Qwen/Qwen3.6-27B"
MODEL_REVISION_EXPECTED="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"   # pin from cc-lens/template-lens/MODEL_PIN_20260821.json (checked, not enforced)
MODEL_DIR="/workspace/models/qwen36-27b"
MODEL_DONE="/workspace/.DONE_model"
LENS_REPO="neuronpedia/jacobian-lens"
LENS_FILE="qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
LENS_PT="/workspace/lens_hosted/$LENS_FILE"
LENS_DONE="/workspace/.DONE_lens"
LENS_SHA256="1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1"  # HF LFS sha256 of LENS_FILE (repo sha 0731326e, read 2026-09-04)
LENS_BYTES="3303032772"
JLENS_URL="https://github.com/anthropics/jacobian-lens"
JLENS_COMMIT="581d398613e5602a5af361e1c34d3a92ea82ba8e"   # "Initial release" 2026-07-01; the commit the prereg's order_ops.json came from
JLENS_DIR="/workspace/jacobian-lens"
PY="/venv/main/bin/python"                 # the image env that carries torch (box fact); system python3 has no torch
LOGS="/workspace/logs"
OUT="/workspace/out"
# Expected (box facts 2026-09-04; the local dry run passed on the same transformers): warn, don't die, on drift.
EXPECT_TORCH="2.11.0"
EXPECT_TRANSFORMERS="5.14.1"
STALL_SECONDS="${STALL_SECONDS:-300}"      # zero-growth window after which the watch WARNS (it never kills the operator's downloads)
DL_TIMEOUT_MIN="${DL_TIMEOUT_MIN:-120}"

die() { echo "FATAL: $*" >&2; exit 1; }
phase() { echo; echo "=================== $(date -u +%FT%TZ)  $*"; }

# ----------------------------------------------------------------------------- P0 preflight
phase "P0 preflight"
mkdir -p /workspace "$LOGS" "$OUT"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || die "nvidia-smi"
df -h /workspace
# The PyTorch template auto-starts a llama-server that eats VRAM; stop it via supervisor (plain pkill respawns).
if command -v supervisorctl >/dev/null 2>&1; then supervisorctl stop llama >/dev/null 2>&1 || true; fi
pkill -x llama-server >/dev/null 2>&1 || true
[ -s /workspace/.hf_token ] || die "missing /workspace/.hf_token (push it over stdin per CLAUDE.md; never on a command line)"
chmod 600 /workspace/.hf_token
export HF_TOKEN="$(< /workspace/.hf_token)"
export HF_HOME=/workspace/.hf_home
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"

[ -x "$PY" ] || die "$PY missing (box fact: the image env lives at /venv/main)"
echo "python: $PY ($("$PY" --version 2>&1))"
# Gate 1 (CLAUDE.md): torch+CUDA must import on the IMAGE python before anything else is built on it.
"$PY" - <<'PY' || die "image torch/CUDA check failed -> this box is unusable; stop here and tell Jeffrey"
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info()
print(f"torch {torch.__version__} cuda {torch.version.cuda} gpu {name} free {free/2**30:.1f} GiB / {total/2**30:.1f} GiB")
assert total / 2**30 >= 90, f"need a ~96 GB card, got {total/2**30:.1f} GiB"
PY
echo "PHASE_P0_OK"

# ----------------------------------------------------------------------------- P1 venv (reuse the image's torch)
phase "P1 env: /venv/main + (only-if-needed) huggingface_hub safetensors accelerate datasets numpy pyarrow hf_xet + jacobian-lens @ $JLENS_COMMIT"
# No -U and no version specs: pip leaves an already-installed package alone ("Requirement already satisfied"), so the
# image's torch 2.11 / transformers 5.14.1 stack is never touched; only genuinely missing pieces are added.
"$PY" -m pip install -q huggingface_hub safetensors accelerate datasets numpy pyarrow hf_xet transformers
if [ ! -d "$JLENS_DIR/.git" ]; then git clone -q "$JLENS_URL" "$JLENS_DIR"; fi
git -C "$JLENS_DIR" fetch -q origin "$JLENS_COMMIT" 2>/dev/null || true
git -C "$JLENS_DIR" checkout -q "$JLENS_COMMIT"
[ "$(git -C "$JLENS_DIR" rev-parse HEAD)" = "$JLENS_COMMIT" ] || die "jacobian-lens is not at $JLENS_COMMIT"
# --no-deps: its pyproject lists torch/transformers/numpy/huggingface_hub; all present — never let pip swap torch.
"$PY" -m pip install -q --no-deps -e "$JLENS_DIR"

# Gate 2: the full stack imports on the venv python and the banned fused kernels are ABSENT — checked BEFORE waiting on downloads.
EXPECT_TORCH="$EXPECT_TORCH" EXPECT_TRANSFORMERS="$EXPECT_TRANSFORMERS" "$PY" - <<'PY' || die "env verification failed"
import importlib, sys
import torch, transformers, safetensors, accelerate, datasets, numpy, huggingface_hub, jlens
from jlens.hf import from_hf
from jlens.lens import JacobianLens
from jlens.fitting import fit
import os
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_ok", torch.cuda.is_available())
print("transformers", transformers.__version__, "| safetensors", safetensors.__version__, "| accelerate", accelerate.__version__,
      "| datasets", datasets.__version__, "| numpy", numpy.__version__, "| huggingface_hub", huggingface_hub.__version__)
for name, have, want in (("torch", torch.__version__, os.environ.get("EXPECT_TORCH", "")), ("transformers", transformers.__version__, os.environ.get("EXPECT_TRANSFORMERS", ""))):
    if want and not have.startswith(want):
        print(f"WARNING: {name} {have} != expected {want} (box fact / local dry-run version); record it, then continue")
for mod in ("fla", "flash_linear_attention", "causal_conv1d"):
    try:
        importlib.import_module(mod)
    except ImportError:
        print(f"OK: {mod} absent")
    else:
        sys.exit(f"FATAL: {mod} is importable — uninstall it (pure-PyTorch DeltaNet path required)")
assert torch.cuda.is_available()
PY
echo "PHASE_P1_OK"

# ----------------------------------------------------------------------------- P2 wait on the operator's downloads
phase "P2 wait on $MODEL_DONE and $LENS_DONE (byte-growth watch; WARN after ${STALL_SECONDS}s of zero growth, no relaunch)"
bytes_now() { du -sb "$MODEL_DIR" /workspace/lens_hosted "$HF_HOME" 2>/dev/null | awk '{s+=$1} END{print s+0}'; }
last_bytes="$(bytes_now)"; last_change="$(date +%s)"
deadline=$(( $(date +%s) + DL_TIMEOUT_MIN*60 ))
while :; do
  done_model=0; done_lens=0
  [ -f "$MODEL_DONE" ] && done_model=1
  [ -f "$LENS_DONE" ] && done_lens=1
  now="$(date +%s)"; cur="$(bytes_now)"
  rate=$(( (cur - last_bytes) / 30 / 1000000 ))
  echo "  $(date -u +%T) model=$done_model lens=$done_lens total=$((cur/1000000)) MB  ~${rate} MB/s  $(df -h /workspace | tail -1 | awk '{print $4" free"}')"
  if [ "$done_model" = 1 ] && [ "$done_lens" = 1 ]; then break; fi
  if [ "$cur" -gt "$last_bytes" ]; then last_bytes="$cur"; last_change="$now"; fi
  if [ $(( now - last_change )) -ge "$STALL_SECONDS" ]; then
    echo "  WARNING: STALL — no byte growth for $(( now - last_change ))s. Per CLAUDE.md kill+relaunch is nearly free"
    echo "           (snapshot_download resumes); the operator owns these procs, so this script only warns. Check $LOGS/ and ps."
  fi
  [ "$now" -lt "$deadline" ] || die "downloads incomplete after ${DL_TIMEOUT_MIN} min"
  sleep 30
done
echo "PHASE_P2_OK"

# ----------------------------------------------------------------------------- P3 verify artifacts
phase "P3 verify artifacts"
[ -f "$LENS_PT" ] || die "hosted lens missing at $LENS_PT"
got_bytes="$(stat -c %s "$LENS_PT")"
[ "$got_bytes" = "$LENS_BYTES" ] || die "hosted lens size $got_bytes != $LENS_BYTES"
got_sha="$(sha256sum "$LENS_PT" | awk '{print $1}')"
[ "$got_sha" = "$LENS_SHA256" ] || die "hosted lens sha256 $got_sha != $LENS_SHA256"
echo "hosted lens OK: $LENS_PT ($got_bytes bytes, sha256 $got_sha)"
[ -f "$MODEL_DIR/config.json" ] || die "model config.json missing"
n_shards="$(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l)"
[ "$n_shards" -ge 1 ] || die "no safetensors shards in $MODEL_DIR"
[ -f "$MODEL_DIR/model.safetensors.index.json" ] || die "model.safetensors.index.json missing (partial snapshot?)"
n_expected="$("$PY" -c "import json;print(len(set(json.load(open('$MODEL_DIR/model.safetensors.index.json'))['weight_map'].values())))")"
[ "$n_shards" -ge "$n_expected" ] || die "only $n_shards of $n_expected shards present"
# Revision: snapshot_download to a local_dir does not record it; compare the local config.json to the pinned revision via the hub cache if present.
rev_seen="$(ls "$HF_HOME/hub/models--Qwen--Qwen3.6-27B/snapshots" 2>/dev/null | head -1 || true)"
echo "model OK: $MODEL_DIR ($n_shards shards, $(du -sh "$MODEL_DIR" | cut -f1)); hub-cache snapshot rev: ${rev_seen:-n/a} (pin expected $MODEL_REVISION_EXPECTED)"
if [ -n "$rev_seen" ] && [ "$rev_seen" != "$MODEL_REVISION_EXPECTED" ]; then echo "  NOTE: revision differs from the August pin — record it in the note; not fatal."; fi

# Introspect the hosted .pt (keys / dtype / layers): its layout was inferred from file size + Neuronpedia's loader, never opened locally.
"$PY" - "$LENS_PT" <<'PY' || die "hosted lens introspection failed"
import sys, torch, json
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
keys = sorted(ck)
J = ck["J"]; layers = sorted(int(k) for k in J)
sample = J[layers[0]]
info = dict(keys=keys, n_layers=len(layers), first=layers[0], last=layers[-1], shape=list(sample.shape), dtype=str(sample.dtype),
            n_prompts=ck.get("n_prompts"), d_model=ck.get("d_model"), source_layers=ck.get("source_layers"),
            provenance=ck.get("provenance"))
print(json.dumps(info, default=str))
json.dump(info, open("/workspace/lens_hosted/introspect.json", "w"), indent=1, default=str)
assert list(sample.shape) == [5120, 5120], sample.shape
PY

"$PY" - <<PY > /workspace/bootstrap_provenance.json
import json, subprocess, torch, transformers, datasets, safetensors, accelerate, huggingface_hub
print(json.dumps(dict(
  when_utc=subprocess.run(["date","-u","+%FT%TZ"],capture_output=True,text=True).stdout.strip(),
  model_id="$MODEL_ID", model_dir="$MODEL_DIR", model_revision_hub_cache="${rev_seen:-}", model_revision_pin_expected="$MODEL_REVISION_EXPECTED",
  lens_repo="$LENS_REPO", lens_file="$LENS_FILE", lens_path="$LENS_PT", lens_sha256="$got_sha", lens_bytes=$got_bytes,
  jlens_url="$JLENS_URL", jlens_commit="$JLENS_COMMIT",
  torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
  transformers=transformers.__version__, datasets=datasets.__version__, safetensors=safetensors.__version__,
  accelerate=accelerate.__version__, huggingface_hub=huggingface_hub.__version__,
  nvidia_smi=subprocess.run(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],capture_output=True,text=True).stdout.strip(),
), indent=1))
PY
cat /workspace/bootstrap_provenance.json
echo
echo "PHASE_P3_OK"
echo "READY. python=$PY  model=$MODEL_DIR  hosted_lens=$LENS_PT  jlens=$JLENS_DIR@$JLENS_COMMIT  logs=$LOGS  out=$OUT"
echo "Next: RUNBOOK.md step 5 (fit_neel_recipe.py), then step 6 (read_items.py)."
