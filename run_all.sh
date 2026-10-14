#!/usr/bin/env bash
# =============================================================================
# One-command, resumable pipeline for the multi-cohort CAFN experiments:
#   environment -> self-check -> download -> preprocess -> train / validate / test
#
#   ./run_all.sh                 # full run (downloads ~68 GB, trains overnight)
#   ./run_all.sh --smoke         # 1 subject per diagnostic group, 2 epochs: plumbing check
#   ./run_all.sh --help
#
# Safe to interrupt and re-run: downloads resume, preprocessed subjects are
# cached, and finished training runs are skipped.
# Needs: bash, python >= 3.10, ~85 GB free disk. Windows: use WSL2 (CUDA works there) or Git Bash.
# =============================================================================
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ---- constants that make the run reproducible --------------------------------
SEED=42                                  # keep in sync with SEED in src/cohort_experiments.py
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=":4096:8" # deterministic cuBLAS
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export PATH="$PATH:/usr/lib/wsl/lib"     # nvidia-smi lives here inside WSL2

# ---- options ------------------------------------------------------------------
SMOKE=0; TIER=core; REPEATS=3; EPOCHS=120; BATCH=8; DEVICE=auto; OUT=results/cohorts
SKIP_DOWNLOAD=0; SKIP_SETUP=0; SKIP_TESTS=0; WORKERS=2; EXTRA_DL=(); RAW=data/raw; PROCESSED=data/processed
usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; cat <<'USAGE'
Options:
  --smoke            tiny end-to-end check (1 subject/group, 2 epochs, 1 repeat)
  --tier core|all    datasets to download (core = ds003568 + ds007181, ~68 GB; all adds ds008894, +54 GB)
  --repeats N        independent repeats with seeds 42..42+N-1 (default 3)
  --epochs N         max epochs per run (default 120; early stopping patience 25)
  --batch-size N     starting batch size (default 8; halves automatically on GPU OOM)
  --device D         auto | cuda | cpu | mps (default auto)
  --out DIR          results directory (default results/cohorts)
  --workers N        preprocessing worker processes (default 2)
  --raw DIR          raw data directory (default data/raw; downloads always go to data/raw)
  --processed DIR    preprocessed data directory (default data/processed)
  --skip-download    use whatever is already in data/raw
  --skip-setup       do not create/upgrade the virtualenv
  --skip-tests       skip the pre-flight unit tests
USAGE
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; REPEATS=1; EPOCHS=2; EXTRA_DL=(--max-subjects-per-group 1) ;;
    --tier) TIER="$2"; shift ;;
    --repeats) REPEATS="$2"; shift ;;
    --epochs) EPOCHS="$2"; shift ;;
    --batch-size) BATCH="$2"; shift ;;
    --device) DEVICE="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    --workers) WORKERS="$2"; shift ;;
    --raw) RAW="$2"; shift ;;
    --processed) PROCESSED="$2"; shift ;;
    --skip-download) SKIP_DOWNLOAD=1 ;;
    --skip-setup) SKIP_SETUP=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1"; usage; exit 2 ;;
  esac
  shift
done
[[ $SMOKE == 1 ]] && OUT="${OUT}_smoke"

# ---- logging ------------------------------------------------------------------
mkdir -p logs "$OUT"
LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
STEP="startup"; T0=$(date +%s)
step() { STEP="$1"; echo; echo "=================== [$(date +%H:%M:%S)] $1 ==================="; }
on_err() { local rc=$?; echo; echo "!! FAILED during: $STEP  (exit code $rc)."; echo "!! Fix the cause and re-run the same command; finished work is kept and resumed."; echo "!! Log: $LOG"; }
trap on_err ERR

# ---- 0. environment -------------------------------------------------------------
step "0/5 environment"
if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python
elif [[ -x .venv/Scripts/python.exe ]]; then PY=.venv/Scripts/python.exe
else PY=""; fi
if [[ $SKIP_SETUP == 0 ]]; then
  if [[ -z "$PY" ]]; then
    # CPython >= 3.10 only (PyPy cannot run PyTorch, and `python3` is PyPy on some boxes)
    BASE=""
    for c in python3.12 python3.11 python3.10 python3 python; do
      p=$(command -v "$c" 2>/dev/null) || continue
      if "$p" -c 'import sys; sys.exit(0 if sys.implementation.name=="cpython" and sys.version_info>=(3,10) else 1)' 2>/dev/null; then BASE="$p"; break; fi
    done
    [[ -n "$BASE" ]] || { echo "no CPython >= 3.10 found (install python3-venv)"; exit 1; }
    echo "using $BASE ($("$BASE" --version))"
    "$BASE" -m venv .venv
    [[ -x .venv/bin/python ]] && PY=.venv/bin/python || PY=.venv/Scripts/python.exe
  fi
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -r requirements.txt
else
  PY=${PY:-python3}
fi
# On Windows/Linux, make sure the installed torch can see an NVIDIA GPU.
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "NVIDIA GPU found but this torch build has no CUDA; installing the CUDA build..."
    "$PY" -m pip install --quiet --upgrade torch --index-url https://download.pytorch.org/whl/cu124 || \
      echo "!! CUDA torch install failed; continuing (will fall back to CPU)."
  fi
fi
"$PY" - <<'PYEOF'
import platform, shutil, torch
print(f"python {platform.python_version()}  torch {torch.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  {p.total_memory/2**30:.1f} GiB VRAM  (model needs < 2 GiB at batch 8)")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("GPU: Apple MPS (3D pooling falls back to CPU)")
else:
    print("WARNING: no GPU detected -> training runs on CPU (slower but works)")
print(f"free disk here: {shutil.disk_usage('.').free/1e9:.0f} GB")
PYEOF
{
  echo "date: $(date -u +%FT%TZ)"; echo "git: $(git rev-parse HEAD 2>/dev/null || echo n/a)"; echo "seed: $SEED"
  echo "args: smoke=$SMOKE tier=$TIER repeats=$REPEATS epochs=$EPOCHS batch=$BATCH device=$DEVICE"
  "$PY" -m pip freeze
} > "$OUT/environment.txt"

# ---- 1. self-check ----------------------------------------------------------------
if [[ $SKIP_TESTS == 0 ]]; then
  step "1/5 pre-flight tests (cohort loaders + experiment runner)"
  "$PY" -m pytest -q tests/unit/test_cohorts.py tests/unit/test_cohort_experiments.py
fi

# ---- 2. download ------------------------------------------------------------------
if [[ $SKIP_DOWNLOAD == 0 ]]; then
  step "2/5 download datasets (resumable; sizes listed first)"
  "$PY" scripts/download_datasets.py --tier "$TIER" "${EXTRA_DL[@]+"${EXTRA_DL[@]}"}" --yes
else
  step "2/5 download skipped (--skip-download)"
fi

# ---- 3. preprocess ----------------------------------------------------------------
step "3/5 preprocess (T1w -> 64^3 npy, resting MEG/EEG -> 500-pt log-PSD, manifest)"
DATASETS=$([[ $TIER == all ]] && echo "ds003568 ds007181 ds008894" || echo "ds003568 ds007181")
"$PY" -m src.data.cohorts --raw "$RAW" --out "$PROCESSED" --workers "$WORKERS" --datasets $DATASETS

# ---- 4. train / validate / test -----------------------------------------------------
step "4/5 train + validate + test (pooled, then leave-one-cohort-out)"
EXP_ARGS=(--manifest "$PROCESSED/manifest.csv" --out "$OUT" --repeats "$REPEATS" --epochs "$EPOCHS" --batch-size "$BATCH" --device "$DEVICE")
[[ $SMOKE == 1 ]] && EXP_ARGS+=(--smoke)
"$PY" -m src.cohort_experiments "${EXP_ARGS[@]}"

# ---- 5. summary -------------------------------------------------------------------
step "5/5 done"
echo "elapsed: $(( ($(date +%s) - T0) / 60 )) min"
echo "results : $OUT/summary.md  (also results.json, predictions.csv, roc_test.png, per-run curves/checkpoints)"
echo "log     : $LOG"
echo; cat "$OUT/summary.md"
