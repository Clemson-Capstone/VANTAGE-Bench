#!/usr/bin/env bash
# Full setup script for Gemma4 evaluation on a fresh Brev instance.
# Assumes the vlmevalkit repo has already been synced to ~/vlmevalkit via brev copy.
# Run as: HF_TOKEN=<token> bash ~/setup_gemma4_instance.sh
set -euo pipefail

HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" ]]; then
  echo "ERROR: HF_TOKEN not set. Pass it as: HF_TOKEN=<token> bash $0"
  exit 1
fi

REPO_DIR="$HOME/vlmevalkit"
CONDA_ENV="vlmeval"
RESULTS_DIR="$HOME/results"

echo "========================================"
echo " Gemma4 VLMEvalKit Instance Setup"
echo "========================================"

# ── 1. Install Miniconda if absent ─────────────────────────────────────────
if ! command -v conda &>/dev/null && [[ ! -f "$HOME/miniconda3/bin/conda" ]]; then
  echo ">>> Installing Miniconda..."
  MINICONDA_INSTALLER=/tmp/miniconda.sh
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$MINICONDA_INSTALLER"
  bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
  rm "$MINICONDA_INSTALLER"
  echo ">>> Miniconda installed."
fi
CONDA_BIN="$HOME/miniconda3/bin/conda"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || eval "$($CONDA_BIN shell.bash hook)"
$CONDA_BIN init bash 2>/dev/null || true

# ── 2. Create / activate conda environment ─────────────────────────────────
if $CONDA_BIN env list | grep -q "^${CONDA_ENV}\s"; then
  echo ">>> Conda env '${CONDA_ENV}' already exists."
else
  echo ">>> Creating conda env '${CONDA_ENV}' (Python 3.10)..."
  $CONDA_BIN create -y -n "$CONDA_ENV" python=3.10
fi
conda activate "$CONDA_ENV"

# ── 3. Install Python dependencies ─────────────────────────────────────────
echo ">>> Installing PyTorch (CUDA 12.1)..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ">>> Installing transformers 5.x..."
pip install -q "transformers>=5.0.0" accelerate "huggingface_hub>=0.26"

echo ">>> Installing VLMEvalKit core deps..."
pip install -q \
  decord \
  opencv-python \
  pillow \
  qwen-vl-utils \
  sentencepiece \
  datasets \
  pandas \
  numpy \
  rich \
  portalocker \
  python-dotenv \
  requests \
  scipy \
  scikit-learn \
  matplotlib \
  omegaconf \
  einops \
  s3fs \
  math-verify \
  json_repair \
  openai \
  google-genai \
  imageio

echo ">>> Installing flash-attention..."
pip install -q flash-attn --no-build-isolation 2>/dev/null && echo "  flash-attn OK" || \
  echo "  flash-attn skipped (will use SDPA, fine for inference)"

# ── 4. Install VLMEvalKit in editable mode ─────────────────────────────────
if [[ -d "$REPO_DIR" ]]; then
  echo ">>> Installing vlmevalkit from $REPO_DIR..."
  pip install -q -e "$REPO_DIR"
else
  echo "ERROR: $REPO_DIR not found. Run brev copy to sync the repo first."
  exit 1
fi

# ── 5. Configure HuggingFace token ─────────────────────────────────────────
echo ">>> Configuring HuggingFace credentials..."
python -c "
import huggingface_hub, os
huggingface_hub.login(token='${HF_TOKEN}', add_to_git_credential=False)
print('HF login OK')
" 2>/dev/null || true

# Persist credentials for future sessions
grep -q "HF_TOKEN" "$HOME/.bashrc" || echo "export HF_TOKEN='${HF_TOKEN}'" >> "$HOME/.bashrc"
grep -q "HUGGING_FACE_HUB_TOKEN" "$HOME/.bashrc" || echo "export HUGGING_FACE_HUB_TOKEN='${HF_TOKEN}'" >> "$HOME/.bashrc"

# ── 6. Configure dataset root ──────────────────────────────────────────────
# Place VANTAGE benchmark data under $LMUData/datasets/ before running evaluations.
# Expected layout:
#   $LMUData/datasets/VANTAGE_VQA/VANTAGE_VQA.tsv + videos/
#   $LMUData/datasets/VANTAGE_Temporal/VANTAGE_Temporal.tsv + videos/
#   ... (see README.md for full layout)
LMUDATA_ROOT="${LMUData:-$HOME/LMUData}"
mkdir -p "$LMUDATA_ROOT/datasets"

grep -q "export LMUData=" "$HOME/.bashrc" || echo "export LMUData='${LMUDATA_ROOT}'" >> "$HOME/.bashrc"
grep -q "export LMUData=" "$HOME/.profile" || echo "export LMUData='${LMUDATA_ROOT}'" >> "$HOME/.profile"
echo ">>> LMUData root set to: $LMUDATA_ROOT"

# ── 8. Persist conda activation ───────────────────────────────────────────
grep -q "conda activate ${CONDA_ENV}" "$HOME/.bashrc" || \
  echo "conda activate ${CONDA_ENV}" >> "$HOME/.bashrc"

# ── 9. Verify everything ──────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Verification"
echo "========================================"
cd "$REPO_DIR"
python - <<'PYEOF'
import torch
print(f"torch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  {p.total_memory//1024**3}GB VRAM")

import transformers
print(f"transformers: {transformers.__version__}")
from transformers import Gemma4ForConditionalGeneration
print("Gemma4ForConditionalGeneration: OK")

from vlmeval.vlm import Gemma4
from vlmeval.config import supported_VLM
g4 = [k for k in supported_VLM if "Gemma4" in k]
print("Gemma4 wrapper import: OK")
print("Registered models:", g4)

import os
lmu = os.environ.get("LMUData", "NOT SET")
print(f"LMUData env: {lmu}")
import os.path as osp
if lmu != "NOT SET":
    ds = osp.join(lmu, "datasets")
    print(f"datasets/ exists: {osp.isdir(ds)}")
    if osp.isdir(ds):
        print("  datasets:", sorted(os.listdir(ds)))
PYEOF

mkdir -p "$RESULTS_DIR"

echo ""
echo "========================================"
echo " Setup complete!"
echo "========================================"
echo ""
echo "Next — run first smoke test:"
echo "  cd $REPO_DIR"
echo "  python run.py --data VANTAGE_VQA_8frame --model Gemma4-E2B --nproc 1 --work-dir $RESULTS_DIR/gemma4-e2b"
