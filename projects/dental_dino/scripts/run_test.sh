#!/usr/bin/env bash
# 在验证集上评估（与 tools/test.py 一致）。用法见 run_train.sh。
# 默认 conda 环境 dinov3（MMDET_PYTHON / MMDET_CONDA_ENV 可覆盖）。
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=0 $0 <config.py> <checkpoint.pth> [extra args...]" >&2
  exit 1
fi

CFG="$1"
CKPT="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"

MMDET_CONDA_ENV="${MMDET_CONDA_ENV:-dinov3}"
if [[ -n "${MMDET_PYTHON:-}" ]]; then
  PYTHON="${MMDET_PYTHON}"
elif [[ -x "${HOME}/.conda/envs/${MMDET_CONDA_ENV}/bin/python" ]]; then
  PYTHON="${HOME}/.conda/envs/${MMDET_CONDA_ENV}/bin/python"
elif [[ -x "/usr/local/anaconda3/envs/${MMDET_CONDA_ENV}/bin/python" ]]; then
  PYTHON="/usr/local/anaconda3/envs/${MMDET_CONDA_ENV}/bin/python"
else
  PYTHON="python"
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "========== Dental MMDet test =========="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "python_bin: ${PYTHON}"
echo "config: ${CFG}"
echo "checkpoint: ${CKPT}"
if command -v git >/dev/null && git -C "${ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "git_commit: $(git -C "${ROOT}" rev-parse HEAD)"
fi
echo "========================================"

exec "${PYTHON}" tools/test.py "${CFG}" "${CKPT}" "$@"
