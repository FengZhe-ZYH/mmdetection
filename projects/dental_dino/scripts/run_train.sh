#!/usr/bin/env bash
# Plan.md：启动训练并打印 CUDA / git / 配置路径（在 mmdetection 仓库根目录执行）。
# 默认使用 conda 环境 dinov3 的 Python（可用 MMDET_PYTHON 覆盖整路径，或 MMDET_CONDA_ENV 换环境名）。
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=0 $0 <config.py> [extra args to tools/train.py ...]" >&2
  exit 1
fi

CFG="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/ -> dental_dino/ -> projects/ -> mmdetection 仓库根
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

echo "========== Dental MMDet train banner =========="
echo "time: $(date -Iseconds)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "python_bin: ${PYTHON}"
echo "cwd: ${ROOT}"
echo "config: ${CFG}"
if command -v git >/dev/null && git -C "${ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "git_commit: $(git -C "${ROOT}" rev-parse HEAD)"
  echo "git_status:"
  git -C "${ROOT}" status -sb || true
else
  echo "git: not a git repo or git missing"
fi
echo "=============================================="

exec "${PYTHON}" tools/train.py "${CFG}" "$@"
