#!/usr/bin/env bash
# 定时入口：在「服务器本地时间」指定日期的 01:00 / 06:00 由 cron 调用。
# 用法: run_scheduled_dental.sh E|F
# - 仅在日期为 2026-05-09 时真正执行（防止每年 5 月 9 日重复跑）。
# - 物理 GPU 2：通过 CUDA_VISIBLE_DEVICES=2。
# - 若 GPU2 显存/利用率仍高，每小时重试，直到空闲或超过 MAX_WAIT_HOURS。
# - 训练启动后约 30 分钟自动 tail 日志写入 .health 文件（非人工抽查）。

set -euo pipefail

EXPERIMENT="${1:?第一个参数为 E 或 F}"

MMROOT="/hdd1/zyh/Dental/mutil_repo/mmdetection"
PY="${PY:-/home/zyh/.conda/envs/dinov3/bin/python}"
LOGROOT="${MMROOT}/projects/dental_dino/logs/scheduled"
mkdir -p "${LOGROOT}"

RUN_DATE="2026-05-09"
TODAY="$(date +%F)"
if [[ "${TODAY}" != "${RUN_DATE}" ]]; then
  echo "$(date -Is) skip: today=${TODAY} expected=${RUN_DATE} experiment=${EXPERIMENT}" >>"${LOGROOT}/cron_skip.log"
  exit 0
fi

case "${EXPERIMENT}" in
  E) CFG="projects/dental_dino/configs/dino-4scale_r50_8xb2-12e_perio_toothprior_queries.py" ;;
  F) CFG="projects/dental_dino/configs/dino-4scale_r50_8xb2-12e_perio_toothprior_aux.py" ;;
  *) echo "bad experiment: ${EXPERIMENT}"; exit 2 ;;
esac

# 同实验同日只跑一次（cron 重试 / 手动误触）
LOCK="${LOGROOT}/.lock_exp${EXPERIMENT}_${RUN_DATE//-/}"
if [[ -f "${LOCK}" ]]; then
  echo "$(date -Is) already ran ${EXPERIMENT} lock=${LOCK}" >>"${LOGROOT}/cron_skip.log"
  exit 0
fi

GPU_INDEX=2
# 认为「可启动训练」：GPU2 已用显存低于此值（MiB）且利用率低于 UTIL_MAX
MEM_USED_MAX_MB="${MEM_USED_MAX_MB:-12000}"
UTIL_MAX="${UTIL_MAX:-25}"
POLL_INTERVAL="${POLL_INTERVAL:-3600}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-36}"
STABLE_POLLS="${STABLE_POLLS:-2}"
STABLE_SLEEP="${STABLE_SLEEP:-120}"

gpu_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_INDEX}" 2>/dev/null | head -1 | tr -d ' ' || echo 999999
}

gpu_util() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${GPU_INDEX}" 2>/dev/null | head -1 | tr -d ' ' || echo 100
}

wait_for_quiet_gpu() {
  local deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
  local ok_streak=0
  while (( $(date +%s) < deadline )); do
    local used util
    used="$(gpu_used_mb)"
    util="$(gpu_util)"
    if [[ "${used}" -lt "${MEM_USED_MAX_MB}" && "${util}" -lt "${UTIL_MAX}" ]]; then
      ok_streak=$((ok_streak + 1))
      echo "$(date -Is) GPU${GPU_INDEX} candidate ok streak=${ok_streak}/${STABLE_POLLS} used=${used}MiB util=${util}%" \
        >>"${LOGROOT}/gpu_wait.log"
      if (( ok_streak >= STABLE_POLLS )); then
        return 0
      fi
      sleep "${STABLE_SLEEP}"
    else
      ok_streak=0
      echo "$(date -Is) GPU${GPU_INDEX} busy used=${used}MiB util=${util}% -> sleep ${POLL_INTERVAL}s" \
        >>"${LOGROOT}/gpu_wait.log"
      sleep "${POLL_INTERVAL}"
    fi
  done
  echo "$(date -Is) GPU wait timeout after ${MAX_WAIT_HOURS}h" >>"${LOGROOT}/gpu_wait.log"
  exit 1
}

wait_for_quiet_gpu
touch "${LOCK}"

TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="${LOGROOT}/train_${EXPERIMENT}_${TS}.log"
HEALTH="${LOGFILE}.health"

{
  echo "==== $(date -Is) start ${EXPERIMENT} ===="
  echo "CUDA_VISIBLE_DEVICES=2 PYTHONPATH=${MMROOT}"
  echo "cfg=${CFG}"
  echo "python=${PY}"
} >>"${LOGFILE}"

cd "${MMROOT}"
export PYTHONPATH="${MMROOT}"

CUDA_VISIBLE_DEVICES=2 "${PY}" tools/train.py "${CFG}" >>"${LOGFILE}" 2>&1 &
TRAIN_PID=$!
echo "${TRAIN_PID}" >"${LOGROOT}/train_${EXPERIMENT}_${TS}.pid"
echo "$(date -Is) launched pid=${TRAIN_PID} log=${LOGFILE}" >>"${LOGROOT}/launcher.log"

(
  sleep 1800
  {
    echo "==== health @30min $(date -Is) experiment=${EXPERIMENT} pid=${TRAIN_PID} ===="
    if kill -0 "${TRAIN_PID}" 2>/dev/null; then
      echo "status: train process still alive"
    else
      echo "status: train process NOT running (may have exited or OOM)"
    fi
    echo "--- last 120 lines of training log ---"
    tail -n 120 "${LOGFILE}" 2>/dev/null || echo "(no log yet)"
    echo "--- grep suspicious (last 20 hits) ---"
    grep -iE "traceback|cuda out of memory|oom|killed|error:" "${LOGFILE}" 2>/dev/null | tail -20 || echo "(no matches)"
    echo "--- grep training heartbeat (last 15) ---"
    grep -iE "Epoch|loss|Iter|lr:" "${LOGFILE}" 2>/dev/null | tail -15 || echo "(no matches)"
  } >>"${HEALTH}" 2>&1
) &
echo "$(date -Is) health report will append to ${HEALTH} in ~30min" >>"${LOGROOT}/launcher.log"

wait "${TRAIN_PID}" || true
echo "$(date -Is) train pid ${TRAIN_PID} finished exit=$?" >>"${LOGROOT}/launcher.log"
