Dental DINO — 定时训练（实验 E / F，物理 GPU 2）
==============================================

说明
----
- 时间均为「本机 date 命令看到的本地时区」；cron 使用系统本地时间。
- 仅 2026-05-09 当天会真正跑（脚本内写死日期，避免以后每年 5 月 9 日重复执行）。
- 无法在云端由 AI「凌晨人工抽查」；脚本会在训练启动约 30 分钟后自动写 .health 健康摘要。

已安装 cron（若你允许本仓库安装脚本执行了 crontab -）：
  01:00  实验 E  dino-4scale_r50_8xb2-12e_perio_toothprior_queries.py
  06:00  实验 F  dino-4scale_r50_8xb2-12e_perio_toothprior_aux.py

日志目录
--------
  mmdetection/projects/dental_dino/logs/scheduled/
    train_E_*.log / train_F_*.log     训练标准输出
    train_*_*.log.health              启动约 30 分钟后的自动抽查
    gpu_wait.log                      GPU 等待记录
    launcher.log                      启动/结束时间
    cron_wrapper.log                  cron 外层 bash 输出

GPU 等待阈值（可调环境变量）
----------------------------
  MEM_USED_MAX_MB  默认 12000  （MiB，低于则认为显存较空）
  UTIL_MAX         默认 25     （GPU 利用率 %% 低于则认为较空）
  POLL_INTERVAL    默认 3600   （不满足时睡眠秒数，即「每小时看一次」）
  STABLE_POLLS     默认 2      （需连续满足次数）
  STABLE_SLEEP     默认 120    （两次确认间隔秒数）
  MAX_WAIT_HOURS   默认 36     （超过则放弃本次启动并 exit 1）

手动安装 / 卸载 cron
---------------------
  安装（追加两行，不覆盖其它任务；请先 crontab -l 检查）：
    ( crontab -l 2>/dev/null | grep -v 'run_scheduled_dental.sh'; cat <<'EOF'
0 1 9 5 * /bin/bash /hdd1/zyh/Dental/mutil_repo/mmdetection/projects/dental_dino/scripts/scheduled/run_scheduled_dental.sh E >>/hdd1/zyh/Dental/mutil_repo/mmdetection/projects/dental_dino/logs/scheduled/cron_wrapper.log 2>&1
0 6 9 5 * /bin/bash /hdd1/zyh/Dental/mutil_repo/mmdetection/projects/dental_dino/scripts/scheduled/run_scheduled_dental.sh F >>/hdd1/zyh/Dental/mutil_repo/mmdetection/projects/dental_dino/logs/scheduled/cron_wrapper.log 2>&1
EOF
    ) | crontab -

  卸载（删掉含 run_scheduled_dental 的行）：
    crontab -l 2>/dev/null | grep -v run_scheduled_dental.sh | crontab -

手动立刻试跑（仍会检查日期=2026-05-09）
----------------------------------------
  bash .../run_scheduled_dental.sh E

若 5 月 9 日已过需改日期：编辑 run_scheduled_dental.sh 中 RUN_DATE= 一行。
