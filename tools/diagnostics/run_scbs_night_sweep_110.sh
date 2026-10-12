#!/usr/bin/env bash
# Six independent runs, sequential on one GPU; no deletion/resume/overwrite.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
night_fail() { printf '错误：%s\n' "$*" >&2; exit 1; }
night_usage() {
  printf '%s\n' \
    '用法：bash tools/diagnostics/run_scbs_night_sweep_110.sh [--dry-run] [N1 ... N6]' \
    '默认顺序：N4 N6 N5 N2 N1 N3；每组独立110轮，不叠加，不改损失。' \
    '训练后自动用本组best_stg2/EMA导出混淆矩阵、全部query并离线分析。' \
    '环境变量：NIGHT_GPU=0、NIGHT_RUN_ID=1、NIGHT_CONDA_ENV=wq、NIGHT_OUTPUT_ROOT。' \
    '可设NIGHT_EXPORT=0只训练；NIGHT_STOP_ON_ERROR=1在首个训练/导出失败后停队列。' \
    '正式启动前默认执行六组CUDA FP16小样本反向检查；不是batch32显存保证。' \
    '所有输出必须不存在；--dry-run不创建文件、不检查服务器GPU/数据。' \
    'NIGHT_PYTHON仅用于CPU演练；正式训练使用Conda环境。'
}
night_groups=(); night_dry=0
for night_arg in "$@"; do
  case "$night_arg" in
    --dry-run) night_dry=1 ;;
    -h|--help) night_usage; exit 0 ;;
    N[1-6]) night_groups+=("$night_arg") ;;
    *) night_fail "未知组别/参数：$night_arg" ;;
  esac
done
[[ ${#night_groups[@]} -gt 0 ]] || night_groups=(N4 N6 N5 N2 N1 N3)
night_root="${NIGHT_OUTPUT_ROOT:-/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw}"
night_run="${NIGHT_RUN_ID:-1}"; night_gpu="${NIGHT_GPU:-0}"
night_export="${NIGHT_EXPORT:-1}"; night_stop="${NIGHT_STOP_ON_ERROR:-0}"
night_conda="${NIGHT_CONDA:-${CONDA_EXE:-conda}}"; night_env="${NIGHT_CONDA_ENV:-wq}"
[[ "$night_root" == /* ]] || night_fail '输出根目录必须是绝对路径'
[[ "$night_run" =~ ^[1-9][0-9]*$ && "$night_gpu" =~ ^[0-9]+$ ]] || night_fail 'RUN_ID须为正整数，GPU须为非负整数'
[[ "$night_export" =~ ^[01]$ && "$night_stop" =~ ^[01]$ ]] || night_fail 'EXPORT/STOP_ON_ERROR只能为0或1'
night_queue="$night_root/night_sweep_epochs110_seed0_run${night_run}"
[[ ! -e "$night_queue" ]] || night_fail "队列目录已存在：$night_queue；请用新的NIGHT_RUN_ID"
night_python=("$night_conda" run --no-capture-output -n "$night_env" python)
if [[ "$night_dry" == 1 && -n "${NIGHT_PYTHON:-}" ]]; then
  night_python=("$NIGHT_PYTHON")
else
  command -v "$night_conda" >/dev/null 2>&1 || night_fail '找不到Conda'
fi
export CUDA_VISIBLE_DEVICES="$night_gpu" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
night_indices=(); night_configs=(); night_outputs=(); night_seen=' '
for night_group in "${night_groups[@]}"; do
  case "$night_seen" in *" $night_group "*) night_fail "重复组别：$night_group" ;; esac
  night_seen+="$night_group "
  night_i="${night_group#N}"
  night_config="configs/dfine/dfine_s_scbs_hrw_110_night_n${night_i}.yml"
  night_out="$night_root/dfine_s_scbs_hrw_epochs110_night_n${night_i}_seed0_run${night_run}"
  [[ -f "$night_config" && -f train.py ]] || night_fail "缺少源码/配置：$night_config"
  [[ ! -e "$night_out" ]] || night_fail "结果目录已存在，未启动任何训练：$night_out"
  night_indices+=("$night_i"); night_configs+=("$night_config"); night_outputs+=("$night_out")
  printf '%s：%s\n  输出：%s\n' "$night_group" "$night_config" "$night_out"
done
if [[ "$night_dry" == 1 ]]; then
  "${night_python[@]}" tools/diagnostics/scbs_night_queue.py preflight --dry-run "${night_indices[@]}"
else
  "${night_python[@]}" tools/diagnostics/scbs_night_queue.py preflight "${night_indices[@]}"
fi
if [[ "$night_dry" == 1 ]]; then
  printf '演练通过；未启动训练，未创建输出文件。正式运行删除 --dry-run。\n'
  exit 0
fi

# Lock only this queue, not other users' programs. Never kill a GPU process.
if command -v flock >/dev/null 2>&1; then
  exec 9>"/tmp/dfine-night-sweep-gpu${night_gpu}.lock"
  flock -n 9 || night_fail "同GPU的另一份夜间队列仍在运行：GPU$night_gpu"
fi
night_check_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || night_fail '找不到nvidia-smi，不能确认GPU空闲'
  local night_pids night_pid night_process
  night_pids="$(nvidia-smi -i "$night_gpu" --query-compute-apps=pid --format=csv,noheader,nounits)" || night_fail 'GPU查询失败'
  while IFS= read -r night_pid; do
    night_pid="${night_pid//[[:space:]]/}"
    [[ "$night_pid" =~ ^[0-9]+$ ]] || continue
    # Linux comm can truncate gnome-remote-desktop to 15 characters; inspect
    # full argv so the ordinary display process is not mistaken for training.
    night_process="$(ps -ww -p "$night_pid" -o args= 2>/dev/null || true)"
    [[ -n "$night_process" ]] || continue  # stale nvidia-smi PID
    case "$night_process" in *Xorg*|*gnome-shell*|*gnome-remote-desktop*) continue ;; esac
    night_fail "GPU${night_gpu}仍有进程PID=${night_pid}：${night_process}；未停止其他任务"
  done <<< "$night_pids"
}
night_check_gpu
"${night_python[@]}" tools/diagnostics/test_night_probes.py NightProbeTests.test_cuda_fp16_training_all_six
mkdir -p -- "$night_root"
mkdir -- "$night_queue"
night_failed=0
trap 'printf "队列已中断；保留全部结果，未自动恢复或删除。\n" >&2; exit 130' INT TERM
for ((night_index=0; night_index<${#night_groups[@]}; night_index++)); do
  night_check_gpu
  night_i="${night_indices[$night_index]}"; night_out="${night_outputs[$night_index]}"
  night_config="${night_configs[$night_index]}"
  mkdir -- "$night_out"
  "${night_python[@]}" tools/diagnostics/scbs_night_queue.py snapshot "$night_i" "$night_out"
  printf '\n[%s/%s] 开始%s，110轮，seed=0，无断点/微调：%s\n' \
    "$((night_index+1))" "${#night_groups[@]}" "${night_groups[$night_index]}" "$night_out"
  set +e
  "$night_conda" run --no-capture-output -n "$night_env" \
    torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
    train.py -c "$night_config" --output-dir "$night_out" --use-amp --seed=0 \
    2>&1 | tee "$night_out/console.log"
  night_pipe=("${PIPESTATUS[@]}")
  set -e
  night_code="${night_pipe[0]}"
  [[ "$night_code" != 0 || "${night_pipe[1]}" == 0 ]] || night_code="${night_pipe[1]}"
  if [[ "$night_code" != 0 || ! -f "$night_out/best_stg2.pth" ]]; then
    [[ "$night_code" != 0 ]] || night_code=1
    "${night_python[@]}" tools/diagnostics/scbs_night_queue.py record "$night_i" "$night_out" "$night_queue" training_failed "$night_code"
    night_failed=1
    [[ "$night_stop" == 0 ]] || exit "$night_code"
    continue
  fi
  if [[ "$night_export" == 1 ]]; then
    set +e
    "$night_conda" run --no-capture-output -n "$night_env" \
      torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
      train.py -c "$night_config" -r "$night_out/best_stg2.pth" --test-only \
      --output-dir "$night_out/diagnostics_eval/evaluation" --use-amp --seed=0 \
      -u export_confusion_matrix=true export_query_diagnostics=true \
      query_diagnostic_conf_thresh=0.5 query_diagnostic_iou_thresh=0.5 query_diagnostic_neighbor_iou_thresh=0.8 \
      2>&1 | tee "$night_out/diagnostics_console.log"
    night_pipe=("${PIPESTATUS[@]}")
    set -e
    night_code="${night_pipe[0]}"
    [[ "$night_code" != 0 || "${night_pipe[1]}" == 0 ]] || night_code="${night_pipe[1]}"
    if [[ "$night_code" == 0 ]]; then
      set +e
      "${night_python[@]}" tools/diagnostics/analyze_detection_diagnostics.py \
        --input-dir "$night_out/diagnostics_eval/evaluation" --output-dir "$night_out/diagnostics_eval/analysis" \
        2>&1 | tee "$night_out/analysis_console.log"
      night_pipe=("${PIPESTATUS[@]}")
      set -e
      night_code="${night_pipe[0]}"
      [[ "$night_code" != 0 || "${night_pipe[1]}" == 0 ]] || night_code="${night_pipe[1]}"
    fi
    if [[ "$night_code" != 0 ]]; then
      "${night_python[@]}" tools/diagnostics/scbs_night_queue.py record "$night_i" "$night_out" "$night_queue" diagnostics_failed "$night_code"
      night_failed=1
      [[ "$night_stop" == 0 ]] || exit "$night_code"
      continue
    fi
  fi
  "${night_python[@]}" tools/diagnostics/scbs_night_queue.py record "$night_i" "$night_out" "$night_queue" completed 0
done
printf '\n队列结束。汇总：%s/results.jsonl；训练日志在各组console.log。\n' "$night_queue"
[[ "$night_failed" == 0 ]] || night_fail '存在失败组，请查看results.jsonl；旧结果全部保留'
